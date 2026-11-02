#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::undocumented_unsafe_blocks
)]

use core::mem::size_of;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::Command;
use std::thread;
use std::time::Duration;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN,
    TokenInputSlotHeader,
};
use k3_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, IpcMessage, ShutdownCommand, StatusCode,
    WorkerHeartbeatRequest, global_fory, recv_frame, recv_msg_frame, send_frame, send_msg_frame,
};
use k3_core::ipc::shm::{DEFAULT_LIMIT_MEMLOCK_BYTES, SharedMemoryArena};
use zerocopy::{FromBytes, IntoBytes};

const BIN_PATH: &str = env!("CARGO_BIN_EXE_k3-worker");

/// Helper to create a dummy valid model file
fn create_dummy_model(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut f = File::create(&path).expect("create dummy model");
    f.write_all(b"CHALLENGER_EMPIRICAL_TEST_MODEL_V3")
        .expect("write dummy model");
    path
}

// ----------------------------------------------------------------------------
// Challenge 1: Process Exit Code 0 (Help, Version, Shutdown, EOF, Signals)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_0_variants() {
    let model_path = create_dummy_model("k3_worker_exit0_model.gguf");

    // 1.1 --help and -h
    let res = Command::new(BIN_PATH).arg("--help").output().unwrap();
    assert_eq!(res.status.code(), Some(0));

    let res = Command::new(BIN_PATH).arg("-h").output().unwrap();
    assert_eq!(res.status.code(), Some(0));

    // 1.2 --version and -V
    let res = Command::new(BIN_PATH).arg("--version").output().unwrap();
    assert_eq!(res.status.code(), Some(0));

    let res = Command::new(BIN_PATH).arg("-V").output().unwrap();
    assert_eq!(res.status.code(), Some(0));

    // 1.3 Clean shutdown on socket EOF (Supervisor closes socket)
    let sock_path = std::env::temp_dir().join("k3_worker_eof_test.sock");
    if sock_path.exists() {
        let _ = std::fs::remove_file(&sock_path);
    }

    let mut child = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker");

    // Connect and immediately close socket to trigger EOF
    let mut stream_opt = None;
    for _ in 0..50 {
        if sock_path.exists() {
            if let Ok(s) = UnixStream::connect(&sock_path) {
                stream_opt = Some(s);
                break;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let stream = stream_opt.expect("connect to worker");
    drop(stream); // Trigger EOF

    let exit_status = child.wait().expect("wait for child on EOF");
    assert_eq!(
        exit_status.code(),
        Some(0),
        "Worker must exit 0 cleanly on supervisor EOF"
    );

    // 1.4 Clean shutdown on SIGINT (Ctrl+C)
    let sock_path_sigint = std::env::temp_dir().join("k3_worker_sigint_test.sock");
    if sock_path_sigint.exists() {
        let _ = std::fs::remove_file(&sock_path_sigint);
    }

    let mut child_sigint = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path_sigint.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker for sigint");

    let mut stream_sigint = None;
    for _ in 0..50 {
        if sock_path_sigint.exists() {
            if let Ok(s) = UnixStream::connect(&sock_path_sigint) {
                stream_sigint = Some(s);
                break;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _s_sigint = stream_sigint.expect("connected for sigint");
    thread::sleep(Duration::from_millis(100));

    // Send SIGINT
    unsafe {
        libc::kill(child_sigint.id() as i32, libc::SIGINT);
    }
    let exit_sigint = child_sigint.wait().expect("wait for child on SIGINT");
    assert_eq!(
        exit_sigint.code(),
        Some(0),
        "Worker must exit 0 cleanly on SIGINT"
    );

    // 1.5 Clean shutdown on SIGTERM
    let sock_path_sigterm = std::env::temp_dir().join("k3_worker_sigterm_test.sock");
    if sock_path_sigterm.exists() {
        let _ = std::fs::remove_file(&sock_path_sigterm);
    }

    let mut child_sigterm = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path_sigterm.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker for sigterm");

    let mut stream_sigterm = None;
    for _ in 0..50 {
        if sock_path_sigterm.exists() {
            if let Ok(s) = UnixStream::connect(&sock_path_sigterm) {
                stream_sigterm = Some(s);
                break;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _s_sigterm = stream_sigterm.expect("connected for sigterm");
    thread::sleep(Duration::from_millis(100));

    // Send SIGTERM
    unsafe {
        libc::kill(child_sigterm.id() as i32, libc::SIGTERM);
    }
    let exit_sigterm = child_sigterm.wait().expect("wait for child on SIGTERM");
    assert_eq!(
        exit_sigterm.code(),
        Some(0),
        "Worker must exit 0 cleanly on SIGTERM"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_file(&sock_path_sigint);
    let _ = std::fs::remove_file(&sock_path_sigterm);
}

// ----------------------------------------------------------------------------
// Challenge 2: Process Exit Code 2 (Comprehensive CLI Boundary Violations)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_2_boundary_matrix() {
    let model_path = create_dummy_model("k3_worker_exit2_model.gguf");
    let valid_m = model_path.to_str().unwrap();

    let invalid_cli_cases: &[(&[&str], &str)] = &[
        // Missing arguments
        (&[], "no arguments"),
        (&["--model", valid_m], "missing socket"),
        (&["--socket", "/tmp/k3.sock"], "missing model"),
        // Empty strings
        (
            &["--model", "", "--socket", "/tmp/k3.sock"],
            "empty model path",
        ),
        (
            &["--model", "   ", "--socket", "/tmp/k3.sock"],
            "whitespace model path",
        ),
        (&["--model", valid_m, "--socket", ""], "empty socket path"),
        (
            &["--model", valid_m, "--socket", "   "],
            "whitespace socket path",
        ),
        // Socket FD invalid
        (
            &["--model", valid_m, "--socket-fd", "not_a_number"],
            "non-numeric socket fd",
        ),
        (
            &["--model", valid_m, "--socket-fd", "-1"],
            "negative socket fd",
        ),
        // Cluster invalid
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--cluster",
                "cuda",
            ],
            "unsupported cluster cuda",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--cluster",
                "x200",
            ],
            "unsupported cluster x200",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--cluster",
                "",
            ],
            "empty cluster",
        ),
        // Affinity invalid
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "",
            ],
            "empty affinity",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "   ",
            ],
            "whitespace affinity",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "64",
            ],
            "core index >= 64",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "128",
            ],
            "core index 128",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "0x0",
            ],
            "zero hex affinity mask",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "0X00",
            ],
            "zero hex affinity mask upper",
        ),
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--affinity",
                "xyz",
            ],
            "malformed affinity string",
        ),
        // Unknown arguments
        (
            &[
                "--model",
                valid_m,
                "--socket",
                "/tmp/k3.sock",
                "--invalid-flag",
            ],
            "unrecognized flag",
        ),
        (
            &["--model", valid_m, "--socket", "/tmp/k3.sock", "-Z"],
            "unrecognized short flag",
        ),
    ];

    for (args, desc) in invalid_cli_cases {
        let output = Command::new(BIN_PATH)
            .args(*args)
            .output()
            .unwrap_or_else(|e| panic!("failed to execute {desc}: {e}"));
        assert_eq!(
            output.status.code(),
            Some(2),
            "Expected exit code 2 for {desc}, got {:?}",
            output.status.code()
        );
    }

    let _ = std::fs::remove_file(&model_path);
}

// ----------------------------------------------------------------------------
// Challenge 3: Process Exit Code 10 (CapabilityMismatch & Exclusive Lock)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_10_capability_mismatch() {
    let model_path = create_dummy_model("k3_worker_exit10_model.gguf");
    let m_str = model_path.to_str().unwrap();

    #[cfg(not(target_arch = "riscv64"))]
    {
        // 3.1 Requesting A100 cluster on non-riscv64 host
        let res = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                m_str,
                "--socket",
                "/tmp/test10.sock",
                "--cluster",
                "a100",
            ])
            .output()
            .unwrap();
        assert_eq!(
            res.status.code(),
            Some(10),
            "a100 on non-riscv64 must exit 10"
        );

        // 3.2 Requesting core 8 (Cluster 1) on non-riscv64 host
        let res = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                m_str,
                "--socket",
                "/tmp/test10.sock",
                "--affinity",
                "8",
            ])
            .output()
            .unwrap();
        assert_eq!(
            res.status.code(),
            Some(10),
            "core 8 on non-riscv64 must exit 10"
        );

        // 3.3 Requesting core mask 0xFF00 (Cluster 1) on non-riscv64 host
        let res = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                m_str,
                "--socket",
                "/tmp/test10.sock",
                "--affinity",
                "0xFF00",
            ])
            .output()
            .unwrap();
        assert_eq!(
            res.status.code(),
            Some(10),
            "0xFF00 on non-riscv64 must exit 10"
        );
    }

    // 3.4 Exclusive lock conflict: LOCK_EX held by another process
    let f_lock = File::open(&model_path).unwrap();
    let lock_ret = unsafe { libc::flock(f_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(lock_ret, 0, "Test acquired exclusive flock");

    let res = Command::new(BIN_PATH)
        .args([
            "--model",
            m_str,
            "--socket",
            "/tmp/test10.sock",
            "--cluster",
            "x100",
        ])
        .output()
        .unwrap();
    assert_eq!(
        res.status.code(),
        Some(10),
        "flock(LOCK_EX) conflict must exit 10"
    );

    unsafe { libc::flock(f_lock.as_raw_fd(), libc::LOCK_UN) };
    drop(f_lock);

    let _ = std::fs::remove_file(&model_path);
}

// ----------------------------------------------------------------------------
// Challenge 4: Process Exit Code 11 (LimitMEMLOCK 28 GiB Ceiling Exceeded)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_11_memlock_ceiling_exceeded() {
    let large_model_path = std::env::temp_dir().join("k3_worker_29gib_sparse.gguf");
    let file = File::create(&large_model_path).expect("create sparse file");

    // Create a 29 GiB sparse file (exceeding 28 GiB LimitMEMLOCK)
    // 28 GiB = DEFAULT_LIMIT_MEMLOCK_BYTES = 30_064_771_072 bytes
    let excess_len = DEFAULT_LIMIT_MEMLOCK_BYTES + (1024 * 1024); // 28 GiB + 1 MiB
    file.set_len(excess_len)
        .expect("set sparse file length to 28.001 GiB");

    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            large_model_path.to_str().unwrap(),
            "--socket",
            "/tmp/k3_exit11.sock",
            "--cluster",
            "x100",
        ])
        .output()
        .expect("run k3-worker on 29 GiB file");

    let _ = std::fs::remove_file(&large_model_path);

    assert_eq!(
        output.status.code(),
        Some(11),
        "File exceeding 28 GiB LimitMEMLOCK ceiling must exit with code 11 (InsufficientLockableMemory)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("exceeds operator LimitMEMLOCK bound")
            || stderr.contains("insufficient lockable memory"),
        "stderr should mention LimitMEMLOCK ceiling: {stderr}"
    );
}

// ----------------------------------------------------------------------------
// Challenge 5: Process Exit Code 12 (Missing or Empty Model File)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_12_resource_exhausted() {
    // 5.1 Non-existent file
    let res = Command::new(BIN_PATH)
        .args([
            "--model",
            "/tmp/k3_definitely_nonexistent_model_file_12345.gguf",
            "--socket",
            "/tmp/k3.sock",
        ])
        .output()
        .unwrap();
    assert_eq!(
        res.status.code(),
        Some(12),
        "Missing model file must exit 12"
    );

    // 5.2 0-byte file
    let empty_path = std::env::temp_dir().join("k3_worker_0byte_file.gguf");
    File::create(&empty_path).unwrap();

    let res = Command::new(BIN_PATH)
        .args([
            "--model",
            empty_path.to_str().unwrap(),
            "--socket",
            "/tmp/k3.sock",
        ])
        .output()
        .unwrap();

    let _ = std::fs::remove_file(&empty_path);
    assert_eq!(
        res.status.code(),
        Some(12),
        "0-byte model file must exit 12"
    );
}

// ----------------------------------------------------------------------------
// Challenge 6: Process Exit Code 13 (Protocol Violations & Framing Errors)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_exit_13_protocol_error() {
    let model_path = create_dummy_model("k3_worker_exit13_model.gguf");
    let sock_path = std::env::temp_dir().join("k3_worker_exit13.sock");
    if sock_path.exists() {
        let _ = std::fs::remove_file(&sock_path);
    }

    // 6.1 Send an unknown IPC message type ID (e.g. 999) to worker
    let mut child = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker");

    let mut stream_opt = None;
    for _ in 0..50 {
        if sock_path.exists() {
            if let Ok(s) = UnixStream::connect(&sock_path) {
                stream_opt = Some(s);
                break;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let stream = stream_opt.expect("connect to worker");

    // Send malformed / unexpected type ID 999 frame
    let garbage_payload = b"UNKNOWN_PROTOCOL_PAYLOAD";
    send_frame(stream.as_raw_fd(), 999, garbage_payload, None).expect("send invalid frame");

    let exit_status = child.wait().expect("wait for child after protocol error");
    assert_eq!(
        exit_status.code(),
        Some(13),
        "Unknown message type ID must terminate worker with exit code 13 (ProtocolError)"
    );

    // 6.2 ForwardPassCommand sent before AttachArenaCommand
    let sock_path2 = std::env::temp_dir().join("k3_worker_exit13_noarena.sock");
    if sock_path2.exists() {
        let _ = std::fs::remove_file(&sock_path2);
    }

    let mut child2 = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path2.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker 2");

    let mut stream_opt2 = None;
    for _ in 0..50 {
        if sock_path2.exists() {
            if let Ok(s) = UnixStream::connect(&sock_path2) {
                stream_opt2 = Some(s);
                break;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    let stream2 = stream_opt2.expect("connect to worker 2");

    let forward_cmd = ForwardPassCommand {
        request_id: 42,
        sequence_id: 1,
        token_offset: 0,
        token_count: 5,
        block_indices: vec![0],
        is_prefill: true,
    };
    let (fwd_type, fwd_payload) = IpcMessage::ForwardPassCommand(forward_cmd)
        .serialize(global_fory())
        .unwrap();

    send_frame(stream2.as_raw_fd(), fwd_type, &fwd_payload, None).expect("send forward command");

    let exit_status2 = child2
        .wait()
        .expect("wait for child 2 after premature forward command");
    assert_eq!(
        exit_status2.code(),
        Some(13),
        "ForwardPassCommand before AttachArenaCommand must terminate worker with exit code 13"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_file(&sock_path2);
}

// ----------------------------------------------------------------------------
// Challenge 7: Concurrent Workers with Shared Advisory Lock (LOCK_SH)
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_concurrent_workers_shared_lock() {
    let model_path = create_dummy_model("k3_worker_shared_concurrent_model.gguf");
    let m_str = model_path.to_str().unwrap();

    let sock1 = std::env::temp_dir().join("k3_worker_sh1.sock");
    let sock2 = std::env::temp_dir().join("k3_worker_sh2.sock");
    if sock1.exists() {
        let _ = std::fs::remove_file(&sock1);
    }
    if sock2.exists() {
        let _ = std::fs::remove_file(&sock2);
    }

    // Spawn Worker 1
    let mut child1 = Command::new(BIN_PATH)
        .args([
            "--model",
            m_str,
            "--socket",
            sock1.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker 1");

    // Spawn Worker 2 on the exact same model file
    let mut child2 = Command::new(BIN_PATH)
        .args([
            "--model",
            m_str,
            "--socket",
            sock2.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn worker 2");

    // Both should connect and coexist with LOCK_SH
    let mut s1_opt = None;
    let mut s2_opt = None;
    for _ in 0..50 {
        if s1_opt.is_none() && sock1.exists() {
            if let Ok(s) = UnixStream::connect(&sock1) {
                s1_opt = Some(s);
            }
        }
        if s2_opt.is_none() && sock2.exists() {
            if let Ok(s) = UnixStream::connect(&sock2) {
                s2_opt = Some(s);
            }
        }
        if s1_opt.is_some() && s2_opt.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }

    let s1 = s1_opt.expect("connect to worker 1");
    let s2 = s2_opt.expect("connect to worker 2");

    // Send Shutdown to both
    let shutdown = IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true });
    send_msg_frame(&s1, &shutdown, None).expect("shutdown 1");
    send_msg_frame(&s2, &shutdown, None).expect("shutdown 2");

    assert_eq!(child1.wait().unwrap().code(), Some(0));
    assert_eq!(child2.wait().unwrap().code(), Some(0));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&sock1);
    let _ = std::fs::remove_file(&sock2);
}

// ----------------------------------------------------------------------------
// Challenge 8: Full End-to-End Prefill, Decode, Heartbeat & Subprocess Lifecycles
// ----------------------------------------------------------------------------
#[test]
fn test_challenge_e2e_prefill_and_decode_pipeline() {
    let model_path = create_dummy_model("k3_worker_e2e_full_pipeline.gguf");
    let sock_path = std::env::temp_dir().join("k3_worker_e2e_full_pipeline.sock");
    if sock_path.exists() {
        let _ = std::fs::remove_file(&sock_path);
    }

    // 1. Create UnixListener from supervisor side
    let supervisor_listener = UnixListener::bind(&sock_path).expect("bind supervisor listener");

    // 2. Spawn worker in client mode (connecting to supervisor_listener)
    let mut child = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            sock_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--verbose",
        ])
        .spawn()
        .expect("spawn worker connecting to supervisor");

    // 3. Supervisor accepts connection from worker
    let (stream, _) = supervisor_listener
        .accept()
        .expect("supervisor accepted worker");

    // 4. Create and attach shared memory arena via SCM_RIGHTS
    let arena_size = 65536_usize;
    let mut arena =
        SharedMemoryArena::create("challenger_m4_arena", arena_size).expect("create arena");

    let header = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: arena_size as u64,
        block_count: 8,
        tokens_per_block: 16,
        n_ctx: 128,
        head_dim: 64,
        num_layers: 2,
        head_count_kv: 2,
        vocab_size: 16,
        active_slots: 1,
        meta_table_offset: 128,
        token_slots_offset: 1024,
        kv_blocks_offset: 4096,
        output_slabs_offset: 8192,
        generation_counter: 0,
        swa_window_size: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        derived_weights_offset: 0,
        _reserved: [0; 24],
    };
    let hdr_slice = arena.slice_mut(0, size_of::<ArenaHeader>()).unwrap();
    hdr_slice.copy_from_slice(header.as_bytes());

    let dup_fd = arena.duplicate_fd().expect("dup arena fd");
    let attach_cmd = AttachArenaCommand {
        arena_id: 200,
        arena_bytes: arena_size as u64,
        n_ctx: 128,
        max_blocks: 8,
    };
    let (cmd_type, payload) = IpcMessage::AttachArenaCommand(attach_cmd)
        .serialize(global_fory())
        .unwrap();

    send_frame(
        stream.as_raw_fd(),
        cmd_type,
        &payload,
        Some(dup_fd.as_raw_fd()),
    )
    .expect("send AttachArenaCommand with FD");

    // Receive AttachArenaResponse (Type ID 206)
    let (resp_hdr, resp_payload, _) =
        recv_frame(stream.as_raw_fd()).expect("recv AttachArena response");
    assert_eq!(resp_hdr.type_id, 206);
    let status = u32::from_le_bytes([
        resp_payload[0],
        resp_payload[1],
        resp_payload[2],
        resp_payload[3],
    ]);
    assert_eq!(status, StatusCode::Success as u32);

    // 5. Test Pass 1: Prefill Forward Pass (16 prompt tokens)
    let prompt_tokens: Vec<u32> = (1..=16).collect();
    let slot_hdr = TokenInputSlotHeader {
        slot_seq_id: 1,
        token_len: prompt_tokens.len() as u32,
        start_position: 0,
    };
    let slot_offset = 1024_usize;
    let slot_slice = arena.slice_mut(slot_offset, 16).unwrap();
    slot_slice.copy_from_slice(slot_hdr.as_bytes());

    let tokens_slice = arena
        .slice_mut(slot_offset + 16, prompt_tokens.len() * 4)
        .unwrap();
    for (i, &t) in prompt_tokens.iter().enumerate() {
        tokens_slice[i * 4..(i + 1) * 4].copy_from_slice(&t.to_ne_bytes());
    }

    let prefill_cmd = ForwardPassCommand {
        request_id: 5001,
        sequence_id: 1,
        token_offset: 0,
        token_count: prompt_tokens.len() as u32,
        block_indices: vec![0],
        is_prefill: true,
    };
    let (cmd_type, payload) = IpcMessage::ForwardPassCommand(prefill_cmd)
        .serialize(global_fory())
        .unwrap();

    // Verify zero bulk payload over UDS: command frame is < 128 bytes
    assert!(
        12 + payload.len() < 128,
        "Prefill frame must be compact control metadata"
    );
    send_frame(stream.as_raw_fd(), cmd_type, &payload, None).expect("send prefill command");

    let (msg, _) = recv_msg_frame(&stream).expect("recv prefill completed");
    match msg {
        IpcMessage::ForwardPassCompleted(done) => {
            assert_eq!(done.request_id, 5001);
            assert_eq!(done.committed_tokens, 16);
            assert!(done.prefill_latency_ms > 0.0);
            assert_eq!(done.decode_latency_ms, 0.0);
            assert_eq!(done.status, StatusCode::Success);
        }
        other => panic!("expected ForwardPassCompleted, got {other:?}"),
    }

    // Verify in-place memory mutations for prefill
    let block = arena.block_meta(0).unwrap();
    assert_eq!(block.committed_tokens(), 16);
    assert_eq!(block.tag, KvBlockTag::Persist.as_u8());
    assert_eq!(block.sampler_rng_state[0], 987654321);

    let slab_offset = 8192_usize;
    let slab_slice = arena.slice(slab_offset, 64).unwrap();
    let slab_hdr = OutputSlabHeader::ref_from_bytes(slab_slice).unwrap();
    assert_eq!(slab_hdr.request_id, 5001);
    assert_eq!(slab_hdr.vocab_size, 16);
    assert_eq!(slab_hdr.position, 16);
    assert_eq!(slab_hdr.status, SLAB_STATUS_WRITTEN);

    // 6. Test Pass 2: Autoregressive Decode Forward Pass (1 single token)
    let decode_token = [777_u32];
    let slot_hdr_dec = TokenInputSlotHeader {
        slot_seq_id: 2,
        token_len: 1,
        start_position: 16,
    };
    let slot_slice_dec = arena.slice_mut(slot_offset, 16).unwrap();
    slot_slice_dec.copy_from_slice(slot_hdr_dec.as_bytes());

    let tokens_slice_dec = arena.slice_mut(slot_offset + 16, 4).unwrap();
    tokens_slice_dec.copy_from_slice(&decode_token[0].to_ne_bytes());

    let decode_cmd = ForwardPassCommand {
        request_id: 5002,
        sequence_id: 1,
        token_offset: 16,
        token_count: 1,
        block_indices: vec![0],
        is_prefill: false,
    };
    let (cmd_type, payload) = IpcMessage::ForwardPassCommand(decode_cmd)
        .serialize(global_fory())
        .unwrap();

    send_frame(stream.as_raw_fd(), cmd_type, &payload, None).expect("send decode command");

    let (msg_dec, _) = recv_msg_frame(&stream).expect("recv decode completed");
    match msg_dec {
        IpcMessage::ForwardPassCompleted(done) => {
            assert_eq!(done.request_id, 5002);
            assert_eq!(done.committed_tokens, 1);
            assert_eq!(done.prefill_latency_ms, 0.0);
            assert!(done.decode_latency_ms > 0.0);
            assert_eq!(done.status, StatusCode::Success);
        }
        other => panic!("expected ForwardPassCompleted, got {other:?}"),
    }

    // Verify in-place memory mutations for decode
    let slab_slice_dec = arena.slice(slab_offset, 64).unwrap();
    let slab_hdr_dec = OutputSlabHeader::ref_from_bytes(slab_slice_dec).unwrap();
    assert_eq!(slab_hdr_dec.request_id, 5002);
    assert_eq!(slab_hdr_dec.position, 17); // 16 + 1
    assert_eq!(slab_hdr_dec.status, SLAB_STATUS_WRITTEN);

    // 7. Verify Heartbeat Request/Response
    let hb_req = IpcMessage::WorkerHeartbeatRequest(WorkerHeartbeatRequest {
        timestamp_ns: 1234567890,
    });
    send_msg_frame(&stream, &hb_req, None).expect("send heartbeat");

    let (msg_hb, _) = recv_msg_frame(&stream).expect("recv heartbeat");
    match msg_hb {
        IpcMessage::WorkerHeartbeatResponse(resp) => {
            assert_eq!(resp.timestamp_ns, 1234567890);
            assert_eq!(resp.completed_passes, 2); // Prefill + Decode
            assert!(!resp.is_busy);
        }
        other => panic!("expected WorkerHeartbeatResponse, got {other:?}"),
    }

    // 8. Graceful Shutdown
    let shutdown = IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true });
    send_msg_frame(&stream, &shutdown, None).expect("send shutdown");

    let exit_status = child.wait().expect("wait for child after shutdown");
    assert_eq!(
        exit_status.code(),
        Some(0),
        "Worker must exit 0 on ShutdownCommand"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&sock_path);
}
