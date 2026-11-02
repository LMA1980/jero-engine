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
use std::os::unix::net::UnixStream;
use std::thread;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN,
    TokenInputSlotHeader,
};
use k3_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, IpcMessage, ShutdownCommand, StatusCode,
    WorkerHeartbeatRequest, create_uds_socket_pair, global_fory, recv_frame, recv_msg_frame,
    send_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use k3_engine::worker::cli::WorkerConfig;
use k3_engine::worker::weights::LockedWeights;
use k3_engine::worker::worker::WorkerEngine;
use zerocopy::{FromBytes, IntoBytes};

#[test]
fn test_e2e_worker_engine_socket_loop_and_zerocopy_dispatch() {
    // 1. Prepare dummy model file
    let model_path = std::env::temp_dir().join("k3_worker_e2e_engine_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"SYNTHETIC_MODEL_WEIGHTS_PAYLOAD_V3").unwrap();
    }

    // 2. Prepare shared memory arena
    let arena_size = 32768_usize;
    let mut arena =
        SharedMemoryArena::create("e2e_engine_arena", arena_size).expect("create arena");

    // Initialize ArenaHeader at offset 0
    let header = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: arena_size as u64,
        block_count: 4,
        tokens_per_block: 16,
        n_ctx: 64,
        head_dim: 64,
        num_layers: 2,
        head_count_kv: 2,
        vocab_size: 10,
        active_slots: 1,
        meta_table_offset: 128,
        token_slots_offset: 512,
        kv_blocks_offset: 1024,
        output_slabs_offset: 2048,
        generation_counter: 0,
        swa_window_size: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        derived_weights_offset: 0,
        _reserved: [0; 24],
    };
    let hdr_slice = arena.slice_mut(0, size_of::<ArenaHeader>()).unwrap();
    hdr_slice.copy_from_slice(header.as_bytes());

    // 3. Create connected bidirectional Unix stream socketpair
    let (mgr_fd, worker_fd) = create_uds_socket_pair().expect("socketpair");
    let mgr_sock = UnixStream::from(mgr_fd);
    let worker_sock = UnixStream::from(worker_fd);

    // 4. Spawn worker engine in a background thread
    let config = WorkerConfig {
        socket_path: None,
        socket_fd: Some(worker_sock.as_raw_fd()),
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: None,
        affinity: None,
        verbose: true,
        listen: false,
    };
    let weights = LockedWeights::open(&model_path).expect("open weights");
    let mut engine = WorkerEngine::new(config, weights);

    let worker_handle = thread::spawn(move || engine.run_event_loop(worker_sock));

    // 5. Manager sends AttachArenaCommand (Type ID 201) with attached arena FD
    let dup_fd = arena.duplicate_fd().expect("dup arena fd");
    let attach_cmd = AttachArenaCommand {
        arena_id: 100,
        arena_bytes: arena_size as u64,
        n_ctx: 64,
        max_blocks: 4,
    };
    let (cmd_type, payload) = IpcMessage::AttachArenaCommand(attach_cmd)
        .serialize(global_fory())
        .unwrap();

    send_frame(
        mgr_sock.as_raw_fd(),
        cmd_type,
        &payload,
        Some(dup_fd.as_raw_fd()),
    )
    .expect("send AttachArenaCommand with FD");

    // Manager receives AttachArenaResponse (Type ID 206, StatusCode::Success)
    let (resp_hdr, resp_payload, _) =
        recv_frame(mgr_sock.as_raw_fd()).expect("recv AttachArena response");
    assert_eq!(resp_hdr.type_id, 206);
    let status_val = u32::from_le_bytes([
        resp_payload[0],
        resp_payload[1],
        resp_payload[2],
        resp_payload[3],
    ]);
    assert_eq!(status_val, StatusCode::Success as u32);

    // 6. Manager writes prompt tokens directly into shared memory TokenInputSlot
    let prompt_tokens = [101_u32, 2054, 2003, 1037, 7000]; // 5 prompt tokens
    let slot_hdr = TokenInputSlotHeader {
        slot_seq_id: 1,
        token_len: prompt_tokens.len() as u32,
        start_position: 0,
    };

    let slot_offset = 512_usize;
    let slot_hdr_slice = arena.slice_mut(slot_offset, 16).unwrap();
    slot_hdr_slice.copy_from_slice(slot_hdr.as_bytes());

    let tokens_slice = arena
        .slice_mut(slot_offset + 16, prompt_tokens.len() * 4)
        .unwrap();
    for (i, &t) in prompt_tokens.iter().enumerate() {
        tokens_slice[i * 4..(i + 1) * 4].copy_from_slice(&t.to_ne_bytes());
    }

    // 7. Manager sends ForwardPassCommand (Type ID 202) over UDS
    let forward_cmd = ForwardPassCommand {
        request_id: 1001,
        sequence_id: 1,
        token_offset: 0,
        token_count: prompt_tokens.len() as u32,
        block_indices: vec![0],
        is_prefill: true,
    };

    let (fwd_type, fwd_payload) = IpcMessage::ForwardPassCommand(forward_cmd)
        .serialize(global_fory())
        .unwrap();

    // Verify zero bulk tensor bytes: command frame sent over socket is compact (< 128 bytes)
    assert!(
        12 + fwd_payload.len() < 128,
        "ForwardPassCommand frame must be compact metadata; observed {} bytes",
        12 + fwd_payload.len()
    );

    send_frame(mgr_sock.as_raw_fd(), fwd_type, &fwd_payload, None)
        .expect("send ForwardPassCommand");

    // 8. Manager receives ForwardPassCompleted (Type ID 203) over UDS
    let (completed_msg, _) = recv_msg_frame(&mgr_sock).expect("recv ForwardPassCompleted");
    match completed_msg {
        IpcMessage::ForwardPassCompleted(done) => {
            assert_eq!(done.request_id, 1001);
            assert_eq!(done.sequence_id, 1);
            assert_eq!(done.committed_tokens, 5);
            assert_eq!(done.status, StatusCode::Success);
            assert!(done.prefill_latency_ms > 0.0);
            assert_eq!(done.decode_latency_ms, 0.0);
            assert_eq!(done.error_message, None);
        }
        other => panic!("expected ForwardPassCompleted, got {other:?}"),
    }

    // 9. Verify in-place mutations in shared memory
    // Assertion 1: KvBlockMeta updated in place
    let block = arena.block_meta(0).expect("block 0");
    assert_eq!(block.committed_tokens(), 5);
    assert_eq!(block.tag, KvBlockTag::Persist.as_u8());
    assert_eq!(block.sampler_rng_state[0], 987654321);

    // Assertion 2: OutputSlabHeader updated in place
    let slab_offset = 2048_usize;
    let slab_slice = arena.slice(slab_offset, 64).unwrap();
    let slab_hdr = OutputSlabHeader::ref_from_bytes(slab_slice).unwrap();
    assert_eq!(slab_hdr.request_id, 1001);
    assert_eq!(slab_hdr.vocab_size, 10);
    assert_eq!(slab_hdr.position, 5);
    assert_eq!(slab_hdr.status, SLAB_STATUS_WRITTEN);

    // Assertion 3: Output logits match reference computation bit-for-bit
    let mut expected_logits = vec![0.0_f32; 10];
    for (i, val) in expected_logits.iter_mut().enumerate() {
        *val = (prompt_tokens[i % prompt_tokens.len()] as f32) * 0.01 + (i as f32) * 0.1;
    }

    let logits_slice = arena.slice(slab_offset + 64, 10 * 4).unwrap();
    let mut read_logits = vec![0.0_f32; 10];
    for (i, val) in read_logits.iter_mut().enumerate() {
        *val = f32::from_ne_bytes([
            logits_slice[i * 4],
            logits_slice[i * 4 + 1],
            logits_slice[i * 4 + 2],
            logits_slice[i * 4 + 3],
        ]);
    }
    assert_eq!(
        read_logits, expected_logits,
        "Logits must match reference computation bit-for-bit"
    );

    // 10. Test WorkerHeartbeatRequest -> WorkerHeartbeatResponse
    let hb_req = IpcMessage::WorkerHeartbeatRequest(WorkerHeartbeatRequest {
        timestamp_ns: 9988776655,
    });
    send_msg_frame(&mgr_sock, &hb_req, None).expect("send heartbeat request");

    let (hb_resp, _) = recv_msg_frame(&mgr_sock).expect("recv heartbeat response");
    match hb_resp {
        IpcMessage::WorkerHeartbeatResponse(resp) => {
            assert_eq!(resp.timestamp_ns, 9988776655);
            assert_eq!(resp.completed_passes, 1);
            assert!(!resp.is_busy);
        }
        other => panic!("expected WorkerHeartbeatResponse, got {other:?}"),
    }

    // 11. Test ShutdownCommand -> Graceful exit 0
    let shutdown_cmd = IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true });
    send_msg_frame(&mgr_sock, &shutdown_cmd, None).expect("send shutdown command");

    let thread_res = worker_handle.join().expect("join worker thread");
    assert!(
        thread_res.is_ok(),
        "Worker engine must exit cleanly with Ok(())"
    );

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_e2e_worker_subprocess_lifecycle_and_shutdown() {
    let bin_path = env!("CARGO_BIN_EXE_k3-worker");

    // 1. Prepare dummy model file
    let model_path = std::env::temp_dir().join("k3_worker_e2e_subproc_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"SYNTHETIC_MODEL_WEIGHTS_SUBPROC").unwrap();
    }

    // 2. Prepare UDS socket path for communication
    let socket_path = std::env::temp_dir().join("k3_worker_subproc.sock");
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    // 3. Spawn child k3-worker process in listener mode
    let mut child = std::process::Command::new(bin_path)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            socket_path.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
            "--verbose",
        ])
        .spawn()
        .expect("spawn k3-worker child process");

    // 4. Connect to child worker listener
    let mut stream_opt = None;
    for _ in 0..50 {
        if socket_path.exists() {
            if let Ok(s) = UnixStream::connect(&socket_path) {
                stream_opt = Some(s);
                break;
            }
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }

    let stream = stream_opt.expect("connected to k3-worker child process");

    // 5. Send ShutdownCommand to verify graceful termination (exit code 0)
    let shutdown_cmd = IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true });
    send_msg_frame(&stream, &shutdown_cmd, None).expect("send ShutdownCommand to child");

    let exit_status = child.wait().expect("wait for child process");
    assert_eq!(
        exit_status.code(),
        Some(0),
        "Child worker must exit cleanly with code 0"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&socket_path);
}
