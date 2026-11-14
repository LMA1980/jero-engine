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

use jero_core::QuantType;
use jero_core::gguf::{GGUF_MAGIC, GGUF_VERSION_V3, align_up};
use jero_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN,
    TokenInputSlotHeader,
};
use jero_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, IpcMessage, ShutdownCommand, StatusCode,
    WorkerHeartbeatRequest, create_uds_socket_pair, global_fory, recv_frame, recv_msg_frame,
    send_frame, send_msg_frame,
};
use jero_core::ipc::shm::SharedMemoryArena;
use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_engine::affinity::ClusterPhasePolicy;
use jero_engine::capability::verify_model_capabilities;
use jero_engine::engine::{SinglePromptEngine, StreamAction};
use jero_engine::kv::SingleSlotKvArena;
use jero_engine::weights::LockedModelWeights;
use jero_engine::worker::cli::WorkerConfig;
use jero_engine::worker::weights::LockedWeights;
use jero_engine::worker::worker::WorkerEngine;
use zerocopy::{FromBytes, IntoBytes};

#[test]
fn test_e2e_worker_engine_socket_loop_and_zerocopy_dispatch() {
    // 1. Prepare miniature valid GGUF v3 model file
    let model_path = std::env::temp_dir().join("k3_worker_e2e_engine_model.gguf");
    let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(&gguf_bytes).unwrap();
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
        vocab_size: 16,
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
    let worker_next_token = match completed_msg {
        IpcMessage::ForwardPassCompleted(done) => {
            assert_eq!(done.request_id, 1001);
            assert_eq!(done.sequence_id, 1);
            assert_eq!(done.committed_tokens, 5);
            assert_eq!(done.status, StatusCode::Success);
            assert!(done.prefill_latency_ms > 0.0);
            assert_eq!(done.decode_latency_ms, 0.0);
            assert_eq!(done.error_message, None);
            done.next_token
        }
        other => panic!("expected ForwardPassCompleted, got {other:?}"),
    };

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
    assert_eq!(slab_hdr.vocab_size, 16);
    assert_eq!(slab_hdr.position, 5);
    assert_eq!(slab_hdr.status, SLAB_STATUS_WRITTEN);

    // Assertion 3: Output logits and sampled token match direct SinglePromptEngine bit-for-bit
    let ref_weights = LockedModelWeights::open(&model_path).expect("ref open");
    let ref_gguf = jero_core::GgufFile::parse(ref_weights.as_bytes()).expect("ref gguf");
    let ref_recipe = verify_model_capabilities(&ref_gguf).expect("ref recipe");
    let mut ref_kv = SingleSlotKvArena::from_gguf(&ref_gguf, 64).expect("ref kv");
    let mut ref_engine = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);
    let (expected_logits, expected_next_token, expected_pos) = ref_engine
        .prefill_step_with_pool(
            &*ref_recipe,
            &ref_weights,
            &mut ref_kv,
            None,
            &prompt_tokens,
            0,
            16,
        )
        .expect("ref prefill");
    assert_eq!(expected_pos, 5);
    assert_eq!(worker_next_token, expected_next_token);

    let logits_slice = arena.slice(slab_offset + 64, 16 * 4).unwrap();
    let mut read_logits = vec![0.0_f32; 16];
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
        "Logits must match SinglePromptEngine bit-for-bit"
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
    let bin_path = env!("CARGO_BIN_EXE_jero-worker");

    // 1. Prepare miniature valid GGUF v3 model file
    let model_path = std::env::temp_dir().join("k3_worker_e2e_subproc_model.gguf");
    let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(&gguf_bytes).unwrap();
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

#[cfg_attr(miri, ignore)]
#[test]
fn test_e2e_worker_binary_llama_ipc_matches_single_prompt_engine_bit_for_bit() {
    use jero_core::gguf::test_helpers::write_gguf_value;
    use jero_core::ipc::{
        AttachArenaCommand, ForwardPassCommand, IpcMessage, OutputSlabHeader, ShutdownCommand,
        StatusCode, TokenInputSlotHeader, recv_msg_frame, send_msg_frame,
    };
    use jero_core::{GgufArray, GgufFile, GgufValue};
    use jero_engine::{
        ClusterPhasePolicy, LockedModelWeights, SinglePromptEngine, SingleSlotKvArena,
        verify_model_capabilities,
    };
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::process::Command;
    use std::time::Duration;

    const DIM: usize = 4;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const HIDDEN_DIM: usize = 8;
    const VOCAB_SIZE: usize = 8;
    const CONTEXT_LEN: usize = 64;

    fn build_miniature_llama_gguf_v3() -> Vec<u8> {
        fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        let token_strings: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("tok{i}")).collect();
        let token_refs: Vec<&str> = token_strings.iter().map(String::as_str).collect();

        let kvs: Vec<(&str, GgufValue<'_>)> = vec![
            ("general.architecture", GgufValue::String("llama")),
            ("general.name", GgufValue::String("mini-llama-ipc-parity")),
            (
                "llama.context_length",
                GgufValue::Uint32(CONTEXT_LEN as u32),
            ),
            ("llama.embedding_length", GgufValue::Uint32(DIM as u32)),
            ("llama.block_count", GgufValue::Uint32(1)),
            (
                "llama.feed_forward_length",
                GgufValue::Uint32(HIDDEN_DIM as u32),
            ),
            (
                "llama.attention.head_count",
                GgufValue::Uint32(HEADS as u32),
            ),
            (
                "llama.attention.head_count_kv",
                GgufValue::Uint32(KV_HEADS as u32),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(1e-5),
            ),
            (
                "llama.rope.dimension_count",
                GgufValue::Uint32(HEAD_DIM as u32),
            ),
            ("llama.rope.freq_base", GgufValue::Float32(10000.0)),
            ("tokenizer.ggml.model", GgufValue::String("llama")),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(token_refs)),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
        ];

        struct TensorSpec {
            name: &'static str,
            dims: &'static [u64],
            is_norm: bool,
            salt: f32,
        }

        let specs = [
            TensorSpec {
                name: "token_embd.weight",
                dims: &[DIM as u64, VOCAB_SIZE as u64],
                is_norm: false,
                salt: 0.11,
            },
            TensorSpec {
                name: "blk.0.attn_norm.weight",
                dims: &[DIM as u64],
                is_norm: true,
                salt: 0.02,
            },
            TensorSpec {
                name: "blk.0.attn_q.weight",
                dims: &[DIM as u64, DIM as u64],
                is_norm: false,
                salt: 0.23,
            },
            TensorSpec {
                name: "blk.0.attn_k.weight",
                dims: &[DIM as u64, DIM as u64],
                is_norm: false,
                salt: 0.37,
            },
            TensorSpec {
                name: "blk.0.attn_v.weight",
                dims: &[DIM as u64, DIM as u64],
                is_norm: false,
                salt: 0.41,
            },
            TensorSpec {
                name: "blk.0.attn_output.weight",
                dims: &[DIM as u64, DIM as u64],
                is_norm: false,
                salt: 0.53,
            },
            TensorSpec {
                name: "blk.0.ffn_norm.weight",
                dims: &[DIM as u64],
                is_norm: true,
                salt: 0.03,
            },
            TensorSpec {
                name: "blk.0.ffn_gate.weight",
                dims: &[DIM as u64, HIDDEN_DIM as u64],
                is_norm: false,
                salt: 0.61,
            },
            TensorSpec {
                name: "blk.0.ffn_up.weight",
                dims: &[DIM as u64, HIDDEN_DIM as u64],
                is_norm: false,
                salt: 0.71,
            },
            TensorSpec {
                name: "blk.0.ffn_down.weight",
                dims: &[HIDDEN_DIM as u64, DIM as u64],
                is_norm: false,
                salt: 0.83,
            },
            TensorSpec {
                name: "output_norm.weight",
                dims: &[DIM as u64],
                is_norm: true,
                salt: 0.04,
            },
            TensorSpec {
                name: "output.weight",
                dims: &[DIM as u64, VOCAB_SIZE as u64],
                is_norm: false,
                salt: 0.97,
            },
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        buf.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

        for (k, v) in &kvs {
            push_gguf_str(&mut buf, k);
            write_gguf_value(&mut buf, v);
        }

        let mut tensor_bytes = Vec::new();
        for spec in &specs {
            let rel_offset = align_up(tensor_bytes.len(), 32);
            tensor_bytes.resize(rel_offset, 0);
            push_gguf_str(&mut buf, spec.name);
            buf.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
            for &d in spec.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&QuantType::F32.ggml_type_id().to_le_bytes());
            buf.extend_from_slice(&(rel_offset as u64).to_le_bytes());

            let numel: usize = spec.dims.iter().map(|&d| d as usize).product();
            for i in 0..numel {
                let val = if spec.is_norm {
                    1.0 + (i as f32) * spec.salt
                } else {
                    ((i as f32 + 1.0) * spec.salt).sin() * 0.35 + ((i % 5) as f32 - 2.0) * 0.04
                };
                tensor_bytes.extend_from_slice(&val.to_le_bytes());
            }
        }

        let data_offset = align_up(buf.len(), 32);
        buf.resize(data_offset, 0);
        buf.extend_from_slice(&tensor_bytes);
        buf
    }

    fn write_tokens_to_slot(
        arena: &mut SharedMemoryArena,
        slot_offset: usize,
        slot_seq_id: u64,
        start_position: u32,
        tokens: &[u32],
    ) {
        let hdr = TokenInputSlotHeader {
            slot_seq_id,
            token_len: tokens.len() as u32,
            start_position,
        };
        let slice = arena.as_mut_slice();
        let hdr_bytes: &[u8] = IntoBytes::as_bytes(&hdr);
        slice[slot_offset..slot_offset + hdr_bytes.len()].copy_from_slice(hdr_bytes);
        let tok_base = slot_offset + size_of::<TokenInputSlotHeader>();
        for (i, &tok) in tokens.iter().enumerate() {
            let off = tok_base + i * 4;
            slice[off..off + 4].copy_from_slice(&tok.to_le_bytes());
        }
    }

    fn read_output_logits(
        arena: &SharedMemoryArena,
        slab_offset: usize,
        vocab_size: usize,
    ) -> Vec<f32> {
        let slice = arena.as_slice();
        let logit_base = slab_offset + size_of::<OutputSlabHeader>();
        let mut out = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let off = logit_base + i * 4;
            out.push(f32::from_le_bytes([
                slice[off],
                slice[off + 1],
                slice[off + 2],
                slice[off + 3],
            ]));
        }
        out
    }

    let pid = std::process::id();
    let model_path = std::env::temp_dir().join(format!("test_e2e_llama_parity_{pid}.gguf"));
    let socket_path = std::env::temp_dir().join(format!("test_e2e_llama_parity_{pid}.sock"));
    let shm_name = format!("e2e_llama_p_{pid}");
    let _ = std::fs::remove_file(&socket_path);

    let gguf_bytes = build_miniature_llama_gguf_v3();
    std::fs::write(&model_path, &gguf_bytes).expect("write miniature llama GGUF");

    let arena_bytes: usize = 128 * 1024;
    let mut arena = SharedMemoryArena::create(&shm_name, arena_bytes).expect("create SHM arena");
    let header = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: arena_bytes as u64,
        block_count: 4,
        tokens_per_block: 16,
        n_ctx: CONTEXT_LEN as u32,
        head_dim: HEAD_DIM as u32,
        num_layers: 1,
        head_count_kv: KV_HEADS as u32,
        vocab_size: VOCAB_SIZE as u32,
        active_slots: 1,
        meta_table_offset: 128,
        token_slots_offset: 512,
        kv_blocks_offset: 2048,
        output_slabs_offset: 65536,
        ..ArenaHeader::default()
    };
    arena.as_mut_slice()[..size_of::<ArenaHeader>()].copy_from_slice(IntoBytes::as_bytes(&header));

    let worker_bin = env!("CARGO_BIN_EXE_jero-worker");
    let mut child = Command::new(worker_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            socket_path.to_str().unwrap(),
            "--listen",
            "--cluster",
            "x100",
        ])
        .spawn()
        .expect("spawn jero-worker subprocess");

    let mut stream = None;
    for _ in 0..80 {
        if let Ok(s) = UnixStream::connect(&socket_path) {
            stream = Some(s);
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let stream = stream.expect("connect to jero-worker UNIX socket");

    // 1. AttachArenaCommand
    let attach = AttachArenaCommand {
        arena_id: 1,
        arena_bytes: arena_bytes as u64,
        n_ctx: CONTEXT_LEN as u32,
        max_blocks: 4,
    };
    send_msg_frame(
        &stream,
        &IpcMessage::AttachArenaCommand(attach),
        Some(arena.as_raw_fd()),
    )
    .expect("send AttachArenaCommand");
    let (attach_hdr, attach_ack, _) = recv_frame(stream.as_raw_fd()).expect("recv AttachArena ack");
    assert_eq!(attach_hdr.flags & jero_core::ipc::FLAG_ERROR, 0);
    assert_eq!(attach_ack.first().copied(), Some(0));

    // 2. Direct SinglePromptEngine reference on the exact same GGUF & policy
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse miniature llama GGUF");
    let recipe = verify_model_capabilities(&gguf).expect("verify miniature llama capabilities");
    let direct_weights = LockedModelWeights::from_file(File::open(&model_path).unwrap())
        .expect("load direct LockedModelWeights");
    let mut direct_kv =
        SingleSlotKvArena::new(gguf.metadata(), CONTEXT_LEN).expect("create direct KV arena");
    let mut direct_engine =
        SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);

    let prompt = [1u32, 2u32, 3u32];

    // Prefill IPC vs Direct
    write_tokens_to_slot(
        &mut arena,
        header.token_slots_offset as usize,
        1,
        0,
        &prompt,
    );
    let prefill_cmd = ForwardPassCommand {
        request_id: 1,
        sequence_id: 1,
        token_offset: 0,
        token_count: prompt.len() as u32,
        block_indices: vec![0],
        is_prefill: true,
    };
    send_msg_frame(&stream, &IpcMessage::ForwardPassCommand(prefill_cmd), None)
        .expect("send prefill ForwardPassCommand");
    let (prefill_msg, _) = recv_msg_frame(&stream).expect("recv prefill response");
    let IpcMessage::ForwardPassCompleted(prefill_resp) = prefill_msg else {
        panic!("expected ForwardPassCompleted, got {prefill_msg:?}");
    };
    assert_eq!(prefill_resp.status, StatusCode::Success);

    let (direct_prefill_logits, direct_tok1, direct_pos1) = direct_engine
        .prefill_step_with_pool(
            recipe.as_ref(),
            &direct_weights,
            &mut direct_kv,
            None,
            &prompt,
            0,
            VOCAB_SIZE,
        )
        .expect("direct prefill_step_with_pool");
    assert_eq!(direct_pos1, prompt.len());
    assert_eq!(prefill_resp.next_token, direct_tok1);

    let ipc_prefill_logits =
        read_output_logits(&arena, header.output_slabs_offset as usize, VOCAB_SIZE);
    for (idx, &expected) in direct_prefill_logits.iter().enumerate() {
        let actual = ipc_prefill_logits[idx];
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "prefill logit[{idx}] bit-exact mismatch: ipc={actual} vs direct={expected}"
        );
    }

    // Decode step 1 IPC vs Direct
    write_tokens_to_slot(
        &mut arena,
        header.token_slots_offset as usize,
        1,
        prompt.len() as u32,
        &[direct_tok1],
    );
    let decode1_cmd = ForwardPassCommand {
        request_id: 2,
        sequence_id: 1,
        token_offset: 0,
        token_count: 1,
        block_indices: vec![0],
        is_prefill: false,
    };
    send_msg_frame(&stream, &IpcMessage::ForwardPassCommand(decode1_cmd), None)
        .expect("send decode1 ForwardPassCommand");
    let (decode1_msg, _) = recv_msg_frame(&stream).expect("recv decode1 response");
    let IpcMessage::ForwardPassCompleted(decode1_resp) = decode1_msg else {
        panic!("expected ForwardPassCompleted, got {decode1_msg:?}");
    };
    assert_eq!(decode1_resp.status, StatusCode::Success);

    let (direct_decode1_logits, direct_tok2, direct_pos2) = direct_engine
        .decode_step_with_pool(
            recipe.as_ref(),
            &direct_weights,
            &mut direct_kv,
            None,
            direct_tok1,
            prompt.len(),
            VOCAB_SIZE,
        )
        .expect("direct decode step 1");
    assert_eq!(direct_pos2, prompt.len() + 1);
    assert_eq!(decode1_resp.next_token, direct_tok2);
    let ipc_decode1_logits =
        read_output_logits(&arena, header.output_slabs_offset as usize, VOCAB_SIZE);
    for (idx, &expected) in direct_decode1_logits.iter().enumerate() {
        let actual = ipc_decode1_logits[idx];
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "decode1 logit[{idx}] bit-exact mismatch: ipc={actual} vs direct={expected}"
        );
    }

    // Decode step 2 IPC vs Direct
    write_tokens_to_slot(
        &mut arena,
        header.token_slots_offset as usize,
        1,
        (prompt.len() + 1) as u32,
        &[direct_tok2],
    );
    let decode2_cmd = ForwardPassCommand {
        request_id: 3,
        sequence_id: 1,
        token_offset: 0,
        token_count: 1,
        block_indices: vec![0],
        is_prefill: false,
    };
    send_msg_frame(&stream, &IpcMessage::ForwardPassCommand(decode2_cmd), None)
        .expect("send decode2 ForwardPassCommand");
    let (decode2_msg, _) = recv_msg_frame(&stream).expect("recv decode2 response");
    let IpcMessage::ForwardPassCompleted(decode2_resp) = decode2_msg else {
        panic!("expected ForwardPassCompleted, got {decode2_msg:?}");
    };
    assert_eq!(decode2_resp.status, StatusCode::Success);

    let (direct_decode2_logits, direct_tok3, direct_pos3) = direct_engine
        .decode_step_with_pool(
            recipe.as_ref(),
            &direct_weights,
            &mut direct_kv,
            None,
            direct_tok2,
            prompt.len() + 1,
            VOCAB_SIZE,
        )
        .expect("direct decode step 2");
    assert_eq!(direct_pos3, prompt.len() + 2);
    assert_eq!(decode2_resp.next_token, direct_tok3);
    let ipc_decode2_logits =
        read_output_logits(&arena, header.output_slabs_offset as usize, VOCAB_SIZE);
    for (idx, &expected) in direct_decode2_logits.iter().enumerate() {
        let actual = ipc_decode2_logits[idx];
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "decode2 logit[{idx}] bit-exact mismatch: ipc={actual} vs direct={expected}"
        );
    }

    // Verify full evaluate_prompt matches the exact token trajectory [direct_tok1, direct_tok2, direct_tok3]
    let mut streamed = Vec::new();
    let mut eval_kv =
        SingleSlotKvArena::new(gguf.metadata(), CONTEXT_LEN).expect("create eval KV arena");
    let mut eval_engine = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);
    let summary = eval_engine
        .evaluate_prompt(
            recipe.as_ref(),
            &direct_weights,
            &mut eval_kv,
            &prompt,
            VOCAB_SIZE,
            3,
            |tok| {
                streamed.push(tok);
                Ok(StreamAction::Continue)
            },
        )
        .expect("direct evaluate_prompt");
    assert_eq!(
        summary.generated_tokens,
        vec![direct_tok1, direct_tok2, direct_tok3]
    );
    assert_eq!(streamed, vec![direct_tok1, direct_tok2, direct_tok3]);

    // Shutdown cleanly
    send_msg_frame(
        &stream,
        &IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true }),
        None,
    )
    .expect("send ShutdownCommand");
    let exit_status = child.wait().expect("wait for worker child");
    assert_eq!(exit_status.code(), Some(0));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&socket_path);
}
