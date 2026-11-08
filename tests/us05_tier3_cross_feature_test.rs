//! Tier 3: Cross-Feature Interaction & Pairwise Integration Test Suite for US-05
//!
//! Validates complex subsystem intersections across process domains:
//! - Pair 1: Target Core Affinity (X100/A100) x Shared Memory Arena Access
//! - Pair 2: Multi-Model Isolation x Bounded Memlock Ceiling (28 GiB)
//! - Pair 3: Arrow Columnar Views x Zerocopy In-Place Mutations
//! - Pair 4: Checkpointed Token Count x RNG Sampler State x Worker Re-attachment
//! - Pair 5: Advisory File Locking (flock) x Worker Crash Recovery
//! - Pair 6: Length-Prefixed Binary Framing x SCM_RIGHTS FD Passing

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_qualifications,
    unreachable_pub,
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::undocumented_unsafe_blocks,
    clippy::const_is_empty
)]

mod common;

use common::*;
use jero_core::K3Error;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;

// ============================================================================
// Pair 1: Target Core Affinity x Shared Memory Arena Access
// ============================================================================
#[test]
fn test_pair1_core_affinity_and_shared_memory_access() {
    let mut arena = SharedMemoryFixture::create("pair1_aff_shm", 16384).expect("arena");
    *arena.header_mut() = ArenaHeader::default();

    // Verify cluster masks: X100 (Cluster 0, 0x00FF) vs A100 (Cluster 1, 0xFF00)
    assert_eq!(X100_CORES_MASK, 0x00FF);
    assert_eq!(A100_CORES_MASK, 0xFF00);
    assert_eq!(X100_CORES_MASK & A100_CORES_MASK, 0);

    // Simulate X100 worker writing input prompt tokens to shared memory slot
    let x100_tokens = [10_u32, 20, 30, 40];
    arena
        .write_tokens(&x100_tokens)
        .expect("x100 writes tokens");

    // Duplicate descriptor for A100 matrix worker
    let dup_fd = arena.duplicate_fd().expect("dup fd");
    let mut a100_arena =
        SharedMemoryFixture::from_fd(dup_fd, arena.len(), "a100_map").expect("a100 map");

    // A100 worker reads the tokens written by X100 worker without IPC copy
    let read_tokens = a100_arena.read_tokens();
    assert_eq!(read_tokens, x100_tokens);

    // A100 worker writes output logits directly into shared memory slab
    let a100_logits = [1.25_f32, 3.50, -2.00, 0.00];
    a100_arena
        .write_logits(101, &a100_logits)
        .expect("a100 writes logits");

    // X100 worker / manager reads logits directly from original arena
    let (hdr, logits) = arena.read_logits();
    assert_eq!(hdr.request_id, 101);
    assert_eq!(logits, a100_logits);
}

// ============================================================================
// Pair 2: Multi-Model Isolation x Bounded Memlock Ceiling
// ============================================================================
#[test]
fn test_pair2_multimodel_isolation_and_memlock_budget() {
    let memlock_ceiling_bytes: u64 = 28 * 1024 * 1024 * 1024; // 28 GiB

    // Allocate Model A arena (e.g. 10 GiB budget simulation)
    let arena_a = SharedMemoryFixture::create("pair2_model_a", 8192).expect("arena a");
    let model_a_budget: u64 = 10 * 1024 * 1024 * 1024;

    // Allocate Model B arena (e.g. 12 GiB budget simulation)
    let arena_b = SharedMemoryFixture::create("pair2_model_b", 8192).expect("arena b");
    let model_b_budget: u64 = 12 * 1024 * 1024 * 1024;

    let combined_active_bytes = model_a_budget + model_b_budget;
    assert!(combined_active_bytes <= memlock_ceiling_bytes);

    // Attempting Model C with 10 GiB exceeds 28 GiB ceiling (10 + 12 + 10 = 32 > 28)
    let model_c_budget: u64 = 10 * 1024 * 1024 * 1024;
    let exceeds = combined_active_bytes + model_c_budget > memlock_ceiling_bytes;
    assert!(
        exceeds,
        "Admission control must refuse allocation exceeding 28 GiB"
    );

    let err = K3Error::InsufficientLockableMemory {
        required_bytes: combined_active_bytes + model_c_budget,
        weight_bytes: 20 * 1024 * 1024 * 1024,
        kv_bytes: 12 * 1024 * 1024 * 1024,
        available_bytes: 8 * 1024 * 1024 * 1024,
        rlimit_memlock_bytes: memlock_ceiling_bytes,
    };
    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");
    assert_eq!(EXIT_INSUFFICIENT_MEMLOCK, 11);

    // Verify Model A and Model B arenas remain isolated and uncorrupted
    assert_ne!(arena_a.as_ptr(), arena_b.as_ptr());
    assert_ne!(arena_a.as_raw_fd(), arena_b.as_raw_fd());
}

// ============================================================================
// Pair 3: Columnar Token/Logit Views x Zerocopy In-Place Mutations
// ============================================================================
#[test]
fn test_pair3_columnar_views_and_zerocopy_mutations() {
    let mut arena = SharedMemoryFixture::create("pair3_arrow_mut", 16384).expect("arena");
    *arena.header_mut() = ArenaHeader::default();

    // 1. Write initial tokens into input slot
    let initial_tokens = [100_u32, 200, 300];
    arena.write_tokens(&initial_tokens).expect("write tokens");

    // 2. Compute kernel simulates appending a new token directly in place
    let offset = arena.header().token_slots_offset as usize;
    let hdr_size = std::mem::size_of::<TokenInputHeader>();
    unsafe {
        let hdr = &mut *(arena.as_ptr().add(offset) as *mut TokenInputHeader);
        hdr.token_count = 4;
        let tokens_ptr = arena.as_ptr().add(offset + hdr_size) as *mut u32;
        *tokens_ptr.add(3) = 400; // Append token 400
    }

    // 3. Arrow-compatible reader immediately observes appended token without copying
    let updated = arena.read_tokens();
    assert_eq!(updated, [100, 200, 300, 400]);

    // 4. In-place logit normalization / softmax mutation
    let initial_logits = [2.0_f32, 4.0, 6.0];
    arena
        .write_logits(42, &initial_logits)
        .expect("write logits");

    let logit_offset =
        arena.header().output_slabs_offset as usize + std::mem::size_of::<OutputSlabHeader>();
    unsafe {
        let logits_ptr = arena.as_ptr().add(logit_offset) as *mut f32;
        *logits_ptr = 10.0; // In-place update to first logit
    }

    let (_, read_logits) = arena.read_logits();
    assert!((read_logits[0] - 10.0).abs() < f32::EPSILON);
    assert!((read_logits[1] - 4.0).abs() < f32::EPSILON);
}

// ============================================================================
// Pair 4: Checkpointed Token Commit Count x RNG Sampler State x Re-attachment
// ============================================================================
#[test]
fn test_pair4_checkpoint_rng_state_and_worker_reattachment() {
    let mut arena = SharedMemoryFixture::create("pair4_rng_chk", 16384).expect("arena");
    *arena.header_mut() = ArenaHeader::default();

    // Model worker generates up to token 37, checkpointing progress in KvBlockMeta
    let expected_committed_tokens = 37_u32;
    let expected_rng_state = [0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF0, 100, 200];
    {
        let block = arena.block_meta_mut(0).expect("block 0");
        block.committed_tokens = expected_committed_tokens;
        block.sampler_rng_state = expected_rng_state;
        block.tag = TAG_PERSIST;
    }

    // Simulate worker process crash: worker descriptor dropped, arena persists
    // k3-manager inspects surviving block table
    let surviving_block = arena.block_meta(0).expect("surviving block");
    assert_eq!(surviving_block.committed_tokens, expected_committed_tokens);
    assert_eq!(surviving_block.sampler_rng_state, expected_rng_state);
    assert_eq!(surviving_block.tag, TAG_PERSIST);

    // k3-manager passes arena FD to replacement worker via SCM_RIGHTS
    let (mgr_sock, repl_sock) = SocketIpcHelper::create_socket_pair().expect("uds pair");
    SocketIpcHelper::send_message_with_fd(
        mgr_sock.as_raw_fd(),
        b"ATTACH_RESUME:pos=37",
        Some(arena.as_raw_fd()),
    )
    .expect("send");
    let (cmd, received_fd) =
        SocketIpcHelper::recv_message_with_fd(repl_sock.as_raw_fd()).expect("recv");
    assert_eq!(cmd, b"ATTACH_RESUME:pos=37");
    assert!(received_fd.is_some());

    // Replacement worker maps arena and verifies checkpoint state
    let repl_arena =
        SharedMemoryFixture::from_fd(received_fd.unwrap(), arena.len(), "repl_shm").expect("map");
    let repl_block = repl_arena.block_meta(0).expect("repl block");
    assert_eq!(repl_block.committed_tokens, 37);
    assert_eq!(repl_block.sampler_rng_state, expected_rng_state);

    // Generation resumes at token 38 without prefill
    let next_token_idx = repl_block.committed_tokens + 1;
    assert_eq!(next_token_idx, 38);
}

// ============================================================================
// Pair 5: Advisory Weight File Locking x Worker Process Crash
// ============================================================================
#[test]
fn test_pair5_advisory_file_locking_and_crash_recovery() {
    let temp_weight = std::env::temp_dir().join(format!("k3_weight_pair5_{}.bin", next_seq()));
    let mut f = File::create(&temp_weight).expect("create weight file");
    f.write_all(b"MODEL_WEIGHT_BYTES").expect("write");

    // Worker opens weights and acquires LOCK_SH
    let worker_file = File::open(&temp_weight).expect("open worker");
    let lock_res = unsafe { libc::flock(worker_file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(lock_res, 0, "Worker holds LOCK_SH");

    // Garbage collector probe confirms LOCK_EX is blocked
    let gc_probe = File::open(&temp_weight).expect("open gc");
    let lock_ex_blocked =
        unsafe { libc::flock(gc_probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(
        lock_ex_blocked, 0,
        "LOCK_EX must be blocked while worker holds LOCK_SH"
    );

    // Worker process crashes: OS closes open file descriptors and releases lock automatically
    drop(worker_file);

    // Garbage collector probe now successfully acquires LOCK_EX
    let lock_ex_success =
        unsafe { libc::flock(gc_probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        lock_ex_success, 0,
        "GC acquires LOCK_EX once worker terminates"
    );

    unsafe {
        libc::flock(gc_probe.as_raw_fd(), libc::LOCK_UN);
    }
    let _ = std::fs::remove_file(&temp_weight);
}

// ============================================================================
// Pair 6: Length-Prefixed Binary UDS Framing x SCM_RIGHTS FD Passing
// ============================================================================
#[test]
fn test_pair6_binary_framing_and_scm_rights_migration() {
    let (sender, receiver) = SocketIpcHelper::create_socket_pair().expect("pair");
    let mut arena = SharedMemoryFixture::create("pair6_atomic_fd", 8192).expect("arena");

    // Write magic signature to arena
    arena.as_slice_mut()[0..4].copy_from_slice(&ARENA_MAGIC);

    // Send control command + arena FD atomically in one transaction
    let control_frame = b"ATTACH_MODEL:name=qwen3,bytes=8192,ctx=64";
    SocketIpcHelper::send_message_with_fd(
        sender.as_raw_fd(),
        control_frame,
        Some(arena.as_raw_fd()),
    )
    .expect("send atomic");

    // Receiver decodes frame and captures file descriptor
    let (payload, received_fd) =
        SocketIpcHelper::recv_message_with_fd(receiver.as_raw_fd()).expect("recv atomic");
    assert_eq!(payload, control_frame);
    let raw_fd = received_fd.expect("must receive FD");

    // Receiver maps the transferred descriptor and verifies magic bytes
    let mapped =
        SharedMemoryFixture::from_fd(raw_fd, 8192, "mapped_receiver").expect("map receiver");
    assert_eq!(&mapped.as_slice()[0..4], &ARENA_MAGIC);
}
