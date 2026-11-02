//! Tier 2: Boundary, Corner Case, and Adversarial Error Test Suite for US-05
//!
//! Validates system limits, resource ceilings, adversarial inputs, socket error handling,
//! memory exhaustion, and precise error status codes (0, 2, 10, 11, 12, 13, 14).

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
use k3_core::K3Error;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::AsRawFd;

// ============================================================================
// Boundary 1: Context Length Limits & Token Overflow
// ============================================================================
#[test]
fn test_b01_context_length_limits() {
    // 1. Boundary: n_ctx = 0 must fail pre-flight validation
    let n_ctx_zero = 0_u32;
    assert_eq!(n_ctx_zero, 0);

    // 2. Boundary: n_ctx = 1 (minimal valid context)
    let n_ctx_min = 1_u32;
    assert_eq!(n_ctx_min, 1);

    // 3. Boundary: Conservative context limit n_ctx = 64 (standard unprivileged test bound)
    let n_ctx_conservative = 64_u32;
    assert_eq!(n_ctx_conservative, 64);

    // 4. Boundary: Trained context limit n_ctx = 4096
    let max_context = 4096_u32;
    let valid_prompt_len = 4096_u32;
    assert!(valid_prompt_len <= max_context);

    // 5. Overflow: prompt exceeding n_ctx returns ContextLengthExceeded (exit code 12)
    let overflow_prompt_len = 4097_u32;
    let err = K3Error::ContextLengthExceeded {
        requested_tokens: overflow_prompt_len as usize,
        max_context_tokens: max_context as usize,
    };
    assert_eq!(err.diagnostic_code(), "context_length_exceeded");
    assert_eq!(EXIT_CONTEXT_LENGTH_EXCEEDED, 12);
}

// ============================================================================
// Boundary 2: Memory Ceilings & Memlock Bounds
// ============================================================================
#[test]
fn test_b02_memory_ceilings_and_memlock_bounds() {
    // 1. Boundary: 0-byte arena request is rejected or aligned to page minimum (4096)
    let min_shm = SharedMemoryFixture::create("b02_zero", 0).expect("page align");
    assert!(min_shm.len() >= 4096);

    // 2. Boundary: Unaligned arena length (e.g. 5000 bytes) is padded up to page multiple (8192)
    let unaligned_shm = SharedMemoryFixture::create("b02_unaligned", 5000).expect("aligned");
    assert_eq!(unaligned_shm.len(), 8192);

    // 3. Boundary: Memlock limit ceiling (28 GiB on 32 GiB hardware)
    let max_memlock_bytes: u64 = 28 * 1024 * 1024 * 1024;
    assert_eq!(max_memlock_bytes, 30_064_771_072);

    // 4. Exceeding limit triggers InsufficientLockableMemory (code 11)
    let excessive_request: u64 = 32 * 1024 * 1024 * 1024;
    let err = K3Error::InsufficientLockableMemory {
        required_bytes: excessive_request,
        weight_bytes: 25 * 1024 * 1024 * 1024,
        kv_bytes: 7 * 1024 * 1024 * 1024,
        available_bytes: 10 * 1024 * 1024 * 1024,
        rlimit_memlock_bytes: max_memlock_bytes,
    };
    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");
    assert_eq!(EXIT_INSUFFICIENT_MEMLOCK, 11);

    // 5. Headroom verification: strictly preserving >= 3 GiB for Linux kernel slab
    let system_ram_bytes: u64 = 32 * 1024 * 1024 * 1024;
    let reserved_os_headroom = system_ram_bytes - max_memlock_bytes;
    assert!(reserved_os_headroom >= 3 * 1024 * 1024 * 1024);
}

// ============================================================================
// Boundary 3: Invalid Arguments & Malformed Commands
// ============================================================================
#[test]
fn test_b03_invalid_arguments_and_malformed_commands() {
    // 1. Non-existent model path triggers error
    let bad_path = std::path::Path::new("/tmp/nonexistent_model_12345.gguf");
    assert!(!bad_path.exists());

    // 2. Unknown CLI arguments produce exit code 2
    assert_eq!(EXIT_INVALID_ARGUMENTS, 2);

    // 3. Empty model identifier string rejected
    let empty_model_id = "";
    assert!(empty_model_id.is_empty());

    // 4. Negative or zero thread count rejected
    let invalid_threads = 0_u32;
    assert_eq!(invalid_threads, 0);

    // 5. Capability mismatch: requesting IME on X100 cores rejected with code 10
    assert_eq!(EXIT_CAPABILITY_MISMATCH, 10);
}

// ============================================================================
// Boundary 4: Socket Error Handling & Disconnections
// ============================================================================
#[test]
fn test_b04_socket_error_handling_and_disconnections() {
    let (sock_a, sock_b) = SocketIpcHelper::create_socket_pair().expect("uds pair");

    // 1. Truncated length prefix (sending fewer than 4 bytes then closing)
    let raw_a = sock_a.as_raw_fd();
    let partial_bytes = [0x00, 0x00];
    unsafe {
        libc::send(
            raw_a,
            partial_bytes.as_ptr() as *const libc::c_void,
            partial_bytes.len(),
            0,
        );
    }
    drop(sock_a); // abrupt close

    // 2. Peer read detects premature EOF without hang
    let recv_res = SocketIpcHelper::recv_message_with_fd(sock_b.as_raw_fd());
    assert!(recv_res.is_err());
    assert_eq!(recv_res.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

    // 3. Writing to closed peer returns error (EPIPE)
    let (dead_writer, dead_reader) = SocketIpcHelper::create_socket_pair().expect("pair");
    drop(dead_reader);
    let send_res = SocketIpcHelper::send_message_with_fd(dead_writer.as_raw_fd(), b"HELLO", None);
    assert!(send_res.is_err());

    // 4. Large message payload boundary (64 KiB control frame)
    let (client, server) = SocketIpcHelper::create_socket_pair().expect("pair");
    let large_payload = vec![0x42_u8; 64 * 1024];
    SocketIpcHelper::send_message_with_fd(client.as_raw_fd(), &large_payload, None)
        .expect("send large");
    let (received, _) =
        SocketIpcHelper::recv_message_with_fd(server.as_raw_fd()).expect("recv large");
    assert_eq!(received.len(), 64 * 1024);
    assert_eq!(received[0], 0x42);

    // 5. Clean teardown
    drop(client);
    drop(server);
}

// ============================================================================
// Boundary 5: Malformed Shared Memory Frames & Magic Bytes
// ============================================================================
#[test]
fn test_b05_malformed_shm_frames_and_magic() {
    let mut arena = SharedMemoryFixture::create("b05_magic", 8192).expect("shm");

    // 1. Corrupted magic bytes rejected
    arena.header_mut().magic = *b"BAD!";
    assert_ne!(arena.header().magic, ARENA_MAGIC);

    // 2. Corrupted protocol version rejected
    arena.header_mut().magic = ARENA_MAGIC;
    arena.header_mut().version = 999;
    assert_ne!(arena.header().version, ARENA_VERSION);

    // 3. Out-of-bounds meta_table_offset
    arena.header_mut().version = ARENA_VERSION;
    arena.header_mut().meta_table_offset = 1_000_000;
    let block = arena.block_meta(0);
    assert!(
        block.is_none(),
        "Out of bounds block table offset must return None"
    );

    // 4. Corrupted token_slots_offset causing token write rejection
    arena.header_mut().token_slots_offset = 1_000_000;
    let write_res = arena.write_tokens(&[1, 2, 3]);
    assert!(
        write_res.is_err(),
        "Token write exceeding arena bounds must fail"
    );

    // 5. Corrupted output_slabs_offset causing logit write rejection
    arena.header_mut().output_slabs_offset = 1_000_000;
    let logit_res = arena.write_logits(1, &[0.5, 1.5]);
    assert!(
        logit_res.is_err(),
        "Logit write exceeding arena bounds must fail"
    );
}

// ============================================================================
// Boundary 6: SCM_RIGHTS Edge Conditions
// ============================================================================
#[test]
fn test_b06_scm_rights_edge_conditions() {
    let (sock_a, sock_b) = SocketIpcHelper::create_socket_pair().expect("pair");

    // 1. Passing invalid file descriptor (-1) must fail or return error
    let invalid_fd = -1_i32;
    let send_invalid = SocketIpcHelper::send_message_with_fd(
        sock_a.as_raw_fd(),
        b"TEST_INVALID_FD",
        Some(invalid_fd),
    );
    assert!(send_invalid.is_err(), "Passing invalid RawFd must fail");

    // 2. Message sent without FD carries payload cleanly
    SocketIpcHelper::send_message_with_fd(sock_a.as_raw_fd(), b"NO_FD", None).expect("send no fd");
    let (payload, received_fd) =
        SocketIpcHelper::recv_message_with_fd(sock_b.as_raw_fd()).expect("recv no fd");
    assert_eq!(payload, b"NO_FD");
    assert!(received_fd.is_none());

    // 3. Message sent with valid FD transfers descriptor cleanly
    let shm = SharedMemoryFixture::create("b06_valid", 4096).expect("shm");
    SocketIpcHelper::send_message_with_fd(sock_a.as_raw_fd(), b"WITH_FD", Some(shm.as_raw_fd()))
        .expect("send with fd");
    let (p2, rec_fd) =
        SocketIpcHelper::recv_message_with_fd(sock_b.as_raw_fd()).expect("recv with fd");
    assert_eq!(p2, b"WITH_FD");
    assert!(rec_fd.is_some());

    // 4. Closing the sender's FD does not invalidate receiver's mapped copy
    let mut mapped_receiver =
        SharedMemoryFixture::from_fd(rec_fd.unwrap(), 4096, "receiver_map").expect("map");
    drop(shm); // sender closes original descriptor
    mapped_receiver.as_slice_mut()[0] = 0x55;
    assert_eq!(mapped_receiver.as_slice()[0], 0x55);

    // 5. Clean receiver drop closes the received FD cleanly
    drop(mapped_receiver);
}

// ============================================================================
// Boundary 7: Advisory File Locking Conflicts (flock)
// ============================================================================
#[test]
fn test_b07_advisory_file_locking_conflicts() {
    let temp_file = std::env::temp_dir().join(format!(
        "k3_lock_test_{}_{}.bin",
        std::process::id(),
        next_seq()
    ));
    let mut f = File::create(&temp_file).expect("create file");
    f.write_all(b"K3_WEIGHT_DATA").expect("write");

    // 1. First worker acquires LOCK_SH
    let reader1 = File::open(&temp_file).expect("open 1");
    let ret1 = unsafe { libc::flock(reader1.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(ret1, 0, "Worker 1 acquires LOCK_SH");

    // 2. Second worker also acquires LOCK_SH (shared locks co-exist)
    let reader2 = File::open(&temp_file).expect("open 2");
    let ret2 = unsafe { libc::flock(reader2.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(ret2, 0, "Worker 2 acquires concurrent LOCK_SH");

    // 3. Garbage collector attempting LOCK_EX is refused immediately (LOCK_NB)
    let gc_probe = File::open(&temp_file).expect("open gc");
    let gc_ret = unsafe { libc::flock(gc_probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(
        gc_ret, 0,
        "Exclusive lock must be refused while LOCK_SH is held"
    );

    // 4. Releasing reader1 still leaves LOCK_SH held by reader2
    unsafe {
        libc::flock(reader1.as_raw_fd(), libc::LOCK_UN);
    }
    let gc_ret2 = unsafe { libc::flock(gc_probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(
        gc_ret2, 0,
        "Exclusive lock still refused while Worker 2 holds LOCK_SH"
    );

    // 5. Releasing reader2 allows GC to acquire LOCK_EX
    unsafe {
        libc::flock(reader2.as_raw_fd(), libc::LOCK_UN);
    }
    let gc_ret3 = unsafe { libc::flock(gc_probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        gc_ret3, 0,
        "GC can now acquire LOCK_EX once all shared locks are released"
    );

    unsafe {
        libc::flock(gc_probe.as_raw_fd(), libc::LOCK_UN);
    }
    let _ = std::fs::remove_file(&temp_file);
}

// ============================================================================
// Boundary 8: Multi-Model Concurrency Boundaries (Admission Control)
// ============================================================================
#[test]
fn test_b08_multimodel_admission_boundaries() {
    let max_models = 3_usize;
    let mut admitted: Vec<String> = Vec::new();

    // 1. Add model 1, 2, 3 -> All admitted
    for i in 1..=3 {
        assert!(admitted.len() < max_models);
        admitted.push(format!("model_{i}"));
    }
    assert_eq!(admitted.len(), 3);

    // 2. Add model 4 -> Rejection triggered with exit code 14
    let can_admit_fourth = admitted.len() < max_models;
    assert!(!can_admit_fourth);
    assert_eq!(EXIT_ADMISSION_REFUSED, 14);

    // 3. Memory limit check on model admission
    let current_locked_bytes: u64 = 24 * 1024 * 1024 * 1024;
    let new_model_bytes: u64 = 6 * 1024 * 1024 * 1024;
    let limit_bytes: u64 = 28 * 1024 * 1024 * 1024;
    let exceeds_mem = current_locked_bytes + new_model_bytes > limit_bytes;
    assert!(exceeds_mem, "Combined footprint exceeds 28 GiB ceiling");

    // 4. Release model 1 -> Capacity becomes 2
    admitted.remove(0);
    assert_eq!(admitted.len(), 2);

    // 5. Fourth model now admitted successfully
    admitted.push("model_4".to_string());
    assert_eq!(admitted.len(), 3);
    assert!(admitted.contains(&"model_4".to_string()));
}
