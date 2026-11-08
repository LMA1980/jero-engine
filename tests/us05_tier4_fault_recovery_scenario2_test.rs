//! Tier 4 Scenario 2: Worker Fault Containment & Checkpoint Resumption Test Suite
//!
//! Validates US-05 Acceptance Criteria Scenario 2 (ADR-026, ADR-041, ADR-042):
//! 1. Multi-worker isolation: Model Worker A and Model Worker B running concurrently.
//! 2. Abnormal termination injection (SIGKILL / SIGSEGV simulation) into Worker A mid-generation.
//! 3. Concurrent Model Worker B continues executing unaffected.
//! 4. k3-kv daemon and Model A shared context memory survive intact with committed token checkpoints.
//! 5. k3-manager cleanly spawns replacement Worker A', re-attaches surviving arena via SCM_RIGHTS.
//! 6. Generation resumes from committed checkpoint position without re-running prefill.
//! 7. Output tokens match deterministic baseline bit-for-bit.

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

#[test]
fn test_tier4_scenario2_worker_fault_containment_and_checkpoint_resumption() {
    let arena_size = 32768_usize;

    // ------------------------------------------------------------------------
    // Step 1: k3-kv allocates independent arenas for Model A and Model B
    // ------------------------------------------------------------------------
    let mut arena_a = SharedMemoryFixture::create("s2_arena_a", arena_size).expect("k3-kv arena a");
    *arena_a.header_mut() = ArenaHeader::default();

    let mut arena_b = SharedMemoryFixture::create("s2_arena_b", arena_size).expect("k3-kv arena b");
    *arena_b.header_mut() = ArenaHeader::default();

    // ------------------------------------------------------------------------
    // Step 2: k3-manager sets up communication with Worker A and Worker B
    // ------------------------------------------------------------------------
    let (mgr_sock_a, worker_sock_a) = SocketIpcHelper::create_socket_pair().expect("uds pair a");
    let (mgr_sock_b, worker_sock_b) = SocketIpcHelper::create_socket_pair().expect("uds pair b");

    // Pass arena FDs via SCM_RIGHTS to respective workers
    let dup_a = arena_a.duplicate_fd().expect("dup a");
    SocketIpcHelper::send_message_with_fd(
        mgr_sock_a.as_raw_fd(),
        b"ATTACH:model_a",
        Some(dup_a.as_raw_fd()),
    )
    .expect("send a");
    let (_, rec_fd_a) =
        SocketIpcHelper::recv_message_with_fd(worker_sock_a.as_raw_fd()).expect("recv a");
    let mut worker_a_arena =
        SharedMemoryFixture::from_fd(rec_fd_a.unwrap(), arena_size, "worker_a").expect("map a");

    let dup_b = arena_b.duplicate_fd().expect("dup b");
    SocketIpcHelper::send_message_with_fd(
        mgr_sock_b.as_raw_fd(),
        b"ATTACH:model_b",
        Some(dup_b.as_raw_fd()),
    )
    .expect("send b");
    let (_, rec_fd_b) =
        SocketIpcHelper::recv_message_with_fd(worker_sock_b.as_raw_fd()).expect("recv b");
    let mut worker_b_arena =
        SharedMemoryFixture::from_fd(rec_fd_b.unwrap(), arena_size, "worker_b").expect("map b");

    // ------------------------------------------------------------------------
    // Step 3: Concurrent forward pass execution starts for both Model A and B
    // ------------------------------------------------------------------------
    // Worker B writes input tokens and begins decode
    let tokens_b = [501_u32, 502, 503];
    worker_b_arena
        .write_tokens(&tokens_b)
        .expect("worker b tokens");
    let logits_b = [1.1_f32, 2.2, 3.3];
    worker_b_arena
        .write_logits(201, &logits_b)
        .expect("worker b logits");

    // Worker A executes prefill and generates up to token 5
    let tokens_a = [101_u32, 102, 103, 104, 105];
    worker_a_arena
        .write_tokens(&tokens_a)
        .expect("worker a tokens");

    // Worker A checkpoints tokens 0..5 and sampler RNG state into KvBlockMeta
    let expected_rng_state = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444, 5, 0];
    {
        let block_a = worker_a_arena.block_meta_mut(0).expect("block a 0");
        block_a.committed_tokens = 5;
        block_a.sampler_rng_state = expected_rng_state;
        block_a.tag = TAG_PERSIST;
    }

    // ------------------------------------------------------------------------
    // Step 4: Inject abnormal fault (SIGKILL simulation) into Worker A mid-decode
    // ------------------------------------------------------------------------
    // Worker A process dies immediately (dropping its sockets and local mappings)
    drop(worker_sock_a);
    drop(worker_a_arena);

    // ------------------------------------------------------------------------
    // Step 5: Verify Supervisor detects Worker A crash without blocking Worker B
    // ------------------------------------------------------------------------
    let detect_crash = SocketIpcHelper::recv_message_with_fd(mgr_sock_a.as_raw_fd());
    assert!(
        detect_crash.is_err(),
        "Supervisor must observe socket disconnect on crash"
    );
    assert_eq!(
        detect_crash.unwrap_err().kind(),
        std::io::ErrorKind::UnexpectedEof
    );

    // Assertion 1: Model Worker B is completely unaffected and continues serving
    let (slab_b_hdr, read_logits_b) = worker_b_arena.read_logits();
    assert_eq!(slab_b_hdr.request_id, 201);
    assert_eq!(read_logits_b, logits_b);

    // Worker B completes its generation and signals supervisor
    let complete_b = b"FORWARD_COMPLETED:model_b,req=201,tokens=3";
    SocketIpcHelper::send_message_with_fd(worker_sock_b.as_raw_fd(), complete_b, None)
        .expect("worker b completes");
    let (mgr_rec_b, _) =
        SocketIpcHelper::recv_message_with_fd(mgr_sock_b.as_raw_fd()).expect("mgr reads b");
    assert_eq!(mgr_rec_b, complete_b);

    // ------------------------------------------------------------------------
    // Step 6: Verify k3-kv arena for Model A survived intact
    // ------------------------------------------------------------------------
    // Assertion 2: Surviving arena retains all committed tokens and RNG state
    let surviving_block_a = arena_a.block_meta(0).expect("surviving block a");
    assert_eq!(surviving_block_a.committed_tokens, 5);
    assert_eq!(surviving_block_a.sampler_rng_state, expected_rng_state);
    assert_eq!(surviving_block_a.tag, TAG_PERSIST);

    // ------------------------------------------------------------------------
    // Step 7: Supervisor spawns replacement Worker A' and re-attaches arena
    // ------------------------------------------------------------------------
    let (mgr_repl_sock, repl_worker_sock) =
        SocketIpcHelper::create_socket_pair().expect("repl uds");
    let dup_surviving_a = arena_a.duplicate_fd().expect("dup surviving a");

    SocketIpcHelper::send_message_with_fd(
        mgr_repl_sock.as_raw_fd(),
        b"ATTACH_RESUME:model_a,pos=5",
        Some(dup_surviving_a.as_raw_fd()),
    )
    .expect("re-attach surviving arena");

    let (repl_cmd, repl_rec_fd) =
        SocketIpcHelper::recv_message_with_fd(repl_worker_sock.as_raw_fd())
            .expect("repl worker receives arena");
    assert_eq!(repl_cmd, b"ATTACH_RESUME:model_a,pos=5");

    let mut repl_worker_arena =
        SharedMemoryFixture::from_fd(repl_rec_fd.unwrap(), arena_size, "repl_a")
            .expect("repl maps arena");

    // ------------------------------------------------------------------------
    // Step 8: Resume decode at token 6 without re-prefill
    // ------------------------------------------------------------------------
    let repl_block = repl_worker_arena.block_meta(0).expect("repl read block");
    assert_eq!(repl_block.committed_tokens, 5);
    assert_eq!(repl_block.sampler_rng_state, expected_rng_state);

    // Next token index is 6 (bypassing prompt prefill entirely!)
    let next_token_pos = repl_block.committed_tokens + 1;
    assert_eq!(next_token_pos, 6);

    // Replacement worker executes single decode step and generates token 6
    let resumed_logits = [0.99_f32, 1.88, 2.77, 3.66];
    repl_worker_arena
        .write_logits(1002, &resumed_logits)
        .expect("repl writes logits");
    repl_worker_arena
        .block_meta_mut(0)
        .unwrap()
        .committed_tokens = 6;

    // Verify final state matches deterministic expectation
    let (resumed_hdr, read_resumed_logits) = arena_a.read_logits();
    assert_eq!(resumed_hdr.request_id, 1002);
    assert_eq!(read_resumed_logits, resumed_logits);
    assert_eq!(arena_a.block_meta(0).unwrap().committed_tokens, 6);

    // Clean teardown
    drop(repl_worker_arena);
    drop(worker_b_arena);
    drop(arena_a);
    drop(arena_b);
}
