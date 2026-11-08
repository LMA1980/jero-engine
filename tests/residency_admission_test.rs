//! Integration tests for US-01.3: Pre-flight memory admission, locked weight residency,
//! flock advisory locking, and single-slot KV cache arena lifecycle.

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;

use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{ExecutionContext, K3Error, QuantType};
use jero_engine::{
    LockedModelWeights, MockMemoryInspector, SingleSlotKvArena, check_memory_admission_with,
    verify_model_capabilities,
};

#[test]
fn test_end_to_end_preflight_admission_and_locked_evaluation() {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!("integ_test_model_{}.gguf", std::process::id()));
    {
        let mut f = File::create(&file_path).expect("create model file");
        f.write_all(&bytes).expect("write model bytes");
        f.sync_all().expect("sync model file");
    }

    // 1. Open model with shared lock, mmap, and mlock
    let locked_weights = LockedModelWeights::open(&file_path).expect("open and lock model weights");
    let metadata = locked_weights.metadata();

    // 2. Capability verification (US-01.2)
    let recipe =
        verify_model_capabilities(&jero_core::GgufFile::parse(locked_weights.as_bytes()).unwrap())
            .expect("recipe capability verification succeeds");
    assert_eq!(recipe.architecture_id(), "synthetic");

    // 3. Pre-flight memory admission check (US-01.3.1)
    let n_ctx = 64;
    let inspector = MockMemoryInspector::new(50_000_000, 50_000_000);
    let footprint = check_memory_admission_with(metadata, n_ctx, &inspector)
        .expect("pre-flight memory admission succeeds");
    assert!(footprint.total_required_bytes > 0);
    assert_eq!(footprint.kv_block_count, 4); // 64 / 16 = 4 blocks

    // 4. Single-slot KV cache arena allocation and mlock (US-01.3.3)
    let mut kv_arena = SingleSlotKvArena::new(metadata, n_ctx).expect("allocate kv arena");
    assert_eq!(kv_arena.n_ctx(), 64);
    assert_eq!(kv_arena.num_blocks(), 4);

    // 5. Execution context lifecycle using locked weights and KV arena
    let prompt = [10_u32, 20, 30];
    let mut logits = [0.0_f32; 8];
    let mut ctx = ExecutionContext::new(&prompt, n_ctx, &mut logits, kv_arena.as_mut_slice())
        .expect("initialize execution context");

    recipe
        .prefill_step(&mut ctx, &locked_weights)
        .expect("prefill step succeeds");
    assert_eq!(ctx.position(), 3);

    // 6. Verify shared flock blocks external exclusive lock (US-01.3.2)
    let external_handle = File::open(&file_path).expect("open external file");
    // SAFETY: Testing non-blocking exclusive flock on actively locked model file.
    let ex_lock_res =
        unsafe { libc::flock(external_handle.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(
        ex_lock_res, 0,
        "LOCK_EX must be refused while model is resident"
    );

    // 7. Teardown: dropping locked weights releases flock and mlock cleanly
    drop(locked_weights);
    drop(kv_arena);

    // Now external exclusive lock must succeed immediately
    // SAFETY: Model was dropped, releasing LOCK_SH; LOCK_EX must now succeed.
    let ex_lock_after =
        unsafe { libc::flock(external_handle.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        ex_lock_after, 0,
        "LOCK_EX must succeed after model weights drop"
    );

    // SAFETY: Releasing the exclusive lock.
    unsafe {
        libc::flock(external_handle.as_raw_fd(), libc::LOCK_UN);
    }
    drop(external_handle);
    let _ = std::fs::remove_file(&file_path);
}

#[test]
fn test_insufficient_memory_aborts_upfront_scenario_3() {
    let tensor_specs = [("weight", QuantType::F32, &[16][..])];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
    let model = jero_core::GgufFile::parse(&bytes).expect("parse model");
    let meta = model.metadata();

    // Available RAM is restricted to 256 bytes
    let inspector = MockMemoryInspector::new(256, 100_000_000);
    let err = check_memory_admission_with(meta, 128, &inspector)
        .expect_err("must abort upfront with InsufficientLockableMemory");

    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");
    match err {
        K3Error::InsufficientLockableMemory {
            required_bytes,
            available_bytes,
            rlimit_memlock_bytes,
            ..
        } => {
            assert!(required_bytes > available_bytes);
            assert_eq!(available_bytes, 256);
            assert_eq!(rlimit_memlock_bytes, 100_000_000);
        }
        other => panic!("expected InsufficientLockableMemory, got: {other:?}"),
    }
}
