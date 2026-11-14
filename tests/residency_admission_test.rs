//! Integration tests for US-01.3: Pre-flight memory admission, locked weight residency,
//! flock advisory locking, and single-slot KV cache arena lifecycle.

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;

use jero_core::test_helpers::{create_synthetic_gguf_v3, create_synthetic_gguf_v3_with_kvs};
use jero_core::{ExecutionContext, GgufValue, K3Error, QuantType};
use jero_engine::{
    LayerKvPolicy, LockedModelWeights, MockMemoryInspector, SingleSlotKvArena,
    check_memory_admission_with, estimate_hybrid_kv_cache_bytes, gemma_layer_kv_policies,
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

#[test]
fn test_hybrid_kv_admission() {
    // 36-layer 5:1 Gemma 3/4 topology at n_ctx = 32768 with 1024-token sliding window
    let policies = gemma_layer_kv_policies("gemma3", 36, Some(1024));
    assert_eq!(policies.len(), 36);
    assert_eq!(
        policies
            .iter()
            .filter(|p| matches!(p, LayerKvPolicy::GlobalFullContext))
            .count(),
        6
    );

    let hybrid_bytes = estimate_hybrid_kv_cache_bytes(32_768, 4, 256, &policies)
        .expect("hybrid KV byte estimation succeeds");
    let uniform_full_bytes =
        estimate_hybrid_kv_cache_bytes(32_768, 4, 256, &[LayerKvPolicy::GlobalFullContext; 36])
            .expect("uniform full KV byte estimation succeeds");

    // 6 global @ 32768 + 30 sliding @ 1024 = 227,328 tokens vs 36 * 32768 = 1,179,648 tokens (>5x savings)
    assert!(hybrid_bytes * 5 < uniform_full_bytes);

    // GGUF admission test: gemma3 admits under a ceiling smaller than uniform full KV
    let extra_kvs = [
        ("gemma3.block_count", GgufValue::Uint32(12)),
        ("gemma3.embedding_length", GgufValue::Uint32(512)),
        ("gemma3.attention.head_count", GgufValue::Uint32(4)),
        ("gemma3.attention.head_count_kv", GgufValue::Uint32(2)),
        ("gemma3.context_length", GgufValue::Uint32(8192)),
    ];
    let bytes = create_synthetic_gguf_v3_with_kvs("gemma3", 32, &extra_kvs, &[]);
    let model = jero_core::GgufFile::parse(&bytes).expect("parse gemma3 gguf");
    let meta = model.metadata();

    let gemma_policies = gemma_layer_kv_policies("gemma3", 12, None);
    let expected_hybrid_bytes =
        estimate_hybrid_kv_cache_bytes(8192, 2, 128, &gemma_policies).unwrap();
    let uniform_bytes =
        estimate_hybrid_kv_cache_bytes(8192, 2, 128, &[LayerKvPolicy::GlobalFullContext; 12])
            .unwrap();
    assert!(expected_hybrid_bytes * 3 < uniform_bytes);

    // Pick a tight memlock budget between hybrid footprint (+weights/page headroom) and uniform full footprint
    let tight_budget = u64::try_from(expected_hybrid_bytes + 512 * 1024).unwrap();
    assert!(tight_budget < u64::try_from(uniform_bytes).unwrap());

    let admitting_inspector = MockMemoryInspector::new(tight_budget, tight_budget);
    let fp = check_memory_admission_with(meta, 8192, &admitting_inspector)
        .expect("hybrid gemma3 admission must succeed below uniform-full memlock ceiling");
    assert!(fp.total_required_bytes <= tight_budget);

    // Verify admission still fails closed if ceiling drops below hybrid KV requirement
    let rejecting_inspector = MockMemoryInspector::new(1024 * 1024 * 1024, 65_536);
    let err = check_memory_admission_with(meta, 8192, &rejecting_inspector)
        .expect_err("must reject when hybrid KV exceeds RLIMIT_MEMLOCK");
    assert!(matches!(err, K3Error::InsufficientLockableMemory { .. }));
}

#[test]
fn test_qwen3_30b_a3b_admission() {
    use jero_engine::forward::{Lane, PlacementPlan};
    use jero_engine::{Qwen3MoeRecipe, check_memory_admission_for_weight_bytes};

    // Qwen3-30B-A3B Q4_K_M profile (~17.5 GiB = 18,790,481,920 bytes, 128 experts, top-8 routed).
    let extra_kvs = [
        ("qwen3moe.block_count", GgufValue::Uint32(48)),
        ("qwen3moe.embedding_length", GgufValue::Uint32(2048)),
        ("qwen3moe.feed_forward_length", GgufValue::Uint32(6144)),
        ("qwen3moe.attention.head_count", GgufValue::Uint32(32)),
        ("qwen3moe.attention.head_count_kv", GgufValue::Uint32(4)),
        ("qwen3moe.attention.key_length", GgufValue::Uint32(128)),
        ("qwen3moe.context_length", GgufValue::Uint32(32_768)),
        ("qwen3moe.expert_count", GgufValue::Uint32(128)),
        ("qwen3moe.expert_used_count", GgufValue::Uint32(8)),
    ];
    let bytes = create_synthetic_gguf_v3_with_kvs("qwen3moe", 32, &extra_kvs, &[]);
    let model = jero_core::GgufFile::parse(&bytes).expect("parse qwen3moe gguf");
    let recipe = verify_model_capabilities(&model).expect("verify qwen3moe recipe");
    assert_eq!(recipe.architecture_id(), "qwen3moe");

    let moe = Qwen3MoeRecipe::new_30b_a3b();
    assert_eq!(
        moe.placement_plan(),
        PlacementPlan::op(Lane::A100, Lane::X100)
    );

    // Bounded 28 GiB RLIMIT_MEMLOCK ceiling (29,360,128 KB = 30,064,771,072 bytes) admits
    // 17.5 GiB Q4_K_M weights + 4096-token KV cache without touching host physical RAM/tmpfs.
    let q4km_30b_bytes: usize = 18_790_481_920;
    let target_ceiling_28gib: u64 = 30_064_771_072;
    let inspector_28g = MockMemoryInspector::new(target_ceiling_28gib, target_ceiling_28gib);
    let fp = check_memory_admission_for_weight_bytes(
        model.metadata(),
        q4km_30b_bytes,
        4096,
        &inspector_28g,
    )
    .expect("Qwen3-30B-A3B Q4_K_M @ n_ctx=4096 must pass 28 GiB memlock admission");
    assert!(fp.total_required_bytes <= target_ceiling_28gib);

    let inspector_16g = MockMemoryInspector::new(16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024);
    let err = check_memory_admission_for_weight_bytes(
        model.metadata(),
        q4km_30b_bytes,
        4096,
        &inspector_16g,
    )
    .expect_err("Qwen3-30B-A3B Q4_K_M must be rejected under 16 GiB memlock ceiling");
    assert!(matches!(err, K3Error::InsufficientLockableMemory { .. }));
}
