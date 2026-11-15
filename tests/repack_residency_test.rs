//! Integration tests for `#163` memlock-bounded repack residency admission and `#164`
//! model-scoped `RepackedTensorStore` reuse across repeated prompt evaluations.

use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};
use jero_engine::{
    CoordinationProfile, LockedModelWeights, MockMemoryInspector, SinglePromptEngine,
    SingleSlotKvArena, StreamAction, WorkerPool, check_memory_admission_with, estimate_footprint,
    resolve_affinity,
};

#[test]
#[cfg_attr(miri, ignore)]
fn test_wave3_repeat3_model_view_repack_residency_and_drop_cleanup() {
    let specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 8][..]),
        ("blk.0.attn_v.weight", QuantType::Q4_K_M, &[256, 8][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &specs);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

    // 1. Validate #163 estimate_footprint & tight-budget downgrade behavior
    let estimated = estimate_footprint(gguf.metadata(), 64).expect("estimate footprint");
    assert!(estimated.repack_arena_bytes > 0);
    assert!(estimated.prefill_scratch_bytes > 0);
    assert!(estimated.eager_repack_enabled);
    assert_eq!(
        estimated.total_required_bytes,
        estimated.weight_bytes + estimated.kv_bytes
    );

    let base_budget = estimated.base_required_bytes();
    assert!(base_budget < estimated.total_with_repack_bytes());
    let downgrade_inspector = MockMemoryInspector::new(base_budget, base_budget);
    let downgraded = check_memory_admission_with(gguf.metadata(), 64, &downgrade_inspector)
        .expect("base fits while optional repack is cleanly downgraded");
    assert!(!downgraded.eager_repack_enabled);
    assert_eq!(downgraded.repack_arena_bytes, 0);

    // 2. Open with full budget, prepare once on WorkerPool, evaluate repeat=3 using ModelView
    let temp_path = std::env::temp_dir().join(format!(
        "k3_wave3_residency_e2e_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&temp_path, &bytes).expect("write synthetic gguf");

    let generous = MockMemoryInspector::new(256 * 1024 * 1024, 256 * 1024 * 1024);
    let (locked_weights, recipe, fp) =
        LockedModelWeights::open_with_admission(&temp_path, 64, &generous).expect("open model");
    assert!(fp.eager_repack_enabled);

    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(2));
    let pool = WorkerPool::with_coordination(
        resolve_affinity(None, None, Some(threads)).expect("host affinity"),
        CoordinationProfile::Rayon,
    )
    .expect("spawn pool");

    let init_repack = locked_weights
        .prepare_repacked_weights(Some(&pool))
        .expect("eager parallel repack");
    assert_eq!(init_repack.build_count, 1);

    let model_view = locked_weights.model_view();
    let mut engine = SinglePromptEngine::new();
    let prompt = [1_u32, 2, 3, 4, 5, 6, 7, 8];

    for _run in 0..3 {
        let reuse = locked_weights
            .prepare_repacked_weights(Some(&pool))
            .expect("reuse repacked weights");
        assert_eq!(reuse.build_count, 1);

        let mut kv_arena = SingleSlotKvArena::from_gguf(&gguf, 64).expect("kv arena");
        let summary = engine
            .evaluate_prompt_with_pool(
                recipe.as_ref(),
                &model_view,
                &mut kv_arena,
                Some(&pool),
                &prompt,
                32,
                2,
                |_| Ok(StreamAction::Continue),
            )
            .expect("evaluate prompt");
        assert_eq!(summary.prompt_tokens, prompt.len());
        assert!(summary.timing.prefill_tokens_per_sec() > 0.0);
        assert!(summary.timing.prefill_ns > 0);
    }

    let final_repack = model_view.repack_summary();
    assert_eq!(final_repack.build_count, 1);
    assert_eq!(final_repack.hit_count, 3);

    let weak_entry = locked_weights
        .metadata()
        .tensors()
        .iter()
        .find_map(|desc| locked_weights.repacked_tensor(desc.name()))
        .map(|entry| std::sync::Arc::downgrade(&entry))
        .expect("at least one repacked tensor entry must exist before drop");
    assert!(
        weak_entry.upgrade().is_some(),
        "repacked entry must stay live while LockedModelWeights owns it"
    );

    drop(locked_weights);
    assert!(
        weak_entry.upgrade().is_none(),
        "dropping LockedModelWeights must automatically release all repacked entries without calling clear()"
    );
    let _ = std::fs::remove_file(&temp_path);
}
