//! US-02: Multi-Core Coordination & Jitter Evaluation Engine Integration Tests.
//!
//! Validates the multi-run execution pipeline, thread affinity stability,
//! temperature 0 token determinism, clean teardown, and descriptive sample statistics.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::PathBuf;

use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{CoordinationProfile, QuantType};
use jero_engine::{
    CpuTopologyInspector, LockedModelWeights, SinglePromptEngine, SingleSlotKvArena, StreamAction,
    SystemCpuTopology, SystemMemoryInspector, WorkerPool, X100_CLUSTER_MASK, resolve_affinity,
};

fn write_synthetic_gguf_fixture(tag: &str) -> PathBuf {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
    let path = std::env::temp_dir().join(format!(
        "us02_engine_fixture_{tag}_{}.gguf",
        std::process::id()
    ));
    let mut f = File::create(&path).expect("create synthetic GGUF fixture");
    f.write_all(&bytes).expect("write GGUF fixture bytes");
    f.sync_all().expect("sync GGUF fixture");
    path
}

/// Computes sample mean, min, max, and sample standard deviation ($N-1$ Bessel's correction).
fn calculate_sample_stats(samples: &[f64]) -> (f64, f64, f64, f64) {
    assert!(samples.len() >= 2, "sample size must be >= 2");
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, min, max, variance.sqrt())
}

#[test]
fn test_us02_multicore_pipeline_affinity_stability_and_determinism() {
    let fixture_path = write_synthetic_gguf_fixture("stability");
    let repeat_count = 3;
    let max_tokens = 4;
    let n_ctx = 64;

    // 1. Resolve host core affinity and spawn pinned WorkerPool
    let online_mask = SystemCpuTopology::new()
        .online_cpu_mask()
        .expect("query online CPUs");
    let host_x100_mask = online_mask & X100_CLUSTER_MASK;
    let thread_count = (host_x100_mask.count_ones() as usize).clamp(1, 2);

    let host_affinity = resolve_affinity(None, Some(host_x100_mask), Some(thread_count))
        .expect("resolve host affinity");
    let worker_pool = WorkerPool::new(host_affinity.clone()).expect("spawn worker pool");
    assert_eq!(worker_pool.thread_count(), thread_count);

    // Initial affinity check
    let initial_status = worker_pool
        .verify_live_affinities()
        .expect("verify initial worker affinities");

    // 2. Open model weights and allocate KV arena
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _) =
        LockedModelWeights::open_with_admission(&fixture_path, n_ctx, &inspector)
            .expect("open model weights");

    // 3. Multi-Run Execution Loop (N = 3)
    let mut engine = SinglePromptEngine::new();
    let prompt_tokens = vec![1, 9, 10]; // synthetic vocabulary tokens
    let mut run_tokens: Vec<Vec<u32>> = Vec::new();
    let mut run_latencies_ms: Vec<f64> = Vec::new();

    for iter in 0..repeat_count {
        let mut kv_arena =
            SingleSlotKvArena::new(locked_weights.metadata(), n_ctx).expect("allocate KV arena");
        let start = std::time::Instant::now();

        let summary = engine
            .evaluate_prompt_with_pool(
                recipe,
                &locked_weights,
                &mut kv_arena,
                Some(&worker_pool),
                &prompt_tokens,
                16,
                max_tokens,
                |_tok| Ok(StreamAction::Continue),
            )
            .unwrap_or_else(|e| panic!("iteration {iter} failed: {e}"));

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        run_latencies_ms.push(elapsed_ms);
        run_tokens.push(summary.generated_tokens);

        // Verify thread affinities remain pinned and stable across repeat passes
        let live_status = worker_pool
            .verify_live_affinities()
            .unwrap_or_else(|e| panic!("iteration {iter} affinity check failed: {e}"));
        assert_eq!(
            live_status, initial_status,
            "worker thread core affinity must not drift across iterations"
        );
    }

    // 4. Assert Output Determinism: Bit-identical tokens across all iterations at temp 0
    assert_eq!(run_tokens.len(), repeat_count);
    for i in 1..repeat_count {
        assert_eq!(
            run_tokens[0], run_tokens[i],
            "iteration {i} generated tokens differ from iteration 0: {:?} vs {:?}",
            run_tokens[i], run_tokens[0]
        );
    }

    // 5. Calculate Descriptive Statistics
    let (mean, min, max, stddev) = calculate_sample_stats(&run_latencies_ms);
    assert!(
        min <= mean && mean <= max,
        "min ({min}) <= mean ({mean}) <= max ({max})"
    );
    assert!(stddev >= 0.0, "stddev must be non-negative: {stddev}");

    // 6. Clean Teardown Verification
    drop(locked_weights);
    drop(worker_pool);

    // Verify flock(LOCK_EX) can be acquired immediately after drop
    let probe_file = File::open(&fixture_path).expect("open probe file");
    // SAFETY: Non-blocking flock(LOCK_EX | LOCK_NB) to verify LOCK_SH released.
    let lock_ex = unsafe { libc::flock(probe_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        lock_ex, 0,
        "flock(LOCK_EX) must succeed immediately after drop"
    );
    // SAFETY: Non-blocking flock(LOCK_UN) to release probe file lock.
    unsafe {
        libc::flock(probe_file.as_raw_fd(), libc::LOCK_UN);
    }

    let _ = std::fs::remove_file(&fixture_path);
}

#[test]
fn test_us02_coordination_profiles_comparative_forward_pass() {
    let fixture_path = write_synthetic_gguf_fixture("coord_comparative");
    let max_tokens = 4;
    let n_ctx = 64;

    let online_mask = SystemCpuTopology::new()
        .online_cpu_mask()
        .expect("query online CPUs");
    let host_x100_mask = online_mask & X100_CLUSTER_MASK;
    let thread_count = (host_x100_mask.count_ones() as usize).clamp(1, 2);

    let host_affinity = resolve_affinity(None, Some(host_x100_mask), Some(thread_count))
        .expect("resolve host affinity");

    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _) =
        LockedModelWeights::open_with_admission(&fixture_path, n_ctx, &inspector)
            .expect("open model weights");

    let prompt_tokens = vec![1, 9, 10];
    let mut profile_tokens: Vec<(CoordinationProfile, Vec<u32>)> = Vec::new();

    for &profile in &CoordinationProfile::all() {
        let pool = WorkerPool::with_coordination(host_affinity.clone(), profile)
            .unwrap_or_else(|e| panic!("failed to spawn pool for {profile}: {e}"));
        assert_eq!(pool.profile(), profile);
        assert_eq!(pool.coordination_profile(), profile);
        assert_eq!(pool.thread_count(), thread_count);

        let mut kv_arena =
            SingleSlotKvArena::new(locked_weights.metadata(), n_ctx).expect("allocate KV arena");
        let mut engine = SinglePromptEngine::new();

        let summary = engine
            .evaluate_prompt_with_pool(
                recipe,
                &locked_weights,
                &mut kv_arena,
                Some(&pool),
                &prompt_tokens,
                16,
                max_tokens,
                |_tok| Ok(StreamAction::Continue),
            )
            .unwrap_or_else(|e| panic!("evaluation failed under {profile}: {e}"));

        assert_eq!(summary.prompt_tokens, 3);
        assert_eq!(summary.generated_tokens.len(), max_tokens);
        assert_eq!(summary.timing.decode_steps, 3);
        assert!(
            summary.timing.ttft_ms > 0.0,
            "TTFT must be positive for {profile}"
        );
        assert!(
            summary.timing.decode_tokens_per_sec > 0.0,
            "decode throughput must be positive for {profile}"
        );
        assert!(
            summary.timing.total_wall_time_ms >= summary.timing.ttft_ms,
            "wall time >= TTFT for {profile}"
        );
        assert!(
            summary.timing.cumulative_barrier_wait_ms >= 0.0,
            "cumulative barrier wait >= 0.0 for {profile}"
        );
        assert!(
            summary.timing.compute_utilization_ratio >= 0.0
                && summary.timing.compute_utilization_ratio <= 1.0,
            "compute utilization ratio within [0.0, 1.0] for {profile}"
        );
        assert!(
            summary.timing.barrier_wait_ratio >= 0.0 && summary.timing.barrier_wait_ratio <= 1.0,
            "barrier wait ratio within [0.0, 1.0] for {profile}"
        );

        // Verify accumulator reset
        pool.reset_barrier_wait();
        assert_eq!(pool.cumulative_barrier_wait_ms(), 0.0);

        profile_tokens.push((profile, summary.generated_tokens));
        drop(pool);
    }

    // Output determinism: all 3 profiles must produce bit-identical token sequences
    assert_eq!(profile_tokens.len(), 3);
    let baseline_tokens = &profile_tokens[0].1;
    for (prof, tokens) in &profile_tokens[1..] {
        assert_eq!(
            baseline_tokens, tokens,
            "profile {prof} generated tokens differ from baseline spin: {:?} vs {:?}",
            tokens, baseline_tokens
        );
    }

    drop(locked_weights);
    let _ = std::fs::remove_file(&fixture_path);
}
