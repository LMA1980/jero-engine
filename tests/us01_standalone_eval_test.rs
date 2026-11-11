//! End-to-End Integration Test Suite for US-01 Scenarios 1–3 (`US-01.7.3`, #51, #1).
//!
//! Verifies all three core system assertions of `US-01: Standalone Local Model Evaluation`
//! without starting `k3-manager` or `k3-kv` background daemons:
//! - **Scenario 1**: Direct local model evaluation, architecture validation, `mlock` residency,
//!   in-process prompt tokenization, isolated single-prompt forward pass, and real-time token streaming.
//! - **Scenario 2**: Explicit compute cluster (`x100` / `a100`), core mask, and concurrency targeting
//!   with runtime `/proc/self/task/<tid>/status` (`Cpus_allowed_list`) verification.
//! - **Scenario 3**: Upfront resident memory verification and refusal (`InsufficientLockableMemory`)
//!   prior to execution when required weight + KV footprint exceeds lockable system memory.

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::PathBuf;

use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{GgufFile, K3Error, QuantType};
use jero_engine::{
    ClusterPreset, CpuTopologyInspector, LockedModelWeights, MockCpuTopology, MockMemoryInspector,
    SinglePromptEngine, SingleSlotKvArena, StreamAction, SystemCpuTopology, SystemMemoryInspector,
    VlenDomain, WorkerPool, X100_CLUSTER_MASK, resolve_affinity, resolve_affinity_with,
};
use jero_tokenizer::{DetokenizeStep, GgufTokenizer};

fn write_synthetic_gguf_fixture(tag: &str, architecture: &str) -> PathBuf {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3(architecture, 32, &tensor_specs);
    let path = std::env::temp_dir().join(format!("us01_fixture_{tag}_{}.gguf", std::process::id()));
    let mut f = File::create(&path).expect("create synthetic GGUF fixture");
    f.write_all(&bytes).expect("write GGUF fixture bytes");
    f.sync_all().expect("sync GGUF fixture");
    path
}

#[test]
fn test_us01_scenario_1_direct_local_evaluation_and_realtime_token_streaming() {
    let fixture_path = write_synthetic_gguf_fixture("scenario1", "synthetic");
    let n_ctx = 64;
    let max_tokens = 5;

    // 1. Open model with pre-flight capability filter, RAM admission, flock(LOCK_SH), mmap, and mlock
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, footprint) =
        LockedModelWeights::open_with_admission(&fixture_path, n_ctx, &inspector)
            .expect("Scenario 1: open and mlock supported synthetic model");

    assert_eq!(recipe.architecture_id(), "synthetic");
    assert!(footprint.weight_bytes > 0);
    assert!(footprint.kv_bytes > 0);

    // Confirm LOCK_SH is actively held during evaluation (blocks LOCK_EX)
    let probe_file = File::open(&fixture_path).expect("open probe handle");
    // SAFETY: Non-blocking flock(LOCK_EX | LOCK_NB) on open file descriptor.
    let lock_ex_during =
        unsafe { libc::flock(probe_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(
        lock_ex_during, 0,
        "Scenario 1: LOCK_EX must be blocked while model is resident"
    );

    // 2. Allocate and mlock single-slot 16-token paged KV cache arena
    let mut kv_arena =
        SingleSlotKvArena::new(locked_weights.metadata(), n_ctx).expect("allocate locked KV arena");

    // 3. Extract embedded GGUF vocabulary & chat template and tokenize input prompt
    let gguf_view = GgufFile::parse(locked_weights.as_bytes()).expect("parse locked GGUF view");
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view).expect("extract GGUF tokenizer");
    let prompt_tokens = tokenizer
        .tokenize_prompt("Explain RVV 1.0", true, n_ctx)
        .expect("format and tokenize prompt");
    assert!(!prompt_tokens.is_empty(), "prompt tokens must not be empty");

    // 4. Execute isolated single-prompt forward pass and verify progressive token-by-token streaming
    let mut engine = SinglePromptEngine::new();
    let mut detokenizer = tokenizer.streaming_detokenizer();
    let mut streamed_pieces: Vec<String> = Vec::new();

    let summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_arena,
            &prompt_tokens,
            tokenizer.vocab_size(),
            max_tokens,
            |token_id| match detokenizer.push_token(token_id)? {
                DetokenizeStep::Emit(piece) => {
                    streamed_pieces.push(piece);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(prefix) => {
                    if !prefix.is_empty() {
                        streamed_pieces.push(prefix);
                    }
                    Ok(StreamAction::Stop)
                }
            },
        )
        .expect("Scenario 1: isolated forward pass succeeds");

    assert_eq!(summary.generated_tokens.len(), max_tokens);
    assert_eq!(streamed_pieces.len(), max_tokens);
    assert_eq!(streamed_pieces.concat(), "helloworldExplainRVV1.0");

    // 5. Verify deterministic bit-identical output on repeated run at temperature 0
    let mut streamed_repeat: Vec<String> = Vec::new();
    let mut detok_repeat = tokenizer.streaming_detokenizer();
    let summary_repeat = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_arena,
            &prompt_tokens,
            tokenizer.vocab_size(),
            max_tokens,
            |token_id| match detok_repeat.push_token(token_id)? {
                DetokenizeStep::Emit(piece) => {
                    streamed_repeat.push(piece);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(prefix) => {
                    if !prefix.is_empty() {
                        streamed_repeat.push(prefix);
                    }
                    Ok(StreamAction::Stop)
                }
            },
        )
        .expect("repeat forward pass succeeds");

    assert_eq!(summary.generated_tokens, summary_repeat.generated_tokens);
    assert_eq!(streamed_pieces, streamed_repeat);

    // 6. Drop resident resources and verify flock is released
    drop(kv_arena);
    drop(locked_weights);

    // SAFETY: Re-testing LOCK_EX after resident weights drop.
    let lock_ex_after =
        unsafe { libc::flock(probe_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        lock_ex_after, 0,
        "LOCK_EX must succeed after LockedModelWeights is dropped"
    );
    // SAFETY: Unlocking probe_file.
    unsafe {
        libc::flock(probe_file.as_raw_fd(), libc::LOCK_UN);
    }
    let _ = std::fs::remove_file(&fixture_path);
}

#[test]
fn test_us01_scenario_2_explicit_compute_cluster_and_concurrency_targeting() {
    let fixture_path = write_synthetic_gguf_fixture("scenario2", "synthetic");
    let k3_topo = MockCpuTopology::spacemit_k3();

    // 1. Verify SpacemiT K3 cluster presets (X100: 0x00FF zvl256b; A100: 0xFF00 zvl1024b)
    let x100_resolved = resolve_affinity_with(Some("x100"), None, Some(4), &k3_topo)
        .expect("resolve 4 threads on x100");
    assert_eq!(x100_resolved.cluster, Some(ClusterPreset::X100));
    assert_eq!(x100_resolved.vlen_domain, VlenDomain::Zvl256b);
    assert_eq!(x100_resolved.assigned_cores, vec![0, 1, 2, 3]);
    assert_eq!(x100_resolved.effective_mask, 0x000F);

    let a100_resolved = resolve_affinity_with(Some("a100"), None, Some(4), &k3_topo)
        .expect("resolve 4 threads on a100");
    assert_eq!(a100_resolved.cluster, Some(ClusterPreset::A100));
    assert_eq!(a100_resolved.vlen_domain, VlenDomain::Zvl1024b);
    assert_eq!(a100_resolved.assigned_cores, vec![8, 9, 10, 11]);
    assert_eq!(a100_resolved.effective_mask, 0x0F00);

    // 2. Spawn pinned WorkerPool on host online cores and execute forward pass
    let online_mask = SystemCpuTopology::new()
        .online_cpu_mask()
        .expect("query online CPUs");
    let host_x100_mask = online_mask & X100_CLUSTER_MASK;
    let thread_count = (host_x100_mask.count_ones() as usize).clamp(1, 2);

    let host_affinity = resolve_affinity(None, Some(host_x100_mask), Some(thread_count))
        .expect("resolve host mask");
    let worker_pool = WorkerPool::new(host_affinity.clone()).expect("spawn pinned worker pool");
    assert_eq!(worker_pool.thread_count(), thread_count);

    // Verify /proc/self/task/<tid>/status Cpus_allowed_list before forward pass
    let statuses_before = worker_pool
        .verify_live_affinities()
        .expect("verify live affinities before forward pass");
    for (idx, status) in statuses_before.iter().enumerate() {
        assert_eq!(status.assigned_core, host_affinity.assigned_cores[idx]);
        assert_eq!(
            status.proc_cpus_allowed_list,
            status.assigned_core.to_string()
        );
    }

    // Run evaluation bound to the pinned worker pool
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _) =
        LockedModelWeights::open_with_admission(&fixture_path, 64, &inspector).expect("open model");
    let mut kv_arena = SingleSlotKvArena::new(locked_weights.metadata(), 64).expect("kv arena");
    let mut engine = SinglePromptEngine::new();

    let summary = engine
        .evaluate_prompt_with_pool(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_arena,
            Some(&worker_pool),
            &[1, 9, 10],
            16,
            3,
            |_tok| Ok(StreamAction::Continue),
        )
        .expect("forward pass on pinned worker pool");
    assert_eq!(summary.generated_tokens, vec![9, 10, 11]);

    // Verify /proc/self/task/<tid>/status Cpus_allowed_list after forward pass (zero migration)
    let statuses_after = worker_pool
        .verify_live_affinities()
        .expect("verify live affinities after forward pass");
    assert_eq!(statuses_before, statuses_after);

    let _ = std::fs::remove_file(&fixture_path);
}

#[test]
fn test_us01_scenario_3_resident_memory_verification_refuses_oversized_run_upfront() {
    let fixture_path = write_synthetic_gguf_fixture("scenario3", "synthetic");

    // Simulate constrained physical RAM (512 bytes available) where model weights + KV arena cannot fit
    let constrained_inspector = MockMemoryInspector::new(512, 100_000_000);
    let err = LockedModelWeights::open_with_admission(&fixture_path, 64, &constrained_inspector)
        .expect_err("Scenario 3: must refuse run upfront when model + KV exceeds available RAM");

    assert_eq!(err.diagnostic_code(), "insufficient_lockable_memory");
    match err {
        K3Error::InsufficientLockableMemory {
            required_bytes,
            weight_bytes,
            kv_bytes,
            available_bytes,
            rlimit_memlock_bytes,
        } => {
            assert!(required_bytes > available_bytes);
            assert_eq!(required_bytes, weight_bytes + kv_bytes);
            assert_eq!(available_bytes, 512);
            assert_eq!(rlimit_memlock_bytes, 100_000_000);
        }
        other => panic!("expected InsufficientLockableMemory, got: {other:?}"),
    }

    // Verify that the shared file lock was cleanly released upon upfront refusal
    let probe_file = File::open(&fixture_path).expect("open probe handle");
    // SAFETY: Non-blocking flock(LOCK_EX | LOCK_NB) to verify LOCK_SH was cleaned up.
    let lock_ex = unsafe { libc::flock(probe_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(
        lock_ex, 0,
        "flock(LOCK_SH) must be released after upfront memory refusal"
    );
    // SAFETY: Unlocking probe_file.
    unsafe {
        libc::flock(probe_file.as_raw_fd(), libc::LOCK_UN);
    }

    let _ = std::fs::remove_file(&fixture_path);
}
