//! Integration tests for phase-scoped heterogeneous cluster dispatch (`#121`–`#129`)
//! and dynamic GGUF hyperparameter recipe hydration (`#113`–`#115`).

use std::fs::File;
use std::io::Write;
use std::process::Command;

use jero_core::test_helpers::{create_synthetic_gguf_v3, create_synthetic_gguf_v3_with_kvs};
use jero_core::{GgufFile, GgufValue, QuantType};
use jero_engine::{
    ClusterPhasePolicy, LockedModelWeights, SinglePromptEngine, SingleSlotKvArena, StreamAction,
    SystemMemoryInspector, verify_model_capabilities,
};

const DIAG_BIN: &str = env!("CARGO_BIN_EXE_jero-diag");

#[cfg(feature = "jero-arch-llama")]
#[test]
fn test_issue_113_qwen25_15b_1536_151936_untied_lm_head_forward_pass() {
    let extra_kvs = [
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(1536)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(8960)),
        ("qwen2.attention.head_count", GgufValue::Uint32(12)),
        ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
        ("qwen2.vocab_size", GgufValue::Uint32(151_936)),
        ("qwen2.rope.freq_base", GgufValue::Float32(1_000_000.0)),
        (
            "qwen2.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(1e-6),
        ),
    ];
    let output_shape = [1536_usize, 151_936_usize];
    let norm_shape = [1536_usize];
    let tensor_specs = [
        ("output_norm.weight", QuantType::F32, norm_shape.as_slice()),
        ("output.weight", QuantType::Q4_K_M, output_shape.as_slice()),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &extra_kvs, &tensor_specs);
    let gguf = GgufFile::parse(&bytes).expect("parse Qwen2.5-1.5B synthetic GGUF");
    let recipe =
        verify_model_capabilities(&gguf).expect("verify and hydrate Qwen2.5-1.5B capabilities");
    assert_eq!(recipe.architecture_id(), "qwen2");

    let mut kv_arena = SingleSlotKvArena::from_gguf(&gguf, 16).expect("allocate KV arena");
    assert_eq!(kv_arena.layer_count(), 1);
    assert_eq!(kv_arena.head_count_kv(), 2);
    assert_eq!(kv_arena.head_dim(), 128);

    let mut engine = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::Auto);
    let summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &gguf,
            &mut kv_arena,
            &[1, 2, 3, 4],
            151_936,
            1,
            |_| Ok(StreamAction::Continue),
        )
        .expect(
            "Qwen2.5-1.5B [1536, 151936] LM head projection must succeed without TensorBoundsError",
        );
    assert_eq!(summary.generated_tokens.len(), 1);
    assert!((summary.generated_tokens[0] as usize) < 151_936);

    // Also verify GGUF alignment-padded output.weight ([1536, 151936]) when tokenizer vocab is 151665
    let padded_summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &gguf,
            &mut kv_arena,
            &[1, 2],
            151_665,
            1,
            |_| Ok(StreamAction::Continue),
        )
        .expect(
            "padded Qwen2.5 output.weight [1536, 151936] with tokenizer vocab 151665 must succeed",
        );
    assert_eq!(padded_summary.generated_tokens.len(), 1);
    assert!((padded_summary.generated_tokens[0] as usize) < 151_665);
}

#[cfg(feature = "jero-arch-llama")]
#[test]
fn test_issue_113_locked_model_weights_and_cli_1536_dim_end_to_end() {
    let extra_kvs = [
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(1536)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(8960)),
        ("qwen2.attention.head_count", GgufValue::Uint32(12)),
        ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
    ];
    let embd_shape = [1536_usize, 16_usize];
    let norm_shape = [1536_usize];
    let output_shape = [1536_usize, 16_usize];
    let tensor_specs = [
        (
            "token_embd.weight",
            QuantType::Q4_K_M,
            embd_shape.as_slice(),
        ),
        ("output_norm.weight", QuantType::F32, norm_shape.as_slice()),
        ("output.weight", QuantType::Q4_K_M, output_shape.as_slice()),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &extra_kvs, &tensor_specs);
    let model_path = std::env::temp_dir().join(format!(
        "k3_issue113_qwen2_1536_{}.gguf",
        std::process::id()
    ));
    {
        let mut f = File::create(&model_path).expect("create temp qwen2 gguf");
        f.write_all(&bytes).expect("write qwen2 gguf");
        f.sync_all().expect("sync qwen2 gguf");
    }

    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _) =
        LockedModelWeights::open_with_admission(&model_path, 16, &inspector)
            .expect("open_with_admission for 1536-dim qwen2");
    assert_eq!(recipe.architecture_id(), "qwen2");
    let mut kv_arena = SingleSlotKvArena::new(locked_weights.metadata(), 16).expect("kv arena");
    let mut engine = SinglePromptEngine::new();
    let summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_arena,
            &[1, 2, 3, 4],
            16,
            2,
            |_| Ok(StreamAction::Continue),
        )
        .expect("evaluate 1536-dim qwen2 via LockedModelWeights");
    assert_eq!(summary.generated_tokens.len(), 2);
    drop(kv_arena);
    drop(locked_weights);

    // Verify jero-diag CLI execution with --phase-policy auto on the 1536-dim qwen2 fixture
    let output = Command::new(DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().expect("utf8 path"),
            "--prompt",
            "hello world",
            "--phase-policy",
            "auto",
            "--n-ctx",
            "16",
            "--max-tokens",
            "2",
        ])
        .output()
        .expect("run jero-diag with --phase-policy auto");

    let _ = std::fs::remove_file(&model_path);

    assert!(
        output.status.success(),
        "jero-diag --phase-policy auto failed on 1536-dim qwen2 ({:?}): {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_cli_phase_policy_streaming_multirun_and_error_codes() {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
    let model_path =
        std::env::temp_dir().join(format!("k3_phase_policy_cli_{}.gguf", std::process::id()));
    {
        let mut f = File::create(&model_path).expect("create synthetic gguf");
        f.write_all(&bytes).expect("write synthetic gguf");
        f.sync_all().expect("sync synthetic gguf");
    }
    let path_str = model_path.to_str().expect("utf8 path");

    // 1. Single-pass streaming with --phase-policy x100-only --cores 0x0001 --threads 1
    let out_stream = Command::new(DIAG_BIN)
        .args([
            "--model",
            path_str,
            "--prompt",
            "hello world",
            "--phase-policy",
            "x100-only",
            "--cores",
            "0x0001",
            "--threads",
            "1",
            "--n-ctx",
            "32",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("run x100-only streaming");
    assert!(
        out_stream.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_stream.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out_stream.stdout).trim(),
        "helloworldExplainRVV"
    );

    // 2. Multi-run JSON with --phase-policy auto under restricted K3_CPU_ONLINE_MASK=0x000F fallback
    let out_json = Command::new(DIAG_BIN)
        .env("K3_CPU_ONLINE_MASK", "0x000F")
        .args([
            "--model",
            path_str,
            "--prompt",
            "hello world",
            "--phase-policy",
            "auto",
            "--threads",
            "1",
            "--repeat",
            "3",
            "--n-ctx",
            "32",
            "--max-tokens",
            "3",
            "--json",
        ])
        .output()
        .expect("run auto multi-run json");
    assert!(
        out_json.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_json.stderr)
    );
    let stdout_json = String::from_utf8_lossy(&out_json.stdout);
    assert!(stdout_json.contains("\"coordination_profile\": \"rayon\""));
    assert!(stdout_json.contains("\"determinism_verified\": true"));

    // 3. Explicit --phase-policy a100-only fails closed (exit code 13) when A100 cores are offline
    let out_a100_err = Command::new(DIAG_BIN)
        .env("K3_CPU_ONLINE_MASK", "0x000F")
        .args([
            "--model",
            path_str,
            "--prompt",
            "hello world",
            "--phase-policy",
            "a100-only",
            "--n-ctx",
            "32",
            "--max-tokens",
            "2",
        ])
        .output()
        .expect("run a100-only on x100-only mask");
    assert_eq!(out_a100_err.status.code(), Some(13));

    // 4. Invalid --phase-policy and --phase-policy + --cluster conflict exit with code 2
    let out_bad_flag = Command::new(DIAG_BIN)
        .args(["--model", path_str, "--phase-policy", "invalid-policy"])
        .output()
        .expect("run invalid --phase-policy");
    assert_eq!(out_bad_flag.status.code(), Some(2));

    let out_conflict = Command::new(DIAG_BIN)
        .args([
            "--model",
            path_str,
            "--phase-policy",
            "auto",
            "--cluster",
            "x100",
        ])
        .output()
        .expect("run conflicting --phase-policy and --cluster");
    assert_eq!(out_conflict.status.code(), Some(2));

    let _ = std::fs::remove_file(&model_path);
}
