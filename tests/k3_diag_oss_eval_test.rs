//! Integration tests for `k3-diag` Open-Source AI Runtime Benchmark Evaluator (`--eval-suite`).
//!
//! Validates all 5 `BenchLM` / Confident AI (`DeepEval`) / `EleutherAI` (`lm-evaluation-harness`)
//! capability categories (`mmlu_pro`, `gsm8k`, `humaneval`, `ifeval`, `truthfulqa`), runtime
//! efficiency telemetry (`TTFT`, `prefill_tps`, `decode_tps`, hardware ceilings), Quantization
//! & Kernel Fidelity Delta (`FP16` vs quantized), and `MuObserve` telemetry export.

use std::env;
use std::fs::File;
use std::io::Write;
use std::process::Command;

use jero_core::QuantType;
use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_engine::CpuTopologyInspector;
use serde_json::Value;

fn k3_diag_bin() -> String {
    env!("CARGO_BIN_EXE_jero-diag").to_owned()
}

fn create_temp_synthetic_gguf(tag: &str) -> std::path::PathBuf {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
    let path = env::temp_dir().join(format!("k3_diag_oss_{}_{}.gguf", tag, std::process::id()));
    let mut f = File::create(&path).expect("create temp gguf");
    f.write_all(&bytes).expect("write gguf");
    f.sync_all().expect("sync gguf");
    path
}

#[test]
fn test_k3_diag_eval_suite_all_json_and_export() {
    let model_path = create_temp_synthetic_gguf("all_json");
    let export_path =
        env::temp_dir().join(format!("k3_diag_oss_export_{}.json", std::process::id()));

    let output = Command::new(k3_diag_bin())
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--eval-suite",
            "all",
            "--cluster",
            "x100",
            "--threads",
            "2",
            "--n-ctx",
            "64",
            "--max-tokens",
            "8",
            "--json",
            "--export-eval",
            export_path.to_str().unwrap(),
        ])
        .output()
        .expect("execute k3-diag --eval-suite all");

    assert!(
        output.status.success(),
        "k3-diag failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8(output.stdout).expect("valid utf8 stdout");
    let report: Value = serde_json::from_str(&stdout_str).expect("valid JSON report");

    assert_eq!(report["report_schema_version"], "1.0.0");
    assert_eq!(report["architecture"], "synthetic");
    assert_eq!(report["selected_suite"], "all");
    assert!(report["benchlm_composite_index"].as_f64().unwrap() > 0.0);

    let categories = report["categories"].as_array().expect("categories array");
    assert_eq!(
        categories.len(),
        5,
        "all 5 BenchLM / DeepEval pillars must be present"
    );

    let suites: Vec<&str> = categories
        .iter()
        .map(|c| c["suite"].as_str().unwrap())
        .collect();
    assert!(suites.contains(&"mmlu_pro"));
    assert!(suites.contains(&"gsm8k"));
    assert!(suites.contains(&"humaneval"));
    assert!(suites.contains(&"ifeval"));
    assert!(suites.contains(&"truthfulqa"));

    assert_eq!(
        report["evidence_provenance"]["evidence_confidence"],
        "verified_high"
    );
    assert_eq!(
        report["evidence_provenance"]["contamination_flag"],
        "verified_clean"
    );
    assert!(
        report["quantization_fidelity"]["ime_vs_rvv_numerically_identical"]
            .as_bool()
            .unwrap()
    );
    assert_eq!(
        report["muobserve_records"]
            .as_array()
            .expect("muobserve records")
            .len(),
        5
    );

    assert!(export_path.exists(), "--export-eval file must be created");
    let exported_raw = std::fs::read_to_string(&export_path).expect("read exported json");
    let exported_val: Value = serde_json::from_str(&exported_raw).expect("parse exported json");
    assert_eq!(
        exported_val["benchlm_composite_index"],
        report["benchlm_composite_index"]
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&export_path);
}

#[test]
fn test_k3_diag_eval_custom_jsonl_dataset_and_human_scorecard() {
    let model_path = create_temp_synthetic_gguf("custom_jsonl");
    let dataset_path = env::temp_dir().join(format!(
        "k3_diag_custom_dataset_{}.jsonl",
        std::process::id()
    ));
    let jsonl_content = r#"{"id":"gsm8k_custom_1","suite":"gsm8k","category":"Mathematical Reasoning","prompt":"2 + 2 = ####","expected_answer":"4","scoring_method":"numerical_extract","source_license":"MIT","contamination_status":"verified_clean"}"#;
    std::fs::write(&dataset_path, format!("{jsonl_content}\n")).expect("write custom jsonl");

    let topo = jero_engine::affinity::SystemCpuTopology::new();
    let a100_online = topo.online_cpu_mask().is_ok_and(|m| (m & 0xFF00) != 0);
    let (cluster, ime) = if a100_online {
        ("a100", "on")
    } else {
        ("x100", "off")
    };

    let output = Command::new(k3_diag_bin())
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--eval-suite",
            "gsm8k",
            "--eval-dataset",
            dataset_path.to_str().unwrap(),
            "--cluster",
            cluster,
            "--ime",
            ime,
            "--threads",
            "2",
            "--n-ctx",
            "64",
            "--max-tokens",
            "8",
        ])
        .output()
        .expect("execute k3-diag custom dataset");

    assert!(
        output.status.success(),
        "k3-diag failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout_str = String::from_utf8(output.stdout).expect("valid utf8");
    assert!(stdout_str.contains("Public Open-Source AI Runtime Benchmark Scorecard"));
    assert!(stdout_str.contains("BenchLM Composite Index"));
    assert!(stdout_str.contains("gsm8k"));
    assert!(stdout_str.contains("IME-vs-RVV Bit-Identical=true"));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&dataset_path);
}
