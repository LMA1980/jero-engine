//! US-04: Hardware Ceiling Cost Profiling & Machine-Readable Export
//! End-to-End Integration and CLI Test Suite (`k3-diag`).
//!
//! Validates:
//! - Scenario 1: Hardware ceiling efficiency reporting (ADR-034):
//!   * Prefill throughput (`tokens/s`), latency (`ms`), cost (`ms/token`), and `% of peak_macs`.
//!   * Decode throughput (`tokens/s`), latency (`ms`), cost (`ms/token`), and `% of memory_bandwidth`.
//!   * Objective reporting: no arbitrary pass/fail speed threshold (exits code 0).
//!   * Output in both human-readable table and `--json` modes across single/dual clusters and `--repeat`.
//! - Scenario 2: Machine-readable cost baseline export (ADR-010, ADR-032):
//!   * Export via `--json` and/or `--export-profile <path>`.
//!   * Valid structured `CostProfile` schema (`model_identity`, `quantization_level`, `cluster_placement`,
//!     `allocated_cores`, `split_strategy`, `execution_cost`, `memory_footprint`, `tuning_advisory`).
//!   * Quantization detection across GGUF tensor formats (`Q4_K_M`, `Q8_0`, `F32`).
//!   * Memory footprint consistency (`peak_resident_bytes >= total_required_bytes >= weight_bytes + kv_bytes`).
//!   * Derived static scheduling weight (`static_scheduling_weight`).
//!   * Evidence-based tuning advisory strictly preserving operator tuning authority (ADR-032).
//! - Error handling & CLI boundary testing:
//!   * Exit code 2 on missing `--export-profile` argument.
//!   * Exit code 2 on invalid flag combinations.
//!   * Exit code 10 on capability mismatches.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use jero_core::test_helpers::create_synthetic_gguf_v3;
use jero_core::{CostProfile, QuantType};
use serde_json::Value;

static RAND_SEED: AtomicU64 = AtomicU64::new(100);

fn fastrand() -> u64 {
    let prev = RAND_SEED.fetch_add(1, Ordering::Relaxed);
    prev.wrapping_mul(6364136223846793005).wrapping_add(1)
}

struct TempModelFixture {
    path: PathBuf,
}

#[allow(dead_code)]
impl TempModelFixture {
    fn new(tag: &str) -> Self {
        Self::with_specs(
            tag,
            &[
                ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
                ("output_norm.weight", QuantType::F32, &[16][..]),
            ],
        )
    }

    fn with_specs(tag: &str, specs: &[(&str, QuantType, &[usize])]) -> Self {
        let bytes = create_synthetic_gguf_v3("synthetic", 32, specs);
        let path = std::env::temp_dir().join(format!(
            "k3_us04_e2e_{tag}_{}_{}.gguf",
            std::process::id(),
            fastrand()
        ));
        let mut f = File::create(&path).expect("create temp synthetic GGUF model");
        f.write_all(&bytes).expect("write GGUF bytes");
        f.sync_all().expect("sync GGUF file");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn path_str(&self) -> &str {
        self.path.to_str().expect("valid utf8 path")
    }
}

impl Drop for TempModelFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn run_diag(args: &[&str]) -> (std::process::Output, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jero-diag"));
    cmd.env("K3_CPU_ONLINE_MASK", "0xFFFF");
    for arg in args {
        cmd.arg(arg);
    }
    let out = cmd.output().expect("execute k3-diag");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    (out, stdout, stderr)
}

fn run_diag_json(args: &[&str]) -> (std::process::Output, Value, String) {
    let mut full_args = Vec::from(args);
    if !full_args.contains(&"--json") {
        full_args.push("--json");
    }
    let (out, stdout, stderr) = run_diag(&full_args);
    let val: Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("failed to parse JSON from stdout: {e}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    });
    (out, val, stderr)
}

// ============================================================================
// Scenario 1: Hardware Ceiling Efficiency Reporting (ADR-034)
// ============================================================================

#[test]
fn test_us04_s1_single_cluster_x100_ceiling_reporting() {
    let fixture = TempModelFixture::new("s1_x100");
    let (out, val, _stderr) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "3",
    ]);

    assert!(out.status.success(), "exit status must be 0");

    // R1: Prefill efficiency
    let prefill_tps = val["prefill_tps"].as_f64().expect("prefill_tps as f64");
    let prefill_ms = val["prefill_latency_ms"]
        .as_f64()
        .expect("prefill_latency_ms as f64");
    let prefill_cost_per_tok = val["prefill_cost_per_token_ms"]
        .as_f64()
        .expect("prefill_cost_per_token_ms as f64");
    let prefill_ceiling = val["prefill_hardware_ceiling_pct"]
        .as_f64()
        .expect("prefill_hardware_ceiling_pct as f64");

    assert!(prefill_tps > 0.0, "prefill throughput must be positive");
    assert!(prefill_ms > 0.0, "prefill latency must be positive");
    assert!(
        prefill_cost_per_tok > 0.0,
        "prefill cost per token must be positive"
    );
    assert!(
        prefill_ceiling >= 0.0,
        "prefill ceiling pct must be non-negative"
    );

    // R1: Decode efficiency
    let decode_tps = val["decode_tps"].as_f64().expect("decode_tps as f64");
    let decode_ms = val["decode_latency_ms"]
        .as_f64()
        .expect("decode_latency_ms as f64");
    let decode_cost_per_tok = val["decode_cost_per_token_ms"]
        .as_f64()
        .expect("decode_cost_per_token_ms as f64");
    let decode_ceiling = val["decode_hardware_ceiling_pct"]
        .as_f64()
        .expect("decode_hardware_ceiling_pct as f64");

    assert!(decode_tps > 0.0, "decode throughput must be positive");
    assert!(decode_ms > 0.0, "decode latency must be positive");
    assert!(
        decode_cost_per_tok > 0.0,
        "decode cost per token must be positive"
    );
    assert!(
        decode_ceiling >= 0.0,
        "decode ceiling pct must be non-negative"
    );

    // PhaseReport structured verification
    assert_eq!(val["prefill"]["ceiling_metric"].as_str(), Some("peak_macs"));
    assert_eq!(
        val["decode"]["ceiling_metric"].as_str(),
        Some("memory_bandwidth")
    );
}

#[test]
fn test_us04_s1_single_cluster_a100_ceiling_reporting() {
    let fixture = TempModelFixture::new("s1_a100");

    // Arm 1: A100 with IME off
    let (out_off, val_off, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "off",
        "--max-tokens",
        "3",
    ]);
    assert!(out_off.status.success());
    assert!(val_off["prefill_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
    assert!(val_off["decode_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);

    // Arm 2: A100 with IME on
    let (out_on, val_on, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "3",
    ]);
    assert!(out_on.status.success());
    assert!(val_on["prefill_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
    assert!(val_on["decode_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
    assert!(
        val_on["cost_profile"]["tuning_advisory"]["recommended_ime"]
            .as_bool()
            .unwrap()
    );
}

#[test]
fn test_us04_s1_dual_cluster_splits_ceiling_reporting() {
    let fixture = TempModelFixture::new("s1_both");

    for split in ["phase", "op", "row"] {
        let (out, val, _err) = run_diag_json(&[
            "--model",
            fixture.path_str(),
            "--cluster",
            "both",
            "--split",
            split,
            "--max-tokens",
            "3",
        ]);
        assert!(out.status.success(), "split {split} must exit with 0");
        assert!(val["prefill_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
        assert!(val["decode_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
        assert_eq!(val["split_strategy"].as_str(), Some(split));
        assert_eq!(val["cost_profile"]["split_strategy"].as_str(), Some(split));
    }
}

#[test]
fn test_us04_s1_human_readable_table_format() {
    let fixture = TempModelFixture::new("s1_table");

    // Single-cluster table summary
    let (out, stdout, _stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--summary",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    println!("\n=== Single-Cluster X100 Summary Table ===\n{stdout}");
    assert!(stdout.contains("Prefill"), "must contain Prefill row");
    assert!(stdout.contains("Decode"), "must contain Decode row");
    assert!(stdout.contains("ms/token"), "must display cost in ms/token");
    assert!(
        stdout.contains("% of peak_macs ceiling"),
        "must display compute ceiling ratio"
    );
    assert!(
        stdout.contains("% of memory_bandwidth ceiling"),
        "must display bandwidth ceiling ratio"
    );
    assert!(
        stdout.contains("Cost Profile Baseline (US-04):"),
        "must display baseline header"
    );
    assert!(
        stdout.contains("Tuning Advisory"),
        "must display tuning advisory"
    );

    // Dual-cluster table summary
    let (out2, stdout2, _stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "3",
    ]);
    assert!(out2.status.success());
    println!("\n=== Dual-Cluster (Phase Split) Summary Table ===\n{stdout2}");
    assert!(stdout2.contains("Split Policy         : phase"));
    assert!(stdout2.contains("Cost Profile Baseline (US-04):"));
}

#[test]
fn test_us04_s1_multi_repeat_benchmark() {
    let fixture = TempModelFixture::new("s1_repeat");
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(val["repeat_count"].as_u64(), Some(3));
    assert_eq!(val["determinism_verified"].as_bool(), Some(true));
    assert!(
        val["cost_profile"]["prefill_cost_per_token_ms"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        val["cost_profile"]["decode_cost_per_token_ms"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn test_us04_s1_objective_reporting_no_speed_threshold() {
    let fixture = TempModelFixture::new("s1_obj");
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ADR-034: ceiling attainment is informational with no arbitrary speed pass/fail threshold"
    );
    assert!(val["prefill_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
    assert!(val["decode_hardware_ceiling_pct"].as_f64().unwrap() >= 0.0);
}

// ============================================================================
// Scenario 2: Machine-Readable Cost Baseline Export & Shared Schema
// ============================================================================

#[test]
fn test_us04_s2_json_contains_valid_cost_profile_schema() {
    let fixture = TempModelFixture::new("s2_schema");
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());

    let cp_val = &val["cost_profile"];
    assert!(
        cp_val.is_object(),
        "cost_profile must be a structured JSON object"
    );

    // Deserialize into typed CostProfile
    let cp: CostProfile = serde_json::from_value(cp_val.clone()).expect("parse CostProfile schema");

    // 1. Model identity
    assert!(
        !cp.model_identity.name.is_empty(),
        "model name must not be empty"
    );
    assert_eq!(cp.model_identity.architecture, "synthetic");
    assert_eq!(cp.model_identity.file_path, fixture.path_str());
    assert!(
        !cp.model_identity.file_digest.is_empty(),
        "digest must not be empty"
    );

    // 2. Quantization level
    assert_eq!(cp.quantization_level.primary, "Q4_K_M");
    assert_eq!(cp.quantization_level.summary.get("Q4_K_M"), Some(&1));
    assert_eq!(cp.quantization_level.summary.get("F32"), Some(&1));
    assert_eq!(cp.quantization_level.tensor_bytes.get("Q4_K_M"), Some(&288));
    assert_eq!(cp.quantization_level.tensor_bytes.get("F32"), Some(&64));

    // 3. Cluster placement & core allocation
    // Per ADR-007, x100 defaults to 6 compute threads on cores 0..5 (0x003F), reserving 2 cores for OS duties
    assert_eq!(cp.cluster_placement.cluster, "x100");
    assert_eq!(cp.cluster_placement.effective_core_mask, "0x003F");
    assert_eq!(cp.cluster_placement.allocated_cores, 6);
    assert_eq!(cp.cluster_placement.x100_cores, 6);
    assert_eq!(cp.cluster_placement.a100_cores, 0);
    assert_eq!(cp.split_strategy, "none");
    assert_eq!(cp.allocated_cores.total, 6);
    assert_eq!(cp.allocated_cores.total_cores, 6);
    assert_eq!(cp.allocated_cores.x100_cores, 6);
    assert_eq!(cp.allocated_cores.a100_cores, 0);

    // 4. Execution cost baseline
    assert!(cp.prefill_cost_per_token_ms > 0.0);
    assert!(cp.prefill_tokens_per_sec > 0.0);
    assert!(cp.prefill_hardware_ceiling_pct >= 0.0);
    assert!(cp.decode_tokens_per_sec > 0.0);
    assert!(cp.decode_cost_per_token_ms > 0.0);
    assert!(cp.decode_hardware_ceiling_pct >= 0.0);
    assert!(cp.static_scheduling_weight > 0.0);
    assert_eq!(cp.static_scheduling_weight, cp.scheduling_weight);

    // Mathematical consistency: latency per token should closely match 1000 / throughput
    let expected_decode_cost = 1000.0 / cp.decode_tokens_per_sec;
    let ratio = cp.decode_cost_per_token_ms / expected_decode_cost;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "decode cost ({}) must align with expected ({}) ratio={}",
        cp.decode_cost_per_token_ms,
        expected_decode_cost,
        ratio
    );

    // 5. Memory footprint consistency
    assert!(cp.weight_bytes > 0);
    assert!(cp.kv_bytes > 0);
    assert_eq!(cp.total_required_bytes, cp.weight_bytes + cp.kv_bytes);
    assert!(
        cp.peak_resident_bytes >= cp.total_required_bytes,
        "peak_resident_bytes ({}) must be >= total_required_bytes ({})",
        cp.peak_resident_bytes,
        cp.total_required_bytes
    );

    // 6. Tuning advisory
    assert!(
        cp.tuning_advisory.bottleneck_classification == "bandwidth-bound"
            || cp.tuning_advisory.bottleneck_classification == "compute-bound"
    );
    assert!(!cp.tuning_advisory.recommended_cluster.is_empty());
    assert!(!cp.tuning_advisory.recommendations.is_empty());
    assert!(
        cp.tuning_advisory
            .operator_authority_notice
            .contains("ADR-032")
    );
}

#[test]
fn test_us04_s2_export_profile_file_generation() {
    let fixture = TempModelFixture::new("s2_file");
    let export_path = std::env::temp_dir().join(format!(
        "k3_export_profile_{}_{}.json",
        std::process::id(),
        fastrand()
    ));

    let (out, stdout, _err) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--export-profile",
        export_path.to_str().unwrap(),
        "--max-tokens",
        "3",
    ]);

    assert!(out.status.success());
    assert!(
        export_path.exists(),
        "exported profile JSON file must exist"
    );

    // Stdout was table format because --json was not passed
    assert!(stdout.contains("Cost Profile Baseline (US-04):"));

    // File content is valid CostProfile JSON
    let file_content = std::fs::read_to_string(&export_path).expect("read exported profile");
    let profile: CostProfile =
        serde_json::from_str(&file_content).expect("parse exported profile JSON");

    assert_eq!(profile.model_identity.architecture, "synthetic");
    assert_eq!(profile.cluster_placement.cluster, "x100");
    assert!(profile.peak_resident_bytes >= profile.total_required_bytes);

    let _ = std::fs::remove_file(&export_path);
}

#[test]
fn test_us04_s2_export_profile_and_json_match() {
    let fixture = TempModelFixture::new("s2_match");
    let export_path = std::env::temp_dir().join(format!(
        "k3_export_profile_match_{}_{}.json",
        std::process::id(),
        fastrand()
    ));

    let (out, json_val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--export-profile",
        export_path.to_str().unwrap(),
        "--max-tokens",
        "3",
    ]);

    assert!(out.status.success());
    assert!(export_path.exists());

    let file_content = std::fs::read_to_string(&export_path).expect("read file");
    let profile_from_file: CostProfile = serde_json::from_str(&file_content).expect("parse file");
    let profile_from_json: CostProfile =
        serde_json::from_value(json_val["cost_profile"].clone()).expect("parse json");

    assert_eq!(profile_from_file, profile_from_json);

    let _ = std::fs::remove_file(&export_path);
}

#[test]
fn test_us04_s2_quantization_detection_multi_quant() {
    let specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("blk.0.attn_k.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let fixture = TempModelFixture::with_specs("s2_q4km", &specs);
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());

    let cp: CostProfile = serde_json::from_value(val["cost_profile"].clone()).unwrap();
    assert_eq!(cp.quantization_level.primary, "Q4_K_M");
    assert_eq!(cp.quantization_level.summary.get("Q4_K_M"), Some(&2));
    assert_eq!(cp.quantization_level.summary.get("F32"), Some(&1));
    assert_eq!(cp.quantization_level.tensor_bytes.get("Q4_K_M"), Some(&576));
}

#[test]
fn test_us04_s2_quantization_detection_f32_uniform() {
    let specs = [
        ("blk.0.weight", QuantType::F32, &[16][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let fixture = TempModelFixture::with_specs("s2_f32", &specs);
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());

    let cp: CostProfile = serde_json::from_value(val["cost_profile"].clone()).unwrap();
    assert_eq!(cp.quantization_level.primary, "F32");
    assert_eq!(cp.quantization_level.summary.get("F32"), Some(&2));
    assert_eq!(cp.quantization_level.tensor_bytes.get("F32"), Some(&128));
}

#[test]
fn test_us04_s2_quantization_detection_f16() {
    let specs = [
        ("blk.0.weight", QuantType::F16, &[32][..]),
        ("output_norm.weight", QuantType::F32, &[8][..]),
    ];
    let fixture = TempModelFixture::with_specs("s2_f16", &specs);
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());

    let cp: CostProfile = serde_json::from_value(val["cost_profile"].clone()).unwrap();
    // F16 has 64 bytes (32 * 2), F32 has 32 bytes (8 * 4) -> F16 is primary
    assert_eq!(cp.quantization_level.primary, "F16");
    assert_eq!(cp.quantization_level.summary.get("F16"), Some(&1));
    assert_eq!(cp.quantization_level.summary.get("F32"), Some(&1));
    assert_eq!(cp.quantization_level.tensor_bytes.get("F16"), Some(&64));
    assert_eq!(cp.quantization_level.tensor_bytes.get("F32"), Some(&32));
}

#[test]
fn test_us04_s2_memory_footprint_invariants() {
    let fixture = TempModelFixture::new("s2_mem");
    let (out, val, _err) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--n-ctx",
        "128",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());

    let cp: CostProfile = serde_json::from_value(val["cost_profile"].clone()).unwrap();

    assert!(cp.weight_bytes > 0);
    assert!(cp.kv_bytes > 0);
    assert_eq!(cp.total_required_bytes, cp.weight_bytes + cp.kv_bytes);
    assert!(
        cp.peak_resident_bytes >= cp.total_required_bytes,
        "peak_resident_bytes >= total_required_bytes invariant failed"
    );
    assert_eq!(cp.memory_footprint.weight_bytes, cp.weight_bytes);
    assert_eq!(cp.memory_footprint.kv_bytes, cp.kv_bytes);
    assert_eq!(
        cp.memory_footprint.total_required_bytes,
        cp.total_required_bytes
    );
    assert_eq!(
        cp.memory_footprint.peak_resident_bytes,
        cp.peak_resident_bytes
    );
}

#[test]
fn test_us04_s2_allocated_cores_breakdown() {
    let fixture = TempModelFixture::new("s2_cores");

    // X100: defaults to 6 X100 cores, 0 A100 cores (ADR-007)
    let (out_x, val_x, _) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out_x.status.success());
    let cp_x: CostProfile = serde_json::from_value(val_x["cost_profile"].clone()).unwrap();
    assert_eq!(cp_x.allocated_cores.x100_cores, 6);
    assert_eq!(cp_x.allocated_cores.a100_cores, 0);
    assert_eq!(cp_x.allocated_cores.total, 6);

    // A100: 0 X100 cores, 8 A100 cores
    let (out_a, val_a, _) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "2",
    ]);
    assert!(out_a.status.success());
    let cp_a: CostProfile = serde_json::from_value(val_a["cost_profile"].clone()).unwrap();
    assert_eq!(cp_a.allocated_cores.x100_cores, 0);
    assert_eq!(cp_a.allocated_cores.a100_cores, 8);
    assert_eq!(cp_a.allocated_cores.total, 8);

    // Both (phase split): 6 X100 cores, 8 A100 cores = 14 total cores (ADR-007)
    let (out_b, val_b, _) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "2",
    ]);
    assert!(out_b.status.success());
    let cp_b: CostProfile = serde_json::from_value(val_b["cost_profile"].clone()).unwrap();
    assert_eq!(cp_b.allocated_cores.x100_cores, 6);
    assert_eq!(cp_b.allocated_cores.a100_cores, 8);
    assert_eq!(cp_b.allocated_cores.total, 14);
}

// ============================================================================
// CLI Error Handling & Boundary Testing
// ============================================================================

#[test]
fn test_us04_error_missing_export_profile_argument() {
    let fixture = TempModelFixture::new("err_missing_arg");
    let (out, _stdout, stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--export-profile", // missing path argument
    ]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "missing argument to --export-profile must exit with code 2"
    );
    assert!(stderr.contains("--export-profile requires a file path"));
}

#[test]
fn test_us04_error_capability_mismatch_ime_on_x100() {
    let fixture = TempModelFixture::new("err_ime_x100");
    let (out, _stdout, stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--ime",
        "on",
    ]);
    assert_eq!(
        out.status.code(),
        Some(10),
        "--ime on on cluster x100 must reject with exit code 10"
    );
    assert!(stderr.contains("does not support SpacemiT IME"));
}

#[test]
fn test_us04_error_cluster_both_without_split() {
    let fixture = TempModelFixture::new("err_both_no_split");
    let (out, _stdout, stderr) = run_diag(&["--model", fixture.path_str(), "--cluster", "both"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr.contains("--cluster both requires an explicit --split"));
}

#[test]
fn test_us04_error_split_with_single_cluster() {
    let fixture = TempModelFixture::new("err_single_split");
    let (out, _stdout, stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--split",
        "phase",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr.contains("--split requires --cluster both"));
}

#[test]
fn test_us04_error_empty_export_profile_argument() {
    let fixture = TempModelFixture::new("err_empty_profile_path");
    let (out, _stdout, stderr) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--export-profile",
        "",
    ]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "empty argument to --export-profile must exit with code 2"
    );
    assert!(stderr.contains("--export-profile requires a non-empty file path"));
}

#[test]
fn test_us04_s2_export_profile_nested_directory_creation() {
    let fixture = TempModelFixture::new("s2_nested");
    let unique_dir = std::env::temp_dir().join(format!(
        "k3_nested_test_{}_{}",
        std::process::id(),
        fastrand()
    ));
    let export_path = unique_dir.join("sub").join("nested_profile.json");

    assert!(!unique_dir.exists());

    let (out, _stdout, _err) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--export-profile",
        export_path.to_str().unwrap(),
        "--max-tokens",
        "2",
    ]);

    assert!(out.status.success());
    assert!(
        export_path.exists(),
        "nested exported profile JSON must exist"
    );

    let file_content = std::fs::read_to_string(&export_path).expect("read nested file");
    let profile: CostProfile = serde_json::from_str(&file_content).expect("parse nested JSON");
    assert_eq!(profile.model_identity.architecture, "synthetic");

    let _ = std::fs::remove_dir_all(&unique_dir);
}

#[test]
fn test_us04_s2_export_profile_default_cluster() {
    let fixture = TempModelFixture::new("s2_default_cluster");
    let export_path = std::env::temp_dir().join(format!(
        "k3_export_default_cluster_{}_{}.json",
        std::process::id(),
        fastrand()
    ));

    let (out, _stdout, _err) = run_diag(&[
        "--model",
        fixture.path_str(),
        "--export-profile",
        export_path.to_str().unwrap(),
        "--max-tokens",
        "2",
    ]);

    assert!(out.status.success());
    assert!(export_path.exists());

    let file_content = std::fs::read_to_string(&export_path).expect("read file");
    let profile: CostProfile = serde_json::from_str(&file_content).expect("parse JSON");
    assert_eq!(profile.cluster_placement.cluster, "x100");
    assert_eq!(profile.split_strategy, "none");
    assert!(profile.allocated_cores.total >= 1);

    let _ = std::fs::remove_file(&export_path);
}
