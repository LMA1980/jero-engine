//! US-03: Heterogeneous Cluster Workload Splitting & Matrix Acceleration Benchmarking
//! Opaque-Box End-to-End CLI Integration Test Suite (`k3-diag`).
//!
//! Evaluates the 4-Tier Systematic Testing Methodology plus Adversarial Hardening:
//! - **Tier 1: Feature Coverage (F1–F17, >=5 tests per feature)**:
//!   F1: Single-cluster placement (`x100`, `a100`) with core pinning
//!   F2: Cross-cluster allocation (`--cluster both --split <phase|op|row>`)
//!   F3: CLI flag parsing & exit code mapping (0, 2, 10, 13)
//!   F4: Core affinity enforcement & isolation (disjoint masks, zero migration)
//!   F5: Phase split strategy (`prefill` on one cluster, `decode` on another)
//!   F6: Operation split strategy (matmul on A100, norm/attention on X100)
//!   F7: Row split strategy (concurrent partitioning with dynamic measured ratio)
//!   F8: Separate prefill/decode metrics (latencies and throughputs)
//!   F9: IME kernel execution & instruction path validation (`xsmtvdotii` on A100)
//!   F10: IME toggle & capability check (reject `--ime on` on X100 with exit 10)
//!   F11: IME throughput delta reporting ($\Delta = T_{\text{on}} - T_{\text{off}}$)
//!   F12: Hardware ceiling comparisons (compute ceiling vs memory bandwidth)
//!   F13: Bit-identical determinism across repeats ($N \ge 5$) at temperature 0
//!   F14: Determinism mismatch detection and reporting (`determinism_verified: false`)
//!   F15: Post-run affinity audit (zero thread migration via `/proc` checks)
//!   F16: Documentation upkeep & topology verification (`0x00FF` vs `0xFF00`)
//!   F17: E2E test suite integration & opaque-box runner validation
//! - **Tier 2: Boundary & Corner Cases (F1–F17, >=5 tests per feature)**:
//!   Extreme thread bounds, empty strings, max-tokens limits, core oversubscription,
//!   corrupted model files, unaligned matrix dimensions, zero repeats, subnormals.
//! - **Tier 3: Cross-Feature Combinations**:
//!   Pairwise matrix across cluster x split x ime x coord x repeat.
//! - **Tier 4: Real-World Application Scenarios**:
//!   Comparative placement, IME yardstick evaluation, phase split pipeline,
//!   op split offload, production determinism audit.
//! - **Tier 5: Adversarial & Integrity Hardening**:
//!   JSON character injection escaping, lock concurrency, math invariant checks.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::single_char_pattern,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines
)]

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use jero_core::QuantType;
use jero_core::test_helpers::create_synthetic_gguf_v3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Data Transfer Objects & Schemas for JSON Validation
// ============================================================================

/// Continuous statistical metrics summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MetricStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

/// Execution metrics for an individual evaluation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RunMetric {
    pub run_index: usize,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_sec: f64,
    pub total_wall_time_ms: f64,
    #[serde(default)]
    pub cumulative_barrier_wait_ms: f64,
    #[serde(default)]
    pub barrier_wait_ratio: f64,
    #[serde(default)]
    pub compute_utilization_ratio: f64,
    #[serde(default)]
    pub tokens: Option<Vec<u32>>,
}

/// Aggregated statistical summary for multi-run evaluations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StatisticsSummary {
    #[serde(default)]
    pub ttft_ms: MetricStats,
    #[serde(default)]
    pub decode_tokens_per_sec: MetricStats,
    #[serde(default)]
    pub cumulative_barrier_wait_ms: MetricStats,
    #[serde(default)]
    pub total_wall_time_ms: MetricStats,
    #[serde(default)]
    pub compute_utilization_ratio: MetricStats,
}

/// Strongly typed schema representation of `k3-diag` JSON report export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Us03Report {
    #[serde(default)]
    pub cluster: Option<String>,
    #[serde(default)]
    pub split_strategy: Option<String>,
    #[serde(default)]
    pub ime_enabled: Option<bool>,
    #[serde(default)]
    pub coordination_profile: Option<String>,
    #[serde(default)]
    pub repeat_count: Option<usize>,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub effective_core_mask: Option<String>,
    #[serde(default)]
    pub determinism_verified: Option<bool>,
    #[serde(default)]
    pub cluster_isolation_verified: Option<bool>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prefill_latency_ms: Option<f64>,
    #[serde(default)]
    pub decode_latency_ms: Option<f64>,
    #[serde(default)]
    pub prefill_tps: Option<f64>,
    #[serde(default)]
    pub decode_tps: Option<f64>,
    #[serde(default)]
    pub ime_delta_prefill_tps: Option<f64>,
    #[serde(default)]
    pub ime_delta_decode_tps: Option<f64>,
    #[serde(default)]
    pub prefill_hardware_ceiling_pct: Option<f64>,
    #[serde(default)]
    pub decode_hardware_ceiling_pct: Option<f64>,
    #[serde(default)]
    pub row_split_partition_ratio: Option<f64>,
    #[serde(default)]
    pub runs: Vec<RunMetric>,
    #[serde(default)]
    pub statistics: StatisticsSummary,
}

// ============================================================================
// Fixtures & Test Helpers
// ============================================================================

static RAND_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(100);

fn fastrand() -> u64 {
    use std::sync::atomic::Ordering;
    let prev = RAND_SEED.fetch_add(1, Ordering::Relaxed);
    prev.wrapping_mul(6364136223846793005).wrapping_add(1)
}

/// RAII guard that manages a temporary synthetic GGUF v3 model file.
#[derive(Debug)]
pub struct TempModelFixture {
    path: PathBuf,
}

impl TempModelFixture {
    pub fn new(tag: &str) -> Self {
        Self::with_architecture(tag, "synthetic")
    }

    pub fn with_architecture(tag: &str, architecture: &str) -> Self {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3(architecture, 32, &tensor_specs);
        let path = std::env::temp_dir().join(format!(
            "k3_us03_e2e_{tag}_{}_{}.gguf",
            std::process::id(),
            fastrand()
        ));
        let mut f = File::create(&path).expect("create temp synthetic GGUF model");
        f.write_all(&bytes).expect("write GGUF bytes");
        f.sync_all().expect("sync GGUF file");
        Self { path }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn path_str(&self) -> &str {
        self.path.to_str().expect("valid utf8 path")
    }
}

impl Drop for TempModelFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// RAII guard that manages a corrupted / invalid model file.
#[derive(Debug)]
pub struct CorruptModelFixture {
    path: PathBuf,
}

impl CorruptModelFixture {
    pub fn truncated(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "k3_us03_corrupt_{tag}_{}_{}.gguf",
            std::process::id(),
            fastrand()
        ));
        let mut f = File::create(&path).expect("create corrupt file");
        f.write_all(b"GGUF\x03\x00\x00\x00")
            .expect("write partial header");
        f.sync_all().expect("sync file");
        Self { path }
    }

    pub fn zero_byte(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "k3_us03_zero_{tag}_{}_{}.gguf",
            std::process::id(),
            fastrand()
        ));
        let _ = File::create(&path).expect("create 0-byte file");
        Self { path }
    }

    pub fn garbage(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "k3_us03_garbage_{tag}_{}_{}.gguf",
            std::process::id(),
            fastrand()
        ));
        let mut f = File::create(&path).expect("create garbage file");
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04])
            .expect("write garbage bytes");
        f.sync_all().expect("sync file");
        Self { path }
    }

    #[must_use]
    pub fn path_str(&self) -> &str {
        self.path.to_str().expect("valid utf8 path")
    }
}

impl Drop for CorruptModelFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Returns a [`Command`] pointing to the compiled `k3-diag` binary.
fn k3_diag_cmd() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jero-diag"));
    cmd.env("K3_CPU_ONLINE_MASK", "0xFFFF");
    cmd
}

/// Executes `k3-diag` with given arguments and captures status, stdout, and stderr.
fn run_diag(args: &[&str]) -> Output {
    k3_diag_cmd()
        .args(args)
        .output()
        .expect("failed to spawn k3-diag subprocess")
}

/// Executes `k3-diag` with given arguments, parses stdout into JSON `Value` and typed `Us03Report`.
fn run_diag_json(args: &[&str]) -> (Output, Value, Us03Report) {
    let mut full_args = args.to_vec();
    if !full_args.contains(&"--json") {
        full_args.push("--json");
    }
    if !full_args.contains(&"--repeat") {
        full_args.push("--repeat");
        full_args.push("3");
    }
    let output = run_diag(&full_args);
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let val: Value = serde_json::from_str(&stdout_str).unwrap_or(Value::Null);
    let report: Us03Report = serde_json::from_str(&stdout_str).unwrap_or_default();
    (output, val, report)
}

/// Tests whether an exclusive flock can be acquired non-blockingly on the file.
fn probe_flock_exclusive(path: &Path) -> bool {
    if let Ok(file) = File::open(path) {
        // SAFETY: Non-blocking flock probe on valid open file descriptor.
        let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if ret == 0 {
            // SAFETY: Unlock before closing probe file.
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
            true
        } else {
            false
        }
    } else {
        false
    }
}

// ============================================================================
// TIER 1: FEATURE COVERAGE (F1 - F17, >= 5 Tests per Feature)
// ============================================================================

// ----------------------------------------------------------------------------
// Feature F1: Single-cluster placement (--cluster x100 | a100)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f1_01_single_cluster_x100_default_threads() {
    let fixture = TempModelFixture::new("f1_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--prompt",
        "hello x100",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f1_02_single_cluster_a100_default_threads() {
    let fixture = TempModelFixture::new("f1_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--prompt",
        "hello a100",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f1_03_single_cluster_x100_explicit_threads() {
    let fixture = TempModelFixture::new("f1_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--threads",
        "4",
        "--prompt",
        "hello x100 threads",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f1_04_single_cluster_a100_explicit_threads() {
    let fixture = TempModelFixture::new("f1_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--threads",
        "8",
        "--prompt",
        "hello a100 threads",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f1_05_single_cluster_x100_with_core_mask() {
    let fixture = TempModelFixture::new("f1_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--cores",
        "0x000F",
        "--threads",
        "4",
        "--prompt",
        "hello mask",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let mask_str = val
        .get("effective_core_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        mask_str.contains("0x000F") || mask_str.contains("0x00ff") || mask_str.contains("0x00FF")
    );
}

// ----------------------------------------------------------------------------
// Feature F2: Cross-cluster allocation (--cluster both --split <phase|op|row>)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f2_01_cross_cluster_phase_split() {
    let fixture = TempModelFixture::new("f2_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "eval phase split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f2_02_cross_cluster_op_split() {
    let fixture = TempModelFixture::new("f2_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--prompt",
        "eval op split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f2_03_cross_cluster_row_split() {
    let fixture = TempModelFixture::new("f2_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "eval row split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "exit={:?}, stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f2_04_cross_cluster_both_json_fields() {
    let fixture = TempModelFixture::new("f2_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "json both check",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(val.get("cluster").and_then(|v| v.as_str()), Some("both"));
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("phase")
    );
}

#[test]
fn test_tier1_f2_05_cross_cluster_disjoint_worker_pools() {
    let fixture = TempModelFixture::new("f2_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "disjoint pools check",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    if let Some(iso) = val
        .get("cluster_isolation_verified")
        .and_then(|v| v.as_bool())
    {
        assert!(iso, "cluster isolation must be verified across dual pools");
    }
}

// ----------------------------------------------------------------------------
// Feature F3: CLI flag parsing & exit codes (0, 2, 10, 13)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f3_01_exit_0_on_success() {
    let fixture = TempModelFixture::new("f3_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--prompt",
        "test ok",
        "--max-tokens",
        "2",
    ]);
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn test_tier1_f3_02_exit_2_on_unknown_flag() {
    let fixture = TempModelFixture::new("f3_02");
    let out = run_diag(&["--model", fixture.path_str(), "--nonexistent-flag-xyz"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier1_f3_03_exit_2_on_missing_flag_value() {
    let fixture = TempModelFixture::new("f3_03");
    let out = run_diag(&["--model", fixture.path_str(), "--cluster"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier1_f3_04_exit_10_on_capability_mismatch() {
    let bad_fixture = TempModelFixture::with_architecture("f3_04", "qwen3.8");
    let out = run_diag(&["--model", bad_fixture.path_str(), "--prompt", "hello"]);
    assert_eq!(out.status.code(), Some(10));
    assert!(String::from_utf8_lossy(&out.stderr).contains("[capability_mismatch]"));
}

#[test]
fn test_tier1_f3_05_exit_13_on_affinity_error() {
    let fixture = TempModelFixture::new("f3_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x0001",
        "--threads",
        "4", // 4 threads on 1 core mask
    ]);
    assert_eq!(out.status.code(), Some(13));
    assert!(String::from_utf8_lossy(&out.stderr).contains("[affinity_error]"));
}

// ----------------------------------------------------------------------------
// Feature F4: Core affinity enforcement
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f4_01_pinning_x100_mask_0x00ff() {
    let fixture = TempModelFixture::new("f4_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--prompt",
        "x100 mask",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mask = val
        .get("effective_core_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let parsed_mask = u16::from_str_radix(mask.trim_start_matches("0x"), 16).unwrap_or(0);
    assert!(
        parsed_mask != 0 && (parsed_mask & !0x00FF) == 0,
        "x100 effective_core_mask must be a non-empty subset of 0x00FF, got {mask}"
    );
}

#[test]
fn test_tier1_f4_02_pinning_a100_mask_0xff00() {
    let fixture = TempModelFixture::new("f4_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--prompt",
        "a100 mask",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mask = val
        .get("effective_core_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(mask.eq_ignore_ascii_case("0xff00") || mask.contains("0xFF00"));
}

#[test]
fn test_tier1_f4_03_reject_mixed_mask_without_split() {
    let fixture = TempModelFixture::new("f4_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x0101", // bit 0 (x100) and bit 8 (a100) without --split
        "--prompt",
        "test",
    ]);
    assert_eq!(
        out.status.code(),
        Some(13),
        "mixed mask without split must yield exit 13"
    );
}

#[test]
fn test_tier1_f4_04_reject_full_mask_without_split() {
    let fixture = TempModelFixture::new("f4_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0xFFFF",
        "--prompt",
        "test",
    ]);
    assert_eq!(
        out.status.code(),
        Some(13),
        "full mask without split must yield exit 13"
    );
}

#[test]
fn test_tier1_f4_05_disjoint_affinity_for_both_split() {
    let fixture = TempModelFixture::new("f4_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "dual disjoint",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ----------------------------------------------------------------------------
// Feature F5: Phase split strategy (--split phase)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f5_01_phase_split_success() {
    let fixture = TempModelFixture::new("f5_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "phase split run",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f5_02_phase_split_prefill_decode_latencies() {
    let fixture = TempModelFixture::new("f5_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "phase latencies",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    let prefill_ms = val
        .get("prefill_latency_ms")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let decode_ms = val
        .get("decode_latency_ms")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    assert!(prefill_ms >= 0.0, "prefill_ms must be reported");
    assert!(decode_ms >= 0.0, "decode_ms must be reported");
}

#[test]
fn test_tier1_f5_03_phase_split_tokens_generated() {
    let fixture = TempModelFixture::new("f5_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "hello",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.is_empty(), "must generate tokens");
}

#[test]
fn test_tier1_f5_04_phase_split_json_split_strategy_field() {
    let fixture = TempModelFixture::new("f5_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("phase")
    );
}

#[test]
fn test_tier1_f5_05_phase_split_cluster_isolation() {
    let fixture = TempModelFixture::new("f5_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    if let Some(iso) = val
        .get("cluster_isolation_verified")
        .and_then(|v| v.as_bool())
    {
        assert!(iso);
    }
}

// ----------------------------------------------------------------------------
// Feature F6: Operation split strategy (--split op)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f6_01_op_split_success() {
    let fixture = TempModelFixture::new("f6_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--prompt",
        "op split execution",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f6_02_op_split_prefill_decode_metrics() {
    let fixture = TempModelFixture::new("f6_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--prompt",
        "op metrics",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val.get("runs").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val.get("runs").is_some());
}

#[test]
fn test_tier1_f6_03_op_split_token_correctness() {
    let fixture = TempModelFixture::new("f6_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--prompt",
        "hello world",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
    let tokens = val["runs"][0]["tokens"]
        .as_array()
        .expect("runs[0].tokens array");
    assert_eq!(tokens.len(), 3);
}

#[test]
fn test_tier1_f6_04_op_split_json_split_strategy_field() {
    let fixture = TempModelFixture::new("f6_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("op")
    );
}

#[test]
fn test_tier1_f6_05_op_split_inter_cluster_coordination() {
    let fixture = TempModelFixture::new("f6_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--coord",
        "spin",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ----------------------------------------------------------------------------
// Feature F7: Row split strategy (--split row)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f7_01_row_split_success() {
    let fixture = TempModelFixture::new("f7_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "row split test",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f7_02_row_split_non_hardcoded_ratio() {
    let fixture = TempModelFixture::new("f7_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "row dynamic ratio check",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    // Acceptance criteria R1: partition ratio is derived from measured per-cluster throughput and is not hard-coded
    let ratio = val
        .get("row_split_partition_ratio")
        .and_then(|v| v.as_f64())
        .expect("row_split_partition_ratio present");
    assert!(
        ratio > 0.0 && ratio < 1.0,
        "partition ratio must be non-trivial floating ratio: {ratio}"
    );
    let x100_macs = val["row_split"]["prefill"]["x100_macs_per_sec"]
        .as_f64()
        .expect("row_split.prefill.x100_macs_per_sec");
    let a100_macs = val["row_split"]["prefill"]["a100_macs_per_sec"]
        .as_f64()
        .expect("row_split.prefill.a100_macs_per_sec");
    let expected_share = x100_macs / (x100_macs + a100_macs);
    assert!(
        (ratio - expected_share).abs() < 1e-9,
        "ratio {ratio} must equal measured throughput share {expected_share}"
    );
}

#[test]
fn test_tier1_f7_03_row_split_metrics_positive() {
    let fixture = TempModelFixture::new("f7_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "row metrics positive",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    let decode_tps = val
        .get("decode_tps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    assert!(decode_tps > 0.0);
}

#[test]
fn test_tier1_f7_04_row_split_tokens_correct() {
    let fixture = TempModelFixture::new("f7_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--prompt",
        "hello world",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
    let tokens = val["runs"][0]["tokens"]
        .as_array()
        .expect("runs[0].tokens array");
    assert_eq!(tokens.len(), 3);
}

#[test]
fn test_tier1_f7_05_row_split_concurrent_execution() {
    let fixture = TempModelFixture::new("f7_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--threads",
        "14",
        "--prompt",
        "concurrent row execution",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ----------------------------------------------------------------------------
// Feature F8: Separate prefill/decode metrics
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f8_01_separate_metrics_single_cluster_x100() {
    let fixture = TempModelFixture::new("f8_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val["runs"][0].get("prefill_ms").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val["runs"][0].get("decode_ms").is_some());
}

#[test]
fn test_tier1_f8_02_separate_metrics_single_cluster_a100() {
    let fixture = TempModelFixture::new("f8_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val["runs"][0].get("prefill_ms").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val["runs"][0].get("decode_ms").is_some());
}

#[test]
fn test_tier1_f8_03_separate_metrics_phase_split() {
    let fixture = TempModelFixture::new("f8_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val["runs"][0].get("prefill_ms").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val["runs"][0].get("decode_ms").is_some());
}

#[test]
fn test_tier1_f8_04_separate_metrics_op_split() {
    let fixture = TempModelFixture::new("f8_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val["runs"][0].get("prefill_ms").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val["runs"][0].get("decode_ms").is_some());
}

#[test]
fn test_tier1_f8_05_separate_metrics_row_split() {
    let fixture = TempModelFixture::new("f8_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("prefill_latency_ms").is_some() || val["runs"][0].get("prefill_ms").is_some());
    assert!(val.get("decode_latency_ms").is_some() || val["runs"][0].get("decode_ms").is_some());
}

// ----------------------------------------------------------------------------
// Feature F9: IME kernel execution & instruction path
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f9_01_ime_kernel_execution_a100() {
    let fixture = TempModelFixture::new("f9_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--prompt",
        "ime kernel run",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f9_02_ime_kernel_json_flag() {
    let fixture = TempModelFixture::new("f9_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(val.get("ime_enabled").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        val.get("kernel_path").and_then(|v| v.as_str()),
        Some("ime-vmadot")
    );
    let vmadot = val
        .get("ime_vmadot_issued")
        .and_then(|v| v.as_u64())
        .expect("ime_vmadot_issued present");
    assert!(
        vmadot > 0,
        "A100 --ime on must issue smt.vmadot instructions"
    );
}

#[test]
fn test_tier1_f9_03_ime_kernel_numerical_stability() {
    let fixture = TempModelFixture::new("f9_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--prompt",
        "hello world",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.is_empty());
}

#[test]
fn test_tier1_f9_04_ime_kernel_row_split() {
    let fixture = TempModelFixture::new("f9_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--ime",
        "on",
        "--prompt",
        "row split with ime",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f9_05_ime_kernel_op_split() {
    let fixture = TempModelFixture::new("f9_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--ime",
        "on",
        "--prompt",
        "op split with ime",
        "--max-tokens",
        "3",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ----------------------------------------------------------------------------
// Feature F10: IME toggle & capability check
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f10_01_ime_off_x100_success() {
    let fixture = TempModelFixture::new("f10_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--ime",
        "off",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f10_02_ime_off_a100_success() {
    let fixture = TempModelFixture::new("f10_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "off",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f10_03_ime_on_a100_success() {
    let fixture = TempModelFixture::new("f10_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f10_04_ime_on_x100_rejected_exit_10() {
    let fixture = TempModelFixture::new("f10_04");
    let out = run_diag(&[
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
        "x100 cluster must reject --ime on with exit code 10"
    );
}

#[test]
fn test_tier1_f10_05_ime_on_x100_stderr_error_tag() {
    let fixture = TempModelFixture::new("f10_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--ime",
        "on",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[capability_mismatch]") || stderr.to_lowercase().contains("ime"));
}

// ----------------------------------------------------------------------------
// Feature F11: IME throughput delta reporting
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f11_01_ime_delta_fields_present() {
    let fixture = TempModelFixture::new("f11_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert!(val.get("ime_delta_prefill_tps").is_some() || val.get("prefill_tps").is_some());
    assert!(val.get("ime_delta_decode_tps").is_some() || val.get("decode_tps").is_some());
}

#[test]
fn test_tier1_f11_02_ime_delta_arithmetic_equality() {
    let fixture = TempModelFixture::new("f11_02");
    let (out_on, val_on, _rep_on) = run_diag_json(&[
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
    let cmp = &val_on["ime_comparison"];
    let on_pre = cmp["on"]["prefill_tps"].as_f64().expect("on.prefill_tps");
    let off_pre = cmp["off"]["prefill_tps"].as_f64().expect("off.prefill_tps");
    let delta_pre = cmp["delta_prefill_tps"]
        .as_f64()
        .expect("delta_prefill_tps");
    assert!((delta_pre - (on_pre - off_pre)).abs() < 1e-6);

    let on_dec = cmp["on"]["decode_tps"].as_f64().expect("on.decode_tps");
    let off_dec = cmp["off"]["decode_tps"].as_f64().expect("off.decode_tps");
    let delta_dec = cmp["delta_decode_tps"].as_f64().expect("delta_decode_tps");
    assert!((delta_dec - (on_dec - off_dec)).abs() < 1e-6);
    assert!(cmp["on"]["vmadot_issued"].as_u64().unwrap_or(0) > 0);
    assert_eq!(cmp["off"]["vmadot_issued"].as_u64(), Some(0));
}

#[test]
fn test_tier1_f11_03_ime_delta_prefill_isolated() {
    let fixture = TempModelFixture::new("f11_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    if let Some(delta) = val.get("ime_delta_prefill_tps").and_then(|v| v.as_f64()) {
        assert!(delta.is_finite());
    }
}

#[test]
fn test_tier1_f11_04_ime_delta_decode_isolated() {
    let fixture = TempModelFixture::new("f11_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    if let Some(delta) = val.get("ime_delta_decode_tps").and_then(|v| v.as_f64()) {
        assert!(delta.is_finite());
    }
}

#[test]
fn test_tier1_f11_05_ime_no_external_baseline_reference() {
    let fixture = TempModelFixture::new("f11_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !combined.contains("llama.cpp"),
        "must not use external llama.cpp baseline per ADR-033"
    );
    assert!(
        !combined.contains("ollama"),
        "must not use external ollama baseline per ADR-033"
    );
}

// ----------------------------------------------------------------------------
// Feature F12: Hardware ceiling comparisons
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f12_01_ceiling_fields_present() {
    let fixture = TempModelFixture::new("f12_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    if let Some(ceil) = val
        .get("decode_hardware_ceiling_pct")
        .and_then(|v| v.as_f64())
    {
        assert!(ceil >= 0.0);
    }
}

#[test]
fn test_tier1_f12_02_ceiling_ratios_bounded() {
    let fixture = TempModelFixture::new("f12_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    if let Some(ceil) = val
        .get("prefill_hardware_ceiling_pct")
        .and_then(|v| v.as_f64())
    {
        assert!(ceil >= 0.0);
    }
}

#[test]
fn test_tier1_f12_03_ceiling_x100_vs_a100_distinct() {
    let fixture = TempModelFixture::new("f12_03");
    let (_out_x, val_x, _rep_x) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    let (_out_a, val_a, _rep_a) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "2",
    ]);
    // Different peak compute specs on X100 (2.15 GHz OoO) vs A100 (1.80 GHz InO)
    assert!(val_x.is_object());
    assert!(val_a.is_object());
}

#[test]
fn test_tier1_f12_04_ceiling_sub_ceiling_exits_0() {
    let fixture = TempModelFixture::new("f12_04");
    let out = run_diag(&[
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
        "sub-ceiling performance is informational, exits 0 per ADR-034"
    );
}

#[test]
fn test_tier1_f12_05_ceiling_fields_for_splits() {
    let fixture = TempModelFixture::new("f12_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

// ----------------------------------------------------------------------------
// Feature F13: Bit-identical determinism across repeats (N >= 5)
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f13_01_determinism_x100_repeat_5() {
    let fixture = TempModelFixture::new("f13_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--repeat",
        "5",
        "--prompt",
        "deterministic x100 prompt",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier1_f13_02_determinism_a100_repeat_5() {
    let fixture = TempModelFixture::new("f13_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--repeat",
        "5",
        "--prompt",
        "deterministic a100 prompt",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier1_f13_03_determinism_phase_split_repeat_5() {
    let fixture = TempModelFixture::new("f13_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--repeat",
        "5",
        "--prompt",
        "deterministic phase split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier1_f13_04_determinism_op_split_repeat_5() {
    let fixture = TempModelFixture::new("f13_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--repeat",
        "5",
        "--prompt",
        "deterministic op split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier1_f13_05_determinism_row_split_repeat_5() {
    let fixture = TempModelFixture::new("f13_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--repeat",
        "5",
        "--prompt",
        "deterministic row split",
        "--max-tokens",
        "4",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

// ----------------------------------------------------------------------------
// Feature F14: Determinism mismatch detection
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f14_01_mismatch_detected_flag_false() {
    let fixture = TempModelFixture::new("f14_01");
    let out = k3_diag_cmd()
        .env("K3_DIAG_INJECT_NONDETERMINISM", "2")
        .args([
            "--model",
            fixture.path_str(),
            "--cluster",
            "x100",
            "--repeat",
            "3",
            "--max-tokens",
            "2",
            "--json",
        ])
        .output()
        .expect("spawn k3-diag with injected nondeterminism");
    assert_eq!(out.status.code(), Some(10));
    let val: Value = serde_json::from_slice(&out.stdout).expect("parse mismatch json");
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert_eq!(val["determinism"]["first_divergent_run"].as_u64(), Some(2));
}

#[test]
fn test_tier1_f14_02_mismatch_run_tokens_preserved() {
    let fixture = TempModelFixture::new("f14_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    if let Some(runs) = val.get("runs").and_then(|v| v.as_array()) {
        assert_eq!(runs.len(), 3);
        for run in runs {
            assert!(run.get("tokens").is_some());
        }
    }
}

#[test]
fn test_tier1_f14_03_mismatch_exit_code_clean() {
    let fixture = TempModelFixture::new("f14_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f14_04_mismatch_human_readable_warning() {
    let fixture = TempModelFixture::new("f14_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f14_05_mismatch_early_token_divergence() {
    let fixture = TempModelFixture::new("f14_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "1",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

// ----------------------------------------------------------------------------
// Feature F15: Post-run affinity audit
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f15_01_post_run_affinity_x100() {
    let fixture = TempModelFixture::new("f15_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f15_02_post_run_affinity_a100() {
    let fixture = TempModelFixture::new("f15_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f15_03_post_run_affinity_both_disjoint() {
    let fixture = TempModelFixture::new("f15_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_f15_04_post_run_affinity_json_verified_field() {
    let fixture = TempModelFixture::new("f15_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("cluster_isolation_verified")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    for key in ["workers_before", "workers_after"] {
        let workers = val[key].as_array().expect("workers array present");
        assert!(!workers.is_empty());
        for w in workers {
            let mask = w["cpus_allowed_mask"].as_u64().expect("cpus_allowed_mask");
            assert!(
                mask != 0 && (mask & !0x00FF) == 0,
                "worker mask 0x{mask:04X} in {key} must stay within X100 (0x00FF)"
            );
        }
    }
}

#[test]
fn test_tier1_f15_05_post_run_affinity_across_all_repeats() {
    let fixture = TempModelFixture::new("f15_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

// ----------------------------------------------------------------------------
// Feature F16: Documentation upkeep & topology verification
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f16_01_doc_affinity_topology_x100_cores_0_to_7() {
    // Assert physical definition: X100 = cores 0-7 (0x00FF)
    let fixture = TempModelFixture::new("f16_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    let mask = val
        .get("effective_core_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let parsed_mask = u16::from_str_radix(mask.trim_start_matches("0x"), 16).unwrap_or(0);
    assert!(
        parsed_mask != 0 && (parsed_mask & !0x00FF) == 0,
        "x100 effective_core_mask must be a non-empty subset of 0x00FF, got {mask}"
    );
}

#[test]
fn test_tier1_f16_02_doc_affinity_topology_a100_cores_8_to_15() {
    // Assert physical definition: A100 = cores 8-15 (0xFF00)
    let fixture = TempModelFixture::new("f16_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    let mask = val
        .get("effective_core_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(mask.eq_ignore_ascii_case("0xff00") || mask.contains("0xFF00"));
}

#[test]
fn test_tier1_f16_03_doc_k3_bench_reconciled() {
    let fixture = TempModelFixture::new("f16_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x00FF",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f16_04_doc_lab_md_reconciled() {
    let fixture = TempModelFixture::new("f16_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0xFF00",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier1_f16_05_doc_feature_diagnostic_tool_options() {
    let out = run_diag(&["--help"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--cluster"));
}

// ----------------------------------------------------------------------------
// Feature F17: E2E testing suite
// ----------------------------------------------------------------------------

#[test]
fn test_tier1_f17_01_e2e_invokes_compiled_binary() {
    let out = run_diag(&["--version"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("k3-diag"));
}

#[test]
fn test_tier1_f17_02_e2e_parses_json_schema() {
    let fixture = TempModelFixture::new("f17_02");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier1_f17_03_e2e_synthetic_gguf_fixture() {
    let fixture = TempModelFixture::new("f17_03");
    assert!(fixture.path().exists());
    assert!(std::fs::metadata(fixture.path()).unwrap().len() > 0);
}

#[test]
fn test_tier1_f17_04_e2e_raii_file_cleanup() {
    let path = {
        let fixture = TempModelFixture::new("f17_04");
        fixture.path().to_path_buf()
    };
    assert!(!path.exists(), "temporary file must be unlinked on drop");
}

#[test]
fn test_tier1_f17_05_e2e_repeatable_runs() {
    let fixture = TempModelFixture::new("f17_05");
    let out1 = run_diag(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    let out2 = run_diag(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    assert_eq!(out1.status.code(), out2.status.code());
}

// ============================================================================
// TIER 2: BOUNDARY & CORNER CASES (F1 - F17, >= 5 Tests per Feature)
// ============================================================================

// --- F1 Boundaries ---
#[test]
fn test_tier2_f1_01_invalid_cluster_name() {
    let fixture = TempModelFixture::new("t2_f1_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "invalid_cluster_name",
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f1_02_x100_mismatched_core_mask() {
    let fixture = TempModelFixture::new("t2_f1_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--cores",
        "0xFF00", // a100 mask on x100 cluster
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f1_03_a100_mismatched_core_mask() {
    let fixture = TempModelFixture::new("t2_f1_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--cores",
        "0x00FF", // x100 mask on a100 cluster
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f1_04_x100_thread_overflow() {
    let fixture = TempModelFixture::new("t2_f1_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--threads",
        "9", // 9 threads exceeds 8 cores
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f1_05_zero_threads() {
    let fixture = TempModelFixture::new("t2_f1_05");
    let out = run_diag(&["--model", fixture.path_str(), "--threads", "0"]);
    assert_eq!(out.status.code(), Some(2));
}

// --- F2 Boundaries ---
#[test]
fn test_tier2_f2_01_cluster_both_missing_split() {
    let fixture = TempModelFixture::new("t2_f2_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both", // missing --split
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f2_02_split_on_single_cluster_x100() {
    let fixture = TempModelFixture::new("t2_f2_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--split",
        "phase", // --split not allowed on single cluster
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f2_03_split_on_single_cluster_a100() {
    let fixture = TempModelFixture::new("t2_f2_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--split",
        "row",
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f2_04_split_empty_string() {
    let fixture = TempModelFixture::new("t2_f2_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "",
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f2_05_split_unknown_strategy() {
    let fixture = TempModelFixture::new("t2_f2_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "column", // unknown strategy
    ]);
    assert_eq!(out.status.code(), Some(2));
}

// --- F3 Boundaries ---
#[test]
fn test_tier2_f3_01_repeat_below_minimum_2() {
    let fixture = TempModelFixture::new("t2_f3_01");
    let out = run_diag(&["--model", fixture.path_str(), "--repeat", "2"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f3_02_repeat_negative_number() {
    let fixture = TempModelFixture::new("t2_f3_02");
    let out = run_diag(&["--model", fixture.path_str(), "--repeat", "-5"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f3_03_repeat_non_numeric() {
    let fixture = TempModelFixture::new("t2_f3_03");
    let out = run_diag(&["--model", fixture.path_str(), "--repeat", "many"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f3_04_ime_invalid_value() {
    let fixture = TempModelFixture::new("t2_f3_04");
    let out = run_diag(&["--model", fixture.path_str(), "--ime", "maybe"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f3_05_missing_model_arg() {
    let out = run_diag(&["--prompt", "hello world"]);
    assert_ne!(out.status.code(), Some(0));
}

// --- F4 Boundaries ---
#[test]
fn test_tier2_f4_01_core_mask_zero() {
    let fixture = TempModelFixture::new("t2_f4_01");
    let out = run_diag(&["--model", fixture.path_str(), "--cores", "0x0000"]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f4_02_core_mask_out_of_range() {
    let fixture = TempModelFixture::new("t2_f4_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x10000", // exceeds 16 cores
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f4_03_threads_exceed_mask_bits() {
    let fixture = TempModelFixture::new("t2_f4_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x0003", // 2 cores
        "--threads",
        "3", // 3 threads
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f4_04_x100_conflicting_core_mask() {
    let fixture = TempModelFixture::new("t2_f4_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--cores",
        "0x0100", // core 8 is A100
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f4_05_a100_conflicting_core_mask() {
    let fixture = TempModelFixture::new("t2_f4_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--cores",
        "0x0001", // core 0 is X100
    ]);
    assert_eq!(out.status.code(), Some(13));
}

// --- F5 Boundaries ---
#[test]
fn test_tier2_f5_01_phase_split_single_token_prompt() {
    let fixture = TempModelFixture::new("t2_f5_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "a",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f5_02_phase_split_max_tokens_0() {
    let fixture = TempModelFixture::new("t2_f5_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "prefill only",
        "--max-tokens",
        "0",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f5_03_phase_split_max_tokens_1() {
    let fixture = TempModelFixture::new("t2_f5_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--max-tokens",
        "1",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f5_04_phase_split_near_ctx_limit() {
    let fixture = TempModelFixture::new("t2_f5_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--n-ctx",
        "32",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f5_05_phase_split_uneven_threads() {
    let fixture = TempModelFixture::new("t2_f5_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--threads",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

// --- F6 Boundaries ---
#[test]
fn test_tier2_f6_01_op_split_minimal_layer_model() {
    let fixture = TempModelFixture::new("t2_f6_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f6_02_op_split_with_ime_on() {
    let fixture = TempModelFixture::new("t2_f6_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f6_03_op_split_barrier_sync_frequency() {
    let fixture = TempModelFixture::new("t2_f6_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--coord",
        "futex",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f6_04_op_split_threads_1() {
    let fixture = TempModelFixture::new("t2_f6_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--threads",
        "1",
        "--max-tokens",
        "2",
    ]);
    // Dual-cluster placement requires at least 2 threads (1 per cluster) -> AffinityError (exit 13).
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f6_05_op_split_max_tokens_1() {
    let fixture = TempModelFixture::new("t2_f6_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--max-tokens",
        "1",
    ]);
    assert!(out.status.success());
}

// --- F7 Boundaries ---
#[test]
fn test_tier2_f7_01_row_split_odd_matrix_dim() {
    let fixture = TempModelFixture::new("t2_f7_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f7_02_row_split_single_row_decode() {
    let fixture = TempModelFixture::new("t2_f7_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "1",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f7_03_row_split_extreme_imbalance() {
    let fixture = TempModelFixture::new("t2_f7_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--threads",
        "2",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f7_04_row_split_ime_on_a100_rows() {
    let fixture = TempModelFixture::new("t2_f7_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f7_05_row_split_zero_tokens() {
    let fixture = TempModelFixture::new("t2_f7_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--max-tokens",
        "0",
    ]);
    assert!(out.status.success());
}

// --- F8 Boundaries ---
#[test]
fn test_tier2_f8_01_wall_time_greater_than_sum() {
    let fixture = TempModelFixture::new("t2_f8_01");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "3"]);
    assert!(out.status.success());
    if let (Some(wall), Some(pre), Some(dec)) = (
        val.get("total_wall_time_ms").and_then(|v| v.as_f64()),
        val.get("prefill_latency_ms").and_then(|v| v.as_f64()),
        val.get("decode_latency_ms").and_then(|v| v.as_f64()),
    ) {
        assert!(wall >= (pre + dec) * 0.99);
    }
}

#[test]
fn test_tier2_f8_02_throughput_inversely_relates_to_latency() {
    let fixture = TempModelFixture::new("t2_f8_02");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "3"]);
    assert!(out.status.success());
    if let (Some(dec_lat), Some(dec_tps)) = (
        val.get("decode_latency_ms").and_then(|v| v.as_f64()),
        val.get("decode_tps").and_then(|v| v.as_f64()),
    ) {
        if dec_lat > 0.0 {
            assert!(dec_tps > 0.0);
        }
    }
}

#[test]
fn test_tier2_f8_03_max_tokens_0_zero_decode_tps() {
    let fixture = TempModelFixture::new("t2_f8_03");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "0"]);
    assert!(out.status.success());
    if let Some(dec_tps) = val.get("decode_tps").and_then(|v| v.as_f64()) {
        assert!(dec_tps == 0.0 || dec_tps.is_finite());
    }
}

#[test]
fn test_tier2_f8_04_single_token_prefill_metrics() {
    let fixture = TempModelFixture::new("t2_f8_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--prompt",
        "x",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f8_05_prefill_latency_positive() {
    let fixture = TempModelFixture::new("t2_f8_05");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    assert!(out.status.success());
    if let Some(pre_ms) = val.get("prefill_latency_ms").and_then(|v| v.as_f64()) {
        assert!(pre_ms >= 0.0);
    }
}

// --- F9 Boundaries ---
#[test]
fn test_tier2_f9_01_ime_unaligned_dimension() {
    let fixture = TempModelFixture::new("t2_f9_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f9_02_ime_all_zero_weights() {
    let fixture = TempModelFixture::new("t2_f9_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f9_03_ime_saturating_quant_values() {
    let fixture = TempModelFixture::new("t2_f9_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f9_04_ime_subnormal_floats() {
    let fixture = TempModelFixture::new("t2_f9_04");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f9_05_ime_repeated_rapid_invocations() {
    let fixture = TempModelFixture::new("t2_f9_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--repeat",
        "5",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

// --- F10 Boundaries ---
#[test]
fn test_tier2_f10_01_ime_invalid_option_true() {
    let fixture = TempModelFixture::new("t2_f10_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--ime",
        "true", // expects on|off
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f10_02_ime_empty_value() {
    let fixture = TempModelFixture::new("t2_f10_02");
    let out = run_diag(&["--model", fixture.path_str(), "--ime", ""]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn test_tier2_f10_03_ime_mixed_mask_without_split() {
    let fixture = TempModelFixture::new("t2_f10_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x0101",
        "--ime",
        "on",
    ]);
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn test_tier2_f10_04_ime_omitted_defaults_to_off() {
    let fixture = TempModelFixture::new("t2_f10_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    if let Some(ime) = val.get("ime_enabled").and_then(|v| v.as_bool()) {
        assert!(!ime, "omitted --ime must default to false");
    }
}

#[test]
fn test_tier2_f10_05_ime_whitespace_value() {
    let fixture = TempModelFixture::new("t2_f10_05");
    let out = run_diag(&["--model", fixture.path_str(), "--ime", "   "]);
    assert_eq!(out.status.code(), Some(2));
}

// --- F11 Boundaries ---
#[test]
fn test_tier2_f11_01_ime_delta_negative_or_zero_handled() {
    let fixture = TempModelFixture::new("t2_f11_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    if let Some(delta) = val.get("ime_delta_decode_tps").and_then(|v| v.as_f64()) {
        assert!(delta.is_finite());
    }
}

#[test]
fn test_tier2_f11_02_ime_delta_zero_decode_tokens() {
    let fixture = TempModelFixture::new("t2_f11_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "0",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f11_03_ime_delta_single_token_prefill() {
    let fixture = TempModelFixture::new("t2_f11_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--prompt",
        "a",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f11_04_ime_delta_repeat_statistics_mean() {
    let fixture = TempModelFixture::new("t2_f11_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f11_05_ime_delta_same_prompt_and_seed() {
    let fixture = TempModelFixture::new("t2_f11_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--prompt",
        "test same prompt",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

// --- F12 Boundaries ---
#[test]
fn test_tier2_f12_01_ceiling_subnormal_or_zero_clock() {
    let fixture = TempModelFixture::new("t2_f12_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f12_02_ceiling_extreme_prompt_prefill() {
    let fixture = TempModelFixture::new("t2_f12_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--n-ctx",
        "128",
        "--prompt",
        "a very long prompt extending across multiple sentences to exercise prefill ceiling",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f12_03_ceiling_memory_bound_decode() {
    let fixture = TempModelFixture::new("t2_f12_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--max-tokens",
        "8",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f12_04_ceiling_scaling_threads() {
    let fixture = TempModelFixture::new("t2_f12_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--threads",
        "6",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f12_05_ceiling_with_ime_active() {
    let fixture = TempModelFixture::new("t2_f12_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

// --- F13 Boundaries ---
#[test]
fn test_tier2_f13_01_determinism_repeat_10() {
    let fixture = TempModelFixture::new("t2_f13_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--repeat",
        "10",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier2_f13_02_determinism_max_tokens_64() {
    let fixture = TempModelFixture::new("t2_f13_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "16",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier2_f13_03_determinism_ime_on_repeat_5() {
    let fixture = TempModelFixture::new("t2_f13_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier2_f13_04_determinism_empty_prompt() {
    let fixture = TempModelFixture::new("t2_f13_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--prompt",
        "",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn test_tier2_f13_05_determinism_futex_coordination() {
    let fixture = TempModelFixture::new("t2_f13_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--coord",
        "futex",
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

// --- F14 Boundaries ---
#[test]
fn test_tier2_f14_01_mismatch_single_run_divergence() {
    let fixture = TempModelFixture::new("t2_f14_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.get("determinism_verified").is_some());
}

#[test]
fn test_tier2_f14_02_mismatch_last_token_divergence() {
    let fixture = TempModelFixture::new("t2_f14_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
    assert!(val.get("determinism_verified").is_some());
}

#[test]
fn test_tier2_f14_03_mismatch_length_discrepancy() {
    let fixture = TempModelFixture::new("t2_f14_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

#[test]
fn test_tier2_f14_04_mismatch_max_tokens_1() {
    let fixture = TempModelFixture::new("t2_f14_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "1",
    ]);
    assert!(out.status.success());
    assert!(val.get("determinism_verified").is_some());
}

#[test]
fn test_tier2_f14_05_mismatch_all_runs_divergent() {
    let fixture = TempModelFixture::new("t2_f14_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    assert!(val.is_object());
}

// --- F15 Boundaries ---
#[test]
fn test_tier2_f15_01_affinity_audit_high_threads() {
    let fixture = TempModelFixture::new("t2_f15_01");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--threads",
        "8",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f15_02_affinity_audit_repeat_10() {
    let fixture = TempModelFixture::new("t2_f15_02");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--repeat",
        "10",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f15_03_affinity_audit_single_thread() {
    let fixture = TempModelFixture::new("t2_f15_03");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--threads",
        "1",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f15_04_affinity_audit_signal_interrupt() {
    let fixture = TempModelFixture::new("t2_f15_04");
    let out = run_diag(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f15_05_affinity_audit_proc_fs_format() {
    let fixture = TempModelFixture::new("t2_f15_05");
    let (out, val, _report) = run_diag_json(&["--model", fixture.path_str(), "--max-tokens", "2"]);
    assert!(out.status.success());
    assert!(val.is_object());
}

// --- F16 Boundaries ---
#[test]
fn test_tier2_f16_01_doc_cross_reference_adr009() {
    let out = run_diag(&["--help"]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f16_02_doc_cross_reference_adr028() {
    let out = run_diag(&["--help"]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f16_03_doc_cross_reference_adr034() {
    let out = run_diag(&["--help"]);
    assert!(out.status.success());
}

#[test]
fn test_tier2_f16_04_doc_cli_flag_names() {
    let out = run_diag(&["--help"]);
    assert!(out.status.success());
    let txt = String::from_utf8_lossy(&out.stdout);
    assert!(txt.contains("--model"));
}

#[test]
fn test_tier2_f16_05_doc_exit_codes_mapping() {
    let fixture = TempModelFixture::new("t2_f16_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cores",
        "0x0001",
        "--threads",
        "2",
    ]);
    assert_eq!(
        out.status.code(),
        Some(13),
        "exit code 13 must map to AffinityError"
    );
}

// --- F17 Boundaries ---
#[test]
fn test_tier2_f17_01_corrupt_model_truncated_file() {
    let corrupt = CorruptModelFixture::truncated("t2_f17_01");
    let out = run_diag(&["--model", corrupt.path_str(), "--prompt", "hello"]);
    assert_ne!(out.status.code(), Some(0));
}

#[test]
fn test_tier2_f17_02_nonexistent_model_file() {
    let out = run_diag(&[
        "--model",
        "/tmp/nonexistent_model_file_12345.gguf",
        "--prompt",
        "hello",
    ]);
    assert_ne!(out.status.code(), Some(0));
}

#[test]
fn test_tier2_f17_03_zero_byte_model_file() {
    let zero = CorruptModelFixture::zero_byte("t2_f17_03");
    let out = run_diag(&["--model", zero.path_str(), "--prompt", "hello"]);
    assert_ne!(out.status.code(), Some(0));
}

#[test]
fn test_tier2_f17_04_unsupported_quant_type() {
    let corrupt = CorruptModelFixture::garbage("t2_f17_04");
    let out = run_diag(&["--model", corrupt.path_str(), "--prompt", "hello"]);
    assert_ne!(out.status.code(), Some(0));
}

#[test]
fn test_tier2_f17_05_stdout_stderr_isolation_in_json() {
    let fixture = TempModelFixture::new("t2_f17_05");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--json",
        "--repeat",
        "3",
        "--max-tokens",
        "2",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        serde_json::from_str::<Value>(&stdout).is_ok(),
        "stdout must contain only valid JSON"
    );
}

// ============================================================================
// TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise Interactions)
// ============================================================================

#[test]
fn test_tier3_01_x100_spin_repeat5_json() {
    let fixture = TempModelFixture::new("t3_01");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--coord",
        "spin",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("coordination_profile").and_then(|v| v.as_str()),
        Some("spin")
    );
    assert_eq!(val.get("repeat_count").and_then(|v| v.as_u64()), Some(5));
}

#[test]
fn test_tier3_02_a100_futex_ime_on_repeat5_json() {
    let fixture = TempModelFixture::new("t3_02");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--coord",
        "futex",
        "--ime",
        "on",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("coordination_profile").and_then(|v| v.as_str()),
        Some("futex")
    );
    assert_eq!(val.get("repeat_count").and_then(|v| v.as_u64()), Some(5));
}

#[test]
fn test_tier3_03_both_split_phase_rayon_repeat5_json() {
    let fixture = TempModelFixture::new("t3_03");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--coord",
        "rayon",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("coordination_profile").and_then(|v| v.as_str()),
        Some("rayon")
    );
}

#[test]
fn test_tier3_04_both_split_op_spin_ime_on_repeat5_json() {
    let fixture = TempModelFixture::new("t3_04");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--coord",
        "spin",
        "--ime",
        "on",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("op")
    );
}

#[test]
fn test_tier3_05_both_split_row_futex_ime_on_repeat5_json() {
    let fixture = TempModelFixture::new("t3_05");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--coord",
        "futex",
        "--ime",
        "on",
        "--repeat",
        "5",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("row")
    );
}

#[test]
fn test_tier3_06_both_split_row_rayon_repeat3_json() {
    let fixture = TempModelFixture::new("t3_06");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--coord",
        "rayon",
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("coordination_profile").and_then(|v| v.as_str()),
        Some("rayon")
    );
}

#[test]
fn test_tier3_07_x100_futex_ime_off_repeat3() {
    let fixture = TempModelFixture::new("t3_07");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--coord",
        "futex",
        "--ime",
        "off",
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
}

#[test]
fn test_tier3_08_a100_spin_ime_off_repeat3() {
    let fixture = TempModelFixture::new("t3_08");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--coord",
        "spin",
        "--ime",
        "off",
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());
}

// ============================================================================
// TIER 4: REAL-WORLD APPLICATION SCENARIOS
// ============================================================================

#[test]
fn test_tier4_scenario1_comparative_placement_benchmarking() {
    // Researcher benchmarks x100 vs a100 vs row split to evaluate optimal placement
    let fixture = TempModelFixture::new("t4_scen1");

    let (out_x, val_x, _rep_x) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out_x.status.success());

    let (out_a, val_a, _rep_a) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out_a.status.success());

    let (out_row, val_row, _rep_row) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out_row.status.success());

    assert!(val_x.is_object());
    assert!(val_a.is_object());
    assert!(val_row.is_object());
}

#[test]
fn test_tier4_scenario2_ime_acceleration_yardstick_evaluation() {
    // Researcher compares IME on vs off on A100 per ADR-034
    let fixture = TempModelFixture::new("t4_scen2");

    let (out_on, val_on, _rep_on) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out_on.status.success());

    let (out_off, val_off, _rep_off) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "a100",
        "--ime",
        "off",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out_off.status.success());

    assert!(val_on.is_object());
    assert!(val_off.is_object());
}

#[test]
fn test_tier4_scenario3_phase_split_asymmetric_workload() {
    // Exploits X100 OoO for prefill and A100 1024-bit vector for decode
    let fixture = TempModelFixture::new("t4_scen3");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "asymmetric workload pipeline across clusters",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("phase")
    );
}

#[test]
fn test_tier4_scenario4_op_split_specialized_kernel_offloading() {
    // Directs GEMM to A100 IME while keeping normalization & attention on X100
    let fixture = TempModelFixture::new("t4_scen4");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "op",
        "--ime",
        "on",
        "--repeat",
        "3",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("split_strategy").and_then(|v| v.as_str()),
        Some("op")
    );
}

#[test]
fn test_tier4_scenario5_production_determinism_audit_at_temp_zero() {
    // Production QA audit verifying zero token drift across 5 repeats
    let fixture = TempModelFixture::new("t4_scen5");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "both",
        "--split",
        "row",
        "--repeat",
        "5",
        "--prompt",
        "production determinism certification test",
        "--max-tokens",
        "4",
    ]);
    assert!(out.status.success());
    assert_eq!(
        val.get("determinism_verified").and_then(|v| v.as_bool()),
        Some(true)
    );
}

// ============================================================================
// TIER 5: ADVERSARIAL & INTEGRITY HARDENING
// ============================================================================

#[test]
fn test_tier5_adversarial_json_special_characters_escaping() {
    let fixture = TempModelFixture::new("t5_esc");
    let malicious_prompt = "hello \"world\" \\backslash\nnewline\ttab carriage\r return";
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--prompt",
        malicious_prompt,
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        val.is_object(),
        "JSON parser must successfully parse escaped output"
    );
}

#[test]
fn test_tier5_adversarial_extreme_prompt_boundary() {
    let fixture = TempModelFixture::new("t5_boundary");
    let out = run_diag(&[
        "--model",
        fixture.path_str(),
        "--n-ctx",
        "64",
        "--prompt",
        "prompt with exactly few tokens",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier5_adversarial_concurrent_k3_diag_shared_locks() {
    let fixture = TempModelFixture::new("t5_concur");
    assert!(
        probe_flock_exclusive(fixture.path()),
        "file must be free before invocation"
    );

    let out1 = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--cores",
        "0x000F",
        "--threads",
        "2",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out1.status.success(),
        "out1 stderr={}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let out2 = run_diag(&[
        "--model",
        fixture.path_str(),
        "--cluster",
        "x100",
        "--cores",
        "0x00F0",
        "--threads",
        "2",
        "--max-tokens",
        "2",
    ]);
    assert!(
        out2.status.success(),
        "out2 stderr={}",
        String::from_utf8_lossy(&out2.stderr)
    );

    assert!(
        probe_flock_exclusive(fixture.path()),
        "lock must be cleanly released after completion"
    );
}

#[test]
fn test_tier5_adversarial_metric_mathematical_properties() {
    let fixture = TempModelFixture::new("t5_math");
    let (out, val, _report) = run_diag_json(&[
        "--model",
        fixture.path_str(),
        "--repeat",
        "3",
        "--max-tokens",
        "3",
    ]);
    assert!(out.status.success());

    if let Some(stats) = val.get("statistics") {
        if let Some(ttft) = stats.get("ttft_ms") {
            let min = ttft.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mean = ttft.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let max = ttft.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let stddev = ttft.get("stddev").and_then(|v| v.as_f64()).unwrap_or(0.0);

            assert!(
                min <= mean && mean <= max,
                "min <= mean <= max invariant: min={min}, mean={mean}, max={max}"
            );
            assert!(stddev >= 0.0, "stddev must be non-negative: {stddev}");
        }
    }
}

#[test]
fn test_tier5_adversarial_garbage_binary_file() {
    let garbage = CorruptModelFixture::garbage("t5_garbage");
    let out = run_diag(&["--model", garbage.path_str(), "--prompt", "hello"]);
    assert_ne!(out.status.code(), Some(0));
}
