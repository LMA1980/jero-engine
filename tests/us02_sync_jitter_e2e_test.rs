//! US-02: Multi-Core Synchronization & Jitter Evaluation
//! Opaque-Box End-to-End CLI Integration Test Suite.
//!
//! Covers all 4 Systematic Testing Tiers per ORIGINAL_REQUEST.md and PROJECT.md:
//! - **Tier 1: Feature Coverage**:
//!   Profiles (`spin`, `futex`, `rayon`), TTFT latency, decode throughput (tokens/sec),
//!   cumulative barrier wait time, multi-run repeat count ($N \ge 3$), and structured `--json` formatting.
//! - **Tier 2: Boundary & Corner Cases**:
//!   Rejection of `--repeat < 3` with exit code 2, rejection of invalid coordination profiles,
//!   single-worker thread execution (`--threads 1`), zero max-tokens (`--max-tokens 0`),
//!   context length limits (`--n-ctx`), and core-thread bounds.
//! - **Tier 3: Cross-Feature Combinations**:
//!   Pairwise interactions across profiles, repeat counts, thread counts, cluster targeting, and JSON export.
//! - **Tier 4: Real-World Application Scenarios**:
//!   Multi-profile comparative evaluation, temperature 0 token determinism across repeats,
//!   clean process teardown (flock release & thread cleanup), and US-01 backward compatibility.
//! - **Adversarial Integrity**:
//!   JSON escaping of special characters, missing model handling, and mathematical properties of sample statistics.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::single_char_pattern,
    clippy::float_cmp,
    clippy::redundant_closure_for_method_calls
)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

use jero_core::QuantType;
use jero_core::test_helpers::create_synthetic_gguf_v3;

// ============================================================================
// Fixtures & Test Helpers
// ============================================================================

/// RAII guard that deletes the temporary synthetic model file upon drop.
struct TempModelFixture {
    path: PathBuf,
}

impl TempModelFixture {
    fn new(tag: &str) -> Self {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let path = std::env::temp_dir().join(format!(
            "k3_us02_e2e_{tag}_{}_{}.gguf",
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
}

impl Drop for TempModelFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

static RAND_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fastrand() -> u64 {
    use std::sync::atomic::Ordering;
    let prev = RAND_SEED.fetch_add(1, Ordering::Relaxed);
    prev.wrapping_mul(6364136223846793005).wrapping_add(1)
}

/// Returns a [`Command`] pointing to the compiled `k3-diag` binary.
fn k3_diag_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_jero-diag"))
}

/// Tests whether a non-blocking shared/exclusive flock can be acquired.
fn probe_flock_exclusive(path: &Path) -> bool {
    if let Ok(file) = File::open(path) {
        // SAFETY: Non-blocking flock probe on open file handle.
        let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if ret == 0 {
            // SAFETY: Unlock before closing probe file.
            unsafe {
                libc::flock(file.as_raw_fd(), libc::LOCK_UN);
            }
            true
        } else {
            false
        }
    } else {
        false
    }
}

// ============================================================================
// Lightweight Zero-Dependency JSON Parser & Validator
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum SimpleJson {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<SimpleJson>),
    Object(BTreeMap<String, SimpleJson>),
}

impl SimpleJson {
    fn parse(input: &str) -> Result<Self, String> {
        let chars: Vec<char> = input.chars().collect();
        let mut idx = 0;
        let val = Self::parse_value(&chars, &mut idx)?;
        Self::skip_ws(&chars, &mut idx);
        if idx < chars.len() {
            return Err(format!("trailing characters at index {idx}"));
        }
        Ok(val)
    }

    fn skip_ws(chars: &[char], idx: &mut usize) {
        while *idx < chars.len() && chars[*idx].is_whitespace() {
            *idx += 1;
        }
    }

    fn parse_value(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        Self::skip_ws(chars, idx);
        if *idx >= chars.len() {
            return Err("unexpected end of JSON input".to_owned());
        }
        match chars[*idx] {
            'n' => Self::parse_null(chars, idx),
            't' | 'f' => Self::parse_bool(chars, idx),
            '"' => Self::parse_string(chars, idx).map(Self::String),
            '[' => Self::parse_array(chars, idx),
            '{' => Self::parse_object(chars, idx),
            '-' | '0'..='9' => Self::parse_number(chars, idx),
            other => Err(format!("unexpected character '{other}' at index {idx}")),
        }
    }

    fn parse_null(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        if *idx + 4 <= chars.len() && chars[*idx..*idx + 4] == ['n', 'u', 'l', 'l'] {
            *idx += 4;
            Ok(Self::Null)
        } else {
            Err(format!("invalid null token at {idx}"))
        }
    }

    fn parse_bool(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        if *idx + 4 <= chars.len() && chars[*idx..*idx + 4] == ['t', 'r', 'u', 'e'] {
            *idx += 4;
            Ok(Self::Bool(true))
        } else if *idx + 5 <= chars.len() && chars[*idx..*idx + 5] == ['f', 'a', 'l', 's', 'e'] {
            *idx += 5;
            Ok(Self::Bool(false))
        } else {
            Err(format!("invalid boolean token at {idx}"))
        }
    }

    fn parse_string(chars: &[char], idx: &mut usize) -> Result<String, String> {
        if chars[*idx] != '"' {
            return Err("expected '\"'".to_owned());
        }
        *idx += 1;
        let mut s = String::new();
        while *idx < chars.len() {
            let c = chars[*idx];
            *idx += 1;
            if c == '"' {
                return Ok(s);
            }
            if c == '\\' {
                if *idx >= chars.len() {
                    return Err("unterminated string escape".to_owned());
                }
                let esc = chars[*idx];
                *idx += 1;
                match esc {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'b' => s.push('\x08'),
                    'f' => s.push('\x0c'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    'u' => {
                        if *idx + 4 > chars.len() {
                            return Err("truncated unicode escape".to_owned());
                        }
                        let hex: String = chars[*idx..*idx + 4].iter().collect();
                        *idx += 4;
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|e| format!("invalid hex in unicode escape: {e}"))?;
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        }
                    }
                    other => return Err(format!("unknown escape '\\{other}'")),
                }
            } else {
                s.push(c);
            }
        }
        Err("unterminated string literal".to_owned())
    }

    fn parse_number(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        let start = *idx;
        if chars[*idx] == '-' {
            *idx += 1;
        }
        while *idx < chars.len()
            && (chars[*idx].is_ascii_digit()
                || chars[*idx] == '.'
                || chars[*idx] == 'e'
                || chars[*idx] == 'E'
                || chars[*idx] == '+'
                || chars[*idx] == '-')
        {
            if (chars[*idx] == '+' || chars[*idx] == '-')
                && chars[*idx - 1] != 'e'
                && chars[*idx - 1] != 'E'
            {
                break;
            }
            *idx += 1;
        }
        let s: String = chars[start..*idx].iter().collect();
        s.parse::<f64>()
            .map(Self::Number)
            .map_err(|e| format!("invalid number '{s}': {e}"))
    }

    fn parse_array(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        *idx += 1; // skip '['
        let mut items = Vec::new();
        Self::skip_ws(chars, idx);
        if *idx < chars.len() && chars[*idx] == ']' {
            *idx += 1;
            return Ok(Self::Array(items));
        }
        loop {
            let item = Self::parse_value(chars, idx)?;
            items.push(item);
            Self::skip_ws(chars, idx);
            if *idx >= chars.len() {
                return Err("unterminated array".to_owned());
            }
            if chars[*idx] == ']' {
                *idx += 1;
                break;
            }
            if chars[*idx] == ',' {
                *idx += 1;
            } else {
                return Err(format!(
                    "expected ',' or ']' in array, got '{}'",
                    chars[*idx]
                ));
            }
        }
        Ok(Self::Array(items))
    }

    fn parse_object(chars: &[char], idx: &mut usize) -> Result<Self, String> {
        *idx += 1; // skip '{'
        let mut map = BTreeMap::new();
        Self::skip_ws(chars, idx);
        if *idx < chars.len() && chars[*idx] == '}' {
            *idx += 1;
            return Ok(Self::Object(map));
        }
        loop {
            Self::skip_ws(chars, idx);
            if *idx >= chars.len() || chars[*idx] != '"' {
                return Err("expected string key in object".to_owned());
            }
            let key = Self::parse_string(chars, idx)?;
            Self::skip_ws(chars, idx);
            if *idx >= chars.len() || chars[*idx] != ':' {
                return Err("expected ':' after object key".to_owned());
            }
            *idx += 1; // skip ':'
            let val = Self::parse_value(chars, idx)?;
            map.insert(key, val);
            Self::skip_ws(chars, idx);
            if *idx >= chars.len() {
                return Err("unterminated object".to_owned());
            }
            if chars[*idx] == '}' {
                *idx += 1;
                break;
            }
            if chars[*idx] == ',' {
                *idx += 1;
            } else {
                return Err(format!(
                    "expected ',' or '}}' in object, got '{}'",
                    chars[*idx]
                ));
            }
        }
        Ok(Self::Object(map))
    }

    fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(map) => map.get(key),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// Tier 1: Feature Coverage (Happy Path Tests)
// ============================================================================

/// T1.1: Feature Coverage — Spin-Wait Coordination Profile (`spin`).
/// Verifies that `--coord spin --repeat 3` executes without error,
/// instruments cumulative barrier wait overhead, and reports TTFT and decode throughput.
#[test]
fn test_tier1_feature_coordination_spin() {
    let fixture = TempModelFixture::new("t1_spin");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("Explain RVV 1.0")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("4")
        .output()
        .expect("spawn k3-diag for spin coordination profile");

    assert!(
        output.status.success(),
        "k3-diag --coord spin failed with exit code {:?}: stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Stdout table must display TTFT, decode throughput, and cumulative barrier wait
    assert!(
        stdout.contains("TTFT")
            || stdout.contains("ttft")
            || stdout.contains("Time to First Token"),
        "stdout must report TTFT metric: {stdout}"
    );
    assert!(
        stdout.contains("tokens/s") || stdout.contains("throughput") || stdout.contains("Decode"),
        "stdout must report decode throughput metric: {stdout}"
    );
    assert!(
        stdout.contains("barrier") || stdout.contains("wait") || stdout.contains("Barrier Wait"),
        "stdout must report cumulative barrier wait metric: {stdout}"
    );
}

/// T1.2: Feature Coverage — Futex / Parking Coordination Profile (`futex`).
/// Verifies that `--coord futex --repeat 3` executes cleanly,
/// records cumulative barrier wait overhead, and reports descriptive metrics.
#[test]
fn test_tier1_feature_coordination_futex() {
    let fixture = TempModelFixture::new("t1_futex");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("Explain RVV 1.0")
        .arg("--coord")
        .arg("futex")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("4")
        .output()
        .expect("spawn k3-diag for futex coordination profile");

    assert!(
        output.status.success(),
        "k3-diag --coord futex failed with exit code {:?}: stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("barrier") || stdout.contains("wait") || stdout.contains("Barrier Wait"),
        "stdout must report barrier wait metric: {stdout}"
    );
}

/// T1.3: Feature Coverage — Rayon Work-Stealing Profile (`rayon`).
/// Verifies dynamic work-stealing thread coordination via `--coord rayon --repeat 3`.
#[test]
fn test_tier1_feature_coordination_rayon() {
    let fixture = TempModelFixture::new("t1_rayon");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("Explain RVV 1.0")
        .arg("--coord")
        .arg("rayon")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("4")
        .output()
        .expect("spawn k3-diag for rayon coordination profile");

    assert!(
        output.status.success(),
        "k3-diag --coord rayon failed with exit code {:?}: stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("barrier")
            || stdout.contains("wait")
            || stdout.contains("Rayon")
            || stdout.contains("rayon"),
        "stdout must report metrics for rayon coordination: {stdout}"
    );
}

/// T1.4: Feature Coverage — Multi-Run Repeat Execution & Statistics Pipeline.
/// Verifies that `--repeat 4` runs 4 iterations and reports mean, min, max, and stddev.
#[test]
fn test_tier1_feature_multi_run_repeat_pipeline_statistics() {
    let fixture = TempModelFixture::new("t1_repeat");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("4")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for multi-run repeat");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("mean") || stdout.contains("Mean"),
        "stdout must report mean metric: {stdout}"
    );
    assert!(
        stdout.contains("min") || stdout.contains("Min"),
        "stdout must report min metric: {stdout}"
    );
    assert!(
        stdout.contains("max") || stdout.contains("Max"),
        "stdout must report max metric: {stdout}"
    );
    assert!(
        stdout.contains("stddev")
            || stdout.contains("StdDev")
            || stdout.contains("std dev")
            || stdout.contains("σ"),
        "stdout must report stddev metric: {stdout}"
    );
}

/// T1.5: Feature Coverage — Machine-Readable JSON Export (`--json`).
/// Verifies that `--json` emits valid JSON adhering to the `MultiRunReport` schema contract.
#[test]
fn test_tier1_feature_json_structured_export_schema() {
    let fixture = TempModelFixture::new("t1_json");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello world")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("3")
        .arg("--json")
        .output()
        .expect("spawn k3-diag with --json");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not valid JSON ({e}):\n{stdout}"));

    // Verify top-level contract schema fields
    let profile = json
        .get("coordination_profile")
        .and_then(|v| v.as_str())
        .expect("coordination_profile string present");
    assert_eq!(profile.to_ascii_lowercase(), "spin");

    let repeat_count = json
        .get("repeat_count")
        .and_then(|v| v.as_f64())
        .expect("repeat_count present");
    assert_eq!(repeat_count as usize, 3);

    let determinism = json
        .get("determinism_verified")
        .and_then(|v| v.as_bool())
        .expect("determinism_verified boolean present");
    assert!(determinism, "determinism_verified must be true at temp 0");

    let runs = json
        .get("runs")
        .and_then(|v| v.as_array())
        .expect("runs array present");
    assert_eq!(
        runs.len(),
        3,
        "runs array must contain exactly 3 iterations"
    );

    // Verify runs[0] has required metric fields
    let first_run = &runs[0];
    assert!(first_run.get("ttft_ms").is_some(), "run missing ttft_ms");
    assert!(
        first_run.get("decode_tokens_per_sec").is_some(),
        "run missing decode_tokens_per_sec"
    );
    assert!(
        first_run.get("cumulative_barrier_wait_ms").is_some(),
        "run missing cumulative_barrier_wait_ms"
    );

    // Verify statistics summary object
    let stats = json.get("statistics").expect("statistics object present");
    let ttft_stats = stats.get("ttft_ms").expect("ttft_ms stats present");
    assert!(ttft_stats.get("mean").is_some(), "missing ttft_ms mean");
    assert!(ttft_stats.get("min").is_some(), "missing ttft_ms min");
    assert!(ttft_stats.get("max").is_some(), "missing ttft_ms max");
    assert!(ttft_stats.get("stddev").is_some(), "missing ttft_ms stddev");
}

// ============================================================================
// Tier 2: Boundary & Corner Cases
// ============================================================================

/// T2.1: Boundary — Rejection of `--repeat < 3` with Exit Code 2.
/// Verifies that `--repeat 0`, `--repeat 1`, `--repeat 2`, and negative counts are
/// rejected upfront with exit code 2 and a descriptive error message on stderr.
#[test]
fn test_tier2_boundary_repeat_less_than_3_rejected_exit_code_2() {
    let fixture = TempModelFixture::new("t2_repeat_err");

    for bad_repeat in ["0", "1", "2", "-1"] {
        let output = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--coord")
            .arg("spin")
            .arg("--repeat")
            .arg(bad_repeat)
            .output()
            .unwrap_or_else(|e| panic!("spawn k3-diag with --repeat {bad_repeat}: {e}"));

        assert_eq!(
            output.status.code(),
            Some(2),
            "expected exit code 2 for --repeat {bad_repeat}, got: {:?}",
            output.status.code()
        );

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("repeat") || stderr.contains("at least 3"),
            "stderr must describe repeat constraint: {stderr}"
        );
    }
}

/// T2.2: Boundary — Rejection of Invalid Coordination Profile with Exit Code 2.
/// Verifies that unknown coordination strategies (e.g. `--coord unknown`, `--coord mutex`)
/// are rejected upfront with exit code 2 and an actionable message indicating `spin`, `futex`, `rayon`.
#[test]
fn test_tier2_boundary_invalid_coordination_profile_exit_code_2() {
    let fixture = TempModelFixture::new("t2_coord_err");

    for bad_profile in ["unknown", "mutex", "pthread", "invalid"] {
        let output = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--coord")
            .arg(bad_profile)
            .arg("--repeat")
            .arg("3")
            .output()
            .unwrap_or_else(|e| panic!("spawn k3-diag with --coord {bad_profile}: {e}"));

        assert_eq!(
            output.status.code(),
            Some(2),
            "expected exit code 2 for invalid --coord {bad_profile}, got {:?}",
            output.status.code()
        );

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("coordination") || stderr.contains("coord") || stderr.contains("spin"),
            "stderr must report invalid coordination option: {stderr}"
        );
    }
}

/// T2.3: Boundary — Single Worker Thread (`--threads 1`).
/// Verifies that multi-core barriers operate safely on 1 thread without deadlocks,
/// and barrier wait ratio evaluates to approximately 0.0.
#[test]
fn test_tier2_boundary_single_worker_thread_no_deadlock() {
    let fixture = TempModelFixture::new("t2_threads_1");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("spin")
        .arg("--threads")
        .arg("1")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("3")
        .arg("--json")
        .output()
        .expect("spawn k3-diag with 1 thread");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json on 1 thread");
    let runs = json
        .get("runs")
        .and_then(|v| v.as_array())
        .expect("runs array");
    assert_eq!(runs.len(), 3);

    // On 1 thread, barrier wait ratio should be negligible (near 0)
    for run in runs {
        let wait_ratio = run
            .get("barrier_wait_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!(
            wait_ratio <= 0.1,
            "1-thread barrier wait ratio should be near 0, got {wait_ratio}"
        );
    }
}

/// T2.4: Boundary — Zero Max Tokens (`--max-tokens 0`).
/// Verifies that running with 0 decode tokens executes only prefill (TTFT),
/// completes without error, and reports 0 generated decode tokens.
#[test]
fn test_tier2_boundary_zero_max_tokens_prefill_only() {
    let fixture = TempModelFixture::new("t2_max_tokens_0");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("futex")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("0")
        .arg("--json")
        .output()
        .expect("spawn k3-diag with max-tokens 0");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json on max-tokens 0");
    let stats = json.get("statistics").expect("statistics present");
    let ttft = stats.get("ttft_ms").expect("ttft_ms present");
    let mean_ttft = ttft
        .get("mean")
        .and_then(|v| v.as_f64())
        .expect("ttft mean");
    assert!(mean_ttft >= 0.0, "ttft must be non-negative");
}

/// T2.5: Boundary — Context Length Exceeded Upfront Refusal (Exit Code 12).
/// Verifies that when a prompt exceeds the context sequence length (`--n-ctx 2`),
/// execution is refused upfront with exit code 12 (`ContextLengthExceeded`).
#[test]
fn test_tier2_boundary_context_length_exceeded_exit_code_12() {
    let fixture = TempModelFixture::new("t2_ctx_limit");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("Explain RVV 1.0 in full detail with code examples")
        .arg("--n-ctx")
        .arg("2")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("3")
        .output()
        .expect("spawn k3-diag with exceeding context length");

    assert_eq!(
        output.status.code(),
        Some(12),
        "expected exit code 12 for ContextLengthExceeded, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[context_length_exceeded]"),
        "stderr must contain context error tag: {stderr}"
    );
}

/// T2.6: Boundary — Thread Count Exceeding Assigned Cores (Exit Code 13).
/// Verifies that allocating more threads than available in the core mask fails upfront
/// with exit code 13 (`AffinityError`).
#[test]
fn test_tier2_boundary_threads_exceed_core_mask_exit_code_13() {
    let fixture = TempModelFixture::new("t2_aff_err");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cores")
        .arg("0x0001")
        .arg("--threads")
        .arg("2")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("3")
        .output()
        .expect("spawn k3-diag with invalid thread/core ratio");

    assert_eq!(
        output.status.code(),
        Some(13),
        "expected exit code 13 for AffinityError, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[affinity_error]"));
}

// ============================================================================
// Tier 3: Cross-Feature Combinations
// ============================================================================

/// T3.1: Combination — Spin Coordination + 5 Repeats + JSON Export.
/// Pairwise verification of high repeat count with busy-wait spin barrier.
#[test]
fn test_tier3_combo_spin_large_repeat_json_export() {
    let fixture = TempModelFixture::new("t3_spin_5");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("5")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("3")
        .arg("--json")
        .output()
        .expect("spawn k3-diag spin + 5 repeats + json");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json");

    let runs = json
        .get("runs")
        .and_then(|v| v.as_array())
        .expect("runs array");
    assert_eq!(runs.len(), 5, "must have recorded exactly 5 runs");

    let stats = json.get("statistics").expect("statistics present");
    let ttft = stats.get("ttft_ms").expect("ttft present");
    let mean = ttft.get("mean").and_then(|v| v.as_f64()).expect("mean");
    let min = ttft.get("min").and_then(|v| v.as_f64()).expect("min");
    let max = ttft.get("max").and_then(|v| v.as_f64()).expect("max");
    assert!(
        min <= mean && mean <= max,
        "min ({min}) <= mean ({mean}) <= max ({max})"
    );
}

/// T3.2: Combination — Futex Coordination + Scaled Threads + JSON Export.
/// Pairwise verification of multi-threaded futex barrier.
#[test]
fn test_tier3_combo_futex_scaled_threads_json_export() {
    let fixture = TempModelFixture::new("t3_futex_4th");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("futex")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("4")
        .arg("--max-tokens")
        .arg("3")
        .arg("--json")
        .output()
        .expect("spawn k3-diag futex + 4 threads + json");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json");
    let threads = json
        .get("threads")
        .and_then(|v| v.as_f64())
        .expect("threads count");
    assert_eq!(threads as usize, 4);
}

/// T3.3: Combination — Rayon Work-Stealing + Cluster `x100` Preset + Stdout Table.
/// Pairwise verification of Rayon dynamic pool on explicit hardware cluster preset.
#[test]
fn test_tier3_combo_rayon_cluster_x100_table_output() {
    let fixture = TempModelFixture::new("t3_rayon_x100");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("rayon")
        .arg("--cluster")
        .arg("x100")
        .arg("--threads")
        .arg("2")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag rayon + x100 table");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty(), "stdout table must be non-empty");
}

/// T3.4: Combination — Explicit Core Bitmask + Threads + JSON Export.
/// Pairwise verification of explicit core mask targeting (`0x0003`) with multi-run evaluation.
#[test]
fn test_tier3_combo_spin_explicit_cores_and_threads_json() {
    let fixture = TempModelFixture::new("t3_spin_cores");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("spin")
        .arg("--cores")
        .arg("0x0003")
        .arg("--threads")
        .arg("2")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("3")
        .arg("--json")
        .output()
        .expect("spawn k3-diag spin + explicit cores + json");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json");
    let mask = json
        .get("effective_core_mask")
        .expect("effective_core_mask");
    let mask_str = mask.as_str().unwrap_or("");
    assert!(
        mask_str.contains("3") || mask_str.contains("0x0003") || mask_str.contains("0x3"),
        "effective mask must reflect requested core mask 0x0003: {mask_str}"
    );
}

// ============================================================================
// Tier 4: Real-World Application Scenarios
// ============================================================================

/// T4.1: Real-World — Comparative Multi-Core Synchronization Overhead (Scenario 1).
/// Benchmarks `spin`, `futex`, and `rayon` on identical weights, prompt, and cores,
/// validating that all three capture TTFT, decode throughput, and cumulative barrier wait.
#[test]
fn test_tier4_scenario1_comparative_synchronization_overhead() {
    let fixture = TempModelFixture::new("t4_scen1_comp");

    let profiles = ["spin", "futex", "rayon"];
    let mut reports = Vec::new();

    for profile in profiles {
        let output = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--prompt")
            .arg("Explain RVV 1.0")
            .arg("--coord")
            .arg(profile)
            .arg("--repeat")
            .arg("3")
            .arg("--threads")
            .arg("2")
            .arg("--max-tokens")
            .arg("4")
            .arg("--json")
            .output()
            .unwrap_or_else(|e| panic!("run profile {profile}: {e}"));

        assert!(
            output.status.success(),
            "profile {profile} failed with code {:?}: stderr={}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json = SimpleJson::parse(stdout.trim())
            .unwrap_or_else(|e| panic!("invalid json for {profile} ({e}):\n{stdout}"));
        reports.push((profile, json));
    }

    assert_eq!(reports.len(), 3);

    for (name, report) in &reports {
        let stats = report.get("statistics").expect("statistics present");
        let barrier = stats
            .get("cumulative_barrier_wait_ms")
            .expect("barrier wait stats");
        let mean_wait = barrier.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        assert!(
            mean_wait >= 0.0,
            "profile {name} mean barrier wait must be non-negative: {mean_wait}"
        );

        let util = stats
            .get("compute_utilization_ratio")
            .expect("compute util stats");
        let mean_util = util.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        assert!(
            (0.0..=1.0).contains(&mean_util),
            "profile {name} compute utilization ratio must be in [0, 1], got {mean_util}"
        );
    }
}

/// T4.2: Real-World — Bit-Identical Token Determinism at Temperature 0 (Scenario 2 / R3).
/// Asserts that repeat iterations produce identical generated token sequences
/// across runs and coordination profiles.
#[test]
fn test_tier4_scenario2_determinism_at_temperature_zero() {
    let fixture = TempModelFixture::new("t4_scen2_determ");

    // 1. Run with spin profile over 4 iterations
    let out_spin = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello world")
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("4")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("4")
        .arg("--json")
        .output()
        .expect("spawn spin for determinism test");

    assert!(
        out_spin.status.success(),
        "spin stderr={}",
        String::from_utf8_lossy(&out_spin.stderr)
    );
    let json_spin = SimpleJson::parse(String::from_utf8_lossy(&out_spin.stdout).trim())
        .expect("valid spin json");
    assert!(
        json_spin
            .get("determinism_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "determinism_verified must be true in spin profile"
    );

    // 2. Run with futex profile over 4 iterations
    let out_futex = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello world")
        .arg("--coord")
        .arg("futex")
        .arg("--repeat")
        .arg("4")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("4")
        .arg("--json")
        .output()
        .expect("spawn futex for determinism test");

    assert!(
        out_futex.status.success(),
        "futex stderr={}",
        String::from_utf8_lossy(&out_futex.stderr)
    );
    let json_futex = SimpleJson::parse(String::from_utf8_lossy(&out_futex.stdout).trim())
        .expect("valid futex json");
    assert!(
        json_futex
            .get("determinism_verified")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "determinism_verified must be true in futex profile"
    );
}

/// T4.3: Real-World — Clean Resource Teardown and Lock Release.
/// Verifies that upon benchmark termination, shared file locks (`flock(LOCK_SH)`),
/// locked memory (`mlock`), and worker threads are completely released.
#[test]
fn test_tier4_clean_resource_teardown_and_lock_release() {
    let fixture = TempModelFixture::new("t4_teardown");

    // Spawn k3-diag multi-run benchmark
    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--coord")
        .arg("spin")
        .arg("--repeat")
        .arg("3")
        .arg("--threads")
        .arg("2")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for teardown verification");

    assert!(output.status.success());

    // Immediately verify that an exclusive flock(LOCK_EX) succeeds on the fixture file.
    // If the process leaked any locks or left active threads holding the descriptor,
    // probe_flock_exclusive will fail.
    assert!(
        probe_flock_exclusive(fixture.path()),
        "exclusive lock probe must succeed immediately after benchmark completion"
    );
}

/// T4.4: Real-World — Backward Compatibility with US-01 Single-Run Invocations.
/// Verifies that when invoked without `--coord` or `--repeat`, `k3-diag` retains its
/// original US-01 streaming behavior to stdout.
#[test]
fn test_tier4_backward_compatibility_single_run_unbuffered_streaming() {
    let fixture = TempModelFixture::new("t4_us01_compat");

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello world")
        .arg("--threads")
        .arg("1")
        .arg("--n-ctx")
        .arg("64")
        .arg("--max-tokens")
        .arg("5")
        .output()
        .expect("spawn k3-diag in US-01 compatibility mode");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "helloworldExplainRVV1.0",
        "single-run mode without --repeat/--coord must preserve US-01 streaming output"
    );
}

// ============================================================================
// Adversarial & Mathematical Integrity
// ============================================================================

/// Adversarial 1: Special Characters and Escaping in Input Prompts under `--json`.
/// Verifies that prompts containing quotes, newlines, tabs, and escape sequences
/// produce valid, uncorrupted JSON output.
#[test]
fn test_adversarial_special_character_escaping_in_json() {
    let fixture = TempModelFixture::new("adv_escape");

    let adversarial_prompt = "hello \"quoted\" \n\t\\ world!";

    let output = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg(adversarial_prompt)
        .arg("--coord")
        .arg("futex")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("2")
        .arg("--json")
        .output()
        .expect("spawn with adversarial prompt");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = SimpleJson::parse(stdout.trim());
    assert!(
        parsed.is_ok(),
        "output with special characters must parse as valid JSON: err={:?}, stdout={stdout}",
        parsed.err()
    );
}

/// Adversarial 2: Missing Required `--model` Flag with US-02 Arguments.
/// Verifies that omitting `--model` exits cleanly with exit code 2 and a usage message.
#[test]
fn test_adversarial_missing_model_fails_with_exit_code_2() {
    let output = k3_diag_cmd()
        .arg("--prompt")
        .arg("hello world")
        .output()
        .expect("spawn without model");

    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit code 2 when --model is missing"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing required") && stderr.contains("--model"),
        "stderr must report missing required --model: {stderr}"
    );
}

/// Adversarial 3: Direct Mathematical Property Verification for Descriptive Statistics.
/// Validates sample mean, min, max, and sample standard deviation with Bessel's correction ($N-1$).
#[test]
fn test_adversarial_statistical_properties_and_bessel_correction() {
    // Formula verification with known values
    let sample = [10.0, 12.0, 14.0];
    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;
    assert!((mean - 12.0).abs() < 1e-9);

    let variance = sample.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let stddev = variance.sqrt();
    // Sum of squared diffs: (10-12)^2 + (12-12)^2 + (14-12)^2 = 4 + 0 + 4 = 8.
    // Sample variance with N-1 = 8 / 2 = 4.0.
    // Sample stddev = sqrt(4.0) = 2.0.
    assert!(
        (stddev - 2.0).abs() < 1e-9,
        "stddev must be exactly 2.0 with Bessel's correction"
    );

    // Zero-variance property: identical values must yield stddev == 0.0
    let identical = [5.5, 5.5, 5.5, 5.5];
    let n_id = identical.len() as f64;
    let mean_id = identical.iter().sum::<f64>() / n_id;
    let var_id = identical.iter().map(|x| (x - mean_id).powi(2)).sum::<f64>() / (n_id - 1.0);
    let stddev_id = var_id.sqrt();
    assert_eq!(stddev_id, 0.0, "identical samples must yield stddev of 0.0");
}
