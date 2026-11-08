//! US-03 Milestone 1: Heterogeneous Cluster Placement & Split Validation Test Suite.
//!
//! Subprocess CLI integration tests for `k3-diag`:
//! - Tier 1: Single-cluster and cross-cluster valid execution (`x100`, `a100`, `both` with `phase`/`op`/`row`)
//! - Tier 2: Validation rejection rules:
//!   * Invalid `--split` strategy -> Exit 2
//!   * `--cluster both` without `--split` -> Exit 2
//!   * `--split` with single cluster (`x100` / `a100`) -> Exit 2
//!   * `--ime on` on X100 -> Exit 10 (`CapabilityMismatch`)
//!   * Mixed core mask without explicit split policy -> Exit 13 (`AffinityError`)
//! - Tier 3: Combinations with `--repeat` and `--json` export
//! - Tier 4: Backward compatibility and resource cleanup

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::single_char_pattern
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
            "k3_us03_m1_{tag}_{}_{}.gguf",
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

static RAND_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(100);

fn fastrand() -> u64 {
    use std::sync::atomic::Ordering;
    let prev = RAND_SEED.fetch_add(1, Ordering::Relaxed);
    prev.wrapping_mul(6364136223846793005).wrapping_add(1)
}

fn k3_diag_cmd() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jero-diag"));
    cmd.env("K3_CPU_ONLINE_MASK", "0xFFFF");
    cmd
}

fn probe_flock_exclusive(path: &Path) -> bool {
    if let Ok(file) = File::open(path) {
        // SAFETY: Non-blocking flock probe on a valid open file descriptor.
        let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if ret == 0 {
            // SAFETY: Unlock before closing the probe file descriptor.
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
// Zero-Dependency JSON Parser for Test Assertions
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
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
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
        *idx += 1;
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
        *idx += 1;
        let mut map = BTreeMap::new();
        Self::skip_ws(chars, idx);
        if *idx < chars.len() && chars[*idx] == '}' {
            *idx += 1;
            return Ok(Self::Object(map));
        }
        loop {
            Self::skip_ws(chars, idx);
            let key = Self::parse_string(chars, idx)?;
            Self::skip_ws(chars, idx);
            if *idx >= chars.len() || chars[*idx] != ':' {
                return Err("expected ':' after object key".to_owned());
            }
            *idx += 1;
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
}

// ============================================================================
// Tier 1: Feature Coverage (Happy Path Tests)
// ============================================================================

#[test]
fn test_tier1_feature_single_cluster_x100_execution() {
    let fixture = TempModelFixture::new("t1_x100");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("x100")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for x100 execution");

    assert!(
        out.status.success(),
        "x100 execution failed with {:?}: stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "stdout should contain generated tokens"
    );
}

#[test]
fn test_tier1_feature_single_cluster_a100_execution() {
    let fixture = TempModelFixture::new("t1_a100");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("a100")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for a100 execution");

    assert!(
        out.status.success(),
        "a100 execution failed with {:?}: stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "stdout should contain generated tokens"
    );
}

#[test]
fn test_tier1_feature_cross_cluster_split_phase() {
    let fixture = TempModelFixture::new("t1_both_phase");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("both")
        .arg("--split")
        .arg("phase")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for both+phase");

    assert!(
        out.status.success(),
        "both+phase execution failed with {:?}: stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_feature_cross_cluster_split_op() {
    let fixture = TempModelFixture::new("t1_both_op");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("both")
        .arg("--split")
        .arg("op")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for both+op");

    assert!(
        out.status.success(),
        "both+op execution failed with {:?}: stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_feature_cross_cluster_split_row() {
    let fixture = TempModelFixture::new("t1_both_row");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("both")
        .arg("--split")
        .arg("row")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for both+row");

    assert!(
        out.status.success(),
        "both+row execution failed with {:?}: stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_feature_ime_toggle_off_on_x100() {
    let fixture = TempModelFixture::new("t1_x100_ime_off");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("x100")
        .arg("--ime")
        .arg("off")
        .arg("--max-tokens")
        .arg("2")
        .output()
        .expect("spawn k3-diag for x100 ime off");

    assert!(
        out.status.success(),
        "ime off on x100 should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_tier1_feature_ime_toggle_on_on_a100() {
    let fixture = TempModelFixture::new("t1_a100_ime_on");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("a100")
        .arg("--ime")
        .arg("on")
        .arg("--max-tokens")
        .arg("2")
        .output()
        .expect("spawn k3-diag for a100 ime on");

    assert!(
        out.status.success(),
        "ime on on a100 should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ============================================================================
// Tier 2: Boundary & Corner Cases (Validation & Exit Codes)
// ============================================================================

#[test]
fn test_tier2_boundary_rejection_invalid_split_exit_code_2() {
    let fixture = TempModelFixture::new("t2_invalid_split");
    for bad_split in ["unknown", "invalid", "foo", "batch", "123"] {
        let out = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--cluster")
            .arg("both")
            .arg("--split")
            .arg(bad_split)
            .output()
            .unwrap_or_else(|e| panic!("spawn with --split {bad_split}: {e}"));

        assert_eq!(
            out.status.code(),
            Some(2),
            "expected exit code 2 for invalid --split {bad_split}"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("split") && (stderr.contains("phase") || stderr.contains("invalid")),
            "stderr must describe split error: {stderr}"
        );
    }
}

#[test]
fn test_tier2_boundary_rejection_cluster_both_without_split_exit_code_2() {
    let fixture = TempModelFixture::new("t2_both_no_split");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("both")
        .output()
        .expect("spawn k3-diag with --cluster both and no --split");

    assert_eq!(
        out.status.code(),
        Some(2),
        "expected exit code 2 for --cluster both without --split"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--split") || stderr.contains("split policy"),
        "stderr must note requirement of --split with --cluster both: {stderr}"
    );
}

#[test]
fn test_tier2_boundary_rejection_split_with_single_cluster_exit_code_2() {
    let fixture = TempModelFixture::new("t2_split_on_single");
    for single_cluster in ["x100", "a100"] {
        let out = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--cluster")
            .arg(single_cluster)
            .arg("--split")
            .arg("phase")
            .output()
            .unwrap_or_else(|e| panic!("spawn with cluster {single_cluster}: {e}"));

        assert_eq!(
            out.status.code(),
            Some(2),
            "expected exit code 2 for --split on single cluster {single_cluster}"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--split") || stderr.contains("both"),
            "stderr must explain --split only applies to --cluster both: {stderr}"
        );
    }
}

#[test]
fn test_tier2_boundary_rejection_invalid_cluster_exit_code_2() {
    let fixture = TempModelFixture::new("t2_bad_cluster");
    for bad_cluster in ["all", "c0", "gpu", "invalid", "0"] {
        let out = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--cluster")
            .arg(bad_cluster)
            .output()
            .unwrap_or_else(|e| panic!("spawn with cluster {bad_cluster}: {e}"));

        assert_eq!(
            out.status.code(),
            Some(2),
            "expected exit code 2 for bad cluster {bad_cluster}"
        );
    }
}

#[test]
fn test_tier2_boundary_rejection_invalid_ime_value_exit_code_2() {
    let fixture = TempModelFixture::new("t2_bad_ime");
    for bad_ime in ["true", "false", "yes", "1", "auto"] {
        let out = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--cluster")
            .arg("a100")
            .arg("--ime")
            .arg(bad_ime)
            .output()
            .unwrap_or_else(|e| panic!("spawn with --ime {bad_ime}: {e}"));

        assert_eq!(
            out.status.code(),
            Some(2),
            "expected exit code 2 for invalid --ime {bad_ime}"
        );
    }
}

#[test]
fn test_tier2_boundary_rejection_ime_on_on_x100_exit_code_10() {
    let fixture = TempModelFixture::new("t2_ime_x100_err");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("x100")
        .arg("--ime")
        .arg("on")
        .output()
        .expect("spawn with x100 ime on");

    assert_eq!(
        out.status.code(),
        Some(10),
        "expected exit code 10 for IME on X100, got: {:?}",
        out.status.code()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[capability_mismatch]"),
        "stderr must contain capability_mismatch tag: {stderr}"
    );
    assert!(
        stderr.contains("IME") || stderr.contains("matrix"),
        "stderr must mention IME capability: {stderr}"
    );
}

#[test]
fn test_tier2_boundary_rejection_mixed_core_mask_without_split_exit_code_13() {
    let fixture = TempModelFixture::new("t2_mixed_mask_err");
    // 0x0101 contains Core 0 (X100) and Core 8 (A100)
    for mixed_mask in ["0x0101", "0xFFFF", "0x00FF00FF"] {
        let out = k3_diag_cmd()
            .arg("--model")
            .arg(fixture.path())
            .arg("--cores")
            .arg(mixed_mask)
            .output()
            .unwrap_or_else(|e| panic!("spawn with mixed mask {mixed_mask}: {e}"));

        assert_eq!(
            out.status.code(),
            Some(13),
            "expected exit code 13 for mixed mask {mixed_mask}"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[affinity_error]"),
            "stderr must contain affinity_error tag: {stderr}"
        );
        assert!(
            stderr.contains("mixes") || stderr.contains("split policy"),
            "stderr must describe mixed VLEN domains: {stderr}"
        );
    }
}

#[test]
fn test_tier2_boundary_rejection_cores_outside_cluster_exit_code_13() {
    let fixture = TempModelFixture::new("t2_cores_outside");
    // 0x0100 is Core 8 (A100), but requested cluster is x100 (0x00FF)
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("x100")
        .arg("--cores")
        .arg("0x0100")
        .output()
        .expect("spawn with mismatched cluster and cores");

    assert_eq!(
        out.status.code(),
        Some(13),
        "expected exit code 13 for cores outside cluster"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[affinity_error]"));
}

// ============================================================================
// Tier 3: Combinations & JSON Export Schema
// ============================================================================

#[test]
fn test_tier3_combo_cluster_both_split_phase_repeat_json() {
    let fixture = TempModelFixture::new("t3_both_json");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("both")
        .arg("--split")
        .arg("phase")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("2")
        .arg("--json")
        .output()
        .expect("spawn both+phase+json");

    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json export");

    let cluster = json.get("cluster").and_then(|v| v.as_str());
    assert_eq!(cluster, Some("both"), "json should report cluster: both");

    let split = json.get("split").and_then(|v| v.as_str());
    assert_eq!(split, Some("phase"), "json should report split: phase");
}

#[test]
fn test_tier3_combo_cluster_x100_ime_off_json() {
    let fixture = TempModelFixture::new("t3_x100_json");
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("x100")
        .arg("--ime")
        .arg("off")
        .arg("--repeat")
        .arg("3")
        .arg("--max-tokens")
        .arg("2")
        .arg("--json")
        .output()
        .expect("spawn x100+ime_off+json");

    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json = SimpleJson::parse(stdout.trim()).expect("valid json export");

    let cluster = json.get("cluster").and_then(|v| v.as_str());
    assert_eq!(cluster, Some("x100"));
}

// ============================================================================
// Tier 4: Backward Compatibility & Resource Hygiene
// ============================================================================

#[test]
fn test_tier4_backward_compatibility_us01_and_us02() {
    let fixture = TempModelFixture::new("t4_compat");
    // Default single eval without --cluster or --split
    let out = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--prompt")
        .arg("hello")
        .arg("--max-tokens")
        .arg("2")
        .output()
        .expect("spawn default eval");

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.trim().is_empty());
}

#[test]
fn test_tier4_clean_lock_release_on_cli_rejection() {
    let fixture = TempModelFixture::new("t4_locks");
    // Run an invocation that is rejected upfront (exit 2)
    let _ = k3_diag_cmd()
        .arg("--model")
        .arg(fixture.path())
        .arg("--cluster")
        .arg("both")
        .output();

    // Verify file lock is immediately acquirable
    assert!(
        probe_flock_exclusive(fixture.path()),
        "flock must be free after rejected CLI invocation"
    );
}
