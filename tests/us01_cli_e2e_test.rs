//! Subprocess CLI End-to-End Integration Tests for `k3-diag` (`US-01.7.3`, #51).
//!
//! Invokes the compiled `k3-diag` binary directly against synthetic GGUF v3 fixtures
//! to validate US-01 Scenarios 1, 2, and 3, exit code mappings, and `stdout`/`stderr` output.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use jero_core::QuantType;
use jero_core::test_helpers::create_synthetic_gguf_v3;

fn create_temp_gguf_model(tag: &str, architecture: &str) -> PathBuf {
    let tensor_specs = [
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
        ("output_norm.weight", QuantType::F32, &[16][..]),
    ];
    let bytes = create_synthetic_gguf_v3(architecture, 32, &tensor_specs);
    let path =
        std::env::temp_dir().join(format!("k3_diag_cli_e2e_{tag}_{}.gguf", std::process::id()));
    let mut f = File::create(&path).expect("create temp GGUF");
    f.write_all(&bytes).expect("write GGUF bytes");
    f.sync_all().expect("sync GGUF file");
    path
}

#[test]
fn test_cli_us01_scenario_1_direct_model_eval_and_streaming() {
    let model_path = create_temp_gguf_model("scen1", "synthetic");
    let bin = env!("CARGO_BIN_EXE_jero-diag");

    let output = Command::new(bin)
        .arg("--model")
        .arg(&model_path)
        .arg("--prompt")
        .arg("hello world")
        .arg("--cores")
        .arg("0x0001")
        .arg("--threads")
        .arg("1")
        .arg("--n-ctx")
        .arg("64")
        .arg("--max-tokens")
        .arg("5")
        .output()
        .expect("spawn k3-diag for Scenario 1");

    assert!(
        output.status.success(),
        "k3-diag exited with {:?}: stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout_str.trim(), "helloworldExplainRVV1.0");

    // Verify unsupported architecture fails upfront with exit code 10 (capability_mismatch)
    let unsupported_path = create_temp_gguf_model("scen1_unsupported", "qwen3.8");
    let bad_out = Command::new(bin)
        .arg("--model")
        .arg(&unsupported_path)
        .arg("--prompt")
        .arg("hello")
        .output()
        .expect("spawn k3-diag on unsupported arch");

    assert_eq!(bad_out.status.code(), Some(10));
    let stderr_str = String::from_utf8_lossy(&bad_out.stderr);
    assert!(stderr_str.contains("[capability_mismatch]"));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&unsupported_path);
}

#[test]
fn test_cli_us01_scenario_2_cluster_and_concurrency_targeting() {
    let model_path = create_temp_gguf_model("scen2", "synthetic");
    let bin = env!("CARGO_BIN_EXE_jero-diag");

    // 1. Valid explicit --cluster x100 --cores 0x0001 --threads 1
    let ok_out = Command::new(bin)
        .arg("--model")
        .arg(&model_path)
        .arg("--prompt")
        .arg("hello")
        .arg("--cluster")
        .arg("x100")
        .arg("--cores")
        .arg("0x0001")
        .arg("--threads")
        .arg("1")
        .arg("--n-ctx")
        .arg("64")
        .arg("--max-tokens")
        .arg("3")
        .output()
        .expect("spawn k3-diag for Scenario 2 valid cluster");

    assert!(
        ok_out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&ok_out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&ok_out.stdout).trim(),
        "helloworldExplain"
    );

    // 2. Invalid thread count exceeding cores in mask (2 threads on 1-core mask 0x0001) -> exit code 13
    let err_out = Command::new(bin)
        .arg("--model")
        .arg(&model_path)
        .arg("--prompt")
        .arg("hello")
        .arg("--cores")
        .arg("0x0001")
        .arg("--threads")
        .arg("2")
        .arg("--n-ctx")
        .arg("64")
        .output()
        .expect("spawn k3-diag with invalid thread/core bound");

    assert_eq!(err_out.status.code(), Some(13));
    let stderr_str = String::from_utf8_lossy(&err_out.stderr);
    assert!(stderr_str.contains("[affinity_error]"));

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_us01_scenario_3_upfront_memory_admission_refusal() {
    let model_path = create_temp_gguf_model("scen3", "synthetic");
    let bin = env!("CARGO_BIN_EXE_jero-diag");

    // Constrain lockable memory via K3_RLIMIT_MEMLOCK_BYTES=256 so model + KV exceeds lockable RAM
    let out = Command::new(bin)
        .env("K3_RLIMIT_MEMLOCK_BYTES", "256")
        .arg("--model")
        .arg(&model_path)
        .arg("--prompt")
        .arg("hello world")
        .arg("--n-ctx")
        .arg("64")
        .output()
        .expect("spawn k3-diag under constrained lockable memory");

    assert_eq!(
        out.status.code(),
        Some(11),
        "expected exit code 11 for InsufficientLockableMemory, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr_str = String::from_utf8_lossy(&out.stderr);
    assert!(stderr_str.contains("[insufficient_lockable_memory]"));
    assert!(
        out.stdout.is_empty(),
        "stdout must be empty when run is refused upfront"
    );

    let _ = std::fs::remove_file(&model_path);
}
