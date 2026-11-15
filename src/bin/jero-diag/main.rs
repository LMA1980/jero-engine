//! `k3-diag`: Operator diagnostic CLI binary.
//!
//! Provides a standalone zero-orchestration tool for verifying model correctness,
//! capability filtering, pre-flight memory admission, locked weight residency (`mlock` + `flock`),
//! core affinity, multi-core coordination profiles (`spin`, `futex`, `rayon`), multi-run jitter
//! benchmarking, in-process tokenization, and real-time unbuffered token streaming on
//! `SpacemiT` K3 hardware (US-01, US-02, US-03 cluster splitting and IME benchmarking, ADR-005, ADR-006, ADR-007, ADR-010, ADR-023, ADR-025, ADR-027, ADR-037, ADR-040).

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::missing_errors_doc
)]

pub mod bench;
pub mod eval;
pub mod parity;
pub mod serve;

use eval::EvalSuite;

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use jero_core::{CoordinationProfile, GgufFile, K3Error, ModelIdentity, QuantizationInfo};
use jero_engine::affinity::{SplitStrategy, resolve_affinity_split, resolve_phase_policy};
use jero_engine::forward::MatKernel;
use jero_engine::memory::calculate_footprint;
use jero_engine::{
    A100_CLUSTER_MASK, ClusterPhasePolicy, GenerationConfig, LockedModelWeights,
    SinglePromptEngine, SingleSlotKvArena, StreamAction, SystemMemoryInspector, WorkerPool,
    X100_CLUSTER_MASK, tokenize_prompt_utf8_safe,
};
use jero_kernels_rvv::ime::ime_vmadot_count;
use jero_tokenizer::{DetokenizeStep, GgufTokenizer};
use serde::{Deserialize, Serialize};

/// Global atomic flag set when `SIGINT` or `SIGTERM` is received so the decode loop
/// exits cleanly and triggers RAII `Drop` guards (`munlock`, `munmap`, `flock(LOCK_UN)`).
static ABORT_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_termination_signal(_sig: libc::c_int) {
    ABORT_REQUESTED.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // SAFETY: handle_termination_signal is an async-signal-safe extern "C" function
    // that only performs an atomic relaxed/seqcst store on a static AtomicBool.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_termination_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_termination_signal as *const () as libc::sighandler_t,
        );
    }
}

/// Returns `true` if an operator `SIGINT` or `SIGTERM` abort was requested.
#[must_use]
pub fn is_abort_requested() -> bool {
    ABORT_REQUESTED.load(Ordering::SeqCst)
}

/// Command line arguments parsed for a diagnostic session.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DiagConfig {
    /// Path to the local model file (e.g., GGUF format).
    pub model_path: Option<String>,
    /// Input prompt to evaluate.
    pub prompt: Option<String>,
    /// Target processor cluster ("both", "x100" or "a100").
    pub cluster: Option<String>,
    /// Execution-phase cluster scheduling policy (`auto`, `x100-prefill-a100-decode`, `a100-only`, `x100-only`, `row-split`).
    pub phase_policy: Option<ClusterPhasePolicy>,
    /// Cross-cluster workload split policy, valid only with `--cluster both` (US-03).
    pub split: Option<SplitStrategy>,
    /// SpacemiT IME matrix acceleration toggle (`Some(true)` = `on`, US-03).
    pub ime: Option<bool>,
    /// Worker thread count.
    pub threads: Option<usize>,
    /// Bitmask for pinned core affinity.
    pub core_mask: Option<u64>,
    /// Configured context window limit in tokens.
    pub n_ctx: Option<usize>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Multi-core coordination profile (`spin`, `futex`, `rayon`).
    pub coord: Option<CoordinationProfile>,
    /// Multi-run repeat iteration count ($N \ge 3$).
    pub repeat: Option<usize>,
    /// Emit structured JSON export.
    pub json: bool,
    /// Path to export machine-readable CostProfile JSON baseline (US-04).
    pub export_profile: Option<String>,
    /// Explicit human-readable benchmark summary request.
    pub summary: bool,
    /// Open-source capability benchmark suite selector (`mmlu_pro`, `gsm8k`, `humaneval`, `ifeval`, `truthfulqa`, `all`).
    pub eval_suite: Option<EvalSuite>,
    /// Optional path to JSONL dataset file for `--eval-suite`.
    pub eval_dataset: Option<String>,
    /// Optional path to FP16 reference baseline JSON file for quantization fidelity delta.
    pub reference_baseline: Option<String>,
    /// Optional path to export the machine-readable `PublicBenchmarkReport` JSON.
    pub export_eval: Option<String>,
    /// Optional override for Dory recurrent loop iterations (`1..=4`).
    pub recurrent_loops: Option<u32>,
    /// Optional override for Dory recurrent KV cache mode (`"per_loop"` or `"last_loop"`).
    pub recurrent_kv_mode: Option<String>,
    /// Ephemeral loopback OpenAI-compatible HTTP bind address (`<ip>:<port>`, e.g. `127.0.0.1:0`).
    pub serve_openai: Option<String>,
    /// Optional request count ceiling after which `--serve-openai` exits cleanly.
    pub max_requests: Option<usize>,
    /// Explicit synthetic in-memory hardware matrix calibration mode (`--synthetic-matrix`).
    pub synthetic_matrix: bool,
    /// Path to export deterministic differential parity report JSON (`--dump-parity <path>`).
    pub dump_parity: Option<String>,
    /// Skip family chat-template framing and tokenize the raw prompt verbatim (`--raw-prompt`).
    pub raw_prompt: bool,
    /// Continue decoding past EOS/EOT stop tokens until `max_tokens` (`--ignore-eos`).
    pub ignore_eos: bool,
}

impl DiagConfig {
    /// Parses CLI arguments from an iterator of string arguments.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error message if arguments are malformed or missing values.
    pub fn parse_args<I>(mut args: I) -> Result<Option<Self>, String>
    where
        I: Iterator<Item = String>,
    {
        let mut config = Self::default();

        // Skip binary name if present
        let _ = args.next();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    return Ok(None);
                }
                "--version" | "-v" => {
                    println!("k3-diag {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                "--model" | "-m" => {
                    config.model_path = Some(
                        args.next()
                            .ok_or_else(|| "--model requires a file path".to_owned())?,
                    );
                }
                "--prompt" | "-p" => {
                    config.prompt = Some(
                        args.next()
                            .ok_or_else(|| "--prompt requires an input string".to_owned())?,
                    );
                }
                "--cluster" => {
                    let cluster = args
                        .next()
                        .ok_or_else(|| "--cluster requires 'both', 'x100' or 'a100'".to_owned())?;
                    if cluster != "x100" && cluster != "a100" && cluster != "both" {
                        return Err(format!(
                            "invalid cluster '{cluster}': expected 'both', 'x100' or 'a100'"
                        ));
                    }
                    config.cluster = Some(cluster);
                }
                "--phase-policy" => {
                    let val = args.next().ok_or_else(|| {
                        "--phase-policy requires 'auto', 'x100-prefill-a100-decode', 'a100-only', 'x100-only', or 'row-split'".to_owned()
                    })?;
                    let policy = ClusterPhasePolicy::parse(&val).map_err(|_| {
                        format!(
                            "invalid --phase-policy '{val}': expected 'auto', 'x100-prefill-a100-decode', 'a100-only', 'x100-only', or 'row-split'"
                        )
                    })?;
                    config.phase_policy = Some(policy);
                }
                "--split" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--split requires 'phase', 'op' or 'row'".to_owned())?;
                    config.split = Some(match val.as_str() {
                        "phase" => SplitStrategy::Phase,
                        "op" => SplitStrategy::Op,
                        "row" => SplitStrategy::Row,
                        _ => {
                            return Err(format!(
                                "invalid split strategy '{val}': expected 'phase', 'op' or 'row'"
                            ));
                        }
                    });
                }
                "--ime" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--ime requires 'on' or 'off'".to_owned())?;
                    config.ime = Some(match val.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => {
                            return Err(format!(
                                "invalid IME toggle '{val}': expected 'on' or 'off'"
                            ));
                        }
                    });
                }
                "--threads" | "-t" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--threads requires an integer count".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid thread count '{val}'"))?;
                    if n == 0 {
                        return Err("thread count must be at least 1".to_owned());
                    }
                    config.threads = Some(n);
                }
                "--cores" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--cores requires a hex or decimal mask".to_owned())?;
                    let mask = if let Some(hex) =
                        val.strip_prefix("0x").or_else(|| val.strip_prefix("0X"))
                    {
                        u64::from_str_radix(hex, 16)
                    } else {
                        val.parse::<u64>()
                    }
                    .map_err(|_| format!("invalid core mask '{val}'"))?;
                    config.core_mask = Some(mask);
                }
                "--n-ctx" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--n-ctx requires a token limit integer".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid n-ctx '{val}'"))?;
                    config.n_ctx = Some(n);
                }
                "--max-tokens" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--max-tokens requires an integer".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid max-tokens '{val}'"))?;
                    config.max_tokens = Some(n);
                }
                "--coord" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--coord requires 'spin', 'futex', or 'rayon'".to_owned())?;
                    let profile = val.parse::<CoordinationProfile>().map_err(|_| {
                        format!(
                            "invalid coordination profile '{val}': expected 'spin', 'futex', or 'rayon'"
                        )
                    })?;
                    config.coord = Some(profile);
                }
                "--repeat" => {
                    let val = args.next().ok_or_else(|| {
                        "--repeat requires an integer count of at least 3".to_owned()
                    })?;
                    let n = val.parse::<isize>().map_err(|_| {
                        format!("invalid repeat count '{val}': expected integer of at least 3")
                    })?;
                    if n < 3 {
                        return Err(format!(
                            "--repeat must be an integer of at least 3 (got {val})"
                        ));
                    }
                    config.repeat = Some(n as usize);
                }
                "--json" => {
                    config.json = true;
                }
                "--export-profile" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "--export-profile requires a file path".to_owned())?;
                    if path.trim().is_empty() {
                        return Err("--export-profile requires a non-empty file path".to_owned());
                    }
                    config.export_profile = Some(path);
                }
                "--summary" | "--bench" => {
                    config.summary = true;
                }
                "--eval-suite" => {
                    let val = args.next().ok_or_else(|| {
                        "--eval-suite requires 'mmlu_pro', 'gsm8k', 'humaneval', 'ifeval', 'truthfulqa', or 'all'".to_owned()
                    })?;
                    config.eval_suite = Some(EvalSuite::parse(&val)?);
                }
                "--eval-dataset" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "--eval-dataset requires a JSONL file path".to_owned())?;
                    if path.trim().is_empty() {
                        return Err("--eval-dataset requires a non-empty file path".to_owned());
                    }
                    config.eval_dataset = Some(path);
                }
                "--reference-baseline" => {
                    let path = args.next().ok_or_else(|| {
                        "--reference-baseline requires a JSON file path".to_owned()
                    })?;
                    if path.trim().is_empty() {
                        return Err(
                            "--reference-baseline requires a non-empty file path".to_owned()
                        );
                    }
                    config.reference_baseline = Some(path);
                }
                "--export-eval" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "--export-eval requires a JSON file path".to_owned())?;
                    if path.trim().is_empty() {
                        return Err("--export-eval requires a non-empty file path".to_owned());
                    }
                    config.export_eval = Some(path);
                }
                "--recurrent-loops" => {
                    let val = args.next().ok_or_else(|| {
                        "--recurrent-loops requires an integer in 1..=4".to_owned()
                    })?;
                    let loops = val.parse::<u32>().map_err(|_| {
                        format!("invalid --recurrent-loops '{val}': expected integer in 1..=4")
                    })?;
                    if !(1..=4).contains(&loops) {
                        return Err(format!("--recurrent-loops must be in 1..=4 (got {loops})"));
                    }
                    config.recurrent_loops = Some(loops);
                }
                "--recurrent-kv-mode" => {
                    let mode = args.next().ok_or_else(|| {
                        "--recurrent-kv-mode requires 'per_loop' or 'last_loop'".to_owned()
                    })?;
                    if mode != "per_loop" && mode != "last_loop" {
                        return Err(format!(
                            "invalid --recurrent-kv-mode '{mode}': expected 'per_loop' or 'last_loop'"
                        ));
                    }
                    config.recurrent_kv_mode = Some(mode);
                }
                "--serve-openai" => {
                    let addr = args.next().ok_or_else(|| {
                        "--serve-openai requires a bind address (e.g. 127.0.0.1:0)".to_owned()
                    })?;
                    if addr.trim().is_empty() {
                        return Err("--serve-openai requires a non-empty bind address".to_owned());
                    }
                    config.serve_openai = Some(addr);
                }
                "--max-requests" => {
                    let val = args.next().ok_or_else(|| {
                        "--max-requests requires a positive integer count".to_owned()
                    })?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --max-requests '{val}'"))?;
                    if n == 0 {
                        return Err("--max-requests must be at least 1".to_owned());
                    }
                    config.max_requests = Some(n);
                }
                "--synthetic-matrix" => {
                    config.synthetic_matrix = true;
                }
                "--dump-parity" => {
                    let path = args.next().ok_or_else(|| {
                        "--dump-parity requires an output JSON file path".to_owned()
                    })?;
                    if path.trim().is_empty() {
                        return Err(
                            "--dump-parity requires a non-empty output file path".to_owned()
                        );
                    }
                    config.dump_parity = Some(path);
                }
                "--raw-prompt" => {
                    config.raw_prompt = true;
                }
                "--ignore-eos" => {
                    config.ignore_eos = true;
                }
                unknown => {
                    return Err(format!("unknown option '{unknown}'"));
                }
            }
        }

        // Prevent silent or ambiguous mixing of real GGUF evaluation with synthetic hardware calibration.
        if config.synthetic_matrix && config.model_path.is_some() {
            return Err(
                "--synthetic-matrix cannot be combined with --model; use --synthetic-matrix alone for hardware calibration or --model <path> for real GGUF execution"
                    .to_owned(),
            );
        }

        // Cluster-boundary rules (US-03): a mixed-cluster mask is allowed only through an
        // explicit split policy, and a split policy only makes sense across both clusters.
        if config.phase_policy.is_some() && config.cluster.is_some() {
            return Err(
                "--phase-policy cannot be combined with --cluster; use one or the other".to_owned(),
            );
        }
        if config.split.is_some() && config.cluster.as_deref() != Some("both") {
            return Err("--split requires --cluster both".to_owned());
        }
        if config.cluster.as_deref() == Some("both") && config.split.is_none() {
            return Err(
                "--cluster both requires an explicit --split <phase|op|row> policy".to_owned(),
            );
        }

        Ok(Some(config))
    }

    /// Returns `true` when the open-source capability benchmark evaluator should run.
    #[must_use]
    pub const fn is_eval_benchmark(&self) -> bool {
        self.eval_suite.is_some()
            || self.eval_dataset.is_some()
            || self.export_eval.is_some()
            || self.reference_baseline.is_some()
    }

    /// Returns `true` ONLY when explicit synthetic hardware matrix calibration (`--synthetic-matrix`) is requested.
    #[must_use]
    pub const fn is_matrix_benchmark(&self) -> bool {
        self.synthetic_matrix
    }

    /// Returns `true` when real GGUF multi-run / placement / cost-profile benchmarking should run.
    #[must_use]
    pub fn is_multi_run_benchmark(&self) -> bool {
        self.repeat.is_some()
            || self.split.is_some()
            || self.ime.is_some()
            || self.cluster.as_deref() == Some("both")
            || self.export_profile.is_some()
            || self.summary
            || self.json
            || (self.cluster.is_some() && self.core_mask.is_none())
    }
}

/// Descriptive statistics for a series of continuous numerical samples ($N \ge 1$).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    #[serde(default)]
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

impl MetricStats {
    /// Computes sample mean, median, min, max, and sample standard deviation with Bessel's correction ($N-1$).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                min: 0.0,
                max: 0.0,
                stddev: 0.0,
            };
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            0.5 * (sorted[mid - 1] + sorted[mid])
        };

        let stddev = if samples.len() < 2 {
            0.0
        } else {
            let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let s = variance.sqrt();
            if s.is_nan() || s < 1e-12 { 0.0 } else { s }
        };

        Self {
            mean,
            median,
            min,
            max,
            stddev,
        }
    }
}

fn default_stop_reason_str() -> String {
    "max_tokens".to_owned()
}

/// Recorded execution metrics for an individual evaluation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetric {
    pub run_index: usize,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    #[serde(default)]
    pub prefill_tokens_per_sec: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_sec: f64,
    pub total_wall_time_ms: f64,
    pub cumulative_barrier_wait_ms: f64,
    pub barrier_wait_ratio: f64,
    pub compute_utilization_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    #[serde(default = "default_stop_reason_str")]
    pub stop_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_id: Option<u32>,
}

/// Aggregated statistical summary for multi-run evaluations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsSummary {
    pub ttft_ms: MetricStats,
    #[serde(default = "zero_metric_stats")]
    pub prefill_tokens_per_sec: MetricStats,
    pub decode_tokens_per_sec: MetricStats,
    pub cumulative_barrier_wait_ms: MetricStats,
    pub total_wall_time_ms: MetricStats,
    pub compute_utilization_ratio: MetricStats,
}

fn zero_metric_stats() -> MetricStats {
    MetricStats::calculate(&[])
}

/// Machine-readable multi-run real-GGUF benchmark report exported via `--json` (US-02, US-03, US-04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiRunReport {
    /// Execution mode discriminator (`"real_gguf"`).
    pub execution_mode: String,
    /// Model architecture read from GGUF metadata (e.g. `llama`, `bitnet`, `dory`, `synthetic`).
    pub architecture: String,
    /// Basename of the evaluated GGUF checkpoint file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_filename: Option<String>,
    /// Concise summary of GGUF weight quantization types and counts.
    pub quant_summary: String,
    pub coordination_profile: String,
    pub repeat_count: usize,
    pub threads: usize,
    pub effective_core_mask: String,
    pub determinism_verified: bool,
    pub prompt: String,
    pub runs: Vec<RunMetric>,
    pub statistics: StatisticsSummary,
    #[serde(default = "default_stop_reason_str")]
    pub stop_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_id: Option<u32>,
    /// Flattened shared placement, ceiling, cost-profile, and determinism fields.
    #[serde(flatten)]
    pub telemetry: bench::BenchmarkTelemetry,
}

/// Maps domain [`K3Error`] variants to distinct process exit codes per US-01.7.1 and US-02.
#[must_use]
pub const fn exit_code_for_error(err: &K3Error) -> u8 {
    match err {
        K3Error::CapabilityMismatch { .. } => 10,
        K3Error::InsufficientLockableMemory { .. } => 11,
        K3Error::ContextLengthExceeded { .. } => 12,
        K3Error::AffinityError { .. } => 13,
        K3Error::TensorBoundsError { .. } => 14,
        K3Error::RoutingUnavailable { .. } => 15,
        K3Error::QueueSaturated { .. } => 23,
    }
}

fn print_usage() {
    println!(
        "Usage: k3-diag --model <path> [options]\n\n\
         Standalone diagnostic model evaluation & jitter benchmarking tool for SpacemiT K3.\n\n\
         Options:\n  \
           -m, --model <path>     Path to local model file (GGUF)\n  \
           -p, --prompt <text>    Input prompt to tokenize and evaluate\n  \
               --cluster <id>     Target processor cluster ('both', 'x100' or 'a100')\n  \
               --phase-policy <p> Phase cluster policy ('auto', 'x100-prefill-a100-decode', 'a100-only', 'x100-only', 'row-split')\n  \
               --split <policy>   Cross-cluster split for '--cluster both' ('phase', 'op', 'row')\n  \
               --ime <on|off>     SpacemiT IME matrix acceleration ('on' also measures the 'off' baseline)\n  \
           -t, --threads <n>      Number of execution worker threads\n  \
               --cores <mask>     Pinned core affinity mask (e.g. 0x00FF)\n  \
               --n-ctx <tokens>   Configured context sequence length (default: 64 for diag/eval, 512 for --serve-openai)\n  \
               --max-tokens <n>   Maximum generation tokens (default: 128 for diag, 16 for eval, 512 for --serve-openai)\n  \
               --raw-prompt       Bypass chat-template framing and tokenize prompt verbatim\n  \
               --ignore-eos       Continue generation until --max-tokens even when EOS/EOT tokens are emitted\n  \
               --coord <strategy> Multi-core coordination profile ('spin', 'futex', 'rayon')\n  \
               --repeat <n>       Repeat benchmark iterations (N >= 3) for jitter measurement\n  \
               --json             Emit structured machine-readable JSON report\n  \
               --export-profile <path> Export structured CostProfile JSON to file\n  \
               --dump-parity <path> Export top-k prefill logits & greedy token parity JSON\n  \
               --eval-suite       Run built-in 5-pillar open-source evaluation benchmark suite\n  \
               --eval-dataset <path> Override 5-pillar suite dataset with custom JSON\n  \
               --reference-baseline <path> Compare 5-pillar accuracy/perplexity against baseline JSON\n  \
               --export-eval <path> Export 5-pillar evaluation report JSON to file\n  \
               --serve-openai <addr> Serve ephemeral OpenAI-compatible HTTP API (e.g. 127.0.0.1:0)\n  \
               --max-requests <n> Exit --serve-openai cleanly after handling N HTTP requests\n  \
               --recurrent-loops <n> Override looped-reasoning iteration count for recurrent architectures\n  \
               --recurrent-kv-mode <mode> Recurrent KV strategy ('last_iteration' or 'shared_per_step')\n  \
               --summary          Emit human-readable summary table\n  \
               --synthetic-matrix Run in-memory synthetic hardware matrix calibration (no --model)\n  \
           -h, --help             Show this help message\n  \
           -v, --version          Show version information"
    );
}

fn resolve_engine_policy(
    split: Option<SplitStrategy>,
    architecture: &str,
    prompt_len: usize,
    effective_mask: u64,
) -> ClusterPhasePolicy {
    let requested = match split {
        Some(SplitStrategy::Row) => ClusterPhasePolicy::RowSplit,
        _ => ClusterPhasePolicy::Auto,
    };
    resolve_phase_policy(requested, architecture, prompt_len, effective_mask)
}

/// Executes multi-iteration evaluation pipeline for real GGUF models with placement, ceiling,
/// cost-profile, and determinism benchmarking (US-02, US-03, US-04).
pub fn run_multi_run_benchmark<W: Write>(config: &DiagConfig, mut out: W) -> Result<(), K3Error> {
    if config.synthetic_matrix {
        return Err(bench::cap_err(
            "cli",
            "--synthetic-matrix cannot be routed through real-GGUF evaluation".to_owned(),
        ));
    }

    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --model <path>".to_owned(),
        })?;
    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(64);
    let repeat_count = config.repeat.unwrap_or(1);
    let ime_on = config.ime == Some(true);
    let profile = config.coord.unwrap_or(if config.phase_policy.is_some() {
        CoordinationProfile::Rayon
    } else {
        CoordinationProfile::Spin
    });

    let inspector = SystemMemoryInspector::new();
    let (
        worker_pool,
        affinity,
        ime_supported,
        before,
        locked_weights,
        recipe,
        tokenizer_vocab_size,
        prompt_tokens,
        effective_core_mask,
        thread_count,
        mut engine,
    ) = if let Some(policy) = config.phase_policy {
        let (locked_weights, recipe, _footprint) =
            LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
        let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
        let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
        let prompt_tokens =
            tokenize_prompt_utf8_safe(&tokenizer, prompt_text, !config.raw_prompt, n_ctx)?;
        let worker_pool = WorkerPool::for_phase_policy(
            policy,
            locked_weights.metadata(),
            prompt_tokens.len(),
            config.core_mask,
            config.threads,
            profile,
        )?;
        let before = worker_pool.verify_live_affinities()?;
        let affinity = worker_pool.affinity().clone();
        let ime_supported = bench::probe_cluster_ime(&affinity, config.cluster.as_deref(), ime_on)?;
        let effective_core_mask = format!("0x{:04X}", affinity.effective_mask);
        let thread_count = worker_pool.thread_count();
        let resolved_policy = worker_pool.phase_policy().unwrap_or(policy);
        let gen_config =
            GenerationConfig::from_tokenizer(&tokenizer).with_ignore_eos(config.ignore_eos);
        let engine = SinglePromptEngine::new()
            .with_phase_policy(resolved_policy)
            .with_generation_config(gen_config);
        (
            worker_pool,
            affinity,
            ime_supported,
            before,
            locked_weights,
            recipe,
            tokenizer.vocab_size(),
            prompt_tokens,
            effective_core_mask,
            thread_count,
            engine,
        )
    } else {
        // 1. Resolve cluster/split/core affinity and spawn pinned worker pool
        let resolved_affinity = resolve_affinity_split(
            config.cluster.as_deref(),
            config.split.as_ref().map(SplitStrategy::as_str),
            config.core_mask,
            config.threads,
        )?;
        let effective_core_mask = format!("0x{:04X}", resolved_affinity.effective_mask);
        let thread_count = resolved_affinity.threads;

        let worker_pool = WorkerPool::with_coordination(resolved_affinity.clone(), profile)?;
        let before = worker_pool.verify_live_affinities()?;
        let ime_supported =
            bench::probe_cluster_ime(&resolved_affinity, config.cluster.as_deref(), ime_on)?;

        // 2. Open model with flock(LOCK_SH), GGUF parse, capability filter, pre-flight RAM admission, and mlock
        let (locked_weights, recipe, _footprint) =
            LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;

        // 3. Extract GGUF vocabulary/template and tokenize prompt in-process
        let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
        let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
        let prompt_tokens =
            tokenize_prompt_utf8_safe(&tokenizer, prompt_text, !config.raw_prompt, n_ctx)?;
        let resolved_policy = resolve_engine_policy(
            config.split,
            recipe.architecture_id(),
            prompt_tokens.len(),
            resolved_affinity.effective_mask,
        );
        let gen_config =
            GenerationConfig::from_tokenizer(&tokenizer).with_ignore_eos(config.ignore_eos);
        (
            worker_pool,
            resolved_affinity,
            ime_supported,
            before,
            locked_weights,
            recipe,
            tokenizer.vocab_size(),
            prompt_tokens,
            effective_core_mask,
            thread_count,
            SinglePromptEngine::new()
                .with_phase_policy(resolved_policy)
                .with_generation_config(gen_config),
        )
    };

    let max_tokens = config
        .max_tokens
        .unwrap_or(128)
        .min(n_ctx.saturating_sub(prompt_tokens.len()));
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let _repack_summary = locked_weights.prepare_repacked_weights(Some(&worker_pool))?;
    let model_view = locked_weights.model_view();

    struct GgufArmOutcome {
        samples: Vec<bench::ArmSample>,
        observed_prefill_cores: u64,
        observed_decode_cores: u64,
        any_decode: bool,
    }

    let run_gguf_arm = |engine: &mut SinglePromptEngine,
                        fault: Option<usize>|
     -> Result<GgufArmOutcome, K3Error> {
        let mut samples = Vec::with_capacity(repeat_count);
        let mut observed_prefill_cores = 0_u64;
        let mut observed_decode_cores = 0_u64;
        let mut any_decode = false;
        for run_idx in 0..repeat_count {
            if is_abort_requested() {
                break;
            }
            let mut kv_arena = SingleSlotKvArena::from_gguf(&gguf_view, n_ctx)?;
            let ime_before = ime_vmadot_count();
            let mut summary = engine.evaluate_prompt_with_pool(
                recipe.as_ref(),
                &model_view,
                &mut kv_arena,
                Some(&worker_pool),
                &prompt_tokens,
                tokenizer_vocab_size,
                max_tokens,
                |_tok| {
                    if is_abort_requested() {
                        Ok(StreamAction::Stop)
                    } else {
                        Ok(StreamAction::Continue)
                    }
                },
            )?;
            let vmadot_issued = ime_vmadot_count().saturating_sub(ime_before);
            observed_prefill_cores |= engine.last_prefill_cpu_mask();
            observed_decode_cores |= engine.last_decode_cpu_mask();
            if summary.timing.decode_steps > 0 {
                any_decode = true;
            }
            bench::inject_fault_if_requested(
                fault,
                run_idx,
                tokenizer_vocab_size,
                &mut summary.generated_tokens,
            );
            samples.push(bench::ArmSample {
                prompt_tokens: summary.prompt_tokens,
                timing: summary.timing,
                generated_tokens: summary.generated_tokens,
                vmadot_issued,
            });
        }
        Ok(GgufArmOutcome {
            samples,
            observed_prefill_cores,
            observed_decode_cores,
            any_decode,
        })
    };

    // 4. Multi-Run Execution Loop via SinglePromptEngine.
    //    SinglePromptEngine/ArchRecipe do not expose an explicit IME kernel toggle,
    //    so real GGUF runs record actual vmadot deltas and omit fabricated IME-off baselines.
    let primary_kernel = if ime_on {
        MatKernel::Ime
    } else {
        MatKernel::Vector
    };
    let GgufArmOutcome {
        samples: primary_samples,
        observed_prefill_cores,
        observed_decode_cores,
        any_decode,
    } = run_gguf_arm(&mut engine, bench::fault_index())?;
    let baseline_samples = None;

    // 5. Re-verify worker affinities and actual per-phase execution masks post-execution
    let after = worker_pool.verify_live_affinities()?;
    let (plan, row_split) = bench::build_plan(
        &worker_pool,
        &affinity,
        primary_kernel,
        Some(engine.phase_policy()),
    )?;
    let observed_all_cores = after
        .iter()
        .filter(|w| w.assigned_core < 64)
        .fold(0_u64, |m, w| m | (1_u64 << w.assigned_core));
    if (observed_all_cores & !affinity.effective_mask) != 0 {
        return Err(K3Error::AffinityError {
            requested_mask: affinity.effective_mask,
            requested_threads: affinity.threads,
            reason: format!(
                "worker executed outside allowed affinity mask (observed 0x{observed_all_cores:04X} vs 0x{:04X})",
                affinity.effective_mask
            ),
        });
    }
    let (expected_prefill_mask, expected_decode_mask) = match engine.phase_policy() {
        ClusterPhasePolicy::A100Only | ClusterPhasePolicy::A100PrefillA100Decode => {
            (A100_CLUSTER_MASK, A100_CLUSTER_MASK)
        }
        ClusterPhasePolicy::A100PrefillX100Decode => (A100_CLUSTER_MASK, X100_CLUSTER_MASK),
        ClusterPhasePolicy::X100Only => (X100_CLUSTER_MASK, X100_CLUSTER_MASK),
        ClusterPhasePolicy::X100PrefillA100Decode => (X100_CLUSTER_MASK, A100_CLUSTER_MASK),
        ClusterPhasePolicy::RowSplit => {
            if !affinity.a100_cores.is_empty() {
                (A100_CLUSTER_MASK, A100_CLUSTER_MASK)
            } else {
                (X100_CLUSTER_MASK, X100_CLUSTER_MASK)
            }
        }
        ClusterPhasePolicy::Auto => (
            X100_CLUSTER_MASK | A100_CLUSTER_MASK,
            X100_CLUSTER_MASK | A100_CLUSTER_MASK,
        ),
    };
    if (observed_prefill_cores & expected_prefill_mask) == 0
        || (observed_prefill_cores & !expected_prefill_mask) != 0
    {
        return Err(K3Error::AffinityError {
            requested_mask: affinity.effective_mask,
            requested_threads: affinity.threads,
            reason: format!(
                "prefill execution mask mismatch (observed 0x{observed_prefill_cores:04X} vs expected 0x{expected_prefill_mask:04X})"
            ),
        });
    }
    if any_decode
        && ((observed_decode_cores & expected_decode_mask) == 0
            || (observed_decode_cores & !expected_decode_mask) != 0)
    {
        return Err(K3Error::AffinityError {
            requested_mask: affinity.effective_mask,
            requested_threads: affinity.threads,
            reason: format!(
                "decode execution mask mismatch (observed 0x{observed_decode_cores:04X} vs expected 0x{expected_decode_mask:04X})"
            ),
        });
    }
    let observed_cores = observed_prefill_cores | observed_decode_cores;
    let observed_matmul = observed_cores;
    let observed_vector = observed_cores;
    let exec_plan = bench::executed_phase_plan(&plan, true, any_decode);
    bench::verify_cluster_masks(&affinity, &exec_plan, observed_matmul, observed_vector)?;

    // 6. Real GGUF model metadata, memory footprint & compute/bandwidth cost-profile telemetry
    let footprint = calculate_footprint(locked_weights.metadata(), n_ctx)?;
    let total_weight_elems: u64 = locked_weights
        .metadata()
        .tensors()
        .iter()
        .map(|t| (t.num_elements() as u64).max(1))
        .sum();
    let macs_per_token = total_weight_elems.max(footprint.weight_bytes).max(1);
    let prefill_macs = (prompt_tokens.len() as u64).saturating_mul(macs_per_token);
    let decode_weight_bytes_per_token = footprint.weight_bytes.max(1);

    let seed = locked_weights
        .as_bytes()
        .iter()
        .take(1 << 20)
        .fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
    let model_name = gguf_view
        .metadata_value("general.name")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned()
        });
    let model_filename = Path::new(model_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_path)
        .to_owned();
    let architecture = gguf_view.architecture().to_owned();
    let model_identity = ModelIdentity {
        name: model_name,
        architecture: architecture.clone(),
        file_path: model_path.to_owned(),
        file_digest: format!("{seed:016x}"),
    };
    let quantization_level =
        QuantizationInfo::from_descriptors(locked_weights.metadata().tensors());
    let quant_summary = bench::quant_summary_from_info(&quantization_level);

    let (telemetry, run_metrics, statistics) =
        bench::build_telemetry_and_stats(bench::TelemetryInputs {
            config,
            pool: &worker_pool,
            affinity: &affinity,
            profile,
            prompt_tokens: &prompt_tokens,
            plan: &plan,
            row_split,
            kernel: primary_kernel,
            ime_supported,
            observed_matmul,
            observed_vector,
            before: &before,
            after: &after,
            primary_samples: &primary_samples,
            baseline_samples,
            prefill_macs,
            decode_weight_bytes_per_token,
            weight_bytes: footprint.weight_bytes,
            kv_bytes: footprint.kv_bytes,
            total_required_bytes: footprint.total_required_bytes,
            model_identity,
            quantization_level,
        })?;

    let determinism_verified = telemetry.determinism.verified;
    let first_divergent = telemetry.determinism.first_divergent_run;
    let report_stop_reason = run_metrics
        .first()
        .map(|r| r.stop_reason.clone())
        .unwrap_or_else(default_stop_reason_str);
    let report_stop_token_id = run_metrics.first().and_then(|r| r.stop_token_id);

    let report = MultiRunReport {
        execution_mode: "real_gguf".to_owned(),
        architecture,
        model_filename: Some(model_filename),
        quant_summary,
        coordination_profile: profile.as_str().to_owned(),
        repeat_count: run_metrics.len(),
        threads: thread_count,
        effective_core_mask,
        determinism_verified,
        prompt: prompt_text.to_owned(),
        runs: run_metrics,
        statistics,
        stop_reason: report_stop_reason,
        stop_token_id: report_stop_token_id,
        telemetry,
    };

    // 7. Output Formatting: JSON or unified human-readable table
    if config.json {
        let json_str =
            serde_json::to_string_pretty(&report).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "json".to_owned(),
                reason: e.to_string(),
            })?;
        let _ = writeln!(out, "{json_str}");
    } else {
        bench::write_benchmark_table(
            &mut out,
            bench::TableHeader {
                execution_mode: &report.execution_mode,
                architecture: &report.architecture,
                model_filename: report.model_filename.as_deref(),
                quant_summary: &report.quant_summary,
                coordination_profile: &report.coordination_profile,
                repeat_count: report.repeat_count,
                threads: report.threads,
                effective_core_mask: &report.effective_core_mask,
                determinism_verified: report.determinism_verified,
            },
            &report.runs,
            &report.statistics,
            &report.telemetry,
        );
    }
    let _ = out.flush();

    if let Some(run) = first_divergent {
        return Err(bench::cap_err(
            "determinism",
            format!(
                "determinism violation: run {} produced a different token sequence than run 0 at temperature 0 (repeat {repeat_count})",
                run + 1
            ),
        ));
    }

    Ok(())
}

/// Executes single-prompt standalone diagnostic evaluation pipeline with real-time streaming (US-01).
pub fn run_single_eval_session<W: Write>(config: &DiagConfig, mut out: W) -> Result<(), K3Error> {
    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --model <path>".to_owned(),
        })?;
    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(64);
    let max_tokens = config.max_tokens.unwrap_or(128);
    let profile = config.coord.unwrap_or(if config.phase_policy.is_some() {
        CoordinationProfile::Rayon
    } else {
        CoordinationProfile::Spin
    });

    let inspector = SystemMemoryInspector::new();
    let ime_on = config.ime == Some(true);
    let (worker_pool, locked_weights, recipe, mut engine) = if let Some(policy) =
        config.phase_policy
    {
        let (locked_weights, recipe, _footprint) =
            LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
        let gguf_probe = GgufFile::parse(locked_weights.as_bytes())?;
        let tokenizer_probe = GgufTokenizer::from_gguf(&gguf_probe)?;
        let probe_tokens =
            tokenize_prompt_utf8_safe(&tokenizer_probe, prompt_text, !config.raw_prompt, n_ctx)?;
        let worker_pool = WorkerPool::for_phase_policy(
            policy,
            locked_weights.metadata(),
            probe_tokens.len(),
            config.core_mask,
            config.threads,
            profile,
        )?;
        worker_pool.verify_live_affinities()?;
        let _ =
            bench::probe_cluster_ime(worker_pool.affinity(), config.cluster.as_deref(), ime_on)?;
        let resolved_policy = worker_pool.phase_policy().unwrap_or(policy);
        let engine = SinglePromptEngine::new().with_phase_policy(resolved_policy);
        (worker_pool, locked_weights, recipe, engine)
    } else {
        // 1. Resolve cluster, split & core affinity and spawn pinned worker pool (US-01.5, US-03)
        let resolved_affinity = resolve_affinity_split(
            config.cluster.as_deref(),
            config.split.as_ref().map(SplitStrategy::as_str),
            config.core_mask,
            config.threads,
        )?;
        let worker_pool = WorkerPool::with_coordination(resolved_affinity.clone(), profile)?;
        worker_pool.verify_live_affinities()?;
        let _ = bench::probe_cluster_ime(&resolved_affinity, config.cluster.as_deref(), ime_on)?;

        // 2. Open model with flock(LOCK_SH), GGUF parse, capability filter, pre-flight RAM admission, and mlock (US-01.2, US-01.3, Scenario 1 & 3)
        let (locked_weights, recipe, _footprint) =
            LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
        let phase_policy = match config.split {
            Some(SplitStrategy::Row) => ClusterPhasePolicy::RowSplit,
            _ => ClusterPhasePolicy::Auto,
        };
        (
            worker_pool,
            locked_weights,
            recipe,
            SinglePromptEngine::new().with_phase_policy(phase_policy),
        )
    };

    // 3. Extract GGUF view/vocabulary/template, allocate 16-token paged KV cache arena, and tokenize prompt in-process
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let mut kv_arena = SingleSlotKvArena::from_gguf(&gguf_view, n_ctx)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens =
        tokenize_prompt_utf8_safe(&tokenizer, prompt_text, !config.raw_prompt, n_ctx)?;
    if config.phase_policy.is_none() {
        engine.set_phase_policy(resolve_engine_policy(
            config.split,
            recipe.architecture_id(),
            prompt_tokens.len(),
            worker_pool.affinity().effective_mask,
        ));
    }
    let gen_config =
        GenerationConfig::from_tokenizer(&tokenizer).with_ignore_eos(config.ignore_eos);
    let stop_ids = gen_config.stop_token_ids.clone();
    engine.set_generation_config(gen_config);

    // 5. Execute isolated single-prompt forward pass and stream tokens unbuffered (US-01.6, US-01.7.2)
    let mut detokenizer = tokenizer.streaming_detokenizer();

    engine.evaluate_prompt_with_pool(
        recipe.as_ref(),
        &locked_weights,
        &mut kv_arena,
        Some(&worker_pool),
        &prompt_tokens,
        tokenizer.vocab_size(),
        max_tokens,
        |token_id| {
            if is_abort_requested() {
                return Ok(StreamAction::Stop);
            }
            if !config.ignore_eos && stop_ids.contains(&token_id) {
                return Ok(StreamAction::Stop);
            }

            match detokenizer.push_token(token_id)? {
                DetokenizeStep::Emit(piece) => {
                    let _ = out.write_all(piece.as_bytes());
                    let _ = out.flush();
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(prefix) => {
                    if !prefix.is_empty() {
                        let _ = out.write_all(prefix.as_bytes());
                        let _ = out.flush();
                    }
                    if config.ignore_eos {
                        detokenizer = tokenizer.streaming_detokenizer();
                        Ok(StreamAction::Continue)
                    } else {
                        Ok(StreamAction::Stop)
                    }
                }
            }
        },
    )?;

    if !detokenizer.is_stopped() && !is_abort_requested() {
        let trailing = detokenizer.finish();
        if !trailing.is_empty() {
            let _ = out.write_all(trailing.as_bytes());
            let _ = out.flush();
        }
    }

    let _ = out.write_all(b"\n");
    let _ = out.flush();

    // Re-verify worker affinities before teardown to guarantee zero cross-cluster drift during evaluation
    worker_pool.verify_live_affinities()?;

    // RAII Drop guards for `kv_arena`, `locked_weights`, and `worker_pool` execute deterministically here,
    // releasing mlock, munmap, and flock(LOCK_UN).
    Ok(())
}

/// Executes diagnostic session dispatcher: routes to differential parity exporter if
/// `--dump-parity` is specified, ephemeral OpenAI HTTP server if `--serve-openai` is specified,
/// open-source capability benchmark if `--eval-suite` is specified, synthetic matrix calibration if
/// `--synthetic-matrix` is specified, multi-run/placement GGUF benchmarking, or single-eval streaming.
pub fn run_diagnostic_session<W: Write>(config: &DiagConfig, out: W) -> Result<(), K3Error> {
    if config.dump_parity.is_some() {
        let _ = parity::run_parity_dump(config, out)?;
        Ok(())
    } else if config.serve_openai.is_some() {
        serve::run_ephemeral_openai_server(config, out)
    } else if config.is_eval_benchmark() {
        eval::run_public_eval_benchmark(config, out)
    } else if config.is_matrix_benchmark() {
        bench::run_matrix_benchmark(config, out)
    } else if config.is_multi_run_benchmark() {
        run_multi_run_benchmark(config, out)
    } else {
        run_single_eval_session(config, out)
    }
}

fn main() -> ExitCode {
    install_signal_handlers();

    let args = env::args();
    let config = match DiagConfig::parse_args(args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => return ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[error] {err}");
            eprintln!("Run 'k3-diag --help' for usage instructions.");
            return ExitCode::from(2);
        }
    };

    if !config.synthetic_matrix && config.model_path.is_none() {
        eprintln!("[error] missing required argument: --model <path> (or --synthetic-matrix)");
        eprintln!("Run 'k3-diag --help' for usage instructions.");
        return ExitCode::from(2);
    }

    let stdout = io::stdout();
    if let Err(err) = run_diagnostic_session(&config, stdout) {
        eprintln!("{err}");
        return ExitCode::from(exit_code_for_error(&err));
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::QuantType;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use std::fs::File;

    #[test]
    fn test_parse_args_valid_options() {
        let args = vec![
            "k3-diag".to_string(),
            "--model".to_string(),
            "/path/to/model.gguf".to_string(),
            "--prompt".to_string(),
            "Explain RVV 1.0".to_string(),
            "--cluster".to_string(),
            "x100".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "--cores".to_string(),
            "0x00FF".to_string(),
            "--n-ctx".to_string(),
            "4096".to_string(),
            "--max-tokens".to_string(),
            "256".to_string(),
            "--coord".to_string(),
            "futex".to_string(),
            "--repeat".to_string(),
            "5".to_string(),
            "--json".to_string(),
        ];

        let config = DiagConfig::parse_args(args.into_iter())
            .expect("parse succeeds")
            .expect("config present");

        assert_eq!(config.model_path.as_deref(), Some("/path/to/model.gguf"));
        assert_eq!(config.prompt.as_deref(), Some("Explain RVV 1.0"));
        assert_eq!(config.cluster.as_deref(), Some("x100"));
        assert_eq!(config.threads, Some(8));
        assert_eq!(config.core_mask, Some(0x00FF));
        assert_eq!(config.n_ctx, Some(4096));
        assert_eq!(config.max_tokens, Some(256));
        assert_eq!(config.coord, Some(CoordinationProfile::Futex));
        assert_eq!(config.repeat, Some(5));
        assert!(config.json);
    }

    #[test]
    fn test_parse_args_invalid_options() {
        let args = vec![
            "k3-diag".to_string(),
            "--cluster".to_string(),
            "invalid".to_string(),
        ];
        let err = DiagConfig::parse_args(args.into_iter()).unwrap_err();
        assert!(err.contains("invalid cluster"));

        let args2 = vec![
            "k3-diag".to_string(),
            "--threads".to_string(),
            "0".to_string(),
        ];
        let err2 = DiagConfig::parse_args(args2.into_iter()).unwrap_err();
        assert!(err2.contains("must be at least 1"));

        let args3 = vec![
            "k3-diag".to_string(),
            "--coord".to_string(),
            "unknown".to_string(),
        ];
        let err3 = DiagConfig::parse_args(args3.into_iter()).unwrap_err();
        assert!(err3.contains("invalid coordination profile"));

        let args4 = vec![
            "k3-diag".to_string(),
            "--repeat".to_string(),
            "2".to_string(),
        ];
        let err4 = DiagConfig::parse_args(args4.into_iter()).unwrap_err();
        assert!(err4.contains("at least 3"));
    }

    #[test]
    fn test_metric_stats_calculation() {
        let samples = [10.0, 12.0, 14.0];
        let stats = MetricStats::calculate(&samples);
        assert!((stats.mean - 12.0).abs() < 1e-9);
        assert!((stats.median - 12.0).abs() < 1e-9);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 14.0);
        assert!((stats.stddev - 2.0).abs() < 1e-9);

        let even_samples = [40.0, 10.0, 20.0, 30.0];
        let even_stats = MetricStats::calculate(&even_samples);
        assert!((even_stats.mean - 25.0).abs() < 1e-9);
        assert!((even_stats.median - 25.0).abs() < 1e-9);

        let identical = [5.0, 5.0, 5.0];
        let stats_id = MetricStats::calculate(&identical);
        assert_eq!(stats_id.mean, 5.0);
        assert_eq!(stats_id.median, 5.0);
        assert_eq!(stats_id.min, 5.0);
        assert_eq!(stats_id.max, 5.0);
        assert_eq!(stats_id.stddev, 0.0);
    }

    #[test]
    fn test_exit_codes_mapping() {
        let cap_err = K3Error::CapabilityMismatch {
            architecture: "foo".to_owned(),
            reason: "bar".to_owned(),
        };
        assert_eq!(exit_code_for_error(&cap_err), 10);

        let mem_err = K3Error::InsufficientLockableMemory {
            required_bytes: 100,
            weight_bytes: 80,
            kv_bytes: 20,
            available_bytes: 50,
            rlimit_memlock_bytes: 50,
        };
        assert_eq!(exit_code_for_error(&mem_err), 11);

        let ctx_err = K3Error::ContextLengthExceeded {
            requested_tokens: 10,
            max_context_tokens: 8,
        };
        assert_eq!(exit_code_for_error(&ctx_err), 12);

        let aff_err = K3Error::AffinityError {
            requested_mask: 0x1,
            requested_threads: 2,
            reason: "mismatch".to_owned(),
        };
        assert_eq!(exit_code_for_error(&aff_err), 13);

        let bounds_err = K3Error::TensorBoundsError {
            tensor_name: "t".to_owned(),
            offset: 0,
            byte_len: 10,
            buffer_len: 5,
            reason: "oob".to_owned(),
        };
        assert_eq!(exit_code_for_error(&bounds_err), 14);
    }

    #[test]
    fn test_run_diagnostic_session_streams_tokens_end_to_end() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let temp_path = env::temp_dir().join(format!("k3_diag_unit_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&temp_path).expect("create temp gguf");
            f.write_all(&bytes).expect("write gguf");
            f.sync_all().expect("sync gguf");
        }

        let cfg = DiagConfig {
            model_path: Some(temp_path.to_string_lossy().into_owned()),
            prompt: Some("hello world".to_owned()),
            cluster: None,
            phase_policy: None,
            split: None,
            ime: None,
            threads: Some(1),
            core_mask: Some(0x0001),
            n_ctx: Some(64),
            max_tokens: Some(4),
            coord: None,
            repeat: None,
            json: false,
            export_profile: None,
            summary: false,
            eval_suite: None,
            eval_dataset: None,
            reference_baseline: None,
            export_eval: None,
            recurrent_loops: None,
            recurrent_kv_mode: None,
            serve_openai: None,
            max_requests: None,
            synthetic_matrix: false,
            dump_parity: None,
            raw_prompt: false,
            ignore_eos: false,
        };

        let mut output_buf = Vec::new();
        run_diagnostic_session(&cfg, &mut output_buf).expect("run diagnostic session");
        let output_str = String::from_utf8(output_buf).expect("valid utf8 output");
        assert_eq!(output_str.trim(), "helloworldExplainRVV");

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_run_diagnostic_session_multi_run_benchmark_json() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let temp_path =
            env::temp_dir().join(format!("k3_diag_unit_multirun_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&temp_path).expect("create temp gguf");
            f.write_all(&bytes).expect("write gguf");
            f.sync_all().expect("sync gguf");
        }

        let cfg = DiagConfig {
            model_path: Some(temp_path.to_string_lossy().into_owned()),
            prompt: Some("hello world".to_owned()),
            cluster: None,
            phase_policy: None,
            split: None,
            ime: None,
            threads: Some(1),
            core_mask: Some(0x0001),
            n_ctx: Some(64),
            max_tokens: Some(3),
            coord: Some(CoordinationProfile::Spin),
            repeat: Some(3),
            json: true,
            export_profile: None,
            summary: false,
            eval_suite: None,
            eval_dataset: None,
            reference_baseline: None,
            export_eval: None,
            recurrent_loops: None,
            recurrent_kv_mode: None,
            serve_openai: None,
            max_requests: None,
            synthetic_matrix: false,
            dump_parity: None,
            raw_prompt: false,
            ignore_eos: false,
        };

        let mut output_buf = Vec::new();
        run_diagnostic_session(&cfg, &mut output_buf).expect("run multi-run benchmark");
        let output_str = String::from_utf8(output_buf).expect("valid utf8 output");
        let report: MultiRunReport =
            serde_json::from_str(&output_str).expect("parse multi-run report JSON");

        assert_eq!(report.execution_mode, "real_gguf");
        assert_eq!(report.architecture, "synthetic");
        assert!(report.model_filename.is_some());
        assert!(report.quant_summary.contains("Q4_K_M"));
        assert_eq!(report.coordination_profile, "spin");
        assert_eq!(report.repeat_count, 3);
        assert_eq!(report.threads, 1);
        assert!(report.determinism_verified);
        assert_eq!(report.runs.len(), 3);
        assert!(report.statistics.ttft_ms.mean >= 0.0);
        assert!(report.statistics.prefill_tokens_per_sec.mean > 0.0);
        assert!(report.statistics.prefill_tokens_per_sec.median > 0.0);
        assert!(report.runs.iter().all(|r| r.prefill_tokens_per_sec > 0.0));

        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_parse_synthetic_matrix_and_real_gguf_routing_separation() {
        // 1. Hardware flags with --model MUST NOT trigger synthetic matrix calibration.
        let real_args = vec![
            "k3-diag".to_owned(),
            "--model".to_owned(),
            "foo.gguf".to_owned(),
            "--cluster".to_owned(),
            "both".to_owned(),
            "--split".to_owned(),
            "phase".to_owned(),
            "--ime".to_owned(),
            "on".to_owned(),
            "--json".to_owned(),
        ];
        let real_cfg = DiagConfig::parse_args(real_args.into_iter())
            .expect("real gguf flags with topology parse cleanly")
            .expect("config present");
        assert!(!real_cfg.synthetic_matrix);
        assert!(!real_cfg.is_matrix_benchmark());
        assert!(real_cfg.is_multi_run_benchmark());

        // 2. Explicit --synthetic-matrix routes cleanly to matrix calibration without --model.
        let syn_args = vec![
            "k3-diag".to_owned(),
            "--synthetic-matrix".to_owned(),
            "--cluster".to_owned(),
            "both".to_owned(),
            "--split".to_owned(),
            "phase".to_owned(),
            "--ime".to_owned(),
            "on".to_owned(),
            "--json".to_owned(),
        ];
        let syn_cfg = DiagConfig::parse_args(syn_args.into_iter())
            .expect("--synthetic-matrix parses cleanly without --model")
            .expect("config present");
        assert!(syn_cfg.synthetic_matrix);
        assert!(syn_cfg.is_matrix_benchmark());
        assert!(syn_cfg.model_path.is_none());

        // 3. Reject conflicting --model + --synthetic-matrix.
        let conflict_args = vec![
            "k3-diag".to_owned(),
            "--model".to_owned(),
            "foo.gguf".to_owned(),
            "--synthetic-matrix".to_owned(),
        ];
        let err = DiagConfig::parse_args(conflict_args.into_iter())
            .expect_err("must reject ambiguous --model + --synthetic-matrix");
        assert!(err.contains("--synthetic-matrix"));
        assert!(err.contains("--model"));
    }

    #[test]
    fn test_real_gguf_vs_synthetic_matrix_provenance_json_and_headers() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // Real GGUF checkpoint execution routes through LlamaRecipe + SinglePromptEngine and
        // emits execution_mode = "real_gguf" in JSON & headers.
        let extra_kvs = [
            ("llama.block_count", GgufValue::Uint32(1)),
            ("llama.embedding_length", GgufValue::Uint32(4)),
            ("llama.attention.head_count", GgufValue::Uint32(1)),
            ("llama.attention.head_count_kv", GgufValue::Uint32(1)),
            ("llama.feed_forward_length", GgufValue::Uint32(8)),
        ];
        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[4, 16][..]),
            ("blk.0.attn_norm.weight", QuantType::F32, &[4][..]),
            ("blk.0.attn_q.weight", QuantType::F32, &[4, 4][..]),
            ("blk.0.attn_k.weight", QuantType::F32, &[4, 4][..]),
            ("blk.0.attn_v.weight", QuantType::F32, &[4, 4][..]),
            ("blk.0.attn_output.weight", QuantType::F32, &[4, 4][..]),
            ("blk.0.ffn_norm.weight", QuantType::F32, &[4][..]),
            ("blk.0.ffn_gate.weight", QuantType::F32, &[4, 8][..]),
            ("blk.0.ffn_up.weight", QuantType::F32, &[4, 8][..]),
            ("blk.0.ffn_down.weight", QuantType::F32, &[8, 4][..]),
            ("output_norm.weight", QuantType::F32, &[4][..]),
            ("output.weight", QuantType::F32, &[4, 16][..]),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("llama", 32, &extra_kvs, &tensor_specs);
        let temp_path = env::temp_dir().join(format!(
            "k3_diag_provenance_llama_{}.gguf",
            std::process::id()
        ));
        {
            let mut f = File::create(&temp_path).expect("create temp gguf");
            f.write_all(&bytes).expect("write gguf");
            f.sync_all().expect("sync gguf");
        }

        let path_str = temp_path.to_string_lossy().into_owned();
        let json_cfg = DiagConfig::parse_args(
            [
                "k3-diag",
                "--model",
                &path_str,
                "--cluster",
                "x100",
                "--cores",
                "0x0001",
                "--ime",
                "off",
                "--max-tokens",
                "2",
                "--json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();
        assert!(!json_cfg.is_matrix_benchmark());
        assert!(json_cfg.is_multi_run_benchmark());

        let mut json_buf = Vec::new();
        run_diagnostic_session(&json_cfg, &mut json_buf).expect("real gguf json session");
        let parsed: serde_json::Value =
            serde_json::from_slice(&json_buf).expect("valid JSON output");
        assert_eq!(parsed["execution_mode"].as_str(), Some("real_gguf"));
        assert_eq!(parsed["architecture"].as_str(), Some("llama"));
        assert!(
            parsed["model_filename"]
                .as_str()
                .is_some_and(|s| s.contains("k3_diag_provenance_llama_"))
        );
        assert!(
            parsed["quant_summary"]
                .as_str()
                .is_some_and(|s| s.contains("F32"))
        );

        let mut text_cfg = json_cfg.clone();
        text_cfg.json = false;
        text_cfg.summary = true;
        let mut text_buf = Vec::new();
        run_diagnostic_session(&text_cfg, &mut text_buf).expect("real gguf table session");
        let text_out = String::from_utf8(text_buf).expect("utf8 table");
        assert!(text_out.contains("Execution Mode       : real_gguf"));
        assert!(text_out.contains("Architecture         : llama"));
        assert!(text_out.contains("Model File           : k3_diag_provenance_llama_"));
        assert!(text_out.contains("Quant Summary        : F32"));

        let _ = std::fs::remove_file(&temp_path);

        // Synthetic matrix mode without --model emits execution_mode = "synthetic_matrix".
        let syn_json_cfg = DiagConfig::parse_args(
            [
                "k3-diag",
                "--synthetic-matrix",
                "--cluster",
                "x100",
                "--cores",
                "0x0001",
                "--max-tokens",
                "2",
                "--json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();
        assert!(syn_json_cfg.is_matrix_benchmark());

        let mut syn_buf = Vec::new();
        run_diagnostic_session(&syn_json_cfg, &mut syn_buf).expect("synthetic matrix json session");
        let syn_report: bench::MatrixBenchReport =
            serde_json::from_slice(&syn_buf).expect("parse MatrixBenchReport");
        assert_eq!(syn_report.execution_mode, "synthetic_matrix");
        assert_eq!(syn_report.architecture, "synthetic");
        assert!(syn_report.model_filename.is_none());
        assert_eq!(syn_report.quant_summary, bench::SYNTHETIC_QUANT_SUMMARY);

        let mut syn_text_cfg = syn_json_cfg;
        syn_text_cfg.json = false;
        let mut syn_text_buf = Vec::new();
        run_diagnostic_session(&syn_text_cfg, &mut syn_text_buf)
            .expect("synthetic matrix table session");
        let syn_text = String::from_utf8(syn_text_buf).expect("utf8 synthetic table");
        assert!(syn_text.contains("Execution Mode       : synthetic_matrix"));
        assert!(syn_text.contains("Architecture         : synthetic"));
        assert!(!syn_text.contains("Model File           :"));
    }

    #[test]
    fn test_parse_dory_recurrent_cli_flags() {
        let args = vec![
            "k3-diag".to_owned(),
            "--model".to_owned(),
            "dory.gguf".to_owned(),
            "--recurrent-loops".to_owned(),
            "4".to_owned(),
            "--recurrent-kv-mode".to_owned(),
            "last_loop".to_owned(),
        ];
        let cfg = DiagConfig::parse_args(args.into_iter())
            .expect("valid args")
            .expect("some config");
        assert_eq!(cfg.recurrent_loops, Some(4));
        assert_eq!(cfg.recurrent_kv_mode.as_deref(), Some("last_loop"));

        let bad_loops = vec![
            "k3-diag".to_owned(),
            "--recurrent-loops".to_owned(),
            "5".to_owned(),
        ];
        assert!(DiagConfig::parse_args(bad_loops.into_iter()).is_err());

        let bad_mode = vec![
            "k3-diag".to_owned(),
            "--recurrent-kv-mode".to_owned(),
            "invalid".to_owned(),
        ];
        assert!(DiagConfig::parse_args(bad_mode.into_iter()).is_err());
    }

    #[test]
    fn test_parse_certification_cli_flags() {
        let args = vec![
            "k3-diag".to_owned(),
            "--model".to_owned(),
            "llama.gguf".to_owned(),
            "--serve-openai".to_owned(),
            "127.0.0.1:0".to_owned(),
            "--max-requests".to_owned(),
            "5".to_owned(),
            "--dump-parity".to_owned(),
            "/tmp/parity.json".to_owned(),
        ];
        let cfg = DiagConfig::parse_args(args.into_iter())
            .expect("valid certification flags")
            .expect("some config");
        assert_eq!(cfg.serve_openai.as_deref(), Some("127.0.0.1:0"));
        assert_eq!(cfg.max_requests, Some(5));
        assert_eq!(cfg.dump_parity.as_deref(), Some("/tmp/parity.json"));

        let zero_req = vec![
            "k3-diag".to_owned(),
            "--max-requests".to_owned(),
            "0".to_owned(),
        ];
        assert!(DiagConfig::parse_args(zero_req.into_iter()).is_err());

        let empty_dump = vec![
            "k3-diag".to_owned(),
            "--dump-parity".to_owned(),
            "   ".to_owned(),
        ];
        assert!(DiagConfig::parse_args(empty_dump.into_iter()).is_err());
    }

    #[test]
    fn test_parse_wave2_raw_prompt_ignore_eos_and_stop_reason_serde() {
        let def = DiagConfig::default();
        assert_eq!(def.n_ctx, None);
        assert_eq!(def.max_tokens, None);
        assert!(!def.raw_prompt);
        assert!(!def.ignore_eos);

        let args = vec![
            "jero-diag".to_owned(),
            "--model".to_owned(),
            "m.gguf".to_owned(),
            "--raw-prompt".to_owned(),
            "--ignore-eos".to_owned(),
        ];
        let cfg = DiagConfig::parse_args(args.into_iter())
            .expect("valid")
            .expect("some");
        assert!(cfg.raw_prompt);
        assert!(cfg.ignore_eos);

        let metric = RunMetric {
            run_index: 0,
            ttft_ms: 1.0,
            prefill_ms: 0.5,
            prefill_tokens_per_sec: 16_000.0,
            decode_ms: 0.5,
            decode_tokens_per_sec: 100.0,
            total_wall_time_ms: 1.0,
            cumulative_barrier_wait_ms: 0.0,
            barrier_wait_ratio: 0.0,
            compute_utilization_ratio: 1.0,
            tokens: Some(vec![7, 8]),
            stop_reason: "eos_token".to_owned(),
            stop_token_id: Some(128_009),
        };
        let v = serde_json::to_value(&metric).expect("json");
        assert_eq!(v["stop_reason"], "eos_token");
        assert_eq!(v["stop_token_id"], 128_009);
        assert_eq!(v["prefill_tokens_per_sec"], 16_000.0);
    }
}
