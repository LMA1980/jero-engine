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

mod bench;
pub mod eval;

use eval::EvalSuite;

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use k3_core::{CoordinationProfile, GgufFile, K3Error};
use k3_engine::affinity::SplitStrategy;
use k3_engine::{
    LockedModelWeights, SinglePromptEngine, SingleSlotKvArena, StreamAction, SystemMemoryInspector,
    WorkerPool, resolve_affinity,
};
use k3_tokenizer::{DetokenizeStep, GgufTokenizer};
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagConfig {
    /// Path to the local model file (e.g., GGUF format).
    pub model_path: Option<String>,
    /// Input prompt to evaluate.
    pub prompt: Option<String>,
    /// Target processor cluster ("both", "x100" or "a100").
    pub cluster: Option<String>,
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
}

impl Default for DiagConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            prompt: None,
            cluster: None,
            split: None,
            ime: None,
            threads: None,
            core_mask: None,
            n_ctx: Some(64),
            max_tokens: Some(128),
            coord: None,
            repeat: None,
            json: false,
            export_profile: None,
            summary: false,
            eval_suite: None,
            eval_dataset: None,
            reference_baseline: None,
            export_eval: None,
        }
    }
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
                unknown => {
                    return Err(format!("unknown option '{unknown}'"));
                }
            }
        }

        // Cluster-boundary rules (US-03): a mixed-cluster mask is allowed only through an
        // explicit split policy, and a split policy only makes sense across both clusters.
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

    /// Returns `true` when the matrix-forward benchmark / hardware ceiling profiling should run.
    #[must_use]
    pub fn is_matrix_benchmark(&self) -> bool {
        self.split.is_some()
            || self.ime.is_some()
            || self.cluster.as_deref() == Some("both")
            || self.export_profile.is_some()
            || self.summary
            || (self.cluster.is_some() && self.json)
            || (self.cluster.is_some() && self.core_mask.is_none())
    }
}

/// Descriptive statistics for a series of continuous numerical samples ($N \ge 1$).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

impl MetricStats {
    /// Computes sample mean, min, max, and sample standard deviation with Bessel's correction ($N-1$).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                stddev: 0.0,
            };
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let stddev = if samples.len() < 2 {
            0.0
        } else {
            let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let s = variance.sqrt();
            if s.is_nan() || s < 1e-12 { 0.0 } else { s }
        };

        Self {
            mean,
            min,
            max,
            stddev,
        }
    }
}

/// Recorded execution metrics for an individual evaluation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetric {
    pub run_index: usize,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_sec: f64,
    pub total_wall_time_ms: f64,
    pub cumulative_barrier_wait_ms: f64,
    pub barrier_wait_ratio: f64,
    pub compute_utilization_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
}

/// Aggregated statistical summary for multi-run evaluations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsSummary {
    pub ttft_ms: MetricStats,
    pub decode_tokens_per_sec: MetricStats,
    pub cumulative_barrier_wait_ms: MetricStats,
    pub total_wall_time_ms: MetricStats,
    pub compute_utilization_ratio: MetricStats,
}

/// Machine-readable multi-run benchmark report exported via `--json` (US-02 R2, R3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiRunReport {
    pub coordination_profile: String,
    pub repeat_count: usize,
    pub threads: usize,
    pub effective_core_mask: String,
    pub determinism_verified: bool,
    pub prompt: String,
    pub runs: Vec<RunMetric>,
    pub statistics: StatisticsSummary,
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
               --split <policy>   Cross-cluster split for '--cluster both' ('phase', 'op', 'row')\n  \
               --ime <on|off>     SpacemiT IME matrix acceleration ('on' also measures the 'off' baseline)\n  \
           -t, --threads <n>      Number of execution worker threads\n  \
               --cores <mask>     Pinned core affinity mask (e.g. 0x00FF)\n  \
               --n-ctx <tokens>   Configured context sequence length (default: 2048)\n  \
               --max-tokens <n>   Maximum generation tokens (default: 128)\n  \
               --coord <strategy> Multi-core coordination profile ('spin', 'futex', 'rayon')\n  \
               --repeat <n>       Repeat benchmark iterations (N >= 3) for jitter measurement\n  \
               --json             Emit structured machine-readable JSON report\n  \
               --export-profile <path> Export structured CostProfile JSON to file\n  \
               --summary          Emit human-readable summary table\n  \
           -h, --help             Show this help message\n  \
           -v, --version          Show version information"
    );
}

/// Executes multi-iteration evaluation pipeline for multi-core coordination & jitter measurement (US-02).
pub fn run_multi_run_benchmark<W: Write>(config: &DiagConfig, mut out: W) -> Result<(), K3Error> {
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
    let repeat_count = config.repeat.unwrap_or(3);
    let profile = config.coord.unwrap_or(CoordinationProfile::Spin);

    // 1. Resolve cluster & core affinity and spawn pinned worker pool with selected coordination profile
    let resolved_affinity =
        resolve_affinity(config.cluster.as_deref(), config.core_mask, config.threads)?;
    let effective_core_mask = format!("0x{:04X}", resolved_affinity.effective_mask);
    let thread_count = resolved_affinity.threads;

    let worker_pool = WorkerPool::with_coordination(resolved_affinity, profile)?;
    worker_pool.verify_live_affinities()?;

    // 2. Open model with flock(LOCK_SH), GGUF parse, capability filter, pre-flight RAM admission, and mlock
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _footprint) =
        LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;

    // 3. Extract GGUF vocabulary/template and tokenize prompt in-process
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens = tokenizer.tokenize_prompt(prompt_text, true, n_ctx)?;

    // 4. Multi-Run Execution Loop
    let mut run_metrics: Vec<RunMetric> = Vec::with_capacity(repeat_count);
    let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(repeat_count);
    let mut engine = SinglePromptEngine::new();

    for run_idx in 0..repeat_count {
        if is_abort_requested() {
            break;
        }

        let mut kv_arena = SingleSlotKvArena::new(locked_weights.metadata(), n_ctx)?;
        let summary = engine.evaluate_prompt_with_pool(
            recipe,
            &locked_weights,
            &mut kv_arena,
            Some(&worker_pool),
            &prompt_tokens,
            tokenizer.vocab_size(),
            max_tokens,
            |_tok| {
                if is_abort_requested() {
                    Ok(StreamAction::Stop)
                } else {
                    Ok(StreamAction::Continue)
                }
            },
        )?;

        let run_metric = RunMetric {
            run_index: run_idx,
            ttft_ms: summary.timing.ttft_ms,
            prefill_ms: summary.timing.prefill_ms,
            decode_ms: summary.timing.decode_ms,
            decode_tokens_per_sec: summary.timing.decode_tokens_per_sec,
            total_wall_time_ms: summary.timing.total_wall_time_ms,
            cumulative_barrier_wait_ms: summary.timing.cumulative_barrier_wait_ms,
            barrier_wait_ratio: summary.timing.barrier_wait_ratio,
            compute_utilization_ratio: summary.timing.compute_utilization_ratio,
            tokens: Some(summary.generated_tokens.clone()),
        };

        all_tokens.push(summary.generated_tokens);
        run_metrics.push(run_metric);
    }

    // 5. Re-verify worker affinities post-execution to guarantee zero cross-cluster migration
    worker_pool.verify_live_affinities()?;

    // 6. Assert Output Determinism: Bit-identical tokens across repeat runs at temp 0
    let mut determinism_verified = true;
    if all_tokens.len() >= 2 {
        for i in 1..all_tokens.len() {
            if all_tokens[i] != all_tokens[0] {
                determinism_verified = false;
                break;
            }
        }
    }

    // 7. Calculate Descriptive Statistics
    let ttft_samples: Vec<f64> = run_metrics.iter().map(|r| r.ttft_ms).collect();
    let throughput_samples: Vec<f64> = run_metrics
        .iter()
        .map(|r| r.decode_tokens_per_sec)
        .collect();
    let barrier_samples: Vec<f64> = run_metrics
        .iter()
        .map(|r| r.cumulative_barrier_wait_ms)
        .collect();
    let wall_time_samples: Vec<f64> = run_metrics.iter().map(|r| r.total_wall_time_ms).collect();
    let util_samples: Vec<f64> = run_metrics
        .iter()
        .map(|r| r.compute_utilization_ratio)
        .collect();

    let statistics = StatisticsSummary {
        ttft_ms: MetricStats::calculate(&ttft_samples),
        decode_tokens_per_sec: MetricStats::calculate(&throughput_samples),
        cumulative_barrier_wait_ms: MetricStats::calculate(&barrier_samples),
        total_wall_time_ms: MetricStats::calculate(&wall_time_samples),
        compute_utilization_ratio: MetricStats::calculate(&util_samples),
    };

    // 8. Output Formatting: JSON or Table
    if config.json {
        let report = MultiRunReport {
            coordination_profile: profile.as_str().to_owned(),
            repeat_count: run_metrics.len(),
            threads: thread_count,
            effective_core_mask,
            determinism_verified,
            prompt: prompt_text.to_owned(),
            runs: run_metrics,
            statistics,
        };
        let json_str =
            serde_json::to_string_pretty(&report).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "json".to_owned(),
                reason: e.to_string(),
            })?;
        let _ = writeln!(out, "{json_str}");
    } else {
        let _ = writeln!(
            out,
            "=========================================================="
        );
        let _ = writeln!(
            out,
            " k3-diag: Multi-Core Synchronization & Jitter Benchmark"
        );
        let _ = writeln!(
            out,
            "=========================================================="
        );
        let _ = writeln!(out, "Coordination Profile : {}", profile.as_str());
        let _ = writeln!(out, "Repeat Iterations    : {}", run_metrics.len());
        let _ = writeln!(out, "Worker Threads       : {thread_count}");
        let _ = writeln!(out, "Effective Core Mask  : {effective_core_mask}");
        let _ = writeln!(
            out,
            "Output Determinism   : {}",
            if determinism_verified {
                "verified (bit-identical across all runs at temp 0)"
            } else {
                "MISMATCH"
            }
        );
        let _ = writeln!(
            out,
            "----------------------------------------------------------"
        );
        let _ = writeln!(out, "Iteration Breakdown:");
        let _ = writeln!(
            out,
            "  Run | TTFT (ms) | Decode (tokens/s) | Barrier Wait (ms) | Total Wall (ms)"
        );
        for r in &run_metrics {
            let _ = writeln!(
                out,
                "  {:3} | {:9.2} | {:17.2} | {:17.2} | {:15.2}",
                r.run_index + 1,
                r.ttft_ms,
                r.decode_tokens_per_sec,
                r.cumulative_barrier_wait_ms,
                r.total_wall_time_ms
            );
        }
        let _ = writeln!(
            out,
            "----------------------------------------------------------"
        );
        let _ = writeln!(out, "Statistical Jitter Summary (Mean, Min, Max, StdDev):");
        let _ = writeln!(
            out,
            "  Metric                      |      Mean |       Min |       Max |    StdDev"
        );
        let _ = writeln!(
            out,
            "  TTFT (Time to First Token)  | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
            statistics.ttft_ms.mean,
            statistics.ttft_ms.min,
            statistics.ttft_ms.max,
            statistics.ttft_ms.stddev
        );
        let _ = writeln!(
            out,
            "  Decode Throughput (tokens/s)| {:9.2} | {:9.2} | {:9.2} | {:9.2}",
            statistics.decode_tokens_per_sec.mean,
            statistics.decode_tokens_per_sec.min,
            statistics.decode_tokens_per_sec.max,
            statistics.decode_tokens_per_sec.stddev
        );
        let _ = writeln!(
            out,
            "  Barrier Wait Overhead (ms)  | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
            statistics.cumulative_barrier_wait_ms.mean,
            statistics.cumulative_barrier_wait_ms.min,
            statistics.cumulative_barrier_wait_ms.max,
            statistics.cumulative_barrier_wait_ms.stddev
        );
        let _ = writeln!(
            out,
            "  Total Wall Time (ms)        | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
            statistics.total_wall_time_ms.mean,
            statistics.total_wall_time_ms.min,
            statistics.total_wall_time_ms.max,
            statistics.total_wall_time_ms.stddev
        );
        let _ = writeln!(
            out,
            "  Compute Utilization Ratio   | {:9.4} | {:9.4} | {:9.4} | {:9.4}",
            statistics.compute_utilization_ratio.mean,
            statistics.compute_utilization_ratio.min,
            statistics.compute_utilization_ratio.max,
            statistics.compute_utilization_ratio.stddev
        );
        let _ = writeln!(
            out,
            "=========================================================="
        );
    }
    let _ = out.flush();

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
    let profile = config.coord.unwrap_or(CoordinationProfile::Spin);

    // 1. Resolve cluster & core affinity and spawn pinned worker pool (US-01.5, Scenario 2)
    let resolved_affinity =
        resolve_affinity(config.cluster.as_deref(), config.core_mask, config.threads)?;
    let worker_pool = WorkerPool::with_coordination(resolved_affinity, profile)?;
    worker_pool.verify_live_affinities()?;

    // 2. Open model with flock(LOCK_SH), GGUF parse, capability filter, pre-flight RAM admission, and mlock (US-01.2, US-01.3, Scenario 1 & 3)
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _footprint) =
        LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;

    // 3. Allocate and mlock single-slot 16-token paged KV cache arena (US-01.3.3)
    let mut kv_arena = SingleSlotKvArena::new(locked_weights.metadata(), n_ctx)?;

    // 4. Extract GGUF vocabulary/template and tokenize prompt in-process (US-01.4)
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens = tokenizer.tokenize_prompt(prompt_text, true, n_ctx)?;

    // 5. Execute isolated single-prompt forward pass and stream tokens unbuffered (US-01.6, US-01.7.2)
    let mut engine = SinglePromptEngine::new();
    let mut detokenizer = tokenizer.streaming_detokenizer();

    engine.evaluate_prompt_with_pool(
        recipe,
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
                    Ok(StreamAction::Stop)
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

/// Executes diagnostic session dispatcher: routes to open-source capability benchmark if
/// `--eval-suite` is specified, matrix-forward benchmark, multi-run benchmark if `--repeat` is
/// specified, or single-eval streaming pipeline.
pub fn run_diagnostic_session<W: Write>(config: &DiagConfig, out: W) -> Result<(), K3Error> {
    if config.is_eval_benchmark() {
        eval::run_public_eval_benchmark(config, out)
    } else if config.is_matrix_benchmark() {
        bench::run_matrix_benchmark(config, out)
    } else if config.repeat.is_some() {
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

    if config.model_path.is_none() {
        eprintln!("[error] missing required argument: --model <path>");
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
    use k3_core::QuantType;
    use k3_core::test_helpers::create_synthetic_gguf_v3;
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
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 14.0);
        assert!((stats.stddev - 2.0).abs() < 1e-9);

        let identical = [5.0, 5.0, 5.0];
        let stats_id = MetricStats::calculate(&identical);
        assert_eq!(stats_id.mean, 5.0);
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
        };

        let mut output_buf = Vec::new();
        run_diagnostic_session(&cfg, &mut output_buf).expect("run multi-run benchmark");
        let output_str = String::from_utf8(output_buf).expect("valid utf8 output");
        let report: MultiRunReport =
            serde_json::from_str(&output_str).expect("parse multi-run report JSON");

        assert_eq!(report.coordination_profile, "spin");
        assert_eq!(report.repeat_count, 3);
        assert_eq!(report.threads, 1);
        assert!(report.determinism_verified);
        assert_eq!(report.runs.len(), 3);
        assert!(report.statistics.ttft_ms.mean >= 0.0);

        let _ = std::fs::remove_file(&temp_path);
    }
}
