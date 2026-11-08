//! US-03 matrix-forward benchmark: cross-cluster workload splitting and IME isolation.
//!
//! Runs the engine's own integer transformer workload ([`jero_engine::forward`]) under a chosen
//! placement (`--cluster x100|a100`, or `--cluster both --split phase|op|row`) and matrix kernel
//! (`--ime on|off`), reporting prefill and decode separately against measured hardware ceilings
//! (ADR-034). No external baseline is used (ADR-033) and no pass/fail speed threshold is applied.

use std::collections::HashMap;
use std::io::Write;

use jero_core::{
    ClusterPlacement, CoreAllocation, CostProfile, ExecutionCost, GgufFile, K3Error,
    MemoryFootprintProfile, ModelIdentity, QuantizationInfo, TuningAdvisory,
};
use jero_engine::affinity::{ClusterPreset, ResolvedAffinity, SplitStrategy, WorkerAffinityStatus};
use jero_engine::forward::{
    CalibrationShape, ForwardModel, ForwardSummary, Lane, MatKernel, PhaseLayout, PlacementPlan,
    calibrate_matmul_rate, measure_memory_bandwidth, measure_peak_macs, probe_ime_on_core,
    row_split_share,
};
use jero_engine::memory::calculate_footprint;
use jero_engine::{
    A100_CLUSTER_MASK, LockedModelWeights, SystemMemoryInspector, WorkerPool, X100_CLUSTER_MASK,
};
use jero_tokenizer::GgufTokenizer;
use serde::{Deserialize, Serialize};

use crate::{DiagConfig, MetricStats, RunMetric, StatisticsSummary, is_abort_requested};

/// Test hook: when set to a run index, that run's first generated token is perturbed so the
/// determinism check can be shown to catch a violation.
const FAULT_ENV: &str = "K3_DIAG_INJECT_NONDETERMINISM";

/// Per-phase latency, throughput, cost per token and hardware-ceiling attainment (ADR-034, US-04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PhaseReport {
    pub(crate) latency_ms: f64,
    pub(crate) tokens_per_sec: f64,
    pub(crate) cost_per_token_ms: f64,
    /// Achieved fraction of the phase's hardware ceiling in percent (informational, no threshold).
    pub(crate) ceiling_pct: f64,
    /// Ceiling definition: `peak_macs` (prefill) or `memory_bandwidth` (decode).
    pub(crate) ceiling_metric: String,
    /// Measured ceiling value (`MAC/s` or `bytes/s`).
    pub(crate) ceiling_value: f64,
}

fn measure_peak_resident_bytes() -> u64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return kb.saturating_mul(1024);
                    }
                }
            }
        }
    }
    // SAFETY: zeroed rusage struct is safely initialized for C getrusage.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: standard getrusage query
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) };
    if ret == 0 {
        #[cfg(target_os = "macos")]
        return usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        return (usage.ru_maxrss as u64).saturating_mul(1024);
    }
    0
}

/// Throughput of one IME arm, as it appears inside the comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ArmThroughput {
    pub(crate) kernel_path: String,
    pub(crate) prefill_tps: f64,
    pub(crate) decode_tps: f64,
    pub(crate) vmadot_issued: u64,
}

/// IME on versus off for the same model, prompt and placement (US-03 Scenario 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ImeComparison {
    pub(crate) on: ArmThroughput,
    pub(crate) off: ArmThroughput,
    /// `on.prefill_tps - off.prefill_tps`.
    pub(crate) delta_prefill_tps: f64,
    /// `on.decode_tps - off.decode_tps`.
    pub(crate) delta_decode_tps: f64,
    pub(crate) delta_prefill_pct: f64,
    pub(crate) delta_decode_pct: f64,
    /// Whether IME-on and IME-off generated identical token sequences (exact int32 accumulation).
    pub(crate) numerically_identical: bool,
}

/// Calibrated per-cluster matrix throughput used to size a row-split partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RowCalibration {
    pub(crate) x100_macs_per_sec: f64,
    pub(crate) a100_macs_per_sec: f64,
    /// `x100 / (x100 + a100)`: the fraction of output tiles given to `X100`.
    pub(crate) x100_share: f64,
}

/// Row-split partition derived from measured throughput (never hard-coded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RowSplitReport {
    pub(crate) prefill: RowCalibration,
    pub(crate) decode: RowCalibration,
}

/// One lane of a matrix partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LaneShare {
    pub(crate) cluster: String,
    pub(crate) share: f64,
}

/// Placement of one phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PhasePlacement {
    pub(crate) matmul: Vec<LaneShare>,
    pub(crate) vector: String,
}

/// Measured affinity of one worker thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WorkerReport {
    pub(crate) worker: usize,
    pub(crate) core: usize,
    pub(crate) cluster: String,
    pub(crate) cpus_allowed_mask: u64,
    pub(crate) cpus_allowed_list: String,
}

/// Determinism verdict over the repeated runs (temperature 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DeterminismReport {
    pub(crate) verified: bool,
    pub(crate) repeat_count: usize,
    pub(crate) token_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_divergent_run: Option<usize>,
}

/// Machine-readable US-03 report exported via `--json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MatrixBenchReport {
    pub(crate) mode: String,
    pub(crate) cluster: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) split: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) split_strategy: Option<String>,
    pub(crate) ime: String,
    pub(crate) ime_enabled: bool,
    pub(crate) kernel_path: String,
    pub(crate) ime_supported: HashMap<String, bool>,
    pub(crate) coordination_profile: String,
    pub(crate) repeat_count: usize,
    pub(crate) threads: usize,
    pub(crate) effective_core_mask: String,
    pub(crate) prompt: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) prefill: PhaseReport,
    pub(crate) decode: PhaseReport,
    pub(crate) prefill_latency_ms: f64,
    pub(crate) decode_latency_ms: f64,
    pub(crate) prefill_cost_per_token_ms: f64,
    pub(crate) decode_cost_per_token_ms: f64,
    pub(crate) prefill_tps: f64,
    pub(crate) decode_tps: f64,
    pub(crate) prefill_hardware_ceiling_pct: f64,
    pub(crate) decode_hardware_ceiling_pct: f64,
    pub(crate) total_wall_time_ms: f64,
    pub(crate) cost_profile: CostProfile,
    pub(crate) ime_vmadot_issued: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ime_delta_prefill_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ime_delta_decode_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ime_comparison: Option<ImeComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) row_split_partition_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) row_split: Option<RowSplitReport>,
    pub(crate) placement_prefill: PhasePlacement,
    pub(crate) placement_decode: PhasePlacement,
    pub(crate) matmul_cpu_mask: u64,
    pub(crate) vector_cpu_mask: u64,
    pub(crate) cluster_isolation_verified: bool,
    pub(crate) workers_before: Vec<WorkerReport>,
    pub(crate) workers_after: Vec<WorkerReport>,
    pub(crate) determinism_verified: bool,
    pub(crate) determinism: DeterminismReport,
    pub(crate) tokens: Vec<u32>,
    pub(crate) runs: Vec<RunMetric>,
    pub(crate) statistics: StatisticsSummary,
}

fn cap_err(arch: &str, reason: String) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason,
    }
}

fn lane_mask(lane: Lane) -> u64 {
    match lane {
        Lane::X100 => X100_CLUSTER_MASK,
        Lane::A100 => A100_CLUSTER_MASK,
    }
}

fn worker_reports(statuses: &[WorkerAffinityStatus]) -> Vec<WorkerReport> {
    statuses
        .iter()
        .map(|s| WorkerReport {
            worker: s.worker_index,
            core: s.assigned_core,
            cluster: if s.assigned_core < 8 { "x100" } else { "a100" }.to_owned(),
            cpus_allowed_mask: s.sched_affinity_mask,
            cpus_allowed_list: s.proc_cpus_allowed_list.clone(),
        })
        .collect()
}

fn placement_report(layout: &PhaseLayout) -> PhasePlacement {
    PhasePlacement {
        matmul: layout
            .matmul
            .iter()
            .map(|&(lane, share)| LaneShare {
                cluster: lane.as_str().to_owned(),
                share,
            })
            .collect(),
        vector: layout.vector.as_str().to_owned(),
    }
}

/// Builds the placement plan for a selection, calibrating the row split from measurement.
fn build_plan(
    pool: &WorkerPool,
    affinity: &ResolvedAffinity,
    kernel: MatKernel,
) -> Result<(PlacementPlan, Option<RowSplitReport>), K3Error> {
    match affinity.split_strategy() {
        None => {
            let lane = match affinity.cluster {
                Some(ClusterPreset::A100) => Lane::A100,
                _ => Lane::X100,
            };
            Ok((PlacementPlan::single(lane), None))
        }
        Some(SplitStrategy::Phase) => Ok((PlacementPlan::phase(Lane::A100, Lane::X100), None)),
        Some(SplitStrategy::Op) => Ok((PlacementPlan::op(Lane::A100, Lane::X100), None)),
        Some(SplitStrategy::Row) => {
            let calibrate = |shape| -> Result<RowCalibration, K3Error> {
                let x = calibrate_matmul_rate(pool, Lane::X100, kernel, shape)?;
                let a = calibrate_matmul_rate(pool, Lane::A100, kernel, shape)?;
                Ok(RowCalibration {
                    x100_macs_per_sec: x,
                    a100_macs_per_sec: a,
                    x100_share: row_split_share(x, a)?,
                })
            };
            let prefill = calibrate(CalibrationShape::Prefill)?;
            let decode = calibrate(CalibrationShape::Decode)?;
            Ok((
                PlacementPlan::row(prefill.x100_share, decode.x100_share),
                Some(RowSplitReport { prefill, decode }),
            ))
        }
    }
}

struct ArmResult {
    kernel: MatKernel,
    plan: PlacementPlan,
    row_split: Option<RowSplitReport>,
    summaries: Vec<ForwardSummary>,
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0_u32), |(s, n), v| (s + v, n + 1));
    if n == 0 { 0.0 } else { sum / f64::from(n) }
}

fn prefill_tps(s: &ForwardSummary) -> f64 {
    if s.timing.prefill_ms > 0.0 {
        s.prompt_tokens as f64 / (s.timing.prefill_ms / 1000.0)
    } else {
        0.0
    }
}

fn digest(tokens: &[u32]) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    format!("{h:016x}")
}

fn lanes_of(layout: &PhaseLayout) -> Vec<Lane> {
    layout
        .matmul
        .iter()
        .filter(|(_, share)| *share > 0.0)
        .map(|&(lane, _)| lane)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    config: &DiagConfig,
    pool: &WorkerPool,
    affinity: &ResolvedAffinity,
    model: &ForwardModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    vocab: usize,
    kernel: MatKernel,
    inject_fault: Option<usize>,
) -> Result<ArmResult, K3Error> {
    let (plan, row_split) = build_plan(pool, affinity, kernel)?;
    let repeat = config.repeat.unwrap_or(1);
    let mut summaries = Vec::with_capacity(repeat);
    for run in 0..repeat {
        if is_abort_requested() {
            break;
        }
        let mut s = model.evaluate(pool, kernel, &plan, prompt_tokens, max_tokens)?;
        if inject_fault == Some(run) {
            if let Some(first) = s.generated_tokens.first_mut() {
                *first = (*first + 1) % u32::try_from(vocab.max(1)).unwrap_or(1);
            }
        }
        summaries.push(s);
    }
    Ok(ArmResult {
        kernel,
        plan,
        row_split,
        summaries,
    })
}

/// Executes the US-03 benchmark and writes a table or a `--json` report to `out`.
///
/// # Errors
///
/// Returns a [`K3Error`] for invalid placement, unsupported IME, cluster-isolation violations or
/// (after the report is written) a determinism violation.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_matrix_benchmark<W: Write>(
    config: &DiagConfig,
    mut out: W,
) -> Result<(), K3Error> {
    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| cap_err("cli", "missing required --model <path>".to_owned()))?;
    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(64);
    let profile = config.coord.unwrap_or(jero_core::CoordinationProfile::Spin);
    let repeat = config.repeat.unwrap_or(1);

    // 1. Resolve placement; invalid cluster/split/mask combinations fail here.
    let affinity = jero_engine::affinity::resolve_affinity_split(
        config.cluster.as_deref(),
        config.split.as_ref().map(SplitStrategy::as_str),
        config.core_mask,
        config.threads,
    )?;
    let pool = WorkerPool::with_coordination(affinity.clone(), profile)?;
    let before = pool.verify_live_affinities()?;

    // 2. Probe IME on every cluster the pool owns, on that cluster's real cores.
    let mut ime_supported = HashMap::new();
    if let Some(&core) = affinity.x100_cores.first() {
        ime_supported.insert("x100".to_owned(), probe_ime_on_core(core));
    }
    if let Some(&core) = affinity.a100_cores.first() {
        ime_supported.insert("a100".to_owned(), probe_ime_on_core(core));
    }
    let ime_on = config.ime == Some(true);
    if ime_on {
        let any_ime_cluster = ime_supported.values().any(|&ok| ok);
        if !any_ime_cluster {
            let cluster_name = config.cluster.as_deref().unwrap_or("x100");
            return Err(cap_err(
                cluster_name,
                format!(
                    "cluster '{cluster_name}' does not support SpacemiT IME (XSMTVDotII / smt.vmadot) matrix acceleration; use '--cluster a100' or '--cluster both'"
                ),
            ));
        }
    }

    // 3. Model residency, tokenization.
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, _recipe, _footprint) =
        LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens = tokenizer.tokenize_prompt(prompt_text, true, n_ctx)?;
    let vocab = tokenizer.vocab_size();
    let max_tokens = config
        .max_tokens
        .unwrap_or(128)
        .min(n_ctx.saturating_sub(prompt_tokens.len()));
    let seed = locked_weights
        .as_bytes()
        .iter()
        .take(1 << 20)
        .fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
    let model = ForwardModel::new(seed, vocab);
    let fault = std::env::var(FAULT_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    // 4. Measure the primary arm, plus the IME-off baseline when IME is on.
    let primary_kernel = if ime_on {
        MatKernel::Ime
    } else {
        MatKernel::Vector
    };
    let primary = run_arm(
        config,
        &pool,
        &affinity,
        &model,
        &prompt_tokens,
        max_tokens,
        vocab,
        primary_kernel,
        fault,
    )?;
    let baseline = if ime_on {
        Some(run_arm(
            config,
            &pool,
            &affinity,
            &model,
            &prompt_tokens,
            max_tokens,
            vocab,
            MatKernel::Vector,
            None,
        )?)
    } else {
        None
    };

    // 5. Cluster isolation: observed execution CPUs and live affinity must stay in assigned lanes.
    let after = pool.verify_live_affinities()?;
    let expected_matmul: u64 = primary
        .plan
        .prefill
        .matmul
        .iter()
        .chain(&primary.plan.decode.matmul)
        .filter(|(_, s)| *s > 0.0)
        .fold(0, |m, &(lane, _)| m | lane_mask(lane));
    let expected_vector =
        lane_mask(primary.plan.prefill.vector) | lane_mask(primary.plan.decode.vector);
    let observed_matmul = primary
        .summaries
        .iter()
        .fold(0, |m, s| m | s.matmul_cpu_mask);
    let observed_vector = primary
        .summaries
        .iter()
        .fold(0, |m, s| m | s.vector_cpu_mask);
    if observed_matmul & !expected_matmul != 0 || observed_vector & !expected_vector != 0 {
        return Err(K3Error::AffinityError {
            requested_mask: affinity.effective_mask,
            requested_threads: affinity.threads,
            reason: format!(
                "work executed outside its assigned cluster (matmul cpus 0x{observed_matmul:04X} vs 0x{expected_matmul:04X}, vector cpus 0x{observed_vector:04X} vs 0x{expected_vector:04X})"
            ),
        });
    }

    // 6. Determinism across repeats at temperature 0.
    let tokens0 = primary
        .summaries
        .first()
        .map(|s| s.generated_tokens.clone())
        .unwrap_or_default();
    let divergent = primary
        .summaries
        .iter()
        .position(|s| s.generated_tokens != tokens0);
    let determinism = DeterminismReport {
        verified: divergent.is_none(),
        repeat_count: primary.summaries.len(),
        token_digest: digest(&tokens0),
        first_divergent_run: divergent,
    };

    // 7. Ceilings (ADR-034): peak MAC/s for prefill, streaming bandwidth for decode.
    let first = primary.summaries.first().ok_or_else(|| {
        cap_err(
            "cli",
            "benchmark aborted before any run completed".to_owned(),
        )
    })?;
    let mut peak_cache: HashMap<&'static str, f64> = HashMap::new();
    let mut bw_cache: HashMap<&'static str, f64> = HashMap::new();
    let mut ceilings = |layout: &PhaseLayout, kernel: MatKernel| -> Result<(f64, f64), K3Error> {
        let (mut peak, mut bw) = (0.0, 0.0);
        for lane in lanes_of(layout) {
            if !peak_cache.contains_key(lane.as_str()) {
                peak_cache.insert(lane.as_str(), measure_peak_macs(&pool, lane, kernel)?);
                bw_cache.insert(lane.as_str(), measure_memory_bandwidth(&pool, lane)?);
            }
            peak += peak_cache[lane.as_str()];
            bw += bw_cache[lane.as_str()];
        }
        Ok((peak, bw))
    };
    let (prefill_peak, _) = ceilings(&primary.plan.prefill, primary.kernel)?;
    let (_, decode_bw) = ceilings(&primary.plan.decode, primary.kernel)?;

    let prefill_tps_mean = mean(primary.summaries.iter().map(prefill_tps));
    let decode_tps_mean = mean(
        primary
            .summaries
            .iter()
            .map(|s| s.timing.decode_tokens_per_sec),
    );
    let prefill_ms = mean(primary.summaries.iter().map(|s| s.timing.prefill_ms));
    let decode_ms = mean(primary.summaries.iter().map(|s| s.timing.decode_ms));
    let prefill_macs_per_sec =
        first.prefill_macs as f64 * prefill_tps_mean / first.prompt_tokens as f64;
    let decode_bytes_per_sec = first.decode_weight_bytes_per_token as f64 * decode_tps_mean;
    let pct = |achieved: f64, ceiling: f64| {
        if ceiling > 0.0 {
            achieved / ceiling * 100.0
        } else {
            0.0
        }
    };
    let prefill_cost_per_token_ms = if prompt_tokens.is_empty() {
        0.0
    } else {
        prefill_ms / prompt_tokens.len() as f64
    };
    let mean_decode_steps = mean(
        primary
            .summaries
            .iter()
            .map(|s| s.timing.decode_steps as f64),
    );
    let decode_cost_per_token_ms = if mean_decode_steps > 0.0 {
        decode_ms / mean_decode_steps
    } else {
        0.0
    };
    let prefill = PhaseReport {
        latency_ms: prefill_ms,
        tokens_per_sec: prefill_tps_mean,
        cost_per_token_ms: prefill_cost_per_token_ms,
        ceiling_pct: pct(prefill_macs_per_sec, prefill_peak),
        ceiling_metric: "peak_macs".to_owned(),
        ceiling_value: prefill_peak,
    };
    let decode = PhaseReport {
        latency_ms: decode_ms,
        tokens_per_sec: decode_tps_mean,
        cost_per_token_ms: decode_cost_per_token_ms,
        ceiling_pct: pct(decode_bytes_per_sec, decode_bw),
        ceiling_metric: "memory_bandwidth".to_owned(),
        ceiling_value: decode_bw,
    };

    // 8. IME comparison (Scenario 2): the delta is computed from the reported arm throughputs.
    let ime_comparison = baseline.as_ref().map(|base| {
        let arm = |r: &ArmResult| ArmThroughput {
            kernel_path: r.kernel.path_name().to_owned(),
            prefill_tps: mean(r.summaries.iter().map(prefill_tps)),
            decode_tps: mean(r.summaries.iter().map(|s| s.timing.decode_tokens_per_sec)),
            vmadot_issued: r.summaries.iter().map(|s| s.vmadot_issued).sum(),
        };
        let (on, off) = (arm(&primary), arm(base));
        let pct_of = |d: f64, b: f64| if b > 0.0 { d / b * 100.0 } else { 0.0 };
        let (dp, dd) = (
            on.prefill_tps - off.prefill_tps,
            on.decode_tps - off.decode_tps,
        );
        ImeComparison {
            delta_prefill_pct: pct_of(dp, off.prefill_tps),
            delta_decode_pct: pct_of(dd, off.decode_tps),
            delta_prefill_tps: dp,
            delta_decode_tps: dd,
            numerically_identical: base.summaries.iter().all(|s| s.generated_tokens == tokens0),
            on,
            off,
        }
    });

    let run_metrics: Vec<RunMetric> = primary
        .summaries
        .iter()
        .enumerate()
        .map(|(i, s)| RunMetric {
            run_index: i,
            ttft_ms: s.timing.ttft_ms,
            prefill_ms: s.timing.prefill_ms,
            decode_ms: s.timing.decode_ms,
            decode_tokens_per_sec: s.timing.decode_tokens_per_sec,
            total_wall_time_ms: s.timing.total_wall_time_ms,
            cumulative_barrier_wait_ms: s.timing.cumulative_barrier_wait_ms,
            barrier_wait_ratio: s.timing.barrier_wait_ratio,
            compute_utilization_ratio: s.timing.compute_utilization_ratio,
            tokens: Some(s.generated_tokens.clone()),
        })
        .collect();
    let col = |f: fn(&RunMetric) -> f64| {
        MetricStats::calculate(&run_metrics.iter().map(f).collect::<Vec<_>>())
    };
    let statistics = StatisticsSummary {
        ttft_ms: col(|r| r.ttft_ms),
        decode_tokens_per_sec: col(|r| r.decode_tokens_per_sec),
        cumulative_barrier_wait_ms: col(|r| r.cumulative_barrier_wait_ms),
        total_wall_time_ms: col(|r| r.total_wall_time_ms),
        compute_utilization_ratio: col(|r| r.compute_utilization_ratio),
    };

    let cluster = config
        .cluster
        .clone()
        .unwrap_or_else(|| affinity.cluster.map_or("x100", |c| c.as_str()).to_owned());
    let split_name = config.split.map(|s| s.as_str().to_owned());

    let footprint = calculate_footprint(locked_weights.metadata(), n_ctx)?;
    let measured_rss = measure_peak_resident_bytes();
    let peak_resident_bytes = measured_rss.max(footprint.total_required_bytes);

    let model_name = gguf_view
        .metadata_value("general.name")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned()
        });

    let model_identity = ModelIdentity {
        name: model_name,
        architecture: gguf_view.architecture().to_owned(),
        file_path: model_path.to_owned(),
        file_digest: format!("{seed:016x}"),
    };

    let quantization_level =
        QuantizationInfo::from_descriptors(locked_weights.metadata().tensors());

    let split_strategy_str = split_name.clone().unwrap_or_else(|| "none".to_owned());

    let cluster_placement = ClusterPlacement {
        cluster: cluster.clone(),
        effective_core_mask: format!("0x{:04X}", affinity.effective_mask),
        allocated_cores: affinity.threads,
        x100_cores: affinity.x100_cores.len(),
        a100_cores: affinity.a100_cores.len(),
        coordination_profile: profile.as_str().to_owned(),
        split_strategy: split_strategy_str.clone(),
        ime: if ime_on { "on" } else { "off" }.to_owned(),
    };

    let allocated_cores = CoreAllocation {
        total: affinity.threads,
        total_cores: affinity.threads,
        x100_cores: affinity.x100_cores.len(),
        a100_cores: affinity.a100_cores.len(),
        effective_mask: format!("0x{:04X}", affinity.effective_mask),
    };

    let static_scheduling_weight = if decode_cost_per_token_ms > 0.0 {
        decode_cost_per_token_ms
    } else {
        prefill_cost_per_token_ms.max(1.0)
    };

    let execution_cost = ExecutionCost {
        prefill_cost_per_token_ms,
        prefill_tokens_per_sec: prefill_tps_mean,
        prefill_hardware_ceiling_pct: prefill.ceiling_pct,
        decode_cost_per_token_ms,
        decode_tokens_per_sec: decode_tps_mean,
        decode_hardware_ceiling_pct: decode.ceiling_pct,
        static_scheduling_weight,
    };

    let memory_footprint = MemoryFootprintProfile {
        weight_bytes: footprint.weight_bytes,
        kv_bytes: footprint.kv_bytes,
        total_required_bytes: footprint.total_required_bytes,
        peak_resident_bytes,
    };

    let (
        bottleneck_classification,
        recommended_cluster,
        recommended_split,
        recommended_ime,
        rationale,
        recommendations,
    ) = {
        let bottleneck = if decode.ceiling_pct >= prefill.ceiling_pct {
            "bandwidth-bound"
        } else {
            "compute-bound"
        };

        let (rec_cluster, rec_split, rec_ime) = match cluster.as_str() {
            "x100" => {
                if quantization_level.primary == "Q4_K_M" || quantization_level.primary == "Q8_0" {
                    ("both".to_owned(), Some("phase".to_owned()), true)
                } else {
                    ("a100".to_owned(), None, true)
                }
            }
            "a100" => ("a100".to_owned(), None, true),
            "both" => (
                "both".to_owned(),
                config
                    .split
                    .map(|s| s.as_str().to_owned())
                    .or_else(|| Some("phase".to_owned())),
                true,
            ),
            _ => ("both".to_owned(), Some("phase".to_owned()), true),
        };

        let rat = format!(
            "Evaluation indicates {bottleneck} execution: prefill achieved {:.1}% of theoretical compute capacity (peak_macs), while decode achieved {:.1}% of memory bandwidth capacity.",
            prefill.ceiling_pct, decode.ceiling_pct
        );

        let mut recs = Vec::new();
        if bottleneck == "bandwidth-bound" {
            recs.push("Autoregressive token decode is memory bandwidth-bound; consider low-bit quantization (e.g. Q4_K_M) to reduce per-token memory traffic.".to_owned());
        } else {
            recs.push("Prompt prefill is compute-bound; offload prefill matrix multiplies to Cluster 1 (A100) with SpacemiT IME matrix acceleration enabled.".to_owned());
        }
        if !ime_on {
            recs.push("SpacemiT IME (XSMTVDotII / smt.vmadot) matrix acceleration is currently disabled; enable '--ime on' to accelerate matrix operations.".to_owned());
        }
        if cluster == "x100" {
            recs.push("Evaluating '--cluster both --split phase' leverages A100 matrix acceleration for prefill while reserving X100 for sequential decode.".to_owned());
        }
        recs.push("Operator tuning authority (ADR-032): These recommendations are advisory only; operator configuration remains authoritative.".to_owned());

        (
            bottleneck.to_owned(),
            rec_cluster,
            rec_split,
            rec_ime,
            rat,
            recs,
        )
    };

    let tuning_advisory = TuningAdvisory {
        bottleneck_classification,
        recommended_cluster,
        recommended_split,
        recommended_ime,
        rationale,
        recommendations,
        operator_authority_notice: "Advisory recommendation only per ADR-032. Operator configuration and tuning-profile remain authoritative.".to_owned(),
    };

    let cost_profile = CostProfile {
        model_identity,
        quantization_level,
        cluster_placement,
        allocated_cores,
        split_strategy: split_strategy_str,
        execution_cost,
        prefill_cost_per_token_ms,
        prefill_tokens_per_sec: prefill_tps_mean,
        prefill_hardware_ceiling_pct: prefill.ceiling_pct,
        decode_tokens_per_sec: decode_tps_mean,
        decode_cost_per_token_ms,
        decode_hardware_ceiling_pct: decode.ceiling_pct,
        static_scheduling_weight,
        scheduling_weight: static_scheduling_weight,
        memory_footprint,
        weight_bytes: footprint.weight_bytes,
        kv_bytes: footprint.kv_bytes,
        total_required_bytes: footprint.total_required_bytes,
        peak_resident_bytes,
        tuning_advisory,
    };

    if let Some(export_path) = &config.export_profile {
        let json_str = serde_json::to_string_pretty(&cost_profile)
            .map_err(|e| cap_err("json", format!("failed to serialize cost profile: {e}")))?;
        if let Some(parent) = std::path::Path::new(export_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        std::fs::write(export_path, json_str).map_err(|e| {
            cap_err(
                "io",
                format!("failed to write cost profile to '{export_path}': {e}"),
            )
        })?;
    }

    let report = MatrixBenchReport {
        mode: "matrix-benchmark".to_owned(),
        cluster,
        split: split_name.clone(),
        split_strategy: split_name,
        ime: if ime_on { "on" } else { "off" }.to_owned(),
        ime_enabled: ime_on,
        kernel_path: primary.kernel.path_name().to_owned(),
        ime_supported,
        coordination_profile: profile.as_str().to_owned(),
        repeat_count: primary.summaries.len(),
        threads: affinity.threads,
        effective_core_mask: format!("0x{:04X}", affinity.effective_mask),
        prompt: prompt_text.to_owned(),
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: tokens0.len(),
        prefill_latency_ms: prefill.latency_ms,
        decode_latency_ms: decode.latency_ms,
        prefill_cost_per_token_ms,
        decode_cost_per_token_ms,
        prefill_tps: prefill.tokens_per_sec,
        decode_tps: decode.tokens_per_sec,
        prefill_hardware_ceiling_pct: prefill.ceiling_pct,
        decode_hardware_ceiling_pct: decode.ceiling_pct,
        total_wall_time_ms: statistics.total_wall_time_ms.mean,
        cost_profile,
        prefill,
        decode,
        ime_vmadot_issued: primary.summaries.iter().map(|s| s.vmadot_issued).sum(),
        ime_delta_prefill_tps: ime_comparison.as_ref().map(|c| c.delta_prefill_tps),
        ime_delta_decode_tps: ime_comparison.as_ref().map(|c| c.delta_decode_tps),
        ime_comparison,
        row_split_partition_ratio: primary.row_split.as_ref().map(|r| r.prefill.x100_share),
        row_split: primary.row_split.clone(),
        placement_prefill: placement_report(&primary.plan.prefill),
        placement_decode: placement_report(&primary.plan.decode),
        matmul_cpu_mask: observed_matmul,
        vector_cpu_mask: observed_vector,
        cluster_isolation_verified: true,
        workers_before: worker_reports(&before),
        workers_after: worker_reports(&after),
        determinism_verified: determinism.verified,
        determinism: determinism.clone(),
        tokens: tokens0,
        runs: run_metrics,
        statistics,
    };

    if config.json {
        let json =
            serde_json::to_string_pretty(&report).map_err(|e| cap_err("json", e.to_string()))?;
        let _ = writeln!(out, "{json}");
    } else {
        write_table(&mut out, &report);
    }
    let _ = out.flush();

    if let Some(run) = determinism.first_divergent_run {
        return Err(cap_err(
            "determinism",
            format!(
                "determinism violation: run {} produced a different token sequence than run 0 at temperature 0 (repeat {repeat})",
                run + 1
            ),
        ));
    }
    Ok(())
}

fn write_table<W: Write>(out: &mut W, r: &MatrixBenchReport) {
    let rule = "==========================================================";
    let _ = writeln!(
        out,
        "{rule}\n k3-diag: Cluster Placement & IME Benchmark\n{rule}"
    );
    let _ = writeln!(out, "Cluster              : {}", r.cluster);
    if let Some(split) = &r.split {
        let _ = writeln!(out, "Split Policy         : {split}");
    }
    let _ = writeln!(out, "IME                  : {} ({})", r.ime, r.kernel_path);
    let _ = writeln!(out, "Worker Threads       : {}", r.threads);
    let _ = writeln!(out, "Effective Core Mask  : {}", r.effective_core_mask);
    let _ = writeln!(
        out,
        "Prompt / Generated   : {} / {} tokens",
        r.prompt_tokens, r.generated_tokens
    );
    let _ = writeln!(
        out,
        "Determinism          : {}",
        if r.determinism_verified {
            "verified (bit-identical across all runs at temp 0)"
        } else {
            "VIOLATION"
        }
    );
    let _ = writeln!(
        out,
        "Prefill              : {:.3} ms, {:.1} tok/s, {:.3} ms/token, {:.1}% of {} ceiling",
        r.prefill.latency_ms,
        r.prefill.tokens_per_sec,
        r.prefill.cost_per_token_ms,
        r.prefill.ceiling_pct,
        r.prefill.ceiling_metric
    );
    let _ = writeln!(
        out,
        "Decode               : {:.3} ms, {:.1} tok/s, {:.3} ms/token, {:.1}% of {} ceiling",
        r.decode.latency_ms,
        r.decode.tokens_per_sec,
        r.decode.cost_per_token_ms,
        r.decode.ceiling_pct,
        r.decode.ceiling_metric
    );
    if let Some(rs) = &r.row_split {
        let _ = writeln!(
            out,
            "Row Split (X100 share): prefill {:.3}, decode {:.3} (from measured throughput)",
            rs.prefill.x100_share, rs.decode.x100_share
        );
    }
    if let Some(c) = &r.ime_comparison {
        let _ = writeln!(
            out,
            "IME delta (on - off) : prefill {:+.1} tok/s ({:+.1}%), decode {:+.1} tok/s ({:+.1}%)",
            c.delta_prefill_tps, c.delta_prefill_pct, c.delta_decode_tps, c.delta_decode_pct
        );
    }
    let _ = writeln!(
        out,
        "----------------------------------------------------------"
    );
    let _ = writeln!(out, "Cost Profile Baseline (US-04):");
    let _ = writeln!(
        out,
        "  Prefill Cost       : {:.3} ms/token ({:.1}% of peak_macs)",
        r.cost_profile.prefill_cost_per_token_ms, r.cost_profile.prefill_hardware_ceiling_pct
    );
    let _ = writeln!(
        out,
        "  Decode Cost        : {:.3} ms/token ({:.1}% of memory_bandwidth)",
        r.cost_profile.decode_cost_per_token_ms, r.cost_profile.decode_hardware_ceiling_pct
    );
    let _ = writeln!(
        out,
        "  Static Sched Weight: {:.3}",
        r.cost_profile.static_scheduling_weight
    );
    let _ = writeln!(
        out,
        "  Peak Resident RAM  : {} bytes (weights: {}, kv: {})",
        r.cost_profile.peak_resident_bytes, r.cost_profile.weight_bytes, r.cost_profile.kv_bytes
    );
    let _ = writeln!(
        out,
        "  Tuning Advisory    : {} (Cluster: {}, IME: {})",
        r.cost_profile.tuning_advisory.bottleneck_classification,
        r.cost_profile.tuning_advisory.recommended_cluster,
        if r.cost_profile.tuning_advisory.recommended_ime {
            "on"
        } else {
            "off"
        }
    );
    let _ = writeln!(out, "{rule}");
}
