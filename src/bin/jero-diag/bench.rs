//! US-03 matrix-forward benchmark: cross-cluster workload splitting and IME isolation.
//!
//! Runs the engine's own integer transformer workload ([`jero_engine::forward`]) under `--synthetic-matrix`
//! with a chosen placement (`--cluster x100|a100`, or `--cluster both --split phase|op|row`) and matrix
//! kernel (`--ime on|off`), reporting prefill and decode separately against measured hardware ceilings
//! (ADR-034). No external baseline is used (ADR-033) and no pass/fail speed threshold is applied.
//! Also provides shared placement, ceiling, cost-profile, determinism, and formatting helpers used
//! by real GGUF execution in [`crate::run_multi_run_benchmark`].

use std::collections::HashMap;
use std::io::Write;

use jero_core::{
    ClusterPlacement, CoordinationProfile, CoreAllocation, CostProfile, ExecutionCost, K3Error,
    MemoryFootprintProfile, ModelIdentity, QuantizationInfo, TuningAdvisory,
};
use jero_engine::affinity::{ClusterPreset, ResolvedAffinity, SplitStrategy, WorkerAffinityStatus};
use jero_engine::engine::PassTiming;
use jero_engine::forward::{
    CalibrationShape, ForwardModel, ForwardSummary, Lane, MatKernel, PhaseLayout, PlacementPlan,
    calibrate_bandwidth_split_ratio, calibrate_matmul_rate, measure_memory_bandwidth,
    measure_peak_macs, probe_ime_on_core, row_split_share,
};
use jero_engine::{A100_CLUSTER_MASK, ClusterPhasePolicy, WorkerPool, X100_CLUSTER_MASK};
use serde::{Deserialize, Serialize};

use crate::{DiagConfig, MetricStats, RunMetric, StatisticsSummary, is_abort_requested};

/// Test hook: when set to a run index, that run's first generated token is perturbed so the
/// determinism check can be shown to catch a violation.
pub const FAULT_ENV: &str = "K3_DIAG_INJECT_NONDETERMINISM";

/// Default synthetic-matrix quantization summary string.
pub const SYNTHETIC_QUANT_SUMMARY: &str = "synthetic-matrix (in-memory I8/F32)";

/// Per-phase latency, throughput, cost per token and hardware-ceiling attainment (ADR-034, US-04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseReport {
    /// Mean phase duration in milliseconds.
    pub latency_ms: f64,
    /// Mean phase throughput in tokens per second.
    pub tokens_per_sec: f64,
    /// Mean compute cost per token in milliseconds.
    pub cost_per_token_ms: f64,
    /// Achieved fraction of the phase's hardware ceiling in percent (informational, no threshold).
    pub ceiling_pct: f64,
    /// Ceiling definition: `peak_macs` (prefill) or `memory_bandwidth` (decode).
    pub ceiling_metric: String,
    /// Measured ceiling value (`MAC/s` or `bytes/s`).
    pub ceiling_value: f64,
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
pub struct ArmThroughput {
    /// Executed kernel path identifier (e.g. `ime` or `rvv_vector`).
    pub kernel_path: String,
    /// Mean prefill throughput in tokens per second.
    pub prefill_tps: f64,
    /// Mean decode throughput in tokens per second.
    pub decode_tps: f64,
    /// Number of IME `smt.vmadot` instructions observed.
    pub vmadot_issued: u64,
}

/// IME on versus off for the same model, prompt and placement (US-03 Scenario 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImeComparison {
    /// Metrics with IME acceleration enabled.
    pub on: ArmThroughput,
    /// Metrics with IME acceleration disabled (`--ime off` baseline).
    pub off: ArmThroughput,
    /// `on.prefill_tps - off.prefill_tps`.
    pub delta_prefill_tps: f64,
    /// `on.decode_tps - off.decode_tps`.
    pub delta_decode_tps: f64,
    /// Relative prefill throughput change percentage.
    pub delta_prefill_pct: f64,
    /// Relative decode throughput change percentage.
    pub delta_decode_pct: f64,
    /// Whether IME-on and IME-off generated identical token sequences (exact int32 accumulation).
    pub numerically_identical: bool,
}

/// Calibrated per-cluster matrix throughput used to size a row-split partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowCalibration {
    /// Measured Cluster 0 (`X100`) matrix multiply throughput in MACs/s.
    pub x100_macs_per_sec: f64,
    /// Measured Cluster 1 (`A100`) matrix multiply throughput in MACs/s.
    pub a100_macs_per_sec: f64,
    /// `x100 / (x100 + a100)`: the fraction of output tiles given to `X100`.
    pub x100_share: f64,
}

/// Row-split partition derived from measured throughput (never hard-coded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowSplitReport {
    /// Calibrated partition for prompt prefill.
    pub prefill: RowCalibration,
    /// Calibrated partition for autoregressive decode.
    pub decode: RowCalibration,
}

/// One lane of a matrix partition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneShare {
    /// Cluster lane identifier (`x100` or `a100`).
    pub cluster: String,
    /// Fraction of output tiles assigned to this lane.
    pub share: f64,
}

/// Placement of one phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhasePlacement {
    /// Matrix multiplication cluster assignment and shares.
    pub matmul: Vec<LaneShare>,
    /// Cluster executing vector/normalization/attention ops.
    pub vector: String,
}

/// Measured affinity of one worker thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerReport {
    /// Worker index within the pool.
    pub worker: usize,
    /// Assigned physical core ID.
    pub core: usize,
    /// Cluster owning the assigned core (`x100` or `a100`).
    pub cluster: String,
    /// OS `sched_getaffinity` bitmask.
    pub cpus_allowed_mask: u64,
    /// Human-readable `/proc/self/status` `Cpus_allowed_list`.
    pub cpus_allowed_list: String,
}

/// Determinism verdict over the repeated runs (temperature 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterminismReport {
    /// Whether every repeat produced identical output tokens.
    pub verified: bool,
    /// Total number of runs compared.
    pub repeat_count: usize,
    /// 64-bit FNV-1a digest of run 0 tokens.
    pub token_digest: String,
    /// Zero-based index of the first run that diverged from run 0, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergent_run: Option<usize>,
}

/// Shared placement, ceiling, cost-profile, and determinism telemetry for both real-GGUF
/// multi-run benchmarks and `--synthetic-matrix` hardware calibration reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkTelemetry {
    /// Resolved target cluster string (`x100`, `a100`, or `both`).
    pub cluster: String,
    /// Optional split strategy (`phase`, `op`, or `row`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    /// Alias for `split` consumed by downstream US-03 schemas.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_strategy: Option<String>,
    /// Requested IME mode (`on` or `off`).
    pub ime: String,
    /// Boolean flag indicating whether IME acceleration was enabled.
    pub ime_enabled: bool,
    /// Active matrix kernel path name.
    pub kernel_path: String,
    /// Probed IME hardware support per participating cluster.
    pub ime_supported: HashMap<String, bool>,
    /// Number of tokenized prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated output tokens in run 0.
    pub generated_tokens: usize,
    /// Prefill phase metrics and compute ceiling attainment.
    pub prefill: PhaseReport,
    /// Decode phase metrics and memory-bandwidth ceiling attainment.
    pub decode: PhaseReport,
    /// Mean prefill latency in milliseconds.
    pub prefill_latency_ms: f64,
    /// Mean decode latency in milliseconds.
    pub decode_latency_ms: f64,
    /// Mean prefill cost per token in milliseconds.
    pub prefill_cost_per_token_ms: f64,
    /// Mean decode cost per token in milliseconds.
    pub decode_cost_per_token_ms: f64,
    /// Mean prefill throughput in tokens per second.
    pub prefill_tps: f64,
    /// Mean decode throughput in tokens per second.
    pub decode_tps: f64,
    /// Mean prefill hardware ceiling percentage (`peak_macs`).
    pub prefill_hardware_ceiling_pct: f64,
    /// Mean decode hardware ceiling percentage (`memory_bandwidth`).
    pub decode_hardware_ceiling_pct: f64,
    /// Mean total wall-clock pass duration in milliseconds.
    pub total_wall_time_ms: f64,
    /// Structured US-04 hardware cost baseline and tuning advisory.
    pub cost_profile: CostProfile,
    /// Measured `smt.vmadot` instructions executed during the primary arm.
    pub ime_vmadot_issued: u64,
    /// Prefill throughput difference (`ime on - ime off`), if `--ime on`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ime_delta_prefill_tps: Option<f64>,
    /// Decode throughput difference (`ime on - ime off`), if `--ime on`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ime_delta_decode_tps: Option<f64>,
    /// Detailed IME on/off arm comparison, if `--ime on`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ime_comparison: Option<ImeComparison>,
    /// Measured `X100` prefill output-tile share, if `--split row`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_split_partition_ratio: Option<f64>,
    /// Calibrated row-split parameters for prefill and decode, if `--split row`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_split: Option<RowSplitReport>,
    /// Effective placement during prompt prefill.
    pub placement_prefill: PhasePlacement,
    /// Effective placement during autoregressive decode.
    pub placement_decode: PhasePlacement,
    /// Union CPU core mask observed/verified for matrix operations.
    pub matmul_cpu_mask: u64,
    /// Union CPU core mask observed/verified for vector/norm operations.
    pub vector_cpu_mask: u64,
    /// True when pre-run and post-run OS affinities strictly match the requested clusters.
    pub cluster_isolation_verified: bool,
    /// Worker affinities measured prior to benchmark execution.
    pub workers_before: Vec<WorkerReport>,
    /// Worker affinities measured after benchmark execution.
    pub workers_after: Vec<WorkerReport>,
    /// Structured determinism audit across repeats.
    pub determinism: DeterminismReport,
    /// Output token IDs generated on run 0.
    pub tokens: Vec<u32>,
}

/// Machine-readable US-03 `--synthetic-matrix` calibration report exported via `--json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixBenchReport {
    /// Execution mode discriminator (`"synthetic_matrix"`).
    pub execution_mode: String,
    /// Model architecture identifier (`"synthetic"`).
    pub architecture: String,
    /// Optional model filename (`None` for `--synthetic-matrix`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_filename: Option<String>,
    /// Quantization summary string.
    pub quant_summary: String,
    /// Benchmark mode identifier (`"matrix-benchmark"`).
    pub mode: String,
    /// Active coordination profile (`spin`, `futex`, `rayon`).
    pub coordination_profile: String,
    /// Number of benchmark iterations completed.
    pub repeat_count: usize,
    /// Allocated worker thread count.
    pub threads: usize,
    /// Hex bitmask of allowed cores.
    pub effective_core_mask: String,
    /// Whether every repeat produced identical output tokens.
    pub determinism_verified: bool,
    /// Evaluated prompt text.
    pub prompt: String,
    /// Per-iteration execution telemetry.
    pub runs: Vec<RunMetric>,
    /// Aggregate mean/min/max/stddev statistics across repeats.
    pub statistics: StatisticsSummary,
    /// Flattened shared placement, ceiling, cost-profile, and determinism fields.
    #[serde(flatten)]
    pub telemetry: BenchmarkTelemetry,
}

/// Single run outcome normalized across both real GGUF evaluation and synthetic matrix calibration.
#[derive(Debug, Clone)]
pub(crate) struct ArmSample {
    pub(crate) prompt_tokens: usize,
    pub(crate) timing: PassTiming,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) vmadot_issued: u64,
}

/// Borrowed header metadata for rendering unified human-readable CLI benchmark tables.
pub(crate) struct TableHeader<'a> {
    pub(crate) execution_mode: &'a str,
    pub(crate) architecture: &'a str,
    pub(crate) model_filename: Option<&'a str>,
    pub(crate) quant_summary: &'a str,
    pub(crate) coordination_profile: &'a str,
    pub(crate) repeat_count: usize,
    pub(crate) threads: usize,
    pub(crate) effective_core_mask: &'a str,
    pub(crate) determinism_verified: bool,
}

/// Inputs for assembling shared [`BenchmarkTelemetry`], [`RunMetric`]s, and [`StatisticsSummary`].
pub(crate) struct TelemetryInputs<'a> {
    pub(crate) config: &'a DiagConfig,
    pub(crate) pool: &'a WorkerPool,
    pub(crate) affinity: &'a ResolvedAffinity,
    pub(crate) profile: CoordinationProfile,
    pub(crate) prompt_tokens: &'a [u32],
    pub(crate) plan: &'a PlacementPlan,
    pub(crate) row_split: Option<RowSplitReport>,
    pub(crate) kernel: MatKernel,
    pub(crate) ime_supported: HashMap<String, bool>,
    pub(crate) observed_matmul: u64,
    pub(crate) observed_vector: u64,
    pub(crate) before: &'a [WorkerAffinityStatus],
    pub(crate) after: &'a [WorkerAffinityStatus],
    pub(crate) primary_samples: &'a [ArmSample],
    pub(crate) baseline_samples: Option<(MatKernel, Vec<ArmSample>)>,
    pub(crate) prefill_macs: u64,
    pub(crate) decode_weight_bytes_per_token: u64,
    pub(crate) weight_bytes: u64,
    pub(crate) kv_bytes: u64,
    pub(crate) total_required_bytes: u64,
    pub(crate) model_identity: ModelIdentity,
    pub(crate) quantization_level: QuantizationInfo,
}

pub(crate) fn cap_err(arch: &str, reason: String) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason,
    }
}

pub(crate) fn lane_mask(lane: Lane) -> u64 {
    match lane {
        Lane::X100 => X100_CLUSTER_MASK,
        Lane::A100 => A100_CLUSTER_MASK,
    }
}

/// Formats a concise human-readable quantization summary string from [`QuantizationInfo`].
#[must_use]
pub fn quant_summary_from_info(info: &QuantizationInfo) -> String {
    if info.summary.is_empty() {
        return "none".to_owned();
    }
    let breakdown: Vec<String> = info
        .summary
        .iter()
        .map(|(q, count)| format!("{q}:{count}"))
        .collect();
    format!("{} ({})", info.primary, breakdown.join(", "))
}

pub(crate) fn worker_reports(statuses: &[WorkerAffinityStatus]) -> Vec<WorkerReport> {
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

pub(crate) fn placement_report(layout: &PhaseLayout) -> PhasePlacement {
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

/// Probes IME availability across the clusters owned by `affinity` and rejects `--ime on`
/// when no assigned cluster supports SpacemiT IME (`XSMTVDotII`).
pub(crate) fn probe_cluster_ime(
    affinity: &ResolvedAffinity,
    cluster_arg: Option<&str>,
    ime_on: bool,
) -> Result<HashMap<String, bool>, K3Error> {
    let mut ime_supported = HashMap::new();
    if let Some(&core) = affinity.x100_cores.first() {
        ime_supported.insert("x100".to_owned(), probe_ime_on_core(core));
    }
    if let Some(&core) = affinity.a100_cores.first() {
        ime_supported.insert("a100".to_owned(), probe_ime_on_core(core));
    }
    if ime_on && !ime_supported.values().any(|&ok| ok) {
        let cluster_name = cluster_arg.unwrap_or("x100");
        return Err(cap_err(
            cluster_name,
            format!(
                "cluster '{cluster_name}' does not support SpacemiT IME (XSMTVDotII / smt.vmadot) matrix acceleration; use '--cluster a100' or '--cluster both'"
            ),
        ));
    }
    Ok(ime_supported)
}

/// Reads optional `K3_DIAG_INJECT_NONDETERMINISM` test hook index.
pub(crate) fn fault_index() -> Option<usize> {
    std::env::var(FAULT_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// Perturbs the first generated token on `fault == Some(run)` to test determinism checks.
pub(crate) fn inject_fault_if_requested(
    fault: Option<usize>,
    run: usize,
    vocab: usize,
    generated_tokens: &mut [u32],
) {
    if fault == Some(run) {
        if let Some(first) = generated_tokens.first_mut() {
            *first = (*first + 1) % u32::try_from(vocab.max(1)).unwrap_or(1);
        }
    }
}

/// Builds the placement plan for a selection, calibrating the row split from measurement.
pub(crate) fn build_plan(
    pool: &WorkerPool,
    affinity: &ResolvedAffinity,
    kernel: MatKernel,
    phase_policy: Option<ClusterPhasePolicy>,
) -> Result<(PlacementPlan, Option<RowSplitReport>), K3Error> {
    let default_single_lane = if affinity.x100_cores.is_empty() && !affinity.a100_cores.is_empty() {
        Lane::A100
    } else {
        match affinity.cluster {
            Some(ClusterPreset::A100) => Lane::A100,
            _ => Lane::X100,
        }
    };

    if let Some(policy) = phase_policy {
        let has_both = !affinity.x100_cores.is_empty() && !affinity.a100_cores.is_empty();
        return match policy {
            ClusterPhasePolicy::X100Only => Ok((PlacementPlan::single(Lane::X100), None)),
            ClusterPhasePolicy::A100Only | ClusterPhasePolicy::A100PrefillA100Decode => {
                Ok((PlacementPlan::single(Lane::A100), None))
            }
            ClusterPhasePolicy::A100PrefillX100Decode if has_both => {
                Ok((PlacementPlan::phase(Lane::A100, Lane::X100), None))
            }
            ClusterPhasePolicy::X100PrefillA100Decode if has_both => {
                Ok((PlacementPlan::phase(Lane::X100, Lane::A100), None))
            }
            ClusterPhasePolicy::RowSplit if has_both => calibrate_row_plan(pool, kernel),
            ClusterPhasePolicy::Auto if has_both => {
                if jero_engine::affinity::has_a100_ime_for_mask(affinity.effective_mask) {
                    Ok((PlacementPlan::single(Lane::A100), None))
                } else {
                    Ok((PlacementPlan::single(Lane::X100), None))
                }
            }
            _ => Ok((PlacementPlan::single(default_single_lane), None)),
        };
    }

    match affinity.split_strategy() {
        None => Ok((PlacementPlan::single(default_single_lane), None)),
        Some(SplitStrategy::Phase) => Ok((PlacementPlan::phase(Lane::A100, Lane::X100), None)),
        Some(SplitStrategy::Op) => Ok((PlacementPlan::op(Lane::A100, Lane::X100), None)),
        Some(SplitStrategy::Row) => calibrate_row_plan(pool, kernel),
    }
}

fn calibrate_row_plan(
    pool: &WorkerPool,
    kernel: MatKernel,
) -> Result<(PlacementPlan, Option<RowSplitReport>), K3Error> {
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
    let mut decode = calibrate(CalibrationShape::Decode)?;
    let (_, _, bw_share) = calibrate_bandwidth_split_ratio(pool)?;
    decode.x100_share = bw_share;
    Ok((
        PlacementPlan::row(prefill.x100_share, decode.x100_share),
        Some(RowSplitReport { prefill, decode }),
    ))
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0_u32), |(s, n), v| (s + v, n + 1));
    if n == 0 { 0.0 } else { sum / f64::from(n) }
}

fn sample_prefill_tps(s: &ArmSample) -> f64 {
    let tps = s.timing.prefill_tokens_per_sec();
    if tps > 0.0 {
        tps
    } else if s.timing.prefill_ms > 0.0 && s.prompt_tokens > 0 {
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

pub(crate) fn expected_phase_masks(plan: &PlacementPlan) -> (u64, u64) {
    let expected_matmul = plan
        .prefill
        .matmul
        .iter()
        .chain(&plan.decode.matmul)
        .filter(|(_, s)| *s > 0.0)
        .fold(0, |m, &(lane, _)| m | lane_mask(lane));
    let expected_vector = lane_mask(plan.prefill.vector) | lane_mask(plan.decode.vector);
    (expected_matmul, expected_vector)
}

pub(crate) fn executed_phase_plan(
    plan: &PlacementPlan,
    single_cluster_phases: bool,
    has_decode: bool,
) -> PlacementPlan {
    if !has_decode {
        let prefill = if single_cluster_phases && plan.prefill.matmul.len() > 1 {
            PhaseLayout {
                matmul: vec![(Lane::A100, 1.0)],
                vector: Lane::A100,
            }
        } else {
            plan.prefill.clone()
        };
        return PlacementPlan {
            prefill: prefill.clone(),
            decode: prefill,
        };
    }
    if single_cluster_phases && plan.prefill.matmul.len() > 1 {
        // In SinglePromptEngine RowSplit mode, prefill and decode both run on A100.
        return PlacementPlan::single(Lane::A100);
    }
    plan.clone()
}

pub(crate) fn verify_cluster_masks(
    affinity: &ResolvedAffinity,
    plan: &PlacementPlan,
    observed_matmul: u64,
    observed_vector: u64,
) -> Result<(), K3Error> {
    let (expected_matmul, expected_vector) = expected_phase_masks(plan);
    let missing_expected_cluster = |expected: u64, observed: u64| -> bool {
        observed == 0
            || ((expected & X100_CLUSTER_MASK) != 0 && (observed & X100_CLUSTER_MASK) == 0)
            || ((expected & A100_CLUSTER_MASK) != 0 && (observed & A100_CLUSTER_MASK) == 0)
    };
    if (observed_matmul & !expected_matmul) != 0
        || (observed_vector & !expected_vector) != 0
        || missing_expected_cluster(expected_matmul, observed_matmul)
        || missing_expected_cluster(expected_vector, observed_vector)
    {
        return Err(K3Error::AffinityError {
            requested_mask: affinity.effective_mask,
            requested_threads: affinity.threads,
            reason: format!(
                "cluster execution mask mismatch (matmul cpus 0x{observed_matmul:04X} vs expected 0x{expected_matmul:04X}, vector cpus 0x{observed_vector:04X} vs expected 0x{expected_vector:04X})"
            ),
        });
    }
    Ok(())
}

/// Computes `BenchmarkTelemetry`, per-run `RunMetric` rows, and aggregate `StatisticsSummary`,
/// measuring live hardware ceilings and exporting `--export-profile` if requested.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_telemetry_and_stats(
    inputs: TelemetryInputs<'_>,
) -> Result<(BenchmarkTelemetry, Vec<RunMetric>, StatisticsSummary), K3Error> {
    let first = inputs.primary_samples.first().ok_or_else(|| {
        cap_err(
            "cli",
            "benchmark aborted before any run completed".to_owned(),
        )
    })?;

    let tokens0 = first.generated_tokens.clone();
    let divergent = inputs
        .primary_samples
        .iter()
        .position(|s| s.generated_tokens != tokens0);
    let determinism = DeterminismReport {
        verified: divergent.is_none(),
        repeat_count: inputs.primary_samples.len(),
        token_digest: digest(&tokens0),
        first_divergent_run: divergent,
    };

    // Ceilings (ADR-034): peak MAC/s for prefill, streaming bandwidth for decode.
    let mut peak_cache: HashMap<&'static str, f64> = HashMap::new();
    let mut bw_cache: HashMap<&'static str, f64> = HashMap::new();
    let mut ceilings = |layout: &PhaseLayout, kernel: MatKernel| -> Result<(f64, f64), K3Error> {
        let (mut peak, mut bw) = (0.0, 0.0);
        for lane in lanes_of(layout) {
            if !peak_cache.contains_key(lane.as_str()) {
                peak_cache.insert(lane.as_str(), measure_peak_macs(inputs.pool, lane, kernel)?);
                bw_cache.insert(lane.as_str(), measure_memory_bandwidth(inputs.pool, lane)?);
            }
            peak += peak_cache[lane.as_str()];
            bw += bw_cache[lane.as_str()];
        }
        Ok((peak, bw))
    };

    let ime_on = inputs.config.ime == Some(true);
    let (prefill_peak, _) = ceilings(&inputs.plan.prefill, inputs.kernel)?;
    let (_, decode_bw) = ceilings(&inputs.plan.decode, inputs.kernel)?;
    let ime_vmadot_issued: u64 = if ime_on {
        inputs.primary_samples.iter().map(|s| s.vmadot_issued).sum()
    } else {
        0
    };

    let prefill_tps_mean = mean(inputs.primary_samples.iter().map(sample_prefill_tps));
    let decode_tps_mean = mean(
        inputs
            .primary_samples
            .iter()
            .map(|s| s.timing.decode_tokens_per_sec),
    );
    let prefill_ms = mean(inputs.primary_samples.iter().map(|s| s.timing.prefill_ms));
    let decode_ms = mean(inputs.primary_samples.iter().map(|s| s.timing.decode_ms));
    let prefill_macs_per_sec = if first.prompt_tokens > 0 {
        inputs.prefill_macs as f64 * prefill_tps_mean / first.prompt_tokens as f64
    } else {
        0.0
    };
    let decode_bytes_per_sec = inputs.decode_weight_bytes_per_token as f64 * decode_tps_mean;
    let pct = |achieved: f64, ceiling: f64| {
        if ceiling > 0.0 {
            achieved / ceiling * 100.0
        } else {
            0.0
        }
    };
    let prefill_cost_per_token_ms = if inputs.prompt_tokens.is_empty() {
        0.0
    } else {
        prefill_ms / inputs.prompt_tokens.len() as f64
    };
    let mean_decode_steps = mean(
        inputs
            .primary_samples
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

    let ime_comparison = inputs
        .baseline_samples
        .as_ref()
        .map(|(base_kernel, base_runs)| {
            let on = ArmThroughput {
                kernel_path: inputs.kernel.path_name().to_owned(),
                prefill_tps: prefill_tps_mean,
                decode_tps: decode_tps_mean,
                vmadot_issued: ime_vmadot_issued,
            };
            let off = ArmThroughput {
                kernel_path: base_kernel.path_name().to_owned(),
                prefill_tps: mean(base_runs.iter().map(sample_prefill_tps)),
                decode_tps: mean(base_runs.iter().map(|s| s.timing.decode_tokens_per_sec)),
                vmadot_issued: base_runs.iter().map(|s| s.vmadot_issued).sum(),
            };
            let pct_of = |d: f64, b: f64| if b > 0.0 { d / b * 100.0 } else { 0.0 };
            let dp = on.prefill_tps - off.prefill_tps;
            let dd = on.decode_tps - off.decode_tps;
            ImeComparison {
                delta_prefill_pct: pct_of(dp, off.prefill_tps),
                delta_decode_pct: pct_of(dd, off.decode_tps),
                delta_prefill_tps: dp,
                delta_decode_tps: dd,
                numerically_identical: base_runs.iter().all(|s| s.generated_tokens == tokens0),
                on,
                off,
            }
        });

    let run_metrics: Vec<RunMetric> = inputs
        .primary_samples
        .iter()
        .enumerate()
        .map(|(i, s)| RunMetric {
            run_index: i,
            ttft_ms: s.timing.ttft_ms,
            prefill_ms: s.timing.prefill_ms,
            prefill_tokens_per_sec: sample_prefill_tps(s),
            decode_ms: s.timing.decode_ms,
            decode_tokens_per_sec: s.timing.decode_tokens_per_sec,
            total_wall_time_ms: s.timing.total_wall_time_ms,
            cumulative_barrier_wait_ms: s.timing.cumulative_barrier_wait_ms,
            barrier_wait_ratio: s.timing.barrier_wait_ratio,
            compute_utilization_ratio: s.timing.compute_utilization_ratio,
            stop_reason: s.timing.stop_reason.as_str().to_owned(),
            stop_token_id: s.timing.stop_reason.token_id(),
            tokens: Some(s.generated_tokens.clone()),
        })
        .collect();
    let col = |f: fn(&RunMetric) -> f64| {
        MetricStats::calculate(&run_metrics.iter().map(f).collect::<Vec<_>>())
    };
    let statistics = StatisticsSummary {
        ttft_ms: col(|r| r.ttft_ms),
        prefill_tokens_per_sec: col(|r| r.prefill_tokens_per_sec),
        decode_tokens_per_sec: col(|r| r.decode_tokens_per_sec),
        cumulative_barrier_wait_ms: col(|r| r.cumulative_barrier_wait_ms),
        total_wall_time_ms: col(|r| r.total_wall_time_ms),
        compute_utilization_ratio: col(|r| r.compute_utilization_ratio),
    };

    let cluster = inputs.config.cluster.clone().unwrap_or_else(|| {
        if !inputs.affinity.x100_cores.is_empty() && !inputs.affinity.a100_cores.is_empty() {
            "both".to_owned()
        } else if !inputs.affinity.a100_cores.is_empty() {
            "a100".to_owned()
        } else {
            inputs
                .affinity
                .cluster
                .map_or("x100", |c| c.as_str())
                .to_owned()
        }
    });
    let split_name = inputs
        .config
        .split
        .map(|s| s.as_str().to_owned())
        .or_else(|| {
            inputs.config.phase_policy.and_then(|p| match p {
                ClusterPhasePolicy::X100PrefillA100Decode
                | ClusterPhasePolicy::A100PrefillX100Decode => Some("phase".to_owned()),
                ClusterPhasePolicy::RowSplit => Some("row".to_owned()),
                _ => None,
            })
        });
    let split_strategy_str = split_name.clone().unwrap_or_else(|| "none".to_owned());

    let measured_rss = measure_peak_resident_bytes();
    let peak_resident_bytes = measured_rss.max(inputs.total_required_bytes);

    let cluster_placement = ClusterPlacement {
        cluster: cluster.clone(),
        effective_core_mask: format!("0x{:04X}", inputs.affinity.effective_mask),
        allocated_cores: inputs.affinity.threads,
        x100_cores: inputs.affinity.x100_cores.len(),
        a100_cores: inputs.affinity.a100_cores.len(),
        coordination_profile: inputs.profile.as_str().to_owned(),
        split_strategy: split_strategy_str.clone(),
        ime: if ime_on { "on" } else { "off" }.to_owned(),
    };

    let allocated_cores = CoreAllocation {
        total: inputs.affinity.threads,
        total_cores: inputs.affinity.threads,
        x100_cores: inputs.affinity.x100_cores.len(),
        a100_cores: inputs.affinity.a100_cores.len(),
        effective_mask: format!("0x{:04X}", inputs.affinity.effective_mask),
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
        weight_bytes: inputs.weight_bytes,
        kv_bytes: inputs.kv_bytes,
        total_required_bytes: inputs.total_required_bytes,
        peak_resident_bytes,
    };

    let (
        bottleneck_classification,
        recommended_cluster,
        recommended_split,
        recommended_ime,
        rationale,
        recommendations,
    ) = build_advisory(
        inputs.config,
        &cluster,
        &inputs.quantization_level,
        &prefill,
        &decode,
        ime_on,
    );

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
        model_identity: inputs.model_identity,
        quantization_level: inputs.quantization_level,
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
        weight_bytes: inputs.weight_bytes,
        kv_bytes: inputs.kv_bytes,
        total_required_bytes: inputs.total_required_bytes,
        peak_resident_bytes,
        tuning_advisory,
    };

    export_cost_profile_if_requested(inputs.config, &cost_profile)?;

    let telemetry = BenchmarkTelemetry {
        cluster,
        split: split_name.clone(),
        split_strategy: split_name,
        ime: if ime_on { "on" } else { "off" }.to_owned(),
        ime_enabled: ime_on,
        kernel_path: inputs.kernel.path_name().to_owned(),
        ime_supported: inputs.ime_supported,
        prompt_tokens: inputs.prompt_tokens.len(),
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
        ime_vmadot_issued,
        ime_delta_prefill_tps: ime_comparison.as_ref().map(|c| c.delta_prefill_tps),
        ime_delta_decode_tps: ime_comparison.as_ref().map(|c| c.delta_decode_tps),
        ime_comparison,
        row_split_partition_ratio: inputs.row_split.as_ref().map(|r| r.prefill.x100_share),
        row_split: inputs.row_split,
        placement_prefill: placement_report(&inputs.plan.prefill),
        placement_decode: placement_report(&inputs.plan.decode),
        matmul_cpu_mask: inputs.observed_matmul,
        vector_cpu_mask: inputs.observed_vector,
        cluster_isolation_verified: true,
        workers_before: worker_reports(inputs.before),
        workers_after: worker_reports(inputs.after),
        determinism,
        tokens: tokens0,
    };

    Ok((telemetry, run_metrics, statistics))
}

fn build_advisory(
    config: &DiagConfig,
    cluster: &str,
    quantization_level: &QuantizationInfo,
    prefill: &PhaseReport,
    decode: &PhaseReport,
    ime_on: bool,
) -> (String, String, Option<String>, bool, String, Vec<String>) {
    let bottleneck = if decode.ceiling_pct >= prefill.ceiling_pct {
        "bandwidth-bound"
    } else {
        "compute-bound"
    };

    let (rec_cluster, rec_split, rec_ime) = match cluster {
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
}

pub(crate) fn export_cost_profile_if_requested(
    config: &DiagConfig,
    cost_profile: &CostProfile,
) -> Result<(), K3Error> {
    if let Some(export_path) = &config.export_profile {
        let json_str = serde_json::to_string_pretty(cost_profile)
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
    Ok(())
}

struct MatrixArmResult {
    kernel: MatKernel,
    plan: PlacementPlan,
    row_split: Option<RowSplitReport>,
    summaries: Vec<ForwardSummary>,
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic_arm(
    config: &DiagConfig,
    pool: &WorkerPool,
    affinity: &ResolvedAffinity,
    model: &ForwardModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    vocab: usize,
    kernel: MatKernel,
    inject_fault: Option<usize>,
) -> Result<MatrixArmResult, K3Error> {
    let (plan, row_split) = build_plan(pool, affinity, kernel, config.phase_policy)?;
    let repeat = config.repeat.unwrap_or(1);
    let mut summaries = Vec::with_capacity(repeat);
    for run in 0..repeat {
        if is_abort_requested() {
            break;
        }
        let mut s = model.evaluate(pool, kernel, &plan, prompt_tokens, max_tokens)?;
        inject_fault_if_requested(inject_fault, run, vocab, &mut s.generated_tokens);
        summaries.push(s);
    }
    Ok(MatrixArmResult {
        kernel,
        plan,
        row_split,
        summaries,
    })
}

fn to_arm_samples(summaries: &[ForwardSummary]) -> Vec<ArmSample> {
    summaries
        .iter()
        .map(|s| ArmSample {
            prompt_tokens: s.prompt_tokens,
            timing: s.timing.clone(),
            generated_tokens: s.generated_tokens.clone(),
            vmadot_issued: s.vmadot_issued,
        })
        .collect()
}

/// Executes the US-03 synthetic matrix calibration benchmark (`--synthetic-matrix`) and writes
/// a table or a `--json` report to `out`.
///
/// # Errors
///
/// Returns a [`K3Error`] for invalid placement, unsupported IME, cluster-isolation violations,
/// conflicting `--model` specification, or (after the report is written) a determinism violation.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_matrix_benchmark<W: Write>(
    config: &DiagConfig,
    mut out: W,
) -> Result<(), K3Error> {
    if config.model_path.is_some() {
        return Err(cap_err(
            "cli",
            "--synthetic-matrix cannot be combined with --model; use --synthetic-matrix alone for hardware calibration or --model <path> for real GGUF execution"
                .to_owned(),
        ));
    }

    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(64);
    let profile = config.coord.unwrap_or(CoordinationProfile::Spin);
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
    let ime_on = config.ime == Some(true);
    let ime_supported = probe_cluster_ime(&affinity, config.cluster.as_deref(), ime_on)?;

    // 3. Synthetic byte-level prompt tokens & deterministic in-memory ForwardModel.
    let vocab = 256_usize;
    let prompt_limit = n_ctx.saturating_sub(1).max(1);
    let mut prompt_tokens: Vec<u32> = prompt_text
        .bytes()
        .map(u32::from)
        .take(prompt_limit)
        .collect();
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let max_tokens = config
        .max_tokens
        .unwrap_or(128)
        .min(n_ctx.saturating_sub(prompt_tokens.len()));
    let seed = prompt_text
        .as_bytes()
        .iter()
        .fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
    let model = ForwardModel::new(seed, vocab);
    let fault = fault_index();

    // 4. Measure the primary arm, plus the IME-off baseline when IME is on.
    let primary_kernel = if ime_on {
        MatKernel::Ime
    } else {
        MatKernel::Vector
    };
    let primary = run_synthetic_arm(
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
        Some(run_synthetic_arm(
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
    let observed_matmul = primary
        .summaries
        .iter()
        .fold(0, |m, s| m | s.matmul_cpu_mask);
    let observed_vector = primary
        .summaries
        .iter()
        .fold(0, |m, s| m | s.vector_cpu_mask);
    let has_decode = primary.summaries.iter().any(|s| s.timing.decode_steps > 0);
    let exec_plan = executed_phase_plan(&primary.plan, false, has_decode);
    verify_cluster_masks(&affinity, &exec_plan, observed_matmul, observed_vector)?;

    let first_summary = primary.summaries.first().ok_or_else(|| {
        cap_err(
            "cli",
            "benchmark aborted before any run completed".to_owned(),
        )
    })?;
    let prefill_macs = first_summary.prefill_macs;
    let decode_weight_bytes_per_token = first_summary.decode_weight_bytes_per_token;
    let weight_bytes = decode_weight_bytes_per_token;
    let kv_bytes = (n_ctx as u64).saturating_mul(64);
    let total_required_bytes = weight_bytes.saturating_add(kv_bytes);

    let model_identity = ModelIdentity {
        name: "synthetic-matrix".to_owned(),
        architecture: "synthetic".to_owned(),
        file_path: "<synthetic-matrix>".to_owned(),
        file_digest: format!("{seed:016x}"),
    };
    let quantization_level = QuantizationInfo::from_descriptors(&[]);
    let primary_samples = to_arm_samples(&primary.summaries);
    let baseline_samples = baseline
        .as_ref()
        .map(|b| (b.kernel, to_arm_samples(&b.summaries)));

    let (telemetry, run_metrics, statistics) = build_telemetry_and_stats(TelemetryInputs {
        config,
        pool: &pool,
        affinity: &affinity,
        profile,
        prompt_tokens: &prompt_tokens,
        plan: &primary.plan,
        row_split: primary.row_split,
        kernel: primary.kernel,
        ime_supported,
        observed_matmul,
        observed_vector,
        before: &before,
        after: &after,
        primary_samples: &primary_samples,
        baseline_samples,
        prefill_macs,
        decode_weight_bytes_per_token,
        weight_bytes,
        kv_bytes,
        total_required_bytes,
        model_identity,
        quantization_level,
    })?;

    let report = MatrixBenchReport {
        execution_mode: "synthetic_matrix".to_owned(),
        architecture: "synthetic".to_owned(),
        model_filename: None,
        quant_summary: SYNTHETIC_QUANT_SUMMARY.to_owned(),
        mode: "matrix-benchmark".to_owned(),
        coordination_profile: profile.as_str().to_owned(),
        repeat_count: primary.summaries.len(),
        threads: affinity.threads,
        effective_core_mask: format!("0x{:04X}", affinity.effective_mask),
        determinism_verified: telemetry.determinism.verified,
        prompt: prompt_text.to_owned(),
        runs: run_metrics,
        statistics,
        telemetry,
    };

    if config.json {
        let json =
            serde_json::to_string_pretty(&report).map_err(|e| cap_err("json", e.to_string()))?;
        let _ = writeln!(out, "{json}");
    } else {
        write_benchmark_table(
            &mut out,
            TableHeader {
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

    if let Some(run) = report.telemetry.determinism.first_divergent_run {
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

/// Renders unified human-readable header, phase ceilings, US-04 cost profile, and US-02 jitter stats.
pub(crate) fn write_benchmark_table<W: Write>(
    out: &mut W,
    hdr: TableHeader<'_>,
    runs: &[RunMetric],
    statistics: &StatisticsSummary,
    t: &BenchmarkTelemetry,
) {
    let rule = "==========================================================";
    let _ = writeln!(
        out,
        "{rule}\n k3-diag: Cluster Placement, Ceiling & Jitter Benchmark\n{rule}"
    );
    let _ = writeln!(out, "Execution Mode       : {}", hdr.execution_mode);
    let _ = writeln!(out, "Architecture         : {}", hdr.architecture);
    if let Some(file) = hdr.model_filename {
        let _ = writeln!(out, "Model File           : {file}");
    }
    let _ = writeln!(out, "Quant Summary        : {}", hdr.quant_summary);
    let _ = writeln!(out, "Coordination Profile : {}", hdr.coordination_profile);
    let _ = writeln!(out, "Repeat Iterations    : {}", hdr.repeat_count);
    let _ = writeln!(out, "Cluster              : {}", t.cluster);
    if let Some(split) = &t.split {
        let _ = writeln!(out, "Split Policy         : {split}");
    }
    let _ = writeln!(out, "IME                  : {} ({})", t.ime, t.kernel_path);
    let _ = writeln!(out, "Worker Threads       : {}", hdr.threads);
    let _ = writeln!(out, "Effective Core Mask  : {}", hdr.effective_core_mask);
    let _ = writeln!(
        out,
        "Prompt / Generated   : {} / {} tokens",
        t.prompt_tokens, t.generated_tokens
    );
    let _ = writeln!(
        out,
        "Output Determinism   : {}",
        if hdr.determinism_verified {
            "verified (bit-identical across all runs at temp 0)"
        } else {
            "VIOLATION"
        }
    );
    let _ = writeln!(
        out,
        "Prefill              : {:.3} ms, {:.1} tok/s, {:.3} ms/token, {:.1}% of {} ceiling",
        t.prefill.latency_ms,
        t.prefill.tokens_per_sec,
        t.prefill.cost_per_token_ms,
        t.prefill.ceiling_pct,
        t.prefill.ceiling_metric
    );
    let _ = writeln!(
        out,
        "Decode               : {:.3} ms, {:.1} tok/s, {:.3} ms/token, {:.1}% of {} ceiling",
        t.decode.latency_ms,
        t.decode.tokens_per_sec,
        t.decode.cost_per_token_ms,
        t.decode.ceiling_pct,
        t.decode.ceiling_metric
    );
    if let Some(rs) = &t.row_split {
        let _ = writeln!(
            out,
            "Row Split (X100 share): prefill {:.3}, decode {:.3} (from measured throughput)",
            rs.prefill.x100_share, rs.decode.x100_share
        );
    }
    if let Some(c) = &t.ime_comparison {
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
        t.cost_profile.prefill_cost_per_token_ms, t.cost_profile.prefill_hardware_ceiling_pct
    );
    let _ = writeln!(
        out,
        "  Decode Cost        : {:.3} ms/token ({:.1}% of memory_bandwidth)",
        t.cost_profile.decode_cost_per_token_ms, t.cost_profile.decode_hardware_ceiling_pct
    );
    let _ = writeln!(
        out,
        "  Static Sched Weight: {:.3}",
        t.cost_profile.static_scheduling_weight
    );
    let _ = writeln!(
        out,
        "  Peak Resident RAM  : {} bytes (weights: {}, kv: {})",
        t.cost_profile.peak_resident_bytes, t.cost_profile.weight_bytes, t.cost_profile.kv_bytes
    );
    let _ = writeln!(
        out,
        "  Tuning Advisory    : {} (Cluster: {}, IME: {})",
        t.cost_profile.tuning_advisory.bottleneck_classification,
        t.cost_profile.tuning_advisory.recommended_cluster,
        if t.cost_profile.tuning_advisory.recommended_ime {
            "on"
        } else {
            "off"
        }
    );
    let _ = writeln!(
        out,
        "----------------------------------------------------------"
    );
    let _ = writeln!(out, "Iteration Breakdown:");
    let _ = writeln!(
        out,
        "  Run | TTFT (ms) | Prefill (tok/s) | Decode (tokens/s) | Barrier Wait (ms) | Total Wall (ms)"
    );
    for r in runs {
        let _ = writeln!(
            out,
            "  {:3} | {:9.2} | {:15.2} | {:17.2} | {:17.2} | {:15.2}",
            r.run_index + 1,
            r.ttft_ms,
            r.prefill_tokens_per_sec,
            r.decode_tokens_per_sec,
            r.cumulative_barrier_wait_ms,
            r.total_wall_time_ms
        );
    }
    let _ = writeln!(
        out,
        "----------------------------------------------------------"
    );
    let _ = writeln!(
        out,
        "Statistical Jitter Summary (Mean, Median, Min, Max, StdDev):"
    );
    let _ = writeln!(
        out,
        "  Metric                      |      Mean |    Median |       Min |       Max |    StdDev"
    );
    let _ = writeln!(
        out,
        "  TTFT (Time to First Token)  | {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
        statistics.ttft_ms.mean,
        statistics.ttft_ms.median,
        statistics.ttft_ms.min,
        statistics.ttft_ms.max,
        statistics.ttft_ms.stddev
    );
    let _ = writeln!(
        out,
        "  Prefill Throughput (tok/s)  | {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
        statistics.prefill_tokens_per_sec.mean,
        statistics.prefill_tokens_per_sec.median,
        statistics.prefill_tokens_per_sec.min,
        statistics.prefill_tokens_per_sec.max,
        statistics.prefill_tokens_per_sec.stddev
    );
    let _ = writeln!(
        out,
        "  Decode Throughput (tokens/s)| {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
        statistics.decode_tokens_per_sec.mean,
        statistics.decode_tokens_per_sec.median,
        statistics.decode_tokens_per_sec.min,
        statistics.decode_tokens_per_sec.max,
        statistics.decode_tokens_per_sec.stddev
    );
    let _ = writeln!(
        out,
        "  Barrier Wait Overhead (ms)  | {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
        statistics.cumulative_barrier_wait_ms.mean,
        statistics.cumulative_barrier_wait_ms.median,
        statistics.cumulative_barrier_wait_ms.min,
        statistics.cumulative_barrier_wait_ms.max,
        statistics.cumulative_barrier_wait_ms.stddev
    );
    let _ = writeln!(
        out,
        "  Total Wall Time (ms)        | {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
        statistics.total_wall_time_ms.mean,
        statistics.total_wall_time_ms.median,
        statistics.total_wall_time_ms.min,
        statistics.total_wall_time_ms.max,
        statistics.total_wall_time_ms.stddev
    );
    let _ = writeln!(
        out,
        "  Compute Utilization Ratio   | {:9.4} | {:9.4} | {:9.4} | {:9.4} | {:9.4}",
        statistics.compute_utilization_ratio.mean,
        statistics.compute_utilization_ratio.median,
        statistics.compute_utilization_ratio.min,
        statistics.compute_utilization_ratio.max,
        statistics.compute_utilization_ratio.stddev
    );
    let _ = writeln!(out, "{rule}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_affinity(effective_mask: u64, x100: Vec<usize>, a100: Vec<usize>) -> ResolvedAffinity {
        ResolvedAffinity {
            cluster: None,
            selection: Some(jero_engine::affinity::ClusterSelection::Both {
                split: SplitStrategy::Phase,
            }),
            vlen_domain: jero_engine::affinity::VlenDomain::Heterogeneous,
            effective_mask,
            threads: x100.len() + a100.len(),
            assigned_cores: x100.iter().copied().chain(a100.iter().copied()).collect(),
            x100_cores: x100,
            a100_cores: a100,
        }
    }

    #[test]
    fn test_verify_cluster_masks_rejects_empty_or_missing_expected_cluster() {
        let dual_aff = test_affinity(0xFFFF, vec![0, 1], vec![8, 9]);
        let phase_plan = PlacementPlan::phase(Lane::X100, Lane::A100);

        // Zero observed mask must fail rather than passing as a tautology.
        assert!(matches!(
            verify_cluster_masks(&dual_aff, &phase_plan, 0, 0),
            Err(K3Error::AffinityError { .. })
        ));

        // Missing A100 decode cluster bits on a dual-cluster phase plan must fail.
        assert!(matches!(
            verify_cluster_masks(&dual_aff, &phase_plan, 0x0003, 0x0003),
            Err(K3Error::AffinityError { .. })
        ));

        // Non-empty bits covering both expected clusters must succeed.
        verify_cluster_masks(&dual_aff, &phase_plan, 0x0303, 0x0303)
            .expect("both X100 and A100 observed bits valid");

        // Wrong-cluster bits for A100Only must fail.
        let a100_plan = PlacementPlan::single(Lane::A100);
        assert!(matches!(
            verify_cluster_masks(&dual_aff, &a100_plan, 0x0101, 0x0100),
            Err(K3Error::AffinityError { .. })
        ));
    }

    #[test]
    fn test_executed_phase_plan_narrows_to_prefill_without_decode() {
        let plan = PlacementPlan::phase(Lane::X100, Lane::A100);
        let prefill_only = executed_phase_plan(&plan, true, false);
        assert_eq!(expected_phase_masks(&prefill_only), (0x00FF, 0x00FF));

        let with_decode = executed_phase_plan(&plan, true, true);
        assert_eq!(expected_phase_masks(&with_decode), (0xFFFF, 0xFFFF));

        let a100_prefill_plan = PlacementPlan::phase(Lane::A100, Lane::X100);
        let a100_prefill_only = executed_phase_plan(&a100_prefill_plan, true, false);
        assert_eq!(expected_phase_masks(&a100_prefill_only), (0xFF00, 0xFF00));
        let a100_x100_decode = executed_phase_plan(&a100_prefill_plan, true, true);
        assert_eq!(expected_phase_masks(&a100_x100_decode), (0xFFFF, 0xFFFF));

        let row = PlacementPlan::row(0.5, 0.5);
        let row_engine = executed_phase_plan(&row, true, true);
        assert_eq!(expected_phase_masks(&row_engine), (0xFF00, 0xFF00));
    }
}
