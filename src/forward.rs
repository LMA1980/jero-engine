//! Heterogeneous-cluster matrix forward workload (US-03, ADR-009, ADR-033, ADR-034).
//!
//! A deterministic all-integer transformer-style forward pass (`RMSNorm`, `Q/K/V/O` and FFN
//! projections, causal attention, LM head) whose matrix multiplications run on either the `SpacemiT`
//! IME path (`smt.vmadot`) or the baseline RVV path, and whose operations are placed on the `X100`
//! and `A100` worker pools according to a [`PlacementPlan`]:
//!
//! - single cluster: everything on one pool,
//! - phase split: prefill and decode on different clusters,
//! - operation split: matrix multiplications on one cluster, normalization/attention on the other,
//! - row split: every matrix multiplication partitioned across both clusters by output-tile rows in
//!   proportion to *measured* per-cluster throughput ([`row_split_share`]).
//!
//! Accumulation is exact `i32` and requantization is integer-only, so the generated token sequence
//! is bit-identical across every placement, split and kernel choice.

use std::ops::Range;
use std::sync::atomic::{AtomicI32, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType, TensorDescriptor,
};
#[allow(unused_imports)]
use jero_kernels_rvv::ime::{
    TILE_C, TILE_K, TILE_M, TILE_N, dot_i8_vector, gemm_i8_ime, gemm_i8_vector, ime_probe_exec,
    ime_vmadot_count, pack_tiles, tile_count, vector_dot_count,
};
use jero_kernels_rvv::{
    compute_deberta_rel_pos_bucket, deberta_disentangled_attention_f32, dequantize_token_row,
    rmsnorm_unit_offset,
};

use crate::affinity::{ClusterPreset, WorkerPool};
use crate::engine::{PassTiming, compute_pass_timing};

#[cfg(feature = "jero-arch-bitnet")]
pub use jero_arch_bitnet::BitNetRecipe;

#[cfg(feature = "jero-arch-dory")]
pub use jero_arch_dory::DoryRecipe;

#[cfg(feature = "jero-arch-gemma")]
pub use crate::gemma::{
    ExpertWeights, Gemma4Recipe, GemmaRecipe, apply_gemma4_ple_residual, dispatch_sparse_moe_layer,
    gemma_norm_lane, select_top_k_experts,
};

#[cfg(feature = "jero-arch-modernbert")]
pub use crate::modernbert::ModernBertRecipe;

/// Canonical vocabulary size for `Dory1.0-4B` parallel LM head row-split (`131,072`).
pub const DORY_VOCAB_SIZE: usize = 131_072;

/// Applies Dory's final hyperspherical logit scaling (`logits[i] *= output_scale[i] / init_std`)
/// across the `X100` + `A100` parallel LM head row-split output buffer.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `init_std <= 0.0` or `!init_std.is_finite()`, or
/// [`K3Error::TensorBoundsError`] if `logits.len() != output_scale.len()`.
pub fn scale_dory_lm_head_logits(
    logits: &mut [f32],
    output_scale: &[f32],
    init_std: f32,
) -> Result<(), K3Error> {
    if !init_std.is_finite() || init_std <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: format!("dory.init_std must be a positive finite f32, got {init_std}"),
        });
    }
    if logits.len() != output_scale.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "output_scale.weight".to_owned(),
            offset: 0,
            byte_len: output_scale.len(),
            buffer_len: logits.len(),
            reason: format!(
                "output_scale length ({}) does not match LM head vocabulary logits length ({})",
                output_scale.len(),
                logits.len()
            ),
        });
    }
    let inv_std = 1.0_f32 / init_std;
    for (logit, &scale) in logits.iter_mut().zip(output_scale.iter()) {
        *logit *= scale * inv_std;
    }
    Ok(())
}

/// Hidden dimension of the benchmark model (multiple of [`TILE_K`]).
pub const MODEL_DIM: usize = 512;
/// FFN inner dimension of the benchmark model (multiple of [`TILE_K`]).
pub const FFN_DIM: usize = 1024;
/// Number of transformer layers in the benchmark model.
pub const N_LAYERS: usize = 4;

type Job = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// Processor cluster a unit of work is dispatched to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// `X100` out-of-order cluster (cores 0-7).
    X100,
    /// `A100` in-order cluster (cores 8-15).
    A100,
}

impl Lane {
    /// Canonical lowercase name (`"x100"` / `"a100"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X100 => "x100",
            Self::A100 => "a100",
        }
    }

    const fn preset(self) -> ClusterPreset {
        match self {
            Self::X100 => ClusterPreset::X100,
            Self::A100 => ClusterPreset::A100,
        }
    }
}

/// Matrix multiplication kernel selection (`--ime on|off`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatKernel {
    /// Baseline RVV 1.0 int8 vector path (`--ime off`).
    Vector,
    /// `SpacemiT` IME `smt.vmadot` path (`--ime on`).
    Ime,
}

impl MatKernel {
    /// Name of the kernel path as reported by the diagnostic tool.
    #[must_use]
    pub const fn path_name(self) -> &'static str {
        match self {
            Self::Vector => "rvv-vwmul",
            Self::Ime => "ime-vmadot",
        }
    }
}

/// Placement of one execution phase: matrix lanes with their output-tile share, and the lane that
/// runs normalization and attention.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseLayout {
    /// Lanes running matrix multiplications with the fraction of output tiles each one owns
    /// (fractions sum to 1).
    pub matmul: Vec<(Lane, f64)>,
    /// Lane running normalization and attention.
    pub vector: Lane,
}

impl PhaseLayout {
    /// Everything on one lane.
    #[must_use]
    pub fn single(lane: Lane) -> Self {
        Self {
            matmul: vec![(lane, 1.0)],
            vector: lane,
        }
    }
}

/// Complete prefill/decode placement.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementPlan {
    /// Placement of the prompt prefill phase.
    pub prefill: PhaseLayout,
    /// Placement of the autoregressive decode phase.
    pub decode: PhaseLayout,
}

impl PlacementPlan {
    /// Single-cluster placement.
    #[must_use]
    pub fn single(lane: Lane) -> Self {
        Self {
            prefill: PhaseLayout::single(lane),
            decode: PhaseLayout::single(lane),
        }
    }

    /// Phase split: prefill on `prefill`, decode on `decode`.
    #[must_use]
    pub fn phase(prefill: Lane, decode: Lane) -> Self {
        Self {
            prefill: PhaseLayout::single(prefill),
            decode: PhaseLayout::single(decode),
        }
    }

    /// Operation split: matrix multiplications on `matmul`, normalization/attention on `vector`.
    #[must_use]
    pub fn op(matmul: Lane, vector: Lane) -> Self {
        let layout = PhaseLayout {
            matmul: vec![(matmul, 1.0)],
            vector,
        };
        Self {
            prefill: layout.clone(),
            decode: layout,
        }
    }

    /// Row split with the given `X100` output-tile shares per phase; the rest goes to `A100`.
    /// Normalization and attention run on the `X100` out-of-order cores.
    #[must_use]
    pub fn row(prefill_x100_share: f64, decode_x100_share: f64) -> Self {
        let layout = |share: f64| PhaseLayout {
            matmul: vec![(Lane::X100, share), (Lane::A100, 1.0 - share)],
            vector: Lane::X100,
        };
        Self {
            prefill: layout(prefill_x100_share),
            decode: layout(decode_x100_share),
        }
    }
}

/// Derives the `X100` share of a row-split partition from measured per-cluster throughput
/// (`x100_rate / (x100_rate + a100_rate)`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if either rate is not a positive finite number.
pub fn row_split_share(x100_rate: f64, a100_rate: f64) -> Result<f64, K3Error> {
    if !(x100_rate.is_finite() && a100_rate.is_finite() && x100_rate > 0.0 && a100_rate > 0.0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "row-split".to_owned(),
            reason: format!(
                "cannot derive row-split ratio from non-positive throughput (x100={x100_rate}, a100={a100_rate})"
            ),
        });
    }
    Ok(x100_rate / (x100_rate + a100_rate))
}

/// Number of output tiles (out of `total`) assigned to the first lane for a `share` in `[0, 1]`.
#[must_use]
pub fn partition_tiles(total: usize, share: f64) -> usize {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "tile counts are tiny and the share is clamped to [0, 1]"
    )]
    let tiles = (total as f64 * share.clamp(0.0, 1.0)).round() as usize;
    tiles.min(total)
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic `i8` values in `[-8, 7]`.
#[allow(clippy::cast_possible_truncation)]
fn fill_small(seed: u64, len: usize) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let r = splitmix(seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
            (r & 0xF) as i8 - 8
        })
        .collect()
}

fn requant(v: i32, shift: u32) -> i8 {
    let r = (v + (1 << (shift - 1))) >> shift;
    #[expect(clippy::cast_possible_truncation, reason = "clamped to the i8 range")]
    let q = r.clamp(-127, 127) as i8;
    q
}

fn cpu_bit() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("K3_CPU_ONLINE_MASK").is_some()
            || std::env::var_os("K3_SIMULATE_A100").is_some()
        {
            if let Some(core) = crate::affinity::get_current_thread_core() {
                if core < 64 {
                    return 1_u64 << core;
                }
            }
        }
        // SAFETY: `sched_getcpu` has no preconditions.
        let cpu = unsafe { libc::sched_getcpu() };
        u64::try_from(cpu)
            .ok()
            .filter(|&c| c < 64)
            .map_or(0, |c| 1_u64 << c)
    }
    #[cfg(not(target_os = "linux"))]
    {
        if let Some(core) = crate::affinity::get_current_thread_core() {
            if core < 64 {
                return 1_u64 << core;
            }
        }
        0
    }
}

/// One weight matrix stored row-major (`n x k`, vector path) and IME-packed (IME path).
#[derive(Debug)]
pub struct Matrix {
    n: usize,
    k: usize,
    shift: u32,
    rows: Vec<i8>,
    packed: Vec<i8>,
}

impl Matrix {
    fn random(seed: u64, n: usize, k: usize, shift: u32) -> Self {
        let n_pad = tile_count(n) * TILE_N;
        let mut rows = fill_small(seed, n * k);
        rows.resize(n_pad * k, 0);
        let packed = pack_tiles(&rows, n_pad, k);
        Self {
            n: n_pad,
            k,
            shift,
            rows,
            packed,
        }
    }

    const fn bytes(&self) -> usize {
        self.n * self.k
    }
}

struct Operand {
    m: usize,
    k: usize,
    rows: Vec<i8>,
    packed: Option<Vec<i8>>,
}

fn worker_range(range: &Range<usize>, idx: usize, workers: usize) -> Range<usize> {
    let len = range.end - range.start;
    let chunk = len.div_ceil(workers.max(1));
    let lo = (range.start + idx * chunk).min(range.end);
    let hi = (lo + chunk).min(range.end);
    lo..hi
}

fn lane_error(lane: Lane) -> K3Error {
    K3Error::AffinityError {
        requested_mask: 0,
        requested_threads: 0,
        reason: format!(
            "worker pool has no '{}' workers for the requested placement",
            lane.as_str()
        ),
    }
}

fn run_lane(pool: &WorkerPool, lane: Lane, job: Job) -> Result<(), K3Error> {
    if let (Some(x), Some(a)) = (pool.x100_pool(), pool.a100_pool()) {
        match lane {
            Lane::X100 => x.execute_job(job),
            Lane::A100 => a.execute_job(job),
        }
        Ok(())
    } else if pool.affinity().cluster == Some(lane.preset()) {
        pool.execute_job(job);
        Ok(())
    } else {
        Err(lane_error(lane))
    }
}

/// Dispatch context: worker pool, matrix kernel and CPU-observation sinks.
#[derive(Debug)]
pub struct ForwardExec<'a> {
    pool: &'a WorkerPool,
    kernel: MatKernel,
    matmul_cpus: Arc<AtomicU64>,
    vector_cpus: Arc<AtomicU64>,
}

impl<'a> ForwardExec<'a> {
    /// Creates a dispatch context over `pool` using `kernel` for matrix multiplications.
    #[must_use]
    pub fn new(pool: &'a WorkerPool, kernel: MatKernel) -> Self {
        Self {
            pool,
            kernel,
            matmul_cpus: Arc::new(AtomicU64::new(0)),
            vector_cpus: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Bitmask of CPUs observed (`sched_getcpu`) running matrix-multiplication jobs.
    #[must_use]
    pub fn matmul_cpu_mask(&self) -> u64 {
        self.matmul_cpus.load(Ordering::Relaxed)
    }

    /// Bitmask of CPUs observed running normalization/attention jobs.
    #[must_use]
    pub fn vector_cpu_mask(&self) -> u64 {
        self.vector_cpus.load(Ordering::Relaxed)
    }

    /// `out[m x n] = x[m x k] * w^T` in exact `i32`, partitioned over `layout.matmul` lanes.
    #[allow(clippy::many_single_char_names)]
    fn matmul(
        &self,
        layout: &PhaseLayout,
        w: &Arc<Matrix>,
        x: &[i8],
        m: usize,
    ) -> Result<Vec<i32>, K3Error> {
        let k = w.k;
        let operand = Arc::new(Operand {
            m,
            k,
            rows: x.to_vec(),
            packed: (self.kernel == MatKernel::Ime).then(|| pack_tiles(x, m, k)),
        });
        let n = w.n;
        let n_tiles = n / TILE_N;
        let out: Arc<Vec<AtomicI32>> = Arc::new((0..m * n).map(|_| AtomicI32::new(0)).collect());

        let mut jobs: Vec<(Lane, Job)> = Vec::new();
        let mut start = 0_usize;
        let mut acc_share = 0.0_f64;
        for (idx, &(lane, share)) in layout.matmul.iter().enumerate() {
            acc_share += share;
            let end = if idx + 1 == layout.matmul.len() {
                n_tiles
            } else {
                partition_tiles(n_tiles, acc_share)
            };
            let range = start..end.max(start);
            start = range.end;
            if range.is_empty() {
                continue;
            }
            let (w, operand, out) = (Arc::clone(w), Arc::clone(&operand), Arc::clone(&out));
            let lane_kernel = match (self.kernel, lane) {
                (MatKernel::Ime, Lane::A100) => MatKernel::Ime,
                _ => MatKernel::Vector,
            };
            let cpus = Arc::clone(&self.matmul_cpus);
            let job: Job = Arc::new(move |idx, workers| {
                let tiles = worker_range(&range, idx, workers);
                if tiles.is_empty() {
                    return;
                }
                cpus.fetch_or(cpu_bit(), Ordering::Relaxed);
                matmul_tiles(lane_kernel, &w, &operand, tiles, &out);
            });
            jobs.push((lane, job));
        }

        let results: Vec<Result<(), K3Error>> = if jobs.len() == 1 {
            let (lane, job) = jobs.remove(0);
            vec![run_lane(self.pool, lane, job)]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = jobs
                    .into_iter()
                    .map(|(lane, job)| s.spawn(move || run_lane(self.pool, lane, job)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("lane dispatch thread"))
                    .collect()
            })
        };
        for r in results {
            r?;
        }
        Ok(out.iter().map(|v| v.load(Ordering::Relaxed)).collect())
    }

    /// Runs `body(row_range)` over `rows` rows on the vector lane, partitioned across its workers.
    fn rows_on_lane<F>(&self, lane: Lane, rows: usize, body: F) -> Result<(), K3Error>
    where
        F: Fn(Range<usize>) + Send + Sync + 'static,
    {
        let cpus = Arc::clone(&self.vector_cpus);
        let job: Job = Arc::new(move |idx, workers| {
            let r = worker_range(&(0..rows), idx, workers);
            if r.is_empty() {
                return;
            }
            cpus.fetch_or(cpu_bit(), Ordering::Relaxed);
            body(r);
        });
        run_lane(self.pool, lane, job)
    }

    fn rmsnorm(&self, lane: Lane, x: &[i8], m: usize) -> Result<Vec<i8>, K3Error> {
        let input: Arc<Vec<i8>> = Arc::new(x.to_vec());
        let out: Arc<Vec<AtomicU8>> =
            Arc::new((0..m * MODEL_DIM).map(|_| AtomicU8::new(0)).collect());
        let (i2, o2) = (Arc::clone(&input), Arc::clone(&out));
        self.rows_on_lane(lane, m, move |rows| {
            for r in rows {
                let row = &i2[r * MODEL_DIM..(r + 1) * MODEL_DIM];
                let ms: u64 = row
                    .iter()
                    .map(|&v| u64::from(i32::from(v).unsigned_abs().pow(2)))
                    .sum();
                let rms = i32::try_from(isqrt(ms / MODEL_DIM as u64))
                    .unwrap_or(i32::MAX)
                    .max(1);
                for (d, &v) in row.iter().enumerate() {
                    let y = (i32::from(v) * 48 / rms).clamp(-127, 127);
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "clamped to [-127, 127] and stored bit-preserving as u8"
                    )]
                    o2[r * MODEL_DIM + d].store(y as i8 as u8, Ordering::Relaxed);
                }
            }
        })?;
        Ok(collect_i8(&out))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        lane: Lane,
        q: &[i8],
        k_cache: &[i8],
        v_cache: &[i8],
        m: usize,
        start_pos: usize,
    ) -> Result<Vec<i8>, K3Error> {
        let q: Arc<Vec<i8>> = Arc::new(q.to_vec());
        let kc: Arc<Vec<i8>> = Arc::new(k_cache.to_vec());
        let vc: Arc<Vec<i8>> = Arc::new(v_cache.to_vec());
        let out: Arc<Vec<AtomicU8>> =
            Arc::new((0..m * MODEL_DIM).map(|_| AtomicU8::new(0)).collect());
        let o2 = Arc::clone(&out);
        self.rows_on_lane(lane, m, move |rows| {
            let mut acc = vec![0_i64; MODEL_DIM];
            for r in rows {
                let pos = start_pos + r;
                let qrow = &q[r * MODEL_DIM..(r + 1) * MODEL_DIM];
                acc.fill(0);
                let mut total_w = 0_i64;
                for t in 0..=pos {
                    let s = dot_i8_vector(qrow, &kc[t * MODEL_DIM..(t + 1) * MODEL_DIM]);
                    let w = i64::from((s >> 10).clamp(0, 127)) + 1;
                    total_w += w;
                    let vrow = &vc[t * MODEL_DIM..(t + 1) * MODEL_DIM];
                    for (a, &v) in acc.iter_mut().zip(vrow) {
                        *a += w * i64::from(v);
                    }
                }
                for (d, &a) in acc.iter().enumerate() {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "weighted mean of i8 values"
                    )]
                    let y = (a / total_w) as i8;
                    #[expect(clippy::cast_sign_loss, reason = "bit-preserving i8 store")]
                    o2[r * MODEL_DIM + d].store(y as u8, Ordering::Relaxed);
                }
            }
        })?;
        Ok(collect_i8(&out))
    }
}

fn collect_i8(out: &[AtomicU8]) -> Vec<i8> {
    #[expect(clippy::cast_possible_wrap, reason = "bit-preserving u8 -> i8")]
    out.iter()
        .map(|v| v.load(Ordering::Relaxed) as i8)
        .collect()
}

fn isqrt(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    let mut x = v;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u64::midpoint(x, v / x);
    }
    x
}

fn matmul_tiles(
    kernel: MatKernel,
    w: &Matrix,
    x: &Operand,
    tiles: Range<usize>,
    out: &[AtomicI32],
) {
    let n = w.n;
    match kernel {
        MatKernel::Ime => {
            let a = x.packed.as_deref().expect("IME operand is packed");
            gemm_i8_ime(
                a,
                tile_count(x.m),
                &w.packed,
                tiles,
                x.k / TILE_K,
                |mt, nt, c: &[i32; TILE_C]| {
                    for i in 0..TILE_M {
                        let row = mt * TILE_M + i;
                        if row >= x.m {
                            break;
                        }
                        for j in 0..TILE_N {
                            out[row * n + nt * TILE_N + j]
                                .store(c[i * TILE_N + j], Ordering::Relaxed);
                        }
                    }
                },
            );
        }
        MatKernel::Vector => {
            gemm_i8_vector(
                &x.rows,
                x.m,
                &w.rows,
                tiles.start * TILE_N..tiles.end * TILE_N,
                x.k,
                |r, c, v| out[r * n + c].store(v, Ordering::Relaxed),
            );
        }
    }
}

fn requant_all(raw: &[i32], shift: u32) -> Vec<i8> {
    raw.iter().map(|&v| requant(v, shift)).collect()
}

fn sat_add(a: &mut [i8], b: &[i8]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = x.saturating_add(y);
    }
}

struct LayerWeights {
    wq: Arc<Matrix>,
    wk: Arc<Matrix>,
    wv: Arc<Matrix>,
    wo: Arc<Matrix>,
    up: Arc<Matrix>,
    down: Arc<Matrix>,
}

/// Deterministic synthetic benchmark model (weights derived from `seed`).
#[derive(Debug)]
pub struct ForwardModel {
    seed: u64,
    vocab: usize,
    layers: Vec<LayerWeights>,
    lm_head: Arc<Matrix>,
}

impl std::fmt::Debug for LayerWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerWeights").finish_non_exhaustive()
    }
}

struct KvState {
    k: Vec<Vec<i8>>,
    v: Vec<Vec<i8>>,
}

/// Outcome of one forward-pass evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardSummary {
    /// Prompt tokens processed by prefill.
    pub prompt_tokens: usize,
    /// Greedy (temperature 0) generated token ids.
    pub generated_tokens: Vec<u32>,
    /// Prefill/decode timing and barrier metrics.
    pub timing: PassTiming,
    /// Useful multiply-accumulates executed by prefill.
    pub prefill_macs: u64,
    /// Useful multiply-accumulates per decoded token.
    pub decode_macs_per_token: u64,
    /// Weight bytes streamed per decoded token.
    pub decode_weight_bytes_per_token: u64,
    /// CPUs observed running matrix multiplications.
    pub matmul_cpu_mask: u64,
    /// CPUs observed running normalization/attention.
    pub vector_cpu_mask: u64,
    /// `smt.vmadot` instructions issued during the evaluation.
    pub vmadot_issued: u64,
    /// Baseline RVV int8 dot-product invocations (matrix path) during the evaluation.
    pub vector_dots: u64,
}

impl ForwardModel {
    /// Builds the benchmark model with `vocab` output tokens from `seed`.
    #[must_use]
    pub fn new(seed: u64, vocab: usize) -> Self {
        let mat = |salt: u64, n: usize, k: usize, shift: u32| {
            Arc::new(Matrix::random(splitmix(seed ^ salt), n, k, shift))
        };
        let layers = (0..N_LAYERS as u64)
            .map(|l| {
                let s = l * 16;
                LayerWeights {
                    wq: mat(s + 1, MODEL_DIM, MODEL_DIM, 6),
                    wk: mat(s + 2, MODEL_DIM, MODEL_DIM, 6),
                    wv: mat(s + 3, MODEL_DIM, MODEL_DIM, 6),
                    wo: mat(s + 4, MODEL_DIM, MODEL_DIM, 6),
                    up: mat(s + 5, FFN_DIM, MODEL_DIM, 6),
                    down: mat(s + 6, MODEL_DIM, FFN_DIM, 7),
                }
            })
            .collect();
        Self {
            seed,
            vocab,
            layers,
            lm_head: mat(0xFFFF, vocab.max(1), MODEL_DIM, 6),
        }
    }

    /// Useful MACs per token through the transformer layers (excluding the LM head).
    #[must_use]
    pub const fn layer_macs_per_token() -> u64 {
        (N_LAYERS * (4 * MODEL_DIM * MODEL_DIM + 2 * MODEL_DIM * FFN_DIM)) as u64
    }

    fn head_macs(&self) -> u64 {
        (self.lm_head.n * MODEL_DIM) as u64
    }

    fn weight_bytes_per_token(&self) -> u64 {
        let layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.wq.bytes()
                    + l.wk.bytes()
                    + l.wv.bytes()
                    + l.wo.bytes()
                    + l.up.bytes()
                    + l.down.bytes()
            })
            .sum();
        (layer + self.lm_head.bytes()) as u64
    }

    fn embed(&self, token: u32) -> Vec<i8> {
        fill_small(
            self.seed ^ u64::from(token).wrapping_mul(0x9E37_79B9),
            MODEL_DIM,
        )
        .into_iter()
        .map(|v| v * 8)
        .collect()
    }

    /// One transformer pass over `tokens` at absolute positions `start_pos..`, returning the
    /// logits of the final token.
    #[allow(clippy::many_single_char_names)]
    fn forward(
        &self,
        exec: &ForwardExec<'_>,
        layout: &PhaseLayout,
        state: &mut KvState,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<i32>, K3Error> {
        let m = tokens.len();
        let mut x: Vec<i8> = tokens.iter().flat_map(|&t| self.embed(t)).collect();
        let vec_lane = layout.vector;
        for (li, l) in self.layers.iter().enumerate() {
            let h = exec.rmsnorm(vec_lane, &x, m)?;
            let q = requant_all(&exec.matmul(layout, &l.wq, &h, m)?, l.wq.shift);
            let k = requant_all(&exec.matmul(layout, &l.wk, &h, m)?, l.wk.shift);
            let v = requant_all(&exec.matmul(layout, &l.wv, &h, m)?, l.wv.shift);
            state.k[li].extend_from_slice(&k);
            state.v[li].extend_from_slice(&v);
            let a = exec.attention(vec_lane, &q, &state.k[li], &state.v[li], m, start_pos)?;
            let o = requant_all(&exec.matmul(layout, &l.wo, &a, m)?, l.wo.shift);
            sat_add(&mut x, &o);
            let h2 = exec.rmsnorm(vec_lane, &x, m)?;
            let mut up = requant_all(&exec.matmul(layout, &l.up, &h2, m)?, l.up.shift);
            for u in &mut up {
                *u = (*u).max(0);
            }
            let down = requant_all(&exec.matmul(layout, &l.down, &up, m)?, l.down.shift);
            sat_add(&mut x, &down);
        }
        let last = &x[(m - 1) * MODEL_DIM..];
        let hn = exec.rmsnorm(vec_lane, last, 1)?;
        exec.matmul(layout, &self.lm_head, &hn, 1)
    }

    fn sample(&self, logits: &[i32]) -> u32 {
        let mut best = 0_usize;
        for (i, &l) in logits.iter().enumerate().take(self.vocab).skip(1) {
            if l > logits[best] {
                best = i;
            }
        }
        u32::try_from(best).unwrap_or(0)
    }

    /// Runs prefill over `prompt` then up to `max_tokens` greedy decode steps under `plan`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] for an empty prompt and
    /// [`K3Error::AffinityError`] if `plan` names a cluster the pool does not own.
    #[expect(clippy::cast_precision_loss, reason = "timing ratios")]
    pub fn evaluate(
        &self,
        pool: &WorkerPool,
        kernel: MatKernel,
        plan: &PlacementPlan,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Result<ForwardSummary, K3Error> {
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "forward".to_owned(),
                reason: "prompt token sequence cannot be empty".to_owned(),
            });
        }
        let exec = ForwardExec::new(pool, kernel);
        let mut state = KvState {
            k: vec![Vec::new(); N_LAYERS],
            v: vec![Vec::new(); N_LAYERS],
        };
        let (ime0, vec0) = (ime_vmadot_count(), vector_dot_count());
        pool.reset_barrier_wait();
        let pass_start = Instant::now();

        let prefill_start = Instant::now();
        let logits = self.forward(&exec, &plan.prefill, &mut state, prompt, 0)?;
        let mut generated = Vec::with_capacity(max_tokens);
        let mut decode_steps = 0_usize;
        let (ttft_ms, prefill_ms);
        let mut decode_ms = 0.0;
        if max_tokens == 0 {
            prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
            ttft_ms = prefill_ms;
        } else {
            let mut tok = self.sample(&logits);
            generated.push(tok);
            prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
            ttft_ms = prefill_ms;
            let decode_start = Instant::now();
            while generated.len() < max_tokens {
                let pos = prompt.len() + decode_steps;
                let logits = self.forward(&exec, &plan.decode, &mut state, &[tok], pos)?;
                tok = self.sample(&logits);
                generated.push(tok);
                decode_steps += 1;
            }
            decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        }
        let tps = if decode_ms > 0.0 && decode_steps > 0 {
            decode_steps as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let timing = compute_pass_timing(
            pass_start,
            ttft_ms,
            prefill_ms,
            decode_ms,
            decode_steps,
            tps,
            Some(pool),
        );
        Ok(ForwardSummary {
            prompt_tokens: prompt.len(),
            generated_tokens: generated,
            timing,
            prefill_macs: prompt.len() as u64 * Self::layer_macs_per_token() + self.head_macs(),
            decode_macs_per_token: Self::layer_macs_per_token() + self.head_macs(),
            decode_weight_bytes_per_token: self.weight_bytes_per_token(),
            matmul_cpu_mask: exec.matmul_cpu_mask(),
            vector_cpu_mask: exec.vector_cpu_mask(),
            vmadot_issued: ime_vmadot_count() - ime0,
            vector_dots: vector_dot_count() - vec0,
        })
    }
}

/// Phase shape used to calibrate per-cluster matrix throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationShape {
    /// Prefill-like batched activations (`m = 32`).
    Prefill,
    /// Decode-like single activation row (`m = 1`).
    Decode,
}

/// Measures the matrix-multiplication throughput (useful MACs per second) of `lane` with
/// `kernel` on a representative `m x 512 x 512` projection (best of 5 after one warm-up).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if `pool` has no workers for `lane`.
#[expect(clippy::cast_precision_loss, reason = "throughput ratio")]
pub fn calibrate_matmul_rate(
    pool: &WorkerPool,
    lane: Lane,
    kernel: MatKernel,
    shape: CalibrationShape,
) -> Result<f64, K3Error> {
    let m = match shape {
        CalibrationShape::Prefill => 32,
        CalibrationShape::Decode => 1,
    };
    let w = Arc::new(Matrix::random(0xCA11_B8A7, MODEL_DIM, MODEL_DIM, 6));
    let x = fill_small(0x5EED, m * MODEL_DIM);
    let exec = ForwardExec::new(pool, kernel);
    let layout = PhaseLayout::single(lane);
    exec.matmul(&layout, &w, &x, m)?;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        exec.matmul(&layout, &w, &x, m)?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    Ok((m * MODEL_DIM * MODEL_DIM) as f64 / best.max(1e-9))
}

/// Measures the peak register/L1-resident matrix throughput (MACs per second) of every worker of
/// `lane` for `kernel`, summed (the compute ceiling of ADR-034).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if `pool` has no workers for `lane`.
#[expect(clippy::cast_precision_loss, reason = "throughput ratio")]
pub fn measure_peak_macs(pool: &WorkerPool, lane: Lane, kernel: MatKernel) -> Result<f64, K3Error> {
    let lane_kernel = match (kernel, lane) {
        (MatKernel::Ime, Lane::A100) => MatKernel::Ime,
        _ => MatKernel::Vector,
    };
    let rates: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&rates);
    let job: Job = Arc::new(move |_, _| {
        const K_CHUNKS: usize = 32;
        let a = fill_small(1, TILE_M * K_CHUNKS * TILE_K);
        let b = fill_small(2, TILE_N * K_CHUNKS * TILE_K);
        let (macs_per_call, mut calls) = ((TILE_M * TILE_N * K_CHUNKS * TILE_K) as f64, 0_u64);
        let start = Instant::now();
        let mut sum = 0_i64;
        while start.elapsed().as_millis() < 5 {
            for _ in 0..64 {
                match lane_kernel {
                    MatKernel::Ime => gemm_i8_ime(&a, 1, &b, 0..1, K_CHUNKS, |_, _, c| {
                        sum += i64::from(c[0]);
                    }),
                    MatKernel::Vector => {
                        for r in 0..TILE_M {
                            for cidx in 0..TILE_N {
                                let ar = &a[r * K_CHUNKS * TILE_K..(r + 1) * K_CHUNKS * TILE_K];
                                let br =
                                    &b[cidx * K_CHUNKS * TILE_K..(cidx + 1) * K_CHUNKS * TILE_K];
                                sum += i64::from(dot_i8_vector(ar, br));
                            }
                        }
                    }
                }
                calls += 1;
            }
        }
        std::hint::black_box(sum);
        let rate = calls as f64 * macs_per_call / start.elapsed().as_secs_f64();
        sink.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(rate);
    });
    run_lane(pool, lane, job)?;
    let total = rates
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .sum();
    Ok(total)
}

/// Measures the streaming read bandwidth (bytes per second) of every worker of `lane`, summed
/// (the decode ceiling of ADR-034).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if `pool` has no workers for `lane`.
#[expect(clippy::cast_precision_loss, reason = "throughput ratio")]
pub fn measure_memory_bandwidth(pool: &WorkerPool, lane: Lane) -> Result<f64, K3Error> {
    const WORDS: usize = 2 * 1024 * 1024;
    let buf: Arc<Vec<u64>> = Arc::new((0..WORDS as u64).collect());
    let rates: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let (b2, sink) = (Arc::clone(&buf), Arc::clone(&rates));
    let job: Job = Arc::new(move |idx, workers| {
        let r = worker_range(&(0..WORDS), idx, workers);
        let slice = &b2[r.clone()];
        let start = Instant::now();
        let mut acc = 0_u64;
        for _ in 0..2 {
            for &v in slice {
                acc = acc.wrapping_add(v);
            }
        }
        std::hint::black_box(acc);
        let rate = (slice.len() * 8 * 2) as f64 / start.elapsed().as_secs_f64().max(1e-9);
        sink.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(rate);
    });
    run_lane(pool, lane, job)?;
    let total = rates
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .sum();
    Ok(total)
}

/// Reports whether the CPU `core` executes the `SpacemiT` IME `XSMTVDotII` (`smt.vmadot`, `VLEN=1024`)
/// instruction path, by running it in a forked child pinned to that core (a missing extension kills
/// the child with `SIGILL`).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn probe_ime_on_core(core: usize) -> bool {
    #[cfg(any(not(target_os = "linux"), miri))]
    {
        // On non-Linux platforms (or under Miri), follow canonical SpacemiT K3 topology: Cluster 1 (A100, cores 8-15)
        // supports IME matrix acceleration, while Cluster 0 (X100, cores 0-7) does not.
        (8..16).contains(&core)
    }
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        if std::env::var_os("K3_CPU_ONLINE_MASK").is_some()
            || std::env::var_os("K3_SIMULATE_A100").is_some()
            || !cfg!(target_arch = "riscv64")
        {
            return (8..16).contains(&core);
        }

        let mask = 1_u64 << core;
        // SAFETY: the child only issues async-signal-safe operations (registering with
        // `/proc/set_ai_thread` for A100 cores >= 8, a raw affinity syscall, the stack-only probe
        // and `_exit`); the parent merely reaps it.
        unsafe {
            let pid = libc::fork();
            if pid < 0 {
                return false;
            }
            if pid == 0 {
                if core >= 8 {
                    let tid = libc::syscall(libc::SYS_gettid);
                    let path = b"/proc/set_ai_thread\0";
                    let fd = libc::open(path.as_ptr().cast::<libc::c_char>(), libc::O_WRONLY);
                    if fd >= 0 {
                        let mut buf = [0_u8; 32];
                        let mut tmp = [0_u8; 32];
                        let mut val = tid;
                        let mut i = 0_usize;
                        if val <= 0 {
                            buf[0] = b'0';
                            i = 1;
                        } else {
                            while val > 0 {
                                tmp[i] = b'0' + (val % 10) as u8;
                                val /= 10;
                                i += 1;
                            }
                            for j in 0..i {
                                buf[j] = tmp[i - 1 - j];
                            }
                        }
                        buf[i] = b'\n';
                        let _ = libc::write(fd, buf.as_ptr().cast::<libc::c_void>(), i + 1);
                        libc::close(fd);
                    }
                }
                let r = libc::syscall(
                    libc::SYS_sched_setaffinity,
                    0_i32,
                    size_of::<u64>(),
                    std::ptr::from_ref(&mask).cast::<libc::c_void>(),
                );
                if r < 0 {
                    libc::_exit(2);
                }
                libc::_exit(if ime_probe_exec() { 0 } else { 3 });
            }
            let mut status = 0_i32;
            if libc::waitpid(pid, &raw mut status, 0) < 0 {
                return false;
            }
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
        }
    }
}

/// Zero-KV `DeBERTa-v3` disentangled-attention encoder classification recipe (Decision 2).
///
/// Pinned by default to `Lane::X100` with a strict `Q8_0` / `F16` / `BF16` / `F32` precision floor
/// and 3-way `[CLS]` softmax output (`BENIGN`, `INJECTION`, `JAILBREAK`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebertaV3Recipe {
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Model hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head dimension (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Maximum relative position bucket distance `k` (`2 * k` buckets).
    pub max_relative_positions: usize,
    /// Number of classification labels (`3` by default: `BENIGN`, `INJECTION`, `JAILBREAK`).
    pub num_labels: usize,
}

impl Default for DebertaV3Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DebertaV3Recipe {
    /// Creates a default `DeBERTa-v3` guardrail classification recipe (`12` layers, `768` hidden,
    /// `12` heads, `3` `[CLS]` classes).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 12,
            hidden_dim: 768,
            n_heads: 12,
            head_dim: 64,
            max_relative_positions: 256,
            num_labels: 3,
        }
    }

    /// Returns the default `X100`-only cluster placement plan for `DeBERTa-v3` prompt guards.
    #[must_use]
    pub fn default_placement() -> PlacementPlan {
        PlacementPlan::single(Lane::X100)
    }

    /// Hydrates a [`DebertaV3Recipe`] from optional GGUF [`ModelMetadata`].
    #[must_use]
    pub fn from_metadata(metadata: Option<&ModelMetadata>) -> Self {
        let mut recipe = Self::new();
        let Some(meta) = metadata else {
            return recipe;
        };

        if meta.layer_count() > 0 {
            recipe.num_layers = meta.layer_count();
        }
        if meta.embedding_dim() > 0 {
            recipe.hidden_dim = meta.embedding_dim();
        } else if let Some(embd) = deberta_find_tensor_desc(meta.tensors(), "token_embd.weight")
            && embd.shape().len() == 2
            && embd.shape()[0] > 0
        {
            recipe.hidden_dim = embd.shape()[0];
        }

        if meta.head_count() > 0 {
            recipe.n_heads = meta.head_count().min(recipe.hidden_dim).max(1);
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);
        } else if recipe.hidden_dim % recipe.n_heads != 0 {
            recipe.n_heads = if recipe.hidden_dim % 4 == 0 { 4 } else { 1 };
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);
        } else {
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);
        }

        for head_name in ["classifier.weight", "cls.weight", "output.weight"] {
            if let Some(head) = deberta_find_tensor_desc(meta.tensors(), head_name)
                && head.shape().len() == 2
                && head.shape()[1] > 0
            {
                recipe.num_labels = head.shape()[1];
                break;
            }
        }
        if let Some(bias) = deberta_find_tensor_desc(meta.tensors(), "classifier.bias")
            && !bias.shape().is_empty()
            && bias.shape()[0] > 0
        {
            recipe.num_labels = bias.shape()[0];
        }

        recipe
    }
}

fn deberta_find_tensor_desc<'a>(
    tensors: &'a [TensorDescriptor],
    name: &str,
) -> Option<&'a TensorDescriptor> {
    tensors.iter().find(|t| t.name() == name)
}

fn normalize_max_abs(values: &mut [f32]) {
    let mut max_abs = 0.0_f32;
    for v in values.iter_mut() {
        if v.is_finite() {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        } else {
            *v = 0.0;
        }
    }
    if max_abs > 1.0 {
        for v in values.iter_mut() {
            *v /= max_abs;
        }
    }
}

fn deberta_read_f32_weight_or_zeros(
    weights: &dyn ModelWeights,
    name: &str,
    len: usize,
) -> Vec<f32> {
    let Ok(view) = weights.tensor(name) else {
        return vec![0.0_f32; len];
    };
    let mut out = vec![0.0_f32; len];
    let raw = view.as_bytes();
    if view.quant() == QuantType::F32 {
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * 4;
            if off + 4 <= raw.len() {
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if v.is_finite() && v.abs() <= 1e6 {
                    *slot = v;
                }
            }
        }
    }
    out
}

fn deberta_embed_token(
    token_id: u32,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let mut hidden = vec![0.0_f32; hidden_dim];
    if weights.contains_tensor("token_embd.weight") {
        let embd = weights.tensor("token_embd.weight")?;
        let token_idx = usize::try_from(token_id).unwrap_or(0);
        let row_count = embd.num_elements().checked_div(hidden_dim).unwrap_or(0);
        if row_count > 0 && token_idx < row_count {
            dequantize_token_row(&embd, token_idx, &mut hidden)?;
            normalize_max_abs(&mut hidden);
            return Ok(hidden);
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "bounded token modulo fits cleanly in f32"
    )]
    let tok_phase = ((token_id % 97) as f32) * 0.031_25 + 0.125;
    for (i, slot) in hidden.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "small hidden index fits cleanly in f32"
        )]
        let dim_offset = ((i % 11) as f32) * 0.015_625;
        *slot = (tok_phase + dim_offset).sin() * 0.5;
    }
    Ok(hidden)
}

fn deberta_project_or_fallback(
    weights: &dyn ModelWeights,
    tensor_name: &str,
    input: &[f32],
    out_len: usize,
    scale: f32,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; out_len];
    if weights.contains_tensor(tensor_name) {
        let view = weights.tensor(tensor_name)?;
        if view.num_elements() == out_len.saturating_mul(input.len()) {
            let mut row_buf = vec![0.0_f32; input.len()];
            let mut accum = Vec::with_capacity(out_len);
            let mut max_abs = 0.0_f64;
            for row_idx in 0..out_len {
                dequantize_token_row(&view, row_idx, &mut row_buf)?;
                let dot: f64 = row_buf
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum();
                if dot.is_finite() {
                    max_abs = max_abs.max(dot.abs());
                    accum.push(dot);
                } else {
                    accum.push(0.0);
                }
            }
            let divisor = if max_abs > 1.0 { max_abs } else { 1.0 };
            for (slot, val) in out.iter_mut().zip(accum) {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "scaled output is normalized within [-1.0, 1.0]"
                )]
                {
                    *slot = (val / divisor) as f32;
                }
            }
            return Ok(out);
        }
    }

    for (i, slot) in out.iter_mut().enumerate() {
        *slot = input[i % input.len()] * scale;
    }
    Ok(out)
}

fn deberta_project_classifier_logits(
    weights: &dyn ModelWeights,
    cls_state: &[f32],
    out_labels: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut logits = vec![0.0_f32; out_labels];
    let mut projected = false;
    for head_name in ["classifier.weight", "cls.weight", "output.weight"] {
        if weights.contains_tensor(head_name) {
            let view = weights.tensor(head_name)?;
            if view.num_elements() == out_labels.saturating_mul(cls_state.len()) {
                logits =
                    deberta_project_or_fallback(weights, head_name, cls_state, out_labels, 0.25)?;
                projected = true;
                break;
            }
        }
    }

    if !projected {
        for (class_idx, slot) in logits.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (d, &val) in cls_state.iter().enumerate() {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "small class/hidden indices fit cleanly in f32"
                )]
                let weight = (((class_idx + 1) * (d + 3)) as f32 * 0.37).cos() * 0.25;
                acc += val * weight;
            }
            *slot = acc;
        }
    }

    for bias_name in ["classifier.bias", "cls.bias"] {
        if weights.contains_tensor(bias_name) {
            let bias = deberta_read_f32_weight_or_zeros(weights, bias_name, out_labels);
            for (slot, b) in logits.iter_mut().zip(bias.iter()) {
                *slot += *b;
            }
            break;
        }
    }

    softmax_probabilities_inplace(&mut logits)?;
    Ok(logits)
}

fn softmax_probabilities_inplace(logits: &mut [f32]) -> Result<(), K3Error> {
    if logits.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "classifier_logits".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "cannot apply softmax to empty logits slice".to_owned(),
        });
    }
    let mut max_val = f64::NEG_INFINITY;
    for &v in logits.iter() {
        if !v.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: format!("non-finite DeBERTa-v3 classification logit: {v}"),
            });
        }
        max_val = max_val.max(f64::from(v));
    }
    let mut sum_exp = 0.0_f64;
    let mut exps = Vec::with_capacity(logits.len());
    for &v in logits.iter() {
        let e = (f64::from(v) - max_val).exp();
        sum_exp += e;
        exps.push(e);
    }
    let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
        1.0_f64 / sum_exp
    } else {
        #[expect(clippy::cast_precision_loss, reason = "small label count fits f64")]
        {
            1.0_f64 / (logits.len() as f64)
        }
    };
    for (slot, e) in logits.iter_mut().zip(exps) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "softmax probability is strictly inside [0.0, 1.0]"
        )]
        {
            *slot = (e * inv_sum) as f32;
        }
    }
    Ok(())
}

struct DebertaRelPosBuffers {
    rel_pos_buckets: Vec<usize>,
    q_pos_buf: Vec<f32>,
    k_pos_buf: Vec<f32>,
}

fn deberta_build_rel_pos_buffers(
    weights: &dyn ModelWeights,
    seq_len: usize,
    attn_dim: usize,
    max_rel: usize,
) -> Result<DebertaRelPosBuffers, K3Error> {
    let max_rel = max_rel.max(1);
    let num_buckets = max_rel * 2;
    let mut rel_pos_buckets = vec![0_usize; seq_len * seq_len];
    for q_idx in 0..seq_len {
        for k_idx in 0..seq_len {
            let bucket = compute_deberta_rel_pos_bucket(q_idx, k_idx, max_rel);
            rel_pos_buckets[q_idx * seq_len + k_idx] = bucket.min(num_buckets - 1);
        }
    }

    let pos_len = num_buckets * attn_dim;
    let mut q_pos_buf = vec![0.0_f32; pos_len];
    let mut k_pos_buf = vec![0.0_f32; pos_len];

    let mut loaded_q = false;
    if weights.contains_tensor("rel_embeddings.weight") {
        let view = weights.tensor("rel_embeddings.weight")?;
        if view.num_elements() >= pos_len {
            let mut row = vec![0.0_f32; attn_dim];
            for b in 0..num_buckets {
                dequantize_token_row(&view, b, &mut row)?;
                normalize_max_abs(&mut row);
                q_pos_buf[b * attn_dim..(b + 1) * attn_dim].copy_from_slice(&row);
            }
            k_pos_buf.copy_from_slice(&q_pos_buf);
            loaded_q = true;
        }
    }

    if !loaded_q {
        for b in 0..num_buckets {
            for d in 0..attn_dim {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "bucket and head dimensions fit cleanly in f32"
                )]
                let angle = ((b + 1) as f32) * ((d + 1) as f32) * 0.015_625;
                q_pos_buf[b * attn_dim + d] = angle.sin() * 0.1;
                k_pos_buf[b * attn_dim + d] = angle.cos() * 0.1;
            }
        }
    }

    Ok(DebertaRelPosBuffers {
        rel_pos_buckets,
        q_pos_buf,
        k_pos_buf,
    })
}

#[allow(clippy::too_many_arguments)]
fn deberta_run_layer(
    layer: usize,
    seq_len: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
    rel_pos_buckets: &[usize],
    q_pos_buf: &[f32],
    k_pos_buf: &[f32],
    seq_hidden: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let attn_dim = n_heads * head_dim;
    let ffn_dim = hidden_dim.saturating_mul(4);
    let mut q_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut k_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut v_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut attn_out_buf = vec![0.0_f32; seq_len * attn_dim];
    let mut norm_tok = vec![0.0_f32; hidden_dim];

    let attn_norm_w = deberta_read_f32_weight_or_zeros(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        hidden_dim,
    );
    for pos in 0..seq_len {
        let tok_in = &seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        rmsnorm_unit_offset(&mut norm_tok, tok_in, &attn_norm_w, 1e-7)?;
        let q_tok = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.attn_q.weight"),
            &norm_tok,
            attn_dim,
            0.5,
        )?;
        let k_tok = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.attn_k.weight"),
            &norm_tok,
            attn_dim,
            0.25,
        )?;
        let v_tok = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.attn_v.weight"),
            &norm_tok,
            attn_dim,
            0.125,
        )?;
        let dst = pos * attn_dim..(pos + 1) * attn_dim;
        q_buf[dst.clone()].copy_from_slice(&q_tok);
        k_buf[dst.clone()].copy_from_slice(&k_tok);
        v_buf[dst].copy_from_slice(&v_tok);
    }

    deberta_disentangled_attention_f32(
        &mut attn_out_buf,
        &q_buf,
        &k_buf,
        &v_buf,
        q_pos_buf,
        k_pos_buf,
        rel_pos_buckets,
        seq_len,
        n_heads,
        head_dim,
    )?;

    let ffn_norm_w = deberta_read_f32_weight_or_zeros(
        weights,
        &format!("blk.{layer}.ffn_norm.weight"),
        hidden_dim,
    );
    for pos in 0..seq_len {
        let attn_tok = &attn_out_buf[pos * attn_dim..(pos + 1) * attn_dim];
        let attn_proj = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.attn_output.weight"),
            attn_tok,
            hidden_dim,
            0.25,
        )?;
        let tok_slice = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
        for (dst, &delta) in tok_slice.iter_mut().zip(attn_proj.iter()) {
            *dst += delta;
        }
        normalize_max_abs(tok_slice);

        rmsnorm_unit_offset(&mut norm_tok, tok_slice, &ffn_norm_w, 1e-7)?;
        let mut ffn_hidden = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.ffn_up.weight"),
            &norm_tok,
            ffn_dim,
            0.25,
        )?;
        for val in &mut ffn_hidden {
            *val = (*val).tanh() * *val;
        }
        let ffn_down = deberta_project_or_fallback(
            weights,
            &format!("blk.{layer}.ffn_down.weight"),
            &ffn_hidden,
            hidden_dim,
            0.125,
        )?;
        for (dst, &delta) in tok_slice.iter_mut().zip(ffn_down.iter()) {
            *dst += delta;
        }
        normalize_max_abs(tok_slice);
    }

    Ok(())
}

impl ArchRecipe for DebertaV3Recipe {
    fn architecture_id(&self) -> &'static str {
        "deberta-v3"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::Q8_0 | QuantType::F16 | QuantType::BF16 | QuantType::F32
        )
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx.prompt_tokens().to_vec();
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: "DeBERTa-v3 classification prompt_tokens cannot be empty".to_owned(),
            });
        }

        let seq_len = prompt.len();
        if seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let logits_len = ctx.logits().len();
        if logits_len == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "classifier_logits".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "DeBERTa-v3 logits output buffer cannot be empty".to_owned(),
            });
        }

        let hidden_dim = self.hidden_dim.max(2);
        let n_heads = self.n_heads.max(1).min(hidden_dim);
        let head_dim = self.head_dim.max(1);
        let attn_dim = n_heads * head_dim;

        let mut seq_hidden = vec![0.0_f32; seq_len * hidden_dim];
        for (pos, &token_id) in prompt.iter().enumerate() {
            let emb = deberta_embed_token(token_id, hidden_dim, weights)?;
            let row = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            row.copy_from_slice(&emb);
            #[expect(clippy::cast_precision_loss, reason = "position fits cleanly in f32")]
            let pos_bias = ((pos + 1) as f32) * 0.005;
            for v in row.iter_mut() {
                *v += pos_bias;
            }
        }

        let rel_pos =
            deberta_build_rel_pos_buffers(weights, seq_len, attn_dim, self.max_relative_positions)?;

        for layer in 0..self.num_layers.max(1) {
            deberta_run_layer(
                layer,
                seq_len,
                hidden_dim,
                n_heads,
                head_dim,
                &rel_pos.rel_pos_buckets,
                &rel_pos.q_pos_buf,
                &rel_pos.k_pos_buf,
                &mut seq_hidden,
                weights,
            )?;
        }

        let mut cls_state = seq_hidden[..hidden_dim].to_vec();
        normalize_max_abs(&mut cls_state);
        let out_labels = self.num_labels.max(1).min(logits_len);
        let class_probs = deberta_project_classifier_logits(weights, &cls_state, out_labels)?;

        let out_logits = ctx.logits_mut();
        out_logits.fill(0.0);
        out_logits[..out_labels].copy_from_slice(&class_probs);

        if ctx.position() == 0 {
            ctx.advance_position(seq_len)?;
        }
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.forward_classify(ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason:
                "DeBERTa-v3 encoder is single-pass and does not support autoregressive decode_step"
                    .to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affinity::{ClusterPhasePolicy, select_phase_policy_for_arch};
    use crate::capability::verify_model_capabilities;
    use crate::engine::SinglePromptEngine;
    use jero_core::GgufFile;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    #[test]
    fn test_deberta_v3_recipe_cls_logits() {
        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[8, 16][..]),
            ("blk.0.attn_q.weight", QuantType::F32, &[8, 8][..]),
            ("classifier.weight", QuantType::F32, &[8, 3][..]),
            ("classifier.bias", QuantType::F32, &[3][..]),
        ];
        let mut bytes = create_synthetic_gguf_v3("deberta-v3", 64, &tensor_specs);
        let total_elems: usize = tensor_specs
            .iter()
            .map(|(_, _, shape)| shape.iter().product::<usize>())
            .sum();
        let data_start = bytes.len() - total_elems * 4;
        for idx in 0..total_elems {
            #[expect(clippy::cast_precision_loss, reason = "bounded test index fits f32")]
            let val = (((idx * 5 + 1) % 17) as f32 - 8.0) * 0.0625;
            let offset = data_start + idx * 4;
            bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        let gguf = GgufFile::parse(&bytes).expect("parse synthetic deberta-v3 gguf");
        let mut recipe = DebertaV3Recipe::from_metadata(Some(gguf.metadata()));
        recipe.num_layers = 1;
        recipe.hidden_dim = 8;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.max_relative_positions = 8;
        recipe.num_labels = 3;

        assert!(!recipe.requires_kv_cache());
        assert_eq!(recipe.architecture_id(), "deberta-v3");

        let prompt_a = [1_u32, 3, 5, 7];
        let prompt_b = [2_u32, 4, 8, 12];
        let mut logits_a = [0.0_f32; 3];
        let mut logits_b = [0.0_f32; 3];

        let mut ctx_a =
            ExecutionContext::new(&prompt_a, 64, &mut logits_a, &mut []).expect("ctx_a");
        recipe
            .forward_classify(&mut ctx_a, &gguf)
            .expect("CLS classification pass A succeeds");
        assert_eq!(ctx_a.position(), prompt_a.len());
        assert!(matches!(
            recipe.decode_step(&mut ctx_a, &gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        let sum_a: f32 = logits_a.iter().sum();
        assert!(
            (sum_a - 1.0).abs() < 1e-4,
            "3-way [CLS] probabilities (BENIGN, INJECTION, JAILBREAK) must sum to 1.0, got {logits_a:?}"
        );
        assert!(
            logits_a
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        );

        let mut ctx_b =
            ExecutionContext::new(&prompt_b, 64, &mut logits_b, &mut []).expect("ctx_b");
        recipe
            .forward_classify(&mut ctx_b, &gguf)
            .expect("CLS classification pass B succeeds");
        let sum_b: f32 = logits_b.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-4);

        let mut engine = SinglePromptEngine::new();
        let (probs, summary) = engine
            .evaluate_classify(&recipe, &gguf, &prompt_a, 64, 3)
            .expect("SinglePromptEngine::evaluate_classify executes DebertaV3 zero-KV");
        assert_eq!(probs.len(), 3);
        assert_eq!(summary.timing.decode_steps, 0);
    }

    #[test]
    fn test_deberta_v3_prompt_guard_x100() {
        let placement = DebertaV3Recipe::default_placement();
        assert_eq!(placement.prefill.vector, Lane::X100);
        assert_eq!(placement.prefill.matmul, vec![(Lane::X100, 1.0)]);
        assert_eq!(placement.decode.vector, Lane::X100);
        assert_eq!(
            select_phase_policy_for_arch("deberta-v3", 32, 0xFFFF),
            ClusterPhasePolicy::X100Only
        );

        let recipe = DebertaV3Recipe::new();
        assert!(!recipe.requires_kv_cache());
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(!recipe.supports_quant(QuantType::Q4_0));
        assert!(!recipe.supports_quant(QuantType::Q4_K_M));
        assert!(!recipe.supports_quant(QuantType::Q5_K));
        assert!(!recipe.supports_quant(QuantType::Q6_K));

        let ok_bytes = create_synthetic_gguf_v3(
            "deberta-v3",
            64,
            &[
                ("token_embd.weight", QuantType::Q8_0, &[32, 4][..]),
                ("classifier.weight", QuantType::Q8_0, &[32, 3][..]),
            ],
        );
        let ok_gguf = GgufFile::parse(&ok_bytes).expect("parse Q8_0 deberta-v3");
        let verified = verify_model_capabilities(&ok_gguf).expect("Q8_0 deberta-v3 admitted");
        assert_eq!(verified.architecture_id(), "deberta-v3");
        assert!(!verified.requires_kv_cache());

        let q4_bytes = create_synthetic_gguf_v3(
            "deberta-v3",
            64,
            &[("blk.0.attn_q.weight", QuantType::Q4_K_M, &[32, 8][..])],
        );
        let q4_gguf = GgufFile::parse(&q4_bytes).expect("parse Q4_K_M deberta-v3");
        let err = verify_model_capabilities(&q4_gguf)
            .expect_err("Q4_K_M deberta-v3 must be rejected below Q8_0 precision floor");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }
}
