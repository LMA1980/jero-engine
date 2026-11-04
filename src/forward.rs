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

use k3_core::K3Error;
#[allow(unused_imports)]
use kernels_rvv::ime::{
    TILE_C, TILE_K, TILE_M, TILE_N, dot_i8_vector, gemm_i8_ime, gemm_i8_vector, ime_probe_exec,
    ime_vmadot_count, pack_tiles, tile_count, vector_dot_count,
};

use crate::affinity::{ClusterPreset, WorkerPool};
use crate::engine::{PassTiming, compute_pass_timing};

#[cfg(feature = "arch-bitnet")]
pub use k3_arch_bitnet::BitNetRecipe;

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
