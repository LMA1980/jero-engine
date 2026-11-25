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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use jero_core::K3Error;
#[allow(unused_imports)]
use jero_kernels_rvv::ime::{
    TILE_C, TILE_K, TILE_M, TILE_N, dot_i8_vector, gemm_i8_ime, gemm_i8_vector, ime_probe_exec,
    ime_vmadot_count, pack_tiles, tile_count, vector_dot_count,
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

#[cfg(feature = "jero-arch-machine")]
pub use crate::machine_models::{
    DecisionTransformerRecipe, FtTransformerRecipe, MmprojLatentBridge, TabPfnV2Recipe, TtmRecipe,
    forward_with_soft_prompt, project_machine_latent_to_soft_prompt,
};
#[cfg(feature = "jero-arch-encoder")]
pub use crate::modernbert::{
    CrossEncoderRerankRecipe, DebertaV3Recipe, EncoderFfnActivation, ModernBertRecipe,
    RerankScoringMode, resolve_encoder_recipe,
};

pub use crate::memory::HybridLayerKind;
#[cfg(feature = "jero-arch-qwen")]
pub use crate::qwen_deepseek::{
    DeepSeek2MlaConfig, DeepSeek2Recipe, Qwen2Recipe, Qwen3MoePlacementExt, Qwen3MoeRecipe,
    Qwen3Recipe, Qwen35MoeRecipe, Qwen35Recipe, QwenEmbeddingRecipe,
};
#[cfg(feature = "jero-arch-ssm")]
pub use crate::ssm::{
    GraniteRecipe, HybridSsmTransformerRecipe, Lfm2MoeRecipe, Mamba2Recipe, Rwkv7Recipe,
    lfm2_shortconv_step, resolve_hybrid_layer_schedule,
    resolve_hybrid_layer_schedule_with_metadata,
};
#[cfg(feature = "jero-arch-vision")]
pub use jero_arch_vision::{
    ClipVisionRecipe, SiglipVisionRecipe, VisionEncoderConfig, VisionTokens, VlmMmprojBridge,
};
#[cfg(feature = "jero-arch-whisper")]
pub use jero_arch_whisper::{
    WhisperRecipe, WhisperVariant, build_slaney_mel_filters, pcm_to_log_mel,
    pcm_to_log_mel_with_filters,
};

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

/// Minimum output row count required before cross-cluster `RowSplit` fans out to both clusters.
/// Projections with fewer than `ROW_SPLIT_MIN_ROWS` output rows stay on the dominant cluster.
pub const ROW_SPLIT_MIN_ROWS: usize = 256;
/// Output row alignment granularity for single-row GEMV (`m == 1`, Vector kernel).
pub const GEMV_ROW_ALIGN: usize = 4;
/// Output row alignment granularity for tiled / IME matrix kernels (`TILE_N = 8`).
pub const IME_ROW_ALIGN: usize = TILE_N;

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

/// Computes an aligned row index partitioning `0..n_rows` across two clusters according to
/// `first_share` (`[0.0, 1.0]`).
///
/// Bypasses cross-cluster splitting when `n_rows < ROW_SPLIT_MIN_ROWS`, returning `n_rows` when
/// `first_share > 0.5` (all work on the first cluster) or `0` otherwise (all work on the second
/// cluster). For larger matrices, rounds the split point to the nearest multiple of `align`
/// (`GEMV_ROW_ALIGN = 4` for single-row Vector GEMV; `TILE_N = 8` for IME or batched projections).
#[must_use]
pub fn calibrate_split_row(n_rows: usize, first_share: f64, align: usize) -> usize {
    if n_rows == 0 {
        return 0;
    }
    let share = if first_share.is_finite() {
        first_share.clamp(0.0, 1.0)
    } else {
        0.5
    };
    if n_rows < ROW_SPLIT_MIN_ROWS {
        return if share > 0.5 { n_rows } else { 0 };
    }
    let align = align.max(1);
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "matrix row counts fit cleanly in f64 and share is clamped to [0, 1]"
    )]
    let approx = (n_rows as f64 * share).round() as usize;
    let aligned = ((approx + align / 2) / align) * align;
    aligned.min(n_rows)
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
    crate::affinity::current_execution_cpu_mask()
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

#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);

impl<T> SyncMutPtr<T> {
    const fn as_ptr(&self) -> *mut T {
        self.0
    }
}

// SAFETY: Matrix and vector workers write strictly disjoint index ranges guarded by barrier join.
unsafe impl<T: Send> Send for SyncMutPtr<T> {}
// SAFETY: Shared read-only copy of disjoint base pointers for `Send` elements.
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}

fn worker_range(range: &Range<usize>, idx: usize, workers: usize) -> Range<usize> {
    let len = range.end - range.start;
    let chunk = len.div_ceil(workers.max(1));
    let lo = (range.start + idx * chunk).min(range.end);
    let hi = (lo + chunk).min(range.end);
    lo..hi
}

fn aligned_row_subrange(
    range: &Range<usize>,
    idx: usize,
    workers: usize,
    align: usize,
) -> Range<usize> {
    if range.is_empty() {
        return range.end..range.end;
    }
    let align = align.max(1);
    let len = range.end - range.start;
    let units = len.div_ceil(align);
    let workers = workers.max(1);
    let u_lo = (idx * units) / workers;
    let u_hi = ((idx + 1) * units) / workers;
    let lo = (range.start + u_lo * align).min(range.end);
    let hi = (range.start + u_hi * align).min(range.end);
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

fn validate_lane(pool: &WorkerPool, lane: Lane) -> Result<(), K3Error> {
    if (pool.x100_pool().is_some() && pool.a100_pool().is_some())
        || pool.affinity().cluster == Some(lane.preset())
    {
        Ok(())
    } else {
        Err(lane_error(lane))
    }
}

fn run_lane_scoped<F>(pool: &WorkerPool, lane: Lane, task: F) -> Result<(), K3Error>
where
    F: Fn(usize, usize) + Sync,
{
    if let (Some(x), Some(a)) = (pool.x100_pool(), pool.a100_pool()) {
        match lane {
            Lane::X100 => x.execute_scoped(&task),
            Lane::A100 => a.execute_scoped(&task),
        }
        Ok(())
    } else if pool.affinity().cluster == Some(lane.preset()) {
        pool.execute_scoped(&task);
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
    matmul_cpus: AtomicU64,
    vector_cpus: AtomicU64,
}

impl<'a> ForwardExec<'a> {
    /// Creates a dispatch context over `pool` using `kernel` for matrix multiplications.
    #[must_use]
    pub const fn new(pool: &'a WorkerPool, kernel: MatKernel) -> Self {
        Self {
            pool,
            kernel,
            matmul_cpus: AtomicU64::new(0),
            vector_cpus: AtomicU64::new(0),
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

    const fn lane_kernel(&self, lane: Lane) -> MatKernel {
        match (self.kernel, lane) {
            (MatKernel::Ime, Lane::A100) => MatKernel::Ime,
            _ => MatKernel::Vector,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches both m=1 split and batched m>1 paths"
    )]
    fn run_matmul_slice<'b>(
        &'b self,
        lane: Lane,
        w: &'b Matrix,
        x: &'b [i8],
        packed_x: Option<&'b [i8]>,
        m: usize,
        abs_rows: &Range<usize>,
        out_ptr: SyncMutPtr<i32>,
        out_row_offset: usize,
        out_stride: usize,
    ) -> impl Fn(usize, usize) + Sync + 'b {
        let lk = self.lane_kernel(lane);
        let align = match lk {
            MatKernel::Ime => TILE_N,
            MatKernel::Vector => GEMV_ROW_ALIGN,
        };
        let rows = abs_rows.clone();
        move |idx, workers| {
            let sub = aligned_row_subrange(&rows, idx, workers, align);
            if sub.is_empty() {
                return;
            }
            self.matmul_cpus.fetch_or(cpu_bit(), Ordering::Relaxed);
            // SAFETY: `sub` ranges are mutually disjoint across `idx in 0..workers` and reside
            // inside `out_ptr`'s caller-owned `out` buffer until synchronous barrier completion.
            unsafe {
                matmul_rows(
                    lk,
                    w,
                    x,
                    packed_x,
                    m,
                    sub,
                    out_ptr,
                    out_row_offset,
                    out_stride,
                );
            }
        }
    }

    /// `out[m x n] = x[m x k] * w^T` in exact `i32`, partitioned over `layout.matmul` lanes.
    #[allow(clippy::many_single_char_names)]
    fn matmul(
        &self,
        layout: &PhaseLayout,
        w: &Matrix,
        x: &[i8],
        m: usize,
    ) -> Result<Vec<i32>, K3Error> {
        let n = w.n;
        let k = w.k;
        let mut out = vec![0_i32; m * n];
        if layout.matmul.is_empty() || m == 0 || n == 0 {
            return Ok(out);
        }

        let packed_x = (self.kernel == MatKernel::Ime).then(|| pack_tiles(x, m, k));
        let packed_ref = packed_x.as_deref();

        if layout.matmul.len() == 1 {
            let lane = layout.matmul[0].0;
            let p = SyncMutPtr(out.as_mut_ptr());
            let task = self.run_matmul_slice(lane, w, x, packed_ref, m, &(0..n), p, 0, n);
            run_lane_scoped(self.pool, lane, task)?;
            return Ok(out);
        }

        if layout.matmul.len() == 2 {
            let (lane0, share0) = layout.matmul[0];
            let (lane1, _share1) = layout.matmul[1];
            let has_ime_lane =
                self.kernel == MatKernel::Ime && (lane0 == Lane::A100 || lane1 == Lane::A100);
            let row_align = if m == 1 && !has_ime_lane {
                GEMV_ROW_ALIGN
            } else {
                IME_ROW_ALIGN
            };
            let split_row = calibrate_split_row(n, share0, row_align);
            if split_row == 0 {
                let p = SyncMutPtr(out.as_mut_ptr());
                let task = self.run_matmul_slice(lane1, w, x, packed_ref, m, &(0..n), p, 0, n);
                run_lane_scoped(self.pool, lane1, task)?;
                return Ok(out);
            }
            if split_row == n {
                let p = SyncMutPtr(out.as_mut_ptr());
                let task = self.run_matmul_slice(lane0, w, x, packed_ref, m, &(0..n), p, 0, n);
                run_lane_scoped(self.pool, lane0, task)?;
                return Ok(out);
            }

            validate_lane(self.pool, lane0)?;
            validate_lane(self.pool, lane1)?;

            if lane0 != lane1 && self.pool.is_dual() {
                let (r0, r1) = (0..split_row, split_row..n);
                let (p0, p1, off1) = if m == 1 {
                    let (s0, s1) = out.split_at_mut(split_row);
                    (
                        SyncMutPtr(s0.as_mut_ptr()),
                        SyncMutPtr(s1.as_mut_ptr()),
                        split_row,
                    )
                } else {
                    (
                        SyncMutPtr(out.as_mut_ptr()),
                        SyncMutPtr(out.as_mut_ptr()),
                        0,
                    )
                };
                let t0 = self.run_matmul_slice(lane0, w, x, packed_ref, m, &r0, p0, 0, n);
                let t1 = self.run_matmul_slice(lane1, w, x, packed_ref, m, &r1, p1, off1, n);
                match (lane0, lane1) {
                    (Lane::X100, Lane::A100) => self.pool.execute_split_scoped(&t0, &t1),
                    (Lane::A100, Lane::X100) => self.pool.execute_split_scoped(&t1, &t0),
                    _ => unreachable!("distinct dual lanes validated"),
                }
                return Ok(out);
            }
        }

        // General multi-lane sequential fallback for synthetic plans with > 2 entries.
        let n_tiles = n / TILE_N;
        let mut start = 0_usize;
        let mut acc_share = 0.0_f64;
        let p = SyncMutPtr(out.as_mut_ptr());
        for (idx, &(lane, share)) in layout.matmul.iter().enumerate() {
            acc_share += share;
            let end_tile = if idx + 1 == layout.matmul.len() {
                n_tiles
            } else {
                partition_tiles(n_tiles, acc_share)
            };
            let range = (start * TILE_N)..(end_tile.max(start) * TILE_N);
            start = end_tile.max(start);
            if range.is_empty() {
                continue;
            }
            let task = self.run_matmul_slice(lane, w, x, packed_ref, m, &range, p, 0, n);
            run_lane_scoped(self.pool, lane, task)?;
        }
        Ok(out)
    }

    /// Runs `body(row_range)` over `rows` rows on the vector lane, partitioned across its workers.
    fn rows_on_lane<F>(&self, lane: Lane, rows: usize, body: F) -> Result<(), K3Error>
    where
        F: Fn(Range<usize>) + Sync,
    {
        run_lane_scoped(self.pool, lane, |idx, workers| {
            let r = worker_range(&(0..rows), idx, workers);
            if r.is_empty() {
                return;
            }
            self.vector_cpus.fetch_or(cpu_bit(), Ordering::Relaxed);
            body(r);
        })
    }

    fn rmsnorm(&self, lane: Lane, x: &[i8], m: usize) -> Result<Vec<i8>, K3Error> {
        let mut out = vec![0_i8; m * MODEL_DIM];
        let out_ptr = SyncMutPtr(out.as_mut_ptr());
        self.rows_on_lane(lane, m, |rows| {
            for r in rows {
                let row = &x[r * MODEL_DIM..(r + 1) * MODEL_DIM];
                let ms: u64 = row
                    .iter()
                    .map(|&v| u64::from(i32::from(v).unsigned_abs().pow(2)))
                    .sum();
                let rms = i32::try_from(isqrt(ms / MODEL_DIM as u64))
                    .unwrap_or(i32::MAX)
                    .max(1);
                // SAFETY: Distinct `r` across workers writes disjoint `[r*MODEL_DIM..(r+1)*MODEL_DIM]` rows.
                let out_row = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.as_ptr().add(r * MODEL_DIM), MODEL_DIM)
                };
                for (dst, &v) in out_row.iter_mut().zip(row.iter()) {
                    let y = (i32::from(v) * 48 / rms).clamp(-127, 127);
                    #[expect(clippy::cast_possible_truncation, reason = "clamped to [-127, 127]")]
                    {
                        *dst = y as i8;
                    }
                }
            }
        })?;
        Ok(out)
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
        let mut out = vec![0_i8; m * MODEL_DIM];
        let out_ptr = SyncMutPtr(out.as_mut_ptr());
        self.rows_on_lane(lane, m, |rows| {
            let mut acc = vec![0_i64; MODEL_DIM];
            for r in rows {
                let pos = start_pos + r;
                let qrow = &q[r * MODEL_DIM..(r + 1) * MODEL_DIM];
                acc.fill(0);
                let mut total_w = 0_i64;
                for t in 0..=pos {
                    let s = dot_i8_vector(qrow, &k_cache[t * MODEL_DIM..(t + 1) * MODEL_DIM]);
                    let w = i64::from((s >> 10).clamp(0, 127)) + 1;
                    total_w += w;
                    let vrow = &v_cache[t * MODEL_DIM..(t + 1) * MODEL_DIM];
                    for (a, &v) in acc.iter_mut().zip(vrow) {
                        *a += w * i64::from(v);
                    }
                }
                // SAFETY: Distinct `r` across workers writes disjoint `[r*MODEL_DIM..(r+1)*MODEL_DIM]` rows.
                let out_row = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.as_ptr().add(r * MODEL_DIM), MODEL_DIM)
                };
                for (dst, &a) in out_row.iter_mut().zip(acc.iter()) {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "weighted mean of i8 values"
                    )]
                    let y = (a / total_w) as i8;
                    *dst = y;
                }
            }
        })?;
        Ok(out)
    }
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

// SAFETY: Every `(row, col)` with `row < m` and `col in abs_rows` maps to a strictly disjoint
// offset `(row * out_stride) + (col - out_row_offset)` inside caller's mutable slice.
#[expect(clippy::too_many_arguments, reason = "kernel row-slice dispatch")]
unsafe fn matmul_rows(
    kernel: MatKernel,
    w: &Matrix,
    x: &[i8],
    packed_x: Option<&[i8]>,
    m: usize,
    abs_rows: Range<usize>,
    out_ptr: SyncMutPtr<i32>,
    out_row_offset: usize,
    out_stride: usize,
) {
    match kernel {
        MatKernel::Ime => {
            let a = packed_x.expect("IME operand is packed");
            let tile_start = abs_rows.start / TILE_N;
            let tile_end = abs_rows.end / TILE_N;
            gemm_i8_ime(
                a,
                tile_count(m),
                &w.packed,
                tile_start..tile_end,
                w.k / TILE_K,
                |mt, nt, c: &[i32; TILE_C]| {
                    for i in 0..TILE_M {
                        let row = mt * TILE_M + i;
                        if row >= m {
                            break;
                        }
                        for j in 0..TILE_N {
                            let col = nt * TILE_N + j;
                            if col >= abs_rows.start && col < abs_rows.end {
                                let dst_idx = row * out_stride + (col - out_row_offset);
                                // SAFETY: Disjoint element guaranteed by `abs_rows` ownership.
                                unsafe { out_ptr.as_ptr().add(dst_idx).write(c[i * TILE_N + j]) };
                            }
                        }
                    }
                },
            );
        }
        MatKernel::Vector => {
            gemm_i8_vector(x, m, &w.rows, abs_rows, w.k, |row, col, v| {
                let dst_idx = row * out_stride + (col - out_row_offset);
                // SAFETY: Disjoint element guaranteed by `abs_rows` ownership.
                unsafe { out_ptr.as_ptr().add(dst_idx).write(v) };
            });
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
        let (ttft_ms, prefill_ms, prefill_nanos);
        let mut decode_ms = 0.0;
        if max_tokens == 0 {
            let prefill_elapsed = prefill_start.elapsed();
            prefill_ms = prefill_elapsed.as_secs_f64() * 1000.0;
            prefill_nanos = u64::try_from(prefill_elapsed.as_nanos()).unwrap_or(u64::MAX);
            ttft_ms = prefill_ms;
        } else {
            let mut tok = self.sample(&logits);
            generated.push(tok);
            let prefill_elapsed = prefill_start.elapsed();
            prefill_ms = prefill_elapsed.as_secs_f64() * 1000.0;
            prefill_nanos = u64::try_from(prefill_elapsed.as_nanos()).unwrap_or(u64::MAX);
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
            prompt.len(),
            prefill_nanos,
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
    let rates: Mutex<Vec<f64>> = Mutex::new(Vec::new());
    run_lane_scoped(pool, lane, |_, _| {
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
        rates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(rate);
    })?;
    let total = rates
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
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
    let buf: Vec<u64> = (0..WORDS as u64).collect();
    let rates: Mutex<Vec<f64>> = Mutex::new(Vec::new());
    run_lane_scoped(pool, lane, |idx, workers| {
        let r = worker_range(&(0..WORDS), idx, workers);
        let slice = &buf[r];
        let start = Instant::now();
        let mut acc = 0_u64;
        for _ in 0..2 {
            for &v in slice {
                acc = acc.wrapping_add(v);
            }
        }
        std::hint::black_box(acc);
        let rate = (slice.len() * 8 * 2) as f64 / start.elapsed().as_secs_f64().max(1e-9);
        rates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(rate);
    })?;
    let total = rates
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .sum();
    Ok(total)
}

/// Measures dual-cluster DRAM streaming bandwidth via [`measure_memory_bandwidth`] on both
/// `Lane::X100` and `Lane::A100` and returns `(x100_bytes_per_sec, a100_bytes_per_sec, x100_share)`.
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if `pool` lacks workers for either cluster, or
/// [`K3Error::CapabilityMismatch`] if either measured bandwidth is non-positive.
pub fn calibrate_bandwidth_split_ratio(pool: &WorkerPool) -> Result<(f64, f64, f64), K3Error> {
    let x100_bw = measure_memory_bandwidth(pool, Lane::X100)?;
    let a100_bw = measure_memory_bandwidth(pool, Lane::A100)?;
    let share = row_split_share(x100_bw, a100_bw)?;
    Ok((x100_bw, a100_bw, share))
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

/// Cluster lane placement metadata extension for [`DebertaV3Recipe`].
#[cfg(feature = "jero-arch-encoder")]
pub trait DebertaV3PlacementExt {
    /// Returns the default `X100`-only cluster placement plan for `DeBERTa-v3` prompt guards.
    fn default_placement() -> PlacementPlan;
}

#[cfg(feature = "jero-arch-encoder")]
impl DebertaV3PlacementExt for DebertaV3Recipe {
    fn default_placement() -> PlacementPlan {
        PlacementPlan::single(Lane::X100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affinity::{ClusterPhasePolicy, select_phase_policy_for_arch};
    use crate::capability::verify_model_capabilities;
    use crate::engine::SinglePromptEngine;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    #[cfg(feature = "jero-arch-encoder")]
    fn build_deberta_v3_test_gguf(
        hidden_dim: usize,
        ffn_dim: usize,
        num_buckets: usize,
        num_labels: usize,
    ) -> Vec<u8> {
        use jero_core::QuantType;

        let specs = [
            ("token_embd.weight", vec![hidden_dim, 16]),
            ("rel_embeddings.weight", vec![hidden_dim, num_buckets]),
            ("blk.0.attn_norm.weight", vec![hidden_dim]),
            ("blk.0.attn_q.weight", vec![hidden_dim, hidden_dim]),
            ("blk.0.attn_k.weight", vec![hidden_dim, hidden_dim]),
            ("blk.0.attn_v.weight", vec![hidden_dim, hidden_dim]),
            ("blk.0.attn_output.weight", vec![hidden_dim, hidden_dim]),
            ("blk.0.ffn_norm.weight", vec![hidden_dim]),
            ("blk.0.ffn_up.weight", vec![hidden_dim, ffn_dim]),
            ("blk.0.ffn_down.weight", vec![ffn_dim, hidden_dim]),
            ("classifier.weight", vec![hidden_dim, num_labels]),
            ("classifier.bias", vec![num_labels]),
        ];
        let chunks: Vec<Vec<u8>> = specs
            .iter()
            .enumerate()
            .map(|(idx, (name, shape))| {
                let elems: usize = shape.iter().product();
                (0..elems)
                    .flat_map(|i| {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "bounded test index fits f32"
                        )]
                        let val = if name.ends_with("norm.weight") {
                            1.0_f32
                        } else {
                            (((i * 5 + idx * 3 + 1) % 17) as f32 - 8.0) * 0.0625
                        };
                        val.to_le_bytes()
                    })
                    .collect()
            })
            .collect();
        let borrowed: Vec<(&str, QuantType, &[usize])> = specs
            .iter()
            .map(|(n, s)| (*n, QuantType::F32, s.as_slice()))
            .collect();
        let mut bytes = create_synthetic_gguf_v3("deberta-v3", 64, &borrowed);
        let total_bytes: usize = chunks.iter().map(Vec::len).sum();
        let mut cursor = bytes.len() - total_bytes;
        for chunk in chunks {
            bytes[cursor..cursor + chunk.len()].copy_from_slice(&chunk);
            cursor += chunk.len();
        }
        bytes
    }

    #[cfg(feature = "jero-arch-encoder")]
    #[test]
    fn test_deberta_v3_recipe_cls_logits() {
        use jero_core::{ArchRecipe, ExecutionContext, GgufFile};

        let hidden_dim = 8_usize;
        let num_labels = 3_usize;
        let bytes = build_deberta_v3_test_gguf(hidden_dim, hidden_dim * 4, 16, num_labels);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic deberta-v3 gguf");
        let mut recipe = DebertaV3Recipe::from_metadata(Some(gguf.metadata()));
        recipe.num_layers = 1;
        recipe.hidden_dim = hidden_dim;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.max_relative_positions = 8;
        recipe.num_labels = num_labels;

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
            "3-way [CLS] probabilities must sum to 1.0, got {logits_a:?}"
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

    #[cfg(feature = "jero-arch-encoder")]
    #[test]
    fn test_deberta_v3_prompt_guard_x100() {
        use jero_core::{ArchRecipe, GgufFile, QuantType};

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

    #[test]
    fn test_calibrate_split_row_alignment_and_min_rows_threshold() {
        assert_eq!(ROW_SPLIT_MIN_ROWS, 256);
        assert_eq!(GEMV_ROW_ALIGN, 4);
        assert_eq!(IME_ROW_ALIGN, TILE_N);

        // Matrices smaller than ROW_SPLIT_MIN_ROWS bypass splitting cleanly.
        assert_eq!(calibrate_split_row(128, 0.75, GEMV_ROW_ALIGN), 128);
        assert_eq!(calibrate_split_row(128, 0.25, GEMV_ROW_ALIGN), 0);
        assert_eq!(calibrate_split_row(255, 0.60, IME_ROW_ALIGN), 255);
        assert_eq!(calibrate_split_row(0, 0.5, GEMV_ROW_ALIGN), 0);

        // Larger matrices align strictly to the requested lane granularity.
        let gemv_split = calibrate_split_row(513, 0.37, GEMV_ROW_ALIGN);
        assert_eq!(gemv_split % GEMV_ROW_ALIGN, 0);
        assert!(gemv_split > 0 && gemv_split <= 513);

        let ime_split = calibrate_split_row(512, 0.31, IME_ROW_ALIGN);
        assert_eq!(ime_split % IME_ROW_ALIGN, 0);
        assert!(ime_split > 0 && ime_split <= 512);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_split_matmul_bit_identical_and_zero_new_threads() {
        use crate::affinity::{ClusterPhasePolicy, MockCpuTopology};
        use jero_core::{CoordinationProfile, GgufFile, QuantType};
        use std::collections::HashSet;
        use std::sync::Mutex;

        let bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let topo = MockCpuTopology::spacemit_k3();
        for profile in [CoordinationProfile::Spin, CoordinationProfile::Futex] {
            let single_pool = WorkerPool::for_phase_policy(
                ClusterPhasePolicy::X100Only,
                gguf.metadata(),
                8,
                None,
                Some(2),
                profile,
            )
            .expect("single-cluster pool");
            let single_exec = ForwardExec::new(&single_pool, MatKernel::Vector);

            let Ok(dual_pool) = WorkerPool::for_phase_policy_with(
                ClusterPhasePolicy::RowSplit,
                gguf.metadata(),
                32,
                None,
                Some(2),
                profile,
                &topo,
            ) else {
                // Skip dual-cluster pin validation on <16-core/cpuset-restricted CI hosts.
                continue;
            };
            let (x100_bw, a100_bw, bw_share) =
                calibrate_bandwidth_split_ratio(&dual_pool).expect("bandwidth ratio calibration");
            assert!(bw_share.is_finite() && (0.0..=1.0).contains(&bw_share));
            assert!(x100_bw > 0.0 && a100_bw > 0.0);

            let split_exec = ForwardExec::new(&dual_pool, MatKernel::Vector);
            let single_layout = PlacementPlan::single(Lane::X100);
            let row_split_layout = PlacementPlan::row(bw_share, bw_share);

            // Check both < ROW_SPLIT_MIN_ROWS (128) and >= ROW_SPLIT_MIN_ROWS (512) for m = 1 & m = 4.
            for (m, n, k) in [(1, 128, 32), (1, 512, 64), (4, 512, 64)] {
                let w = Matrix::random(0x1234_5678 + n as u64, n, k, 6);
                let x = fill_small(0xDEAD_BEEF + m as u64, m * k);
                let ref_out = single_exec
                    .matmul(&single_layout.decode, &w, &x, m)
                    .expect("single-cluster reference matmul");
                let split_out = split_exec
                    .matmul(&row_split_layout.decode, &w, &x, m)
                    .expect("row-split matmul");
                assert_eq!(ref_out, split_out, "bit-identity mismatch for m={m}, n={n}");
            }

            // 100 consecutive RowSplit GEMV steps must reuse fixed OS worker threads only.
            let w512 = Matrix::random(0xCAFE_BABE, 512, 64, 6);
            let x512 = fill_small(0xFACE_B00C, 64);
            let seen_tids: Mutex<HashSet<std::thread::ThreadId>> = Mutex::new(HashSet::new());
            for _ in 0..100 {
                let out = split_exec
                    .matmul(&row_split_layout.decode, &w512, &x512, 1)
                    .expect("consecutive row-split matmul");
                assert_eq!(out.len(), w512.n);
            }
            dual_pool.execute_split_scoped(
                &|_, _| {
                    seen_tids
                        .lock()
                        .expect("seen tids")
                        .insert(std::thread::current().id());
                },
                &|_, _| {
                    seen_tids
                        .lock()
                        .expect("seen tids")
                        .insert(std::thread::current().id());
                },
            );
            let distinct = seen_tids.lock().expect("seen tids").len();
            assert!(
                distinct <= dual_pool.thread_count(),
                "dual pool with {} workers used {distinct} threads",
                dual_pool.thread_count()
            );
            single_pool.reset_barrier_wait();
        }
    }
}
