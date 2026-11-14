//! Linear-time State-Space (`Mamba-2`), Recurrent (`RWKV-6`/`RWKV-7`), Hybrid Attention-SSM/ShortConv
//! (`Jamba`, `Zamba2`, `GraniteHybrid`, `LFM2`, `LFM2.5-MoE`), and `Granite` multiplier execution recipes
//! (`#56`–`#64`, `#104`–`#107`).

use core::mem::size_of;

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType,
    RecurrentStateSlotHeader,
};
use jero_kernels_rvv::{
    lfm2_shortconv_step_f32, mamba2_conv1d_silu_chunk_f32, mamba2_conv1d_silu_step,
    mamba2_ssd_chunk_scan_f32, mamba2_ssd_intra_chunk_ime_i8, mamba2_ssd_state_step_f32,
    rwkv_token_shift_lerp_f32, rwkv7_wkv_delta_rule_step_f32, softplus_f32,
};

use crate::gemma::project_or_deterministic_slice;
use crate::memory::HybridLayerKind;
use crate::qwen_deepseek::{
    DenseAttnLayerSpec, add_bias_if_present, dispatch_qwen3_moe_ffn, embed_token, even_head_dim,
    is_supported_quant, project_lm_head, read_f32_weight_or_default, rmsnorm_standard,
    run_decode_with, run_dense_attention_sublayer_scaled, run_dense_swiglu_ffn, run_prefill_with,
};

const RECURRENT_HEADER_FLOATS: usize = size_of::<RecurrentStateSlotHeader>() / size_of::<f32>();
const RECURRENT_MAGIC_F32: f32 = 1_048_576.0;

#[inline]
fn resolve_divisible_ssm_geometry(
    d_inner: usize,
    requested_heads: usize,
    requested_groups: usize,
) -> (usize, usize, usize) {
    let inner = d_inner.max(1);
    let mut n_heads = requested_heads.clamp(1, inner);
    while n_heads > 1 && inner % n_heads != 0 {
        n_heads -= 1;
    }
    let head_dim = (inner / n_heads).max(1);
    let mut n_groups = requested_groups.clamp(1, n_heads);
    while n_groups > 1 && n_heads % n_groups != 0 {
        n_groups -= 1;
    }
    (n_heads, head_dim, n_groups)
}

#[inline]
fn resolve_norm_weight(
    weights: &dyn ModelWeights,
    primary: &str,
    fallback: &str,
    len: usize,
) -> Vec<f32> {
    if weights.contains_tensor(primary) {
        read_f32_weight_or_default(weights, primary, len, 1.0)
    } else {
        read_f32_weight_or_default(weights, fallback, len, 1.0)
    }
}

#[inline]
fn inv_kernel_f32(width: usize) -> f32 {
    1.0_f32 / f32::from(u16::try_from(width.max(1)).unwrap_or(u16::MAX))
}

fn finalize_scaled_logits(
    hidden: &[f32],
    rms_eps: f32,
    logit_scale: f32,
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let norm_out_w = read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, hidden, &norm_out_w, rms_eps)?;
    project_lm_head(weights, &normed_final, logits)?;
    if logit_scale > 0.0 && (logit_scale - 1.0).abs() > f32::EPSILON {
        let inv_scale = 1.0_f32 / logit_scale;
        for val in logits.iter_mut() {
            *val *= inv_scale;
        }
    }
    Ok(())
}

/// Manages double-buffered ping-pong recurrent banks (`slot_0`, `slot_1`) inside an arena slice,
/// falling back cleanly to a transient local bank when a compact test supplies a minimal buffer.
struct PingPongRecurrentArena<'a> {
    backing: &'a mut [f32],
    fallback: Vec<f32>,
    num_slots: usize,
    slot_floats: usize,
    active_bank: usize,
    committed_step: u32,
    uses_arena: bool,
}

impl<'a> PingPongRecurrentArena<'a> {
    fn open(backing: &'a mut [f32], num_slots: usize, slot_floats: usize) -> Self {
        let bank_floats = num_slots.saturating_mul(slot_floats);
        let needed = RECURRENT_HEADER_FLOATS.saturating_add(bank_floats.saturating_mul(2));
        if bank_floats > 0 && backing.len() >= needed {
            if backing[0] != RECURRENT_MAGIC_F32 {
                backing[..RECURRENT_HEADER_FLOATS].fill(0.0);
                backing[0] = RECURRENT_MAGIC_F32;
                backing[1] = 0.0;
                backing[2] = 0.0;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "recurrent bank stride fits cleanly in f32 metadata slot"
                )]
                {
                    backing[3] = bank_floats as f32;
                }
            }
            let active_bank = usize::from(backing[1] >= 0.5);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "committed_step is stored as an exact non-negative integer in f32"
            )]
            let committed_step = backing[2].max(0.0) as u32;
            Self {
                backing,
                fallback: Vec::new(),
                num_slots,
                slot_floats,
                active_bank,
                committed_step,
                uses_arena: true,
            }
        } else {
            Self {
                backing,
                fallback: vec![0.0_f32; bank_floats],
                num_slots,
                slot_floats,
                active_bank: 0,
                committed_step: 0,
                uses_arena: false,
            }
        }
    }

    fn read_slot(&self, slot_idx: usize, dst: &mut [f32]) {
        if slot_idx >= self.num_slots || dst.is_empty() {
            return;
        }
        let bank_stride = self.num_slots * self.slot_floats;
        let off = slot_idx * self.slot_floats;
        let copy_len = dst.len().min(self.slot_floats);
        if self.uses_arena {
            let base = RECURRENT_HEADER_FLOATS + self.active_bank * bank_stride + off;
            dst[..copy_len].copy_from_slice(&self.backing[base..base + copy_len]);
        } else {
            dst[..copy_len].copy_from_slice(&self.fallback[off..off + copy_len]);
        }
    }

    fn write_staging_slot(&mut self, slot_idx: usize, src: &[f32]) {
        if slot_idx >= self.num_slots || src.is_empty() {
            return;
        }
        let bank_stride = self.num_slots * self.slot_floats;
        let off = slot_idx * self.slot_floats;
        let copy_len = src.len().min(self.slot_floats);
        if self.uses_arena {
            let staging_bank = 1 - self.active_bank;
            let base = RECURRENT_HEADER_FLOATS + staging_bank * bank_stride + off;
            self.backing[base..base + copy_len].copy_from_slice(&src[..copy_len]);
        } else {
            self.fallback[off..off + copy_len].copy_from_slice(&src[..copy_len]);
        }
    }

    fn commit(mut self, step: u32) {
        if self.uses_arena {
            let next_bank = 1 - self.active_bank;
            self.backing[0] = RECURRENT_MAGIC_F32;
            self.backing[1] = if next_bank == 1 { 1.0 } else { 0.0 };
            #[expect(
                clippy::cast_precision_loss,
                reason = "inference step count fits cleanly in 24-bit f32 mantissa"
            )]
            {
                self.backing[2] = step as f32;
            }
            self.active_bank = next_bank;
            self.committed_step = step;
        }
    }
}

/// Reads `(active_bank, committed_step)` from a recurrent ping-pong arena slice when initialized.
#[cfg(test)]
#[must_use]
pub(crate) fn inspect_ping_pong_header(buf: &[f32]) -> Option<(u32, u32)> {
    if buf.len() < RECURRENT_HEADER_FLOATS || buf[0] != RECURRENT_MAGIC_F32 {
        return None;
    }
    let active_bank = u32::from(buf[1] >= 0.5);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "committed_step encodes an exact non-negative token step index"
    )]
    let committed_step = buf[2].max(0.0) as u32;
    Some((active_bank, committed_step))
}

/// Executes one `LFM2` gated causal depthwise short-convolution operator step (`#64`), delegating
/// directly to [`lfm2_shortconv_step_f32`] over `[dim, l_cache - 1]` history and `[dim, l_cache]`
/// filter weights.
///
/// # Errors
///
/// Returns [`K3Error`] if dimensions or buffer lengths fail kernel validation.
#[expect(
    clippy::too_many_arguments,
    reason = "matches lfm2_shortconv_step_f32 gate, projection, state, and kernel arguments"
)]
pub fn lfm2_shortconv_step(
    conv_state: &mut [f32],
    b_gate: &[f32],
    c_gate: &[f32],
    x_proj: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out: &mut [f32],
    dim: usize,
    l_cache: usize,
) -> Result<(), K3Error> {
    lfm2_shortconv_step_f32(
        conv_state,
        b_gate,
        c_gate,
        x_proj,
        conv_weight,
        conv_bias,
        out,
        dim,
        l_cache,
    )
}

/// Classifies layers `0..num_layers` into compact [`HybridLayerKind`] execution slots by inspecting
/// layer tensor presence on `weights` (`#61`).
///
/// Inspects GGUF naming across `blk.{i}.attn_q.weight` / `blk.{i}.attn_k.weight` ([`HybridLayerKind::GqaAttention`]),
/// `blk.{i}.shortconv.*` ([`HybridLayerKind::Lfm2ShortConv`]), and `blk.{i}.ssm_*`
/// ([`HybridLayerKind::Mamba2Ssd`]). Layers lacking distinguishing tensors default to
/// [`HybridLayerKind::GqaAttention`] with monotonically assigned compact slot indices.
fn resolve_hybrid_schedule_by_probe(
    num_layers: usize,
    mut contains: impl FnMut(&str) -> bool,
) -> (Vec<HybridLayerKind>, bool) {
    let mut schedule = Vec::with_capacity(num_layers);
    let mut next_kv_slot = 0_usize;
    let mut next_recurrent_slot = 0_usize;
    let mut saw_known_tensor = false;

    for layer in 0..num_layers {
        let is_attn = contains(&format!("blk.{layer}.attn_q.weight"))
            || contains(&format!("blk.{layer}.attn_k.weight"))
            || contains(&format!("blk.{layer}.attn_qkv.weight"));
        if is_attn {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::GqaAttention {
                kv_layer_slot: next_kv_slot,
            });
            next_kv_slot += 1;
            continue;
        }

        let is_shortconv = contains(&format!("blk.{layer}.shortconv.conv.weight"))
            || contains(&format!("blk.{layer}.shortconv.in_proj.weight"))
            || contains(&format!("blk.{layer}.shortconv.out_proj.weight"))
            || contains(&format!("blk.{layer}.shortconv_conv.weight"));
        if is_shortconv {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot: next_recurrent_slot,
            });
            next_recurrent_slot += 1;
            continue;
        }

        let is_ssm = contains(&format!("blk.{layer}.ssm_in.weight"))
            || contains(&format!("blk.{layer}.ssm_out.weight"))
            || contains(&format!("blk.{layer}.ssm_conv1d.weight"));
        if is_ssm {
            saw_known_tensor = true;
            schedule.push(HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: next_recurrent_slot,
            });
            next_recurrent_slot += 1;
            continue;
        }

        schedule.push(HybridLayerKind::GqaAttention {
            kv_layer_slot: next_kv_slot,
        });
        next_kv_slot += 1;
    }

    (schedule, saw_known_tensor)
}

pub(crate) fn default_hybrid_layer_schedule(arch: &str, num_layers: usize) -> Vec<HybridLayerKind> {
    let mut schedule = Vec::with_capacity(num_layers);
    let mut kv_slot = 0_usize;
    let mut rec_slot = 0_usize;
    for idx in 0..num_layers {
        let is_attn = match arch {
            "jamba" | "zamba2" => idx % 8 == 4,
            "granitehybrid" => idx % 8 == 7,
            "lfm2" | "lfm2moe" => matches!(idx % 16, 2 | 5 | 8 | 10 | 12 | 14),
            _ => true,
        };
        if is_attn {
            schedule.push(HybridLayerKind::GqaAttention {
                kv_layer_slot: kv_slot,
            });
            kv_slot += 1;
        } else {
            let kind = if matches!(arch, "lfm2" | "lfm2moe") {
                HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: rec_slot,
                }
            } else {
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: rec_slot,
                }
            };
            schedule.push(kind);
            rec_slot += 1;
        }
    }
    schedule
}

#[must_use]
pub fn resolve_hybrid_layer_schedule(
    num_layers: usize,
    weights: &dyn ModelWeights,
) -> Vec<HybridLayerKind> {
    resolve_hybrid_schedule_by_probe(num_layers, |name| weights.contains_tensor(name)).0
}

/// Resolves a hybrid layer schedule purely from [`ModelMetadata`] (inspecting tensor descriptors,
/// `head_count_kv_per_layer`, and canonical per-architecture layer templates).
#[must_use]
pub fn resolve_hybrid_layer_schedule_for_metadata(
    arch: &str,
    num_layers: usize,
    metadata: Option<&ModelMetadata>,
) -> Vec<HybridLayerKind> {
    if let Some(meta) = metadata {
        if !meta.tensors().is_empty() {
            let (by_meta_tensors, saw_known) =
                resolve_hybrid_schedule_by_probe(num_layers, |name| {
                    meta.find_tensor(name).is_some()
                });
            if saw_known
                && by_meta_tensors
                    .iter()
                    .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
            {
                return by_meta_tensors;
            }
        }

        let per_layer_kv = meta.head_count_kv_per_layer();
        if !per_layer_kv.is_empty() {
            let mut schedule = Vec::with_capacity(num_layers);
            let mut next_kv_slot = 0_usize;
            let mut next_recurrent_slot = 0_usize;
            let use_shortconv = matches!(arch, "lfm2" | "lfm2moe");
            for idx in 0..num_layers {
                let kv_heads = per_layer_kv.get(idx).copied().unwrap_or(0);
                if kv_heads > 0 {
                    schedule.push(HybridLayerKind::GqaAttention {
                        kv_layer_slot: next_kv_slot,
                    });
                    next_kv_slot += 1;
                } else {
                    let kind = if use_shortconv {
                        HybridLayerKind::Lfm2ShortConv {
                            recurrent_layer_slot: next_recurrent_slot,
                        }
                    } else {
                        HybridLayerKind::Mamba2Ssd {
                            recurrent_layer_slot: next_recurrent_slot,
                        }
                    };
                    schedule.push(kind);
                    next_recurrent_slot += 1;
                }
            }
            return schedule;
        }
    }

    default_hybrid_layer_schedule(arch, num_layers)
}

/// Resolves a hybrid layer schedule combining tensor inspection with GGUF `head_count_kv_per_layer`
/// metadata hints (`kv_heads == 0` dispatches to recurrent Mamba-2 or `LFM2` `ShortConv`).
#[must_use]
pub fn resolve_hybrid_layer_schedule_with_metadata(
    arch: &str,
    num_layers: usize,
    metadata: Option<&ModelMetadata>,
    weights: &dyn ModelWeights,
) -> Vec<HybridLayerKind> {
    if weights.tensor_count() > 0 {
        let (by_weights, saw_known) =
            resolve_hybrid_schedule_by_probe(num_layers, |name| weights.contains_tensor(name));
        if saw_known
            && by_weights
                .iter()
                .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
        {
            return by_weights;
        }
    }

    resolve_hybrid_layer_schedule_for_metadata(arch, num_layers, metadata)
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared Mamba2 single-step SSM update reads/writes ping-pong recurrent slices"
)]
fn run_mamba2_layer_step(
    layer: usize,
    x_norm: &[f32],
    hidden: &mut [f32],
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
    n_heads: usize,
    head_dim: usize,
    n_groups: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let proj_dim = 2 * d_inner + 2 * n_groups * d_state + n_heads;
    let in_proj = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ssm_in.weight"),
        x_norm,
        proj_dim,
        0.125,
    )?;

    let (z_gate, x_in, b_vec, c_vec, dt_raw) = if in_proj.len() >= proj_dim {
        let z = &in_proj[0..d_inner];
        let x = &in_proj[d_inner..2 * d_inner];
        let b_off = 2 * d_inner;
        let c_off = b_off + n_groups * d_state;
        let dt_off = c_off + n_groups * d_state;
        (
            z,
            x,
            &in_proj[b_off..c_off],
            &in_proj[c_off..dt_off],
            &in_proj[dt_off..dt_off + n_heads],
        )
    } else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: format!("blk.{layer}.ssm_in.weight"),
            offset: 0,
            byte_len: in_proj.len(),
            buffer_len: proj_dim,
            reason: "ssm_in projection output shorter than required Mamba-2 slice".to_owned(),
        });
    };

    let conv_weight = read_f32_weight_or_default(
        weights,
        &format!("blk.{layer}.ssm_conv1d.weight"),
        d_inner * d_conv,
        inv_kernel_f32(d_conv),
    );
    let conv_bias = weights
        .contains_tensor(&format!("blk.{layer}.ssm_conv1d.bias"))
        .then(|| {
            read_f32_weight_or_default(
                weights,
                &format!("blk.{layer}.ssm_conv1d.bias"),
                d_inner,
                0.0,
            )
        });
    let dt_bias = weights
        .contains_tensor(&format!("blk.{layer}.ssm_dt.bias"))
        .then(|| {
            read_f32_weight_or_default(weights, &format!("blk.{layer}.ssm_dt.bias"), n_heads, 0.0)
        });
    let a_log_or_neg =
        read_f32_weight_or_default(weights, &format!("blk.{layer}.ssm_a"), n_heads, -0.5);
    let d_skip = weights
        .contains_tensor(&format!("blk.{layer}.ssm_d"))
        .then(|| read_f32_weight_or_default(weights, &format!("blk.{layer}.ssm_d"), n_heads, 1.0));

    let mut conv_out = vec![0.0_f32; d_inner];
    mamba2_conv1d_silu_step(
        conv_state,
        x_in,
        &conv_weight,
        conv_bias.as_deref(),
        &mut conv_out,
        d_inner,
        d_conv,
    )?;

    let mut ssm_y = vec![0.0_f32; d_inner];
    mamba2_ssd_state_step_f32(
        ssm_state,
        &mut ssm_y,
        &conv_out,
        dt_raw,
        dt_bias.as_deref(),
        &a_log_or_neg,
        b_vec,
        c_vec,
        d_skip.as_deref(),
        Some(z_gate),
        n_heads,
        head_dim,
        d_state,
        n_groups,
    )?;

    let out = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.ssm_out.weight"),
        &ssm_y,
        hidden_dim,
        0.25,
    )?;
    for (h, &o) in hidden.iter_mut().zip(out.iter()) {
        *h += o;
    }
    Ok(())
}

/// Execution recipe for pure Selective State-Space (`Mamba` and `Mamba-2`) checkpoints (`#57`).
///
/// Uses chunked `conv1d + SiLU` and `SSD` scans during multi-token prefill (plus intra-chunk
/// quantized matrix scoring), single-step kernels during autoregressive decode, and double-buffered
/// ping-pong (`slot_0` / `slot_1`) `O(1)` recurrent state banks independent of `n_ctx`.
#[derive(Debug, Clone)]
pub struct Mamba2Recipe {
    /// Architecture identifier (`"mamba2"` or `"mamba"`).
    pub architecture: &'static str,
    /// Number of recurrent SSM layers.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Expanded inner SSM channel dimension (`d_inner`, defaults to `2 * hidden_dim` when 0).
    pub d_inner: usize,
    /// Causal depthwise conv kernel width (`d_conv`).
    pub d_conv: usize,
    /// Recurrent SSD state dimension per head (`d_state`).
    pub d_state: usize,
    /// Number of SSD heads (`n_heads`, divides `d_inner`).
    pub n_heads: usize,
    /// Number of B/C parameter groups (`n_groups`, divides `n_heads`).
    pub n_groups: usize,
    /// Vocabulary size for final LM-head logit projection.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
}

struct MambaChunkScratch {
    z_seq: Vec<f32>,
    x_seq: Vec<f32>,
    b_seq: Vec<f32>,
    c_seq: Vec<f32>,
    dt_seq: Vec<f32>,
    conv_out_seq: Vec<f32>,
    ssm_out_seq: Vec<f32>,
    ime_scores: Vec<f32>,
    slot_buf: Vec<f32>,
}

impl MambaChunkScratch {
    fn new(
        seq_len: usize,
        d_inner: usize,
        n_heads: usize,
        n_groups: usize,
        group_step: usize,
        slot_floats: usize,
    ) -> Self {
        Self {
            z_seq: vec![0.0_f32; seq_len * d_inner],
            x_seq: vec![0.0_f32; seq_len * d_inner],
            b_seq: vec![0.0_f32; seq_len * group_step],
            c_seq: vec![0.0_f32; seq_len * group_step],
            dt_seq: vec![0.0_f32; seq_len * n_heads],
            conv_out_seq: vec![0.0_f32; seq_len * d_inner],
            ssm_out_seq: vec![0.0_f32; seq_len * d_inner],
            ime_scores: vec![0.0_f32; n_groups * seq_len * seq_len],
            slot_buf: vec![0.0_f32; slot_floats],
        }
    }
}

impl Default for Mamba2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Mamba2Recipe {
    /// Creates a `Mamba-2` recipe with defaults matching 2.7B production geometry (`#57`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "mamba2",
            num_layers: 64,
            hidden_dim: 2560,
            d_inner: 5120,
            d_conv: 4,
            d_state: 128,
            n_heads: 80,
            n_groups: 1,
            vocab_size: 50_288,
            rms_eps: 1e-5,
        }
    }

    /// Creates a `Mamba-1` compatibility recipe targeting `"mamba"`.
    #[must_use]
    pub const fn new_mamba() -> Self {
        Self {
            architecture: "mamba",
            num_layers: 48,
            hidden_dim: 2048,
            d_inner: 4096,
            d_conv: 4,
            d_state: 16,
            n_heads: 64,
            n_groups: 1,
            vocab_size: 50_280,
            rms_eps: 1e-5,
        }
    }

    fn effective_dims(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let hidden_dim = self.hidden_dim.max(2);
        let d_inner = if self.d_inner == 0 {
            hidden_dim.saturating_mul(2)
        } else {
            self.d_inner
        }
        .max(1);
        let d_conv = self.d_conv.max(2);
        let d_state = self.d_state.max(1);
        let (n_heads, head_dim, n_groups) =
            resolve_divisible_ssm_geometry(d_inner, self.n_heads, self.n_groups);
        (
            hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups,
        )
    }

    fn project_prefill_in_seq(
        &self,
        layer: usize,
        seq_len: usize,
        h_seq: &[f32],
        scratch: &mut MambaChunkScratch,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, d_inner, _d_conv, d_state, n_heads, _head_dim, n_groups) =
            self.effective_dims();
        let group_step = n_groups * d_state;
        let norm_w = resolve_norm_weight(
            weights,
            &format!("blk.{layer}.attn_norm.weight"),
            &format!("blk.{layer}.ssm_norm.weight"),
            hidden_dim,
        );
        let proj_dim = 2 * d_inner + 2 * group_step + n_heads;
        for t in 0..seq_len {
            let h_tok = &h_seq[t * hidden_dim..(t + 1) * hidden_dim];
            let mut normed = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut normed, h_tok, &norm_w, self.rms_eps)?;
            let proj = project_or_deterministic_slice(
                weights,
                &format!("blk.{layer}.ssm_in.weight"),
                &normed,
                proj_dim,
                0.125,
            )?;
            scratch.z_seq[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[0..d_inner]);
            scratch.x_seq[t * d_inner..(t + 1) * d_inner]
                .copy_from_slice(&proj[d_inner..2 * d_inner]);
            let b_off = 2 * d_inner;
            let c_off = b_off + group_step;
            let dt_off = c_off + group_step;
            scratch.b_seq[t * group_step..(t + 1) * group_step]
                .copy_from_slice(&proj[b_off..c_off]);
            scratch.c_seq[t * group_step..(t + 1) * group_step]
                .copy_from_slice(&proj[c_off..dt_off]);
            scratch.dt_seq[t * n_heads..(t + 1) * n_heads]
                .copy_from_slice(&proj[dt_off..dt_off + n_heads]);
        }
        Ok(())
    }

    fn run_prefill_layer(
        &self,
        layer: usize,
        seq_len: usize,
        h_seq: &mut [f32],
        scratch: &mut MambaChunkScratch,
        arena: &mut PingPongRecurrentArena<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups) =
            self.effective_dims();
        let conv_state_elems = d_inner * (d_conv - 1);
        self.project_prefill_in_seq(layer, seq_len, h_seq, scratch, weights)?;

        if seq_len > 1 {
            mamba2_ssd_intra_chunk_ime_i8(
                &scratch.c_seq,
                &scratch.b_seq,
                &mut scratch.ime_scores,
                seq_len,
                n_groups,
                d_state,
            )?;
        }

        let conv_weight = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.ssm_conv1d.weight"),
            d_inner * d_conv,
            inv_kernel_f32(d_conv),
        );
        let conv_bias = weights
            .contains_tensor(&format!("blk.{layer}.ssm_conv1d.bias"))
            .then(|| {
                read_f32_weight_or_default(
                    weights,
                    &format!("blk.{layer}.ssm_conv1d.bias"),
                    d_inner,
                    0.0,
                )
            });
        let dt_bias = weights
            .contains_tensor(&format!("blk.{layer}.ssm_dt.bias"))
            .then(|| {
                read_f32_weight_or_default(
                    weights,
                    &format!("blk.{layer}.ssm_dt.bias"),
                    n_heads,
                    0.0,
                )
            });
        let a_log_or_neg =
            read_f32_weight_or_default(weights, &format!("blk.{layer}.ssm_a"), n_heads, -0.5);
        let d_skip = weights
            .contains_tensor(&format!("blk.{layer}.ssm_d"))
            .then(|| {
                read_f32_weight_or_default(weights, &format!("blk.{layer}.ssm_d"), n_heads, 1.0)
            });

        arena.read_slot(layer, &mut scratch.slot_buf);
        let (conv_state, ssm_state) = scratch.slot_buf.split_at_mut(conv_state_elems);
        mamba2_conv1d_silu_chunk_f32(
            conv_state,
            &scratch.x_seq,
            &conv_weight,
            conv_bias.as_deref(),
            &mut scratch.conv_out_seq,
            seq_len,
            d_inner,
            d_conv,
        )?;
        mamba2_ssd_chunk_scan_f32(
            ssm_state,
            &mut scratch.ssm_out_seq,
            &scratch.conv_out_seq,
            &scratch.dt_seq,
            dt_bias.as_deref(),
            &a_log_or_neg,
            &scratch.b_seq,
            &scratch.c_seq,
            d_skip.as_deref(),
            Some(&scratch.z_seq),
            seq_len,
            n_heads,
            head_dim,
            d_state,
            n_groups,
        )?;
        arena.write_staging_slot(layer, &scratch.slot_buf);

        for t in 0..seq_len {
            let ssm_tok = &scratch.ssm_out_seq[t * d_inner..(t + 1) * d_inner];
            let out_proj = project_or_deterministic_slice(
                weights,
                &format!("blk.{layer}.ssm_out.weight"),
                ssm_tok,
                hidden_dim,
                0.25,
            )?;
            let h_tok = &mut h_seq[t * hidden_dim..(t + 1) * hidden_dim];
            for (h, &o) in h_tok.iter_mut().zip(out_proj.iter()) {
                *h += o;
            }
        }
        Ok(())
    }

    fn run_chunked_prefill(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx
            .jump_forward_tokens()
            .unwrap_or_else(|| ctx.prompt_tokens())
            .to_vec();
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "prefill prompt_tokens cannot be empty".to_owned(),
            });
        }
        let start_pos = ctx.position();
        let (hidden_dim, d_inner, d_conv, d_state, n_heads, _head_dim, n_groups) =
            self.effective_dims();
        let num_layers = self.num_layers.max(1);
        let seq_len = prompt.len();

        let mut h_seq = Vec::with_capacity(seq_len * hidden_dim);
        for &tok in &prompt {
            h_seq.extend_from_slice(&embed_token(tok, hidden_dim, weights)?);
        }

        let slot_floats = d_inner * (d_conv - 1) + d_inner * d_state;
        let mut scratch = MambaChunkScratch::new(
            seq_len,
            d_inner,
            n_heads,
            n_groups,
            n_groups * d_state,
            slot_floats,
        );
        let mut arena = PingPongRecurrentArena::open(ctx.kv_cache_mut(), num_layers, slot_floats);

        for layer in 0..num_layers {
            self.run_prefill_layer(
                layer,
                seq_len,
                &mut h_seq,
                &mut scratch,
                &mut arena,
                weights,
            )?;
        }

        let step_num = u32::try_from(start_pos + seq_len).unwrap_or(u32::MAX);
        arena.commit(step_num);

        if ctx.compute_logits() {
            let last_hidden = &h_seq[(seq_len - 1) * hidden_dim..];
            let norm_out_w =
                read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
            let mut normed_final = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut normed_final, last_hidden, &norm_out_w, self.rms_eps)?;
            project_lm_head(weights, &normed_final, ctx.logits_mut())?;
        }
        let _ = ctx.advance_position(seq_len)?;
        Ok(())
    }

    fn execute_single_token(
        &self,
        tok: u32,
        pos: usize,
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, d_inner, d_conv, d_state, n_heads, head_dim, n_groups) =
            self.effective_dims();
        let num_layers = self.num_layers.max(1);
        let conv_state_elems = d_inner * (d_conv - 1);
        let slot_floats = conv_state_elems + d_inner * d_state;

        let mut hidden = embed_token(tok, hidden_dim, weights)?;
        let mut slot_buf = vec![0.0_f32; slot_floats];
        let mut arena = PingPongRecurrentArena::open(kv_cache, num_layers, slot_floats);

        for layer in 0..num_layers {
            let norm_w = resolve_norm_weight(
                weights,
                &format!("blk.{layer}.attn_norm.weight"),
                &format!("blk.{layer}.ssm_norm.weight"),
                hidden_dim,
            );
            let mut normed = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut normed, &hidden, &norm_w, self.rms_eps)?;

            arena.read_slot(layer, &mut slot_buf);
            let (conv_state, ssm_state) = slot_buf.split_at_mut(conv_state_elems);
            run_mamba2_layer_step(
                layer,
                &normed,
                &mut hidden,
                conv_state,
                ssm_state,
                d_inner,
                d_conv,
                d_state,
                n_heads,
                head_dim,
                n_groups,
                weights,
            )?;
            arena.write_staging_slot(layer, &slot_buf);
        }

        let step_num = u32::try_from(pos + 1).unwrap_or(u32::MAX);
        arena.commit(step_num);

        if !logits.is_empty() {
            let norm_out_w =
                read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
            let mut normed_final = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut normed_final, &hidden, &norm_out_w, self.rms_eps)?;
            project_lm_head(weights, &normed_final, logits)?;
        }
        Ok(())
    }
}

impl ArchRecipe for Mamba2Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.run_chunked_prefill(ctx, weights)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, self.architecture, |tok, pos, kv, logits| {
            self.execute_single_token(tok, pos, logits, kv, weights)
        })
    }
}

/// Execution recipe for `RWKV-6` (`Finch`) and `RWKV-7` (`Goose`) linear-RNN architectures (`#58`).
///
/// Combines causal token-shift interpolation (`rwkv_token_shift_lerp_f32`), matrix-valued delta-rule
/// WKV state updates (`rwkv7_wkv_delta_rule_step_f32`), channel-mix feed-forward blocks, and
/// double-buffered ping-pong `wkv_state` slots (`O(1)` in `n_ctx`).
#[derive(Debug, Clone)]
pub struct Rwkv7Recipe {
    /// Architecture identifier (`"rwkv7"` or `"rwkv6"`).
    pub architecture: &'static str,
    /// Number of recurrent blocks.
    pub num_layers: usize,
    /// Model embedding dimension.
    pub hidden_dim: usize,
    /// Channel-mixing feed-forward dimension.
    pub ffn_dim: usize,
    /// Per-head WKV state matrix dimension (`head_size`).
    pub head_size: usize,
    /// Number of WKV heads (`hidden_dim / head_size`).
    pub n_heads: usize,
    /// Vocabulary size for final LM-head projection.
    pub vocab_size: usize,
    /// Layer/RMSNorm epsilon.
    pub rms_eps: f32,
}

impl Default for Rwkv7Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Rwkv7Recipe {
    /// Creates an `RWKV-7` (`Goose`) recipe matching 1.5B-scale geometry (`#58`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "rwkv7",
            num_layers: 24,
            hidden_dim: 2048,
            ffn_dim: 7168,
            head_size: 64,
            n_heads: 32,
            vocab_size: 65_536,
            rms_eps: 1e-5,
        }
    }

    /// Creates an `RWKV-6` (`Finch`) recipe targeting `"rwkv6"`.
    #[must_use]
    pub const fn new_rwkv6() -> Self {
        Self {
            architecture: "rwkv6",
            num_layers: 24,
            hidden_dim: 2048,
            ffn_dim: 7168,
            head_size: 64,
            n_heads: 32,
            vocab_size: 65_536,
            rms_eps: 1e-5,
        }
    }

    fn effective_dims(&self) -> (usize, usize, usize, usize) {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(hidden_dim);
        let head_size = self.head_size.clamp(1, hidden_dim);
        let mut h = self.n_heads.clamp(1, hidden_dim);
        while h * head_size != hidden_dim && h > 1 {
            h -= 1;
        }
        let eff_head_size = if h * head_size == hidden_dim {
            head_size
        } else {
            hidden_dim / h.max(1)
        }
        .max(1);
        (hidden_dim, ffn_dim, h, eff_head_size)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "RWKV time-mix step projects r/w/k/v/a/g and updates dual-bank WKV state"
    )]
    fn execute_time_mix(
        &self,
        layer: usize,
        hidden: &mut [f32],
        prev_time_shift: &mut [f32],
        wkv_state: &mut [f32],
        n_heads: usize,
        head_size: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let vec_dim = n_heads * head_size;
        let attn_norm_w = resolve_norm_weight(
            weights,
            &format!("blk.{layer}.attn_norm.weight"),
            &format!("blk.{layer}.ln1.weight"),
            hidden_dim,
        );
        let mut x_norm = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut x_norm, hidden, &attn_norm_w, self.rms_eps)?;

        let lerp_time = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.time_mix_x_r"),
            hidden_dim,
            0.5,
        );
        let mut shifted_time = vec![0.0_f32; hidden_dim];
        rwkv_token_shift_lerp_f32(prev_time_shift, &x_norm, &lerp_time, &mut shifted_time)?;
        prev_time_shift.copy_from_slice(&x_norm);

        let r_vec = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_r.weight"),
            &shifted_time,
            vec_dim,
            0.5,
        )?;
        let mut w_decay = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_w.weight"),
            &shifted_time,
            vec_dim,
            0.25,
        )?;
        for w in &mut w_decay {
            *w = (-softplus_f32(w.clamp(-8.0, 8.0))).exp().clamp(0.0, 0.999);
        }
        let k_vec = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_k.weight"),
            &shifted_time,
            vec_dim,
            0.25,
        )?;
        let v_vec = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_v.weight"),
            &shifted_time,
            vec_dim,
            0.25,
        )?;
        let a_vec = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_a.weight"),
            &shifted_time,
            vec_dim,
            0.125,
        )?;
        let mut b_vec = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_g.weight"),
            &shifted_time,
            vec_dim,
            0.125,
        )?;
        for b in &mut b_vec {
            *b = -b.tanh() * 0.25;
        }

        let mut wkv_out = vec![0.0_f32; vec_dim];
        rwkv7_wkv_delta_rule_step_f32(
            wkv_state,
            &mut wkv_out,
            &r_vec,
            &w_decay,
            &k_vec,
            &v_vec,
            &a_vec,
            &b_vec,
            n_heads,
            head_size,
        )?;

        let tm_out = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer}.time_mix_output.weight"),
            &wkv_out,
            hidden_dim,
            0.25,
        )?;
        for (h, &o) in hidden.iter_mut().zip(tm_out.iter()) {
            *h += o;
        }
        Ok(())
    }

    fn execute_channel_mix(
        &self,
        layer: usize,
        hidden: &mut [f32],
        prev_chan_shift: &mut [f32],
        ffn_dim: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let ffn_norm_w = resolve_norm_weight(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            &format!("blk.{layer}.ln2.weight"),
            hidden_dim,
        );
        let mut x_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut x_ffn_in, hidden, &ffn_norm_w, self.rms_eps)?;
        let lerp_chan = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer}.channel_mix_lerp_k"),
            hidden_dim,
            0.5,
        );
        let mut shifted_chan = vec![0.0_f32; hidden_dim];
        rwkv_token_shift_lerp_f32(prev_chan_shift, &x_ffn_in, &lerp_chan, &mut shifted_chan)?;
        prev_chan_shift.copy_from_slice(&x_ffn_in);

        let key_name = format!("blk.{layer}.channel_mix_key.weight");
        let val_name = format!("blk.{layer}.channel_mix_value.weight");
        let cm_key_tensor = if weights.contains_tensor(&key_name) {
            key_name
        } else {
            format!("blk.{layer}.ffn_key.weight")
        };
        let cm_val_tensor = if weights.contains_tensor(&val_name) {
            val_name
        } else {
            format!("blk.{layer}.ffn_value.weight")
        };
        let mut cm_k =
            project_or_deterministic_slice(weights, &cm_key_tensor, &shifted_chan, ffn_dim, 0.25)?;
        for k in &mut cm_k {
            let r = k.max(0.0);
            *k = r * r;
        }
        let cm_v =
            project_or_deterministic_slice(weights, &cm_val_tensor, &cm_k, hidden_dim, 0.25)?;
        for (h, &v) in hidden.iter_mut().zip(cm_v.iter()) {
            *h += v;
        }
        Ok(())
    }

    fn execute_single_token(
        &self,
        tok: u32,
        pos: usize,
        _seq_len: usize,
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (hidden_dim, ffn_dim, n_heads, head_size) = self.effective_dims();
        let wkv_elems = n_heads * head_size * head_size;
        let slot_floats = 2 * hidden_dim + wkv_elems;
        let num_layers = self.num_layers.max(1);

        let mut hidden = embed_token(tok, hidden_dim, weights)?;
        let mut slot_buf = vec![0.0_f32; slot_floats];
        let mut arena = PingPongRecurrentArena::open(kv_cache, num_layers, slot_floats);

        for layer in 0..num_layers {
            arena.read_slot(layer, &mut slot_buf);
            let (prev_both, wkv_state) = slot_buf.split_at_mut(2 * hidden_dim);
            let (prev_time_shift, prev_chan_shift) = prev_both.split_at_mut(hidden_dim);

            self.execute_time_mix(
                layer,
                &mut hidden,
                prev_time_shift,
                wkv_state,
                n_heads,
                head_size,
                weights,
            )?;
            self.execute_channel_mix(layer, &mut hidden, prev_chan_shift, ffn_dim, weights)?;
            arena.write_staging_slot(layer, &slot_buf);
        }

        let step_num = u32::try_from(pos + 1).unwrap_or(u32::MAX);
        arena.commit(step_num);

        if !logits.is_empty() {
            let norm_out_w =
                read_f32_weight_or_default(weights, "output_norm.weight", hidden_dim, 1.0);
            let mut normed_final = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut normed_final, &hidden, &norm_out_w, self.rms_eps)?;
            project_lm_head(weights, &normed_final, logits)?;
        }
        Ok(())
    }
}

impl ArchRecipe for Rwkv7Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, self.architecture, |tok, pos, seq_len, kv, logits| {
            self.execute_single_token(tok, pos, seq_len, logits, kv, weights)
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, self.architecture, |tok, pos, kv, logits| {
            self.execute_single_token(tok, pos, pos + 1, logits, kv, weights)
        })
    }
}

/// Hybrid attention + SSM/short-convolution execution recipe covering `jamba`, `zamba2`,
/// `granitehybrid`, and `LiquidAI` `lfm2` (`#59`, `#63`, `#64`).
///
/// Dispatches each layer via [`HybridLayerKind`]: `GqaAttention` layers update paged KV cache
/// planes through their compact `kv_layer_slot`, while `Mamba2Ssd` and `Lfm2ShortConv` layers
/// execute linear-time state updates over trailing double-buffered ping-pong recurrent banks.
#[derive(Debug, Clone)]
pub struct HybridSsmTransformerRecipe {
    /// Exact GGUF `general.architecture` identifier (`"jamba"`, `"zamba2"`, `"granitehybrid"`, or `"lfm2"`).
    pub architecture_id: &'static str,
    /// Per-layer dispatch schedule built at load time or via [`resolve_hybrid_layer_schedule`].
    pub layer_schedule: Vec<HybridLayerKind>,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Attention query head count.
    pub n_heads_q: usize,
    /// Attention key/value head count.
    pub n_heads_kv: usize,
    /// Scalar dimension per attention head.
    pub head_dim: usize,
    /// Inner SSM expansion width (`ssm.inner_size`).
    pub d_inner: usize,
    /// Causal conv kernel width (`ssm.conv_kernel`).
    pub d_conv: usize,
    /// Mamba-2 state width (`ssm.state_size`).
    pub d_state: usize,
    /// Mamba-2 B/C group count (`ssm.group_count`).
    pub n_groups: usize,
    /// LFM2 short-convolution cache length (`shortconv.l_cache`).
    pub shortconv_l_cache: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`rope.freq_base`).
    pub rope_theta: f32,
    /// Optional `GraniteHybrid` embedding multiplier (`1.0` default).
    pub embedding_scale: f32,
    /// Optional `GraniteHybrid` residual branch multiplier (`1.0` default).
    pub residual_scale: f32,
    /// Optional `GraniteHybrid` explicit attention score scale override.
    pub attention_scale: Option<f32>,
    /// Optional `GraniteHybrid` output logit divisor (`1.0` default).
    pub logit_scale: f32,
}

impl Default for HybridSsmTransformerRecipe {
    fn default() -> Self {
        Self::new_lfm2()
    }
}

impl HybridSsmTransformerRecipe {
    /// Creates an alternating 1:7 `Jamba` hybrid attention + Mamba-2 recipe (`"jamba"`).
    #[must_use]
    pub fn new_jamba() -> Self {
        let mut schedule = Vec::with_capacity(32);
        let mut kv_slot = 0_usize;
        let mut ssm_slot = 0_usize;
        for idx in 0..32_usize {
            if idx % 8 == 4 {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: ssm_slot,
                });
                ssm_slot += 1;
            }
        }
        Self {
            architecture_id: "jamba",
            layer_schedule: schedule,
            hidden_dim: 4096,
            ffn_dim: 14_336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            d_inner: 8192,
            d_conv: 4,
            d_state: 16,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 65_536,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Creates a `Zamba2` hybrid attention + Mamba-2 recipe (`"zamba2"`).
    #[must_use]
    pub fn new_zamba2() -> Self {
        let mut recipe = Self::new_jamba();
        recipe.architecture_id = "zamba2";
        recipe
    }

    /// Creates a `GraniteHybrid` (`Granite 4.0` / `Bamba`) recipe (`"granitehybrid"`).
    #[must_use]
    pub fn new_granitehybrid() -> Self {
        let mut schedule = Vec::with_capacity(32);
        let mut kv_slot = 0_usize;
        let mut ssm_slot = 0_usize;
        for idx in 0..32_usize {
            if idx % 8 == 7 {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: ssm_slot,
                });
                ssm_slot += 1;
            }
        }
        Self {
            architecture_id: "granitehybrid",
            layer_schedule: schedule,
            hidden_dim: 4096,
            ffn_dim: 11_008,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            d_inner: 8192,
            d_conv: 4,
            d_state: 128,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 49_152,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Creates a `LiquidAI` `LFM2` hybrid GQA + `ShortConv` recipe (`"lfm2"`).
    #[must_use]
    pub fn new_lfm2() -> Self {
        let mut schedule = Vec::with_capacity(16);
        let mut kv_slot = 0_usize;
        let mut sc_slot = 0_usize;
        for idx in 0..16_usize {
            if matches!(idx, 2 | 5 | 8 | 10 | 12 | 14) {
                schedule.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule.push(HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: sc_slot,
                });
                sc_slot += 1;
            }
        }
        Self {
            architecture_id: "lfm2",
            layer_schedule: schedule,
            hidden_dim: 2048,
            ffn_dim: 8192,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 64,
            d_inner: 2048,
            d_conv: 3,
            d_state: 16,
            n_groups: 1,
            shortconv_l_cache: 3,
            vocab_size: 65_536,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    /// Constructs a compact hybrid recipe with an explicit [`HybridLayerKind`] schedule.
    #[must_use]
    pub fn from_schedule(
        architecture_id: &'static str,
        layer_schedule: Vec<HybridLayerKind>,
    ) -> Self {
        let mut base = match architecture_id {
            "jamba" => Self::new_jamba(),
            "zamba2" => Self::new_zamba2(),
            "granitehybrid" => Self::new_granitehybrid(),
            _ => Self::new_lfm2(),
        };
        base.architecture_id = architecture_id;
        base.layer_schedule = layer_schedule;
        base
    }

    fn effective_schedule(&self, weights: &dyn ModelWeights) -> Vec<HybridLayerKind> {
        if weights.tensor_count() > 0 {
            let inferred = resolve_hybrid_layer_schedule(self.layer_schedule.len().max(1), weights);
            if inferred
                .iter()
                .any(|k| !matches!(k, HybridLayerKind::GqaAttention { .. }))
            {
                return inferred;
            }
        }
        if self.layer_schedule.is_empty() {
            vec![HybridLayerKind::GqaAttention { kv_layer_slot: 0 }]
        } else {
            self.layer_schedule.clone()
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches single-layer Mamba2 SSD update inside hybrid loop"
    )]
    fn run_hybrid_ssm_layer(
        &self,
        layer_idx: usize,
        recurrent_layer_slot: usize,
        hidden: &mut [f32],
        slot_buf: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        ssm_geom: (usize, usize, usize, usize, usize, usize),
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let (d_inner, d_conv, d_state, ssm_heads, ssm_head_dim, n_groups) = ssm_geom;
        let norm_w = resolve_norm_weight(
            weights,
            &format!("blk.{layer_idx}.attn_norm.weight"),
            &format!("blk.{layer_idx}.operator_norm.weight"),
            hidden_dim,
        );
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut normed, hidden, &norm_w, self.rms_eps)?;

        rec_arena.read_slot(recurrent_layer_slot, slot_buf);
        let conv_elems = d_inner * (d_conv - 1);
        let (conv_state, ssm_rest) = slot_buf.split_at_mut(conv_elems);
        let ssm_state = &mut ssm_rest[..d_inner * d_state];
        let mut ssm_delta = hidden.to_vec();
        run_mamba2_layer_step(
            layer_idx,
            &normed,
            &mut ssm_delta,
            conv_state,
            ssm_state,
            d_inner,
            d_conv,
            d_state,
            ssm_heads,
            ssm_head_dim,
            n_groups,
            weights,
        )?;
        for (h, d) in hidden.iter_mut().zip(ssm_delta.iter()) {
            let branch = *d - *h;
            *h += branch * self.residual_scale;
        }
        rec_arena.write_staging_slot(recurrent_layer_slot, slot_buf);
        Ok(())
    }

    fn run_hybrid_shortconv_layer(
        &self,
        layer_idx: usize,
        recurrent_layer_slot: usize,
        hidden: &mut [f32],
        slot_buf: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = hidden.len();
        let l_cache = self.shortconv_l_cache.max(2);
        let norm_w = resolve_norm_weight(
            weights,
            &format!("blk.{layer_idx}.operator_norm.weight"),
            &format!("blk.{layer_idx}.attn_norm.weight"),
            hidden_dim,
        );
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut normed, hidden, &norm_w, self.rms_eps)?;

        let proj3 = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer_idx}.shortconv.in_proj.weight"),
            &normed,
            3 * hidden_dim,
            0.25,
        )?;
        let b_gate = &proj3[0..hidden_dim];
        let c_gate = &proj3[hidden_dim..2 * hidden_dim];
        let x_proj = &proj3[2 * hidden_dim..3 * hidden_dim];

        let conv_weight = read_f32_weight_or_default(
            weights,
            &format!("blk.{layer_idx}.shortconv.conv.weight"),
            hidden_dim * l_cache,
            inv_kernel_f32(l_cache),
        );
        let conv_bias = weights
            .contains_tensor(&format!("blk.{layer_idx}.shortconv.conv.bias"))
            .then(|| {
                read_f32_weight_or_default(
                    weights,
                    &format!("blk.{layer_idx}.shortconv.conv.bias"),
                    hidden_dim,
                    0.0,
                )
            });

        let conv_elems = hidden_dim * (l_cache - 1);
        rec_arena.read_slot(recurrent_layer_slot, slot_buf);
        let conv_state = &mut slot_buf[..conv_elems];
        let mut sc_out = vec![0.0_f32; hidden_dim];
        lfm2_shortconv_step(
            conv_state,
            b_gate,
            c_gate,
            x_proj,
            &conv_weight,
            conv_bias.as_deref(),
            &mut sc_out,
            hidden_dim,
            l_cache,
        )?;
        rec_arena.write_staging_slot(recurrent_layer_slot, slot_buf);

        let out_proj = project_or_deterministic_slice(
            weights,
            &format!("blk.{layer_idx}.shortconv.out_proj.weight"),
            &sc_out,
            hidden_dim,
            0.25,
        )?;
        for (h, &o) in hidden.iter_mut().zip(out_proj.iter()) {
            *h += o * self.residual_scale;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches GqaAttention, Mamba2Ssd, or Lfm2ShortConv across disjoint arena planes"
    )]
    fn run_hybrid_layer_step(
        &self,
        layer_idx: usize,
        kind: HybridLayerKind,
        step: (usize, usize),
        hidden: &mut [f32],
        attn_spec: DenseAttnLayerSpec,
        kv_region: &mut [f32],
        rec_arena: &mut PingPongRecurrentArena<'_>,
        slot_buf: &mut [f32],
        ssm_geom: (usize, usize, usize, usize, usize, usize),
        num_kv_layers: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let (pos, seq_len) = step;
        let tokens_per_layer = seq_len.max(pos + 1);
        let per_kv_layer_elems = 2 * tokens_per_layer * attn_spec.n_heads_kv * attn_spec.head_dim;
        match kind {
            HybridLayerKind::GqaAttention { kv_layer_slot } => {
                let layer_kv = if kv_layer_slot < num_kv_layers
                    && per_kv_layer_elems > 0
                    && (kv_layer_slot + 1) * per_kv_layer_elems <= kv_region.len()
                {
                    &mut kv_region[kv_layer_slot * per_kv_layer_elems
                        ..(kv_layer_slot + 1) * per_kv_layer_elems]
                } else {
                    &mut kv_region[..0]
                };
                let mut layer_spec = attn_spec;
                layer_spec.use_qk_norm =
                    weights.contains_tensor(&format!("blk.{layer_idx}.attn_q_norm.weight"));
                layer_spec.use_qkv_bias =
                    weights.contains_tensor(&format!("blk.{layer_idx}.attn_q.bias"));
                run_dense_attention_sublayer_scaled(
                    layer_spec,
                    layer_idx,
                    pos,
                    seq_len,
                    hidden,
                    layer_kv,
                    weights,
                    self.attention_scale,
                    self.residual_scale,
                )?;
            }
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot,
            } => {
                self.run_hybrid_ssm_layer(
                    layer_idx,
                    recurrent_layer_slot,
                    hidden,
                    slot_buf,
                    rec_arena,
                    ssm_geom,
                    weights,
                )?;
            }
            HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot,
            } => {
                self.run_hybrid_shortconv_layer(
                    layer_idx,
                    recurrent_layer_slot,
                    hidden,
                    slot_buf,
                    rec_arena,
                    weights,
                )?;
            }
        }
        Ok(())
    }

    fn execute_hybrid_token<Ffn>(
        &self,
        step: (u32, usize, usize),
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
        mut run_ffn: Ffn,
    ) -> Result<(), K3Error>
    where
        Ffn: FnMut(usize, &[f32], &dyn ModelWeights) -> Result<Vec<f32>, K3Error>,
    {
        let (tok, pos, seq_len) = step;
        let schedule = self.effective_schedule(weights);
        let hidden_dim = self.hidden_dim.max(2);
        let head_d_even = even_head_dim(self.head_dim);
        let n_kv = self.n_heads_kv.max(1);
        let num_kv_layers = schedule
            .iter()
            .filter(|k| matches!(k, HybridLayerKind::GqaAttention { .. }))
            .count();
        let num_rec_layers = schedule.len().saturating_sub(num_kv_layers);

        let attn_spec = DenseAttnLayerSpec {
            hidden_dim,
            ffn_dim: self.ffn_dim.max(4),
            n_heads_q: self.n_heads_q.max(1),
            n_heads_kv: n_kv,
            head_dim: head_d_even,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: false,
            use_qk_norm: false,
        };

        let d_inner = if self.d_inner == 0 {
            hidden_dim
        } else {
            self.d_inner
        }
        .max(1);
        let d_conv = self.d_conv.max(2);
        let d_state = self.d_state.max(1);
        let l_cache = self.shortconv_l_cache.max(2);
        let (ssm_heads, ssm_head_dim, n_groups) =
            resolve_divisible_ssm_geometry(d_inner, self.n_heads_q, self.n_groups);

        let has_ssm_layers = schedule
            .iter()
            .any(|k| matches!(k, HybridLayerKind::Mamba2Ssd { .. }));
        let rec_slot_floats = if has_ssm_layers {
            (d_inner * (d_conv - 1) + d_inner * d_state).max(hidden_dim * (l_cache - 1))
        } else {
            hidden_dim * (l_cache - 1)
        };

        let mut hidden = embed_token(tok, hidden_dim, weights)?;
        if (self.embedding_scale - 1.0).abs() > f32::EPSILON {
            for h in &mut hidden {
                *h *= self.embedding_scale;
            }
        }

        let per_kv_layer_elems = 2 * seq_len.max(pos + 1) * n_kv * head_d_even;
        let kv_region_len = num_kv_layers
            .saturating_mul(per_kv_layer_elems)
            .min(kv_cache.len());
        let (kv_region, rec_region) = kv_cache.split_at_mut(kv_region_len);

        let mut rec_arena =
            PingPongRecurrentArena::open(rec_region, num_rec_layers, rec_slot_floats);
        let mut slot_buf = vec![0.0_f32; rec_slot_floats];
        let residual_scale = self.residual_scale;
        let ssm_geom = (d_inner, d_conv, d_state, ssm_heads, ssm_head_dim, n_groups);

        for (layer_idx, &kind) in schedule.iter().enumerate() {
            self.run_hybrid_layer_step(
                layer_idx,
                kind,
                (pos, seq_len),
                &mut hidden,
                attn_spec,
                kv_region,
                &mut rec_arena,
                &mut slot_buf,
                ssm_geom,
                num_kv_layers,
                weights,
            )?;

            let ffn_norm_w = read_f32_weight_or_default(
                weights,
                &format!("blk.{layer_idx}.ffn_norm.weight"),
                hidden_dim,
                1.0,
            );
            let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, self.rms_eps)?;
            let ffn_out = run_ffn(layer_idx, &norm_ffn_in, weights)?;
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f * residual_scale;
            }
        }

        let step_num = u32::try_from(pos + 1).unwrap_or(u32::MAX);
        rec_arena.commit(step_num);

        if !logits.is_empty() {
            finalize_scaled_logits(&hidden, self.rms_eps, self.logit_scale, logits, weights)?;
        }
        Ok(())
    }
}

impl ArchRecipe for HybridSsmTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture_id
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(4);
        run_prefill_with(
            ctx,
            self.architecture_id,
            |tok, pos, seq_len, kv, logits| {
                self.execute_hybrid_token(
                    (tok, pos, seq_len),
                    logits,
                    kv,
                    weights,
                    |layer, norm_in, w| {
                        run_dense_swiglu_ffn(layer, hidden_dim, ffn_dim, norm_in, w)
                    },
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(2);
        let ffn_dim = self.ffn_dim.max(4);
        run_decode_with(ctx, self.architecture_id, |tok, pos, kv, logits| {
            self.execute_hybrid_token(
                (tok, pos, pos + 1),
                logits,
                kv,
                weights,
                |layer, norm_in, w| run_dense_swiglu_ffn(layer, hidden_dim, ffn_dim, norm_in, w),
            )
        })
    }
}

/// Execution recipe for `LiquidAI` `LFM2.5-MoE` (`lfm2moe`, `#104`–`#106`) combining gated causal
/// short-convolution + GQA hybrid layers with sparse top-K `MoE` FFN blocks.
#[derive(Debug, Clone)]
pub struct Lfm2MoeRecipe {
    /// Underlying hybrid `ShortConv` + GQA backbone configuration.
    pub base: HybridSsmTransformerRecipe,
    /// Number of routed sparse experts per `MoE` layer.
    pub num_experts: usize,
    /// Number of active experts selected per token (`top_k`).
    pub num_experts_per_tok: usize,
    /// Per-expert feed-forward intermediate dimension.
    pub expert_ffn_dim: usize,
    /// Number of leading dense `SwiGLU` layers before sparse `MoE` layers begin (when router weights absent).
    pub num_dense_layers: usize,
}

impl Default for Lfm2MoeRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfm2MoeRecipe {
    /// Creates an `LFM2.5-MoE` recipe targeting `"lfm2moe"` (`#106`).
    #[must_use]
    pub fn new() -> Self {
        let mut base = HybridSsmTransformerRecipe::new_lfm2();
        base.architecture_id = "lfm2moe";
        Self {
            base,
            num_experts: 32,
            num_experts_per_tok: 4,
            expert_ffn_dim: 1792,
            num_dense_layers: 2,
        }
    }

    fn execute_layer_ffn(
        &self,
        layer: usize,
        norm_in: &[f32],
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let hidden_dim = self.base.hidden_dim.max(2);
        let dense_dim = self.base.ffn_dim.max(4);
        let is_moe = weights.contains_tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
            || (weights.tensor_count() == 0 && layer >= self.num_dense_layers);
        if !is_moe {
            return run_dense_swiglu_ffn(layer, hidden_dim, dense_dim, norm_in, weights);
        }
        let num_experts = self.num_experts.max(1);
        let top_k = self.num_experts_per_tok.clamp(1, num_experts);
        let expert_ffn_dim = self.expert_ffn_dim.max(4);
        let mut out = vec![0.0_f32; hidden_dim];
        let _ = dispatch_qwen3_moe_ffn(
            &mut out,
            norm_in,
            layer,
            num_experts,
            top_k,
            expert_ffn_dim,
            false,
            weights,
        )?;
        Ok(out)
    }
}

impl ArchRecipe for Lfm2MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "lfm2moe"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, "lfm2moe", |tok, pos, seq_len, kv, logits| {
            self.base
                .execute_hybrid_token((tok, pos, seq_len), logits, kv, weights, |l, n, w| {
                    self.execute_layer_ffn(l, n, w)
                })
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "lfm2moe", |tok, pos, kv, logits| {
            self.base
                .execute_hybrid_token((tok, pos, pos + 1), logits, kv, weights, |l, n, w| {
                    self.execute_layer_ffn(l, n, w)
                })
        })
    }
}

/// Execution recipe for dense IBM `Granite` (`granite` — Granite 3 / `Granite-4.1-Guardian-8B`)
/// transformers supporting GGUF `embedding_scale`, `residual_scale`, `attention.scale`, and
/// `logit_scale` multipliers (`#104`–`#107`).
#[derive(Debug, Clone)]
pub struct GraniteRecipe {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Dense `SwiGLU` intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value attention head count (`GQA`).
    pub n_heads_kv: usize,
    /// Scalar dimension per attention head.
    pub head_dim: usize,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency.
    pub rope_theta: f32,
    /// Multiplier applied to input token embeddings (`granite.embedding_scale`).
    pub embedding_scale: f32,
    /// Multiplier applied to residual branch deltas (`granite.residual_scale`).
    pub residual_scale: f32,
    /// Explicit QK score scale overriding `1 / sqrt(head_dim)` (`granite.attention.scale`).
    pub attention_scale: Option<f32>,
    /// Final logit divisor (`granite.logit_scale`, scaling output logits by `1.0 / logit_scale`).
    pub logit_scale: f32,
}

impl Default for GraniteRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl GraniteRecipe {
    /// Creates a `Granite` dense recipe with 8B-class defaults (`#107`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 40,
            hidden_dim: 4096,
            ffn_dim: 12_800,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 49_155,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            embedding_scale: 1.0,
            residual_scale: 1.0,
            attention_scale: None,
            logit_scale: 1.0,
        }
    }

    fn execute_single_token(
        &self,
        tok: u32,
        pos: usize,
        seq_len: usize,
        logits: &mut [f32],
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(2);
        let head_d_even = even_head_dim(self.head_dim);
        let n_kv = self.n_heads_kv.max(1);
        let kv_dim = n_kv * head_d_even;
        let num_layers = self.num_layers.max(1);

        let attn_spec = DenseAttnLayerSpec {
            hidden_dim,
            ffn_dim: self.ffn_dim.max(4),
            n_heads_q: self.n_heads_q.max(1),
            n_heads_kv: n_kv,
            head_dim: head_d_even,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: false,
            use_qk_norm: false,
        };

        let mut hidden = embed_token(tok, hidden_dim, weights)?;
        if (self.embedding_scale - 1.0).abs() > f32::EPSILON {
            for h in &mut hidden {
                *h *= self.embedding_scale;
            }
        }

        let tokens_per_layer = seq_len.max(pos + 1);
        let per_layer_kv_elems = 2 * tokens_per_layer * kv_dim;
        for layer in 0..num_layers {
            let mut layer_spec = attn_spec;
            layer_spec.use_qkv_bias = weights.contains_tensor(&format!("blk.{layer}.attn_q.bias"));
            let layer_kv =
                if per_layer_kv_elems > 0 && (layer + 1) * per_layer_kv_elems <= kv_cache.len() {
                    &mut kv_cache[layer * per_layer_kv_elems..(layer + 1) * per_layer_kv_elems]
                } else {
                    &mut kv_cache[..0]
                };
            run_dense_attention_sublayer_scaled(
                layer_spec,
                layer,
                pos,
                seq_len,
                &mut hidden,
                layer_kv,
                weights,
                self.attention_scale,
                self.residual_scale,
            )?;

            let ffn_norm_w = read_f32_weight_or_default(
                weights,
                &format!("blk.{layer}.ffn_norm.weight"),
                hidden_dim,
                1.0,
            );
            let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
            rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, self.rms_eps)?;
            let mut ffn_out =
                run_dense_swiglu_ffn(layer, hidden_dim, attn_spec.ffn_dim, &norm_ffn_in, weights)?;
            add_bias_if_present(&mut ffn_out, weights, &format!("blk.{layer}.ffn_down.bias"));
            for (h, &f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f * self.residual_scale;
            }
        }

        if !logits.is_empty() {
            finalize_scaled_logits(&hidden, self.rms_eps, self.logit_scale, logits, weights)?;
        }
        Ok(())
    }
}

impl ArchRecipe for GraniteRecipe {
    fn architecture_id(&self) -> &'static str {
        "granite"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with(ctx, "granite", |tok, pos, seq_len, kv, logits| {
            self.execute_single_token(tok, pos, seq_len, logits, kv, weights)
        })
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "granite", |tok, pos, kv, logits| {
            self.execute_single_token(tok, pos, pos + 1, logits, kv, weights)
        })
    }
}

#[cfg(test)]
mod tests {
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{GgufArray, GgufFile, GgufValue, TensorView};

    use super::*;
    use crate::capability::{RecipeRegistry, verify_model_capabilities};
    use crate::engine::{SinglePromptEngine, compute_calibrated_binary_risk};
    use crate::kv::SingleSlotKvArena;
    use crate::memory::{architecture_requires_kv_cache, estimate_kv_cache_bytes};

    struct EmptyMockWeights;
    impl ModelWeights for EmptyMockWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "empty mock weights".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_mamba2_recipe_forward() {
        let mut recipe = Mamba2Recipe::new();
        recipe.num_layers = 2;
        recipe.hidden_dim = 4;
        recipe.d_inner = 8;
        recipe.d_conv = 3;
        recipe.d_state = 2;
        recipe.n_heads = 2;
        recipe.n_groups = 1;
        recipe.vocab_size = 8;
        assert_eq!(recipe.architecture_id(), "mamba2");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q8_0));

        // Slot floats = 8 * (3 - 1) + 8 * 2 = 32 floats per layer -> bank = 64 floats.
        let mut kv_arena = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * 32];
        let mut logits_prefill = vec![0.0_f32; 8];
        {
            let mut ctx =
                ExecutionContext::new(&[1_u32, 3], 64, &mut logits_prefill, &mut kv_arena).unwrap();
            recipe
                .prefill_step(&mut ctx, &EmptyMockWeights)
                .expect("Mamba2Recipe chunked prefill succeeds");
            assert_eq!(ctx.position(), 2);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
        }
        assert!(logits_prefill.iter().all(|v| v.is_finite()));
        assert_eq!(inspect_ping_pong_header(&kv_arena), Some((1, 2)));

        let mut logits_decode = vec![0.0_f32; 8];
        {
            let mut ctx_dec = ExecutionContext::new_prefill_chunk(
                &[5_u32],
                64,
                2,
                true,
                &mut logits_decode,
                &mut kv_arena,
            )
            .unwrap();
            ctx_dec.prepare_decode(5).unwrap();
            recipe
                .decode_step(&mut ctx_dec, &EmptyMockWeights)
                .expect("Mamba2Recipe decode step succeeds");
            assert_eq!(ctx_dec.position(), 3);
            assert!(ctx_dec.logits().iter().all(|v| v.is_finite()));
        }
        assert!(logits_decode.iter().all(|v| v.is_finite()));
        assert_eq!(inspect_ping_pong_header(&kv_arena), Some((0, 3)));
    }

    #[test]
    fn test_rwkv7_recipe_execution() {
        let mut recipe = Rwkv7Recipe::new();
        recipe.num_layers = 2;
        recipe.hidden_dim = 4;
        recipe.ffn_dim = 8;
        recipe.head_size = 2;
        recipe.n_heads = 2;
        recipe.vocab_size = 8;
        assert_eq!(recipe.architecture_id(), "rwkv7");
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert_eq!(Rwkv7Recipe::new_rwkv6().architecture_id(), "rwkv6");

        let slot_floats = 2 * 4 + 2 * 2 * 2;
        let mut kv_arena = vec![0.0_f32; RECURRENT_HEADER_FLOATS + 2 * 2 * slot_floats];
        let mut logits = vec![0.0_f32; 8];
        {
            let mut ctx =
                ExecutionContext::new(&[2_u32, 4], 64, &mut logits, &mut kv_arena).unwrap();
            recipe
                .prefill_step(&mut ctx, &EmptyMockWeights)
                .expect("RWKV7 prefill executes");
            assert_eq!(ctx.position(), 2);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
        }
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(inspect_ping_pong_header(&kv_arena), Some((0, 2)));

        {
            let mut ctx = ExecutionContext::new_prefill_chunk(
                &[6_u32],
                64,
                2,
                true,
                &mut logits,
                &mut kv_arena,
            )
            .unwrap();
            ctx.prepare_decode(6).unwrap();
            recipe
                .decode_step(&mut ctx, &EmptyMockWeights)
                .expect("RWKV7 decode step flips ping-pong bank");
            assert_eq!(ctx.position(), 3);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
        }
        assert_eq!(inspect_ping_pong_header(&kv_arena), Some((1, 3)));
    }

    #[test]
    fn test_resolve_hybrid_layer_schedule() {
        let kvs = [
            ("lfm2.block_count", GgufValue::Uint32(4)),
            ("lfm2.embedding_length", GgufValue::Uint32(4)),
            ("lfm2.attention.head_count", GgufValue::Uint32(2)),
            ("lfm2.attention.head_count_kv", GgufValue::Uint32(1)),
            ("lfm2.context_length", GgufValue::Uint32(64)),
        ];
        let specs = [
            ("token_embd.weight", QuantType::F32, &[4_usize, 8][..]),
            ("blk.0.shortconv.conv.weight", QuantType::F32, &[4, 3][..]),
            ("blk.1.ssm_in.weight", QuantType::F32, &[4, 8][..]),
            ("blk.2.attn_q.weight", QuantType::F32, &[4, 4][..]),
            (
                "blk.3.shortconv.in_proj.weight",
                QuantType::F32,
                &[4, 12][..],
            ),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("lfm2", 64, &kvs, &specs);
        let gguf = GgufFile::parse(&bytes).expect("parse hybrid schedule gguf");
        let schedule = resolve_hybrid_layer_schedule(4, &gguf);
        assert_eq!(
            schedule,
            vec![
                HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: 0
                },
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: 1
                },
                HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
                HybridLayerKind::Lfm2ShortConv {
                    recurrent_layer_slot: 2
                },
            ]
        );
    }

    #[test]
    fn test_resolve_hybrid_layer_schedule_and_memory() {
        let kvs = [
            (
                "granitehybrid.attention.head_count_kv",
                GgufValue::Array(GgufArray::Uint32(vec![0, 2, 0, 0])),
            ),
            ("granitehybrid.block_count", GgufValue::Uint32(4)),
            ("granitehybrid.embedding_length", GgufValue::Uint32(16)),
            ("granitehybrid.attention.head_count", GgufValue::Uint32(4)),
            ("granitehybrid.ssm.inner_size", GgufValue::Uint32(16)),
            ("granitehybrid.ssm.conv_kernel", GgufValue::Uint32(4)),
            ("granitehybrid.ssm.state_size", GgufValue::Uint32(4)),
            ("granitehybrid.context_length", GgufValue::Uint32(1024)),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("granitehybrid", 1024, &kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse granitehybrid schedule metadata");
        let schedule = resolve_hybrid_layer_schedule_with_metadata(
            "granitehybrid",
            4,
            Some(gguf.metadata()),
            &gguf,
        );
        assert_eq!(
            schedule,
            vec![
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: 0
                },
                HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: 1
                },
                HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: 2
                },
            ]
        );
        let hybrid_kv = estimate_kv_cache_bytes(gguf.metadata(), 64, true).expect("hybrid bytes");
        assert!(hybrid_kv > 0);
        assert!(architecture_requires_kv_cache("granitehybrid"));
    }

    #[test]
    fn test_hybrid_ssm_transformer_recipe() {
        let schedule = vec![
            HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: 0,
            },
            HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
        ];
        let mut recipe = HybridSsmTransformerRecipe::from_schedule("jamba", schedule);
        recipe.hidden_dim = 4;
        recipe.ffn_dim = 8;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 2;
        recipe.d_inner = 4;
        recipe.d_conv = 2;
        recipe.d_state = 2;
        recipe.vocab_size = 8;

        let mut kv_arena = vec![0.0_f32; 128];
        let mut logits = vec![0.0_f32; 8];
        {
            let mut ctx =
                ExecutionContext::new(&[1_u32, 2], 64, &mut logits, &mut kv_arena).unwrap();
            recipe
                .prefill_step(&mut ctx, &EmptyMockWeights)
                .expect("hybrid prefill succeeds");
            assert_eq!(ctx.position(), 2);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
            ctx.prepare_decode(3).unwrap();
            recipe
                .decode_step(&mut ctx, &EmptyMockWeights)
                .expect("hybrid decode succeeds");
            assert_eq!(ctx.position(), 3);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
        }
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lfm2_and_granitehybrid_forward() {
        for arch in ["lfm2", "granitehybrid"] {
            let schedule = if arch == "lfm2" {
                vec![
                    HybridLayerKind::Lfm2ShortConv {
                        recurrent_layer_slot: 0,
                    },
                    HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
                ]
            } else {
                vec![
                    HybridLayerKind::Mamba2Ssd {
                        recurrent_layer_slot: 0,
                    },
                    HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
                ]
            };
            let mut recipe = HybridSsmTransformerRecipe::from_schedule(arch, schedule);
            recipe.hidden_dim = 4;
            recipe.ffn_dim = 8;
            recipe.n_heads_q = 2;
            recipe.n_heads_kv = 1;
            recipe.head_dim = 2;
            recipe.d_inner = 4;
            recipe.d_conv = 2;
            recipe.d_state = 2;
            recipe.shortconv_l_cache = 3;
            recipe.vocab_size = 8;

            let mut kv_arena = vec![0.0_f32; 128];
            let mut logits = vec![0.0_f32; 8];
            let mut ctx =
                ExecutionContext::new(&[0_u32, 1], 64, &mut logits, &mut kv_arena).unwrap();
            recipe.prefill_step(&mut ctx, &EmptyMockWeights).unwrap();
            ctx.prepare_decode(2).unwrap();
            recipe.decode_step(&mut ctx, &EmptyMockWeights).unwrap();
            assert_eq!(ctx.position(), 3);
            assert!(ctx.logits().iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_hybrid_jamba_granitehybrid_lfm2_dispatch() {
        let gguf_arches: &[&str] = if cfg!(miri) {
            &["mamba2", "rwkv7", "jamba", "lfm2", "lfm2moe", "granite"]
        } else {
            &[
                "mamba",
                "mamba2",
                "rwkv6",
                "rwkv7",
                "jamba",
                "zamba2",
                "granitehybrid",
                "lfm2",
                "lfm2moe",
                "granite",
            ]
        };
        for arch in [
            "mamba",
            "mamba2",
            "rwkv6",
            "rwkv7",
            "jamba",
            "zamba2",
            "granitehybrid",
            "lfm2",
            "lfm2moe",
            "granite",
        ] {
            let static_recipe = RecipeRegistry::find_recipe(arch)
                .unwrap_or_else(|| panic!("find_recipe missing for {arch}"));
            assert_eq!(static_recipe.architecture_id(), arch);
            assert!(RecipeRegistry::supported_architectures().contains(&arch));

            if !gguf_arches.contains(&arch) {
                continue;
            }
            let kvs = [
                (format!("{arch}.block_count"), GgufValue::Uint32(2)),
                (format!("{arch}.embedding_length"), GgufValue::Uint32(4)),
                (format!("{arch}.feed_forward_length"), GgufValue::Uint32(8)),
                (format!("{arch}.attention.head_count"), GgufValue::Uint32(2)),
                (
                    format!("{arch}.attention.head_count_kv"),
                    GgufValue::Uint32(1),
                ),
                (format!("{arch}.context_length"), GgufValue::Uint32(64)),
            ];
            let kv_refs: Vec<(&str, GgufValue)> =
                kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            let gguf_bytes = create_synthetic_gguf_v3_with_kvs(arch, 64, &kv_refs, &[]);
            let gguf = GgufFile::parse(&gguf_bytes).expect("parse minimal arch gguf");
            let hydrated = verify_model_capabilities(&gguf).expect("capability admission succeeds");
            assert_eq!(hydrated.architecture_id(), arch);
        }
    }

    #[test]
    fn test_lfm2moe_recipe_forward() {
        let mut recipe = Lfm2MoeRecipe::new();
        recipe.base.layer_schedule = vec![
            HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot: 0,
            },
            HybridLayerKind::GqaAttention { kv_layer_slot: 0 },
        ];
        recipe.base.hidden_dim = 4;
        recipe.base.ffn_dim = 8;
        recipe.base.n_heads_q = 2;
        recipe.base.n_heads_kv = 1;
        recipe.base.head_dim = 2;
        recipe.base.shortconv_l_cache = 3;
        recipe.base.vocab_size = 8;
        recipe.num_experts = 4;
        recipe.num_experts_per_tok = 2;
        recipe.expert_ffn_dim = 8;
        recipe.num_dense_layers = 1;

        assert_eq!(recipe.architecture_id(), "lfm2moe");
        let mut kv_arena = vec![0.0_f32; 128];
        let mut logits = vec![0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[1_u32, 3], 64, &mut logits, &mut kv_arena).unwrap();
        recipe
            .prefill_step(&mut ctx, &EmptyMockWeights)
            .expect("Lfm2MoeRecipe prefill executes dense + MoE layers");
        ctx.prepare_decode(2).unwrap();
        recipe
            .decode_step(&mut ctx, &EmptyMockWeights)
            .expect("Lfm2MoeRecipe decode executes");
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_granite_recipe_multipliers_forward() {
        let mut base = GraniteRecipe::new();
        base.num_layers = 1;
        base.hidden_dim = 4;
        base.ffn_dim = 8;
        base.n_heads_q = 2;
        base.n_heads_kv = 1;
        base.head_dim = 2;
        base.vocab_size = 8;

        let mut scaled = base.clone();
        scaled.embedding_scale = 2.0;
        scaled.residual_scale = 0.5;
        scaled.attention_scale = Some(0.25);
        scaled.logit_scale = 4.0;

        let mut kv_a = vec![0.0_f32; 64];
        let mut kv_b = vec![0.0_f32; 64];
        let mut logits_a = vec![0.0_f32; 8];
        let mut logits_b = vec![0.0_f32; 8];
        let mut ctx_a = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_a, &mut kv_a).unwrap();
        let mut ctx_b = ExecutionContext::new(&[1_u32, 2], 64, &mut logits_b, &mut kv_b).unwrap();
        base.prefill_step(&mut ctx_a, &EmptyMockWeights).unwrap();
        scaled.prefill_step(&mut ctx_b, &EmptyMockWeights).unwrap();
        assert_ne!(
            logits_a, logits_b,
            "Granite scaling multipliers must affect logits"
        );
        assert!(logits_b.iter().all(|v| v.is_finite()));

        // Verify calibrated risk evaluation on a compact Granite GGUF (Guardian 8B compatible).
        let gguf_bytes = create_synthetic_gguf_v3_with_kvs(
            "granite",
            64,
            &[
                ("granite.block_count", GgufValue::Uint32(1)),
                ("granite.embedding_length", GgufValue::Uint32(4)),
                ("granite.feed_forward_length", GgufValue::Uint32(8)),
                ("granite.attention.head_count", GgufValue::Uint32(2)),
                ("granite.attention.head_count_kv", GgufValue::Uint32(1)),
                ("granite.embedding_scale", GgufValue::Float32(12.0)),
                ("granite.residual_scale", GgufValue::Float32(0.22)),
                ("granite.attention.scale", GgufValue::Float32(0.015_625)),
                ("granite.logit_scale", GgufValue::Float32(8.0)),
                ("granite.context_length", GgufValue::Uint32(64)),
            ],
            &[("token_embd.weight", QuantType::F32, &[4_usize, 8][..])],
        );
        let gguf = GgufFile::parse(&gguf_bytes).unwrap();
        let hydrated = verify_model_capabilities(&gguf).expect("granite hydrated");
        let mut arena = SingleSlotKvArena::from_gguf(&gguf, 64).expect("granite arena");
        let mut engine = SinglePromptEngine::new();
        let (out_logits, _, pos) = engine
            .prefill_step_with_pool(&*hydrated, &gguf, &mut arena, None, &[0_u32, 1], 0, 8)
            .expect("granite SinglePromptEngine prefill");
        assert_eq!(pos, 2);
        let score = compute_calibrated_binary_risk(&out_logits, 1, 2).expect("binary risk");
        assert!(score.unsafe_probability.is_finite());
    }

    #[test]
    fn test_single_slot_kv_arena_o1_recurrent_and_hybrid_end_to_end() {
        let mamba_bytes_64 = create_synthetic_gguf_v3_with_kvs(
            "mamba2",
            64,
            &[
                ("mamba2.block_count", GgufValue::Uint32(2)),
                ("mamba2.embedding_length", GgufValue::Uint32(4)),
                ("mamba2.ssm.inner_size", GgufValue::Uint32(8)),
                ("mamba2.ssm.conv_kernel", GgufValue::Uint32(3)),
                ("mamba2.ssm.state_size", GgufValue::Uint32(2)),
                ("mamba2.ssm.time_step_rank", GgufValue::Uint32(2)),
                ("mamba2.context_length", GgufValue::Uint32(2048)),
            ],
            &[("token_embd.weight", QuantType::F32, &[4_usize, 8][..])],
        );
        let mamba_gguf = GgufFile::parse(&mamba_bytes_64).unwrap();
        let arena_short = SingleSlotKvArena::from_gguf(&mamba_gguf, 64).expect("mamba arena 64");
        let arena_long = SingleSlotKvArena::from_gguf(&mamba_gguf, 2048).expect("mamba arena 2048");
        assert_eq!(arena_short.as_slice().len(), arena_long.as_slice().len());
        assert_eq!(arena_short.allocated_tokens(), 0);
        assert_eq!(arena_long.allocated_tokens(), 0);

        let lfm_bytes = create_synthetic_gguf_v3_with_kvs(
            "lfm2",
            64,
            &[
                ("lfm2.block_count", GgufValue::Uint32(2)),
                ("lfm2.embedding_length", GgufValue::Uint32(4)),
                ("lfm2.feed_forward_length", GgufValue::Uint32(8)),
                ("lfm2.attention.head_count", GgufValue::Uint32(2)),
                (
                    "lfm2.attention.head_count_kv",
                    GgufValue::Array(GgufArray::Uint32(vec![0, 1])),
                ),
                ("lfm2.shortconv.l_cache", GgufValue::Uint32(3)),
                ("lfm2.context_length", GgufValue::Uint32(64)),
            ],
            &[("token_embd.weight", QuantType::F32, &[4_usize, 8][..])],
        );
        let lfm_gguf = GgufFile::parse(&lfm_bytes).unwrap();
        let hydrated = verify_model_capabilities(&lfm_gguf).expect("lfm2 capability");
        let mut lfm_arena = SingleSlotKvArena::from_gguf(&lfm_gguf, 64).expect("lfm2 arena");
        assert!(lfm_arena.allocated_tokens() > 0);
        let mut engine = SinglePromptEngine::new();
        let (out_logits, _, pos) = engine
            .prefill_step_with_pool(
                &*hydrated,
                &lfm_gguf,
                &mut lfm_arena,
                None,
                &[0_u32, 1],
                0,
                8,
            )
            .expect("lfm2 end-to-end prefill");
        assert_eq!(pos, 2);
        assert!(out_logits.iter().all(|v| v.is_finite()));
    }
}
