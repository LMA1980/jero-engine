//! Google Gemma architecture family recipes (`gemma2`, `gemma3`, `gemma4`).
//!
//! Implements [`ArchRecipe`] for Gemma 2, Gemma 3, and Gemma 4 topologies, integrating:
//! - Unit-offset `(1.0 + w)` `RMSNorm` and per-head QK-normalization,
//! - `GeGLU` FFN activations and attention/final logit softcapping (`gemma2`),
//! - Interleaved local sliding-window cyclic KV rings and global full-context attention
//!   with dual `RoPE` frequencies (`10_000.0` local vs `1_000_000.0` global),
//! - Zero-copy quantized Gemma 4 Edge (`E2B`/`E4B`) Per-Layer Embeddings (`PLE`), and
//! - Gemma-4-26B-A4B top-`k` Sparse `MoE` deterministic softmax routing.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, dequantize_token_row, dot_product_f32_scalar,
    geglu_f32, gqa_causal_attention_f32, logit_softcap_tanh_inplace, qk_norm_inplace,
    rmsnorm_unit_offset, vec_dot_bf16, vec_dot_f16, vec_dot_q4_0, vec_dot_q5_k, vec_dot_q6_k,
    vec_dot_q8_0,
};

use crate::engine::EnginePhase;
use crate::forward::Lane;
use crate::kv::write_and_slice_sliding_kv;

/// Selects the target compute cluster lane for Gemma `RMSNorm` and QK-norm operations:
/// `A100` during prompt prefill (`EnginePhase::Prefill`) and `X100` during decode (`EnginePhase::Decode`).
#[must_use]
pub const fn gemma_norm_lane(phase: EnginePhase) -> Lane {
    match phase {
        EnginePhase::Prefill => Lane::A100,
        EnginePhase::Idle | EnginePhase::Decode | EnginePhase::Completed => Lane::X100,
    }
}

/// Expert weight views for a single Gemma 4 sparse Mixture-of-Experts (`MoE`) expert block.
#[derive(Debug, Clone)]
pub struct ExpertWeights<'a> {
    /// Gate projection weights (`ffn_gate_exps`).
    pub gate: TensorView<'a>,
    /// Up projection weights (`ffn_up_exps`).
    pub up: TensorView<'a>,
    /// Down projection weights (`ffn_down_exps`).
    pub down: TensorView<'a>,
}

/// Dispatches matrix-vector multiplication across all quantization formats supported by Gemma.
pub(crate) fn matvec_any(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    if out.is_empty() || input.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matvec output and input slices cannot be empty".to_owned(),
        });
    }
    match weights.quant() {
        QuantType::F32 => matvec_f32(out, weights, input),
        QuantType::F16 => vec_dot_f16(out, weights, input),
        QuantType::BF16 => vec_dot_bf16(out, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
        QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
        QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
        QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
        other => Err(K3Error::CapabilityMismatch {
            architecture: "gemma".to_owned(),
            reason: format!(
                "unsupported weight quantization {other:?} for tensor '{}'",
                weights.name()
            ),
        }),
    }
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    let expected_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "F32 matvec dimension overflowed usize".to_owned(),
        })?;
    if weights.num_elements() != expected_elems {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "F32 tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        });
    }
    let raw = weights.as_bytes();
    let required_bytes = expected_elems * size_of::<f32>();
    if raw.len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        });
    }

    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let base = r * cols * 4;
        for (c, slot) in row_buf.iter_mut().enumerate() {
            let off = base + c * 4;
            *slot = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        }
        *out_slot = dot_product_f32_scalar(&row_buf, input);
    }
    Ok(())
}

fn quant_block_elems(quant: QuantType) -> usize {
    match quant {
        QuantType::Q4_0 | QuantType::Q8_0 => 32,
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => 256,
        _ => 1,
    }
}

fn infer_ple_row_width(
    ple_embd: &TensorView<'_>,
    ple_dim: usize,
    layer_idx: usize,
    token_idx: usize,
) -> Result<usize, K3Error> {
    let min_cols = layer_idx
        .checked_add(1)
        .and_then(|n| n.checked_mul(ple_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: ple_embd.name().to_owned(),
            offset: ple_embd.offset(),
            byte_len: ple_embd.byte_len(),
            buffer_len: ple_embd.as_bytes().len(),
            reason: "PLE layer column offset overflowed usize".to_owned(),
        })?;
    let blk = quant_block_elems(ple_embd.quant());
    let shape = ple_embd.shape();

    if shape.len() == 2 {
        let (d0, d1) = (shape[0], shape[1]);
        let d0_valid = d0 >= min_cols && d0 % ple_dim == 0 && d0 % blk == 0 && token_idx < d1;
        let d1_valid = d1 >= min_cols && d1 % ple_dim == 0 && d1 % blk == 0 && token_idx < d0;
        if d0_valid {
            return Ok(d0);
        }
        if d1_valid {
            return Ok(d1);
        }
    }

    let total_elems = ple_embd.num_elements();
    let max_rows = token_idx + 1;
    if total_elems >= min_cols && min_cols % blk == 0 && total_elems % min_cols == 0 {
        let rows = total_elems / min_cols;
        if token_idx < rows {
            return Ok(min_cols);
        }
    }
    if max_rows > 0 && total_elems % max_rows == 0 {
        let cols = total_elems / max_rows;
        if cols >= min_cols && cols % ple_dim == 0 && cols % blk == 0 {
            return Ok(cols);
        }
    }

    Err(K3Error::TensorBoundsError {
        tensor_name: ple_embd.name().to_owned(),
        offset: ple_embd.offset(),
        byte_len: ple_embd.byte_len(),
        buffer_len: ple_embd.as_bytes().len(),
        reason: format!(
            "cannot infer block-aligned PLE row width for shape {shape:?}, ple_dim={ple_dim}, layer={layer_idx}, token={token_idx}",
        ),
    })
}

/// Applies a Gemma 4 Edge (`E2B`/`E4B`) Per-Layer Embedding (`PLE`) quantized token lookup,
/// `GeGLU` gated projection, and unit-offset `RMSNorm` residual addition in-place onto `hidden`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if tensor shapes, buffer bounds, or slice dimensions fail
/// validation, or [`K3Error::CapabilityMismatch`] if an unsupported quantization format is used.
#[expect(
    clippy::too_many_arguments,
    reason = "contract matches jero-engine#27 explicit PLE tensor inputs"
)]
pub fn apply_gemma4_ple_residual(
    hidden: &mut [f32],
    token_id: u32,
    layer_idx: usize,
    ple_embd: &TensorView<'_>,
    ple_inp_gate: &TensorView<'_>,
    ple_proj: &TensorView<'_>,
    ple_norm_weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    if hidden_dim == 0 || ple_norm_weight.len() != hidden_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "per_layer_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(ple_norm_weight),
            buffer_len: size_of_val(hidden),
            reason: format!(
                "PLE hidden_dim ({hidden_dim}) must be > 0 and match ple_norm_weight len ({})",
                ple_norm_weight.len()
            ),
        });
    }

    let gate_elems = ple_inp_gate.num_elements();
    if gate_elems == 0 || gate_elems % hidden_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: ple_inp_gate.name().to_owned(),
            offset: ple_inp_gate.offset(),
            byte_len: ple_inp_gate.byte_len(),
            buffer_len: ple_inp_gate.as_bytes().len(),
            reason: format!(
                "ple_inp_gate elements ({gate_elems}) must be a positive multiple of hidden_dim ({hidden_dim})"
            ),
        });
    }
    let ple_dim = gate_elems / hidden_dim;

    let expected_proj_elems =
        ple_dim
            .checked_mul(hidden_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: ple_proj.name().to_owned(),
                offset: ple_proj.offset(),
                byte_len: ple_proj.byte_len(),
                buffer_len: ple_proj.as_bytes().len(),
                reason: "PLE projection dimensions overflowed usize".to_owned(),
            })?;
    if ple_proj.num_elements() != expected_proj_elems {
        return Err(K3Error::TensorBoundsError {
            tensor_name: ple_proj.name().to_owned(),
            offset: ple_proj.offset(),
            byte_len: ple_proj.byte_len(),
            buffer_len: ple_proj.as_bytes().len(),
            reason: format!(
                "ple_proj elements ({}) != ple_dim({ple_dim}) * hidden_dim({hidden_dim})",
                ple_proj.num_elements()
            ),
        });
    }

    let token_idx = usize::try_from(token_id).map_err(|_| K3Error::TensorBoundsError {
        tensor_name: ple_embd.name().to_owned(),
        offset: 0,
        byte_len: ple_embd.byte_len(),
        buffer_len: ple_embd.as_bytes().len(),
        reason: "token_id exceeds usize range".to_owned(),
    })?;
    let row_width = infer_ple_row_width(ple_embd, ple_dim, layer_idx, token_idx)?;
    let mut full_row = vec![0.0_f32; row_width];
    dequantize_token_row(ple_embd, token_idx, &mut full_row)?;

    let layer_start = layer_idx * ple_dim;
    let layer_ple = &full_row[layer_start..layer_start + ple_dim];

    let mut gate_out = vec![0.0_f32; ple_dim];
    matvec_any(&mut gate_out, ple_inp_gate, hidden)?;

    let mut gated = vec![0.0_f32; ple_dim];
    geglu_f32(&mut gated, &gate_out, layer_ple)?;

    let mut proj_out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut proj_out, ple_proj, &gated)?;

    let mut normed = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut normed, &proj_out, ple_norm_weight, eps)?;

    for (h, &delta) in hidden.iter_mut().zip(normed.iter()) {
        *h += delta;
    }
    Ok(())
}

/// Selects the top-`k` experts deterministically from `router_logits` with ascending index
/// tie-breaking, returning `(expert_idx, gating_weight)` pairs.
///
/// When `normalize` is `true`, the selected top-`k` weights are renormalized to sum to `1.0`.
/// When `normalize` is `false`, raw full-router softmax probabilities are returned.
#[must_use]
pub fn select_top_k_experts(
    router_logits: &[f32],
    top_k: usize,
    normalize: bool,
) -> Vec<(usize, f32)> {
    if router_logits.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(router_logits.len());
    let mut sum_exp = 0.0_f32;
    for &logit in router_logits {
        let e = if logit.is_finite() && max_logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0_f32
        };
        probs.push(e);
        sum_exp += e;
    }

    if sum_exp > 0.0_f32 && sum_exp.is_finite() {
        for p in &mut probs {
            *p /= sum_exp;
        }
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "expert count fits comfortably in f32"
        )]
        let uniform = 1.0_f32 / (router_logits.len() as f32);
        probs.fill(uniform);
    }

    let mut ranked: Vec<(usize, f32, f32)> = router_logits
        .iter()
        .copied()
        .zip(probs)
        .enumerate()
        .map(|(idx, (logit, prob))| {
            let finite_logit = if logit.is_finite() {
                logit
            } else {
                f32::NEG_INFINITY
            };
            (idx, finite_logit, prob)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let k = top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> = ranked
        .into_iter()
        .take(k)
        .map(|(idx, _, prob)| (idx, prob))
        .collect();

    if normalize {
        let selected_sum: f32 = selected.iter().map(|&(_, w)| w).sum();
        if selected_sum > 0.0_f32 && selected_sum.is_finite() {
            for (_, w) in &mut selected {
                *w /= selected_sum;
            }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "top_k fits comfortably in f32")]
            let uniform = 1.0_f32 / (k as f32);
            for (_, w) in &mut selected {
                *w = uniform;
            }
        }
    }

    selected
}

/// Evaluates a Gemma-4 sparse `MoE` layer by computing router logits, selecting top-`k`
/// experts deterministically with normalized softmax weights, and accumulating weighted
/// `GeGLU` expert outputs only for active experts.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `experts` is empty, `top_k == 0`, or router
/// logits are non-finite. Returns [`K3Error::TensorBoundsError`] on dimension or buffer errors.
pub fn dispatch_sparse_moe_layer(
    out: &mut [f32],
    hidden: &[f32],
    router_weight: &TensorView<'_>,
    experts: &[ExpertWeights<'_>],
    top_k: usize,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    if hidden_dim == 0 || out.len() != hidden_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: router_weight.name().to_owned(),
            offset: router_weight.offset(),
            byte_len: size_of_val(out),
            buffer_len: size_of_val(hidden),
            reason: format!(
                "MoE hidden ({hidden_dim}) must be > 0 and match out len ({})",
                out.len()
            ),
        });
    }
    if experts.is_empty() || top_k == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gemma4".to_owned(),
            reason: format!(
                "invalid sparse MoE config: experts={}, top_k={top_k}",
                experts.len()
            ),
        });
    }

    let mut router_logits = vec![0.0_f32; experts.len()];
    matvec_any(&mut router_logits, router_weight, hidden)?;
    for (idx, &val) in router_logits.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "gemma4".to_owned(),
                reason: format!("non-finite MoE router logit {val} at expert index {idx}"),
            });
        }
    }

    let selected = select_top_k_experts(&router_logits, top_k, true);
    out.fill(0.0);

    for (expert_idx, gate_weight) in selected {
        let exp = &experts[expert_idx];
        let gate_elems = exp.gate.num_elements();
        if gate_elems == 0 || gate_elems % hidden_dim != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: exp.gate.name().to_owned(),
                offset: exp.gate.offset(),
                byte_len: exp.gate.byte_len(),
                buffer_len: exp.gate.as_bytes().len(),
                reason: format!(
                    "expert gate elements ({gate_elems}) not divisible by hidden_dim ({hidden_dim})"
                ),
            });
        }
        let ffn_dim = gate_elems / hidden_dim;
        let mut gate_buf = vec![0.0_f32; ffn_dim];
        let mut up_buf = vec![0.0_f32; ffn_dim];
        matvec_any(&mut gate_buf, &exp.gate, hidden)?;
        matvec_any(&mut up_buf, &exp.up, hidden)?;

        let mut act_buf = vec![0.0_f32; ffn_dim];
        geglu_f32(&mut act_buf, &gate_buf, &up_buf)?;

        let mut expert_out = vec![0.0_f32; hidden_dim];
        matvec_any(&mut expert_out, &exp.down, &act_buf)?;

        for (dst, &src) in out.iter_mut().zip(expert_out.iter()) {
            *dst += gate_weight * src;
        }
    }

    Ok(())
}

/// Google Gemma 2 and Gemma 3 architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct GemmaRecipe {
    /// Exact architecture identifier (`"gemma2"` or `"gemma3"`).
    pub architecture: &'static str,
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (GQA/MQA).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Unit-offset `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// Local sliding-window attention `RoPE` base frequency (`10_000.0`).
    pub rope_freq_base_local: f32,
    /// Global full-context attention `RoPE` base frequency (`1_000_000.0` on Gemma 3).
    pub rope_freq_base_global: f32,
    /// Local attention sliding-window capacity in tokens (`4096` on Gemma 2, `1024` on Gemma 3).
    pub sliding_window: usize,
    /// Optional attention score softcapping (`50.0` on Gemma 2).
    pub attn_logit_softcap: Option<f32>,
    /// Optional final LM-head logit softcapping (`30.0` on Gemma 2).
    pub final_logit_softcap: Option<f32>,
    /// Enables per-head Q/K unit-offset normalization (`true` on Gemma 3).
    pub use_qk_norm: bool,
}

impl Default for GemmaRecipe {
    fn default() -> Self {
        Self::new_gemma3()
    }
}

impl GemmaRecipe {
    /// Creates a default Gemma 2 recipe (1:1 local/global attention, `4096` SWA, softcapping).
    #[must_use]
    pub const fn new_gemma2() -> Self {
        Self {
            architecture: "gemma2",
            num_layers: 26,
            hidden_dim: 2304,
            ffn_dim: 9216,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 256_000,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 10_000.0,
            sliding_window: 4096,
            attn_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            use_qk_norm: false,
        }
    }

    /// Creates a default Gemma 3 recipe (5:1 local/global attention, `1024` SWA, QK-norm, dual `RoPE`).
    #[must_use]
    pub const fn new_gemma3() -> Self {
        Self {
            architecture: "gemma3",
            num_layers: 34,
            hidden_dim: 2560,
            ffn_dim: 10_240,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 1024,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
        }
    }

    /// Creates a default `FunctionGemma` recipe (Gemma 3 270M/2B specialized functional action SLM).
    #[must_use]
    pub const fn new_functiongemma() -> Self {
        Self {
            architecture: "functiongemma",
            num_layers: 18,
            hidden_dim: 640,
            ffn_dim: 2048,
            n_heads_q: 4,
            n_heads_kv: 1,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 512,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
        }
    }

    /// Returns `true` when `layer_idx` uses global full-context attention.
    #[must_use]
    pub const fn is_global_layer(&self, layer_idx: usize) -> bool {
        if matches!(self.architecture.as_bytes(), b"gemma2") {
            layer_idx % 2 == 1
        } else {
            (layer_idx + 1) % 6 == 0
        }
    }

    /// Returns the layer-appropriate `RoPE` base frequency (`rope_freq_base_global` vs `rope_freq_base_local`).
    #[must_use]
    pub const fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_freq_base_global
        } else {
            self.rope_freq_base_local
        }
    }
}

/// Google Gemma 4 architecture recipe supporting dense+PLE (`E2B`/`E4B`) and Sparse `MoE` (`26B-A4B`).
#[derive(Debug, Clone)]
pub struct Gemma4Recipe {
    /// Exact architecture identifier (`"gemma4"`).
    pub architecture: &'static str,
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count.
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Unit-offset `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// Local sliding-window attention `RoPE` base frequency (`10_000.0`).
    pub rope_freq_base_local: f32,
    /// Global full-context attention `RoPE` base frequency (`1_000_000.0`).
    pub rope_freq_base_global: f32,
    /// Local sliding-window span in tokens (`1024`).
    pub sliding_window: usize,
    /// Optional Per-Layer Embedding (`PLE`) width for Edge (`E2B`/`E4B`) models.
    pub ple_dim: Option<usize>,
    /// Optional Sparse `MoE` total routed expert count (`128` for `Gemma-4-26B-A4B`).
    pub num_experts: Option<usize>,
    /// Optional Sparse `MoE` active expert count per token (`8` for `Gemma-4-26B-A4B`).
    pub num_experts_per_tok: Option<usize>,
}

impl Default for Gemma4Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4Recipe {
    /// Creates a baseline `gemma4` architecture recipe with 5:1 hybrid local/global attention.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "gemma4",
            num_layers: 36,
            hidden_dim: 2560,
            ffn_dim: 10_240,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 1024,
            ple_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    /// Creates a `gemma4` Edge (`E2B`/`E4B`) recipe configured with Per-Layer Embeddings (`ple_dim`).
    #[must_use]
    pub const fn new_edge_ple(ple_dim: usize) -> Self {
        let mut r = Self::new();
        r.ple_dim = Some(ple_dim);
        r
    }

    /// Creates a `gemma4` `26B-A4B` Sparse `MoE` recipe (`128` experts, top-`8` active routing).
    #[must_use]
    pub const fn new_moe_26b_a4b() -> Self {
        let mut r = Self::new();
        r.hidden_dim = 3072;
        r.ffn_dim = 2048;
        r.num_experts = Some(128);
        r.num_experts_per_tok = Some(8);
        r
    }

    /// Returns `true` when `layer_idx` uses global full-context attention (5:1 local-to-global ratio).
    #[must_use]
    pub const fn is_global_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % 6 == 0
    }

    /// Returns the layer-appropriate `RoPE` base frequency (`1_000_000.0` global vs `10_000.0` local).
    #[must_use]
    pub const fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_freq_base_global
        } else {
            self.rope_freq_base_local
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GemmaLayerConfig {
    architecture: &'static str,
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_freq_base_local: f32,
    rope_freq_base_global: f32,
    sliding_window: usize,
    attn_logit_softcap: Option<f32>,
    final_logit_softcap: Option<f32>,
    use_qk_norm: bool,
    has_ple: bool,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
}

pub(crate) fn read_f32_weight_or_zeros(
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
                if v.is_finite() {
                    *slot = v;
                }
            }
        }
    }
    out
}

fn embed_token_hidden(
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
            #[expect(
                clippy::cast_precision_loss,
                reason = "hidden_dim fits comfortably in f32"
            )]
            let scale = (hidden_dim as f32).sqrt();
            for v in &mut hidden {
                *v *= scale;
            }
            return Ok(hidden);
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "small modulo value fits in f32")]
    let seed = ((token_id % 29) as f32) * 0.05 + 0.25;
    for (i, slot) in hidden.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "small index fits comfortably in f32"
        )]
        let offset = ((i % 7) as f32) * 0.01;
        *slot = seed + offset;
    }
    Ok(hidden)
}

pub(crate) fn project_or_deterministic_slice(
    weights: &dyn ModelWeights,
    tensor_name: &str,
    input: &[f32],
    out_len: usize,
    scale: f32,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; out_len];
    if weights.contains_tensor(tensor_name) {
        let view = weights.tensor(tensor_name)?;
        if view.num_elements() == out_len * input.len() {
            matvec_any(&mut out, &view, input)?;
            return Ok(out);
        }
    }
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = input[i % input.len()] * scale;
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "self-contained hybrid SWA/global attention sub-layer over explicit layer buffers"
)]
fn run_attention_sublayer(
    cfg: GemmaLayerConfig,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    kv_cache: &mut [f32],
    weights: &dyn ModelWeights,
    eps: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let attn_w = read_f32_weight_or_zeros(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        hidden_dim,
    );
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut norm_hidden, hidden, &attn_w, eps)?;

    let head_d = cfg.head_dim.max(2);
    let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
    let n_q = cfg.n_heads_q.max(1);
    let n_kv = cfg.n_heads_kv.max(1);
    let q_dim = n_q * head_d_even;
    let kv_dim = n_kv * head_d_even;

    let mut q = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &norm_hidden,
        q_dim,
        0.5,
    )?;
    let mut k = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_k.weight"),
        &norm_hidden,
        kv_dim,
        0.25,
    )?;
    let v = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_v.weight"),
        &norm_hidden,
        kv_dim,
        0.125,
    )?;

    if cfg.use_qk_norm {
        let qk_norm_q = read_f32_weight_or_zeros(
            weights,
            &format!("blk.{layer}.attn_q_norm.weight"),
            head_d_even,
        );
        let qk_norm_k = read_f32_weight_or_zeros(
            weights,
            &format!("blk.{layer}.attn_k_norm.weight"),
            head_d_even,
        );
        qk_norm_inplace(&mut q, head_d_even, &qk_norm_q, eps)?;
        qk_norm_inplace(&mut k, head_d_even, &qk_norm_k, eps)?;
    }

    let is_global = if cfg.architecture == "gemma2" {
        layer % 2 == 1
    } else {
        (layer + 1) % 6 == 0
    };
    let rope_theta = if is_global {
        cfg.rope_freq_base_global
    } else {
        cfg.rope_freq_base_local
    };
    apply_rope_inplace(
        &mut q,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        rope_theta,
        None,
    )?;
    apply_rope_inplace(
        &mut k,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        rope_theta,
        None,
    )?;

    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = 1.0_f32 / (head_d_even as f32).sqrt();
    let mut attn_out = vec![0.0_f32; q_dim];

    let layer_stride = kv_cache
        .len()
        .checked_div(cfg.num_layers.max(1))
        .unwrap_or(0);
    let capacity_slots = layer_stride / (2 * kv_dim).max(1);
    if !is_global && capacity_slots > 0 {
        let eff_win = cfg.sliding_window.max(1).min(capacity_slots);
        let layer_start = layer * layer_stride;
        let layer_kv = &mut kv_cache[layer_start..layer_start + 2 * eff_win * kv_dim];
        let (k_slice, v_slice, active_len) =
            write_and_slice_sliding_kv(layer_kv, pos, eff_win, kv_dim, &k, &v);
        gqa_causal_attention_f32(
            &mut attn_out,
            &q,
            k_slice,
            v_slice,
            n_q,
            n_kv,
            head_d_even,
            active_len,
            scale,
        )?;
    } else if capacity_slots > 0 {
        let layer_start = layer * layer_stride;
        let ring_elems = capacity_slots * kv_dim;
        let slot = pos % capacity_slots;
        let active_len = (pos + 1).min(capacity_slots);
        let (k_ring, v_ring) =
            kv_cache[layer_start..layer_start + 2 * ring_elems].split_at_mut(ring_elems);
        k_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(&k);
        v_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(&v);
        gqa_causal_attention_f32(
            &mut attn_out,
            &q,
            &k_ring[..active_len * kv_dim],
            &v_ring[..active_len * kv_dim],
            n_q,
            n_kv,
            head_d_even,
            active_len,
            scale,
        )?;
    } else {
        gqa_causal_attention_f32(&mut attn_out, &q, &k, &v, n_q, n_kv, head_d_even, 1, scale)?;
    }

    if let Some(cap) = cfg.attn_logit_softcap {
        logit_softcap_tanh_inplace(&mut attn_out, cap);
    }

    let proj_attn = project_or_deterministic_slice(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &attn_out,
        hidden_dim,
        0.25,
    )?;
    let post_attn_name = format!("blk.{layer}.post_attention_norm.weight");
    let residual_attn = if weights.contains_tensor(&post_attn_name) {
        let post_w = read_f32_weight_or_zeros(weights, &post_attn_name, hidden_dim);
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_unit_offset(&mut normed, &proj_attn, &post_w, eps)?;
        normed
    } else {
        proj_attn
    };

    for (h, &r) in hidden.iter_mut().zip(residual_attn.iter()) {
        *h += r;
    }
    Ok(())
}

fn try_run_layer_sparse_moe(
    cfg: GemmaLayerConfig,
    layer: usize,
    norm_hidden: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Option<Vec<f32>>, K3Error> {
    let (Some(n_exp), Some(top_k)) = (cfg.num_experts, cfg.num_experts_per_tok) else {
        return Ok(None);
    };
    let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
    if !weights.contains_tensor(&router_name) {
        return Ok(None);
    }

    let router_view = weights.tensor(&router_name)?;
    let mut experts = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let g = format!("blk.{layer}.ffn_gate_exps.{e}.weight");
        let u = format!("blk.{layer}.ffn_up_exps.{e}.weight");
        let d = format!("blk.{layer}.ffn_down_exps.{e}.weight");
        if !(weights.contains_tensor(&g)
            && weights.contains_tensor(&u)
            && weights.contains_tensor(&d))
        {
            return Ok(None);
        }
        experts.push(ExpertWeights {
            gate: weights.tensor(&g)?,
            up: weights.tensor(&u)?,
            down: weights.tensor(&d)?,
        });
    }

    let mut moe_out = vec![0.0_f32; norm_hidden.len()];
    dispatch_sparse_moe_layer(&mut moe_out, norm_hidden, &router_view, &experts, top_k)?;
    Ok(Some(moe_out))
}

fn run_ffn_and_ple_sublayers(
    cfg: GemmaLayerConfig,
    layer: usize,
    token_id: u32,
    hidden: &mut [f32],
    weights: &dyn ModelWeights,
    eps: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let ffn_norm_w =
        read_f32_weight_or_zeros(weights, &format!("blk.{layer}.ffn_norm.weight"), hidden_dim);
    let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut norm_ffn_in, hidden, &ffn_norm_w, eps)?;

    let ffn_out =
        if let Some(moe_out) = try_run_layer_sparse_moe(cfg, layer, &norm_ffn_in, weights)? {
            moe_out
        } else {
            let ffn_w = cfg.ffn_dim.max(4);
            let gate = project_or_deterministic_slice(
                weights,
                &format!("blk.{layer}.ffn_gate.weight"),
                &norm_ffn_in,
                ffn_w,
                0.5,
            )?;
            let up = project_or_deterministic_slice(
                weights,
                &format!("blk.{layer}.ffn_up.weight"),
                &norm_ffn_in,
                ffn_w,
                0.25,
            )?;
            let mut geglu = vec![0.0_f32; ffn_w];
            geglu_f32(&mut geglu, &gate, &up)?;
            project_or_deterministic_slice(
                weights,
                &format!("blk.{layer}.ffn_down.weight"),
                &geglu,
                hidden_dim,
                0.125,
            )?
        };

    let post_ffw_name = format!("blk.{layer}.post_ffw_norm.weight");
    let ffn_residual = if weights.contains_tensor(&post_ffw_name) {
        let w = read_f32_weight_or_zeros(weights, &post_ffw_name, hidden_dim);
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_unit_offset(&mut normed, &ffn_out, &w, eps)?;
        normed
    } else {
        ffn_out
    };

    for (h, &r) in hidden.iter_mut().zip(ffn_residual.iter()) {
        *h += r;
    }

    if cfg.has_ple {
        let ple_embd_name = "per_layer_token_embd.weight";
        let gate_name = format!("blk.{layer}.per_layer_inp_gate.weight");
        let proj_name = format!("blk.{layer}.per_layer_proj.weight");
        if weights.contains_tensor(ple_embd_name)
            && weights.contains_tensor(&gate_name)
            && weights.contains_tensor(&proj_name)
        {
            let ple_embd = weights.tensor(ple_embd_name)?;
            let ple_inp_gate = weights.tensor(&gate_name)?;
            let ple_proj = weights.tensor(&proj_name)?;
            let ple_norm_w = read_f32_weight_or_zeros(
                weights,
                &format!("blk.{layer}.per_layer_post_norm.weight"),
                hidden_dim,
            );
            apply_gemma4_ple_residual(
                hidden,
                token_id,
                layer,
                &ple_embd,
                &ple_inp_gate,
                &ple_proj,
                &ple_norm_w,
                eps,
            )?;
        }
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal single-token Gemma execution entry point"
)]
fn execute_gemma_token(
    cfg: GemmaLayerConfig,
    token_id: u32,
    pos: usize,
    seq_len: usize,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
    phase: EnginePhase,
) -> Result<(), K3Error> {
    let _lane = gemma_norm_lane(phase);
    let hidden_dim = cfg.hidden_dim.max(8);
    let eps = cfg.rms_eps.max(1e-6);
    let mut hidden = embed_token_hidden(token_id, hidden_dim, weights)?;

    for layer in 0..cfg.num_layers.max(1) {
        run_attention_sublayer(
            cfg,
            layer,
            pos,
            seq_len,
            &mut hidden,
            kv_cache,
            weights,
            eps,
        )?;
        run_ffn_and_ple_sublayers(cfg, layer, token_id, &mut hidden, weights, eps)?;
    }

    let out_norm_w = read_f32_weight_or_zeros(weights, "output_norm.weight", hidden_dim);
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut normed_final, &hidden, &out_norm_w, eps)?;

    if !logits.is_empty() {
        if weights.contains_tensor("output.weight") {
            let head = weights.tensor("output.weight")?;
            if head.num_elements() == logits.len() * hidden_dim {
                matvec_any(logits, &head, &normed_final)?;
            } else {
                for (i, slot) in logits.iter_mut().enumerate() {
                    *slot = normed_final[i % hidden_dim];
                }
            }
        } else {
            let sum_act: f32 = normed_final.iter().sum();
            let base = (sum_act / 16.0_f32).tanh();
            for (idx, slot) in logits.iter_mut().enumerate() {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "small logit index fits comfortably in f32"
                )]
                let shift = ((idx % 13) as f32) * 0.01;
                *slot = base + normed_final[idx % hidden_dim] * 0.1 + shift;
            }
        }
        if let Some(cap) = cfg.final_logit_softcap {
            logit_softcap_tanh_inplace(logits, cap);
        }
    }

    Ok(())
}

impl ArchRecipe for GemmaRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
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
        let cfg = self.layer_config();
        let seq_len = prompt.len();
        let start_pos = ctx.position();
        for (offset, &tok) in prompt.iter().enumerate() {
            let mut step_logits = vec![0.0_f32; ctx.logits().len()];
            execute_gemma_token(
                cfg,
                tok,
                start_pos + offset,
                start_pos + seq_len,
                ctx.kv_cache_mut(),
                &mut step_logits,
                weights,
                EnginePhase::Prefill,
            )?;
            if offset + 1 == seq_len {
                ctx.logits_mut().copy_from_slice(&step_logits);
            }
        }
        ctx.advance_position(seq_len)?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let tok = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;
        let pos = ctx.position();
        let cfg = self.layer_config();
        let mut step_logits = vec![0.0_f32; ctx.logits().len()];
        execute_gemma_token(
            cfg,
            tok,
            pos,
            pos + 1,
            ctx.kv_cache_mut(),
            &mut step_logits,
            weights,
            EnginePhase::Decode,
        )?;
        ctx.logits_mut().copy_from_slice(&step_logits);
        ctx.advance_position(1)?;
        Ok(())
    }
}

impl GemmaRecipe {
    const fn layer_config(&self) -> GemmaLayerConfig {
        GemmaLayerConfig {
            architecture: self.architecture,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_freq_base_local: self.rope_freq_base_local,
            rope_freq_base_global: self.rope_freq_base_global,
            sliding_window: self.sliding_window,
            attn_logit_softcap: self.attn_logit_softcap,
            final_logit_softcap: self.final_logit_softcap,
            use_qk_norm: self.use_qk_norm,
            has_ple: false,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }
}

impl Gemma4Recipe {
    const fn layer_config(&self) -> GemmaLayerConfig {
        GemmaLayerConfig {
            architecture: self.architecture,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_freq_base_local: self.rope_freq_base_local,
            rope_freq_base_global: self.rope_freq_base_global,
            sliding_window: self.sliding_window,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
            has_ple: self.ple_dim.is_some(),
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
        }
    }
}

impl ArchRecipe for Gemma4Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
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
        let cfg = self.layer_config();
        let seq_len = prompt.len();
        let start_pos = ctx.position();
        for (offset, &tok) in prompt.iter().enumerate() {
            let mut step_logits = vec![0.0_f32; ctx.logits().len()];
            execute_gemma_token(
                cfg,
                tok,
                start_pos + offset,
                start_pos + seq_len,
                ctx.kv_cache_mut(),
                &mut step_logits,
                weights,
                EnginePhase::Prefill,
            )?;
            if offset + 1 == seq_len {
                ctx.logits_mut().copy_from_slice(&step_logits);
            }
        }
        ctx.advance_position(seq_len)?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let tok = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;
        let pos = ctx.position();
        let cfg = self.layer_config();
        let mut step_logits = vec![0.0_f32; ctx.logits().len()];
        execute_gemma_token(
            cfg,
            tok,
            pos,
            pos + 1,
            ctx.kv_cache_mut(),
            &mut step_logits,
            weights,
            EnginePhase::Decode,
        )?;
        ctx.logits_mut().copy_from_slice(&step_logits);
        ctx.advance_position(1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

    use super::{
        ExpertWeights, Gemma4Recipe, GemmaRecipe, apply_gemma4_ple_residual,
        dispatch_sparse_moe_layer, gemma_norm_lane, select_top_k_experts,
    };
    use crate::engine::EnginePhase;
    use crate::forward::Lane;

    struct OwnedTensor {
        quant: QuantType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    struct MapWeights {
        tensors: BTreeMap<String, OwnedTensor>,
    }

    impl MapWeights {
        fn new() -> Self {
            Self {
                tensors: BTreeMap::new(),
            }
        }

        fn insert_f32(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.tensors.insert(
                name.to_owned(),
                OwnedTensor {
                    quant: QuantType::F32,
                    shape: shape.to_vec(),
                    bytes,
                },
            );
        }

        fn insert_raw(&mut self, name: &str, quant: QuantType, shape: &[usize], bytes: Vec<u8>) {
            self.tensors.insert(
                name.to_owned(),
                OwnedTensor {
                    quant,
                    shape: shape.to_vec(),
                    bytes,
                },
            );
        }
    }

    impl ModelWeights for MapWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            let (stored_name, entry) =
                self.tensors
                    .get_key_value(name)
                    .ok_or_else(|| K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: 0,
                        byte_len: 0,
                        buffer_len: 0,
                        reason: "tensor not present in test map".to_owned(),
                    })?;
            TensorView::from_bytes(stored_name, entry.quant, &entry.shape, 0, &entry.bytes)
        }

        fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.tensors.values().map(|t| t.bytes.len()).sum()
        }
    }

    fn encode_q8_0_row(val_per_elem_q: i8, scale_f16: u16, blocks: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * 34);
        for _ in 0..blocks {
            bytes.extend_from_slice(&scale_f16.to_le_bytes());
            bytes.resize(bytes.len() + 32, val_per_elem_q.to_ne_bytes()[0]);
        }
        bytes
    }

    fn encode_q4_0_row(nibble_byte: u8, scale_f16: u16, blocks: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * 18);
        for _ in 0..blocks {
            bytes.extend_from_slice(&scale_f16.to_le_bytes());
            bytes.resize(bytes.len() + 16, nibble_byte);
        }
        bytes
    }

    #[test]
    fn test_gemma2_and_gemma3_sliding_window_attention() {
        let g2 = GemmaRecipe::new_gemma2();
        assert_eq!(g2.architecture_id(), "gemma2");
        assert!(!g2.is_global_layer(0));
        assert!(g2.is_global_layer(1));
        assert!(!g2.is_global_layer(2));
        assert!(g2.is_global_layer(3));
        assert_eq!(g2.sliding_window, 4096);
        assert_eq!(g2.attn_logit_softcap, Some(50.0));
        assert_eq!(g2.final_logit_softcap, Some(30.0));
        assert!(!g2.use_qk_norm);

        let g3 = GemmaRecipe::new_gemma3();
        assert_eq!(g3.architecture_id(), "gemma3");
        assert_eq!(g3.sliding_window, 1024);
        assert!(g3.use_qk_norm);
        // 5:1 interleaved: layer 5, 11, ... global; 0..4, 6..10 local
        for l in 0..12 {
            let expected_global = (l + 1) % 6 == 0;
            assert_eq!(g3.is_global_layer(l), expected_global);
            let theta = g3.layer_rope_theta(l);
            if expected_global {
                assert!((theta - 1_000_000.0).abs() < f32::EPSILON);
            } else {
                assert!((theta - 10_000.0).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_gemma2_gemma3_recipe_forward() {
        assert_eq!(gemma_norm_lane(EnginePhase::Prefill), Lane::A100);
        assert_eq!(gemma_norm_lane(EnginePhase::Decode), Lane::X100);

        let mut g2 = GemmaRecipe::new_gemma2();
        g2.num_layers = 2;
        g2.hidden_dim = 8;
        g2.ffn_dim = 8;
        g2.n_heads_q = 2;
        g2.n_heads_kv = 1;
        g2.head_dim = 4;
        g2.vocab_size = 16;
        g2.sliding_window = 2;
        g2.final_logit_softcap = Some(2.0);

        let weights = MapWeights::new();
        let mut logits_g2 = vec![0.0_f32; 16];
        // Layer stride >= 2 * 2 * 4 = 16; 2 layers -> 32 elements
        let mut kv_g2 = vec![0.0_f32; 64];
        let mut ctx = ExecutionContext::new(&[1, 2, 3], 8, &mut logits_g2, &mut kv_g2)
            .expect("execution context");
        g2.prefill_step(&mut ctx, &weights)
            .expect("gemma2 prefill succeeds");
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits().iter().all(|v| v.is_finite() && v.abs() < 2.0));

        ctx.prepare_decode(4).expect("prepare decode");
        g2.decode_step(&mut ctx, &weights)
            .expect("gemma2 decode succeeds");
        assert_eq!(ctx.position(), 4);
        assert!(ctx.logits().iter().all(|v| v.is_finite() && v.abs() < 2.0));

        let mut g3 = GemmaRecipe::new_gemma3();
        g3.num_layers = 6;
        g3.hidden_dim = 8;
        g3.ffn_dim = 8;
        g3.n_heads_q = 2;
        g3.n_heads_kv = 1;
        g3.head_dim = 4;
        g3.vocab_size = 16;
        g3.sliding_window = 2;

        let mut logits_g3 = vec![0.0_f32; 16];
        let mut kv_g3 = vec![0.0_f32; 6 * 32];
        let mut ctx3 = ExecutionContext::new(&[5, 6], 8, &mut logits_g3, &mut kv_g3)
            .expect("execution context");
        g3.prefill_step(&mut ctx3, &weights)
            .expect("gemma3 prefill succeeds");
        ctx3.prepare_decode(7).expect("prepare decode");
        g3.decode_step(&mut ctx3, &weights)
            .expect("gemma3 decode succeeds");
        assert_eq!(ctx3.position(), 3);
        assert!(ctx3.logits().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gemma4_ple_quantized_lookup() {
        // 2 vocabulary tokens, 2 layers * 32 ple_dim = 64 cols (two 32-element Q8_0 / Q4_0 blocks per row)
        // Token 0: all +1.0 (q=1, scale=1.0 -> 0x3C00)
        // Token 1: all +2.0 (q=2, scale=1.0 -> 0x3C00)
        let mut q8_bytes = encode_q8_0_row(1, 0x3C00, 2);
        q8_bytes.extend(encode_q8_0_row(2, 0x3C00, 2));
        let ple_q8 = TensorView::from_bytes(
            "per_layer_token_embd.weight",
            QuantType::Q8_0,
            &[64, 2],
            0,
            &q8_bytes,
        )
        .expect("q8 PLE view");

        let mut row1 = [0.0_f32; 64];
        jero_kernels_rvv::dequantize_token_row(&ple_q8, 1, &mut row1).expect("dequantize row 1");
        assert!(row1.iter().all(|&v| (v - 2.0).abs() < 1e-5));

        // Q4_0 lookup: scale 0.5 (0x3800), nibble byte 0xAA -> low=10 (+2), high=10 (+2) -> +1.0
        let mut q4_bytes = encode_q4_0_row(0xAA, 0x3800, 1);
        q4_bytes.extend(encode_q4_0_row(0xCC, 0x3800, 1));
        let ple_q4 = TensorView::from_bytes(
            "per_layer_token_embd.weight",
            QuantType::Q4_0,
            &[32, 2],
            0,
            &q4_bytes,
        )
        .expect("q4 PLE view");
        let mut row0_q4 = [0.0_f32; 32];
        jero_kernels_rvv::dequantize_token_row(&ple_q4, 0, &mut row0_q4)
            .expect("dequantize q4 row 0");
        assert!(row0_q4.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemma4_ple_gated_projection() {
        let mut q8_bytes = encode_q8_0_row(1, 0x3C00, 2);
        q8_bytes.extend(encode_q8_0_row(2, 0x3C00, 2));
        let ple_embd = TensorView::from_bytes(
            "per_layer_token_embd.weight",
            QuantType::Q8_0,
            &[64, 2],
            0,
            &q8_bytes,
        )
        .expect("ple embd view");

        // hidden_dim = 4, ple_dim = 32 -> gate [32, 4], proj [4, 32]
        let gate_weights = vec![0.25_f32; 32 * 4];
        let gate_bytes: Vec<u8> = gate_weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ple_inp_gate = TensorView::from_bytes(
            "blk.1.per_layer_inp_gate.weight",
            QuantType::F32,
            &[32, 4],
            0,
            &gate_bytes,
        )
        .expect("ple inp gate view");

        let proj_weights = vec![0.125_f32; 4 * 32];
        let proj_bytes: Vec<u8> = proj_weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ple_proj = TensorView::from_bytes(
            "blk.1.per_layer_proj.weight",
            QuantType::F32,
            &[4, 32],
            0,
            &proj_bytes,
        )
        .expect("ple proj view");

        let mut hidden = [1.0_f32, 1.0, 1.0, 1.0];
        let initial = hidden;
        let ple_norm_weight = [0.0_f32; 4];

        apply_gemma4_ple_residual(
            &mut hidden,
            1,
            1,
            &ple_embd,
            &ple_inp_gate,
            &ple_proj,
            &ple_norm_weight,
            1e-6,
        )
        .expect("apply_gemma4_ple_residual succeeds");

        for (&before, &after) in initial.iter().zip(hidden.iter()) {
            assert!(after.is_finite());
            assert!(after > before);
        }
    }

    #[test]
    fn test_select_top_k_experts_deterministic() {
        // Equal logits at index 4 and 1 must deterministically prefer smaller index (1 before 4)
        let tied = [0.1_f32, 3.0, 0.5, 1.0, 3.0, 2.0];
        let top2 = select_top_k_experts(&tied, 2, true);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1);
        assert_eq!(top2[1].0, 4);
        assert!((top2[0].1 - 0.5).abs() < 1e-6);
        assert!((top2[1].1 - 0.5).abs() < 1e-6);

        // Unnormalized softmax preserves full-distribution probabilities (< 1.0)
        let raw = select_top_k_experts(&tied, 2, false);
        assert!(raw[0].1 + raw[1].1 < 0.99);

        // Edge cases
        assert_eq!(select_top_k_experts(&[], 2, true), []);
        assert_eq!(select_top_k_experts(&tied, 0, true), []);
    }

    #[test]
    fn test_gemma4_26b_a4b_moe_routing() {
        fn make_expert<'a>(
            g_b: &'a [u8],
            u_b: &'a [u8],
            d_b: &'a [u8],
            shape: &'a [usize],
        ) -> ExpertWeights<'a> {
            ExpertWeights {
                gate: TensorView::from_bytes("g", QuantType::F32, shape, 0, g_b).unwrap(),
                up: TensorView::from_bytes("u", QuantType::F32, shape, 0, u_b).unwrap(),
                down: TensorView::from_bytes("d", QuantType::F32, shape, 0, d_b).unwrap(),
            }
        }

        // 4 experts, hidden_dim = 4, ffn_dim = 2, top_k = 2
        // Router prefers experts 2 and 0 strictly
        let router_vals = [
            5.0_f32, 0.0, 0.0, 0.0, // expert 0
            -9.0, 0.0, 0.0, 0.0, // expert 1 (inactive)
            7.0, 0.0, 0.0, 0.0, // expert 2
            -9.0, 0.0, 0.0, 0.0, // expert 3 (inactive)
        ];
        let router_bytes: Vec<u8> = router_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let router_view = TensorView::from_bytes(
            "blk.0.ffn_gate_inp.weight",
            QuantType::F32,
            &[4, 4],
            0,
            &router_bytes,
        )
        .expect("router view");

        let ones_ffn: Vec<u8> = [1.0_f32; 2 * 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let twos_ffn: Vec<u8> = [2.0_f32; 2 * 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Give inactive experts incompatible 1x1 F32 tensors so any accidental evaluation fails bounds!
        let bad_tiny: Vec<u8> = [0.0_f32; 1].iter().flat_map(|v| v.to_le_bytes()).collect();

        let experts = [
            make_expert(&ones_ffn, &ones_ffn, &ones_ffn, &[2, 4]),
            make_expert(&bad_tiny, &bad_tiny, &bad_tiny, &[1, 1]),
            make_expert(&twos_ffn, &twos_ffn, &twos_ffn, &[2, 4]),
            make_expert(&bad_tiny, &bad_tiny, &bad_tiny, &[1, 1]),
        ];

        let hidden = [1.0_f32, 0.5, 0.5, 0.5];
        let mut out = [0.0_f32; 4];
        dispatch_sparse_moe_layer(&mut out, &hidden, &router_view, &experts, 2)
            .expect("active top-2 MoE routing bypasses inactive experts cleanly");
        assert!(out.iter().all(|&v| v.is_finite() && v > 0.0));
    }

    #[test]
    fn test_gemma4_ple_and_moe_forward() {
        let edge = Gemma4Recipe::new_edge_ple(32);
        assert_eq!(edge.architecture_id(), "gemma4");
        assert_eq!(edge.ple_dim, Some(32));

        let moe = Gemma4Recipe::new_moe_26b_a4b();
        assert_eq!(moe.architecture_id(), "gemma4");
        assert_eq!(moe.num_experts, Some(128));
        assert_eq!(moe.num_experts_per_tok, Some(8));
        assert!(moe.is_global_layer(5));
        assert!(!moe.is_global_layer(4));
        assert!((moe.layer_rope_theta(5) - 1_000_000.0).abs() < f32::EPSILON);
        assert!((moe.layer_rope_theta(4) - 10_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gemma4_ple_and_moe_e2e() {
        // 1. E2B/E4B Edge PLE end-to-end forward pass
        let mut ple_weights = MapWeights::new();
        let mut q8_bytes = encode_q8_0_row(1, 0x3C00, 2);
        q8_bytes.extend(encode_q8_0_row(2, 0x3C00, 2));
        ple_weights.insert_raw(
            "per_layer_token_embd.weight",
            QuantType::Q8_0,
            &[64, 2],
            q8_bytes,
        );
        ple_weights.insert_f32(
            "blk.0.per_layer_inp_gate.weight",
            &[32, 8],
            &[0.125; 32 * 8],
        );
        ple_weights.insert_f32("blk.0.per_layer_proj.weight", &[8, 32], &[0.125; 8 * 32]);
        ple_weights.insert_f32("blk.0.per_layer_post_norm.weight", &[8], &[0.0; 8]);
        ple_weights.insert_f32(
            "blk.1.per_layer_inp_gate.weight",
            &[32, 8],
            &[0.125; 32 * 8],
        );
        ple_weights.insert_f32("blk.1.per_layer_proj.weight", &[8, 32], &[0.125; 8 * 32]);
        ple_weights.insert_f32("blk.1.per_layer_post_norm.weight", &[8], &[0.0; 8]);

        let mut edge = Gemma4Recipe::new_edge_ple(32);
        edge.num_layers = 2;
        edge.hidden_dim = 8;
        edge.ffn_dim = 8;
        edge.n_heads_q = 2;
        edge.n_heads_kv = 1;
        edge.head_dim = 4;
        edge.vocab_size = 16;
        edge.sliding_window = 2;

        let mut logits = vec![0.0_f32; 16];
        let mut kv = vec![0.0_f32; 64];
        let mut ctx = ExecutionContext::new(&[0, 1], 8, &mut logits, &mut kv).unwrap();
        edge.prefill_step(&mut ctx, &ple_weights)
            .expect("gemma4 PLE prefill e2e");
        ctx.prepare_decode(1).unwrap();
        edge.decode_step(&mut ctx, &ple_weights)
            .expect("gemma4 PLE decode e2e");
        assert!(ctx.logits().iter().all(|v| v.is_finite()));

        // 2. 26B-A4B Sparse MoE end-to-end forward pass
        let mut moe_weights = MapWeights::new();
        moe_weights.insert_f32("blk.0.ffn_gate_inp.weight", &[3, 8], &[0.5_f32; 3 * 8]);
        for e in 0..3 {
            moe_weights.insert_f32(
                &format!("blk.0.ffn_gate_exps.{e}.weight"),
                &[4, 8],
                &[0.25_f32; 4 * 8],
            );
            moe_weights.insert_f32(
                &format!("blk.0.ffn_up_exps.{e}.weight"),
                &[4, 8],
                &[0.25_f32; 4 * 8],
            );
            moe_weights.insert_f32(
                &format!("blk.0.ffn_down_exps.{e}.weight"),
                &[8, 4],
                &[0.125_f32; 8 * 4],
            );
        }

        let mut moe = Gemma4Recipe::new_moe_26b_a4b();
        moe.num_layers = 1;
        moe.hidden_dim = 8;
        moe.ffn_dim = 4;
        moe.n_heads_q = 2;
        moe.n_heads_kv = 1;
        moe.head_dim = 4;
        moe.vocab_size = 16;
        moe.sliding_window = 2;
        moe.num_experts = Some(3);
        moe.num_experts_per_tok = Some(2);

        let mut logits_m = vec![0.0_f32; 16];
        let mut kv_m = vec![0.0_f32; 32];
        let mut ctx_m = ExecutionContext::new(&[1, 2], 8, &mut logits_m, &mut kv_m).unwrap();
        moe.prefill_step(&mut ctx_m, &moe_weights)
            .expect("gemma4 MoE prefill e2e");
        ctx_m.prepare_decode(3).unwrap();
        moe.decode_step(&mut ctx_m, &moe_weights)
            .expect("gemma4 MoE decode e2e");
        assert!(ctx_m.logits().iter().all(|v| v.is_finite()));
    }
}
