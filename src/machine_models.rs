//! Non-LLM machine-model architecture recipes (`TTM`, `FT-Transformer`, `TabPFN-v2`),
//! sequential offline-RL policy execution (`DecisionTransformer`), and the multimodal soft-prompt
//! bridge (`MmprojLatentBridge`) linking machine latents to target LLM prefill Layer 0.
//!
//! All execution recipes enforce strict fail-closed validation on GGUF tensor existence, exact
//! tensor geometry, supported quantization formats, bounded context budgets, and upfront element
//! finiteness (`NaN` / `±Inf` rejection). Missing weights or shape mismatches always return a
//! structured [`K3Error`] rather than synthesizing dummy or identity tensors.

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType, TensorView,
};
#[cfg(any(
    test,
    feature = "jero-arch-ttm",
    feature = "jero-arch-ft-transformer",
    feature = "jero-arch-tabpfn-v2",
    feature = "jero-arch-decision-transformer",
))]
use jero_core::{GgufFile, GgufValue};
use jero_kernels_rvv::{dequantize_token_row, gelu_inplace_f32, layernorm_f32};

#[cfg(any(feature = "jero-arch-ft-transformer", feature = "jero-arch-tabpfn-v2"))]
use jero_kernels_rvv::tokenize_continuous_features_f32;
#[cfg(feature = "jero-arch-ttm")]
use jero_kernels_rvv::{
    AxisMlpWeights, ttm_time_channel_mix_scratch_len, ttm_time_channel_mix_with_scratch_f32,
};
#[cfg(feature = "jero-arch-tabpfn-v2")]
use jero_kernels_rvv::{TabPfnAttentionAxis, TabPfnAttentionShape, tabpfn_dual_axis_attention_f32};

use crate::gemma::matvec_any;
#[cfg(any(
    feature = "jero-arch-ttm",
    feature = "jero-arch-ft-transformer",
    feature = "jero-arch-tabpfn-v2",
    feature = "jero-arch-decision-transformer",
))]
use crate::qwen_deepseek::is_supported_quant;
#[cfg(feature = "jero-arch-decision-transformer")]
use crate::qwen_deepseek::{
    DenseAttnLayerSpec, rmsnorm_standard, run_dense_attention_sublayer_scaled, run_dense_swiglu_ffn,
};

fn bounds_err(tensor_name: &str, expected: usize, actual: usize, reason: String) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset: 0,
        byte_len: expected.saturating_mul(size_of::<f32>()),
        buffer_len: actual.saturating_mul(size_of::<f32>()),
        reason,
    }
}

fn mismatch_err(arch: &str, reason: impl Into<String>) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: reason.into(),
    }
}

fn checked_mul(name: &str, a: usize, b: usize) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| {
        bounds_err(
            name,
            usize::MAX,
            0,
            format!("{name} dimensions ({a} * {b}) overflowed usize"),
        )
    })
}

fn validate_finite_slice(name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: idx.saturating_mul(size_of::<f32>()),
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(data),
                reason: format!("non-finite f32 ({val}) in '{name}' at index {idx}"),
            });
        }
    }
    Ok(())
}

fn dequantize_all_finite(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>, K3Error> {
    let total = view.num_elements();
    if total == 0 {
        return Err(bounds_err(
            name,
            1,
            0,
            format!("tensor '{name}' cannot be empty"),
        ));
    }
    let mut out = vec![0.0_f32; total];
    if view.quant() == QuantType::F32 {
        let raw = view.as_bytes();
        let req_bytes = total.saturating_mul(size_of::<f32>());
        if raw.len() < req_bytes {
            return Err(bounds_err(
                name,
                total,
                raw.len() / size_of::<f32>(),
                format!("F32 tensor '{name}' byte slice truncated"),
            ));
        }
        for (dst, chunk) in out.iter_mut().zip(raw.chunks_exact(size_of::<f32>())) {
            *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    } else {
        let row_len = match view.shape() {
            [cols, ..] if *cols > 0 => *cols,
            _ => total,
        };
        let row_count = total.checked_div(row_len).unwrap_or(1).max(1);
        for row_idx in 0..row_count {
            let start = row_idx * row_len;
            dequantize_token_row(view, row_idx, &mut out[start..start + row_len])?;
        }
    }
    validate_finite_slice(name, &out)?;
    Ok(out)
}

fn require_tensor_1d(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = weights.tensor(name)?;
    if view.num_elements() != expected_len {
        return Err(bounds_err(
            name,
            expected_len,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected_len} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    dequantize_all_finite(&view, name)
}

#[cfg(feature = "jero-arch-ttm")]
fn require_tensor_2d(
    weights: &dyn ModelWeights,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let expected = checked_mul(name, in_dim, out_dim)?;
    let view = weights.tensor(name)?;
    let shape = view.shape();
    if shape.len() == 2 && !(shape[0] == in_dim && shape[1] == out_dim) {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!("2D tensor '{name}' expected shape [{in_dim}, {out_dim}], got {shape:?}"),
        ));
    }
    if view.num_elements() != expected {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    dequantize_all_finite(&view, name)
}

fn require_matvec_view<'w>(
    weights: &'w dyn ModelWeights,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let expected = checked_mul(name, in_dim, out_dim)?;
    let view = weights.tensor(name)?;
    let shape = view.shape();
    if shape.len() == 2 && !(shape[0] == in_dim && shape[1] == out_dim) {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!("2D tensor '{name}' expected shape [{in_dim}, {out_dim}], got {shape:?}"),
        ));
    }
    if view.num_elements() != expected {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    if view.quant() == QuantType::F32 {
        let _ = dequantize_all_finite(&view, name)?;
    }
    Ok(view)
}

#[cfg(feature = "jero-arch-ft-transformer")]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn project_mat_view(
    view: &TensorView<'_>,
    input: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; out_dim];
    matvec_any(&mut out, view, input)?;
    if let Some(b) = bias {
        if b.len() != out_dim {
            return Err(bounds_err(
                view.name(),
                out_dim,
                b.len(),
                format!(
                    "bias length {} does not match projection out_dim {out_dim}",
                    b.len()
                ),
            ));
        }
        for (dst, &bv) in out.iter_mut().zip(b.iter()) {
            *dst += bv;
        }
    }
    validate_finite_slice(view.name(), &out)?;
    Ok(out)
}

#[cfg(feature = "jero-arch-ft-transformer")]
fn softmax_inplace(scores: &mut [f32]) -> Result<(), K3Error> {
    if scores.is_empty() {
        return Ok(());
    }
    validate_finite_slice("attention_scores", scores)?;
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let exp = (*s - max_val).exp();
        *s = exp;
        sum += exp;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(bounds_err(
            "attention_scores",
            scores.len(),
            scores.len(),
            "softmax normalizer became non-finite or non-positive".to_owned(),
        ));
    }
    let inv = 1.0_f32 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
    Ok(())
}

#[cfg(feature = "jero-arch-ft-transformer")]
fn run_bidirectional_mha(
    q_seq: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let d_model = checked_mul("mha_d_model", n_heads, head_dim)?;
    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; seq_len * d_model];
    let mut scores = vec![0.0_f32; seq_len];

    for q_pos in 0..seq_len {
        let q_row = &q_seq[q_pos * d_model..(q_pos + 1) * d_model];
        let out_row = &mut out[q_pos * d_model..(q_pos + 1) * d_model];
        for h in 0..n_heads {
            let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let k_head =
                    &k_seq[k_pos * d_model + h * head_dim..k_pos * d_model + (h + 1) * head_dim];
                scores[k_pos] = dot_slice(q_head, k_head) * scale;
            }
            softmax_inplace(&mut scores)?;
            let out_head = &mut out_row[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let weight = scores[k_pos];
                let v_head =
                    &v_seq[k_pos * d_model + h * head_dim..k_pos * d_model + (h + 1) * head_dim];
                for (dst, &vv) in out_head.iter_mut().zip(v_head.iter()) {
                    *dst += weight * vv;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(any(
    feature = "jero-arch-ttm",
    feature = "jero-arch-ft-transformer",
    feature = "jero-arch-decision-transformer",
))]
fn lookup_optional_bias(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    if weights.contains_tensor(name) {
        Ok(Some(require_tensor_1d(weights, name, expected_len)?))
    } else {
        Ok(None)
    }
}

/// Target-compatible multimodal soft-prompt projector (`MmprojLatentBridge`) connecting a
/// machine-model latent vector to target LLM prefill embeddings in Layer 0.
#[derive(Debug, Clone, PartialEq)]
pub struct MmprojLatentBridge {
    /// Expected target LLM GGUF `general.architecture` identifier (e.g., `"qwen2"`, `"llama"`).
    pub target_architecture_id: String,
    /// Expected target LLM recipe revision string (e.g., `"1.0.0"`).
    pub target_recipe_version: String,
    /// Machine-model latent feature dimension expected on input.
    pub machine_latent_dim: usize,
    /// Target LLM embedding length (`n_embd`).
    pub target_embedding_length: usize,
    /// Number of Layer-0 soft-prompt tokens emitted per machine latent vector.
    pub num_soft_tokens: usize,
    /// Optional `LayerNorm` epsilon applied to projected soft-prompt vectors.
    pub eps: f32,
}

impl MmprojLatentBridge {
    /// Creates a new [`MmprojLatentBridge`] targeting `(target_architecture_id, target_recipe_version, target_embedding_length)`.
    #[must_use]
    pub fn new(
        target_architecture_id: impl Into<String>,
        target_recipe_version: impl Into<String>,
        machine_latent_dim: usize,
        target_embedding_length: usize,
        num_soft_tokens: usize,
    ) -> Self {
        Self {
            target_architecture_id: target_architecture_id.into(),
            target_recipe_version: target_recipe_version.into(),
            machine_latent_dim,
            target_embedding_length,
            num_soft_tokens,
            eps: 1e-5,
        }
    }

    /// Validates static bridge geometry (`machine_latent_dim > 0`, `target_embedding_length > 0`,
    /// `num_soft_tokens > 0`, finite positive `eps`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any bridge dimension or parameter is invalid.
    pub fn validate_geometry(&self) -> Result<(), K3Error> {
        if self.machine_latent_dim == 0
            || self.target_embedding_length == 0
            || self.num_soft_tokens == 0
        {
            return Err(mismatch_err(
                &self.target_architecture_id,
                format!(
                    "mmproj dimensions must all be non-zero (latent={}, embed={}, tokens={})",
                    self.machine_latent_dim, self.target_embedding_length, self.num_soft_tokens
                ),
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(mismatch_err(
                &self.target_architecture_id,
                format!("mmproj eps ({}) must be finite and > 0", self.eps),
            ));
        }
        let _ = checked_mul(
            "mmproj_output",
            self.num_soft_tokens,
            self.target_embedding_length,
        )?;
        Ok(())
    }

    /// Verifies that the target LLM `(architecture_id, recipe_version, embedding_length)` tuple
    /// matches this projector bridge before execution.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on any mismatch across `architecture_id`,
    /// `recipe_version`, or `embedding_length`.
    pub fn verify_target_compatibility(
        &self,
        target_arch: &str,
        target_version: &str,
        target_embedding_len: usize,
    ) -> Result<(), K3Error> {
        self.validate_geometry()?;
        if self.target_architecture_id != target_arch {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target architecture mismatch: bridge expects '{}', target LLM is '{}'",
                    self.target_architecture_id, target_arch
                ),
            ));
        }
        if self.target_recipe_version != target_version {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target recipe version mismatch for '{target_arch}': bridge expects '{}', target LLM is '{}'",
                    self.target_recipe_version, target_version
                ),
            ));
        }
        if self.target_embedding_length != target_embedding_len {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target embedding_length mismatch for '{target_arch}': bridge expects {}, target LLM has {target_embedding_len}",
                    self.target_embedding_length
                ),
            ));
        }
        Ok(())
    }

    /// Verifies compatibility against a concrete [`ArchRecipe`] and its [`ModelMetadata`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any tuple field diverges.
    pub fn verify_target_recipe(
        &self,
        target_recipe: &dyn ArchRecipe,
        metadata: &ModelMetadata,
    ) -> Result<(), K3Error> {
        self.verify_target_compatibility(
            target_recipe.architecture_id(),
            target_recipe.recipe_version(),
            metadata.embedding_dim(),
        )
    }
}

/// Projects a single machine-model latent vector (`[bridge.machine_latent_dim]`) into `K` Layer-0
/// LLM soft-prompt tokens (`[bridge.num_soft_tokens * bridge.target_embedding_length]`) using
/// resident `mmproj.*` GGUF projection weights (`mmproj.fc1.weight` + `mmproj.fc2.weight`, or
/// single linear `mmproj.weight`).
///
/// Fails closed with [`K3Error::TensorBoundsError`] when required projector weights are absent,
/// mismatched in geometry, or produce non-finite outputs.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if `bridge` parameters are invalid, or
/// [`K3Error::TensorBoundsError`] on slice/weight shape mismatch or non-finite values.
pub fn project_machine_latent_to_soft_prompt(
    bridge: &MmprojLatentBridge,
    latent: &[f32],
    weights: &dyn ModelWeights,
    out_embeds: &mut [f32],
) -> Result<(), K3Error> {
    bridge.validate_geometry()?;
    if latent.len() != bridge.machine_latent_dim {
        return Err(bounds_err(
            "machine_latent",
            bridge.machine_latent_dim,
            latent.len(),
            format!(
                "expected machine latent of length {}, got {}",
                bridge.machine_latent_dim,
                latent.len()
            ),
        ));
    }
    validate_finite_slice("machine_latent", latent)?;

    let total_out = checked_mul(
        "mmproj_output",
        bridge.num_soft_tokens,
        bridge.target_embedding_length,
    )?;
    if out_embeds.len() != total_out {
        return Err(bounds_err(
            "soft_prompt_embeds",
            total_out,
            out_embeds.len(),
            format!(
                "expected out_embeds length {total_out} ({}x{}), got {}",
                bridge.num_soft_tokens,
                bridge.target_embedding_length,
                out_embeds.len()
            ),
        ));
    }

    let projected = project_mmproj_weights(bridge, latent, weights, total_out)?;
    out_embeds.copy_from_slice(&projected);
    apply_mmproj_norm(bridge, weights, out_embeds)?;
    validate_finite_slice("soft_prompt_embeds", out_embeds)?;
    Ok(())
}

fn project_mmproj_weights(
    bridge: &MmprojLatentBridge,
    latent: &[f32],
    weights: &dyn ModelWeights,
    total_out: usize,
) -> Result<Vec<f32>, K3Error> {
    if weights.contains_tensor("mmproj.fc1.weight") {
        let fc1_view = weights.tensor("mmproj.fc1.weight")?;
        let shape = fc1_view.shape();
        if shape.len() != 2 || shape[0] != bridge.machine_latent_dim || shape[1] == 0 {
            return Err(bounds_err(
                "mmproj.fc1.weight",
                bridge.machine_latent_dim,
                fc1_view.num_elements(),
                format!(
                    "mmproj.fc1.weight expected shape [{}, hidden_dim], got {shape:?}",
                    bridge.machine_latent_dim
                ),
            ));
        }
        if fc1_view.quant() == QuantType::F32 {
            let _ = dequantize_all_finite(&fc1_view, "mmproj.fc1.weight")?;
        }
        let hidden_dim = shape[1];
        let fc1_bias = if weights.contains_tensor("mmproj.fc1.bias") {
            Some(require_tensor_1d(weights, "mmproj.fc1.bias", hidden_dim)?)
        } else {
            None
        };
        let mut mid = project_mat_view(&fc1_view, latent, hidden_dim, fc1_bias.as_deref())?;
        gelu_inplace_f32(&mut mid)?;
        let fc2_view = require_matvec_view(weights, "mmproj.fc2.weight", hidden_dim, total_out)?;
        let fc2_bias = if weights.contains_tensor("mmproj.fc2.bias") {
            Some(require_tensor_1d(weights, "mmproj.fc2.bias", total_out)?)
        } else {
            None
        };
        project_mat_view(&fc2_view, &mid, total_out, fc2_bias.as_deref())
    } else if weights.contains_tensor("mmproj.weight") {
        let w_view = require_matvec_view(
            weights,
            "mmproj.weight",
            bridge.machine_latent_dim,
            total_out,
        )?;
        let bias = if weights.contains_tensor("mmproj.bias") {
            Some(require_tensor_1d(weights, "mmproj.bias", total_out)?)
        } else {
            None
        };
        project_mat_view(&w_view, latent, total_out, bias.as_deref())
    } else {
        Err(weights.tensor("mmproj.fc1.weight").unwrap_err())
    }
}

fn apply_mmproj_norm(
    bridge: &MmprojLatentBridge,
    weights: &dyn ModelWeights,
    out_embeds: &mut [f32],
) -> Result<(), K3Error> {
    if !weights.contains_tensor("mmproj.norm.weight") {
        return Ok(());
    }
    let norm_w = require_tensor_1d(
        weights,
        "mmproj.norm.weight",
        bridge.target_embedding_length,
    )?;
    let norm_b = if weights.contains_tensor("mmproj.norm.bias") {
        Some(require_tensor_1d(
            weights,
            "mmproj.norm.bias",
            bridge.target_embedding_length,
        )?)
    } else {
        None
    };
    let mut normed_tok = vec![0.0_f32; bridge.target_embedding_length];
    for chunk in out_embeds.chunks_exact_mut(bridge.target_embedding_length) {
        layernorm_f32(
            &mut normed_tok,
            chunk,
            Some(&norm_w),
            norm_b.as_deref(),
            bridge.eps,
        )?;
        chunk.copy_from_slice(&normed_tok);
    }
    Ok(())
}

/// Executes an LLM prefill forward pass (`recipe.prefill_step`) with Layer-0 continuous
/// `soft_prompt` embeddings prepended ahead of discrete `prompt_tokens`, returning the advanced
/// KV sequence position (`ctx.position()`).
///
/// # Errors
/// Returns [`K3Error`] if context bounds are exceeded, soft-prompt geometry fails validation, or
/// recipe execution fails.
#[expect(
    clippy::too_many_arguments,
    reason = "bridges raw caller prompt/soft-prompt slices into ExecutionContext without mutating jero-core"
)]
pub fn forward_with_soft_prompt(
    recipe: &dyn ArchRecipe,
    prompt_tokens: &[u32],
    n_ctx: usize,
    soft_prompt: &[f32],
    num_soft_tokens: usize,
    embed_dim: usize,
    logits: &mut [f32],
    kv_cache: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<usize, K3Error> {
    let total_tokens =
        num_soft_tokens
            .checked_add(prompt_tokens.len())
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: n_ctx,
            })?;
    if total_tokens > n_ctx {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: total_tokens,
            max_context_tokens: n_ctx,
        });
    }
    let mut ctx = ExecutionContext::new(prompt_tokens, n_ctx, logits, kv_cache)?
        .with_soft_prompt_embeds(soft_prompt, num_soft_tokens, embed_dim)?;
    recipe.prefill_step(&mut ctx, weights)?;
    Ok(ctx.position())
}

// ============================================================================
// 1. TinyTimeMixers (TTM) Recipe — Multivariate Zero-Shot Time-Series Forecast
// ============================================================================

/// Zero-KV `TinyTimeMixers` (`TTM`) patch-time and channel-mixing time-series forecasting recipe.
#[cfg(feature = "jero-arch-ttm")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TtmRecipe {
    /// Input lookback series length (`L`, e.g., 64 or 512).
    pub context_length: usize,
    /// Patch window length (`P`, dividing `context_length`).
    pub patch_length: usize,
    /// Forecast horizon length (`F`, e.g., 16 or 96).
    pub forecast_length: usize,
    /// Number of multivariate telemetry channels (`C`).
    pub num_channels: usize,
    /// Number of stacked time+channel MLP mixer layers.
    pub num_mixer_layers: usize,
    /// Hidden width of the time/patch-axis MLP.
    pub time_hidden_dim: usize,
    /// Hidden width of the channel-axis MLP.
    pub channel_hidden_dim: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

#[cfg(feature = "jero-arch-ttm")]
impl Default for TtmRecipe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "jero-arch-ttm")]
impl TtmRecipe {
    /// Creates a baseline `ttm` forecasting recipe sized conservatively (`n_ctx = 64`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            context_length: 64,
            patch_length: 8,
            forecast_length: 16,
            num_channels: 4,
            num_mixer_layers: 2,
            time_hidden_dim: 16,
            channel_hidden_dim: 8,
            norm_eps: 1e-5,
        }
    }

    /// Returns the number of temporal patches (`context_length / patch_length`).
    #[must_use]
    pub const fn num_patches(&self) -> usize {
        match self.context_length.checked_div(self.patch_length) {
            Some(v) => v,
            None => 0,
        }
    }

    /// Hydrates a [`TtmRecipe`] from [`ModelMetadata`] and GGUF tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.context_length() > 0 {
            recipe.context_length = metadata.context_length();
        }
        if metadata.layer_count() > 0 {
            recipe.num_mixer_layers = metadata.layer_count();
        }
        if let Some(desc) = metadata.find_tensor("blk.0.channel_norm.weight")
            && desc.num_elements() > 0
        {
            recipe.num_channels = desc.num_elements();
        }
        if let Some(desc) = metadata.find_tensor("blk.0.time_norm.weight")
            && desc.num_elements() > 0
        {
            let patches = desc.num_elements();
            if recipe.context_length % patches == 0 {
                recipe.patch_length = (recipe.context_length / patches).max(1);
            } else {
                recipe.patch_length = 1;
                recipe.context_length = patches;
            }
        }
        if let Some(desc) = metadata.find_tensor("blk.0.time_mlp.fc1.bias")
            && desc.num_elements() > 0
        {
            recipe.time_hidden_dim = desc.num_elements();
        }
        if let Some(desc) = metadata.find_tensor("blk.0.channel_mlp.fc1.bias")
            && desc.num_elements() > 0
        {
            recipe.channel_hidden_dim = desc.num_elements();
        }
        if let Some(desc) = metadata.find_tensor("ttm.forecast_head.weight")
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            recipe.forecast_length = desc.shape()[1];
        }
        recipe
    }

    /// Hydrates a [`TtmRecipe`] from a parsed [`GgufFile`], reading optional `ttm.*` GGUF keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        if let Some(pl) = gguf
            .metadata_value("ttm.patch_length")
            .and_then(GgufValue::as_usize)
            && pl > 0
        {
            recipe.patch_length = pl;
        }
        if let Some(fl) = gguf
            .metadata_value("ttm.forecast_length")
            .and_then(GgufValue::as_usize)
            && fl > 0
        {
            recipe.forecast_length = fl;
        }
        if let Some(nc) = gguf
            .metadata_value("ttm.num_channels")
            .and_then(GgufValue::as_usize)
            && nc > 0
        {
            recipe.num_channels = nc;
        }
        if let Some(th) = gguf
            .metadata_value("ttm.time_hidden_dim")
            .and_then(GgufValue::as_usize)
            && th > 0
        {
            recipe.time_hidden_dim = th;
        }
        if let Some(ch) = gguf
            .metadata_value("ttm.channel_hidden_dim")
            .and_then(GgufValue::as_usize)
            && ch > 0
        {
            recipe.channel_hidden_dim = ch;
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        if self.context_length == 0
            || self.patch_length == 0
            || self.context_length % self.patch_length != 0
            || self.forecast_length == 0
            || self.num_channels == 0
            || self.num_mixer_layers == 0
            || self.time_hidden_dim == 0
            || self.channel_hidden_dim == 0
        {
            return Err(mismatch_err(
                "ttm",
                format!(
                    "invalid TTM geometry (ctx={}, patch={}, forecast={}, channels={}, layers={})",
                    self.context_length,
                    self.patch_length,
                    self.forecast_length,
                    self.num_channels,
                    self.num_mixer_layers
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "ttm",
                format!("ttm norm_eps ({}) must be finite and > 0", self.norm_eps),
            ));
        }
        Ok(self.context_length / self.patch_length)
    }

    fn prepare_patches(
        &self,
        features: &[f32],
        num_rows: usize,
        num_cols: usize,
        n_patches: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        if num_cols != self.num_channels {
            return Err(bounds_err(
                "feature_input",
                self.num_channels,
                num_cols,
                format!(
                    "ttm expected {} channels (cols), got {num_cols}",
                    self.num_channels
                ),
            ));
        }
        if num_rows == n_patches
            && (self.patch_length == 1 || !weights.contains_tensor("ttm.patch_proj.weight"))
        {
            return Ok(features.to_vec());
        }
        if num_rows != self.context_length {
            return Err(bounds_err(
                "feature_input",
                self.context_length * self.num_channels,
                features.len(),
                format!(
                    "ttm expected feature rows == context_length ({}) or n_patches ({n_patches}), got {num_rows}",
                    self.context_length
                ),
            ));
        }
        let proj_w = require_tensor_1d(weights, "ttm.patch_proj.weight", self.patch_length)?;
        let proj_bias = lookup_optional_bias(weights, "ttm.patch_proj.bias", 1)?;
        let bias = proj_bias.map_or(0.0_f32, |b| b[0]);
        let mut patched = vec![0.0_f32; n_patches * self.num_channels];
        for p in 0..n_patches {
            let t_base = p * self.patch_length;
            for c in 0..self.num_channels {
                let mut acc = bias;
                for (k, &w) in proj_w.iter().enumerate() {
                    acc += w * features[(t_base + k) * self.num_channels + c];
                }
                patched[p * self.num_channels + c] = acc;
            }
        }
        validate_finite_slice("ttm_patched", &patched)?;
        Ok(patched)
    }
}

#[cfg(feature = "jero-arch-ttm")]
struct TtmLayerTensors {
    time_norm_w: Vec<f32>,
    time_norm_b: Vec<f32>,
    time_fc1_w: Vec<f32>,
    time_fc1_b: Vec<f32>,
    time_fc2_w: Vec<f32>,
    time_fc2_b: Vec<f32>,
    chan_norm_w: Vec<f32>,
    chan_norm_b: Vec<f32>,
    chan_fc1_w: Vec<f32>,
    chan_fc1_b: Vec<f32>,
    chan_fc2_w: Vec<f32>,
    chan_fc2_b: Vec<f32>,
}

#[cfg(feature = "jero-arch-ttm")]
impl TtmLayerTensors {
    fn load(
        weights: &dyn ModelWeights,
        layer: usize,
        n_patches: usize,
        n_channels: usize,
        time_hidden: usize,
        chan_hidden: usize,
    ) -> Result<Self, K3Error> {
        Ok(Self {
            time_norm_w: require_tensor_1d(
                weights,
                &format!("blk.{layer}.time_norm.weight"),
                n_patches,
            )?,
            time_norm_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.time_norm.bias"),
                n_patches,
            )?,
            time_fc1_w: require_tensor_2d(
                weights,
                &format!("blk.{layer}.time_mlp.fc1.weight"),
                n_patches,
                time_hidden,
            )?,
            time_fc1_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.time_mlp.fc1.bias"),
                time_hidden,
            )?,
            time_fc2_w: require_tensor_2d(
                weights,
                &format!("blk.{layer}.time_mlp.fc2.weight"),
                time_hidden,
                n_patches,
            )?,
            time_fc2_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.time_mlp.fc2.bias"),
                n_patches,
            )?,
            chan_norm_w: require_tensor_1d(
                weights,
                &format!("blk.{layer}.channel_norm.weight"),
                n_channels,
            )?,
            chan_norm_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.channel_norm.bias"),
                n_channels,
            )?,
            chan_fc1_w: require_tensor_2d(
                weights,
                &format!("blk.{layer}.channel_mlp.fc1.weight"),
                n_channels,
                chan_hidden,
            )?,
            chan_fc1_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.channel_mlp.fc1.bias"),
                chan_hidden,
            )?,
            chan_fc2_w: require_tensor_2d(
                weights,
                &format!("blk.{layer}.channel_mlp.fc2.weight"),
                chan_hidden,
                n_channels,
            )?,
            chan_fc2_b: require_tensor_1d(
                weights,
                &format!("blk.{layer}.channel_mlp.fc2.bias"),
                n_channels,
            )?,
        })
    }

    fn as_axis_weights(
        &self,
        n_patches: usize,
        n_channels: usize,
        time_hidden: usize,
        chan_hidden: usize,
    ) -> (AxisMlpWeights<'_>, AxisMlpWeights<'_>) {
        (
            AxisMlpWeights {
                norm_weight: &self.time_norm_w,
                norm_bias: &self.time_norm_b,
                fc1_weight: &self.time_fc1_w,
                fc1_bias: &self.time_fc1_b,
                fc2_weight: &self.time_fc2_w,
                fc2_bias: &self.time_fc2_b,
                dim: n_patches,
                hidden_dim: time_hidden,
            },
            AxisMlpWeights {
                norm_weight: &self.chan_norm_w,
                norm_bias: &self.chan_norm_b,
                fc1_weight: &self.chan_fc1_w,
                fc1_bias: &self.chan_fc1_b,
                fc2_weight: &self.chan_fc2_w,
                fc2_bias: &self.chan_fc2_b,
                dim: n_channels,
                hidden_dim: chan_hidden,
            },
        )
    }
}

#[cfg(feature = "jero-arch-ttm")]
impl ArchRecipe for TtmRecipe {
    fn architecture_id(&self) -> &'static str {
        "ttm"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let n_patches = self.validate_geometry()?;
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.context_length * self.num_channels,
                0,
                "ttm requires ExecutionContext::feature_input continuous telemetry matrix"
                    .to_owned(),
            )
        })?;
        let (num_rows, num_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "ttm requires ExecutionContext::feature_dims".to_owned(),
            )
        })?;
        validate_finite_slice("feature_input", features)?;

        let out_elems = checked_mul("ttm_forecast", self.forecast_length, self.num_channels)?;
        if ctx.logits().len() < out_elems {
            return Err(bounds_err(
                "logits",
                out_elems,
                ctx.logits().len(),
                format!(
                    "ttm forecast output buffer too small: need {out_elems} ({}x{}), got {}",
                    self.forecast_length,
                    self.num_channels,
                    ctx.logits().len()
                ),
            ));
        }

        let mut state = self.prepare_patches(features, num_rows, num_cols, n_patches, weights)?;
        let req_scratch = ttm_time_channel_mix_scratch_len(
            n_patches,
            self.num_channels,
            self.time_hidden_dim,
            self.channel_hidden_dim,
        )
        .ok_or_else(|| {
            bounds_err(
                "ttm_scratch",
                0,
                0,
                "ttm scratch dimension calculation overflowed usize".to_owned(),
            )
        })?;
        let mut scratch = vec![0.0_f32; req_scratch];
        let mut next_state = vec![0.0_f32; state.len()];

        for layer in 0..self.num_mixer_layers {
            let tensors = TtmLayerTensors::load(
                weights,
                layer,
                n_patches,
                self.num_channels,
                self.time_hidden_dim,
                self.channel_hidden_dim,
            )?;
            let (time_w, chan_w) = tensors.as_axis_weights(
                n_patches,
                self.num_channels,
                self.time_hidden_dim,
                self.channel_hidden_dim,
            );
            ttm_time_channel_mix_with_scratch_f32(
                &mut next_state,
                &state,
                n_patches,
                self.num_channels,
                &time_w,
                &chan_w,
                self.norm_eps,
                &mut scratch,
            )?;
            state.copy_from_slice(&next_state);
        }

        let forecast = self.project_forecast_head(weights, &state, n_patches, out_elems)?;
        validate_finite_slice("ttm_forecast", &forecast)?;
        ctx.logits_mut()[..out_elems].copy_from_slice(&forecast);
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
        Err(mismatch_err(
            "ttm",
            "TinyTimeMixers is a zero-KV single-pass forecast model and does not support autoregressive decode_step",
        ))
    }
}

#[cfg(feature = "jero-arch-ttm")]
impl TtmRecipe {
    fn project_forecast_head(
        &self,
        weights: &dyn ModelWeights,
        state: &[f32],
        n_patches: usize,
        out_elems: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let head_view = require_matvec_view(
            weights,
            "ttm.forecast_head.weight",
            n_patches,
            self.forecast_length,
        )?;
        let head_bias =
            lookup_optional_bias(weights, "ttm.forecast_head.bias", self.forecast_length)?;
        let mut chan_col = vec![0.0_f32; n_patches];
        let mut forecast = vec![0.0_f32; out_elems];
        for c in 0..self.num_channels {
            for p in 0..n_patches {
                chan_col[p] = state[p * self.num_channels + c];
            }
            let chan_out = project_mat_view(
                &head_view,
                &chan_col,
                self.forecast_length,
                head_bias.as_deref(),
            )?;
            for (h, &val) in chan_out.iter().enumerate() {
                forecast[h * self.num_channels + c] = val;
            }
        }
        Ok(forecast)
    }
}

// ============================================================================
// 2. FT-Transformer Recipe — Continuous + Categorical Tabular Tokenizer
// ============================================================================

/// Zero-KV `FT-Transformer` (Feature Tokenizer + bidirectional Transformer) tabular recipe.
#[cfg(feature = "jero-arch-ft-transformer")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FtTransformerRecipe {
    /// Number of continuous numerical input features.
    pub num_continuous: usize,
    /// Number of categorical input feature columns.
    pub num_categorical: usize,
    /// Per-feature token embedding width (`d_token`).
    pub d_token: usize,
    /// Number of bidirectional Transformer encoder layers.
    pub num_layers: usize,
    /// Number of attention heads dividing `d_token`.
    pub n_heads: usize,
    /// Feed-forward intermediate width per encoder block.
    pub ffn_dim: usize,
    /// Output scoring/logit dimension (`1` for regression/binary score or `num_classes`).
    pub out_dim: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

#[cfg(feature = "jero-arch-ft-transformer")]
impl Default for FtTransformerRecipe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "jero-arch-ft-transformer")]
impl FtTransformerRecipe {
    /// Creates a baseline `ft-transformer` recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_continuous: 4,
            num_categorical: 0,
            d_token: 16,
            num_layers: 2,
            n_heads: 2,
            ffn_dim: 32,
            out_dim: 2,
            norm_eps: 1e-5,
        }
    }

    /// Hydrates an [`FtTransformerRecipe`] from [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.embedding_dim() > 0 {
            recipe.d_token = metadata.embedding_dim();
        }
        if metadata.layer_count() > 0 {
            recipe.num_layers = metadata.layer_count();
        }
        if metadata.head_count() > 0 {
            recipe.n_heads = metadata.head_count().min(recipe.d_token).max(1);
        }
        if let Some(ffn) = metadata.feed_forward_length()
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        } else {
            recipe.ffn_dim = recipe.d_token.saturating_mul(2).max(4);
        }
        if let Some(desc) = metadata.find_tensor("ft.cont_weight")
            && recipe.d_token > 0
            && desc.num_elements() >= recipe.d_token
            && desc.num_elements() % recipe.d_token == 0
        {
            recipe.num_continuous = desc.num_elements() / recipe.d_token;
        }
        if let Some(desc) = metadata.find_tensor("classifier.weight")
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            recipe.out_dim = desc.shape()[1];
        } else if let Some(desc) = metadata.find_tensor("ft.head.weight")
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            recipe.out_dim = desc.shape()[1];
        }
        recipe
    }

    /// Hydrates an [`FtTransformerRecipe`] from a parsed [`GgufFile`], reading optional `ft.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (prefix, field) in [
            ("ft-transformer.num_continuous", &mut recipe.num_continuous),
            (
                "ft-transformer.num_categorical",
                &mut recipe.num_categorical,
            ),
            ("ft-transformer.out_dim", &mut recipe.out_dim),
        ] {
            if let Some(v) = gguf.metadata_value(prefix).and_then(GgufValue::as_usize) {
                *field = v;
            }
        }
        if let Some(v) = gguf
            .metadata_value("ft-transformer.feed_forward_length")
            .and_then(GgufValue::as_usize)
            && v > 0
        {
            recipe.ffn_dim = v;
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        let total_features = self.num_continuous + self.num_categorical;
        if total_features == 0
            || self.d_token == 0
            || self.num_layers == 0
            || self.n_heads == 0
            || self.d_token % self.n_heads != 0
            || self.ffn_dim == 0
            || self.out_dim == 0
        {
            return Err(mismatch_err(
                "ft-transformer",
                format!(
                    "invalid FT-Transformer geometry (cont={}, cat={}, d_token={}, heads={}, ffn={}, out={})",
                    self.num_continuous,
                    self.num_categorical,
                    self.d_token,
                    self.n_heads,
                    self.ffn_dim,
                    self.out_dim
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "ft-transformer",
                format!(
                    "ft-transformer norm_eps ({}) must be finite and > 0",
                    self.norm_eps
                ),
            ));
        }
        Ok(self.d_token / self.n_heads)
    }

    fn encode_row_tokens(
        &self,
        row_features: &[f32],
        prompt_cat_ids: &[u32],
        cls_embd: &[f32],
        cont_wb: Option<(&[f32], &[f32])>,
        cat_embd_and_bias: Option<(&TensorView<'_>, Option<&[f32]>)>,
    ) -> Result<Vec<f32>, K3Error> {
        let seq_len = 1 + self.num_continuous + self.num_categorical;
        let mut seq = vec![0.0_f32; seq_len * self.d_token];
        seq[..self.d_token].copy_from_slice(cls_embd);

        let mut token_offset = self.d_token;
        if self.num_continuous > 0 {
            let (cw, cb) = cont_wb.ok_or_else(|| {
                bounds_err(
                    "ft.cont_weight",
                    1,
                    0,
                    "missing continuous tokenizer weights".to_owned(),
                )
            })?;
            let cont_slice = &row_features[..self.num_continuous];
            let cont_len = self.num_continuous * self.d_token;
            tokenize_continuous_features_f32(
                &mut seq[token_offset..token_offset + cont_len],
                cont_slice,
                cw,
                cb,
                self.num_continuous,
                self.d_token,
            )?;
            token_offset += cont_len;
        }

        if self.num_categorical > 0 {
            let (cat_view, cat_bias) = cat_embd_and_bias.ok_or_else(|| {
                bounds_err(
                    "ft.cat_embd.weight",
                    1,
                    0,
                    "missing categorical embedding table".to_owned(),
                )
            })?;
            let cat_rows = cat_view.num_elements() / self.d_token;
            for cat_idx in 0..self.num_categorical {
                let id_usize = if row_features.len() == self.num_continuous + self.num_categorical {
                    let raw = row_features[self.num_continuous + cat_idx];
                    #[expect(
                        clippy::float_cmp,
                        reason = "categorical ID validated non-negative and exact integer before cast"
                    )]
                    if !raw.is_finite() || raw < 0.0 || raw.trunc() != raw {
                        return Err(bounds_err(
                            "feature_input",
                            cat_idx,
                            row_features.len(),
                            format!(
                                "categorical feature at col {} must be non-negative integer, got {raw}",
                                self.num_continuous + cat_idx
                            ),
                        ));
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "validated exact non-negative finite integer"
                    )]
                    {
                        raw as usize
                    }
                } else {
                    usize::try_from(prompt_cat_ids[cat_idx]).map_err(|_| {
                        bounds_err(
                            "prompt_tokens",
                            cat_idx,
                            prompt_cat_ids.len(),
                            "categorical ID overflow".to_owned(),
                        )
                    })?
                };
                if id_usize >= cat_rows {
                    return Err(bounds_err(
                        "ft.cat_embd.weight",
                        id_usize + 1,
                        cat_rows,
                        format!(
                            "categorical index {id_usize} out of bounds for table rows {cat_rows}"
                        ),
                    ));
                }
                let dst = &mut seq[token_offset + cat_idx * self.d_token
                    ..token_offset + (cat_idx + 1) * self.d_token];
                dequantize_token_row(cat_view, id_usize, dst)?;
                if let Some(bias) = cat_bias {
                    let b_slice = &bias[cat_idx * self.d_token..(cat_idx + 1) * self.d_token];
                    for (d, &bv) in dst.iter_mut().zip(b_slice.iter()) {
                        *d += bv;
                    }
                }
                validate_finite_slice("ft_cat_token", dst)?;
            }
        }
        Ok(seq)
    }

    fn run_encoder_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        seq_len: usize,
        head_dim: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.d_token;
        let attn_norm_w = require_tensor_1d(weights, &format!("blk.{layer}.attn_norm.weight"), d)?;
        let attn_norm_b = require_tensor_1d(weights, &format!("blk.{layer}.attn_norm.bias"), d)?;
        let q_view = require_matvec_view(weights, &format!("blk.{layer}.attn_q.weight"), d, d)?;
        let k_view = require_matvec_view(weights, &format!("blk.{layer}.attn_k.weight"), d, d)?;
        let v_view = require_matvec_view(weights, &format!("blk.{layer}.attn_v.weight"), d, d)?;
        let o_view =
            require_matvec_view(weights, &format!("blk.{layer}.attn_output.weight"), d, d)?;

        let mut q_seq = Vec::with_capacity(seq_len * d);
        let mut k_seq = Vec::with_capacity(seq_len * d);
        let mut v_seq = Vec::with_capacity(seq_len * d);
        let mut norm_buf = vec![0.0_f32; d];
        for tok in hidden.chunks_exact(d) {
            layernorm_f32(
                &mut norm_buf,
                tok,
                Some(&attn_norm_w),
                Some(&attn_norm_b),
                self.norm_eps,
            )?;
            q_seq.extend_from_slice(&project_mat_view(&q_view, &norm_buf, d, None)?);
            k_seq.extend_from_slice(&project_mat_view(&k_view, &norm_buf, d, None)?);
            v_seq.extend_from_slice(&project_mat_view(&v_view, &norm_buf, d, None)?);
        }

        let mha = run_bidirectional_mha(&q_seq, &k_seq, &v_seq, seq_len, self.n_heads, head_dim)?;
        for (t_idx, tok_out) in mha.chunks_exact(d).enumerate() {
            let proj = project_mat_view(&o_view, tok_out, d, None)?;
            for (dst, &r) in hidden[t_idx * d..(t_idx + 1) * d]
                .iter_mut()
                .zip(proj.iter())
            {
                *dst += r;
            }
        }

        let ffn_norm_w = require_tensor_1d(weights, &format!("blk.{layer}.ffn_norm.weight"), d)?;
        let ffn_norm_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_norm.bias"), d)?;
        let fc1_view = require_matvec_view(
            weights,
            &format!("blk.{layer}.ffn_fc1.weight"),
            d,
            self.ffn_dim,
        )?;
        let fc1_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_fc1.bias"), self.ffn_dim)?;
        let fc2_view = require_matvec_view(
            weights,
            &format!("blk.{layer}.ffn_fc2.weight"),
            self.ffn_dim,
            d,
        )?;
        let fc2_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_fc2.bias"), d)?;

        for tok in hidden.chunks_exact_mut(d) {
            layernorm_f32(
                &mut norm_buf,
                tok,
                Some(&ffn_norm_w),
                Some(&ffn_norm_b),
                self.norm_eps,
            )?;
            let mut mid = project_mat_view(&fc1_view, &norm_buf, self.ffn_dim, Some(&fc1_b))?;
            gelu_inplace_f32(&mut mid)?;
            let down = project_mat_view(&fc2_view, &mid, d, Some(&fc2_b))?;
            for (dst, &r) in tok.iter_mut().zip(down.iter()) {
                *dst += r;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "jero-arch-ft-transformer")]
impl ArchRecipe for FtTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        "ft-transformer"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let head_dim = self.validate_geometry()?;
        let (features, num_rows, num_cols, needed_logits) = self.validate_ft_inputs(ctx)?;
        let prompt_cats = ctx.prompt_tokens();

        let cls_embd = require_tensor_1d(weights, "ft.cls_token.weight", self.d_token)?;
        let cont_wb = if self.num_continuous > 0 {
            let need = checked_mul("ft.cont_weight", self.num_continuous, self.d_token)?;
            let w = require_tensor_1d(weights, "ft.cont_weight", need)?;
            let b = require_tensor_1d(weights, "ft.cont_bias", need)?;
            Some((w, b))
        } else {
            None
        };
        let cat_view = if self.num_categorical > 0 {
            let view = weights.tensor("ft.cat_embd.weight")?;
            if view.num_elements() < self.d_token || view.num_elements() % self.d_token != 0 {
                return Err(bounds_err(
                    "ft.cat_embd.weight",
                    self.d_token,
                    view.num_elements(),
                    "ft.cat_embd.weight must hold whole d_token rows".to_owned(),
                ));
            }
            if view.quant() == QuantType::F32 {
                let _ = dequantize_all_finite(&view, "ft.cat_embd.weight")?;
            }
            Some(view)
        } else {
            None
        };
        let cat_bias = if self.num_categorical > 0 {
            lookup_optional_bias(weights, "ft.cat_bias", self.num_categorical * self.d_token)?
        } else {
            None
        };

        let out_norm_w = require_tensor_1d(weights, "output_norm.weight", self.d_token)?;
        let out_norm_b = require_tensor_1d(weights, "output_norm.bias", self.d_token)?;
        let (head_name, bias_name) = if weights.contains_tensor("ft.head.weight") {
            ("ft.head.weight", "ft.head.bias")
        } else {
            ("classifier.weight", "classifier.bias")
        };
        let head_view = require_matvec_view(weights, head_name, self.d_token, self.out_dim)?;
        let head_bias = require_tensor_1d(weights, bias_name, self.out_dim)?;
        let seq_len = 1 + self.num_continuous + self.num_categorical;

        let mut all_logits = Vec::with_capacity(needed_logits);
        for row_idx in 0..num_rows {
            let row = &features[row_idx * num_cols..(row_idx + 1) * num_cols];
            let mut hidden = self.encode_row_tokens(
                row,
                prompt_cats,
                &cls_embd,
                cont_wb.as_ref().map(|(w, b)| (w.as_slice(), b.as_slice())),
                cat_view.as_ref().map(|v| (v, cat_bias.as_deref())),
            )?;
            for layer in 0..self.num_layers {
                self.run_encoder_layer(weights, layer, seq_len, head_dim, &mut hidden)?;
            }
            let mut cls_normed = vec![0.0_f32; self.d_token];
            layernorm_f32(
                &mut cls_normed,
                &hidden[..self.d_token],
                Some(&out_norm_w),
                Some(&out_norm_b),
                self.norm_eps,
            )?;
            let row_logits =
                project_mat_view(&head_view, &cls_normed, self.out_dim, Some(&head_bias))?;
            all_logits.extend_from_slice(&row_logits);
        }

        validate_finite_slice("ft_logits", &all_logits)?;
        ctx.logits_mut()[..needed_logits].copy_from_slice(&all_logits);
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
        Err(mismatch_err(
            "ft-transformer",
            "FT-Transformer is a zero-KV single-pass tabular model and does not support autoregressive decode_step",
        ))
    }
}

#[cfg(feature = "jero-arch-ft-transformer")]
impl FtTransformerRecipe {
    fn validate_ft_inputs<'a>(
        &self,
        ctx: &'a ExecutionContext<'_>,
    ) -> Result<(&'a [f32], usize, usize, usize), K3Error> {
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.num_continuous + self.num_categorical,
                0,
                "ft-transformer requires ExecutionContext::feature_input tabular matrix".to_owned(),
            )
        })?;
        let (num_rows, num_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "ft-transformer requires feature_dims".to_owned(),
            )
        })?;
        validate_finite_slice("feature_input", features)?;

        let prompt_cats = ctx.prompt_tokens();
        let expects_combined = self.num_continuous + self.num_categorical;
        let uses_prompt_cats =
            self.num_categorical > 0 && num_cols == self.num_continuous && num_rows == 1;
        if num_cols != expects_combined && !uses_prompt_cats {
            return Err(bounds_err(
                "feature_input",
                expects_combined,
                num_cols,
                format!(
                    "ft-transformer expected {expects_combined} feature cols (or {} continuous cols with prompt categorical IDs), got {num_cols}",
                    self.num_continuous
                ),
            ));
        }
        if uses_prompt_cats && prompt_cats.len() != self.num_categorical {
            return Err(bounds_err(
                "prompt_tokens",
                self.num_categorical,
                prompt_cats.len(),
                format!(
                    "ft-transformer expected {} categorical IDs in prompt_tokens, got {}",
                    self.num_categorical,
                    prompt_cats.len()
                ),
            ));
        }

        let needed_logits = checked_mul("ft_logits", num_rows, self.out_dim)?;
        if ctx.logits().len() < needed_logits {
            return Err(bounds_err(
                "logits",
                needed_logits,
                ctx.logits().len(),
                format!(
                    "ft-transformer logits slice length {} < required {needed_logits}",
                    ctx.logits().len()
                ),
            ));
        }
        Ok((features, num_rows, num_cols, needed_logits))
    }
}

// ============================================================================
// 3. TabPFN-v2 Recipe — Dual-Axis In-Context Prior-Data Fitted Network
// ============================================================================

#[cfg(feature = "jero-arch-tabpfn-v2")]
type QkvProjection = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Zero-KV `TabPFN-v2` dual-axis (`Feature` + context-restricted `Sample`) tabular classifier.
#[cfg(feature = "jero-arch-tabpfn-v2")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabPfnV2Recipe {
    /// Number of tabular feature columns (`F`).
    pub num_features: usize,
    /// Per-cell hidden representation width (`d_model = n_heads * head_dim`).
    pub hidden_dim: usize,
    /// Number of dual-axis attention + FFN layers.
    pub num_layers: usize,
    /// Number of attention heads dividing `hidden_dim`.
    pub n_heads: usize,
    /// Intermediate FFN width.
    pub ffn_dim: usize,
    /// Number of discrete target classes.
    pub num_classes: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

#[cfg(feature = "jero-arch-tabpfn-v2")]
impl Default for TabPfnV2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "jero-arch-tabpfn-v2")]
impl TabPfnV2Recipe {
    /// Creates a baseline `tabpfn-v2` prior-data in-context classification recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_features: 4,
            hidden_dim: 16,
            num_layers: 2,
            n_heads: 2,
            ffn_dim: 32,
            num_classes: 2,
            norm_eps: 1e-5,
        }
    }

    /// Hydrates a [`TabPfnV2Recipe`] from [`ModelMetadata`] and GGUF tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.embedding_dim() > 0 {
            recipe.hidden_dim = metadata.embedding_dim();
        }
        if metadata.layer_count() > 0 {
            recipe.num_layers = metadata.layer_count();
        }
        if metadata.head_count() > 0 {
            recipe.n_heads = metadata.head_count().min(recipe.hidden_dim).max(1);
        }
        if let Some(ffn) = metadata.feed_forward_length()
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        } else {
            recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2).max(4);
        }
        if let Some(desc) = metadata.find_tensor("tabpfn.feature_enc.weight")
            && recipe.hidden_dim > 0
            && desc.num_elements() >= recipe.hidden_dim
            && desc.num_elements() % recipe.hidden_dim == 0
        {
            recipe.num_features = desc.num_elements() / recipe.hidden_dim;
        }
        if let Some(desc) = metadata.find_tensor("classifier.weight")
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            recipe.num_classes = desc.shape()[1];
        }
        recipe
    }

    /// Hydrates a [`TabPfnV2Recipe`] from a parsed [`GgufFile`], reading optional `tabpfn-v2.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (key, dst) in [
            ("tabpfn-v2.num_features", &mut recipe.num_features),
            ("tabpfn-v2.num_classes", &mut recipe.num_classes),
            ("tabpfn-v2.feed_forward_length", &mut recipe.ffn_dim),
        ] {
            if let Some(v) = gguf.metadata_value(key).and_then(GgufValue::as_usize)
                && v > 0
            {
                *dst = v;
            }
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        if self.num_features == 0
            || self.hidden_dim == 0
            || self.num_layers == 0
            || self.n_heads == 0
            || self.hidden_dim % self.n_heads != 0
            || self.ffn_dim == 0
            || self.num_classes < 2
        {
            return Err(mismatch_err(
                "tabpfn-v2",
                format!(
                    "invalid TabPFN-v2 geometry (features={}, hidden={}, layers={}, heads={}, classes={})",
                    self.num_features,
                    self.hidden_dim,
                    self.num_layers,
                    self.n_heads,
                    self.num_classes
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "tabpfn-v2",
                format!(
                    "tabpfn-v2 norm_eps ({}) must be finite and > 0",
                    self.norm_eps
                ),
            ));
        }
        Ok(self.hidden_dim / self.n_heads)
    }

    fn project_cell_qkv(
        &self,
        hidden: &[f32],
        norm_w: &[f32],
        norm_b: &[f32],
        q_view: &TensorView<'_>,
        k_view: &TensorView<'_>,
        v_view: &TensorView<'_>,
    ) -> Result<QkvProjection, K3Error> {
        let d = self.hidden_dim;
        let total = hidden.len();
        let mut q = Vec::with_capacity(total);
        let mut k = Vec::with_capacity(total);
        let mut v = Vec::with_capacity(total);
        let mut norm_buf = vec![0.0_f32; d];
        for cell in hidden.chunks_exact(d) {
            layernorm_f32(
                &mut norm_buf,
                cell,
                Some(norm_w),
                Some(norm_b),
                self.norm_eps,
            )?;
            q.extend_from_slice(&project_mat_view(q_view, &norm_buf, d, None)?);
            k.extend_from_slice(&project_mat_view(k_view, &norm_buf, d, None)?);
            v.extend_from_slice(&project_mat_view(v_view, &norm_buf, d, None)?);
        }
        Ok((q, k, v))
    }

    fn run_dual_axis_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        shape: TabPfnAttentionShape,
        scale: f32,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.hidden_dim;
        let feat_norm_w = require_tensor_1d(weights, &format!("blk.{layer}.feat_norm.weight"), d)?;
        let feat_norm_b = require_tensor_1d(weights, &format!("blk.{layer}.feat_norm.bias"), d)?;
        let fq = require_matvec_view(weights, &format!("blk.{layer}.feat_q.weight"), d, d)?;
        let fk = require_matvec_view(weights, &format!("blk.{layer}.feat_k.weight"), d, d)?;
        let fv = require_matvec_view(weights, &format!("blk.{layer}.feat_v.weight"), d, d)?;
        let fo = require_matvec_view(weights, &format!("blk.{layer}.feat_out.weight"), d, d)?;

        let (q_f, k_f, v_f) =
            self.project_cell_qkv(hidden, &feat_norm_w, &feat_norm_b, &fq, &fk, &fv)?;
        let mut feat_attn = vec![0.0_f32; hidden.len()];
        tabpfn_dual_axis_attention_f32(
            &mut feat_attn,
            &q_f,
            &k_f,
            &v_f,
            shape,
            TabPfnAttentionAxis::Feature,
            scale,
        )?;
        for (dst_cell, attn_cell) in hidden.chunks_exact_mut(d).zip(feat_attn.chunks_exact(d)) {
            let proj = project_mat_view(&fo, attn_cell, d, None)?;
            for (dst, &r) in dst_cell.iter_mut().zip(proj.iter()) {
                *dst += r;
            }
        }

        let samp_norm_w =
            require_tensor_1d(weights, &format!("blk.{layer}.sample_norm.weight"), d)?;
        let samp_norm_b = require_tensor_1d(weights, &format!("blk.{layer}.sample_norm.bias"), d)?;
        let sq = require_matvec_view(weights, &format!("blk.{layer}.sample_q.weight"), d, d)?;
        let sk = require_matvec_view(weights, &format!("blk.{layer}.sample_k.weight"), d, d)?;
        let sv = require_matvec_view(weights, &format!("blk.{layer}.sample_v.weight"), d, d)?;
        let so = require_matvec_view(weights, &format!("blk.{layer}.sample_out.weight"), d, d)?;

        let (q_s, k_s, v_s) =
            self.project_cell_qkv(hidden, &samp_norm_w, &samp_norm_b, &sq, &sk, &sv)?;
        let mut samp_attn = vec![0.0_f32; hidden.len()];
        tabpfn_dual_axis_attention_f32(
            &mut samp_attn,
            &q_s,
            &k_s,
            &v_s,
            shape,
            TabPfnAttentionAxis::Sample,
            scale,
        )?;
        for (dst_cell, attn_cell) in hidden.chunks_exact_mut(d).zip(samp_attn.chunks_exact(d)) {
            let proj = project_mat_view(&so, attn_cell, d, None)?;
            for (dst, &r) in dst_cell.iter_mut().zip(proj.iter()) {
                *dst += r;
            }
        }

        let ffn_norm_w = require_tensor_1d(weights, &format!("blk.{layer}.ffn_norm.weight"), d)?;
        let ffn_norm_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_norm.bias"), d)?;
        let fc1_view = require_matvec_view(
            weights,
            &format!("blk.{layer}.ffn_fc1.weight"),
            d,
            self.ffn_dim,
        )?;
        let fc1_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_fc1.bias"), self.ffn_dim)?;
        let fc2_view = require_matvec_view(
            weights,
            &format!("blk.{layer}.ffn_fc2.weight"),
            self.ffn_dim,
            d,
        )?;
        let fc2_b = require_tensor_1d(weights, &format!("blk.{layer}.ffn_fc2.bias"), d)?;

        let mut norm_buf = vec![0.0_f32; d];
        for cell in hidden.chunks_exact_mut(d) {
            layernorm_f32(
                &mut norm_buf,
                cell,
                Some(&ffn_norm_w),
                Some(&ffn_norm_b),
                self.norm_eps,
            )?;
            let mut mid = project_mat_view(&fc1_view, &norm_buf, self.ffn_dim, Some(&fc1_b))?;
            gelu_inplace_f32(&mut mid)?;
            let down = project_mat_view(&fc2_view, &mid, d, Some(&fc2_b))?;
            for (dst, &r) in cell.iter_mut().zip(down.iter()) {
                *dst += r;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "jero-arch-tabpfn-v2")]
impl ArchRecipe for TabPfnV2Recipe {
    fn architecture_id(&self) -> &'static str {
        "tabpfn-v2"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let head_dim = self.validate_geometry()?;
        let (features, n_samples, n_ctx, needed_logits) = self.validate_tabpfn_inputs(ctx)?;
        let context_labels = ctx.prompt_tokens();
        let mut hidden =
            self.encode_tabpfn_cells(features, context_labels, weights, n_samples, n_ctx)?;

        let shape = TabPfnAttentionShape::new(n_samples, self.num_features, self.n_heads, head_dim)
            .with_context_samples(n_ctx);
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        for layer in 0..self.num_layers {
            self.run_dual_axis_layer(weights, layer, shape, scale, &mut hidden)?;
        }

        let out_logits =
            self.pool_tabpfn_logits(weights, &hidden, n_samples, n_ctx, needed_logits)?;
        ctx.logits_mut()[..needed_logits].copy_from_slice(&out_logits);
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
        Err(mismatch_err(
            "tabpfn-v2",
            "TabPFN-v2 is a zero-KV in-context tabular model and does not support autoregressive decode_step",
        ))
    }
}

#[cfg(feature = "jero-arch-tabpfn-v2")]
impl TabPfnV2Recipe {
    fn validate_tabpfn_inputs<'a>(
        &self,
        ctx: &'a ExecutionContext<'_>,
    ) -> Result<(&'a [f32], usize, usize, usize), K3Error> {
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.num_features,
                0,
                "tabpfn-v2 requires ExecutionContext::feature_input [n_samples, n_features]"
                    .to_owned(),
            )
        })?;
        let (n_samples, n_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "tabpfn-v2 requires feature_dims".to_owned(),
            )
        })?;
        if n_cols != self.num_features || n_samples == 0 {
            return Err(bounds_err(
                "feature_input",
                self.num_features,
                n_cols,
                format!(
                    "tabpfn-v2 expected feature_dims (_, {}), got ({n_samples}, {n_cols})",
                    self.num_features
                ),
            ));
        }
        validate_finite_slice("feature_input", features)?;

        let n_ctx = ctx.prompt_tokens().len();
        if n_ctx == 0 || n_ctx > n_samples {
            return Err(bounds_err(
                "prompt_tokens",
                n_samples,
                n_ctx,
                format!(
                    "tabpfn-v2 context label count ({n_ctx}) in prompt_tokens must lie within 1..={n_samples}"
                ),
            ));
        }
        let n_query = if n_ctx < n_samples {
            n_samples - n_ctx
        } else {
            n_samples
        };
        let needed_logits = checked_mul("tabpfn_logits", n_query, self.num_classes)?;
        if ctx.logits().len() < needed_logits {
            return Err(bounds_err(
                "logits",
                needed_logits,
                ctx.logits().len(),
                format!(
                    "tabpfn-v2 logits length {} < required {needed_logits} ({n_query} queries x {} classes)",
                    ctx.logits().len(),
                    self.num_classes
                ),
            ));
        }
        Ok((features, n_samples, n_ctx, needed_logits))
    }

    fn encode_tabpfn_cells(
        &self,
        features: &[f32],
        context_labels: &[u32],
        weights: &dyn ModelWeights,
        n_samples: usize,
        n_ctx: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let row_elems = checked_mul(
            "tabpfn.feature_enc.weight",
            self.num_features,
            self.hidden_dim,
        )?;
        let enc_w = require_tensor_1d(weights, "tabpfn.feature_enc.weight", row_elems)?;
        let enc_b = require_tensor_1d(weights, "tabpfn.feature_enc.bias", row_elems)?;
        let y_embd_view = require_matvec_view(
            weights,
            "tabpfn.y_embd.weight",
            self.hidden_dim,
            self.num_classes,
        )?;
        let y_unknown = require_tensor_1d(weights, "tabpfn.y_unknown.weight", self.hidden_dim)?;

        let total_elems = checked_mul("tabpfn_hidden", n_samples, row_elems)?;
        let mut hidden = vec![0.0_f32; total_elems];
        let mut label_vec = vec![0.0_f32; self.hidden_dim];
        for r in 0..n_samples {
            let x_row = &features[r * self.num_features..(r + 1) * self.num_features];
            let dst_row = &mut hidden[r * row_elems..(r + 1) * row_elems];
            tokenize_continuous_features_f32(
                dst_row,
                x_row,
                &enc_w,
                &enc_b,
                self.num_features,
                self.hidden_dim,
            )?;
            let cond = if r < n_ctx {
                let lbl = usize::try_from(context_labels[r]).unwrap_or(usize::MAX);
                if lbl >= self.num_classes {
                    return Err(bounds_err(
                        "prompt_tokens",
                        self.num_classes,
                        lbl,
                        format!(
                            "context label {lbl} at row {r} out of bounds for num_classes={}",
                            self.num_classes
                        ),
                    ));
                }
                dequantize_token_row(&y_embd_view, lbl, &mut label_vec)?;
                validate_finite_slice("tabpfn.y_embd.weight", &label_vec)?;
                &label_vec[..]
            } else {
                &y_unknown[..]
            };
            for cell in dst_row.chunks_exact_mut(self.hidden_dim) {
                for (d_val, &lv) in cell.iter_mut().zip(cond.iter()) {
                    *d_val += lv;
                }
            }
        }
        Ok(hidden)
    }

    fn pool_tabpfn_logits(
        &self,
        weights: &dyn ModelWeights,
        hidden: &[f32],
        n_samples: usize,
        n_ctx: usize,
        needed_logits: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let out_norm_w = require_tensor_1d(weights, "output_norm.weight", self.hidden_dim)?;
        let out_norm_b = require_tensor_1d(weights, "output_norm.bias", self.hidden_dim)?;
        let cls_view = require_matvec_view(
            weights,
            "classifier.weight",
            self.hidden_dim,
            self.num_classes,
        )?;
        let cls_bias = require_tensor_1d(weights, "classifier.bias", self.num_classes)?;
        let row_elems = self.num_features * self.hidden_dim;
        let query_range = if n_ctx < n_samples {
            n_ctx..n_samples
        } else {
            0..n_samples
        };

        #[expect(
            clippy::cast_precision_loss,
            reason = "num_features fits cleanly in f32"
        )]
        let inv_f = 1.0_f32 / (self.num_features as f32);
        let mut pooled = vec![0.0_f32; self.hidden_dim];
        let mut normed = vec![0.0_f32; self.hidden_dim];
        let mut out_logits = Vec::with_capacity(needed_logits);

        for r in query_range {
            pooled.fill(0.0);
            let row = &hidden[r * row_elems..(r + 1) * row_elems];
            for cell in row.chunks_exact(self.hidden_dim) {
                for (acc, &val) in pooled.iter_mut().zip(cell.iter()) {
                    *acc += val * inv_f;
                }
            }
            layernorm_f32(
                &mut normed,
                &pooled,
                Some(&out_norm_w),
                Some(&out_norm_b),
                self.norm_eps,
            )?;
            let logits = project_mat_view(&cls_view, &normed, self.num_classes, Some(&cls_bias))?;
            out_logits.extend_from_slice(&logits);
        }

        validate_finite_slice("tabpfn_logits", &out_logits)?;
        Ok(out_logits)
    }
}

// ============================================================================
// 4. DecisionTransformer Recipe — Causal (RTG, State, Action) Autoregression
// ============================================================================

/// Causal offline-RL `DecisionTransformer` recipe operating over `(return_to_go, state, action)`
/// trajectory triples with bounded KV caching (`n_ctx = 64` compatible).
#[cfg(feature = "jero-arch-decision-transformer")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionTransformerRecipe {
    /// Continuous environment state dimension.
    pub state_dim: usize,
    /// Discrete action space size (`act_dim`).
    pub act_dim: usize,
    /// Causal transformer hidden width (`n_embd`).
    pub hidden_dim: usize,
    /// Number of causal decoder layers.
    pub num_layers: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value attention head count (GQA compatible).
    pub n_heads_kv: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// `SwiGLU` feed-forward width.
    pub ffn_dim: usize,
    /// Maximum episode timestep index (`embed_timestep` table capacity).
    pub max_ep_len: usize,
    /// `RMSNorm` epsilon (`1e-6` default).
    pub rms_eps: f32,
    /// `RoPE` frequency base (`10_000.0` default).
    pub rope_theta: f32,
}

#[cfg(feature = "jero-arch-decision-transformer")]
impl Default for DecisionTransformerRecipe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "jero-arch-decision-transformer")]
impl DecisionTransformerRecipe {
    /// Creates a baseline `decision-transformer` recipe fitting comfortably in `n_ctx = 64`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state_dim: 4,
            act_dim: 4,
            hidden_dim: 16,
            num_layers: 2,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            max_ep_len: 64,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    /// Hydrates a [`DecisionTransformerRecipe`] from [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.layer_count() > 0 {
            recipe.num_layers = metadata.layer_count();
        }
        if metadata.embedding_dim() > 0 {
            recipe.hidden_dim = metadata.embedding_dim();
        }
        if metadata.head_count() > 0 {
            recipe.n_heads_q = metadata.head_count();
        }
        if metadata.head_count_kv() > 0 {
            recipe.n_heads_kv = metadata.head_count_kv().min(recipe.n_heads_q);
        } else {
            recipe.n_heads_kv = recipe.n_heads_q;
        }
        if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads_q).max(2);
        }
        if let Some(ffn) = metadata.feed_forward_length()
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        } else {
            recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2).max(8);
        }
        if let Some(desc) = metadata.find_tensor("dt.embed_state.weight")
            && desc.shape().len() == 2
            && desc.shape()[0] > 0
        {
            recipe.state_dim = desc.shape()[0];
        }
        if let Some(desc) = metadata.find_tensor("dt.predict_action.weight")
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            recipe.act_dim = desc.shape()[1];
        }
        if let Some(desc) = metadata.find_tensor("dt.embed_timestep.weight")
            && recipe.hidden_dim > 0
            && desc.num_elements() >= recipe.hidden_dim
        {
            recipe.max_ep_len = (desc.num_elements() / recipe.hidden_dim).max(1);
        }
        recipe
    }

    /// Hydrates a [`DecisionTransformerRecipe`] from a parsed [`GgufFile`].
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (key, dst) in [
            ("decision-transformer.state_dim", &mut recipe.state_dim),
            ("decision-transformer.act_dim", &mut recipe.act_dim),
            ("decision-transformer.max_ep_len", &mut recipe.max_ep_len),
        ] {
            if let Some(v) = gguf.metadata_value(key).and_then(GgufValue::as_usize)
                && v > 0
            {
                *dst = v;
            }
        }
        recipe
    }

    fn validate_geometry(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        if self.state_dim == 0
            || self.act_dim == 0
            || self.hidden_dim < 4
            || self.num_layers == 0
            || self.n_heads_q == 0
            || self.n_heads_kv == 0
            || self.n_heads_q % self.n_heads_kv != 0
            || self.head_dim < 2
            || self.head_dim % 2 != 0
            || self.ffn_dim < 4
            || self.max_ep_len == 0
            || !self.rms_eps.is_finite()
            || self.rms_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(mismatch_err(
                "decision-transformer",
                "invalid DecisionTransformer hyperparameters (requires hidden_dim >= 4, ffn_dim >= 4, even head_dim >= 2, and n_heads_q divisible by n_heads_kv)",
            ));
        }
        // Fail closed upfront if ANY required weight tensor is missing or wrongly shaped.
        let h = self.hidden_dim;
        let q_dim = self.n_heads_q * self.head_dim;
        let kv_dim = self.n_heads_kv * self.head_dim;
        let _ = require_matvec_view(weights, "dt.embed_return.weight", 1, h)?;
        let _ = require_matvec_view(weights, "dt.embed_state.weight", self.state_dim, h)?;
        let _ = require_matvec_view(weights, "dt.embed_action.weight", h, self.act_dim)?;
        let _ = require_matvec_view(weights, "dt.embed_timestep.weight", h, self.max_ep_len)?;
        let _ = require_tensor_1d(weights, "output_norm.weight", h)?;
        let _ = require_matvec_view(weights, "dt.predict_action.weight", h, self.act_dim)?;
        for layer in 0..self.num_layers {
            let _ = require_tensor_1d(weights, &format!("blk.{layer}.attn_norm.weight"), h)?;
            let _ = require_matvec_view(weights, &format!("blk.{layer}.attn_q.weight"), h, q_dim)?;
            let _ = require_matvec_view(weights, &format!("blk.{layer}.attn_k.weight"), h, kv_dim)?;
            let _ = require_matvec_view(weights, &format!("blk.{layer}.attn_v.weight"), h, kv_dim)?;
            let _ = require_matvec_view(
                weights,
                &format!("blk.{layer}.attn_output.weight"),
                q_dim,
                h,
            )?;
            let _ = require_tensor_1d(weights, &format!("blk.{layer}.ffn_norm.weight"), h)?;
            let _ = require_matvec_view(
                weights,
                &format!("blk.{layer}.ffn_gate.weight"),
                h,
                self.ffn_dim,
            )?;
            let _ = require_matvec_view(
                weights,
                &format!("blk.{layer}.ffn_up.weight"),
                h,
                self.ffn_dim,
            )?;
            let _ = require_matvec_view(
                weights,
                &format!("blk.{layer}.ffn_down.weight"),
                self.ffn_dim,
                h,
            )?;
        }
        Ok(())
    }

    fn attn_spec(&self) -> DenseAttnLayerSpec {
        DenseAttnLayerSpec {
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: false,
            use_qk_norm: false,
        }
    }

    fn run_token_hidden(
        &self,
        mut hidden: Vec<f32>,
        pos: usize,
        seq_len: usize,
        kv_cache: &mut [f32],
        logits: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let spec = self.attn_spec();
        let layer_stride = kv_cache.len() / self.num_layers;
        for layer in 0..self.num_layers {
            let layer_kv = if layer_stride > 0 {
                &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride]
            } else {
                &mut []
            };
            run_dense_attention_sublayer_scaled(
                spec,
                layer,
                pos,
                seq_len,
                &mut hidden,
                layer_kv,
                weights,
                None,
                1.0,
            )?;
            let ffn_norm_w = require_tensor_1d(
                weights,
                &format!("blk.{layer}.ffn_norm.weight"),
                self.hidden_dim,
            )?;
            let mut norm_ffn = vec![0.0_f32; self.hidden_dim];
            rmsnorm_standard(&mut norm_ffn, &hidden, &ffn_norm_w, self.rms_eps)?;
            let ffn_out =
                run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, &norm_ffn, weights)?;
            for (h_val, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h_val += r;
            }
        }
        let out_norm_w = require_tensor_1d(weights, "output_norm.weight", self.hidden_dim)?;
        let mut normed = vec![0.0_f32; self.hidden_dim];
        rmsnorm_standard(&mut normed, &hidden, &out_norm_w, self.rms_eps)?;
        let head_view = require_matvec_view(
            weights,
            "dt.predict_action.weight",
            self.hidden_dim,
            self.act_dim,
        )?;
        let head_bias = lookup_optional_bias(weights, "dt.predict_action.bias", self.act_dim)?;
        let action_logits =
            project_mat_view(&head_view, &normed, self.act_dim, head_bias.as_deref())?;
        logits[..self.act_dim].copy_from_slice(&action_logits);
        Ok(())
    }

    fn timestep_embedding(
        &self,
        step_idx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let clamped = step_idx.min(self.max_ep_len - 1);
        let ts_view = weights.tensor("dt.embed_timestep.weight")?;
        let mut ts = vec![0.0_f32; self.hidden_dim];
        dequantize_token_row(&ts_view, clamped, &mut ts)?;
        validate_finite_slice("dt.embed_timestep.weight", &ts)?;
        Ok(ts)
    }

    fn action_embedding(
        &self,
        action_id: u32,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let idx = usize::try_from(action_id).unwrap_or(usize::MAX);
        if idx >= self.act_dim {
            return Err(bounds_err(
                "action_id",
                self.act_dim,
                idx,
                format!("action_id {idx} >= act_dim {}", self.act_dim),
            ));
        }
        let act_view = weights.tensor("dt.embed_action.weight")?;
        let mut emb = vec![0.0_f32; self.hidden_dim];
        dequantize_token_row(&act_view, idx, &mut emb)?;
        validate_finite_slice("dt.embed_action.weight", &emb)?;
        Ok(emb)
    }

    fn build_prefill_sequence(
        &self,
        ctx: &ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<Vec<f32>>, K3Error> {
        let mut seq = Vec::new();
        if let Some(sp) = ctx.soft_prompt_embeds() {
            let (n_sp, sp_dim) = ctx.soft_prompt_dims().ok_or_else(|| {
                mismatch_err(
                    "decision-transformer",
                    "soft_prompt_embeds missing soft_prompt_dims",
                )
            })?;
            if sp_dim != self.hidden_dim {
                return Err(mismatch_err(
                    "decision-transformer",
                    format!(
                        "soft prompt embed_dim {sp_dim} != hidden_dim {}",
                        self.hidden_dim
                    ),
                ));
            }
            for s in 0..n_sp {
                seq.push(sp[s * sp_dim..(s + 1) * sp_dim].to_vec());
            }
        }

        let actions = ctx.prompt_tokens();
        let ret_view = weights.tensor("dt.embed_return.weight")?;
        let state_view = weights.tensor("dt.embed_state.weight")?;

        if let Some(features) = ctx.feature_input() {
            let (n_steps, n_cols) = ctx.feature_dims().ok_or_else(|| {
                bounds_err("feature_input", 1, 0, "missing feature_dims".to_owned())
            })?;
            let expected_cols = 1 + self.state_dim;
            if n_cols != expected_cols {
                return Err(bounds_err(
                    "feature_input",
                    expected_cols,
                    n_cols,
                    format!(
                        "decision-transformer expected [n_steps, 1 + state_dim] ({expected_cols} cols), got {n_cols}"
                    ),
                ));
            }
            if actions.len() > n_steps {
                return Err(bounds_err(
                    "prompt_tokens",
                    n_steps,
                    actions.len(),
                    format!(
                        "action count {} exceeds trajectory steps {n_steps}",
                        actions.len()
                    ),
                ));
            }
            for t in 0..n_steps {
                let row = &features[t * n_cols..(t + 1) * n_cols];
                let ts = self.timestep_embedding(t, weights)?;
                let mut r_emb = project_mat_view(&ret_view, &row[..1], self.hidden_dim, None)?;
                let mut s_emb = project_mat_view(&state_view, &row[1..], self.hidden_dim, None)?;
                for (dst, &tv) in r_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                for (dst, &tv) in s_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                seq.push(r_emb);
                seq.push(s_emb);
                if let Some(&act_id) = actions.get(t) {
                    let mut a_emb = self.action_embedding(act_id, weights)?;
                    for (dst, &tv) in a_emb.iter_mut().zip(ts.iter()) {
                        *dst += tv;
                    }
                    seq.push(a_emb);
                }
            }
        } else {
            if actions.is_empty() && seq.is_empty() {
                return Err(mismatch_err(
                    "decision-transformer",
                    "prefill requires feature_input ([return_to_go, state]) or prompt action tokens",
                ));
            }
            for (t, &act_id) in actions.iter().enumerate() {
                let ts = self.timestep_embedding(t, weights)?;
                let mut a_emb = self.action_embedding(act_id, weights)?;
                for (dst, &tv) in a_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                seq.push(a_emb);
            }
        }
        Ok(seq)
    }
}

#[cfg(feature = "jero-arch-decision-transformer")]
impl ArchRecipe for DecisionTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        "decision-transformer"
    }

    fn requires_kv_cache(&self) -> bool {
        true
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_geometry(weights)?;
        if ctx.logits().len() < self.act_dim {
            return Err(bounds_err(
                "logits",
                self.act_dim,
                ctx.logits().len(),
                format!(
                    "decision-transformer logits length {} < act_dim {}",
                    ctx.logits().len(),
                    self.act_dim
                ),
            ));
        }

        let seq = self.build_prefill_sequence(ctx, weights)?;
        let start_pos = ctx.position();
        let seq_len = seq.len();
        if start_pos + seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: start_pos + seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let mut step_logits = vec![0.0_f32; self.act_dim];
        for (offset, hidden) in seq.into_iter().enumerate() {
            self.run_token_hidden(
                hidden,
                start_pos + offset,
                start_pos + seq_len,
                ctx.kv_cache_mut(),
                &mut step_logits,
                weights,
            )?;
            if offset + 1 == seq_len {
                ctx.logits_mut()[..self.act_dim].copy_from_slice(&step_logits);
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
        self.validate_geometry(weights)?;
        let act_id = ctx.decode_token().ok_or_else(|| {
            mismatch_err(
                "decision-transformer",
                "decode_step called without staged decode_token",
            )
        })?;
        let pos = ctx.position();
        if pos >= ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: pos + 1,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        if ctx.logits().len() < self.act_dim {
            return Err(bounds_err(
                "logits",
                self.act_dim,
                ctx.logits().len(),
                format!(
                    "logits length {} < act_dim {}",
                    ctx.logits().len(),
                    self.act_dim
                ),
            ));
        }

        let ts = self.timestep_embedding(pos / 3, weights)?;
        let mut hidden = self.action_embedding(act_id, weights)?;
        for (dst, &tv) in hidden.iter_mut().zip(ts.iter()) {
            *dst += tv;
        }
        let mut out_logits = vec![0.0_f32; self.act_dim];
        self.run_token_hidden(
            hidden,
            pos,
            pos + 1,
            ctx.kv_cache_mut(),
            &mut out_logits,
            weights,
        )?;
        ctx.logits_mut()[..self.act_dim].copy_from_slice(&out_logits);
        ctx.advance_position(1)?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded synthetic test fixtures fit cleanly in u32 and f32"
)]
mod tests {
    use super::*;
    #[cfg(all(
        feature = "jero-arch-ttm",
        feature = "jero-arch-ft-transformer",
        feature = "jero-arch-tabpfn-v2",
        feature = "jero-arch-decision-transformer"
    ))]
    use crate::capability::RecipeRegistry;
    #[cfg(any(
        feature = "jero-arch-ttm",
        feature = "jero-arch-ft-transformer",
        feature = "jero-arch-tabpfn-v2",
        feature = "jero-arch-decision-transformer",
    ))]
    use crate::capability::verify_model_capabilities;
    #[cfg(all(
        feature = "jero-arch-ttm",
        feature = "jero-arch-ft-transformer",
        feature = "jero-arch-tabpfn-v2",
        feature = "jero-arch-decision-transformer"
    ))]
    use crate::memory::{architecture_requires_kv_cache, estimate_kv_cache_bytes};
    use crate::qwen_deepseek::Qwen2Recipe;
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

    fn build_gguf_f32<'a>(
        arch: &str,
        extra_kvs: &[(&'a str, GgufValue<'a>)],
        tensors: &[(&str, &[usize], Vec<f32>)],
    ) -> Vec<u8> {
        let specs: Vec<(&str, QuantType, &[usize])> = tensors
            .iter()
            .map(|(name, shape, _)| (*name, QuantType::F32, *shape))
            .collect();
        let mut buf = create_synthetic_gguf_v3_with_kvs(arch, 32, extra_kvs, &specs);
        let offsets: Vec<(String, usize, usize)> = {
            let gguf = GgufFile::parse(&buf).expect("parse scaffold gguf");
            gguf.metadata()
                .tensors()
                .iter()
                .map(|d| (d.name().to_owned(), d.absolute_offset(), d.byte_len()))
                .collect()
        };
        for ((name, _, values), (off_name, abs_off, byte_len)) in tensors.iter().zip(offsets.iter())
        {
            assert_eq!(name, off_name);
            assert_eq!(values.len() * 4, *byte_len);
            for (idx, &v) in values.iter().enumerate() {
                let start = abs_off + idx * 4;
                buf[start..start + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[cfg(feature = "jero-arch-ttm")]
    fn make_ttm_tensors(recipe: &TtmRecipe) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let n_p = recipe.num_patches();
        let c = recipe.num_channels;
        let th = recipe.time_hidden_dim;
        let ch = recipe.channel_hidden_dim;
        let mut out = vec![
            (
                "ttm.patch_proj.weight",
                vec![recipe.patch_length],
                vec![1.0 / (recipe.patch_length as f32); recipe.patch_length],
            ),
            ("ttm.patch_proj.bias", vec![1], vec![0.0]),
            (
                "ttm.forecast_head.weight",
                vec![n_p, recipe.forecast_length],
                (0..n_p * recipe.forecast_length)
                    .map(|i| 0.1 + ((i % 5) as f32) * 0.03)
                    .collect(),
            ),
            (
                "ttm.forecast_head.bias",
                vec![recipe.forecast_length],
                vec![0.02; recipe.forecast_length],
            ),
        ];
        // Layer 0 + Layer 1
        assert_eq!(recipe.num_mixer_layers, 1);
        out.extend([
            ("blk.0.time_norm.weight", vec![n_p], vec![1.0; n_p]),
            ("blk.0.time_norm.bias", vec![n_p], vec![0.0; n_p]),
            (
                "blk.0.time_mlp.fc1.weight",
                vec![n_p, th],
                vec![0.05; n_p * th],
            ),
            ("blk.0.time_mlp.fc1.bias", vec![th], vec![0.01; th]),
            (
                "blk.0.time_mlp.fc2.weight",
                vec![th, n_p],
                vec![0.04; th * n_p],
            ),
            ("blk.0.time_mlp.fc2.bias", vec![n_p], vec![0.0; n_p]),
            ("blk.0.channel_norm.weight", vec![c], vec![1.0; c]),
            ("blk.0.channel_norm.bias", vec![c], vec![0.0; c]),
            (
                "blk.0.channel_mlp.fc1.weight",
                vec![c, ch],
                vec![0.06; c * ch],
            ),
            ("blk.0.channel_mlp.fc1.bias", vec![ch], vec![0.01; ch]),
            (
                "blk.0.channel_mlp.fc2.weight",
                vec![ch, c],
                vec![0.03; ch * c],
            ),
            ("blk.0.channel_mlp.fc2.bias", vec![c], vec![0.0; c]),
        ]);
        out
    }

    #[cfg(feature = "jero-arch-ttm")]
    #[test]
    fn test_ttm_recipe_telemetry_forecast() {
        let recipe = TtmRecipe {
            context_length: 16,
            patch_length: 4,
            forecast_length: 6,
            num_channels: 3,
            num_mixer_layers: 1,
            time_hidden_dim: 8,
            channel_hidden_dim: 6,
            norm_eps: 1e-5,
        };
        assert!(!recipe.requires_kv_cache());
        let tensor_data = make_ttm_tensors(&recipe);
        let tensor_refs: Vec<(&str, &[usize], Vec<f32>)> = tensor_data
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let kvs = [
            ("ttm.context_length", GgufValue::Uint32(16)),
            ("ttm.block_count", GgufValue::Uint32(1)),
            ("ttm.patch_length", GgufValue::Uint32(4)),
            ("ttm.forecast_length", GgufValue::Uint32(6)),
            ("ttm.num_channels", GgufValue::Uint32(3)),
            ("ttm.time_hidden_dim", GgufValue::Uint32(8)),
            ("ttm.channel_hidden_dim", GgufValue::Uint32(6)),
        ];
        let bytes = build_gguf_f32("ttm", &kvs, &tensor_refs);
        let gguf = GgufFile::parse(&bytes).expect("parse ttm gguf");
        let verified = verify_model_capabilities(&gguf).expect("verify ttm capabilities");
        assert_eq!(verified.architecture_id(), "ttm");
        assert!(!verified.requires_kv_cache());

        let series: Vec<f32> = (0..16 * 3).map(|i| (i as f32) * 0.05).collect();
        let mut logits = vec![0.0_f32; 6 * 3];
        let mut kv = vec![0.0_f32; 0];
        let mut ctx = ExecutionContext::new_feature_input(&series, 16, 3, 64, &mut logits, &mut kv)
            .expect("feature context");
        verified
            .forward_classify(&mut ctx, &gguf)
            .expect("ttm telemetry forecast succeeds");
        assert!(ctx.logits().iter().all(|v| v.is_finite()));
        assert!(ctx.logits().iter().any(|v| v.abs() > 1e-4));
        assert!(verified.decode_step(&mut ctx, &gguf).is_err());
    }

    #[cfg(feature = "jero-arch-ttm")]
    #[test]
    fn test_ttm_zero_shot_forecast_parity_and_nan_rejection() {
        let recipe = TtmRecipe {
            context_length: 8,
            patch_length: 2,
            forecast_length: 4,
            num_channels: 2,
            num_mixer_layers: 1,
            time_hidden_dim: 6,
            channel_hidden_dim: 4,
            norm_eps: 1e-5,
        };
        let tensor_data = make_ttm_tensors(&recipe);
        let tensor_refs: Vec<(&str, &[usize], Vec<f32>)> = tensor_data
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let bytes = build_gguf_f32("ttm", &[], &tensor_refs);
        let gguf = GgufFile::parse(&bytes).expect("parse ttm gguf");

        let series_a: Vec<f32> = (0..8 * 2).map(|i| 0.2 + (i as f32) * 0.04).collect();
        let series_b: Vec<f32> = (0..8 * 2).map(|i| 1.5 - (i as f32) * 0.06).collect();
        let mut out_a = vec![0.0_f32; 8];
        let mut out_b = vec![0.0_f32; 8];
        let mut kv = [];

        let mut ctx_a =
            ExecutionContext::new_feature_input(&series_a, 8, 2, 64, &mut out_a, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_a, &gguf).unwrap();

        let mut ctx_b =
            ExecutionContext::new_feature_input(&series_b, 8, 2, 64, &mut out_b, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_b, &gguf).unwrap();
        assert_ne!(out_a, out_b);

        // NaN rejection at context boundary AND fail-closed missing tensor rejection
        let mut bad_series = series_a;
        bad_series[3] = f32::NAN;
        let mut bad_out = vec![0.0_f32; 8];
        assert!(matches!(
            ExecutionContext::new_feature_input(&bad_series, 8, 2, 64, &mut bad_out, &mut kv),
            Err(K3Error::TensorBoundsError { .. })
        ));

        let missing_bytes = create_synthetic_gguf_v3_with_kvs("ttm", 32, &[], &[]);
        let empty_gguf = GgufFile::parse(&missing_bytes).unwrap();
        let mut ctx_miss =
            ExecutionContext::new_feature_input(&[1.0_f32; 16], 8, 2, 64, &mut bad_out, &mut kv)
                .unwrap();
        assert!(matches!(
            recipe.forward_classify(&mut ctx_miss, &empty_gguf),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[cfg(feature = "jero-arch-ft-transformer")]
    fn ft_layer0_tensors(
        recipe: &FtTransformerRecipe,
    ) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let d = recipe.d_token;
        vec![
            ("blk.0.attn_norm.weight", vec![d], vec![1.0; d]),
            ("blk.0.attn_norm.bias", vec![d], vec![0.0; d]),
            (
                "blk.0.attn_q.weight",
                vec![d, d],
                (0..d * d)
                    .map(|i| 0.04 + ((i % 11) as f32) * 0.015)
                    .collect(),
            ),
            (
                "blk.0.attn_k.weight",
                vec![d, d],
                (0..d * d)
                    .map(|i| 0.03 + (((i + 3) % 9) as f32) * 0.015)
                    .collect(),
            ),
            (
                "blk.0.attn_v.weight",
                vec![d, d],
                (0..d * d)
                    .map(|i| 0.05 + (((i * 3) % 13) as f32 - 4.0) * 0.02)
                    .collect(),
            ),
            (
                "blk.0.attn_output.weight",
                vec![d, d],
                (0..d * d)
                    .map(|i| 0.04 + (((i * 5) % 11) as f32 - 3.0) * 0.02)
                    .collect(),
            ),
            ("blk.0.ffn_norm.weight", vec![d], vec![1.0; d]),
            ("blk.0.ffn_norm.bias", vec![d], vec![0.0; d]),
            (
                "blk.0.ffn_fc1.weight",
                vec![d, recipe.ffn_dim],
                (0..d * recipe.ffn_dim)
                    .map(|i| 0.03 + ((i % 9) as f32 - 3.0) * 0.02)
                    .collect(),
            ),
            (
                "blk.0.ffn_fc1.bias",
                vec![recipe.ffn_dim],
                vec![0.01; recipe.ffn_dim],
            ),
            (
                "blk.0.ffn_fc2.weight",
                vec![recipe.ffn_dim, d],
                (0..recipe.ffn_dim * d)
                    .map(|i| 0.02 + ((i % 7) as f32 - 2.0) * 0.02)
                    .collect(),
            ),
            ("blk.0.ffn_fc2.bias", vec![d], vec![0.0; d]),
        ]
    }

    #[cfg(feature = "jero-arch-ft-transformer")]
    fn build_ft_gguf(recipe: &FtTransformerRecipe, cat_vocab: usize) -> Vec<u8> {
        let d = recipe.d_token;
        let cont_len = recipe.num_continuous * d;
        let mut tensors = vec![
            (
                "ft.cls_token.weight",
                vec![d],
                (0..d).map(|i| 0.1 + (i as f32) * 0.07).collect(),
            ),
            (
                "ft.cont_weight",
                vec![cont_len],
                (0..cont_len)
                    .map(|i| 0.15 + ((i % 7) as f32) * 0.04)
                    .collect(),
            ),
            (
                "ft.cont_bias",
                vec![cont_len],
                (0..cont_len).map(|i| ((i % 5) as f32) * 0.02).collect(),
            ),
            ("output_norm.weight", vec![d], vec![1.0; d]),
            ("output_norm.bias", vec![d], vec![0.0; d]),
            (
                "classifier.weight",
                vec![d, recipe.out_dim],
                (0..d * recipe.out_dim)
                    .map(|i| {
                        let in_idx = i / recipe.out_dim;
                        let out_idx = i % recipe.out_dim;
                        let sign = if (in_idx + out_idx) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sign * (0.1 + (in_idx as f32) * 0.05)
                            + if out_idx == 0 { 0.15 } else { -0.1 }
                    })
                    .collect(),
            ),
            (
                "classifier.bias",
                vec![recipe.out_dim],
                vec![0.05; recipe.out_dim],
            ),
        ];
        tensors.extend(ft_layer0_tensors(recipe));
        if recipe.num_categorical > 0 {
            tensors.push((
                "ft.cat_embd.weight",
                vec![d, cat_vocab],
                (0..d * cat_vocab)
                    .map(|i| {
                        let cat_id = i / d;
                        let dim = i % d;
                        if dim == (cat_id % d) {
                            1.2 + (cat_id as f32) * 0.3
                        } else {
                            (((dim + cat_id) % 3) as f32) * 0.15 - 0.1
                        }
                    })
                    .collect(),
            ));
            tensors.push((
                "ft.cat_bias",
                vec![recipe.num_categorical * d],
                vec![0.02; recipe.num_categorical * d],
            ));
        }
        let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let kvs = [
            ("ft-transformer.block_count", GgufValue::Uint32(1)),
            (
                "ft-transformer.embedding_length",
                GgufValue::Uint32(d as u32),
            ),
            (
                "ft-transformer.attention.head_count",
                GgufValue::Uint32(recipe.n_heads as u32),
            ),
            (
                "ft-transformer.feed_forward_length",
                GgufValue::Uint32(recipe.ffn_dim as u32),
            ),
            (
                "ft-transformer.num_continuous",
                GgufValue::Uint32(recipe.num_continuous as u32),
            ),
            (
                "ft-transformer.num_categorical",
                GgufValue::Uint32(recipe.num_categorical as u32),
            ),
            (
                "ft-transformer.out_dim",
                GgufValue::Uint32(recipe.out_dim as u32),
            ),
        ];
        build_gguf_f32("ft-transformer", &kvs, &refs)
    }

    #[cfg(feature = "jero-arch-ft-transformer")]
    #[test]
    fn test_ft_transformer_recipe_forward() {
        let recipe = FtTransformerRecipe {
            num_continuous: 3,
            num_categorical: 0,
            d_token: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            out_dim: 2,
            norm_eps: 1e-5,
        };
        let bytes = build_ft_gguf(&recipe, 0);
        let gguf = GgufFile::parse(&bytes).unwrap();
        let verified = verify_model_capabilities(&gguf).expect("verify ft-transformer");
        assert_eq!(verified.architecture_id(), "ft-transformer");
        assert!(!verified.requires_kv_cache());

        let mut logits = [0.0_f32; 2];
        let mut kv = [];
        let row = [1.2_f32, -0.4, 0.8];
        let mut ctx =
            ExecutionContext::new_feature_input(&row, 1, 3, 64, &mut logits, &mut kv).unwrap();
        verified
            .forward_classify(&mut ctx, &gguf)
            .expect("ft forward");
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_ne!(logits[0], logits[1]);
    }

    #[cfg(feature = "jero-arch-ft-transformer")]
    #[test]
    fn test_ft_transformer_continuous_categorical_scoring() {
        let recipe = FtTransformerRecipe {
            num_continuous: 2,
            num_categorical: 2,
            d_token: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            out_dim: 2,
            norm_eps: 1e-5,
        };
        let bytes = build_ft_gguf(&recipe, 4);
        let gguf = GgufFile::parse(&bytes).unwrap();
        let mut kv = [];

        let row_cat0 = [0.5_f32, -0.3, 0.0, 1.0];
        let row_cat3 = [0.5_f32, -0.3, 3.0, 2.0];
        let mut out_a = [0.0_f32; 2];
        let mut out_b = [0.0_f32; 2];
        let mut ctx_a =
            ExecutionContext::new_feature_input(&row_cat0, 1, 4, 64, &mut out_a, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_a, &gguf).unwrap();
        let mut ctx_b =
            ExecutionContext::new_feature_input(&row_cat3, 1, 4, 64, &mut out_b, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_b, &gguf).unwrap();
        assert_ne!(out_a, out_b);
    }

    #[cfg(all(feature = "jero-arch-ttm", feature = "jero-arch-ft-transformer"))]
    #[test]
    fn test_ttm_and_ft_transformer_forward() {
        let ttm = TtmRecipe {
            context_length: 8,
            patch_length: 2,
            forecast_length: 4,
            num_channels: 2,
            num_mixer_layers: 1,
            time_hidden_dim: 6,
            channel_hidden_dim: 4,
            norm_eps: 1e-5,
        };
        let ttm_data = make_ttm_tensors(&ttm);
        let ttm_refs: Vec<(&str, &[usize], Vec<f32>)> = ttm_data
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let ttm_gguf_bytes = build_gguf_f32("ttm", &[], &ttm_refs);
        let ttm_gguf = GgufFile::parse(&ttm_gguf_bytes).unwrap();
        let mut ttm_out = [0.0_f32; 8];
        let mut kv = [];
        let mut ctx1 =
            ExecutionContext::new_feature_input(&[0.3_f32; 16], 8, 2, 64, &mut ttm_out, &mut kv)
                .unwrap();
        ttm.forward_classify(&mut ctx1, &ttm_gguf).unwrap();
        assert!(ttm_out.iter().all(|v| v.is_finite()));

        let ft = FtTransformerRecipe {
            num_continuous: 3,
            num_categorical: 0,
            d_token: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            out_dim: 1,
            norm_eps: 1e-5,
        };
        let ft_bytes = build_ft_gguf(&ft, 0);
        let ft_gguf = GgufFile::parse(&ft_bytes).unwrap();
        let mut ft_out = [0.0_f32; 1];
        let mut ctx2 = ExecutionContext::new_feature_input(
            &[0.5_f32, 1.0, -0.2],
            1,
            3,
            64,
            &mut ft_out,
            &mut kv,
        )
        .unwrap();
        ft.forward_classify(&mut ctx2, &ft_gguf).unwrap();
        assert!(ft_out[0].is_finite());
    }

    #[cfg(feature = "jero-arch-tabpfn-v2")]
    fn build_tabpfn_gguf(recipe: &TabPfnV2Recipe) -> Vec<u8> {
        let d = recipe.hidden_dim;
        let enc_len = recipe.num_features * d;
        let tensors = vec![
            (
                "tabpfn.feature_enc.weight",
                vec![enc_len],
                (0..enc_len)
                    .map(|i| 0.1 + ((i % 5) as f32) * 0.03)
                    .collect(),
            ),
            (
                "tabpfn.feature_enc.bias",
                vec![enc_len],
                vec![0.02; enc_len],
            ),
            (
                "tabpfn.y_embd.weight",
                vec![d, recipe.num_classes],
                (0..d * recipe.num_classes)
                    .map(|i| if i < d { 0.4 } else { -0.4 })
                    .collect(),
            ),
            ("tabpfn.y_unknown.weight", vec![d], vec![0.0; d]),
            ("blk.0.feat_norm.weight", vec![d], vec![1.0; d]),
            ("blk.0.feat_norm.bias", vec![d], vec![0.0; d]),
            ("blk.0.feat_q.weight", vec![d, d], vec![0.08; d * d]),
            ("blk.0.feat_k.weight", vec![d, d], vec![0.07; d * d]),
            ("blk.0.feat_v.weight", vec![d, d], vec![0.09; d * d]),
            ("blk.0.feat_out.weight", vec![d, d], vec![0.05; d * d]),
            ("blk.0.sample_norm.weight", vec![d], vec![1.0; d]),
            ("blk.0.sample_norm.bias", vec![d], vec![0.0; d]),
            ("blk.0.sample_q.weight", vec![d, d], vec![0.1; d * d]),
            ("blk.0.sample_k.weight", vec![d, d], vec![0.09; d * d]),
            ("blk.0.sample_v.weight", vec![d, d], vec![0.11; d * d]),
            ("blk.0.sample_out.weight", vec![d, d], vec![0.07; d * d]),
            ("blk.0.ffn_norm.weight", vec![d], vec![1.0; d]),
            ("blk.0.ffn_norm.bias", vec![d], vec![0.0; d]),
            (
                "blk.0.ffn_fc1.weight",
                vec![d, recipe.ffn_dim],
                vec![0.05; d * recipe.ffn_dim],
            ),
            (
                "blk.0.ffn_fc1.bias",
                vec![recipe.ffn_dim],
                vec![0.01; recipe.ffn_dim],
            ),
            (
                "blk.0.ffn_fc2.weight",
                vec![recipe.ffn_dim, d],
                vec![0.04; recipe.ffn_dim * d],
            ),
            ("blk.0.ffn_fc2.bias", vec![d], vec![0.0; d]),
            ("output_norm.weight", vec![d], vec![1.0; d]),
            ("output_norm.bias", vec![d], vec![0.0; d]),
            (
                "classifier.weight",
                vec![d, recipe.num_classes],
                (0..d * recipe.num_classes)
                    .map(|i| if i < d { 0.2 } else { -0.25 })
                    .collect(),
            ),
            (
                "classifier.bias",
                vec![recipe.num_classes],
                vec![0.01; recipe.num_classes],
            ),
        ];
        let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let kvs = [
            ("tabpfn-v2.block_count", GgufValue::Uint32(1)),
            ("tabpfn-v2.embedding_length", GgufValue::Uint32(d as u32)),
            (
                "tabpfn-v2.attention.head_count",
                GgufValue::Uint32(recipe.n_heads as u32),
            ),
            (
                "tabpfn-v2.feed_forward_length",
                GgufValue::Uint32(recipe.ffn_dim as u32),
            ),
            (
                "tabpfn-v2.num_features",
                GgufValue::Uint32(recipe.num_features as u32),
            ),
            (
                "tabpfn-v2.num_classes",
                GgufValue::Uint32(recipe.num_classes as u32),
            ),
        ];
        build_gguf_f32("tabpfn-v2", &kvs, &refs)
    }

    #[cfg(feature = "jero-arch-tabpfn-v2")]
    #[test]
    fn test_tabpfn_v2_dual_axis_attention() {
        let shape = TabPfnAttentionShape::new(4, 3, 2, 4).with_context_samples(2);
        let total = shape.total_elements().unwrap();
        let q: Vec<f32> = (0..total).map(|i| 0.1 + ((i % 7) as f32) * 0.05).collect();
        let k: Vec<f32> = (0..total).map(|i| 0.2 - ((i % 5) as f32) * 0.04).collect();
        let mut v_a: Vec<f32> = (0..total).map(|i| ((i % 9) as f32) * 0.1).collect();
        let mut v_b = v_a.clone();
        // Corrupt ONLY query row 3 (index >= n_ctx = 2) values in V
        for slot in &mut v_b[3 * 3 * 8..] {
            *slot = 999.0;
        }
        let mut out_a = vec![0.0_f32; total];
        let mut out_b = vec![0.0_f32; total];
        tabpfn_dual_axis_attention_f32(
            &mut out_a,
            &q,
            &k,
            &v_a,
            shape,
            TabPfnAttentionAxis::Sample,
            0.5,
        )
        .unwrap();
        tabpfn_dual_axis_attention_f32(
            &mut out_b,
            &q,
            &k,
            &v_b,
            shape,
            TabPfnAttentionAxis::Sample,
            0.5,
        )
        .unwrap();
        #[cfg(not(miri))]
        assert_eq!(
            out_a, out_b,
            "query V entries must never leak into sample attention"
        );
        for (&a, &b) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "query V entries leaked into sample attention: {a} vs {b}"
            );
        }
        v_a[0] += 0.5;
        let mut out_feat = vec![0.0_f32; total];
        tabpfn_dual_axis_attention_f32(
            &mut out_feat,
            &q,
            &k,
            &v_a,
            shape,
            TabPfnAttentionAxis::Feature,
            0.5,
        )
        .unwrap();
        assert!(out_feat.iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "jero-arch-tabpfn-v2")]
    #[test]
    fn test_tabpfn_v2_in_context_tabular_classification() {
        let recipe = TabPfnV2Recipe {
            num_features: 3,
            hidden_dim: 8,
            num_layers: 1,
            n_heads: 2,
            ffn_dim: 16,
            num_classes: 2,
            norm_eps: 1e-5,
        };
        let bytes = build_tabpfn_gguf(&recipe);
        let gguf = GgufFile::parse(&bytes).unwrap();
        let verified = verify_model_capabilities(&gguf).expect("verify tabpfn-v2");
        assert_eq!(verified.architecture_id(), "tabpfn-v2");
        assert!(!verified.requires_kv_cache());

        // 2 context rows (labels [0, 1]) + 2 query rows
        let rows_a = [
            0.1_f32, 0.2, 0.3, // ctx 0 (class 0)
            0.9, 0.8, 0.7, // ctx 1 (class 1)
            0.15, 0.25, 0.35, // query 0
            0.85, 0.75, 0.65, // query 1
        ];
        let mut rows_b = rows_a;
        rows_b[9] = 42.0; // change query 1 only

        let mut logits_a = [0.0_f32; 4];
        let mut logits_b = [0.0_f32; 4];
        let mut kv = [];
        let mut ctx_a = ExecutionContext::new(&[0_u32, 1], 64, &mut logits_a, &mut kv)
            .unwrap()
            .with_feature_input(&rows_a, 4, 3)
            .unwrap();
        verified.forward_classify(&mut ctx_a, &gguf).unwrap();

        let mut ctx_b = ExecutionContext::new(&[0_u32, 1], 64, &mut logits_b, &mut kv)
            .unwrap()
            .with_feature_input(&rows_b, 4, 3)
            .unwrap();
        verified.forward_classify(&mut ctx_b, &gguf).unwrap();

        assert!(logits_a.iter().all(|v| v.is_finite()));
        assert_eq!(
            logits_a[0..2],
            logits_b[0..2],
            "query 1 must not leak into query 0 classification logits"
        );
    }

    #[cfg(feature = "jero-arch-decision-transformer")]
    fn build_dt_gguf(recipe: &DecisionTransformerRecipe) -> Vec<u8> {
        let mut tensors = dt_embedding_and_head_tensors(recipe);
        tensors.extend(dt_layer0_tensors(recipe));
        let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();
        let kvs = [
            ("decision-transformer.block_count", GgufValue::Uint32(1)),
            (
                "decision-transformer.embedding_length",
                GgufValue::Uint32(u32::try_from(recipe.hidden_dim).unwrap()),
            ),
            (
                "decision-transformer.attention.head_count",
                GgufValue::Uint32(u32::try_from(recipe.n_heads_q).unwrap()),
            ),
            (
                "decision-transformer.attention.head_count_kv",
                GgufValue::Uint32(u32::try_from(recipe.n_heads_kv).unwrap()),
            ),
            (
                "decision-transformer.feed_forward_length",
                GgufValue::Uint32(u32::try_from(recipe.ffn_dim).unwrap()),
            ),
            (
                "decision-transformer.state_dim",
                GgufValue::Uint32(u32::try_from(recipe.state_dim).unwrap()),
            ),
            (
                "decision-transformer.act_dim",
                GgufValue::Uint32(u32::try_from(recipe.act_dim).unwrap()),
            ),
            (
                "decision-transformer.max_ep_len",
                GgufValue::Uint32(u32::try_from(recipe.max_ep_len).unwrap()),
            ),
        ];
        build_gguf_f32("decision-transformer", &kvs, &refs)
    }

    #[cfg(feature = "jero-arch-decision-transformer")]
    fn dt_embedding_and_head_tensors(
        recipe: &DecisionTransformerRecipe,
    ) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let h = recipe.hidden_dim;
        vec![
            ("dt.embed_return.weight", vec![1, h], vec![0.2_f32; h]),
            (
                "dt.embed_state.weight",
                vec![recipe.state_dim, h],
                vec![0.15_f32; recipe.state_dim * h],
            ),
            (
                "dt.embed_action.weight",
                vec![h, recipe.act_dim],
                (0..h * recipe.act_dim)
                    .map(|i| 0.05 + f32::from(u16::try_from(i / h).unwrap()) * 0.1)
                    .collect(),
            ),
            (
                "dt.embed_timestep.weight",
                vec![h, recipe.max_ep_len],
                (0..h * recipe.max_ep_len)
                    .map(|i| f32::from(u16::try_from(i / h).unwrap()) * 0.01)
                    .collect(),
            ),
            ("output_norm.weight", vec![h], vec![1.0; h]),
            (
                "dt.predict_action.weight",
                vec![h, recipe.act_dim],
                (0..h * recipe.act_dim)
                    .map(|i| if (i / h) % 2 == 0 { 0.25 } else { -0.15 })
                    .collect(),
            ),
            (
                "dt.predict_action.bias",
                vec![recipe.act_dim],
                vec![0.02; recipe.act_dim],
            ),
        ]
    }

    #[cfg(feature = "jero-arch-decision-transformer")]
    fn dt_layer0_tensors(
        recipe: &DecisionTransformerRecipe,
    ) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let h = recipe.hidden_dim;
        let q_dim = recipe.n_heads_q * recipe.head_dim;
        let kv_dim = recipe.n_heads_kv * recipe.head_dim;
        vec![
            ("blk.0.attn_norm.weight", vec![h], vec![1.0; h]),
            ("blk.0.attn_q.weight", vec![h, q_dim], vec![0.08; h * q_dim]),
            (
                "blk.0.attn_k.weight",
                vec![h, kv_dim],
                vec![0.07; h * kv_dim],
            ),
            (
                "blk.0.attn_v.weight",
                vec![h, kv_dim],
                vec![0.09; h * kv_dim],
            ),
            (
                "blk.0.attn_output.weight",
                vec![q_dim, h],
                vec![0.06; q_dim * h],
            ),
            ("blk.0.ffn_norm.weight", vec![h], vec![1.0; h]),
            (
                "blk.0.ffn_gate.weight",
                vec![h, recipe.ffn_dim],
                vec![0.05; h * recipe.ffn_dim],
            ),
            (
                "blk.0.ffn_up.weight",
                vec![h, recipe.ffn_dim],
                vec![0.05; h * recipe.ffn_dim],
            ),
            (
                "blk.0.ffn_down.weight",
                vec![recipe.ffn_dim, h],
                vec![0.04; recipe.ffn_dim * h],
            ),
        ]
    }

    #[cfg(feature = "jero-arch-decision-transformer")]
    #[test]
    fn test_decision_transformer_rtg_action_autoregression() {
        let recipe = DecisionTransformerRecipe {
            state_dim: 3,
            act_dim: 4,
            hidden_dim: 8,
            num_layers: 1,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            ffn_dim: 16,
            max_ep_len: 16,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let bytes = build_dt_gguf(&recipe);
        let gguf = GgufFile::parse(&bytes).unwrap();
        let verified = verify_model_capabilities(&gguf).expect("verify decision-transformer");
        assert_eq!(verified.architecture_id(), "decision-transformer");
        assert!(verified.requires_kv_cache());

        // 2 steps: (R0=10, s0=[0.1,0.2,0.3], a0=1), (R1=8, s1=[0.4,0.5,0.6]) -> 5 sequence tokens
        let traj = [10.0_f32, 0.1, 0.2, 0.3, 8.0, 0.4, 0.5, 0.6];
        let mut logits = [0.0_f32; 4];
        let mut kv_cache = vec![0.0_f32; 64 * 2 * 8 * 2];
        let mut ctx = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 2, 4)
            .unwrap();
        verified.prefill_step(&mut ctx, &gguf).expect("dt prefill");
        assert_eq!(ctx.position(), 5);
        assert!(ctx.logits().iter().all(|v| v.is_finite()));

        ctx.prepare_decode(2).unwrap();
        verified.decode_step(&mut ctx, &gguf).expect("dt decode");
        assert_eq!(ctx.position(), 6);
        assert!(ctx.logits().iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "jero-arch-decision-transformer")]
    #[test]
    fn test_decision_transformer_rejects_small_ffn_dim_and_indivisible_gqa_heads() {
        let small_ffn = DecisionTransformerRecipe {
            state_dim: 3,
            act_dim: 4,
            hidden_dim: 8,
            num_layers: 1,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            ffn_dim: 3,
            max_ep_len: 16,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let bytes = build_dt_gguf(&small_ffn);
        let gguf = GgufFile::parse(&bytes).unwrap();
        let traj = [10.0_f32, 0.1, 0.2, 0.3];
        let mut logits = [0.0_f32; 4];
        let mut kv_cache = vec![0.0_f32; 64 * 2 * 8 * 2];
        let mut ctx = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 1, 4)
            .unwrap();
        assert!(matches!(
            small_ffn.prefill_step(&mut ctx, &gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let bad_gqa = DecisionTransformerRecipe {
            n_heads_q: 3,
            n_heads_kv: 2,
            ffn_dim: 16,
            ..small_ffn
        };
        let bad_bytes = build_dt_gguf(&bad_gqa);
        let bad_gguf = GgufFile::parse(&bad_bytes).unwrap();
        let mut ctx_gqa = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 1, 4)
            .unwrap();
        assert!(matches!(
            bad_gqa.prefill_step(&mut ctx_gqa, &bad_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[cfg(all(
        feature = "jero-arch-ttm",
        feature = "jero-arch-ft-transformer",
        feature = "jero-arch-tabpfn-v2",
        feature = "jero-arch-decision-transformer"
    ))]
    #[test]
    fn test_ttm_tabpfn_ft_decision_transformer_recipes() {
        let archs = RecipeRegistry::supported_architectures();
        for arch in ["ttm", "ft-transformer", "tabpfn-v2", "decision-transformer"] {
            assert!(
                archs.contains(&arch),
                "missing {arch} in supported_architectures"
            );
            let r = RecipeRegistry::find_recipe(arch).expect("recipe found");
            assert_eq!(r.architecture_id(), arch);
        }
        assert!(!architecture_requires_kv_cache("ttm"));
        assert!(!architecture_requires_kv_cache("ft-transformer"));
        assert!(!architecture_requires_kv_cache("tabpfn-v2"));
        assert!(architecture_requires_kv_cache("decision-transformer"));

        let ttm_meta_bytes = create_synthetic_gguf_v3_with_kvs("ttm", 32, &[], &[]);
        let ttm_gguf = GgufFile::parse(&ttm_meta_bytes).unwrap();
        assert_eq!(
            estimate_kv_cache_bytes(ttm_gguf.metadata(), 64, false).unwrap(),
            0
        );
    }

    #[test]
    fn test_mmproj_dimension_mismatch_rejected() {
        let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 8, 16, 2);
        assert!(matches!(
            bridge.verify_target_compatibility("qwen3", "1.0.0", 16),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            bridge.verify_target_compatibility("qwen2", "2.0.0", 16),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            bridge.verify_target_compatibility("qwen2", "1.0.0", 32),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let target = Qwen2Recipe::new();
        let bytes = build_gguf_f32(
            "mmproj",
            &[],
            &[("mmproj.weight", &[8, 2 * 16], vec![0.1_f32; 8 * 32])],
        );
        let gguf = GgufFile::parse(&bytes).unwrap();
        let wrong_meta_bytes = create_synthetic_gguf_v3_with_kvs(
            "qwen2",
            32,
            &[("qwen2.embedding_length", GgufValue::Uint32(32))],
            &[],
        );
        let wrong_gguf = GgufFile::parse(&wrong_meta_bytes).unwrap();
        assert!(matches!(
            bridge.verify_target_recipe(&target, wrong_gguf.metadata()),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Wrong latent len or missing weights fail closed
        let mut out = vec![0.0_f32; 32];
        assert!(matches!(
            project_machine_latent_to_soft_prompt(&bridge, &[1.0_f32; 4], &gguf, &mut out),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_mmproj_projects_machine_latent_into_llm_soft_prompt() {
        let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 4, 8, 2);
        let bytes = build_gguf_f32(
            "mmproj",
            &[],
            &[
                ("mmproj.fc1.weight", &[4, 6], vec![0.2_f32; 24]),
                ("mmproj.fc1.bias", &[6], vec![0.05; 6]),
                ("mmproj.fc2.weight", &[6, 16], vec![0.15_f32; 96]),
                ("mmproj.fc2.bias", &[16], vec![0.02; 16]),
            ],
        );
        let gguf = GgufFile::parse(&bytes).unwrap();
        let mut soft_a = vec![0.0_f32; 16];
        let mut soft_b = vec![0.0_f32; 16];
        project_machine_latent_to_soft_prompt(&bridge, &[0.5, -0.3, 0.8, 0.1], &gguf, &mut soft_a)
            .expect("mmproj projection succeeds");
        project_machine_latent_to_soft_prompt(&bridge, &[-0.9, 0.7, 0.2, 1.4], &gguf, &mut soft_b)
            .expect("mmproj second latent succeeds");
        assert!(soft_a.iter().all(|v| v.is_finite()));
        assert_ne!(soft_a, soft_b);

        // Layer-0 soft-prompt injection onto Qwen2Recipe shifts logits compared to plain prompt
        let llm_bytes = build_gguf_f32(
            "qwen2",
            &[
                ("qwen2.block_count", GgufValue::Uint32(1)),
                ("qwen2.embedding_length", GgufValue::Uint32(8)),
                ("qwen2.attention.head_count", GgufValue::Uint32(2)),
                ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
                ("qwen2.feed_forward_length", GgufValue::Uint32(16)),
            ],
            &[
                (
                    "token_embd.weight",
                    &[8, 8],
                    (0..64).map(|i| 0.1 + ((i % 8) as f32) * 0.04).collect(),
                ),
                (
                    "output.weight",
                    &[8, 8],
                    (0..64).map(|i| 0.05 + (i % 7) as f32 * 0.03).collect(),
                ),
                ("output_norm.weight", &[8], vec![1.0; 8]),
                ("blk.0.attn_norm.weight", &[8], vec![1.0; 8]),
                ("blk.0.attn_q.weight", &[8, 8], vec![0.1; 64]),
                ("blk.0.attn_q.bias", &[8], vec![0.01; 8]),
                ("blk.0.attn_k.weight", &[8, 8], vec![0.1; 64]),
                ("blk.0.attn_k.bias", &[8], vec![0.01; 8]),
                ("blk.0.attn_v.weight", &[8, 8], vec![0.1; 64]),
                ("blk.0.attn_v.bias", &[8], vec![0.01; 8]),
                ("blk.0.attn_output.weight", &[8, 8], vec![0.1; 64]),
                ("blk.0.ffn_norm.weight", &[8], vec![1.0; 8]),
                ("blk.0.ffn_gate.weight", &[8, 16], vec![0.05; 128]),
                ("blk.0.ffn_up.weight", &[8, 16], vec![0.05; 128]),
                ("blk.0.ffn_down.weight", &[16, 8], vec![0.05; 128]),
            ],
        );
        let llm_gguf = GgufFile::parse(&llm_bytes).unwrap();
        let llm = Qwen2Recipe {
            num_layers: 1,
            hidden_dim: 8,
            ffn_dim: 16,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 4,
            vocab_size: 8,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        bridge
            .verify_target_compatibility(
                llm.architecture_id(),
                llm.recipe_version(),
                llm.hidden_dim,
            )
            .unwrap();

        let mut logits_plain = [0.0_f32; 8];
        let mut logits_soft = [0.0_f32; 8];
        let mut kv_plain = vec![0.0_f32; 64 * 2 * 8];
        let mut kv_soft = vec![0.0_f32; 64 * 2 * 8];
        let mut ctx_plain =
            ExecutionContext::new(&[2_u32, 3], 64, &mut logits_plain, &mut kv_plain).unwrap();
        llm.prefill_step(&mut ctx_plain, &llm_gguf).unwrap();
        let end_pos = forward_with_soft_prompt(
            &llm,
            &[2_u32, 3],
            64,
            &soft_a,
            2,
            8,
            &mut logits_soft,
            &mut kv_soft,
            &llm_gguf,
        )
        .expect("layer-0 soft prompt forward succeeds");
        assert_eq!(end_pos, 4);
        assert!(logits_soft.iter().all(|v| v.is_finite()));
        assert_ne!(logits_plain, logits_soft);
    }

    #[test]
    fn test_mmproj_latent_bridge_tuple_check_and_projection() {
        let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 4, 8, 2);
        bridge
            .verify_target_compatibility("qwen2", "1.0.0", 8)
            .expect("matching tuple passes");

        // Fail-closed: absent projector weights MUST error instead of synthesizing identity
        let empty_bytes = create_synthetic_gguf_v3_with_kvs("mmproj", 32, &[], &[]);
        let empty_gguf = GgufFile::parse(&empty_bytes).unwrap();
        let mut out = vec![0.0_f32; 16];
        assert!(matches!(
            project_machine_latent_to_soft_prompt(&bridge, &[1.0_f32; 4], &empty_gguf, &mut out),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Linear projector path (`mmproj.weight` + `mmproj.norm.weight`)
        let w_bytes = build_gguf_f32(
            "mmproj",
            &[],
            &[
                (
                    "mmproj.weight",
                    &[4, 16],
                    (0..64).map(|i| 0.1 + (i % 4) as f32 * 0.05).collect(),
                ),
                ("mmproj.bias", &[16], vec![0.02; 16]),
                ("mmproj.norm.weight", &[8], vec![1.0; 8]),
                ("mmproj.norm.bias", &[8], vec![0.0; 8]),
            ],
        );
        let w_gguf = GgufFile::parse(&w_bytes).unwrap();
        project_machine_latent_to_soft_prompt(&bridge, &[0.4, 0.8, -0.2, 0.6], &w_gguf, &mut out)
            .expect("linear mmproj path passes");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
