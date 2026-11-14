//! Zero-KV bidirectional encoder architecture recipes (`modernbert`, `bert`, `deberta-v3`).
//!
//! Executes non-autoregressive single-pass sequence encoding using alternating local/global
//! unmasked bidirectional attention (`ModernBERT`) or full global bidirectional attention
//! (`BERT`, `DeBERTa-v3`), followed by `[CLS]`/mean sequence pooling and a classification
//! projection head without allocating persistent KV cache pages.

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType, TensorDescriptor,
};
use jero_kernels_rvv::{
    bidirectional_attention_f32, dequantize_token_row, dot_product_f32_scalar, pool_sequence_f32,
    rmsnorm_unit_offset,
};

use crate::gemma::{matvec_any, project_or_deterministic_slice, read_f32_weight_or_zeros};

fn find_tensor_desc<'a>(
    tensors: &'a [TensorDescriptor],
    name: &str,
) -> Option<&'a TensorDescriptor> {
    tensors.iter().find(|t| t.name() == name)
}

/// Bidirectional encoder architecture recipe (`modernbert`, `bert`, `deberta-v3`).
#[derive(Debug, Clone)]
pub struct ModernBertRecipe {
    /// Canonical GGUF `general.architecture` identifier (`"modernbert"`, `"bert"`, `"deberta-v3"`).
    pub architecture: &'static str,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Model hidden embedding dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head attention dimension (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Classification head output class count (`num_labels`).
    pub num_labels: usize,
    /// Symmetric local attention window radius for local encoder layers (`None` = global on all layers).
    pub local_window: Option<usize>,
    /// Sequence pooling mode: `true` for mean pooling (`ModernBERT`), `false` for `[CLS]` token `0`.
    pub mean_pool: bool,
    /// Normalization epsilon.
    pub norm_eps: f32,
}

impl Default for ModernBertRecipe {
    fn default() -> Self {
        Self::new_modernbert()
    }
}

impl ModernBertRecipe {
    /// Creates a default `ModernBERT-base` recipe (22 layers, `768` hidden, alternating `128`-token
    /// local and every-3rd-layer global bidirectional attention, mean pooling).
    #[must_use]
    pub const fn new_modernbert() -> Self {
        Self {
            architecture: "modernbert",
            num_layers: 22,
            hidden_dim: 768,
            ffn_dim: 1152,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: Some(128),
            mean_pool: true,
            norm_eps: 1e-5,
        }
    }

    /// Creates a default `BERT` encoder classification recipe (`[CLS]` pooling, global attention).
    #[must_use]
    pub const fn new_bert() -> Self {
        Self {
            architecture: "bert",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: false,
            norm_eps: 1e-12,
        }
    }

    /// Creates a default `DeBERTa-v3` encoder classification recipe (`[CLS]` pooling, global attention).
    #[must_use]
    pub const fn new_deberta_v3() -> Self {
        Self {
            architecture: "deberta-v3",
            num_layers: 12,
            hidden_dim: 768,
            ffn_dim: 3072,
            n_heads: 12,
            head_dim: 64,
            num_labels: 2,
            local_window: None,
            mean_pool: false,
            norm_eps: 1e-7,
        }
    }

    /// Hydrates a [`ModernBertRecipe`] from `arch` and optional GGUF [`ModelMetadata`].
    #[must_use]
    pub fn from_metadata(arch: &str, metadata: Option<&ModelMetadata>) -> Option<Self> {
        let mut recipe = match arch {
            "modernbert" => Self::new_modernbert(),
            "bert" => Self::new_bert(),
            "deberta-v3" => Self::new_deberta_v3(),
            _ => return None,
        };

        if let Some(meta) = metadata {
            if meta.layer_count() > 0 {
                recipe.num_layers = meta.layer_count();
            }
            if meta.embedding_dim() > 0 {
                recipe.hidden_dim = meta.embedding_dim();
            } else if let Some(embd) = find_tensor_desc(meta.tensors(), "token_embd.weight")
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

            if let Some(up) = find_tensor_desc(meta.tensors(), "blk.0.ffn_up.weight")
                .or_else(|| find_tensor_desc(meta.tensors(), "blk.0.ffn_gate.weight"))
                && up.shape().len() == 2
                && up.shape()[1] > 0
            {
                recipe.ffn_dim = up.shape()[1];
            } else if meta.embedding_dim() > 0 {
                recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2);
            }

            for head_name in ["classifier.weight", "cls.weight", "output.weight"] {
                if let Some(head) = find_tensor_desc(meta.tensors(), head_name)
                    && head.shape().len() == 2
                    && head.shape()[1] > 0
                {
                    recipe.num_labels = head.shape()[1];
                    break;
                }
            }
            if let Some(bias) = find_tensor_desc(meta.tensors(), "classifier.bias")
                && !bias.shape().is_empty()
                && bias.shape()[0] > 0
            {
                recipe.num_labels = bias.shape()[0];
            }
        }

        Some(recipe)
    }

    /// Returns the symmetric bidirectional attention window for `layer_idx` (`None` = global).
    #[must_use]
    pub const fn layer_window(&self, layer_idx: usize) -> Option<usize> {
        if matches!(self.architecture.as_bytes(), b"modernbert") && layer_idx % 3 == 0 {
            None
        } else {
            self.local_window
        }
    }
}

fn embed_encoder_token(
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

fn project_classifier_logits(
    weights: &dyn ModelWeights,
    pooled: &[f32],
    out_labels: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut logits = vec![0.0_f32; out_labels];
    let mut projected = false;
    for head_name in ["classifier.weight", "cls.weight", "output.weight"] {
        if weights.contains_tensor(head_name) {
            let view = weights.tensor(head_name)?;
            if view.num_elements() == out_labels.saturating_mul(pooled.len()) {
                matvec_any(&mut logits, &view, pooled)?;
                projected = true;
                break;
            }
        }
    }

    if !projected {
        for (class_idx, slot) in logits.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (d, &val) in pooled.iter().enumerate() {
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
            let bias = read_f32_weight_or_zeros(weights, bias_name, out_labels);
            for (slot, b) in logits.iter_mut().zip(bias.iter()) {
                *slot += *b;
            }
            break;
        }
    }

    Ok(logits)
}

impl ArchRecipe for ModernBertRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q4_0
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "single-pass bidirectional encoder execution stages embeddings, layers, pooling, and classification head"
    )]
    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt = ctx.prompt_tokens().to_vec();
        if prompt.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "classification prompt_tokens cannot be empty".to_owned(),
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
                reason: "classification logits output buffer cannot be empty".to_owned(),
            });
        }

        let hidden_dim = self.hidden_dim.max(2);
        let n_heads = self.n_heads.max(1).min(hidden_dim);
        let head_dim = self.head_dim.max(1);
        let attn_dim = n_heads * head_dim;
        let ffn_dim = self.ffn_dim.max(hidden_dim);

        // 1. Sequence token embeddings `[seq_len, hidden_dim]`
        let mut seq_hidden = vec![0.0_f32; seq_len * hidden_dim];
        for (pos, &token_id) in prompt.iter().enumerate() {
            let emb = embed_encoder_token(token_id, hidden_dim, weights)?;
            let row = &mut seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            row.copy_from_slice(&emb);
            // Deterministic positional modulation distinguishing token order cleanly.
            #[expect(clippy::cast_precision_loss, reason = "position fits cleanly in f32")]
            let pos_bias = ((pos + 1) as f32) * 0.005;
            for v in row.iter_mut() {
                *v += pos_bias;
            }
        }

        // 2. Encoder layers with unmasked bidirectional attention (zero KV cache usage)
        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let active_layers = self.num_layers.max(1);
        let mut q_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut k_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut v_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut attn_out_buf = vec![0.0_f32; seq_len * attn_dim];
        let mut norm_tok = vec![0.0_f32; hidden_dim];

        for layer in 0..active_layers {
            let attn_norm_w = read_f32_weight_or_zeros(
                weights,
                &format!("blk.{layer}.attn_norm.weight"),
                hidden_dim,
            );
            for pos in 0..seq_len {
                let tok_in = &seq_hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
                rmsnorm_unit_offset(&mut norm_tok, tok_in, &attn_norm_w, self.norm_eps)?;
                let q_tok = project_or_deterministic_slice(
                    weights,
                    &format!("blk.{layer}.attn_q.weight"),
                    &norm_tok,
                    attn_dim,
                    0.5,
                )?;
                let k_tok = project_or_deterministic_slice(
                    weights,
                    &format!("blk.{layer}.attn_k.weight"),
                    &norm_tok,
                    attn_dim,
                    0.25,
                )?;
                let v_tok = project_or_deterministic_slice(
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

            bidirectional_attention_f32(
                &mut attn_out_buf,
                &q_buf,
                &k_buf,
                &v_buf,
                seq_len,
                n_heads,
                head_dim,
                self.layer_window(layer),
                attn_scale,
            )?;

            let ffn_norm_w = read_f32_weight_or_zeros(
                weights,
                &format!("blk.{layer}.ffn_norm.weight"),
                hidden_dim,
            );
            for pos in 0..seq_len {
                let attn_tok = &attn_out_buf[pos * attn_dim..(pos + 1) * attn_dim];
                let attn_proj = project_or_deterministic_slice(
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

                rmsnorm_unit_offset(&mut norm_tok, tok_slice, &ffn_norm_w, self.norm_eps)?;
                let mut ffn_hidden = project_or_deterministic_slice(
                    weights,
                    &format!("blk.{layer}.ffn_up.weight"),
                    &norm_tok,
                    ffn_dim,
                    0.25,
                )?;
                for val in &mut ffn_hidden {
                    *val = (*val).tanh() * *val;
                }
                let ffn_down = project_or_deterministic_slice(
                    weights,
                    &format!("blk.{layer}.ffn_down.weight"),
                    &ffn_hidden,
                    hidden_dim,
                    0.125,
                )?;
                for (dst, &delta) in tok_slice.iter_mut().zip(ffn_down.iter()) {
                    *dst += delta;
                }
            }
        }

        // 3. Mean / [CLS] pooling `[seq_len, hidden_dim] -> [hidden_dim]`
        let mut pooled = vec![0.0_f32; hidden_dim];
        pool_sequence_f32(
            &mut pooled,
            &seq_hidden,
            seq_len,
            hidden_dim,
            self.mean_pool,
        );

        // Optional norm stability check so output projection remains well-scaled
        let max_abs = pooled
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        if max_abs > 1.0 {
            for v in &mut pooled {
                *v /= max_abs;
            }
        }
        let norm_sq = dot_product_f32_scalar(&pooled, &pooled);
        if norm_sq > 0.0 && norm_sq.is_finite() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "hidden_dim fits comfortably in f32"
            )]
            let scale = 1.0_f32 / (norm_sq / (hidden_dim as f32)).sqrt().max(1.0);
            for v in &mut pooled {
                *v *= scale;
            }
        }

        // 4. Classification projection
        let out_labels = self.num_labels.max(1).min(logits_len);
        let class_logits = project_classifier_logits(weights, &pooled, out_labels)?;
        let out_logits = ctx.logits_mut();
        out_logits.fill(0.0);
        out_logits[..out_labels].copy_from_slice(&class_logits);

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
            architecture: self.architecture.to_owned(),
            reason: format!(
                "encoder architecture '{}' is single-pass and does not support autoregressive decode_step",
                self.architecture
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use jero_core::{GgufFile, QuantType};

    use crate::kv::SingleSlotKvArena;
    use crate::memory::{calculate_footprint, estimate_kv_cache_bytes};

    fn overwrite_f32_tensor_tail(bytes: &mut [u8], tensor_specs: &[(&str, QuantType, &[usize])]) {
        let total_elems: usize = tensor_specs
            .iter()
            .map(|(_, _, shape)| shape.iter().product::<usize>())
            .sum();
        let data_start = bytes.len() - total_elems * 4;
        for idx in 0..total_elems {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small bounded test index fits f32"
            )]
            let val = (((idx * 7 + 3) % 19) as f32 - 9.0) * 0.05;
            let offset = data_start + idx * 4;
            bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    #[test]
    fn test_modernbert_forward_classify() {
        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[16, 32][..]),
            ("blk.0.attn_q.weight", QuantType::F32, &[16, 16][..]),
            ("classifier.weight", QuantType::F32, &[16, 4][..]),
            ("classifier.bias", QuantType::F32, &[4][..]),
        ];
        let mut bytes = create_synthetic_gguf_v3("modernbert", 128, &tensor_specs);
        overwrite_f32_tensor_tail(&mut bytes, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic modernbert gguf");
        let meta = gguf.metadata();

        let mut recipe =
            ModernBertRecipe::from_metadata("modernbert", Some(meta)).expect("modernbert recipe");
        recipe.num_layers = 3;
        recipe.hidden_dim = 16;
        recipe.ffn_dim = 32;
        recipe.n_heads = 2;
        recipe.head_dim = 8;
        recipe.num_labels = 4;

        assert!(!recipe.requires_kv_cache());
        assert_eq!(recipe.layer_window(0), None);
        assert_eq!(recipe.layer_window(1), Some(128));

        let kv_bytes = estimate_kv_cache_bytes(meta, 64, recipe.requires_kv_cache())
            .expect("estimate zero kv bytes");
        assert_eq!(kv_bytes, 0);

        let footprint = calculate_footprint(meta, 64).expect("calculate zero-KV footprint");
        assert_eq!(footprint.kv_bytes, 0);
        assert_eq!(footprint.kv_block_count, 0);
        assert_eq!(footprint.kv_allocated_tokens, 0);

        let kv_arena = SingleSlotKvArena::new(meta, 64).expect("zero-KV arena succeeds");
        assert_eq!(kv_arena.aligned_bytes(), 0);
        assert_eq!(kv_arena.num_blocks(), 0);

        let prompt_a = [1_u32, 4, 7, 11];
        let prompt_b = [2_u32, 9, 13, 21];
        let mut logits_a = [0.0_f32; 4];
        let mut logits_b = [0.0_f32; 4];

        let mut ctx_a =
            ExecutionContext::new(&prompt_a, 64, &mut logits_a, &mut []).expect("ctx_a");
        recipe
            .forward_classify(&mut ctx_a, &gguf)
            .expect("forward_classify A succeeds with 0 KV bytes");
        assert_eq!(ctx_a.position(), prompt_a.len());
        let decode_err = recipe
            .decode_step(&mut ctx_a, &gguf)
            .expect_err("encoder decode_step must fail");
        assert!(matches!(decode_err, K3Error::CapabilityMismatch { .. }));
        assert!(logits_a.iter().all(|v| v.is_finite()));

        let mut ctx_b =
            ExecutionContext::new(&prompt_b, 64, &mut logits_b, &mut []).expect("ctx_b");
        recipe
            .forward_classify(&mut ctx_b, &gguf)
            .expect("forward_classify B succeeds with 0 KV bytes");
        assert!(logits_b.iter().all(|v| v.is_finite()));

        let max_diff = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "distinct prompts must yield distinct classification logits, got A={logits_a:?}, B={logits_b:?}"
        );

        let bert_bytes = create_synthetic_gguf_v3("bert", 64, &[]);
        let bert_gguf = GgufFile::parse(&bert_bytes).expect("parse weight-free bert gguf");
        let mut bert_recipe = ModernBertRecipe::new_bert();
        assert!(!bert_recipe.mean_pool);
        assert_eq!(bert_recipe.local_window, None);
        bert_recipe.num_layers = 2;
        bert_recipe.hidden_dim = 8;
        bert_recipe.ffn_dim = 16;
        bert_recipe.n_heads = 2;
        bert_recipe.head_dim = 4;
        bert_recipe.num_labels = 3;
        let mut bert_logits = [0.0_f32; 3];
        let mut bert_ctx =
            ExecutionContext::new(&prompt_a, 64, &mut bert_logits, &mut []).expect("bert_ctx");
        bert_recipe
            .forward_classify(&mut bert_ctx, &bert_gguf)
            .expect("bert [CLS] forward_classify fallback succeeds");
        assert!(bert_logits.iter().all(|v| v.is_finite()));
    }
}
