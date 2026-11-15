//! Architecture recipe capability filter and compile-time registry.
//!
//! Enforces upfront capability verification against compiled `arch-*` recipes and compute kernels
//! before reserving or locking memory (ADR-017, ADR-023, ADR-024, US-01 Scenario 1).

use jero_core::{
    ArchRecipe, GgufArray, GgufFile, GgufValue, K3Error, ModelMetadata, TensorDescriptor,
};

/// Architecture hyperparameters extracted from GGUF v3 metadata and tensor shape descriptors.
///
/// Populates dynamic [`ArchRecipe`] instances at model-open time so models across `0.5B`, `1.5B`,
/// `3B`, `7B`, `8B`, and `MoE` scales execute with their authentic dimensions instead of static
/// compile-time `8B` defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufHyperparams {
    /// Canonical `general.architecture` string (e.g. `"llama"`, `"qwen2"`, `"mistral"`).
    pub architecture: String,
    /// Transformer block count (`{arch}.block_count`).
    pub block_count: usize,
    /// Residual / embedding hidden dimension (`{arch}.embedding_length`).
    pub embedding_length: usize,
    /// `SwiGLU` / MLP intermediate dimension (`{arch}.feed_forward_length`).
    pub feed_forward_length: usize,
    /// Query attention head count (`{arch}.attention.head_count`).
    pub head_count: usize,
    /// Grouped-query K/V head count (`{arch}.attention.head_count_kv`).
    pub head_count_kv: usize,
    /// Per-head K/V dimension (`{arch}.attention.key_length` or `embedding_length / head_count`).
    pub key_length: usize,
    /// Explicit `{arch}.attention.key_length` when present in GGUF metadata or inferred from Q/K tensor shapes.
    pub explicit_key_length: Option<usize>,
    /// Optional `RoPE` frequency base (`{arch}.rope.freq_base`).
    pub rope_freq_base: Option<f32>,
    /// Optional `RMSNorm` epsilon (`{arch}.attention.layer_norm_rms_epsilon`).
    pub rms_norm_eps: Option<f32>,
    /// Vocabulary size (`{arch}.vocab_size`, tokenizer vocabulary length, or LM head row count).
    pub vocab_size: usize,
    /// Trained context window length (`{arch}.context_length`).
    pub context_length: usize,
    /// Optional sliding-window attention span (`{arch}.attention.sliding_window`).
    pub sliding_window: Option<u32>,
    /// Optional `RoPE` rotary dimension count (`{arch}.rope.dimension_count`).
    pub rope_dim_count: Option<u32>,
    /// Optional `RoPE` scaling type (`{arch}.rope.scaling.type`).
    pub rope_scaling_type: Option<String>,
    /// Optional `RoPE` scaling factor (`{arch}.rope.scaling.factor`).
    pub rope_scaling_factor: Option<f32>,
    /// Optional `RoPE` original context length (`{arch}.rope.scaling.orig_ctx_len` or
    /// `{arch}.rope.scaling.original_context_length`).
    pub rope_orig_ctx_len: Option<u32>,
    /// Optional explicit per-head value dimension (`{arch}.attention.value_length`).
    pub value_length: Option<u32>,
    /// Optional attention score logit softcapping threshold (`{arch}.attn_logit_softcapping`).
    pub attn_logit_softcapping: Option<f32>,
    /// Optional final LM-head logit softcapping threshold (`{arch}.final_logit_softcapping`).
    pub final_logit_softcapping: Option<f32>,
    /// Optional `MoE` total expert count (`{arch}.expert_count`).
    pub expert_count: Option<usize>,
    /// Optional `MoE` active experts per token (`{arch}.expert_used_count`).
    pub expert_used_count: Option<usize>,
    /// Optional `MoE` shared expert count (`{arch}.expert_shared_count`).
    pub expert_shared_count: Option<usize>,
    /// Optional per-expert `MoE` feed-forward dimension (`{arch}.expert.feed_forward_length`).
    pub expert_feed_forward_length: Option<usize>,
    /// Optional Mamba SSM conv1d kernel width (`{arch}.ssm.conv_kernel`).
    pub ssm_conv_kernel: Option<usize>,
    /// Optional Mamba SSM inner projection width (`{arch}.ssm.inner_size`).
    pub ssm_inner_size: Option<usize>,
    /// Optional Mamba SSM state dimension (`{arch}.ssm.state_size`).
    pub ssm_state_size: Option<usize>,
    /// Optional Mamba SSM time-step head count / rank (`{arch}.ssm.time_step_rank`).
    pub ssm_time_step_rank: Option<usize>,
    /// Optional Mamba-2 SSM group count (`{arch}.ssm.group_count`).
    pub ssm_group_count: Option<usize>,
    /// Optional LFM2 short-convolution cache length (`{arch}.shortconv.l_cache`).
    pub shortconv_l_cache: Option<usize>,
    /// Optional RWKV WKV head dimension (`{arch}.wkv.head_size`).
    pub wkv_head_size: Option<usize>,
    /// Optional embedding multiplier (`{arch}.embedding_scale`).
    pub embedding_scale: Option<f32>,
    /// Optional residual branch multiplier (`{arch}.residual_scale`).
    pub residual_scale: Option<f32>,
    /// Optional attention score multiplier (`{arch}.attention.scale`).
    pub attention_scale: Option<f32>,
    /// Optional output logit divisor (`{arch}.logit_scale`).
    pub logit_scale: Option<f32>,
}

fn find_tensor<'a>(tensors: &'a [TensorDescriptor], name: &str) -> Option<&'a TensorDescriptor> {
    tensors.iter().find(|t| t.name() == name)
}

fn infer_layer_count_from_tensors(tensors: &[TensorDescriptor]) -> usize {
    let mut max_layer: Option<usize> = None;
    for tensor in tensors {
        if let Some(rest) = tensor.name().strip_prefix("blk.")
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            max_layer = Some(max_layer.map_or(idx, |cur| cur.max(idx)));
        }
    }
    max_layer.map_or(0, |m| m.saturating_add(1))
}

fn infer_embedding_dim_from_tensors(tensors: &[TensorDescriptor]) -> usize {
    for norm_name in ["blk.0.attn_norm.weight", "output_norm.weight"] {
        if let Some(desc) = find_tensor(tensors, norm_name)
            && desc.shape().len() == 1
            && desc.shape()[0] > 0
        {
            return desc.shape()[0];
        }
    }
    if let Some(embd) = find_tensor(tensors, "token_embd.weight")
        && embd.shape().len() == 2
        && embd.shape()[0] > 0
    {
        return embd.shape()[0];
    }
    0
}

fn infer_ffn_dim_from_tensors(tensors: &[TensorDescriptor], embedding_dim: usize) -> usize {
    for gate_name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_gate.0.weight",
    ] {
        if let Some(desc) = find_tensor(tensors, gate_name)
            && desc.shape().len() == 2
            && (embedding_dim == 0 || desc.shape()[0] == embedding_dim)
            && desc.shape()[1] > 0
        {
            return desc.shape()[1];
        }
    }
    if let Some(desc) = find_tensor(tensors, "blk.0.ffn_gate_exps.weight")
        && desc.shape().len() >= 2
        && (embedding_dim == 0 || desc.shape()[0] == embedding_dim)
        && desc.shape()[1] > 0
    {
        return desc.shape()[1];
    }
    if let Some(desc) = find_tensor(tensors, "blk.0.ffn_gate_up.weight")
        && desc.shape().len() == 2
        && (embedding_dim == 0 || desc.shape()[0] == embedding_dim)
        && desc.shape()[1] >= 2
    {
        return desc.shape()[1] / 2;
    }
    if let Some(desc) = find_tensor(tensors, "blk.0.ffn_down.weight")
        && desc.shape().len() == 2
        && desc.shape()[0] > 0
    {
        return desc.shape()[0];
    }
    0
}

fn infer_key_length_from_tensors(
    tensors: &[TensorDescriptor],
    embedding_dim: usize,
    head_count: usize,
    head_count_kv: usize,
) -> Option<usize> {
    if embedding_dim == 0 {
        return None;
    }
    if head_count_kv > 0
        && let Some(k_desc) = find_tensor(tensors, "blk.0.attn_k.weight")
        && k_desc.shape().len() == 2
        && k_desc.shape()[0] == embedding_dim
        && k_desc.shape()[1] > 0
        && k_desc.shape()[1] % head_count_kv == 0
    {
        return Some(k_desc.shape()[1] / head_count_kv);
    }
    if head_count > 0
        && let Some(q_desc) = find_tensor(tensors, "blk.0.attn_q.weight")
        && q_desc.shape().len() == 2
        && q_desc.shape()[0] == embedding_dim
        && q_desc.shape()[1] > 0
        && q_desc.shape()[1] % head_count == 0
    {
        return Some(q_desc.shape()[1] / head_count);
    }
    None
}

fn infer_vocab_rows_from_tensors(tensors: &[TensorDescriptor]) -> usize {
    for name in ["output.weight", "token_embd.weight"] {
        if let Some(desc) = find_tensor(tensors, name)
            && desc.shape().len() == 2
            && desc.shape()[1] > 0
        {
            return desc.shape()[1];
        }
    }
    0
}

impl GgufHyperparams {
    /// Extracts model hyperparameters from parsed [`ModelMetadata`] and tensor shape descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let arch = metadata.architecture().to_owned();
        let tensors = metadata.tensors();

        let block_count = if metadata.layer_count() > 0 {
            metadata.layer_count()
        } else {
            infer_layer_count_from_tensors(tensors)
        };

        let embedding_length = if metadata.embedding_dim() > 0 {
            metadata.embedding_dim()
        } else {
            infer_embedding_dim_from_tensors(tensors)
        };

        let feed_forward_length = infer_ffn_dim_from_tensors(tensors, embedding_length);
        let head_count = metadata.head_count();
        let head_count_kv = if metadata.head_count_kv() > 0 {
            metadata.head_count_kv()
        } else {
            head_count
        };

        let tensor_key_len =
            infer_key_length_from_tensors(tensors, embedding_length, head_count, head_count_kv);
        let key_length = tensor_key_len.unwrap_or_else(|| {
            if arch == "dory" && embedding_length == 2560 && head_count == 16 {
                256
            } else if embedding_length > 0 && head_count > 0 {
                embedding_length / head_count
            } else {
                0
            }
        });

        let vocab_size = infer_vocab_rows_from_tensors(tensors);

        let sliding_window = metadata
            .sliding_window()
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0);
        let rope_dim_count = metadata
            .rope_dimension_count()
            .and_then(|v| u32::try_from(v).ok());
        let value_length = metadata.value_length().and_then(|v| u32::try_from(v).ok());
        let explicit_key_length = metadata.key_length().or(tensor_key_len);
        let key_length = explicit_key_length.unwrap_or(key_length);

        Self {
            architecture: arch,
            block_count,
            embedding_length,
            feed_forward_length: metadata
                .feed_forward_length()
                .filter(|&v| v > 0)
                .unwrap_or(feed_forward_length),
            head_count,
            head_count_kv,
            key_length,
            explicit_key_length,
            rope_freq_base: metadata
                .rope_freq_base()
                .filter(|v| v.is_finite() && *v > 0.0),
            rms_norm_eps: metadata
                .rms_norm_eps()
                .filter(|v| v.is_finite() && *v > 0.0),
            vocab_size: metadata
                .vocab_size()
                .filter(|&v| v > 0)
                .unwrap_or(vocab_size),
            context_length: metadata.context_length(),
            sliding_window,
            rope_dim_count,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_orig_ctx_len: None,
            value_length,
            attn_logit_softcapping: metadata.attn_logit_softcapping(),
            final_logit_softcapping: metadata.final_logit_softcapping(),
            expert_count: metadata.expert_count(),
            expert_used_count: metadata.expert_used_count(),
            expert_shared_count: metadata.expert_shared_count(),
            expert_feed_forward_length: metadata.expert_feed_forward_length(),
            ssm_conv_kernel: metadata.ssm_conv_kernel(),
            ssm_inner_size: metadata.ssm_inner_size(),
            ssm_state_size: metadata.ssm_state_size(),
            ssm_time_step_rank: metadata.ssm_time_step_rank(),
            ssm_group_count: metadata.ssm_group_count(),
            shortconv_l_cache: metadata.shortconv_l_cache(),
            wkv_head_size: metadata.wkv_head_size(),
            embedding_scale: metadata.embedding_scale(),
            residual_scale: metadata.residual_scale(),
            attention_scale: metadata.attention_scale(),
            logit_scale: metadata.logit_scale(),
        }
    }

    fn lookup_arch_value<'a>(
        gguf: &'a GgufFile<'a>,
        arch: &str,
        suffix: &str,
    ) -> Option<&'a GgufValue<'a>> {
        gguf.metadata_value(&format!("{arch}.{suffix}"))
            .or_else(|| {
                if arch == "phi4" {
                    gguf.metadata_value(&format!("phi3.{suffix}"))
                } else {
                    None
                }
            })
    }

    /// Extracts full model hyperparameters directly from a parsed [`GgufFile`] header, combining
    /// raw GGUF key-value entries (`feed_forward_length`, `attention.key_length`,
    /// `attention.value_length`, `rope.freq_base`, `rope.dimension_count`, `rope.scaling.*`,
    /// `attention.layer_norm_rms_epsilon`, `vocab_size`, `sliding_window`, softcapping, and
    /// `expert_count`) with tensor-shape fallbacks.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut params = Self::from_metadata(gguf.metadata());
        let arch = params.architecture.clone();

        if let Some(ffn) = Self::lookup_arch_value(gguf, &arch, "feed_forward_length")
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.feed_forward_length = ffn;
        }

        if let Some(kl) = Self::lookup_arch_value(gguf, &arch, "attention.key_length")
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.explicit_key_length = Some(kl);
            params.key_length = kl;
        }

        if let Some(vl) = Self::lookup_arch_value(gguf, &arch, "attention.value_length")
            .and_then(GgufValue::as_u32)
        {
            params.value_length = Some(vl);
        }

        if let Some(freq) = Self::lookup_arch_value(gguf, &arch, "rope.freq_base")
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            params.rope_freq_base = Some(freq);
        }

        if let Some(rd) =
            Self::lookup_arch_value(gguf, &arch, "rope.dimension_count").and_then(GgufValue::as_u32)
        {
            params.rope_dim_count = Some(rd);
        }

        if let Some(st) = Self::lookup_arch_value(gguf, &arch, "rope.scaling.type")
            .and_then(GgufValue::as_str)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
        {
            params.rope_scaling_type = Some(st);
        }

        if let Some(sf) =
            Self::lookup_arch_value(gguf, &arch, "rope.scaling.factor").and_then(GgufValue::as_f32)
        {
            params.rope_scaling_factor = Some(sf);
        }

        if let Some(orig) = Self::lookup_arch_value(gguf, &arch, "rope.scaling.orig_ctx_len")
            .or_else(|| {
                Self::lookup_arch_value(gguf, &arch, "rope.scaling.original_context_length")
            })
            .and_then(GgufValue::as_u32)
        {
            params.rope_orig_ctx_len = Some(orig);
        }

        if let Some(eps) = Self::lookup_arch_value(gguf, &arch, "attention.layer_norm_rms_epsilon")
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            params.rms_norm_eps = Some(eps);
        }

        if let Some(cap) = Self::lookup_arch_value(gguf, &arch, "attn_logit_softcapping")
            .and_then(GgufValue::as_f32)
        {
            params.attn_logit_softcapping = Some(cap);
        }

        if let Some(cap) = Self::lookup_arch_value(gguf, &arch, "final_logit_softcapping")
            .and_then(GgufValue::as_f32)
        {
            params.final_logit_softcapping = Some(cap);
        }

        if let Some(GgufArray::String(tokens)) = gguf
            .metadata_value("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            && !tokens.is_empty()
        {
            params.vocab_size = params.vocab_size.max(tokens.len());
        }

        if let Some(vs) = Self::lookup_arch_value(gguf, &arch, "vocab_size")
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.vocab_size = vs;
        }

        if let Some(sw) = Self::lookup_arch_value(gguf, &arch, "attention.sliding_window")
            .and_then(GgufValue::as_u32)
            .filter(|&v| v > 0)
        {
            params.sliding_window = Some(sw);
        }

        if let Some(ec) = Self::lookup_arch_value(gguf, &arch, "expert_count")
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.expert_count = Some(ec);
        }

        if let Some(euc) = Self::lookup_arch_value(gguf, &arch, "expert_used_count")
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.expert_used_count = Some(euc);
        }

        Self::populate_optional_hyperparams_from_gguf(gguf, &arch, &mut params);
        params
    }

    fn populate_optional_hyperparams_from_gguf(gguf: &GgufFile<'_>, arch: &str, params: &mut Self) {
        for (slot, keys) in [
            (
                &mut params.expert_shared_count,
                &["expert_shared_count", "expert.shared_count"][..],
            ),
            (
                &mut params.expert_feed_forward_length,
                &["expert.feed_forward_length", "expert_feed_forward_length"][..],
            ),
            (&mut params.ssm_conv_kernel, &["ssm.conv_kernel"][..]),
            (&mut params.ssm_inner_size, &["ssm.inner_size"][..]),
            (&mut params.ssm_state_size, &["ssm.state_size"][..]),
            (&mut params.ssm_time_step_rank, &["ssm.time_step_rank"][..]),
            (&mut params.ssm_group_count, &["ssm.group_count"][..]),
            (&mut params.shortconv_l_cache, &["shortconv.l_cache"][..]),
            (&mut params.wkv_head_size, &["wkv.head_size"][..]),
        ] {
            if slot.is_none() {
                *slot = keys.iter().find_map(|key| {
                    Self::lookup_arch_value(gguf, arch, key)
                        .and_then(GgufValue::as_usize)
                        .filter(|&v| v > 0)
                });
            }
        }
        for (key, slot) in [
            ("embedding_scale", &mut params.embedding_scale),
            ("residual_scale", &mut params.residual_scale),
            ("attention.scale", &mut params.attention_scale),
            ("logit_scale", &mut params.logit_scale),
        ] {
            if slot.is_none()
                && let Some(val) = Self::lookup_arch_value(gguf, arch, key)
                    .and_then(GgufValue::as_f32)
                    .filter(|v| v.is_finite() && *v > 0.0)
            {
                *slot = Some(val);
            }
        }
    }
}

/// In-process compile-time registry mapping GGUF `general.architecture` strings
/// to compiled [`ArchRecipe`] implementations (ADR-023, ADR-024).
#[derive(Debug, Clone, Copy, Default)]
pub struct RecipeRegistry;

impl RecipeRegistry {
    #[cfg(feature = "jero-kernels-rvv")]
    fn find_qwen_deepseek_recipe(architecture: &str) -> Option<&'static dyn ArchRecipe> {
        match architecture {
            "qwen2" => {
                static QWEN2: crate::qwen_deepseek::Qwen2Recipe =
                    crate::qwen_deepseek::Qwen2Recipe::new();
                Some(&QWEN2)
            }
            "qwen3" => {
                static QWEN3: crate::qwen_deepseek::Qwen3Recipe =
                    crate::qwen_deepseek::Qwen3Recipe::new();
                Some(&QWEN3)
            }
            "qwen3moe" => {
                static QWEN3MOE: crate::qwen_deepseek::Qwen3MoeRecipe =
                    crate::qwen_deepseek::Qwen3MoeRecipe::new_30b_a3b();
                Some(&QWEN3MOE)
            }
            "deepseek2" => {
                static DEEPSEEK2: crate::qwen_deepseek::DeepSeek2Recipe =
                    crate::qwen_deepseek::DeepSeek2Recipe::new();
                Some(&DEEPSEEK2)
            }
            "qwen35" => {
                static QWEN35: crate::qwen_deepseek::Qwen35Recipe =
                    crate::qwen_deepseek::Qwen35Recipe::new();
                Some(&QWEN35)
            }
            "qwen35moe" => {
                static QWEN35MOE: crate::qwen_deepseek::Qwen35MoeRecipe =
                    crate::qwen_deepseek::Qwen35MoeRecipe::new();
                Some(&QWEN35MOE)
            }
            _ => None,
        }
    }

    /// Returns the compiled [`ArchRecipe`] matching `architecture`, if enabled.
    ///
    /// Minor versions (such as `"qwen3"` vs `"qwen3.8"`) are treated as distinct algorithms
    /// per ADR-023 and must be registered separately.
    #[must_use]
    pub fn find_recipe(architecture: &str) -> Option<&'static dyn ArchRecipe> {
        #[cfg(feature = "jero-arch-synthetic")]
        if architecture == "synthetic" {
            static SYNTHETIC: jero_arch_synthetic::SyntheticRecipe =
                jero_arch_synthetic::SyntheticRecipe::new();
            return Some(&SYNTHETIC);
        }

        #[cfg(feature = "jero-kernels-rvv")]
        if let Some(recipe) = Self::find_qwen_deepseek_recipe(architecture) {
            return Some(recipe);
        }

        #[cfg(feature = "jero-arch-llama")]
        if architecture == "llama" || architecture == "qwen2" {
            static LLAMA: jero_arch_llama::LlamaRecipe = jero_arch_llama::LlamaRecipe::new();
            return Some(&LLAMA);
        } else if architecture == "llama4" {
            static LLAMA4: jero_arch_llama::LlamaRecipe =
                jero_arch_llama::LlamaRecipe::new_llama4();
            return Some(&LLAMA4);
        }

        #[cfg(feature = "jero-arch-mistral")]
        if architecture == "mistral" || architecture == "mistral3" {
            static MISTRAL: jero_arch_mistral::MistralRecipe =
                jero_arch_mistral::MistralRecipe::new();
            return Some(&MISTRAL);
        } else if architecture == "mixtral" {
            static MIXTRAL: jero_arch_mistral::MistralRecipe =
                jero_arch_mistral::MistralRecipe::new_mixtral();
            return Some(&MIXTRAL);
        }

        #[cfg(feature = "jero-arch-phi3")]
        if architecture == "phi3" {
            static PHI3: jero_arch_phi3::Phi3Recipe = jero_arch_phi3::Phi3Recipe::new();
            return Some(&PHI3);
        } else if architecture == "phi4" {
            static PHI4: jero_arch_phi3::Phi3Recipe = jero_arch_phi3::Phi3Recipe::new_phi4();
            return Some(&PHI4);
        }

        #[cfg(feature = "jero-arch-bitnet")]
        if architecture == "bitnet" {
            static BITNET: jero_arch_bitnet::BitNetRecipe = jero_arch_bitnet::BitNetRecipe::new();
            return Some(&BITNET);
        } else if architecture == "bitnet-b1.58" {
            static BITNET_B158: jero_arch_bitnet::BitNetRecipe =
                jero_arch_bitnet::BitNetRecipe::new_b158();
            return Some(&BITNET_B158);
        } else if architecture == "bitnet-25" {
            static BITNET_25: jero_arch_bitnet::BitNetRecipe =
                jero_arch_bitnet::BitNetRecipe::new_25();
            return Some(&BITNET_25);
        }

        #[cfg(feature = "jero-arch-dory")]
        if architecture == "dory" {
            static DORY: jero_arch_dory::DoryRecipe = jero_arch_dory::DoryRecipe::new();
            return Some(&DORY);
        }

        #[cfg(feature = "jero-arch-gemma")]
        if architecture == "gemma2" {
            static GEMMA2: crate::gemma::GemmaRecipe = crate::gemma::GemmaRecipe::new_gemma2();
            return Some(&GEMMA2);
        } else if architecture == "gemma3" {
            static GEMMA3: crate::gemma::GemmaRecipe = crate::gemma::GemmaRecipe::new_gemma3();
            return Some(&GEMMA3);
        } else if architecture == "gemma4" {
            static GEMMA4: crate::gemma::Gemma4Recipe = crate::gemma::Gemma4Recipe::new();
            return Some(&GEMMA4);
        } else if architecture == "functiongemma" {
            static FUNCTIONGEMMA: crate::gemma::GemmaRecipe =
                crate::gemma::GemmaRecipe::new_functiongemma();
            return Some(&FUNCTIONGEMMA);
        }

        #[cfg(feature = "jero-arch-modernbert")]
        if architecture == "modernbert" {
            static MODERNBERT: crate::modernbert::ModernBertRecipe =
                crate::modernbert::ModernBertRecipe::new_modernbert();
            return Some(&MODERNBERT);
        } else if architecture == "bert" {
            static BERT: crate::modernbert::ModernBertRecipe =
                crate::modernbert::ModernBertRecipe::new_bert();
            return Some(&BERT);
        } else if architecture == "deberta-v3" {
            static DEBERTA_V3: crate::forward::DebertaV3Recipe =
                crate::forward::DebertaV3Recipe::new();
            return Some(&DEBERTA_V3);
        }

        #[cfg(feature = "jero-kernels-rvv")]
        if let Some(recipe) = Self::find_ssm_or_granite_recipe(architecture) {
            return Some(recipe);
        }

        let _ = architecture;
        None
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn find_ssm_or_granite_recipe(architecture: &str) -> Option<&'static dyn ArchRecipe> {
        match architecture {
            "mamba" => {
                static MAMBA: crate::ssm::Mamba2Recipe = crate::ssm::Mamba2Recipe::new_mamba();
                Some(&MAMBA)
            }
            "mamba2" => {
                static MAMBA2: crate::ssm::Mamba2Recipe = crate::ssm::Mamba2Recipe::new();
                Some(&MAMBA2)
            }
            "rwkv6" => {
                static RWKV6: crate::ssm::Rwkv7Recipe = crate::ssm::Rwkv7Recipe::new_rwkv6();
                Some(&RWKV6)
            }
            "rwkv7" => {
                static RWKV7: crate::ssm::Rwkv7Recipe = crate::ssm::Rwkv7Recipe::new();
                Some(&RWKV7)
            }
            "jamba" => {
                static JAMBA: std::sync::LazyLock<crate::ssm::HybridSsmTransformerRecipe> =
                    std::sync::LazyLock::new(crate::ssm::HybridSsmTransformerRecipe::new_jamba);
                Some(std::sync::LazyLock::force(&JAMBA))
            }
            "zamba2" => {
                static ZAMBA2: std::sync::LazyLock<crate::ssm::HybridSsmTransformerRecipe> =
                    std::sync::LazyLock::new(crate::ssm::HybridSsmTransformerRecipe::new_zamba2);
                Some(std::sync::LazyLock::force(&ZAMBA2))
            }
            "granitehybrid" => {
                static GRANITE_HYBRID: std::sync::LazyLock<crate::ssm::HybridSsmTransformerRecipe> =
                    std::sync::LazyLock::new(
                        crate::ssm::HybridSsmTransformerRecipe::new_granitehybrid,
                    );
                Some(std::sync::LazyLock::force(&GRANITE_HYBRID))
            }
            "lfm2" => {
                static LFM2: std::sync::LazyLock<crate::ssm::HybridSsmTransformerRecipe> =
                    std::sync::LazyLock::new(crate::ssm::HybridSsmTransformerRecipe::new_lfm2);
                Some(std::sync::LazyLock::force(&LFM2))
            }
            "lfm2moe" => {
                static LFM2_MOE: std::sync::LazyLock<crate::ssm::Lfm2MoeRecipe> =
                    std::sync::LazyLock::new(crate::ssm::Lfm2MoeRecipe::new);
                Some(std::sync::LazyLock::force(&LFM2_MOE))
            }
            "granite" => {
                static GRANITE: crate::ssm::GraniteRecipe = crate::ssm::GraniteRecipe::new();
                Some(&GRANITE)
            }
            _ => None,
        }
    }

    /// Returns a list of architecture identifiers compiled into the current binary.
    #[must_use]
    pub const fn supported_architectures() -> &'static [&'static str] {
        const ARCHS: &[&str] = &[
            #[cfg(feature = "jero-arch-synthetic")]
            "synthetic",
            #[cfg(feature = "jero-arch-llama")]
            "llama",
            #[cfg(feature = "jero-arch-llama")]
            "llama4",
            #[cfg(any(feature = "jero-kernels-rvv", feature = "jero-arch-llama"))]
            "qwen2",
            #[cfg(feature = "jero-kernels-rvv")]
            "qwen3",
            #[cfg(feature = "jero-kernels-rvv")]
            "qwen3moe",
            #[cfg(feature = "jero-kernels-rvv")]
            "deepseek2",
            #[cfg(feature = "jero-kernels-rvv")]
            "qwen35",
            #[cfg(feature = "jero-kernels-rvv")]
            "qwen35moe",
            #[cfg(feature = "jero-kernels-rvv")]
            "mamba",
            #[cfg(feature = "jero-kernels-rvv")]
            "mamba2",
            #[cfg(feature = "jero-kernels-rvv")]
            "rwkv6",
            #[cfg(feature = "jero-kernels-rvv")]
            "rwkv7",
            #[cfg(feature = "jero-kernels-rvv")]
            "jamba",
            #[cfg(feature = "jero-kernels-rvv")]
            "zamba2",
            #[cfg(feature = "jero-kernels-rvv")]
            "granitehybrid",
            #[cfg(feature = "jero-kernels-rvv")]
            "lfm2",
            #[cfg(feature = "jero-kernels-rvv")]
            "lfm2moe",
            #[cfg(feature = "jero-kernels-rvv")]
            "granite",
            #[cfg(feature = "jero-arch-mistral")]
            "mistral",
            #[cfg(feature = "jero-arch-mistral")]
            "mistral3",
            #[cfg(feature = "jero-arch-mistral")]
            "mixtral",
            #[cfg(feature = "jero-arch-phi3")]
            "phi3",
            #[cfg(feature = "jero-arch-phi3")]
            "phi4",
            #[cfg(feature = "jero-arch-bitnet")]
            "bitnet",
            #[cfg(feature = "jero-arch-bitnet")]
            "bitnet-b1.58",
            #[cfg(feature = "jero-arch-bitnet")]
            "bitnet-25",
            #[cfg(feature = "jero-arch-dory")]
            "dory",
            #[cfg(feature = "jero-arch-gemma")]
            "gemma2",
            #[cfg(feature = "jero-arch-gemma")]
            "gemma3",
            #[cfg(feature = "jero-arch-gemma")]
            "gemma4",
            #[cfg(feature = "jero-arch-gemma")]
            "functiongemma",
            #[cfg(feature = "jero-arch-modernbert")]
            "modernbert",
            #[cfg(feature = "jero-arch-modernbert")]
            "bert",
            #[cfg(feature = "jero-arch-modernbert")]
            "deberta-v3",
        ];
        ARCHS
    }

    /// Hydrates a [`jero_arch_llama::LlamaRecipe`] from extracted [`GgufHyperparams`], mutating
    /// only public fields (`"llama"`, `"qwen2"`, `"llama4"`).
    #[cfg(feature = "jero-arch-llama")]
    #[must_use]
    pub fn hydrate_llama_recipe(params: &GgufHyperparams) -> Option<jero_arch_llama::LlamaRecipe> {
        let mut recipe = match params.architecture.as_str() {
            "llama" => jero_arch_llama::LlamaRecipe::new(),
            "qwen2" => {
                let mut r = jero_arch_llama::LlamaRecipe::new();
                r.architecture = "qwen2";
                r.rms_eps = 1e-6;
                r.rope_theta = 1_000_000.0;
                r
            }
            "llama4" => jero_arch_llama::LlamaRecipe::new_llama4(),
            _ => return None,
        };

        if params.block_count > 0 {
            recipe.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            recipe.ffn_dim = params.feed_forward_length;
        }
        if params.head_count > 0 {
            recipe.n_heads_q = params.head_count;
        }
        if params.head_count_kv > 0 {
            recipe.n_heads_kv = params.head_count_kv;
        }
        if params.key_length > 0 {
            recipe.head_dim = params.key_length;
        } else if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = recipe.hidden_dim / recipe.n_heads_q;
        }
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = params.rope_freq_base {
            recipe.rope_theta = freq;
        }
        if let Some(rd) = params.rope_dim_count
            && rd > 0
        {
            recipe.rotary_dim = Some(rd as usize);
        }
        if let Some(scaling_type) = params.rope_scaling_type.as_deref()
            && let Some(factor) = params.rope_scaling_factor
            && factor.is_finite()
            && factor > 0.0
        {
            match scaling_type {
                "linear" => {
                    recipe.rope_scaling =
                        Some(jero_arch_llama::RopeScalingConfig::Linear { factor });
                }
                "llama3" => {
                    recipe.rope_scaling = Some(jero_arch_llama::RopeScalingConfig::Llama3Smooth {
                        factor,
                        low_freq_factor: 1.0,
                        high_freq_factor: 4.0,
                        original_max_position_embeddings: params
                            .rope_orig_ctx_len
                            .filter(|&v| v > 0)
                            .map_or(8192, |v| v as usize),
                    });
                }
                _ => {}
            }
        }
        Some(recipe)
    }

    /// Instantiates a boxed [`ArchRecipe`] dynamically configured from [`GgufHyperparams`] and
    /// optional [`ModelMetadata`] (for architecture-specific extensions such as `dory`).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn instantiate_from_hyperparams(
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
    ) -> Option<Box<dyn ArchRecipe>> {
        let arch = params.architecture.as_str();

        #[cfg(feature = "jero-arch-synthetic")]
        if arch == "synthetic" {
            return Some(Box::new(jero_arch_synthetic::SyntheticRecipe::new()));
        }

        #[cfg(feature = "jero-kernels-rvv")]
        if let Some(recipe) = Self::hydrate_qwen_deepseek_recipe(params, metadata) {
            return Some(recipe);
        }

        #[cfg(feature = "jero-arch-llama")]
        if let Some(recipe) = Self::hydrate_llama_recipe(params) {
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-mistral")]
        if arch == "mistral" || arch == "mistral3" || arch == "mixtral" {
            let mut recipe = if arch == "mixtral" {
                jero_arch_mistral::MistralRecipe::new_mixtral()
            } else {
                let mut r = jero_arch_mistral::MistralRecipe::new();
                if arch == "mistral3" {
                    r.architecture = "mistral3";
                }
                r
            };
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            }
            if params.head_count > 0 {
                recipe.n_heads_q = params.head_count;
            }
            if params.head_count_kv > 0 {
                recipe.n_heads_kv = params.head_count_kv;
            }
            if params.key_length > 0 {
                recipe.head_dim = params.key_length;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            if let Some(eps) = params.rms_norm_eps {
                recipe.rms_eps = eps;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_theta = freq;
            }
            if let Some(sw) = params.sliding_window {
                recipe.sliding_window = Some(sw as usize);
            }
            if let Some(scaling_type) = params.rope_scaling_type.as_deref()
                && let Some(factor) = params.rope_scaling_factor
                && factor.is_finite()
                && factor > 0.0
            {
                match scaling_type {
                    "linear" => {
                        recipe.rope_scaling =
                            Some(jero_arch_mistral::RopeScalingConfig::Linear { factor });
                    }
                    "llama3" => {
                        recipe.rope_scaling =
                            Some(jero_arch_mistral::RopeScalingConfig::Llama3Smooth {
                                factor,
                                low_freq_factor: 1.0,
                                high_freq_factor: 4.0,
                                original_max_position_embeddings: params
                                    .rope_orig_ctx_len
                                    .filter(|&v| v > 0)
                                    .map_or(8192, |v| v as usize),
                            });
                    }
                    _ => {}
                }
            }
            if let Some(ec) = params.expert_count {
                recipe.num_experts = Some(ec);
            }
            if let Some(euc) = params.expert_used_count {
                recipe.num_experts_per_tok = Some(euc);
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-phi3")]
        if arch == "phi3" || arch == "phi4" {
            let mut recipe = if arch == "phi4" {
                jero_arch_phi3::Phi3Recipe::new_phi4()
            } else {
                jero_arch_phi3::Phi3Recipe::new()
            };
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            }
            if params.head_count > 0 {
                recipe.n_heads_q = params.head_count;
            }
            if params.head_count_kv > 0 {
                recipe.n_heads_kv = params.head_count_kv;
            }
            if params.key_length > 0 {
                recipe.head_dim = params.key_length;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            if let Some(eps) = params.rms_norm_eps {
                recipe.rms_eps = eps;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_theta = freq;
            }
            if let Some(rd) = params.rope_dim_count
                && rd > 0
            {
                recipe.rotary_dim = Some(rd as usize);
            }
            if let Some(sw) = params.sliding_window {
                recipe.sliding_window = Some(sw as usize);
            }
            if let Some(orig) = params.rope_orig_ctx_len
                && orig > 0
            {
                recipe.original_max_position_embeddings = orig as usize;
            } else if params.context_length > 0 {
                recipe.original_max_position_embeddings = params.context_length;
            }
            if params.rope_scaling_type.as_deref() == Some("linear")
                && let Some(factor) = params.rope_scaling_factor
                && factor.is_finite()
                && factor > 0.0
            {
                recipe.rope_scaling = jero_arch_phi3::RopeScalingConfig::Linear { factor };
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-bitnet")]
        if arch == "bitnet" || arch == "bitnet-b1.58" || arch == "bitnet-25" {
            let mut recipe = match arch {
                "bitnet-b1.58" => jero_arch_bitnet::BitNetRecipe::new_b158(),
                "bitnet-25" => jero_arch_bitnet::BitNetRecipe::new_25(),
                _ => jero_arch_bitnet::BitNetRecipe::new(),
            };
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            }
            if params.head_count > 0 {
                recipe.n_heads_q = params.head_count;
            }
            if params.head_count_kv > 0 {
                recipe.n_heads_kv = params.head_count_kv;
            }
            if params.key_length > 0 {
                recipe.head_dim = params.key_length;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            if let Some(eps) = params.rms_norm_eps {
                recipe.rms_eps = eps;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_theta = freq;
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-dory")]
        if arch == "dory" {
            let mut recipe = jero_arch_dory::DoryRecipe::new();
            if let Some(dory) = metadata.and_then(ModelMetadata::dory) {
                recipe.n_input_layers = dory.input_layer_count();
                recipe.n_recurrent_layers = dory.recurrent_layer_count();
                recipe.n_output_layers = dory.output_layer_count();
                recipe.n_loops = dory.num_loops();
                recipe.kv_cache_mode = match dory.recurrent_kv_cache_mode() {
                    jero_core::DoryKvCacheMode::PerLoop => jero_arch_dory::DoryKvCacheMode::PerLoop,
                    jero_core::DoryKvCacheMode::LastLoop => {
                        jero_arch_dory::DoryKvCacheMode::LastLoop
                    }
                };
                if !dory.layer_pattern().is_empty() {
                    recipe = recipe.with_layer_pattern_owned(dory.layer_pattern().to_owned());
                }
                let profile = dory.rope_profile().trim();
                if !profile.is_empty() {
                    recipe = recipe.with_rope_profile(profile.to_owned());
                }
                if dory.rope_freq_base_2().is_finite() && dory.rope_freq_base_2() > 0.0 {
                    recipe.rope_theta_swa = dory.rope_freq_base_2();
                }
                if dory.init_std().is_finite() && dory.init_std() > 0.0 {
                    recipe.init_std = dory.init_std();
                }
                if dory.alpha_init().is_finite() {
                    recipe.alpha_init = dory.alpha_init();
                }
                if dory.gate_init().is_finite() {
                    recipe.gate_init = dory.gate_init();
                }
            }
            if let Some(ls) = metadata
                .and_then(ModelMetadata::logit_scale)
                .filter(|v| v.is_finite() && *v > 0.0)
            {
                recipe.logit_scale = Some(ls);
            }
            if let Ok(dim) = u32::try_from(params.embedding_length)
                && dim > 0
            {
                recipe.hidden_dim = dim;
            }
            if let Ok(ffn) = u32::try_from(params.feed_forward_length)
                && ffn > 0
            {
                recipe.ffn_dim = ffn;
            }
            if let Ok(nh) = u32::try_from(params.head_count)
                && nh > 0
            {
                recipe.num_heads = nh;
            }
            if let Ok(nkv) = u32::try_from(params.head_count_kv)
                && nkv > 0
            {
                recipe.num_kv_heads = nkv;
            }
            if let Some(explicit_kl) = params.explicit_key_length
                && let Ok(kl) = u32::try_from(explicit_kl)
                && kl > 0
            {
                recipe.head_dim = kl;
            } else if params.embedding_length != 2560
                && let Ok(kl) = u32::try_from(params.key_length)
                && kl > 0
            {
                recipe.head_dim = kl;
            }
            if let Some(sw) = params.sliding_window
                && sw > 0
            {
                recipe.sliding_window = sw;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_theta_full = freq;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-gemma")]
        if arch == "gemma2" || arch == "gemma3" || arch == "functiongemma" {
            let mut recipe = match arch {
                "gemma2" => crate::gemma::GemmaRecipe::new_gemma2(),
                "functiongemma" => crate::gemma::GemmaRecipe::new_functiongemma(),
                _ => crate::gemma::GemmaRecipe::new_gemma3(),
            };
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            }
            if params.head_count > 0 {
                recipe.n_heads_q = params.head_count;
            }
            if params.head_count_kv > 0 {
                recipe.n_heads_kv = params.head_count_kv;
            }
            if let Some(kl) = params.explicit_key_length
                && kl > 0
            {
                recipe.head_dim = kl;
            } else if params.key_length > 0 && params.head_count > 0 {
                recipe.head_dim = params.key_length;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            if let Some(eps) = params.rms_norm_eps {
                recipe.rms_eps = eps;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_freq_base_global = freq;
                if arch == "gemma2" {
                    recipe.rope_freq_base_local = freq;
                }
            }
            if let Some(sw) = params.sliding_window
                && sw > 0
            {
                recipe.sliding_window = sw as usize;
            }
            if let Some(cap) = params.attn_logit_softcapping
                && cap.is_finite()
                && cap > 0.0
            {
                recipe.attn_logit_softcap = Some(cap);
            }
            if let Some(cap) = params.final_logit_softcapping
                && cap.is_finite()
                && cap > 0.0
            {
                recipe.final_logit_softcap = Some(cap);
            }
            return Some(Box::new(recipe));
        } else if arch == "gemma4" {
            let mut recipe = crate::gemma::Gemma4Recipe::new();
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            }
            if params.head_count > 0 {
                recipe.n_heads_q = params.head_count;
            }
            if params.head_count_kv > 0 {
                recipe.n_heads_kv = params.head_count_kv;
            }
            if let Some(kl) = params.explicit_key_length
                && kl > 0
            {
                recipe.head_dim = kl;
            } else if params.key_length > 0 && params.head_count > 0 {
                recipe.head_dim = params.key_length;
            }
            if params.vocab_size > 0 {
                recipe.vocab_size = params.vocab_size;
            }
            if let Some(eps) = params.rms_norm_eps {
                recipe.rms_eps = eps;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_freq_base_global = freq;
            }
            if let Some(sw) = params.sliding_window
                && sw > 0
            {
                recipe.sliding_window = sw as usize;
            }
            if let Some(ec) = params.expert_count {
                recipe.num_experts = Some(ec);
            }
            if let Some(euc) = params.expert_used_count {
                recipe.num_experts_per_tok = Some(euc);
            }
            if let Some(meta) = metadata {
                if let Some(gate) = find_tensor(meta.tensors(), "blk.0.per_layer_inp_gate.weight")
                    && gate.shape().len() == 2
                    && recipe.hidden_dim > 0
                {
                    let (d0, d1) = (gate.shape()[0], gate.shape()[1]);
                    if d0 == recipe.hidden_dim && d1 > 0 {
                        recipe.ple_dim = Some(d1);
                    } else if d1 == recipe.hidden_dim && d0 > 0 {
                        recipe.ple_dim = Some(d0);
                    }
                } else if find_tensor(meta.tensors(), "per_layer_token_embd.weight").is_some() {
                    recipe.ple_dim = Some(256);
                }
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-modernbert")]
        if arch == "deberta-v3" {
            let mut recipe = crate::forward::DebertaV3Recipe::from_metadata(metadata);
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.head_count > 0 {
                recipe.n_heads = params.head_count.min(recipe.hidden_dim).max(1);
            }
            if recipe.n_heads > 0 && recipe.hidden_dim >= recipe.n_heads {
                recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);
            }
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-modernbert")]
        if let Some(mut recipe) = crate::modernbert::ModernBertRecipe::from_metadata(arch, metadata)
        {
            if params.block_count > 0 {
                recipe.num_layers = params.block_count;
            }
            if params.embedding_length > 0 {
                recipe.hidden_dim = params.embedding_length;
            }
            if params.feed_forward_length > 0 {
                recipe.ffn_dim = params.feed_forward_length;
            } else if params.embedding_length > 0 {
                recipe.ffn_dim = params.embedding_length.saturating_mul(2);
            }
            if params.head_count > 0 {
                recipe.n_heads = params.head_count.min(recipe.hidden_dim).max(1);
            }
            if recipe.n_heads > 0 && recipe.hidden_dim >= recipe.n_heads {
                recipe.head_dim = (recipe.hidden_dim / recipe.n_heads).max(1);
            }
            return Some(Box::new(recipe));
        }

        let _ = (arch, metadata);
        None
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn infer_qwen_head_dim(params: &GgufHyperparams, default: usize) -> usize {
        if params.key_length > 0 {
            params.key_length
        } else if params.head_count > 0 && params.embedding_length >= params.head_count {
            (params.embedding_length / params.head_count).max(1)
        } else {
            default
        }
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_deepseek2_recipe(
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
    ) -> Box<dyn ArchRecipe> {
        let mut recipe = crate::qwen_deepseek::DeepSeek2Recipe::new();
        if params.block_count > 0 {
            recipe.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            recipe.ffn_dim = params.feed_forward_length;
        }
        if params.head_count > 0 {
            recipe.n_heads = params.head_count;
        }
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = params.rope_freq_base {
            recipe.rope_theta = freq;
        }
        if let Some(rope) = params.rope_dim_count
            && rope > 0
        {
            recipe.mla.qk_rope_head_dim = rope as usize;
        }
        if let Some(vl) = params.value_length
            && vl > 0
        {
            recipe.mla.v_head_dim = vl as usize;
        }
        if let Some(meta) = metadata {
            if let Some(rank) = meta.kv_lora_rank() {
                recipe.mla.kv_lora_rank = rank;
            }
            if meta.q_lora_rank().is_some() {
                recipe.mla.q_lora_rank = meta.q_lora_rank();
            }
            if let Some(nope) = meta.qk_nope_head_dim() {
                recipe.mla.qk_nope_head_dim = nope;
            }
            if let Some(rope) = meta.qk_rope_head_dim() {
                recipe.mla.qk_rope_head_dim = rope;
            }
            if let Some(v_dim) = meta.v_head_dim() {
                recipe.mla.v_head_dim = v_dim;
            }
            if let Some(ec) = meta.expert_count() {
                recipe.num_experts = ec;
            }
            if let Some(euc) = meta.expert_used_count() {
                recipe.num_experts_per_tok = euc;
            }
            if let Some(esc) = meta.expert_shared_count() {
                recipe.num_shared_experts = esc;
            }
        }
        Box::new(recipe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_qwen35_recipe(
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
        is_moe: bool,
    ) -> Box<dyn ArchRecipe> {
        let mut base = crate::qwen_deepseek::Qwen35Recipe::new();
        if params.block_count > 0 {
            base.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            base.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            base.ffn_dim = params.feed_forward_length;
        }
        if params.head_count > 0 {
            base.n_heads_q = params.head_count;
        }
        if params.head_count_kv > 0 {
            base.n_heads_kv = params.head_count_kv;
        }
        base.head_dim = Self::infer_qwen_head_dim(params, base.head_dim);
        if params.vocab_size > 0 {
            base.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            base.rms_eps = eps;
        }
        if let Some(freq) = params.rope_freq_base {
            base.rope_theta = freq;
        }
        if !is_moe {
            return Box::new(base);
        }
        let mut moe = crate::qwen_deepseek::Qwen35MoeRecipe::new();
        moe.base = base;
        if let Some(meta) = metadata {
            if let Some(ec) = meta.expert_count() {
                moe.num_experts = ec;
            }
            if let Some(euc) = meta.expert_used_count() {
                moe.num_experts_per_tok = euc;
            }
            if let Some(esc) = meta.expert_shared_count() {
                moe.num_shared_experts = esc;
            }
        }
        Box::new(moe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_mamba_recipe(arch: &str, params: &GgufHyperparams) -> Box<dyn ArchRecipe> {
        let mut recipe = if arch == "mamba" {
            crate::ssm::Mamba2Recipe::new_mamba()
        } else {
            crate::ssm::Mamba2Recipe::new()
        };
        if params.block_count > 0 {
            recipe.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if let Some(d_inner) = params.ssm_inner_size {
            recipe.d_inner = d_inner;
        } else if params.embedding_length > 0 {
            recipe.d_inner = params.embedding_length.saturating_mul(2);
        }
        if let Some(d_conv) = params.ssm_conv_kernel {
            recipe.d_conv = d_conv;
        }
        if let Some(d_state) = params.ssm_state_size {
            recipe.d_state = d_state;
        }
        if let Some(rank) = params.ssm_time_step_rank {
            recipe.n_heads = rank;
        } else if params.head_count > 0 {
            recipe.n_heads = params.head_count;
        } else {
            recipe.n_heads = recipe.n_heads.min(recipe.d_inner.max(1)).max(1);
        }
        if let Some(groups) = params.ssm_group_count {
            recipe.n_groups = groups;
        }
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        Box::new(recipe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_rwkv_recipe(arch: &str, params: &GgufHyperparams) -> Box<dyn ArchRecipe> {
        let mut recipe = if arch == "rwkv6" {
            crate::ssm::Rwkv7Recipe::new_rwkv6()
        } else {
            crate::ssm::Rwkv7Recipe::new()
        };
        if params.block_count > 0 {
            recipe.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            recipe.ffn_dim = params.feed_forward_length;
        }
        if let Some(hs) = params.wkv_head_size {
            recipe.head_size = hs;
        }
        if params.head_count > 0 {
            recipe.n_heads = params.head_count;
        } else {
            recipe.n_heads = (recipe.hidden_dim / recipe.head_size.max(1)).max(1);
        }
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        Box::new(recipe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_hybrid_ssm_base(
        arch: &str,
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
    ) -> crate::ssm::HybridSsmTransformerRecipe {
        let mut recipe = match arch {
            "jamba" => crate::ssm::HybridSsmTransformerRecipe::new_jamba(),
            "zamba2" => crate::ssm::HybridSsmTransformerRecipe::new_zamba2(),
            "granitehybrid" => crate::ssm::HybridSsmTransformerRecipe::new_granitehybrid(),
            "lfm2moe" => {
                let mut base = crate::ssm::HybridSsmTransformerRecipe::new_lfm2();
                base.architecture_id = "lfm2moe";
                base
            }
            _ => crate::ssm::HybridSsmTransformerRecipe::new_lfm2(),
        };
        let num_layers = if params.block_count > 0 {
            params.block_count
        } else {
            recipe.layer_schedule.len()
        };
        recipe.layer_schedule =
            crate::ssm::resolve_hybrid_layer_schedule_for_metadata(arch, num_layers, metadata);
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            recipe.ffn_dim = params.feed_forward_length;
        }
        if params.head_count > 0 {
            recipe.n_heads_q = params.head_count;
        }
        if params.head_count_kv > 0 {
            recipe.n_heads_kv = params.head_count_kv;
        } else if let Some(meta) = metadata
            && let Some(first_kv) = meta
                .head_count_kv_per_layer()
                .iter()
                .copied()
                .find(|&v| v > 0)
        {
            recipe.n_heads_kv = first_kv;
        }
        recipe.n_heads_kv = recipe.n_heads_kv.min(recipe.n_heads_q).max(1);
        recipe.head_dim = Self::infer_qwen_head_dim(params, recipe.head_dim);
        if let Some(d_inner) = params.ssm_inner_size {
            recipe.d_inner = d_inner;
        } else if params.embedding_length > 0 && !matches!(arch, "lfm2" | "lfm2moe") {
            recipe.d_inner = params.embedding_length.saturating_mul(2);
        }
        if let Some(d_conv) = params.ssm_conv_kernel {
            recipe.d_conv = d_conv;
        }
        if let Some(d_state) = params.ssm_state_size {
            recipe.d_state = d_state;
        }
        if let Some(groups) = params.ssm_group_count {
            recipe.n_groups = groups;
        }
        if let Some(l_cache) = params.shortconv_l_cache {
            recipe.shortconv_l_cache = l_cache;
        }
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = params.rope_freq_base {
            recipe.rope_theta = freq;
        }
        if let Some(es) = params.embedding_scale {
            recipe.embedding_scale = es;
        }
        if let Some(rs) = params.residual_scale {
            recipe.residual_scale = rs;
        }
        if params.attention_scale.is_some() {
            recipe.attention_scale = params.attention_scale;
        }
        if let Some(ls) = params.logit_scale {
            recipe.logit_scale = ls;
        }
        recipe
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_lfm2moe_recipe(
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
    ) -> Box<dyn ArchRecipe> {
        let mut recipe = crate::ssm::Lfm2MoeRecipe::new();
        recipe.base = Self::hydrate_hybrid_ssm_base("lfm2moe", params, metadata);
        if let Some(ec) = params.expert_count {
            recipe.num_experts = ec;
        }
        if let Some(euc) = params.expert_used_count {
            recipe.num_experts_per_tok = euc;
        }
        if let Some(eff) = params.expert_feed_forward_length {
            recipe.expert_ffn_dim = eff;
        }
        Box::new(recipe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_granite_recipe(params: &GgufHyperparams) -> Box<dyn ArchRecipe> {
        let mut recipe = crate::ssm::GraniteRecipe::new();
        if params.block_count > 0 {
            recipe.num_layers = params.block_count;
        }
        if params.embedding_length > 0 {
            recipe.hidden_dim = params.embedding_length;
        }
        if params.feed_forward_length > 0 {
            recipe.ffn_dim = params.feed_forward_length;
        }
        if params.head_count > 0 {
            recipe.n_heads_q = params.head_count;
        }
        if params.head_count_kv > 0 {
            recipe.n_heads_kv = params.head_count_kv;
        }
        recipe.n_heads_kv = recipe.n_heads_kv.min(recipe.n_heads_q).max(1);
        recipe.head_dim = Self::infer_qwen_head_dim(params, recipe.head_dim);
        if params.vocab_size > 0 {
            recipe.vocab_size = params.vocab_size;
        }
        if let Some(eps) = params.rms_norm_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = params.rope_freq_base {
            recipe.rope_theta = freq;
        }
        if let Some(es) = params.embedding_scale {
            recipe.embedding_scale = es;
        }
        if let Some(rs) = params.residual_scale {
            recipe.residual_scale = rs;
        }
        if params.attention_scale.is_some() {
            recipe.attention_scale = params.attention_scale;
        }
        if let Some(ls) = params.logit_scale {
            recipe.logit_scale = ls;
        }
        Box::new(recipe)
    }

    #[cfg(feature = "jero-kernels-rvv")]
    fn hydrate_qwen_deepseek_recipe(
        params: &GgufHyperparams,
        metadata: Option<&ModelMetadata>,
    ) -> Option<Box<dyn ArchRecipe>> {
        macro_rules! apply_qwen_dense_fields {
            ($recipe:expr) => {{
                if params.block_count > 0 {
                    $recipe.num_layers = params.block_count;
                }
                if params.embedding_length > 0 {
                    $recipe.hidden_dim = params.embedding_length;
                }
                if params.feed_forward_length > 0 {
                    $recipe.ffn_dim = params.feed_forward_length;
                }
                if params.head_count > 0 {
                    $recipe.n_heads_q = params.head_count;
                }
                if params.head_count_kv > 0 {
                    $recipe.n_heads_kv = params.head_count_kv;
                }
                $recipe.head_dim = Self::infer_qwen_head_dim(params, $recipe.head_dim);
                if params.vocab_size > 0 {
                    $recipe.vocab_size = params.vocab_size;
                }
                if let Some(eps) = params.rms_norm_eps {
                    $recipe.rms_eps = eps;
                }
                if let Some(freq) = params.rope_freq_base {
                    $recipe.rope_theta = freq;
                }
            }};
        }

        match params.architecture.as_str() {
            "qwen2" => {
                let mut r = crate::qwen_deepseek::Qwen2Recipe::new();
                apply_qwen_dense_fields!(r);
                Some(Box::new(r))
            }
            "qwen3" => {
                let mut r = crate::qwen_deepseek::Qwen3Recipe::new();
                apply_qwen_dense_fields!(r);
                Some(Box::new(r))
            }
            "qwen3moe" => {
                let mut r = crate::qwen_deepseek::Qwen3MoeRecipe::new_30b_a3b();
                apply_qwen_dense_fields!(r.base);
                if let Some(meta) = metadata {
                    if let Some(ec) = meta.expert_count() {
                        r.num_experts = ec;
                    }
                    if let Some(euc) = meta.expert_used_count() {
                        r.num_experts_per_tok = euc;
                    }
                }
                Some(Box::new(r))
            }
            "deepseek2" => Some(Self::hydrate_deepseek2_recipe(params, metadata)),
            "qwen35" => Some(Self::hydrate_qwen35_recipe(params, metadata, false)),
            "qwen35moe" => Some(Self::hydrate_qwen35_recipe(params, metadata, true)),
            "mamba" | "mamba2" => Some(Self::hydrate_mamba_recipe(&params.architecture, params)),
            "rwkv6" | "rwkv7" => Some(Self::hydrate_rwkv_recipe(&params.architecture, params)),
            "jamba" | "zamba2" | "granitehybrid" | "lfm2" => Some(Box::new(
                Self::hydrate_hybrid_ssm_base(&params.architecture, params, metadata),
            )),
            "lfm2moe" => Some(Self::hydrate_lfm2moe_recipe(params, metadata)),
            "granite" => Some(Self::hydrate_granite_recipe(params)),
            _ => None,
        }
    }

    /// Instantiates a boxed [`ArchRecipe`] dynamically hydrated from [`ModelMetadata`].
    #[must_use]
    pub fn instantiate_recipe(metadata: &ModelMetadata) -> Option<Box<dyn ArchRecipe>> {
        let params = GgufHyperparams::from_metadata(metadata);
        Self::instantiate_from_hyperparams(&params, Some(metadata))
    }

    /// Instantiates a boxed [`ArchRecipe`] dynamically hydrated from a parsed [`GgufFile`].
    #[must_use]
    pub fn instantiate_recipe_from_gguf(gguf: &GgufFile<'_>) -> Option<Box<dyn ArchRecipe>> {
        let params = GgufHyperparams::from_gguf(gguf);
        #[cfg(any(feature = "jero-arch-phi3", feature = "jero-arch-mistral"))]
        let arch = params.architecture.as_str();

        #[cfg(feature = "jero-arch-phi3")]
        if (arch == "phi3" || arch == "phi4")
            && (GgufHyperparams::lookup_arch_value(gguf, arch, "rope.scaling.short_factor")
                .is_some()
                || GgufHyperparams::lookup_arch_value(gguf, arch, "rope.scaling.long_factor")
                    .is_some())
            && let Ok(recipe) = jero_arch_phi3::Phi3Recipe::from_gguf(gguf)
        {
            return Some(Box::new(recipe));
        }

        #[cfg(feature = "jero-arch-mistral")]
        if matches!(arch, "mistral" | "mistral3" | "mixtral")
            && let Ok(recipe) = jero_arch_mistral::MistralRecipe::from_gguf(gguf)
        {
            return Some(Box::new(recipe));
        }

        Self::instantiate_from_hyperparams(&params, Some(gguf.metadata()))
    }
}

fn validate_raw_gguf_hyperparams(gguf: &GgufFile<'_>) -> Result<(), K3Error> {
    let arch = gguf.metadata().architecture();
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "rope.dimension_count")
        && val.as_u32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "{arch}.rope.dimension_count must be a non-negative integer, got invalid GGUF value"
            ),
        });
    }
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "rope.scaling.factor")
        && val.as_f32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{arch}.rope.scaling.factor must be a float"),
        });
    }
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "rope.scaling.orig_ctx_len")
        .or_else(|| {
            GgufHyperparams::lookup_arch_value(gguf, arch, "rope.scaling.original_context_length")
        })
        && val.as_u32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{arch}.rope.scaling.orig_ctx_len must be a non-negative integer"),
        });
    }
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "attention.value_length")
        && val.as_u32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{arch}.attention.value_length must be a non-negative integer"),
        });
    }
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "attn_logit_softcapping")
        && val.as_f32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{arch}.attn_logit_softcapping must be a float"),
        });
    }
    if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, "final_logit_softcapping")
        && val.as_f32().is_none()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{arch}.final_logit_softcapping must be a float"),
        });
    }
    validate_raw_ssm_and_granite_hyperparams(gguf, arch)?;
    #[cfg(feature = "jero-arch-dory")]
    if arch == "dory"
        && let Some(val) = gguf.metadata_value("dory.rope.profile")
    {
        jero_arch_dory::DoryRecipe::parse_rope_profile(val)?;
    }
    Ok(())
}

fn validate_raw_ssm_and_granite_hyperparams(
    gguf: &GgufFile<'_>,
    arch: &str,
) -> Result<(), K3Error> {
    for key in [
        "ssm.conv_kernel",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.time_step_rank",
        "ssm.group_count",
        "shortconv.l_cache",
        "wkv.head_size",
    ] {
        if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, key) {
            let parsed = val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{arch}.{key} must be a positive integer"),
            })?;
            if parsed == 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!("{arch}.{key} must be greater than 0"),
                });
            }
        }
    }
    for key in [
        "embedding_scale",
        "residual_scale",
        "attention.scale",
        "logit_scale",
    ] {
        if let Some(val) = GgufHyperparams::lookup_arch_value(gguf, arch, key) {
            let parsed = val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{arch}.{key} must be a positive float"),
            })?;
            if !parsed.is_finite() || parsed <= 0.0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!("{arch}.{key} ({parsed}) must be finite and greater than 0"),
                });
            }
        }
    }
    Ok(())
}

fn validate_hyperparams_invariants(
    params: &GgufHyperparams,
    metadata: &ModelMetadata,
) -> Result<(), K3Error> {
    let arch = params.architecture.as_str();
    if let Some(rd) = params.rope_dim_count {
        if rd == 0 || rd % 2 != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "rope.dimension_count ({rd}) for '{arch}' must be positive and even"
                ),
            });
        }
        if params.key_length > 0 && (rd as usize) > params.key_length {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "rope.dimension_count ({rd}) exceeds head_dim ({}) for '{arch}'",
                    params.key_length
                ),
            });
        }
    }
    if let Some(factor) = params.rope_scaling_factor
        && (!factor.is_finite() || factor <= 0.0)
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "rope.scaling.factor ({factor}) for '{arch}' must be finite and greater than 0"
            ),
        });
    }
    if let Some(orig) = params.rope_orig_ctx_len
        && orig == 0
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("rope.scaling.orig_ctx_len for '{arch}' must be greater than 0"),
        });
    }
    if let Some(vl) = params.value_length
        && vl == 0
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("attention.value_length for '{arch}' must be greater than 0"),
        });
    }
    if let Some(cap) = params.attn_logit_softcapping
        && (!cap.is_finite() || cap <= 0.0)
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "attn_logit_softcapping ({cap}) for '{arch}' must be finite and greater than 0"
            ),
        });
    }
    if let Some(cap) = params.final_logit_softcapping
        && (!cap.is_finite() || cap <= 0.0)
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "final_logit_softcapping ({cap}) for '{arch}' must be finite and greater than 0"
            ),
        });
    }
    validate_ssm_and_granite_invariants(params, arch)?;
    if arch == "dory"
        && let Some(dory) = metadata.dory()
        && dory.rope_profile().trim().is_empty()
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: "dory.rope.profile must not be empty".to_owned(),
        });
    }
    Ok(())
}

fn validate_ssm_and_granite_invariants(
    params: &GgufHyperparams,
    arch: &str,
) -> Result<(), K3Error> {
    for (key, value) in [
        ("ssm.conv_kernel", params.ssm_conv_kernel),
        ("ssm.inner_size", params.ssm_inner_size),
        ("ssm.state_size", params.ssm_state_size),
        ("ssm.time_step_rank", params.ssm_time_step_rank),
        ("ssm.group_count", params.ssm_group_count),
        ("shortconv.l_cache", params.shortconv_l_cache),
        ("wkv.head_size", params.wkv_head_size),
    ] {
        if value == Some(0) {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{key} for '{arch}' must be greater than 0"),
            });
        }
    }
    for (key, scale) in [
        ("embedding_scale", params.embedding_scale),
        ("residual_scale", params.residual_scale),
        ("attention.scale", params.attention_scale),
        ("logit_scale", params.logit_scale),
    ] {
        if let Some(v) = scale
            && (!v.is_finite() || v <= 0.0)
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{key} ({v}) for '{arch}' must be finite and greater than 0"),
            });
        }
    }
    Ok(())
}

fn validate_recipe_against_metadata(
    recipe: &dyn ArchRecipe,
    params: &GgufHyperparams,
    metadata: &ModelMetadata,
) -> Result<(), K3Error> {
    let arch = metadata.architecture();
    validate_hyperparams_invariants(params, metadata)?;

    for tensor in metadata.tensors() {
        if !recipe.supports_quant(tensor.quant()) {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "quantization format '{}' of tensor '{}' is not supported by recipe '{arch}'",
                    tensor.quant(),
                    tensor.name()
                ),
            });
        }
    }

    if metadata.context_length() == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "model context length for architecture '{arch}' must be greater than 0"
            ),
        });
    }

    #[cfg(feature = "jero-kernels-rvv")]
    crate::qwen_deepseek::validate_qwen_revision_from_probe(recipe.architecture_id(), |name| {
        metadata.find_tensor(name).is_some()
    })?;

    Ok(())
}

/// Verifies that the model's architecture recipe is compiled, dynamically hydrates its GGUF
/// hyperparameters, and validates that every tensor's quantization layout and structural
/// `RoPE` / softcapping invariant is valid for the active backend (ADR-023, US-01 Scenario 1).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if:
/// - The model's `general.architecture` has no compiled recipe in this binary.
/// - Any tensor in the model uses a quantization layout unsupported by the matching recipe.
/// - Any extracted `RoPE`, softcapping, or `value_length` invariant fails validation.
/// - The trained context window is zero.
pub fn verify_model_capabilities(model: &GgufFile<'_>) -> Result<Box<dyn ArchRecipe>, K3Error> {
    validate_raw_gguf_hyperparams(model)?;
    let params = GgufHyperparams::from_gguf(model);
    let arch = model.metadata().architecture();
    let recipe = RecipeRegistry::instantiate_recipe_from_gguf(model).ok_or_else(|| {
        K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "architecture '{arch}' is not compiled into this binary (compiled recipes: {:?})",
                RecipeRegistry::supported_architectures()
            ),
        }
    })?;
    validate_recipe_against_metadata(recipe.as_ref(), &params, model.metadata())?;
    Ok(recipe)
}

/// Verifies that the extracted [`ModelMetadata`] satisfies compile-time recipe and
/// quantization capabilities before allocating or `mlock`ing weight buffers, returning a
/// dynamically hydrated [`ArchRecipe`].
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if capability checks fail.
pub fn verify_metadata_capabilities(
    metadata: &ModelMetadata,
) -> Result<Box<dyn ArchRecipe>, K3Error> {
    let params = GgufHyperparams::from_metadata(metadata);
    let arch = metadata.architecture();
    let recipe = RecipeRegistry::instantiate_recipe(metadata).ok_or_else(|| {
        K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "architecture '{arch}' is not compiled into this binary (compiled recipes: {:?})",
                RecipeRegistry::supported_architectures()
            ),
        }
    })?;
    validate_recipe_against_metadata(recipe.as_ref(), &params, metadata)?;
    Ok(recipe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::QuantType;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    #[test]
    fn test_supported_architectures_list() {
        let archs = RecipeRegistry::supported_architectures();
        let _ = &archs;
        #[cfg(feature = "jero-arch-synthetic")]
        assert!(archs.contains(&"synthetic"));
        #[cfg(feature = "jero-arch-llama")]
        {
            assert!(archs.contains(&"llama"));
            assert!(archs.contains(&"llama4"));
            assert!(archs.contains(&"qwen2"));
        }
        #[cfg(feature = "jero-arch-mistral")]
        {
            assert!(archs.contains(&"mistral"));
            assert!(archs.contains(&"mistral3"));
            assert!(archs.contains(&"mixtral"));
        }
        #[cfg(feature = "jero-arch-phi3")]
        {
            assert!(archs.contains(&"phi3"));
            assert!(archs.contains(&"phi4"));
        }
        #[cfg(feature = "jero-arch-bitnet")]
        {
            assert!(archs.contains(&"bitnet"));
            assert!(archs.contains(&"bitnet-b1.58"));
            assert!(archs.contains(&"bitnet-25"));
        }
        #[cfg(feature = "jero-arch-dory")]
        assert!(archs.contains(&"dory"));
        #[cfg(feature = "jero-arch-gemma")]
        {
            assert!(archs.contains(&"gemma2"));
            assert!(archs.contains(&"gemma3"));
            assert!(archs.contains(&"gemma4"));
            assert!(archs.contains(&"functiongemma"));
        }
        #[cfg(feature = "jero-arch-modernbert")]
        {
            assert!(archs.contains(&"modernbert"));
            assert!(archs.contains(&"bert"));
            assert!(archs.contains(&"deberta-v3"));
        }
        #[cfg(feature = "jero-kernels-rvv")]
        {
            assert!(archs.contains(&"qwen2"));
            assert!(archs.contains(&"qwen3"));
            assert!(archs.contains(&"qwen3moe"));
            assert!(archs.contains(&"deepseek2"));
            assert!(archs.contains(&"qwen35"));
            assert!(archs.contains(&"qwen35moe"));
            assert!(!archs.contains(&"qwen3.8"));
        }
    }

    #[cfg(feature = "jero-arch-modernbert")]
    #[test]
    fn test_modernbert_and_bert_classifier_pass() {
        use jero_core::{
            ExecutionContext, GgufValue, test_helpers::create_synthetic_gguf_v3_with_kvs,
        };

        for arch in ["modernbert", "bert", "deberta-v3"] {
            let tensor_specs = [
                ("token_embd.weight", QuantType::F32, &[8, 16][..]),
                ("blk.0.attn_q.weight", QuantType::F32, &[8, 8][..]),
                ("classifier.weight", QuantType::F32, &[8, 3][..]),
            ];
            let bytes = if cfg!(miri) {
                let k_blocks = format!("{arch}.block_count");
                let k_emb = format!("{arch}.embedding_length");
                let k_ffn = format!("{arch}.feed_forward_length");
                let k_heads = format!("{arch}.attention.head_count");
                let kvs = [
                    (k_blocks.as_str(), GgufValue::Uint32(1)),
                    (k_emb.as_str(), GgufValue::Uint32(8)),
                    (k_ffn.as_str(), GgufValue::Uint32(16)),
                    (k_heads.as_str(), GgufValue::Uint32(2)),
                ];
                create_synthetic_gguf_v3_with_kvs(arch, 64, &kvs, &tensor_specs)
            } else {
                create_synthetic_gguf_v3(arch, 64, &tensor_specs)
            };
            let model = GgufFile::parse(&bytes).expect("parse encoder model");
            let recipe = verify_model_capabilities(&model).expect("encoder capability check");
            assert_eq!(recipe.architecture_id(), arch);
            assert!(!recipe.requires_kv_cache());

            let prompt = [1_u32, 3, 5];
            let mut logits = [0.0_f32; 3];
            let mut ctx =
                ExecutionContext::new(&prompt, 64, &mut logits, &mut []).expect("zero-kv context");
            recipe
                .forward_classify(&mut ctx, &model)
                .expect("forward_classify executes with zero KV cache");
            assert_eq!(ctx.position(), prompt.len());
            assert!(logits.iter().all(|v| v.is_finite()));
        }
    }

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_verify_valid_synthetic_model_succeeds() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("GGUF parse succeeds");

        let recipe = verify_model_capabilities(&model).expect("capability check succeeds");
        assert_eq!(recipe.architecture_id(), "synthetic");
    }

    #[test]
    fn test_verify_unsupported_architecture_rejected() {
        // qwen3.8 is an uncompiled architecture recipe (ADR-023)
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("qwen3.8", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("GGUF parse succeeds");

        let err = verify_model_capabilities(&model).expect_err("must reject qwen3.8");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        assert!(
            err.to_string()
                .contains("architecture 'qwen3.8' is not compiled")
        );
    }

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_verify_unsupported_quantization_rejected() {
        // SyntheticRecipe does not support Q8_0 (only F32, F16, BF16, Q4_K_M)
        let tensor_specs = [("blk.0.attn_q.weight", QuantType::Q8_0, &[32][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("GGUF parse succeeds");

        let err = verify_model_capabilities(&model).expect_err("must reject Q8_0 on synthetic");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        assert!(err.to_string().contains(
            "quantization format 'Q8_0' of tensor 'blk.0.attn_q.weight' is not supported"
        ));
    }

    #[test]
    fn test_recipe_registry_find_llama_and_mistral() {
        #[cfg(feature = "jero-arch-llama")]
        {
            let llama = RecipeRegistry::find_recipe("llama").expect("llama found");
            assert_eq!(llama.architecture_id(), "llama");
            let llama4 = RecipeRegistry::find_recipe("llama4").expect("llama4 found");
            assert_eq!(llama4.architecture_id(), "llama4");
            let qwen2 = RecipeRegistry::find_recipe("qwen2").expect("qwen2 found");
            #[cfg(feature = "jero-kernels-rvv")]
            assert_eq!(qwen2.architecture_id(), "qwen2");
            #[cfg(not(feature = "jero-kernels-rvv"))]
            assert_eq!(qwen2.architecture_id(), "llama");
        }

        #[cfg(feature = "jero-arch-mistral")]
        {
            let dense_recipe = RecipeRegistry::find_recipe("mistral").expect("mistral found");
            assert_eq!(dense_recipe.architecture_id(), "mistral");
            let mistral3 = RecipeRegistry::find_recipe("mistral3").expect("mistral3 found");
            assert_eq!(mistral3.architecture_id(), "mistral");
            let moe_recipe = RecipeRegistry::find_recipe("mixtral").expect("mixtral found");
            assert_eq!(moe_recipe.architecture_id(), "mixtral");
        }

        #[cfg(feature = "jero-arch-phi3")]
        {
            let phi3 = RecipeRegistry::find_recipe("phi3").expect("phi3 found");
            assert_eq!(phi3.architecture_id(), "phi3");
            let phi4 = RecipeRegistry::find_recipe("phi4").expect("phi4 found");
            assert_eq!(phi4.architecture_id(), "phi4");
        }

        #[cfg(feature = "jero-arch-bitnet")]
        {
            let bitnet = RecipeRegistry::find_recipe("bitnet").expect("bitnet found");
            assert_eq!(bitnet.architecture_id(), "bitnet");
            let bitnet_b158 =
                RecipeRegistry::find_recipe("bitnet-b1.58").expect("bitnet-b1.58 found");
            assert_eq!(bitnet_b158.architecture_id(), "bitnet-b1.58");
            let bitnet_25 = RecipeRegistry::find_recipe("bitnet-25").expect("bitnet-25 found");
            assert_eq!(bitnet_25.architecture_id(), "bitnet-25");
        }
    }

    #[cfg(feature = "jero-arch-dory")]
    #[test]
    fn test_recipe_registry_find_dory() {
        let dory = RecipeRegistry::find_recipe("dory").expect("dory recipe registered");
        assert_eq!(dory.architecture_id(), "dory");
        assert!(dory.supports_quant(QuantType::Q4_K_M));
        assert!(dory.supports_quant(QuantType::Q5_K));
        assert!(dory.supports_quant(QuantType::Q6_K));
        assert!(dory.supports_quant(QuantType::BF16));
        assert!(RecipeRegistry::supported_architectures().contains(&"dory"));
    }

    #[cfg(feature = "jero-arch-llama")]
    #[test]
    fn test_gguf_hyperparams_and_recipe_hydration_across_model_scales() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // 1. Qwen2.5-0.5B (hidden=896, ffn=4864, heads=14, kv_heads=2, head_dim=64, vocab=151936)
        let qwen_05b_kvs = [
            ("qwen2.block_count", GgufValue::Uint32(24)),
            ("qwen2.embedding_length", GgufValue::Uint32(896)),
            ("qwen2.feed_forward_length", GgufValue::Uint32(4864)),
            ("qwen2.attention.head_count", GgufValue::Uint32(14)),
            ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
            ("qwen2.vocab_size", GgufValue::Uint32(151_936)),
            ("qwen2.rope.freq_base", GgufValue::Float64(1_000_000.0)),
            (
                "qwen2.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(1e-6),
            ),
        ];
        let qwen_05b_bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &qwen_05b_kvs, &[]);
        let qwen_05b_gguf = GgufFile::parse(&qwen_05b_bytes).expect("parse qwen2 0.5B");
        let params_05b = GgufHyperparams::from_gguf(&qwen_05b_gguf);
        assert_eq!(params_05b.embedding_length, 896);
        assert_eq!(params_05b.feed_forward_length, 4864);
        assert_eq!(params_05b.key_length, 64);
        assert_eq!(params_05b.vocab_size, 151_936);
        let recipe_05b = RecipeRegistry::hydrate_llama_recipe(&params_05b).expect("hydrate 0.5B");
        assert_eq!(recipe_05b.architecture_id(), "qwen2");
        assert_eq!(recipe_05b.num_layers, 24);
        assert_eq!(recipe_05b.hidden_dim, 896);
        assert_eq!(recipe_05b.ffn_dim, 4864);
        assert_eq!(recipe_05b.n_heads_q, 14);
        assert_eq!(recipe_05b.n_heads_kv, 2);
        assert_eq!(recipe_05b.head_dim, 64);
        assert_eq!(recipe_05b.vocab_size, 151_936);
        assert!((recipe_05b.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((recipe_05b.rms_eps - 1e-6).abs() < 1e-10);

        // 2. Qwen2.5-1.5B (#113: hidden=1536, ffn=8960, heads=12, kv_heads=2, head_dim=128, vocab=151936)
        let qwen_15b_kvs = [
            ("qwen2.block_count", GgufValue::Uint32(28)),
            ("qwen2.embedding_length", GgufValue::Uint32(1536)),
            ("qwen2.feed_forward_length", GgufValue::Uint32(8960)),
            ("qwen2.attention.head_count", GgufValue::Uint32(12)),
            ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
            ("qwen2.vocab_size", GgufValue::Uint32(151_936)),
        ];
        let qwen_15b_bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &qwen_15b_kvs, &[]);
        let qwen_15b_gguf = GgufFile::parse(&qwen_15b_bytes).expect("parse qwen2 1.5B");
        let params_15b = GgufHyperparams::from_gguf(&qwen_15b_gguf);
        let recipe_15b = RecipeRegistry::hydrate_llama_recipe(&params_15b).expect("hydrate 1.5B");
        assert_eq!(recipe_15b.hidden_dim, 1536);
        assert_eq!(recipe_15b.ffn_dim, 8960);
        assert_eq!(recipe_15b.n_heads_q, 12);
        assert_eq!(recipe_15b.n_heads_kv, 2);
        assert_eq!(recipe_15b.head_dim, 128);
        assert_eq!(recipe_15b.vocab_size, 151_936);
        let dyn_15b = verify_model_capabilities(&qwen_15b_gguf).expect("verify qwen2 1.5B");
        assert_eq!(dyn_15b.architecture_id(), "qwen2");

        // 3. Llama-3.2-3B with explicit attention.key_length = 128 (where 3072 / 24 == 128,
        // and verify explicit key_length override = 64 when attention heads are decoupled)
        let llama_3b_kvs = [
            ("llama.block_count", GgufValue::Uint32(28)),
            ("llama.embedding_length", GgufValue::Uint32(3072)),
            ("llama.feed_forward_length", GgufValue::Uint32(8192)),
            ("llama.attention.head_count", GgufValue::Uint32(24)),
            ("llama.attention.head_count_kv", GgufValue::Uint32(8)),
            ("llama.attention.key_length", GgufValue::Uint32(64)),
            ("llama.vocab_size", GgufValue::Uint32(128_256)),
            ("llama.rope.freq_base", GgufValue::Float32(500_000.0)),
        ];
        let llama_3b_bytes = create_synthetic_gguf_v3_with_kvs("llama", 32, &llama_3b_kvs, &[]);
        let llama_3b_gguf = GgufFile::parse(&llama_3b_bytes).expect("parse llama 3B");
        let params_3b = GgufHyperparams::from_gguf(&llama_3b_gguf);
        assert_eq!(params_3b.explicit_key_length, Some(64));
        assert_eq!(params_3b.key_length, 64);
        let recipe_3b = RecipeRegistry::hydrate_llama_recipe(&params_3b).expect("hydrate 3B");
        assert_eq!(recipe_3b.hidden_dim, 3072);
        assert_eq!(recipe_3b.ffn_dim, 8192);
        assert_eq!(recipe_3b.head_dim, 64);
        assert_eq!(recipe_3b.vocab_size, 128_256);
        assert!((recipe_3b.rope_theta - 500_000.0).abs() < 1.0);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_issue_137_rope_value_length_softcap_extraction_and_hydration() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // 1. Llama 3.2 (llama3 RoPE scaling + explicit rope.dimension_count + orig_ctx_len)
        #[cfg(feature = "jero-arch-llama")]
        {
            let kvs = [
                ("llama.block_count", GgufValue::Uint32(16)),
                ("llama.embedding_length", GgufValue::Uint32(2048)),
                ("llama.attention.head_count", GgufValue::Uint32(32)),
                ("llama.attention.head_count_kv", GgufValue::Uint32(8)),
                ("llama.attention.key_length", GgufValue::Uint32(64)),
                ("llama.rope.dimension_count", GgufValue::Uint32(64)),
                ("llama.rope.scaling.type", GgufValue::String("llama3")),
                ("llama.rope.scaling.factor", GgufValue::Float32(32.0)),
                ("llama.rope.scaling.orig_ctx_len", GgufValue::Uint32(8192)),
            ];
            let bytes = create_synthetic_gguf_v3_with_kvs("llama", 32, &kvs, &[]);
            let gguf = GgufFile::parse(&bytes).expect("parse llama 3.2");
            let params = GgufHyperparams::from_gguf(&gguf);
            assert_eq!(params.rope_dim_count, Some(64));
            assert_eq!(params.rope_scaling_type.as_deref(), Some("llama3"));
            assert_eq!(params.rope_scaling_factor, Some(32.0));
            assert_eq!(params.rope_orig_ctx_len, Some(8192));

            let recipe = RecipeRegistry::hydrate_llama_recipe(&params).expect("hydrate llama 3.2");
            assert_eq!(recipe.rotary_dim, Some(64));
            assert!(matches!(
                recipe.rope_scaling,
                Some(jero_arch_llama::RopeScalingConfig::Llama3Smooth {
                    factor,
                    original_max_position_embeddings: 8192,
                    ..
                }) if (factor - 32.0).abs() < 1e-6
            ));
            verify_model_capabilities(&gguf).expect("llama 3.2 capabilities pass");
        }

        // 2. Qwen2.5 (rope.dimension_count + yarn scaling + sliding_window)
        {
            let kvs = [
                ("qwen2.block_count", GgufValue::Uint32(24)),
                ("qwen2.embedding_length", GgufValue::Uint32(896)),
                ("qwen2.attention.head_count", GgufValue::Uint32(14)),
                ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
                ("qwen2.rope.dimension_count", GgufValue::Uint32(64)),
                ("qwen2.rope.scaling.type", GgufValue::String("yarn")),
                ("qwen2.rope.scaling.factor", GgufValue::Float32(4.0)),
                ("qwen2.rope.scaling.orig_ctx_len", GgufValue::Uint32(32_768)),
                ("qwen2.attention.sliding_window", GgufValue::Uint32(32_768)),
            ];
            let bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &kvs, &[]);
            let gguf = GgufFile::parse(&bytes).expect("parse qwen2.5");
            let params = GgufHyperparams::from_gguf(&gguf);
            assert_eq!(params.rope_dim_count, Some(64));
            assert_eq!(params.rope_scaling_type.as_deref(), Some("yarn"));
            assert_eq!(params.rope_scaling_factor, Some(4.0));
            assert_eq!(params.rope_orig_ctx_len, Some(32_768));
            assert_eq!(params.sliding_window, Some(32_768));
            verify_model_capabilities(&gguf).expect("qwen2.5 capabilities pass");
        }

        // 3. Phi-3.5 / Phi-4 (partial rope_dim_count + value_length + linear scaling + orig ctx)
        #[cfg(feature = "jero-arch-phi3")]
        {
            for arch in ["phi3", "phi4"] {
                let k_block = format!("{arch}.block_count");
                let k_emb = format!("{arch}.embedding_length");
                let k_hc = format!("{arch}.attention.head_count");
                let k_hkv = format!("{arch}.attention.head_count_kv");
                let k_key_len = format!("{arch}.attention.key_length");
                let k_val_len = format!("{arch}.attention.value_length");
                let k_rd = format!("{arch}.rope.dimension_count");
                let k_rope_type = format!("{arch}.rope.scaling.type");
                let k_rope_factor = format!("{arch}.rope.scaling.factor");
                let k_rope_orig_ctx = format!("{arch}.rope.scaling.original_context_length");
                let k_sw = format!("{arch}.attention.sliding_window");
                let kvs = [
                    (k_block.as_str(), GgufValue::Uint32(32)),
                    (k_emb.as_str(), GgufValue::Uint32(3072)),
                    (k_hc.as_str(), GgufValue::Uint32(32)),
                    (k_hkv.as_str(), GgufValue::Uint32(8)),
                    (k_key_len.as_str(), GgufValue::Uint32(96)),
                    (k_val_len.as_str(), GgufValue::Uint32(96)),
                    (k_rd.as_str(), GgufValue::Uint32(64)),
                    (k_rope_type.as_str(), GgufValue::String("linear")),
                    (k_rope_factor.as_str(), GgufValue::Float32(2.0)),
                    (k_rope_orig_ctx.as_str(), GgufValue::Uint32(4096)),
                    (k_sw.as_str(), GgufValue::Uint32(2048)),
                ];
                let bytes = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs, &[]);
                let gguf = GgufFile::parse(&bytes).expect("parse phi");
                let params = GgufHyperparams::from_gguf(&gguf);
                assert_eq!(params.rope_dim_count, Some(64));
                assert_eq!(params.value_length, Some(96));
                assert_eq!(params.rope_scaling_type.as_deref(), Some("linear"));
                assert_eq!(params.rope_scaling_factor, Some(2.0));
                assert_eq!(params.rope_orig_ctx_len, Some(4096));
                assert_eq!(params.sliding_window, Some(2048));
                verify_model_capabilities(&gguf).expect("phi capabilities pass");
            }
        }

        // 4. Gemma 2 & Gemma 3 (key_length/value_length + attn/final softcapping + sliding_window)
        #[cfg(feature = "jero-arch-gemma")]
        {
            let gemma2_kvs = [
                ("gemma2.block_count", GgufValue::Uint32(26)),
                ("gemma2.embedding_length", GgufValue::Uint32(2304)),
                ("gemma2.attention.head_count", GgufValue::Uint32(8)),
                ("gemma2.attention.head_count_kv", GgufValue::Uint32(4)),
                ("gemma2.attention.key_length", GgufValue::Uint32(256)),
                ("gemma2.attention.value_length", GgufValue::Uint32(256)),
                ("gemma2.rope.dimension_count", GgufValue::Uint32(256)),
                ("gemma2.attn_logit_softcapping", GgufValue::Float32(50.0)),
                ("gemma2.final_logit_softcapping", GgufValue::Float32(30.0)),
                ("gemma2.attention.sliding_window", GgufValue::Uint32(4096)),
            ];
            let gemma2_bytes = create_synthetic_gguf_v3_with_kvs("gemma2", 32, &gemma2_kvs, &[]);
            let gemma2_gguf = GgufFile::parse(&gemma2_bytes).expect("parse gemma2");
            let g2_params = GgufHyperparams::from_gguf(&gemma2_gguf);
            assert_eq!(g2_params.value_length, Some(256));
            assert_eq!(g2_params.attn_logit_softcapping, Some(50.0));
            assert_eq!(g2_params.final_logit_softcapping, Some(30.0));
            assert_eq!(g2_params.sliding_window, Some(4096));
            verify_model_capabilities(&gemma2_gguf).expect("gemma2 capabilities pass");

            let gemma3_kvs = [
                ("gemma3.block_count", GgufValue::Uint32(26)),
                ("gemma3.embedding_length", GgufValue::Uint32(2304)),
                ("gemma3.attention.head_count", GgufValue::Uint32(8)),
                ("gemma3.attention.head_count_kv", GgufValue::Uint32(4)),
                ("gemma3.attention.key_length", GgufValue::Uint32(256)),
                ("gemma3.attention.value_length", GgufValue::Uint32(256)),
                ("gemma3.rope.dimension_count", GgufValue::Uint32(256)),
                ("gemma3.rope.scaling.type", GgufValue::String("linear")),
                ("gemma3.rope.scaling.factor", GgufValue::Float32(8.0)),
                ("gemma3.attention.sliding_window", GgufValue::Uint32(1024)),
            ];
            let gemma3_bytes = create_synthetic_gguf_v3_with_kvs("gemma3", 32, &gemma3_kvs, &[]);
            let gemma3_gguf = GgufFile::parse(&gemma3_bytes).expect("parse gemma3");
            let g3_params = GgufHyperparams::from_gguf(&gemma3_gguf);
            assert_eq!(g3_params.rope_dim_count, Some(256));
            assert_eq!(g3_params.rope_scaling_type.as_deref(), Some("linear"));
            assert_eq!(g3_params.rope_scaling_factor, Some(8.0));
            assert_eq!(g3_params.value_length, Some(256));
            assert_eq!(g3_params.sliding_window, Some(1024));
            verify_model_capabilities(&gemma3_gguf).expect("gemma3 capabilities pass");
        }

        // 5. DeepSeek-R1-Distill (deepseek2 MLA + value_length & rope.dimension_count)
        #[cfg(feature = "jero-kernels-rvv")]
        {
            let ds_kvs = [
                ("deepseek2.block_count", GgufValue::Uint32(4)),
                ("deepseek2.embedding_length", GgufValue::Uint32(128)),
                ("deepseek2.attention.head_count", GgufValue::Uint32(4)),
                ("deepseek2.attention.head_count_kv", GgufValue::Uint32(4)),
                ("deepseek2.attention.kv_lora_rank", GgufValue::Uint32(16)),
                ("deepseek2.attention.value_length", GgufValue::Uint32(32)),
                ("deepseek2.rope.dimension_count", GgufValue::Uint32(8)),
                ("deepseek2.rope.scaling.type", GgufValue::String("yarn")),
                ("deepseek2.rope.scaling.factor", GgufValue::Float32(40.0)),
                (
                    "deepseek2.rope.scaling.orig_ctx_len",
                    GgufValue::Uint32(4096),
                ),
            ];
            let ds_bytes = create_synthetic_gguf_v3_with_kvs("deepseek2", 32, &ds_kvs, &[]);
            let ds_gguf = GgufFile::parse(&ds_bytes).expect("parse deepseek distill");
            let ds_params = GgufHyperparams::from_gguf(&ds_gguf);
            assert_eq!(ds_params.value_length, Some(32));
            assert_eq!(ds_params.rope_dim_count, Some(8));
            assert_eq!(ds_params.rope_scaling_type.as_deref(), Some("yarn"));
            assert_eq!(ds_params.rope_scaling_factor, Some(40.0));
            assert_eq!(ds_params.rope_orig_ctx_len, Some(4096));
            verify_model_capabilities(&ds_gguf).expect("deepseek2 capabilities pass");
        }
    }

    #[test]
    fn test_issue_137_invalid_rope_softcap_and_value_bounds_rejected() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // rope.dimension_count > head_dim (256/4 = 64 < 96)
        let too_wide = create_synthetic_gguf_v3_with_kvs(
            "llama",
            32,
            &[("llama.rope.dimension_count", GgufValue::Uint32(96))],
            &[],
        );
        let gguf = GgufFile::parse(&too_wide).expect("parse header");
        assert!(matches!(
            verify_model_capabilities(&gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // odd rope.dimension_count
        let odd_rope = create_synthetic_gguf_v3_with_kvs(
            "llama",
            32,
            &[("llama.rope.dimension_count", GgufValue::Uint32(63))],
            &[],
        );
        let gguf = GgufFile::parse(&odd_rope).expect("parse header");
        assert!(matches!(
            verify_model_capabilities(&gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // zero or negative rope.dimension_count
        for bad_val in [GgufValue::Uint32(0), GgufValue::Int32(-4)] {
            let bad_rope = create_synthetic_gguf_v3_with_kvs(
                "llama",
                32,
                &[("llama.rope.dimension_count", bad_val)],
                &[],
            );
            let gguf = GgufFile::parse(&bad_rope).expect("parse header");
            assert!(matches!(
                verify_model_capabilities(&gguf),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }

        // non-positive or non-finite rope.scaling.factor
        for bad_factor in [-2.0_f32, 0.0_f32, f32::NAN, f32::INFINITY] {
            let bad_scale = create_synthetic_gguf_v3_with_kvs(
                "llama",
                32,
                &[
                    ("llama.rope.scaling.type", GgufValue::String("linear")),
                    ("llama.rope.scaling.factor", GgufValue::Float32(bad_factor)),
                ],
                &[],
            );
            let gguf = GgufFile::parse(&bad_scale).expect("parse header");
            assert!(matches!(
                verify_model_capabilities(&gguf),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }

        // zero value_length / orig_ctx_len / non-positive softcapping
        for bad_kv in [
            ("llama.attention.value_length", GgufValue::Uint32(0)),
            ("llama.rope.scaling.orig_ctx_len", GgufValue::Uint32(0)),
            ("llama.attn_logit_softcapping", GgufValue::Float32(0.0)),
            (
                "llama.final_logit_softcapping",
                GgufValue::Float32(f32::NAN),
            ),
        ] {
            let bad = create_synthetic_gguf_v3_with_kvs("llama", 32, &[bad_kv], &[]);
            let gguf = GgufFile::parse(&bad).expect("parse header");
            assert!(matches!(
                verify_model_capabilities(&gguf),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }
    }

    #[cfg(feature = "jero-arch-dory")]
    #[test]
    fn test_issue_133_dory_rope_profile_string_and_integer_admitted() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        assert_eq!(
            jero_arch_dory::DoryRecipe::parse_rope_profile(&GgufValue::String("neox"))
                .expect("string profile")
                .as_ref(),
            "neox"
        );
        assert_eq!(
            jero_arch_dory::DoryRecipe::parse_rope_profile(&GgufValue::Uint32(1))
                .expect("u32 profile")
                .as_ref(),
            "profile_1"
        );
        assert_eq!(
            jero_arch_dory::DoryRecipe::parse_rope_profile(&GgufValue::Int32(2))
                .expect("i32 profile")
                .as_ref(),
            "profile_2"
        );
        for bad_val in [
            GgufValue::Bool(true),
            GgufValue::Int32(-1),
            GgufValue::String("  "),
        ] {
            assert!(matches!(
                jero_arch_dory::DoryRecipe::parse_rope_profile(&bad_val),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }

        // Verify full GGUF capability admission accepts integer profile on dory metadata.
        let kvs = [
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.input_layer_count", GgufValue::Uint32(1)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(1)),
            ("dory.output_layer_count", GgufValue::Uint32(1)),
            ("dory.num_loops", GgufValue::Uint32(1)),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("per_loop"),
            ),
            ("dory.layer_pattern", GgufValue::String("S-*-")),
            ("dory.rope.freq_base_2", GgufValue::Float32(10_000.0)),
            ("dory.init_std", GgufValue::Float32(0.02)),
            ("dory.alpha_init", GgufValue::Float32(0.05)),
            ("dory.gate_init", GgufValue::Float32(0.0)),
            ("dory.embedding_length", GgufValue::Uint32(32)),
            ("dory.attention.head_count", GgufValue::Uint32(2)),
            ("dory.attention.head_count_kv", GgufValue::Uint32(2)),
            ("dory.context_length", GgufValue::Uint32(64)),
            ("dory.rope.profile", GgufValue::Uint32(1)),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("dory", 16, &kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse dory compact");
        assert_eq!(
            gguf.metadata()
                .dory()
                .expect("dory metadata")
                .rope_profile(),
            "profile_1"
        );
        let recipe = verify_model_capabilities(&gguf).expect("verify dory compact");
        assert_eq!(recipe.architecture_id(), "dory");
    }

    #[test]
    fn test_issue_120_mixed_quant_admission_phi3_llama_mistral() {
        let mixed_quants = [
            QuantType::Q4_0,
            QuantType::Q4_K_M,
            QuantType::Q5_K,
            QuantType::Q6_K,
            QuantType::Q8_0,
        ];

        for arch in ["phi3", "phi4", "llama", "mistral", "mistral3", "mixtral"] {
            let recipe = RecipeRegistry::find_recipe(arch)
                .unwrap_or_else(|| panic!("recipe registered for {arch}"));
            for quant in mixed_quants {
                assert!(
                    recipe.supports_quant(quant),
                    "expected {arch} recipe to admit mixed checkpoint quant {quant:?}"
                );
            }
        }
        for arch in ["phi3", "llama"] {
            assert!(
                RecipeRegistry::find_recipe(arch)
                    .expect("recipe")
                    .supports_quant(QuantType::Q5_0)
            );
        }
    }

    #[test]
    fn test_ssm_shortconv_wkv_and_granite_hyperparams_extraction() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        let mamba_bytes = create_synthetic_gguf_v3_with_kvs(
            "mamba2",
            64,
            &[
                ("mamba2.block_count", GgufValue::Uint32(4)),
                ("mamba2.embedding_length", GgufValue::Uint32(16)),
                ("mamba2.ssm.inner_size", GgufValue::Uint32(32)),
                ("mamba2.ssm.conv_kernel", GgufValue::Uint32(4)),
                ("mamba2.ssm.state_size", GgufValue::Uint32(8)),
                ("mamba2.ssm.time_step_rank", GgufValue::Uint32(4)),
                ("mamba2.ssm.group_count", GgufValue::Uint32(2)),
                ("mamba2.context_length", GgufValue::Uint32(64)),
            ],
            &[],
        );
        let mamba_gguf = GgufFile::parse(&mamba_bytes).unwrap();
        let mamba_params = GgufHyperparams::from_gguf(&mamba_gguf);
        assert_eq!(mamba_params.ssm_inner_size, Some(32));
        assert_eq!(mamba_params.ssm_conv_kernel, Some(4));
        assert_eq!(mamba_params.ssm_state_size, Some(8));
        assert_eq!(mamba_params.ssm_time_step_rank, Some(4));
        assert_eq!(mamba_params.ssm_group_count, Some(2));
        assert_eq!(
            verify_model_capabilities(&mamba_gguf)
                .expect("mamba2 capability")
                .architecture_id(),
            "mamba2"
        );

        let lfm_bytes = create_synthetic_gguf_v3_with_kvs(
            "lfm2moe",
            64,
            &[
                ("lfm2moe.block_count", GgufValue::Uint32(2)),
                ("lfm2moe.embedding_length", GgufValue::Uint32(8)),
                ("lfm2moe.attention.head_count", GgufValue::Uint32(2)),
                ("lfm2moe.attention.head_count_kv", GgufValue::Uint32(1)),
                ("lfm2moe.shortconv.l_cache", GgufValue::Uint32(3)),
                ("lfm2moe.expert_count", GgufValue::Uint32(8)),
                ("lfm2moe.expert_used_count", GgufValue::Uint32(2)),
                ("lfm2moe.expert.feed_forward_length", GgufValue::Uint32(16)),
                ("lfm2moe.expert_shared_count", GgufValue::Uint32(1)),
                ("lfm2moe.context_length", GgufValue::Uint32(64)),
            ],
            &[],
        );
        let lfm_gguf = GgufFile::parse(&lfm_bytes).unwrap();
        let lfm_params = GgufHyperparams::from_gguf(&lfm_gguf);
        assert_eq!(lfm_params.shortconv_l_cache, Some(3));
        assert_eq!(lfm_params.expert_feed_forward_length, Some(16));
        assert_eq!(lfm_params.expert_shared_count, Some(1));
        assert_eq!(
            verify_model_capabilities(&lfm_gguf)
                .expect("lfm2moe capability")
                .architecture_id(),
            "lfm2moe"
        );

        let rwkv_bytes = create_synthetic_gguf_v3_with_kvs(
            "rwkv7",
            64,
            &[
                ("rwkv7.block_count", GgufValue::Uint32(2)),
                ("rwkv7.embedding_length", GgufValue::Uint32(8)),
                ("rwkv7.wkv.head_size", GgufValue::Uint32(4)),
                ("rwkv7.context_length", GgufValue::Uint32(64)),
            ],
            &[],
        );
        let rwkv_gguf = GgufFile::parse(&rwkv_bytes).unwrap();
        assert_eq!(
            GgufHyperparams::from_gguf(&rwkv_gguf).wkv_head_size,
            Some(4)
        );
        assert_eq!(
            verify_model_capabilities(&rwkv_gguf)
                .expect("rwkv7 capability")
                .architecture_id(),
            "rwkv7"
        );
    }

    #[test]
    fn test_ssm_shortconv_wkv_and_granite_hyperparams_rejection() {
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        for (arch, bad_kv) in [
            ("mamba2", ("mamba2.ssm.conv_kernel", GgufValue::Uint32(0))),
            ("lfm2", ("lfm2.shortconv.l_cache", GgufValue::String("bad"))),
            ("rwkv7", ("rwkv7.wkv.head_size", GgufValue::Uint32(0))),
            ("granite", ("granite.logit_scale", GgufValue::Float32(0.0))),
            (
                "granite",
                ("granite.residual_scale", GgufValue::Float32(-1.0)),
            ),
            (
                "granite",
                ("granite.embedding_scale", GgufValue::Float32(f32::NAN)),
            ),
            (
                "granite",
                ("granite.attention.scale", GgufValue::String("bad")),
            ),
        ] {
            let kvs = [
                (
                    match arch {
                        "mamba2" => "mamba2.block_count",
                        "lfm2" => "lfm2.block_count",
                        "rwkv7" => "rwkv7.block_count",
                        _ => "granite.block_count",
                    },
                    GgufValue::Uint32(2),
                ),
                (
                    match arch {
                        "mamba2" => "mamba2.context_length",
                        "lfm2" => "lfm2.context_length",
                        "rwkv7" => "rwkv7.context_length",
                        _ => "granite.context_length",
                    },
                    GgufValue::Uint32(64),
                ),
                bad_kv,
            ];
            let bytes = create_synthetic_gguf_v3_with_kvs(arch, 64, &kvs, &[]);
            let gguf = GgufFile::parse(&bytes).unwrap();
            assert!(
                matches!(
                    verify_model_capabilities(&gguf),
                    Err(K3Error::CapabilityMismatch { .. })
                ),
                "expected invalid hyperparameter on {arch} to fail closed"
            );
        }
    }
}
