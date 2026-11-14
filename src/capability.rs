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
    pub sliding_window: Option<usize>,
    /// Optional `MoE` total expert count (`{arch}.expert_count`).
    pub expert_count: Option<usize>,
    /// Optional `MoE` active experts per token (`{arch}.expert_used_count`).
    pub expert_used_count: Option<usize>,
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

        Self {
            architecture: arch,
            block_count,
            embedding_length,
            feed_forward_length,
            head_count,
            head_count_kv,
            key_length,
            explicit_key_length: tensor_key_len,
            rope_freq_base: None,
            rms_norm_eps: None,
            vocab_size,
            context_length: metadata.context_length(),
            sliding_window: None,
            expert_count: None,
            expert_used_count: None,
        }
    }

    /// Extracts full model hyperparameters directly from a parsed [`GgufFile`] header, combining
    /// raw GGUF key-value entries (`feed_forward_length`, `attention.key_length`, `rope.freq_base`,
    /// `attention.layer_norm_rms_epsilon`, `vocab_size`, `sliding_window`, `expert_count`) with
    /// tensor-shape fallbacks.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut params = Self::from_metadata(gguf.metadata());
        let arch = params.architecture.clone();

        if let Some(ffn) = gguf
            .metadata_value(&format!("{arch}.feed_forward_length"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.feed_forward_length = ffn;
        }

        if let Some(kl) = gguf
            .metadata_value(&format!("{arch}.attention.key_length"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.explicit_key_length = Some(kl);
            params.key_length = kl;
        }

        if let Some(freq) = gguf
            .metadata_value(&format!("{arch}.rope.freq_base"))
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            params.rope_freq_base = Some(freq);
        }

        if let Some(eps) = gguf
            .metadata_value(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            params.rms_norm_eps = Some(eps);
        }

        if let Some(GgufArray::String(tokens)) = gguf
            .metadata_value("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            && !tokens.is_empty()
        {
            params.vocab_size = params.vocab_size.max(tokens.len());
        }

        if let Some(vs) = gguf
            .metadata_value(&format!("{arch}.vocab_size"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.vocab_size = vs;
        }

        if let Some(sw) = gguf
            .metadata_value(&format!("{arch}.attention.sliding_window"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.sliding_window = Some(sw);
        }

        if let Some(ec) = gguf
            .metadata_value(&format!("{arch}.expert_count"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.expert_count = Some(ec);
        }

        if let Some(euc) = gguf
            .metadata_value(&format!("{arch}.expert_used_count"))
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
        {
            params.expert_used_count = Some(euc);
        }

        params
    }
}

/// In-process compile-time registry mapping GGUF `general.architecture` strings
/// to compiled [`ArchRecipe`] implementations (ADR-023, ADR-024).
#[derive(Debug, Clone, Copy, Default)]
pub struct RecipeRegistry;

impl RecipeRegistry {
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

        let _ = architecture;
        None
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
            #[cfg(feature = "jero-arch-llama")]
            "qwen2",
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
                recipe.sliding_window = Some(sw);
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
            if params.context_length > 0 {
                recipe.original_max_position_embeddings = params.context_length;
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
                && let Ok(sw_u32) = u32::try_from(sw)
                && sw_u32 > 0
            {
                recipe.sliding_window = sw_u32;
            }
            if let Some(freq) = params.rope_freq_base {
                recipe.rope_theta_full = freq;
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
                recipe.sliding_window = sw;
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
                recipe.sliding_window = sw;
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
        Self::instantiate_from_hyperparams(&params, Some(gguf.metadata()))
    }
}

fn validate_recipe_against_metadata(
    recipe: &dyn ArchRecipe,
    metadata: &ModelMetadata,
) -> Result<(), K3Error> {
    let arch = metadata.architecture();
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

    Ok(())
}

/// Verifies that the model's architecture recipe is compiled, dynamically hydrates its GGUF
/// hyperparameters, and validates that every tensor's quantization layout is supported by the
/// active backend (ADR-023, US-01 Scenario 1).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if:
/// - The model's `general.architecture` has no compiled recipe in this binary.
/// - Any tensor in the model uses a quantization layout unsupported by the matching recipe.
/// - The trained context window is zero.
pub fn verify_model_capabilities(model: &GgufFile<'_>) -> Result<Box<dyn ArchRecipe>, K3Error> {
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
    validate_recipe_against_metadata(recipe.as_ref(), model.metadata())?;
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
    validate_recipe_against_metadata(recipe.as_ref(), metadata)?;
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
    }

    #[cfg(feature = "jero-arch-modernbert")]
    #[test]
    fn test_modernbert_and_bert_classifier_pass() {
        use jero_core::ExecutionContext;

        for arch in ["modernbert", "bert", "deberta-v3"] {
            let tensor_specs = [
                ("token_embd.weight", QuantType::F32, &[8, 16][..]),
                ("blk.0.attn_q.weight", QuantType::F32, &[8, 8][..]),
                ("classifier.weight", QuantType::F32, &[8, 3][..]),
            ];
            let bytes = create_synthetic_gguf_v3(arch, 64, &tensor_specs);
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
}
