//! Architecture recipe capability filter and compile-time registry.
//!
//! Enforces upfront capability verification against compiled `arch-*` recipes and compute kernels
//! before reserving or locking memory (ADR-017, ADR-023, ADR-024, US-01 Scenario 1).

use jero_core::{ArchRecipe, GgufFile, K3Error, ModelMetadata};

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
        ];
        ARCHS
    }
}

/// Verifies that the model's architecture recipe is compiled and that every tensor's
/// quantization layout is supported by the active backend (ADR-023, US-01 Scenario 1).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if:
/// - The model's `general.architecture` has no compiled recipe in this binary.
/// - Any tensor in the model uses a quantization layout unsupported by the matching recipe.
/// - The trained context window is zero.
pub fn verify_model_capabilities(model: &GgufFile<'_>) -> Result<&'static dyn ArchRecipe, K3Error> {
    verify_metadata_capabilities(model.metadata())
}

/// Verifies that the extracted [`ModelMetadata`] satisfies compile-time recipe and
/// quantization capabilities before allocating or `mlock`ing weight buffers.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if capability checks fail.
pub fn verify_metadata_capabilities(
    metadata: &ModelMetadata,
) -> Result<&'static dyn ArchRecipe, K3Error> {
    let arch = metadata.architecture();

    // 1. Upfront architecture recipe lookup (ADR-023)
    let recipe = RecipeRegistry::find_recipe(arch).ok_or_else(|| K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: format!(
            "architecture '{arch}' is not compiled into this binary (compiled recipes: {:?})",
            RecipeRegistry::supported_architectures()
        ),
    })?;

    // 2. Validate every tensor quantization format against the active recipe and kernels
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

    // 3. Validate context window bound (ADR-027)
    if metadata.context_length() == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "model context length for architecture '{arch}' must be greater than 0"
            ),
        });
    }

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
}
