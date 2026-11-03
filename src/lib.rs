//! `k3-engine`: Low-level compute runtime for the `SpacemiT` K3 AI platform.
//!
//! Architecture recipes (`arch-*`) and compute kernels (`kernels-*`) are compiled
//! in-process via Cargo feature gates per ADR-024, targeting `riscv64gc-unknown-linux-gnu`
//! with RVV 1.0 (`+v`) per ADR-029.

pub mod affinity;
pub mod capability;
pub mod engine;
#[cfg(feature = "kernels-rvv")]
pub mod forward;
pub mod kv;
pub mod memory;
pub mod weights;
pub mod worker;
pub use worker as k3_worker;

pub use affinity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, ClusterPreset, CpuTopologyInspector,
    FutexBarrier, IntoClusterPreset, MockCpuTopology, ResolvedAffinity, SpinBarrier,
    SystemCpuTopology, VlenDomain, WorkerAffinityStatus, WorkerPool, WorkerWaitSlot,
    X100_CLUSTER_MASK, X100_DEFAULT_MAX_COMPUTE_THREADS, parse_cpu_list_mask, resolve_affinity,
    resolve_affinity_with,
};
pub use capability::{RecipeRegistry, verify_metadata_capabilities, verify_model_capabilities};
pub use engine::{
    EnginePhase, EvaluationSummary, PassTiming, SinglePromptEngine, StreamAction,
    sample_greedy_argmax,
};
pub use k3_core::{
    self, ArchRecipe, CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX,
    CONSERVATIVE_DEFAULT_X100_CORES, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX,
    ComputePool, CoordinationProfile, ExecutionContext, GgufFile, K3Error, ModelEntry,
    ModelListResponse, ModelLoadAuditEvent, ModelMetadata, ModelProfile, ModelWeights,
    PriorityClass, ProfileStatus, QuantType, TensorDescriptor, TensorView, TuningProfile,
};
pub use kv::SingleSlotKvArena;
pub use memory::{
    DEFAULT_PAGE_SIZE, KV_BLOCK_SIZE, MemoryFootprint, MemoryInspector, MockMemoryInspector,
    SystemMemoryInspector, calculate_footprint, calculate_footprint_with_slots,
    check_memory_admission, check_memory_admission_with, check_memory_admission_with_slots,
    system_page_size,
};
pub use weights::LockedModelWeights;

#[cfg(feature = "arch-llama")]
pub use k3_arch_llama as arch_llama;
#[cfg(feature = "arch-llama")]
pub use k3_arch_llama::{LlamaRecipe, project_lm_head_row_split};

#[cfg(feature = "arch-mistral")]
pub use k3_arch_mistral as arch_mistral;
#[cfg(feature = "arch-mistral")]
pub use k3_arch_mistral::{COMPACT_MOE_ADVISORY, MistralRecipe, evaluate_moe_memlock_admission};

#[cfg(feature = "arch-phi3")]
pub use k3_arch_phi3 as arch_phi3;
#[cfg(feature = "arch-phi3")]
pub use k3_arch_phi3::Phi3Recipe;

#[cfg(feature = "arch-bitnet")]
pub use k3_arch_bitnet as arch_bitnet;
#[cfg(feature = "arch-bitnet")]
pub use k3_arch_bitnet::BitNetRecipe;

#[cfg(feature = "arch-flint")]
pub use k3_arch_flint as arch_flint;
#[cfg(feature = "arch-flint")]
pub use k3_arch_flint::{
    FLINT_MEMLOCK_ADVISORY, FlintAttentionType, FlintRecipe, evaluate_flint_memlock_admission,
    route_sigmoid_topk_experts,
};

#[cfg(feature = "arch-synthetic")]
pub use arch_synthetic;

#[cfg(feature = "kernels-rvv")]
pub use kernels_rvv;

/// Indicates whether the `arch-bitnet` architecture recipe feature gate is enabled.
pub const HAS_ARCH_BITNET: bool = cfg!(feature = "arch-bitnet");

/// Indicates whether the `arch-flint` architecture recipe feature gate is enabled.
pub const HAS_ARCH_FLINT: bool = cfg!(feature = "arch-flint");

/// Indicates whether the `arch-llama` architecture recipe feature gate is enabled.
pub const HAS_ARCH_LLAMA: bool = cfg!(feature = "arch-llama");

/// Indicates whether the `arch-mistral` architecture recipe feature gate is enabled.
pub const HAS_ARCH_MISTRAL: bool = cfg!(feature = "arch-mistral");

/// Indicates whether the `arch-phi3` architecture recipe feature gate is enabled.
pub const HAS_ARCH_PHI3: bool = cfg!(feature = "arch-phi3");

/// Indicates whether the `arch-synthetic` architecture recipe feature gate is enabled.
pub const HAS_ARCH_SYNTHETIC: bool = cfg!(feature = "arch-synthetic");

/// Indicates whether the `kernels-rvv` RVV 1.0 kernel feature gate is enabled.
pub const HAS_KERNELS_RVV: bool = cfg!(feature = "kernels-rvv");

#[cfg(test)]
mod tests {
    use super::{
        HAS_ARCH_BITNET, HAS_ARCH_FLINT, HAS_ARCH_LLAMA, HAS_ARCH_MISTRAL, HAS_ARCH_PHI3,
        HAS_ARCH_SYNTHETIC, HAS_KERNELS_RVV, QuantType,
    };

    #[test]
    fn feature_gates_match_cfg() {
        assert_eq!(HAS_ARCH_BITNET, cfg!(feature = "arch-bitnet"));
        assert_eq!(HAS_ARCH_FLINT, cfg!(feature = "arch-flint"));
        assert_eq!(HAS_ARCH_LLAMA, cfg!(feature = "arch-llama"));
        assert_eq!(HAS_ARCH_MISTRAL, cfg!(feature = "arch-mistral"));
        assert_eq!(HAS_ARCH_PHI3, cfg!(feature = "arch-phi3"));
        assert_eq!(HAS_ARCH_SYNTHETIC, cfg!(feature = "arch-synthetic"));
        assert_eq!(HAS_KERNELS_RVV, cfg!(feature = "kernels-rvv"));
        assert_eq!(QuantType::Q4_K_M.as_str(), "Q4_K_M");
    }

    #[cfg(feature = "arch-flint")]
    #[test]
    fn arch_flint_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_flint::FlintRecipe;

        let recipe = FlintRecipe::new();
        assert_eq!(recipe.architecture_id(), "flint");
        assert_eq!(recipe.num_layers, 49);
        assert_eq!(recipe.num_experts, 256);
        assert_eq!(recipe.num_experts_per_tok, 6);
    }

    #[cfg(feature = "arch-bitnet")]
    #[test]
    fn arch_bitnet_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_bitnet::BitNetRecipe;

        let recipe = BitNetRecipe::new();
        assert_eq!(recipe.architecture_id(), "bitnet");
        let recipe_b158 = BitNetRecipe::new_b158();
        assert_eq!(recipe_b158.architecture_id(), "bitnet-b1.58");
        let recipe_25 = BitNetRecipe::new_25();
        assert_eq!(recipe_25.architecture_id(), "bitnet-25");
    }

    #[cfg(feature = "arch-synthetic")]
    #[test]
    fn arch_synthetic_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_synthetic::SyntheticRecipe;

        let recipe = SyntheticRecipe::new();
        assert_eq!(recipe.architecture_id(), "synthetic");
    }

    #[cfg(feature = "arch-llama")]
    #[test]
    fn arch_llama_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_llama::LlamaRecipe;

        let recipe = LlamaRecipe::new();
        assert_eq!(recipe.architecture_id(), "llama");
        let recipe4 = LlamaRecipe::new_llama4();
        assert_eq!(recipe4.architecture_id(), "llama4");
    }

    #[cfg(feature = "arch-mistral")]
    #[test]
    fn arch_mistral_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_mistral::MistralRecipe;

        let recipe = MistralRecipe::new();
        assert_eq!(recipe.architecture_id(), "mistral");
        let mixtral = MistralRecipe::new_mixtral();
        assert_eq!(mixtral.architecture_id(), "mixtral");
    }

    #[cfg(feature = "arch-phi3")]
    #[test]
    fn arch_phi3_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_phi3::Phi3Recipe;

        let recipe = Phi3Recipe::new();
        assert_eq!(recipe.architecture_id(), "phi3");
        let recipe4 = Phi3Recipe::new_phi4();
        assert_eq!(recipe4.architecture_id(), "phi4");
    }

    #[cfg(feature = "kernels-rvv")]
    #[test]
    fn kernels_rvv_wired() {
        use super::kernels_rvv::{BaselineKernel, VectorKernel};

        let kernel = BaselineKernel::new();
        assert!((kernel.dot_product_f32(&[2.0], &[3.0]) - 6.0).abs() < f32::EPSILON);
    }
}
