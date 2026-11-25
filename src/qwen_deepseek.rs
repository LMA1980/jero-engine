//! Alibaba Qwen (`qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3-embedding`)
//! and `DeepSeek` (`deepseek2`) shim re-exporting `jero-arch-qwen`, plus heterogeneous cluster
//! placement helpers for [`Qwen3MoeRecipe`].

pub use jero_arch_qwen::*;

#[cfg(feature = "jero-kernels-rvv")]
use crate::forward::{Lane, PlacementPlan};

/// Heterogeneous cluster placement extension for [`Qwen3MoeRecipe`].
#[cfg(feature = "jero-kernels-rvv")]
pub trait Qwen3MoePlacementExt {
    /// Returns the heterogeneous `A100` (experts/matmul) + `X100` (router/norm) placement plan.
    fn placement_plan(&self) -> PlacementPlan;
    /// Returns the target lane for `RMSNorm` and 128-way router softmax gating.
    fn router_norm_lane(&self) -> Lane;
    /// Returns the target lane for active top-`k` expert `ime_tile_8x16` matrix multiplications.
    fn expert_compute_lane(&self) -> Lane;
}

#[cfg(feature = "jero-kernels-rvv")]
impl Qwen3MoePlacementExt for Qwen3MoeRecipe {
    fn placement_plan(&self) -> PlacementPlan {
        PlacementPlan::op(Lane::A100, Lane::X100)
    }

    fn router_norm_lane(&self) -> Lane {
        Lane::X100
    }

    fn expert_compute_lane(&self) -> Lane {
        Lane::A100
    }
}

#[cfg(all(test, feature = "jero-kernels-rvv"))]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_moe_placement_ext() {
        let moe = Qwen3MoeRecipe::new_30b_a3b();
        assert_eq!(
            moe.placement_plan(),
            PlacementPlan::op(Lane::A100, Lane::X100)
        );
        assert_eq!(moe.router_norm_lane(), Lane::X100);
        assert_eq!(moe.expert_compute_lane(), Lane::A100);
    }
}
