//! Google Gemma architecture family shim re-exporting `jero-arch-gemma` (`gemma2`, `gemma3`,
//! `gemma4`, `functiongemma`) plus cluster lane placement helpers.

pub use jero_arch_gemma::*;

#[cfg(feature = "jero-kernels-rvv")]
use crate::engine::EnginePhase;
#[cfg(feature = "jero-kernels-rvv")]
use crate::forward::Lane;

/// Selects the target compute cluster lane for Gemma `RMSNorm` and QK-norm operations:
/// `A100` during prompt prefill (`EnginePhase::Prefill`) and `X100` during decode (`EnginePhase::Decode`).
#[cfg(feature = "jero-kernels-rvv")]
#[must_use]
pub const fn gemma_norm_lane(phase: EnginePhase) -> Lane {
    match phase {
        EnginePhase::Prefill => Lane::A100,
        EnginePhase::Idle | EnginePhase::Decode | EnginePhase::Completed => Lane::X100,
    }
}

#[cfg(all(test, feature = "jero-kernels-rvv"))]
mod tests {
    use super::*;

    #[test]
    fn test_gemma_norm_lane_phase_split() {
        assert_eq!(gemma_norm_lane(EnginePhase::Prefill), Lane::A100);
        assert_eq!(gemma_norm_lane(EnginePhase::Decode), Lane::X100);
        assert_eq!(gemma_norm_lane(EnginePhase::Idle), Lane::X100);
        assert_eq!(gemma_norm_lane(EnginePhase::Completed), Lane::X100);
    }
}
