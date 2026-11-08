//! Empirical Adversarial Challenge for k3-engine US-07 conversions and re-exports.

use jero_engine::{
    CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_DEFAULT_X100_CORES,
    CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ClusterPreset, ComputePool,
    IntoClusterPreset, K3Error, ModelEntry, ModelListResponse, ModelLoadAuditEvent, ModelProfile,
    PriorityClass, ProfileStatus, TuningProfile,
};

#[test]
fn test_challenge_engine_cluster_preset_conversions() {
    // 1. Direct function conversion
    assert_eq!(
        ClusterPreset::from_compute_pool(ComputePool::X100),
        Some(ClusterPreset::X100)
    );
    assert_eq!(
        ClusterPreset::from_compute_pool(ComputePool::A100),
        Some(ClusterPreset::A100)
    );
    assert_eq!(
        ClusterPreset::from_compute_pool(ComputePool::Both),
        None,
        "ComputePool::Both cannot be mapped to a single ClusterPreset"
    );

    // 2. Extension trait conversion
    assert_eq!(
        ComputePool::X100.into_cluster_preset(),
        Some(ClusterPreset::X100)
    );
    assert_eq!(
        ComputePool::A100.into_cluster_preset(),
        Some(ClusterPreset::A100)
    );
    assert_eq!(ComputePool::Both.into_cluster_preset(), None);

    // 3. TryFrom conversion
    assert_eq!(
        ClusterPreset::try_from(ComputePool::X100).unwrap(),
        ClusterPreset::X100
    );
    assert_eq!(
        ClusterPreset::try_from(ComputePool::A100).unwrap(),
        ClusterPreset::A100
    );
    let both_err = ClusterPreset::try_from(ComputePool::Both).unwrap_err();
    assert!(
        matches!(both_err, K3Error::CapabilityMismatch { .. }),
        "TryFrom(ComputePool::Both) must return CapabilityMismatch"
    );

    // 4. Reverse conversion to_compute_pool
    assert_eq!(ClusterPreset::X100.to_compute_pool(), ComputePool::X100);
    assert_eq!(ClusterPreset::A100.to_compute_pool(), ComputePool::A100);
}

#[test]
fn test_challenge_engine_reexports_and_conservative_defaults() {
    assert_eq!(CONSERVATIVE_DEFAULT_N_CTX, 2048);
    assert_eq!(CONSTRAINED_TEST_N_CTX, 64);
    assert_eq!(CONSERVATIVE_DEFAULT_X100_CORES, 6);
    assert_eq!(CONSERVATIVE_DEFAULT_A100_CORES, 0);
    assert_eq!(
        CONSERVATIVE_TUNING_ADVISORY,
        "Model is running with safe conservative defaults (unprofiled). Run 'k3-diag' to profile hardware performance and apply an operator tuning profile."
    );

    let default_prof = TuningProfile::conservative_default();
    assert_eq!(default_prof.compute_pool, ComputePool::X100);
    assert_eq!(default_prof.priority_class, PriorityClass::Throughput);
    assert_eq!(default_prof.profile_status, ProfileStatus::Unprofiled);
    assert!(default_prof.is_unprofiled());

    let model = ModelProfile::unprofiled_default("engine-reexport-test");
    let entry: ModelEntry = model.to_model_entry(12345);
    assert_eq!(entry.id, "engine-reexport-test");
    assert!(entry.unprofiled);

    let list = ModelListResponse::new(vec![entry]);
    assert_eq!(list.data.len(), 1);

    let audit = ModelLoadAuditEvent::from_tuning("engine-reexport-test", &default_prof);
    assert_eq!(audit.cluster, "x100");
}
