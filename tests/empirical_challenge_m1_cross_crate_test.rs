//! Empirical Challenger Test Suite for Milestone 1 (US-07):
//! Cross-Crate Conversions, Malformed JSON Fuzzing & Boundary Conditions.
//!
//! Verifies:
//! 1. Cross-crate conversions between `ComputePool` and `ClusterPreset` in `code/engine/src/affinity.rs`:
//!    - `ClusterPreset::from_compute_pool(ComputePool::X100) == Some(ClusterPreset::X100)`
//!    - `ClusterPreset::from_compute_pool(ComputePool::A100) == Some(ClusterPreset::A100)`
//!    - `ClusterPreset::from_compute_pool(ComputePool::Both) == None`
//!    - `ComputePool::into_cluster_preset()` extension trait parity
//!    - `ClusterPreset::try_from(ComputePool)` structured error handling for `ComputePool::Both`
//!    - `ClusterPreset::to_compute_pool()` reverse mapping
//!    - Core mask and default thread limit alignment across crates
//! 2. Malformed, truncated, and unexpected JSON deserialization fuzzing for `TuningProfile`, `ModelEntry`,
//!    `ModelListResponse`, and `ModelLoadAuditEvent`.
//! 3. Boundary conditions: `n_ctx = 0`, `n_ctx = usize::MAX`, extreme core counts, Unicode/path model IDs,
//!    whitespace/case parsing behavior, and enum serialization invariance.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::undocumented_unsafe_blocks
)]

use std::str::FromStr;

use jero_engine::{
    A100_CLUSTER_MASK, CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX,
    CONSERVATIVE_DEFAULT_X100_CORES, CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX,
    ClusterPreset, ComputePool, IntoClusterPreset, K3Error, ModelEntry, ModelListResponse,
    ModelLoadAuditEvent, ModelProfile, PriorityClass, ProfileStatus, TuningProfile,
    X100_CLUSTER_MASK,
};

// ============================================================================
// Challenge 1: Cross-Crate Conversions Between ComputePool and ClusterPreset
// ============================================================================

#[test]
fn test_cross_crate_compute_pool_to_cluster_preset_mapping() {
    // 1. from_compute_pool
    assert_eq!(
        ClusterPreset::from_compute_pool(ComputePool::X100),
        Some(ClusterPreset::X100)
    );
    assert_eq!(
        ClusterPreset::from_compute_pool(ComputePool::A100),
        Some(ClusterPreset::A100)
    );
    assert_eq!(ClusterPreset::from_compute_pool(ComputePool::Both), None);

    // 2. IntoClusterPreset extension trait
    assert_eq!(
        ComputePool::X100.into_cluster_preset(),
        Some(ClusterPreset::X100)
    );
    assert_eq!(
        ComputePool::A100.into_cluster_preset(),
        Some(ClusterPreset::A100)
    );
    assert_eq!(ComputePool::Both.into_cluster_preset(), None);

    // 3. TryFrom<ComputePool> for ClusterPreset
    assert_eq!(
        ClusterPreset::try_from(ComputePool::X100).expect("X100 converts to ClusterPreset::X100"),
        ClusterPreset::X100
    );
    assert_eq!(
        ClusterPreset::try_from(ComputePool::A100).expect("A100 converts to ClusterPreset::A100"),
        ClusterPreset::A100
    );

    let both_err = ClusterPreset::try_from(ComputePool::Both)
        .expect_err("ComputePool::Both must fail TryFrom<ComputePool>");
    match both_err {
        K3Error::CapabilityMismatch {
            architecture,
            reason,
        } => {
            assert_eq!(architecture, "cluster");
            assert!(
                reason.contains("ComputePool::Both cannot be converted to a single ClusterPreset"),
                "Unexpected reason: {reason}"
            );
        }
        other => panic!("Expected K3Error::CapabilityMismatch, got {other:?}"),
    }

    // 4. Reverse to_compute_pool
    assert_eq!(ClusterPreset::X100.to_compute_pool(), ComputePool::X100);
    assert_eq!(ClusterPreset::A100.to_compute_pool(), ComputePool::A100);

    // 5. Bidirectional roundtrip for single clusters
    for preset in [ClusterPreset::X100, ClusterPreset::A100] {
        let pool = preset.to_compute_pool();
        assert_eq!(ClusterPreset::from_compute_pool(pool), Some(preset));
        assert_eq!(ClusterPreset::try_from(pool).ok(), Some(preset));
        assert_eq!(pool.into_cluster_preset(), Some(preset));
    }
}

#[test]
fn test_cross_crate_mask_and_thread_capacity_alignment() {
    // Assert physical core mask equivalence across crates
    assert_eq!(
        ClusterPreset::X100.core_mask(),
        ComputePool::X100.core_mask()
    );
    assert_eq!(ClusterPreset::X100.core_mask(), X100_CLUSTER_MASK);
    assert_eq!(ComputePool::X100.core_mask(), 0x00FF);

    assert_eq!(
        ClusterPreset::A100.core_mask(),
        ComputePool::A100.core_mask()
    );
    assert_eq!(ClusterPreset::A100.core_mask(), A100_CLUSTER_MASK);
    assert_eq!(ComputePool::A100.core_mask(), 0xFF00);

    // Dual-cluster pool mask
    assert_eq!(ComputePool::Both.core_mask(), 0xFFFF);
    assert_eq!(
        ComputePool::Both.core_mask(),
        X100_CLUSTER_MASK | A100_CLUSTER_MASK
    );

    // Assert thread count default capacity equivalence across crates
    assert_eq!(
        ClusterPreset::X100.default_max_threads(),
        ComputePool::X100.default_max_compute_cores()
    );
    assert_eq!(ClusterPreset::X100.default_max_threads(), 6);
    assert_eq!(
        ClusterPreset::X100.default_max_threads(),
        CONSERVATIVE_DEFAULT_X100_CORES
    );

    assert_eq!(
        ClusterPreset::A100.default_max_threads(),
        ComputePool::A100.default_max_compute_cores()
    );
    assert_eq!(ClusterPreset::A100.default_max_threads(), 8);

    assert_eq!(ComputePool::Both.default_max_compute_cores(), 14);
    assert_eq!(CONSERVATIVE_DEFAULT_A100_CORES, 0);
}

// ============================================================================
// Challenge 2: Malformed JSON Deserialization Fuzzing
// ============================================================================

#[test]
fn test_malformed_json_deserialization_fuzz_tuning_profile() {
    let malformed_inputs = [
        // Empty / truncated
        "",
        "   ",
        "{",
        "{}",
        "{\"compute_pool\":",
        "{\"compute_pool\": \"x100\"",
        // Invalid syntax
        "{\"n_ctx\": 2048, }",
        "{\"unprofiled\": True}",
        "{\"cluster\": 'x100'}",
        // Missing required fields
        r#"{"compute_pool":"x100"}"#,
        r#"{"compute_pool":"x100","cluster":"x100"}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput"}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled"}"#,
        // Invalid enum variants
        r#"{"compute_pool":"gpu","cluster":"gpu","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"X100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"realtime","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"THROUGHPUT","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unknown","unprofiled":true}"#,
        // Type mismatches
        r#"{"compute_pool":100,"cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":"six","a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":-1,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":"2048","lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":-2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":"false","borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":"true"}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":18446744073709551616,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        // Nulls on non-optional fields
        r#"{"compute_pool":null,"cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":null,"x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":null,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":null,"n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":null,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":null,"unprofiled":true}"#,
        r#"{"compute_pool":"x100","cluster":"x100","x100_cores":6,"a100_cores":0,"priority_class":"throughput","n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false,"profile_status":"unprofiled","unprofiled":null}"#,
    ];

    for (idx, malformed) in malformed_inputs.iter().enumerate() {
        let res = TuningProfile::from_json_str(malformed);
        assert!(
            res.is_err(),
            "Malformed input #{idx} unexpectedly succeeded: {malformed}"
        );
    }
}

#[test]
fn test_malformed_json_deserialization_fuzz_model_entry() {
    let malformed_inputs = [
        "",
        "{}",
        "{\"id\": \"qwen\"}",
        // Missing fields
        r#"{"id":"test","object":"model"}"#,
        r#"{"id":"test","object":"model","created":1700000000}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3"}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":true}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled"}"#,
        // Type mismatches
        r#"{"id":123,"object":"model","created":1700000000,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":0,"n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"id":"test","object":"model","created":-1,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":0,"n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":"true","profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":0,"n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":-1,"n_ctx":2048,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":0,"n_ctx":-5,"lend_idle_cores":false,"borrow_idle_cores":false}"#,
        r#"{"id":"test","object":"model","created":1700000000,"owned_by":"k3","unprofiled":true,"profile_status":"unprofiled","priority_class":"throughput","cluster":"x100","a100_cores":0,"n_ctx":2048,"lend_idle_cores":"false","borrow_idle_cores":false}"#,
    ];

    for (idx, malformed) in malformed_inputs.iter().enumerate() {
        let res: Result<ModelEntry, _> = serde_json::from_str(malformed);
        assert!(
            res.is_err(),
            "Malformed ModelEntry input #{idx} unexpectedly succeeded: {malformed}"
        );
    }
}

#[test]
fn test_json_extra_fields_tolerance() {
    // Serde by default ignores unknown fields. Verify that adding arbitrary metadata
    // does not crash or corrupt the parsed fields.
    let json_with_extra = serde_json::json!({
        "compute_pool": "x100",
        "cluster": "x100",
        "x100_cores": 6,
        "a100_cores": 0,
        "priority_class": "throughput",
        "n_ctx": 2048,
        "lend_idle_cores": false,
        "borrow_idle_cores": false,
        "profile_status": "unprofiled",
        "unprofiled": true,
        "advisory_notice": CONSERVATIVE_TUNING_ADVISORY,
        "unexpected_future_feature_flag": true,
        "nested_unknown_blob": { "key": [1, 2, 3] }
    });

    let json_str = json_with_extra.to_string();
    let parsed = TuningProfile::from_json_str(&json_str)
        .expect("TuningProfile should cleanly tolerate unknown fields");

    assert_eq!(parsed.compute_pool, ComputePool::X100);
    assert_eq!(parsed.n_ctx, 2048);
    assert!(parsed.unprofiled);
    assert_eq!(
        parsed.advisory_notice.as_deref(),
        Some(CONSERVATIVE_TUNING_ADVISORY)
    );
}

// ============================================================================
// Challenge 3: Boundary Conditions & Invariant Fuzzing
// ============================================================================

#[test]
fn test_boundary_n_ctx_zero_and_max() {
    // 1. Zero context window
    let zero_prof = TuningProfile::conservative_default_with_n_ctx(0);
    assert_eq!(zero_prof.n_ctx, 0);
    let zero_json = zero_prof.to_json_pretty().expect("serialize n_ctx=0");
    let zero_roundtrip = TuningProfile::from_json_str(&zero_json).expect("deserialize n_ctx=0");
    assert_eq!(zero_roundtrip.n_ctx, 0);
    assert_eq!(zero_prof, zero_roundtrip);

    // 2. Max context window (usize::MAX)
    let max_prof = TuningProfile::conservative_default_with_n_ctx(usize::MAX);
    assert_eq!(max_prof.n_ctx, usize::MAX);
    let max_json = max_prof.to_json_pretty().expect("serialize n_ctx=MAX");
    let max_roundtrip = TuningProfile::from_json_str(&max_json).expect("deserialize n_ctx=MAX");
    assert_eq!(max_roundtrip.n_ctx, usize::MAX);
    assert_eq!(max_prof, max_roundtrip);

    // 3. Constrained test bound (64)
    let constrained_prof = TuningProfile::conservative_default_with_n_ctx(CONSTRAINED_TEST_N_CTX);
    assert_eq!(constrained_prof.n_ctx, 64);
    assert_eq!(constrained_prof.n_ctx, CONSTRAINED_TEST_N_CTX);

    // 4. Default bound (2048)
    let default_prof = TuningProfile::conservative_default();
    assert_eq!(default_prof.n_ctx, 2048);
    assert_eq!(default_prof.n_ctx, CONSERVATIVE_DEFAULT_N_CTX);
}

#[test]
fn test_boundary_extreme_core_counts_roundtrip() {
    let mut prof = TuningProfile::conservative_default();
    // Test 0 cores allocation
    prof.apply_operator_tuning(
        ComputePool::X100,
        0,
        0,
        PriorityClass::Throughput,
        1024,
        false,
        false,
    );
    assert_eq!(prof.x100_cores, 0);
    assert_eq!(prof.a100_cores, 0);

    let json = prof.to_json_pretty().unwrap();
    let roundtrip = TuningProfile::from_json_str(&json).unwrap();
    assert_eq!(prof, roundtrip);

    // Test extreme max core allocation
    prof.apply_operator_tuning(
        ComputePool::Both,
        usize::MAX,
        usize::MAX,
        PriorityClass::Latency,
        8192,
        true,
        true,
    );
    assert_eq!(prof.x100_cores, usize::MAX);
    assert_eq!(prof.a100_cores, usize::MAX);

    let json_max = prof.to_json_pretty().unwrap();
    let roundtrip_max = TuningProfile::from_json_str(&json_max).unwrap();
    assert_eq!(prof, roundtrip_max);
}

#[test]
fn test_boundary_string_parsing_and_case_insensitivity() {
    // 1. PriorityClass: case-insensitive ASCII
    for valid in ["throughput", "Throughput", "THROUGHPUT", "tHrOuGhPuT"] {
        assert_eq!(
            PriorityClass::from_str(valid).unwrap(),
            PriorityClass::Throughput
        );
        assert_eq!(
            PriorityClass::parse(valid).unwrap(),
            PriorityClass::Throughput
        );
    }
    for valid in ["latency", "Latency", "LATENCY", "LaTeNcY"] {
        assert_eq!(
            PriorityClass::from_str(valid).unwrap(),
            PriorityClass::Latency
        );
        assert_eq!(PriorityClass::parse(valid).unwrap(), PriorityClass::Latency);
    }
    for invalid in [
        "",
        "   ",
        "throughput ",
        " throughput",
        "latency\0",
        "batch",
        "realtime",
    ] {
        assert!(PriorityClass::from_str(invalid).is_err());
        assert!(PriorityClass::parse(invalid).is_err());
    }

    // 2. ComputePool: case-insensitive ASCII
    for valid in ["x100", "X100", "X100"] {
        assert_eq!(ComputePool::from_str(valid).unwrap(), ComputePool::X100);
        assert_eq!(ComputePool::parse(valid).unwrap(), ComputePool::X100);
    }
    for valid in ["a100", "A100", "A100"] {
        assert_eq!(ComputePool::from_str(valid).unwrap(), ComputePool::A100);
        assert_eq!(ComputePool::parse(valid).unwrap(), ComputePool::A100);
    }
    for valid in ["both", "Both", "BOTH", "BoTh"] {
        assert_eq!(ComputePool::from_str(valid).unwrap(), ComputePool::Both);
        assert_eq!(ComputePool::parse(valid).unwrap(), ComputePool::Both);
    }
    for invalid in ["", " ", "x100 ", " a100", "gpu", "x200", "cluster0"] {
        assert!(ComputePool::from_str(invalid).is_err());
        assert!(ComputePool::parse(invalid).is_err());
    }

    // 3. ProfileStatus: case-insensitive ASCII
    for valid in ["unprofiled", "Unprofiled", "UNPROFILED"] {
        assert_eq!(
            ProfileStatus::from_str(valid).unwrap(),
            ProfileStatus::Unprofiled
        );
        assert_eq!(
            ProfileStatus::parse(valid).unwrap(),
            ProfileStatus::Unprofiled
        );
    }
    for valid in ["profiled", "Profiled", "PROFILED"] {
        assert_eq!(
            ProfileStatus::from_str(valid).unwrap(),
            ProfileStatus::Profiled
        );
        assert_eq!(
            ProfileStatus::parse(valid).unwrap(),
            ProfileStatus::Profiled
        );
    }
    for invalid in ["", "  ", "unprofiled ", "draft", "active", "pending"] {
        assert!(ProfileStatus::from_str(invalid).is_err());
        assert!(ProfileStatus::parse(invalid).is_err());
    }
}

#[test]
fn test_boundary_unicode_and_adversarial_model_ids() {
    let adversarial_ids = [
        "model-with-emoji-🚀-🔥",
        "deepseek-模型-7b",
        "../../etc/shadow",
        "null",
        "true",
        "false",
        "{\"injected\":\"json\"}",
        "spaces in identifier",
        "\t\r\nnewlines\n",
        &"very-long-".repeat(100),
    ];

    for &id in &adversarial_ids {
        let prof = ModelProfile::unprofiled_default(id);
        assert_eq!(prof.model_id, id);
        assert!(prof.is_unprofiled());

        let entry = prof.to_model_entry(1_700_000_000);
        assert_eq!(entry.id, id);

        // Serialize to JSON and deserialize back
        let json = serde_json::to_string(&entry).expect("serialize adversarial ModelEntry");
        let deserialized: ModelEntry =
            serde_json::from_str(&json).expect("deserialize adversarial ModelEntry");
        assert_eq!(deserialized.id, id);
        assert!(deserialized.unprofiled);
        assert_eq!(
            deserialized.advisory_notice.as_deref(),
            Some(CONSERVATIVE_TUNING_ADVISORY)
        );

        // Audit event with adversarial model_id
        let audit = ModelLoadAuditEvent::from_tuning(id, &prof.tuning);
        assert_eq!(audit.model_id, id);
        let audit_json = serde_json::to_string(&audit).expect("serialize audit event");
        let audit_deserialized: ModelLoadAuditEvent =
            serde_json::from_str(&audit_json).expect("deserialize audit event");
        assert_eq!(audit_deserialized.model_id, id);
    }
}

#[test]
fn test_model_list_response_empty_and_bulk_collection() {
    // Empty list
    let empty_list = ModelListResponse::new(vec![]);
    let empty_json = serde_json::to_string(&empty_list).unwrap();
    assert_eq!(empty_json, r#"{"object":"list","data":[]}"#);
    let empty_roundtrip: ModelListResponse = serde_json::from_str(&empty_json).unwrap();
    assert_eq!(empty_list, empty_roundtrip);

    // Bulk collection with 100 entries
    let entries: Vec<ModelEntry> = (0..100)
        .map(|i| {
            let prof = ModelProfile::unprofiled_with_n_ctx(format!("model-{i}"), 64 + i);
            prof.to_model_entry(1_700_000_000 + i as u64)
        })
        .collect();

    let list = ModelListResponse::new(entries);
    let json = serde_json::to_string(&list).unwrap();
    let roundtrip: ModelListResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.data.len(), 100);
    assert_eq!(roundtrip.data[0].id, "model-0");
    assert_eq!(roundtrip.data[99].id, "model-99");
    assert_eq!(roundtrip.data[99].n_ctx, 163);
}

#[test]
fn test_omitted_optional_fields_deserialization() {
    // 1. TuningProfile without optional model_id and advisory_notice
    let minimal_tuning_json = serde_json::json!({
        "compute_pool": "x100",
        "cluster": "x100",
        "x100_cores": 6,
        "a100_cores": 0,
        "priority_class": "throughput",
        "n_ctx": 2048,
        "lend_idle_cores": false,
        "borrow_idle_cores": false,
        "profile_status": "profiled",
        "unprofiled": false
    });
    let profile: TuningProfile =
        serde_json::from_value(minimal_tuning_json).expect("deserialize minimal tuning profile");
    assert!(profile.model_id.is_none());
    assert!(profile.advisory_notice.is_none());
    assert!(!profile.unprofiled);
    assert_eq!(profile.profile_status, ProfileStatus::Profiled);

    // 2. ModelEntry without optional advisory_notice
    let minimal_entry_json = serde_json::json!({
        "id": "minimal-model",
        "object": "model",
        "created": 1700000000,
        "owned_by": "k3",
        "unprofiled": false,
        "profile_status": "profiled",
        "priority_class": "throughput",
        "cluster": "x100",
        "a100_cores": 0,
        "n_ctx": 2048,
        "lend_idle_cores": false,
        "borrow_idle_cores": false
    });
    let entry: ModelEntry =
        serde_json::from_value(minimal_entry_json).expect("deserialize minimal model entry");
    assert_eq!(entry.id, "minimal-model");
    assert!(entry.advisory_notice.is_none());
    assert!(!entry.unprofiled);

    // 3. ModelLoadAuditEvent without optional advisory_notice
    let minimal_audit_json = serde_json::json!({
        "model_id": "minimal-model",
        "unprofiled": false,
        "profile_status": "profiled",
        "cluster": "x100",
        "a100_cores": 0,
        "priority_class": "throughput",
        "n_ctx": 2048
    });
    let audit: ModelLoadAuditEvent =
        serde_json::from_value(minimal_audit_json).expect("deserialize minimal audit event");
    assert_eq!(audit.model_id, "minimal-model");
    assert!(audit.advisory_notice.is_none());
    assert!(!audit.unprofiled);
}

#[test]
fn test_cost_profile_unprofiled_baseline_and_attachment() {
    use jero_engine::jero_core::{CostProfile, ModelIdentity};

    let identity = ModelIdentity {
        name: "test-model".to_owned(),
        architecture: "qwen2".to_owned(),
        file_path: "/models/test-model.gguf".to_owned(),
        file_digest: "fedcba9876543210".to_owned(),
    };

    let cost_baseline = CostProfile::unprofiled_default(identity.clone());

    // Invariant assertions on unprofiled CostProfile
    assert_eq!(cost_baseline.model_identity, identity);
    assert_eq!(cost_baseline.cluster_placement.cluster, "x100");
    assert_eq!(cost_baseline.cluster_placement.a100_cores, 0);
    assert_eq!(cost_baseline.cluster_placement.x100_cores, 6);
    assert_eq!(
        cost_baseline.cluster_placement.effective_core_mask,
        "0x003f"
    );
    assert_eq!(cost_baseline.allocated_cores.a100_cores, 0);
    assert_eq!(cost_baseline.allocated_cores.x100_cores, 6);
    assert_eq!(cost_baseline.allocated_cores.effective_mask, "0x003f");
    assert_eq!(
        cost_baseline.tuning_advisory.bottleneck_classification,
        "unprofiled"
    );

    // Serde roundtrip for CostProfile
    let cost_json = cost_baseline
        .to_json_pretty()
        .expect("serialize CostProfile");
    let deserialized_cost =
        CostProfile::from_json_str(&cost_json).expect("deserialize CostProfile");
    assert_eq!(cost_baseline, deserialized_cost);

    // Attach to ModelProfile
    let mut model_prof = ModelProfile::unprofiled_default("test-model");
    assert!(model_prof.cost_profile.is_none());
    model_prof.attach_cost_profile(cost_baseline);
    assert!(model_prof.cost_profile.is_some());

    // ModelProfile serde roundtrip with CostProfile attached
    let model_prof_json =
        serde_json::to_string_pretty(&model_prof).expect("serialize ModelProfile");
    let deserialized_model_prof: ModelProfile =
        serde_json::from_str(&model_prof_json).expect("deserialize ModelProfile");
    assert_eq!(model_prof, deserialized_model_prof);
}
