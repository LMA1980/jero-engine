//! Empirical Challenge Test Suite: System Boundary Conditions, Mixtral Memlock Admission,
//! Capability Mismatch Exit Codes, and ArenaHeader Layout Compatibility (Milestone 4 Epic 1).
//!
//! Verified by Teamwork Challenger 2 (`challenger_m4_2`).
//!
//! Objectives:
//! 1. Assert ArenaHeader 128-byte size, 64-byte alignment, exact field offsets, and
//!    backward compatibility with legacy zero-initialized arenas (version 1 preservation).
//! 2. Validate Mixtral 8x7B and 8x22B admission checks against the 28 GiB RLIMIT_MEMLOCK ceiling
//!    (`DEFAULT_MEMLOCK_LIMIT_BYTES` = 30,064,771,072 bytes), ensuring models requiring > 28 GiB
//!    are cleanly refused upfront with `InsufficientLockableMemory` before buffer allocation.
//! 3. Validate CapabilityMismatch handling: unsupported architectures, unsupported quantizations,
//!    corrupted router logits, zero context lengths, and exit code 10 verification with zero segfaults.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::undocumented_unsafe_blocks
)]

use core::mem::{align_of, size_of};
use std::fs::File;
use std::io::Write;
use std::process::Command;

use jero_arch_mistral::admission::{
    COMPACT_MOE_ADVISORY, DEFAULT_MEMLOCK_LIMIT_BYTES, MIXTRAL_KV_BYTES_PER_TOKEN,
    evaluate_moe_memlock_admission,
};
use jero_arch_mistral::recipe::route_top2_experts;
use jero_core::ipc::layout::{ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS};
use jero_core::test_helpers::{create_synthetic_gguf_v3, create_synthetic_gguf_v3_with_kvs};
use jero_core::{GgufValue, K3Error, ModelManifest, QuantType};
use zerocopy::{FromBytes, IntoBytes};

const DIAG_BIN: &str = env!("CARGO_BIN_EXE_jero-diag");

// ============================================================================
// Suite 1: ArenaHeader 128-Byte Size, Layout Geometry & Backward Compatibility
// ============================================================================

#[test]
fn test_challenge_arena_header_size_alignment_and_field_offsets() {
    // 1. Fixed struct geometry invariants
    assert_eq!(
        size_of::<ArenaHeader>(),
        128,
        "ArenaHeader must remain exactly 128 bytes"
    );
    assert_eq!(
        align_of::<ArenaHeader>(),
        64,
        "ArenaHeader must be 64-byte cacheline aligned"
    );

    let hdr = ArenaHeader::default();
    let base = &hdr as *const ArenaHeader as usize;

    // 2. Exact byte offsets of all fields
    assert_eq!((&hdr.magic as *const _ as usize) - base, 0);
    assert_eq!((&hdr.version as *const _ as usize) - base, 4);
    assert_eq!((&hdr.total_bytes as *const _ as usize) - base, 8);
    assert_eq!((&hdr.block_count as *const _ as usize) - base, 16);
    assert_eq!((&hdr.tokens_per_block as *const _ as usize) - base, 20);
    assert_eq!((&hdr.n_ctx as *const _ as usize) - base, 24);
    assert_eq!((&hdr.head_dim as *const _ as usize) - base, 28);
    assert_eq!((&hdr.num_layers as *const _ as usize) - base, 32);
    assert_eq!((&hdr.head_count_kv as *const _ as usize) - base, 36);
    assert_eq!((&hdr.vocab_size as *const _ as usize) - base, 40);
    assert_eq!((&hdr.active_slots as *const _ as usize) - base, 44);
    assert_eq!((&hdr.meta_table_offset as *const _ as usize) - base, 48);
    assert_eq!((&hdr.token_slots_offset as *const _ as usize) - base, 56);
    assert_eq!((&hdr.kv_blocks_offset as *const _ as usize) - base, 64);
    assert_eq!((&hdr.output_slabs_offset as *const _ as usize) - base, 72);
    assert_eq!((&hdr.generation_counter as *const _ as usize) - base, 80);

    // Newly carved fields from legacy _reserved ([u8; 40])
    assert_eq!(
        (&hdr.swa_window_size as *const _ as usize) - base,
        88,
        "swa_window_size must start at offset 88"
    );
    assert_eq!(
        (&hdr.num_experts as *const _ as usize) - base,
        92,
        "num_experts must start at offset 92"
    );
    assert_eq!(
        (&hdr.num_experts_per_tok as *const _ as usize) - base,
        94,
        "num_experts_per_tok must start at offset 94"
    );
    assert_eq!(
        (&hdr.derived_weights_offset as *const _ as usize) - base,
        96,
        "derived_weights_offset must start at 8-byte aligned offset 96"
    );
    assert_eq!(
        (&hdr._reserved as *const _ as usize) - base,
        104,
        "_reserved padding must start at offset 104"
    );
    assert_eq!(
        hdr._reserved.len(),
        24,
        "_reserved must be exactly 24 bytes (40 - 16 carved bytes)"
    );
    assert_eq!(104 + hdr._reserved.len(), 128);

    // 3. Verify zero compiler-inserted implicit padding gaps
    let field_sizes_sum = size_of::<u32>()   // magic
        + size_of::<u32>()                   // version
        + size_of::<u64>()                   // total_bytes
        + size_of::<u32>()                   // block_count
        + size_of::<u32>()                   // tokens_per_block
        + size_of::<u32>()                   // n_ctx
        + size_of::<u32>()                   // head_dim
        + size_of::<u32>()                   // num_layers
        + size_of::<u32>()                   // head_count_kv
        + size_of::<u32>()                   // vocab_size
        + size_of::<u32>()                   // active_slots
        + size_of::<u64>()                   // meta_table_offset
        + size_of::<u64>()                   // token_slots_offset
        + size_of::<u64>()                   // kv_blocks_offset
        + size_of::<u64>()                   // output_slabs_offset
        + size_of::<u64>()                   // generation_counter
        + size_of::<u32>()                   // swa_window_size
        + size_of::<u16>()                   // num_experts
        + size_of::<u16>()                   // num_experts_per_tok
        + size_of::<u64>()                   // derived_weights_offset
        + 24; // _reserved
    assert_eq!(
        field_sizes_sum, 128,
        "Sum of field sizes must equal struct size without compiler padding"
    );
}

#[test]
fn test_challenge_arena_header_backward_and_forward_compatibility() {
    // 1. Forward compatibility: Populating new fields and round-tripping through byte representation
    let hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: 1_048_576,
        block_count: 16,
        tokens_per_block: BLOCK_SIZE_TOKENS,
        n_ctx: 256,
        head_dim: 128,
        num_layers: 32,
        head_count_kv: 8,
        vocab_size: 32000,
        active_slots: 1,
        meta_table_offset: 128,
        token_slots_offset: 1152,
        kv_blocks_offset: 4096,
        output_slabs_offset: 65536,
        generation_counter: 42,
        swa_window_size: 4096,
        num_experts: 8,
        num_experts_per_tok: 2,
        derived_weights_offset: 131_072,
        _reserved: [0xAA; 24],
    };

    let bytes = hdr.as_bytes();
    assert_eq!(bytes.len(), 128);

    let recovered = ArenaHeader::read_from_bytes(bytes).expect("zerocopy read_from_bytes");
    assert_eq!(recovered, hdr);
    assert_eq!(recovered.swa_window_size, 4096);
    assert_eq!(recovered.num_experts, 8);
    assert_eq!(recovered.num_experts_per_tok, 2);
    assert_eq!(recovered.derived_weights_offset, 131_072);
    assert_eq!(recovered.version, 1);
    assert_eq!(recovered.validate(), Ok(()));

    // 2. Backward compatibility: Reading a legacy arena where bytes 88..128 were all zero
    let mut legacy_bytes = [0u8; 128];
    // Populate legacy header fields (pre-M4 layout)
    legacy_bytes[0..4].copy_from_slice(&ARENA_MAGIC.to_ne_bytes());
    legacy_bytes[4..8].copy_from_slice(&1u32.to_ne_bytes()); // ARENA_VERSION = 1
    legacy_bytes[8..16].copy_from_slice(&65536u64.to_ne_bytes()); // total_bytes
    legacy_bytes[16..20].copy_from_slice(&4u32.to_ne_bytes()); // block_count
    legacy_bytes[20..24].copy_from_slice(&16u32.to_ne_bytes()); // tokens_per_block = 16
    legacy_bytes[24..28].copy_from_slice(&64u32.to_ne_bytes()); // n_ctx = 64
    legacy_bytes[28..32].copy_from_slice(&64u32.to_ne_bytes()); // head_dim = 64
    legacy_bytes[32..36].copy_from_slice(&2u32.to_ne_bytes()); // num_layers = 2
    legacy_bytes[36..40].copy_from_slice(&2u32.to_ne_bytes()); // head_count_kv = 2
    legacy_bytes[40..44].copy_from_slice(&32000u32.to_ne_bytes()); // vocab_size = 32000
    legacy_bytes[44..48].copy_from_slice(&1u32.to_ne_bytes()); // active_slots = 1
    // offsets ...
    legacy_bytes[48..56].copy_from_slice(&128u64.to_ne_bytes());
    legacy_bytes[56..64].copy_from_slice(&384u64.to_ne_bytes());
    legacy_bytes[64..72].copy_from_slice(&1024u64.to_ne_bytes());
    legacy_bytes[72..80].copy_from_slice(&4096u64.to_ne_bytes());
    legacy_bytes[80..88].copy_from_slice(&1u64.to_ne_bytes());
    // Bytes 88..128 were the legacy `_reserved: [u8; 40]` and were all zeroes.

    let legacy_recovered =
        ArenaHeader::read_from_bytes(&legacy_bytes).expect("parse legacy header bytes");
    assert_eq!(
        legacy_recovered.validate(),
        Ok(()),
        "Legacy header must validate cleanly"
    );
    assert_eq!(
        legacy_recovered.swa_window_size, 0,
        "Zero-initialized legacy field must represent full attention (swa = 0)"
    );
    assert_eq!(
        legacy_recovered.num_experts, 0,
        "Zero-initialized legacy field must represent dense model (experts = 0)"
    );
    assert_eq!(
        legacy_recovered.num_experts_per_tok, 0,
        "Zero-initialized legacy field must represent dense model (experts_per_tok = 0)"
    );
    assert_eq!(
        legacy_recovered.derived_weights_offset, 0,
        "Zero-initialized legacy field must represent no derived weights offset"
    );
    assert_eq!(legacy_recovered._reserved, [0u8; 24]);

    // 3. Header corruption rejection
    let mut corrupt_magic = hdr;
    corrupt_magic.magic = 0xDEAD_BEEF;
    assert_eq!(
        corrupt_magic.validate(),
        Err("invalid arena magic signature")
    );

    let mut corrupt_version = hdr;
    corrupt_version.version = 2;
    assert_eq!(
        corrupt_version.validate(),
        Err("unsupported arena layout version")
    );

    let mut corrupt_tokens = hdr;
    corrupt_tokens.tokens_per_block = 32;
    assert_eq!(
        corrupt_tokens.validate(),
        Err("tokens_per_block must be 16")
    );
}

// ============================================================================
// Suite 2: Mixtral 8x7B & 8x22B Memlock Admission (28 GiB Ceiling)
// ============================================================================

#[test]
fn test_challenge_mixtral_8x7b_admission_under_28gib_ceiling() {
    assert_eq!(
        DEFAULT_MEMLOCK_LIMIT_BYTES,
        29_360_128 * 1024,
        "Target ceiling must match 29,360,128 KB (28.00 GiB)"
    );
    assert_eq!(
        DEFAULT_MEMLOCK_LIMIT_BYTES, 30_064_771_072,
        "Target ceiling must equal 30,064,771,072 bytes"
    );
    assert_eq!(
        MIXTRAL_KV_BYTES_PER_TOKEN,
        32 * 2 * 8 * 128 * 2,
        "Mixtral KV cache must be 131,072 bytes per token (128 KiB)"
    );

    // 2.1 Mixtral-8x7B Q4_0: ~24.5 GiB weights
    // At context_length = 4096: KV = 4096 * 131072 = 536,870,912 bytes (512 MiB)
    // Total = 26,300,000,000 + 536,870,912 = 26,836,870,912 bytes (~25.0 GiB) <= 28 GiB
    let manifest_q4_0_ok = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_0",
        4096,
        "sha256:dummy",
        0, // use default estimated weights for Q4_0
    );
    let res = evaluate_moe_memlock_admission(&manifest_q4_0_ok, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        res.is_ok(),
        "Mixtral-8x7B Q4_0 with 4K context must be admitted under 28 GiB ceiling: {res:?}"
    );

    // At context_length = 32,768 (32K): KV = 32768 * 131072 = 4,294,967,296 bytes (4 GiB)
    // Total = 26,300,000,000 + 4,294,967,296 = 30,594,967,296 bytes (~28.49 GiB) > 28 GiB
    let manifest_q4_0_32k =
        ModelManifest::new("gguf", "mixtral", "1.0.0", "Q4_0", 32768, "sha256:dummy", 0);
    let res = evaluate_moe_memlock_admission(&manifest_q4_0_32k, DEFAULT_MEMLOCK_LIMIT_BYTES);
    match res {
        Err(K3Error::InsufficientLockableMemory {
            required_bytes,
            weight_bytes,
            kv_bytes,
            available_bytes,
            rlimit_memlock_bytes,
        }) => {
            assert!(required_bytes > DEFAULT_MEMLOCK_LIMIT_BYTES);
            assert_eq!(available_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);
            assert_eq!(rlimit_memlock_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);
            assert_eq!(weight_bytes, 26_300_002_304); // 26,300,000,000 aligned up to 4096
            assert_eq!(kv_bytes, 4_294_967_296);
            assert_eq!(required_bytes, weight_bytes + kv_bytes);
        }
        other => {
            panic!("Expected InsufficientLockableMemory for Q4_0 at 32K context, got: {other:?}")
        }
    }

    // 2.2 Mixtral-8x7B Q4_K_M: ~27.0 GiB weights (28,990,000,000 bytes)
    // At context_length = 2048: KV = 2048 * 131072 = 268,435,456 bytes (256 MiB)
    // Total = 28,990,000,000 + 268,435,456 = 29,258,435,456 bytes (~27.25 GiB) <= 28 GiB
    let manifest_q4_k_m_2k = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_K_M",
        2048,
        "sha256:dummy",
        0,
    );
    let res = evaluate_moe_memlock_admission(&manifest_q4_k_m_2k, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        res.is_ok(),
        "Mixtral-8x7B Q4_K_M with 2K context must be admitted: {res:?}"
    );

    // At context_length = 8192: KV = 8192 * 131072 = 1,073,741,824 bytes (1 GiB)
    // Total = 28,990,000,000 + 1,073,741,824 = 30,063,741,824 bytes
    // At context_length = 9000: KV = 9000 * 131072 = 1,179,648,000 bytes
    // Total = 28,990,000,000 + 1,179,648,000 = 30,169,648,000 bytes > 30,064,771,072
    let manifest_q4_k_m_9k = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_K_M",
        9000,
        "sha256:dummy",
        0,
    );
    let res = evaluate_moe_memlock_admission(&manifest_q4_k_m_9k, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral-8x7B Q4_K_M with 9K context must be refused: {res:?}"
    );

    // 2.3 Mixtral-8x7B Q6_K: ~38.0 GiB weights -> Refused immediately even at context_length = 0
    let manifest_q6_k =
        ModelManifest::new("gguf", "mixtral", "1.0.0", "Q6_K", 0, "sha256:dummy", 0);
    let res = evaluate_moe_memlock_admission(&manifest_q6_k, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral-8x7B Q6_K must be refused upfront under 28 GiB ceiling: {res:?}"
    );

    // 2.4 Mixtral-8x7B Q8_0: ~49.5 GiB weights -> Refused immediately
    let manifest_q8_0 =
        ModelManifest::new("gguf", "mixtral", "1.0.0", "Q8_0", 0, "sha256:dummy", 0);
    let res = evaluate_moe_memlock_admission(&manifest_q8_0, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral-8x7B Q8_0 must be refused upfront: {res:?}"
    );

    // 2.5 Mixtral-8x7B F16 / BF16: ~100.2 GiB weights -> Refused immediately
    let manifest_f16 = ModelManifest::new("gguf", "mixtral", "1.0.0", "F16", 0, "sha256:dummy", 0);
    let res = evaluate_moe_memlock_admission(&manifest_f16, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral-8x7B F16 must be refused upfront: {res:?}"
    );

    // 2.6 Verify structured COMPACT_MOE_ADVISORY message
    assert!(
        COMPACT_MOE_ADVISORY.contains("Mixtral-8x7B in Q4_K_M exceeds 28 GiB memlock headroom")
    );
    assert!(COMPACT_MOE_ADVISORY.contains("Qwen3-30B-A3B"));
}

#[test]
fn test_challenge_mixtral_8x22b_admission_under_28gib_ceiling() {
    // Mixtral-8x22B has ~141B parameters.
    // In Q4_0, weights are ~80 GB.
    // In Q8_0, weights are ~145 GB.
    // In F16, weights are ~290 GB.
    // Manifests specifying explicit `weight_size_bytes` must be evaluated accurately.

    // 1. Mixtral 8x22B in Q4_0 (80 GB)
    let manifest_8x22b_q4 = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_0",
        2048,
        "sha256:8x22b_q4",
        80_000_000_000, // 80 GB
    );
    let res = evaluate_moe_memlock_admission(&manifest_8x22b_q4, DEFAULT_MEMLOCK_LIMIT_BYTES);
    match res {
        Err(K3Error::InsufficientLockableMemory {
            required_bytes,
            weight_bytes,
            kv_bytes,
            available_bytes,
            rlimit_memlock_bytes,
        }) => {
            assert!(required_bytes > 80_000_000_000);
            assert_eq!(weight_bytes, 80_000_000_000); // 80B is divisible by 4096
            assert_eq!(kv_bytes, 2048 * 131_072);
            assert_eq!(available_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);
            assert_eq!(rlimit_memlock_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);
        }
        other => {
            panic!("Expected InsufficientLockableMemory for Mixtral 8x22B Q4_0, got: {other:?}")
        }
    }

    // 2. Mixtral 8x22B in Q8_0 (145 GB)
    let manifest_8x22b_q8 = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q8_0",
        2048,
        "sha256:8x22b_q8",
        145_000_000_000,
    );
    let res = evaluate_moe_memlock_admission(&manifest_8x22b_q8, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral 8x22B Q8_0 must be refused: {res:?}"
    );

    // 3. Mixtral 8x22B in F16 (290 GB)
    let manifest_8x22b_f16 = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "F16",
        2048,
        "sha256:8x22b_f16",
        290_000_000_000,
    );
    let res = evaluate_moe_memlock_admission(&manifest_8x22b_f16, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Mixtral 8x22B F16 must be refused: {res:?}"
    );
}

#[test]
fn test_challenge_mixtral_admission_boundary_and_edge_cases() {
    // 1. Non-Mixtral architectures bypass the MoE admission check
    let llama_manifest = ModelManifest::new(
        "gguf",
        "llama",
        "1.0.0",
        "Q4_K_M",
        8192,
        "sha256:dummy",
        50_000_000_000, // 50 GB
    );
    assert_eq!(
        evaluate_moe_memlock_admission(&llama_manifest, DEFAULT_MEMLOCK_LIMIT_BYTES),
        Ok(()),
        "Non-MoE architecture (llama) must not be rejected by evaluate_moe_memlock_admission"
    );

    let mistral_dense_manifest = ModelManifest::new(
        "gguf",
        "mistral",
        "1.0.0",
        "Q4_K_M",
        8192,
        "sha256:dummy",
        50_000_000_000,
    );
    assert_eq!(
        evaluate_moe_memlock_admission(&mistral_dense_manifest, DEFAULT_MEMLOCK_LIMIT_BYTES),
        Ok(()),
        "Dense mistral must not be rejected by evaluate_moe_memlock_admission"
    );

    // 2. Case insensitivity check
    let manifest_caps =
        ModelManifest::new("gguf", "MIXTRAL", "1.0.0", "Q6_K", 0, "sha256:dummy", 0);
    assert!(
        matches!(
            evaluate_moe_memlock_admission(&manifest_caps, DEFAULT_MEMLOCK_LIMIT_BYTES),
            Err(K3Error::InsufficientLockableMemory { .. })
        ),
        "Architecture ID check must be case-insensitive"
    );

    // 3. Exact boundary threshold testing
    let ceiling = DEFAULT_MEMLOCK_LIMIT_BYTES; // 30,064,771,072
    let manifest_exact = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "custom",
        0,
        "sha256:exact",
        ceiling, // Exactly equal to ceiling
    );
    assert_eq!(
        evaluate_moe_memlock_admission(&manifest_exact, ceiling),
        Ok(()),
        "Required memory exactly equal to limit must pass"
    );

    let manifest_one_page_over = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "custom",
        0,
        "sha256:over",
        ceiling + 4096, // 1 page over ceiling
    );
    assert!(
        matches!(
            evaluate_moe_memlock_admission(&manifest_one_page_over, ceiling),
            Err(K3Error::InsufficientLockableMemory { .. })
        ),
        "Required memory exceeding ceiling by 1 page must be refused"
    );

    // 4. Custom memlock limit argument
    let tight_limit = 16 * 1024 * 1024 * 1024; // 16 GiB
    let manifest_q4_0 = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_0",
        2048,
        "sha256:q4_0",
        0, // default 24.5 GiB
    );
    let res = evaluate_moe_memlock_admission(&manifest_q4_0, tight_limit);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Custom 16 GiB limit must reject 24.5 GiB Q4_0"
    );

    // 5. Zero limit argument defaults cleanly to DEFAULT_MEMLOCK_LIMIT_BYTES
    let manifest_f16 = ModelManifest::new("gguf", "mixtral", "1.0.0", "F16", 0, "sha256:f16", 0);
    let res = evaluate_moe_memlock_admission(&manifest_f16, 0);
    match res {
        Err(K3Error::InsufficientLockableMemory {
            rlimit_memlock_bytes,
            ..
        }) => {
            assert_eq!(rlimit_memlock_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);
        }
        other => panic!("Expected InsufficientLockableMemory with default ceiling, got: {other:?}"),
    }

    // 6. Large context length (1,000,000 tokens): Cleanly rejected with InsufficientLockableMemory
    let manifest_1m_ctx = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_0",
        1_000_000,
        "sha256:1m_ctx_test",
        20_000_000_000,
    );
    let res = evaluate_moe_memlock_admission(&manifest_1m_ctx, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Large context length (1M tokens) must be rejected with InsufficientLockableMemory"
    );

    // 7. Empirical Challenge Vulnerability Probe: Integer overflow in align_up_u64
    // When context_length = u64::MAX, saturating_mul yields u64::MAX.
    // align_up_u64(val, 4096) computes (val + 4096 - 1) & !4095.
    // On val = u64::MAX, (val + 4095) overflows u64 and triggers a debug panic!
    let manifest_huge_ctx = ModelManifest::new(
        "gguf",
        "mixtral",
        "1.0.0",
        "Q4_0",
        u64::MAX,
        "sha256:overflow_test",
        20_000_000_000,
    );
    let res = evaluate_moe_memlock_admission(&manifest_huge_ctx, DEFAULT_MEMLOCK_LIMIT_BYTES);
    assert!(
        matches!(res, Err(K3Error::InsufficientLockableMemory { .. })),
        "Huge context length must be cleanly rejected with InsufficientLockableMemory without panicking"
    );
}

// ============================================================================
// Suite 3: Capability Mismatch Handling (Exit Code 10, No Segfaults)
// ============================================================================

#[test]
fn test_challenge_exit_code_10_mapping() {
    let err = K3Error::CapabilityMismatch {
        architecture: "falcon".to_string(),
        reason: "unsupported architecture".to_string(),
    };
    assert_eq!(err.diagnostic_code(), "capability_mismatch");

    let disp = format!("{err}");
    assert!(disp.starts_with("[capability_mismatch]"));
    assert!(disp.contains("falcon"));
    assert!(disp.contains("unsupported architecture"));
}

#[test]
fn test_challenge_k3_diag_process_exit_10_on_unsupported_architecture() {
    // Create synthetic GGUF v3 file with unsupported architecture "falcon"
    let model_bytes = create_synthetic_gguf_v3("falcon", 32, &[]);
    let model_path = std::env::temp_dir().join("k3_adversarial_falcon_arch.gguf");
    let mut f = File::create(&model_path).expect("create falcon gguf");
    f.write_all(&model_bytes).expect("write falcon gguf");
    drop(f);

    let output = Command::new(DIAG_BIN)
        .args(["--model", model_path.to_str().unwrap(), "--summary"])
        .output()
        .expect("execute k3-diag");

    let _ = std::fs::remove_file(&model_path);

    assert_eq!(
        output.status.code(),
        Some(10),
        "k3-diag must exit with code 10 on unsupported architecture"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[capability_mismatch]"),
        "stderr must contain [capability_mismatch]: {stderr}"
    );
    assert!(
        stderr.contains("falcon"),
        "stderr must identify unsupported architecture: {stderr}"
    );
}

#[test]
fn test_challenge_k3_diag_process_exit_10_on_unsupported_quantization() {
    // Create synthetic GGUF v3 file with architecture "llama", but with a tensor
    // of quantization Q4_0 (which LlamaRecipe explicitly does not support).
    let shape = [32usize, 32usize];
    let tensor_specs = [("blk.0.attn_q.weight", QuantType::Q4_0, shape.as_slice())];
    let model_bytes = create_synthetic_gguf_v3("llama", 32, &tensor_specs);
    let model_path = std::env::temp_dir().join("k3_adversarial_llama_q4_0.gguf");
    let mut f = File::create(&model_path).expect("create llama q4_0 gguf");
    f.write_all(&model_bytes).expect("write llama q4_0 gguf");
    drop(f);

    let output = Command::new(DIAG_BIN)
        .args(["--model", model_path.to_str().unwrap(), "--summary"])
        .output()
        .expect("execute k3-diag");

    let _ = std::fs::remove_file(&model_path);

    assert_eq!(
        output.status.code(),
        Some(10),
        "k3-diag must exit with code 10 on unsupported quantization format"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[capability_mismatch]"),
        "stderr must contain [capability_mismatch]: {stderr}"
    );
    assert!(
        stderr.contains("Q4_0"),
        "stderr must identify unsupported quantization format Q4_0: {stderr}"
    );
}

#[test]
fn test_challenge_k3_diag_process_exit_10_on_zero_context_length() {
    // Create synthetic GGUF v3 with context_length = 0
    let extra_kvs = [("synthetic.context_length", GgufValue::Uint32(0))];
    let model_bytes = create_synthetic_gguf_v3_with_kvs("synthetic", 32, &extra_kvs, &[]);
    let model_path = std::env::temp_dir().join("k3_adversarial_zero_ctx.gguf");
    let mut f = File::create(&model_path).expect("create zero ctx gguf");
    f.write_all(&model_bytes).expect("write zero ctx gguf");
    drop(f);

    let output = Command::new(DIAG_BIN)
        .args(["--model", model_path.to_str().unwrap(), "--summary"])
        .output()
        .expect("execute k3-diag");

    let _ = std::fs::remove_file(&model_path);

    assert_eq!(
        output.status.code(),
        Some(10),
        "k3-diag must exit with code 10 on zero context length"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[capability_mismatch]"),
        "stderr must contain [capability_mismatch]: {stderr}"
    );
    assert!(
        stderr.contains("must be greater than 0"),
        "stderr must explain context length requirement: {stderr}"
    );
}

#[test]
fn test_challenge_route_top2_experts_error_handling() {
    // 1. Empty logits -> CapabilityMismatch
    let empty_res = route_top2_experts(&[]);
    assert!(
        matches!(empty_res, Err(K3Error::CapabilityMismatch { .. })),
        "Empty logits must fail closed with CapabilityMismatch"
    );

    // 2. Single logit -> CapabilityMismatch (requires at least 2)
    let single_res = route_top2_experts(&[1.5]);
    assert!(
        matches!(single_res, Err(K3Error::CapabilityMismatch { .. })),
        "Single logit must fail closed with CapabilityMismatch"
    );

    // 3. Non-finite logit (NaN) -> CapabilityMismatch
    let nan_res = route_top2_experts(&[1.0, f32::NAN, 2.0]);
    assert!(
        matches!(nan_res, Err(K3Error::CapabilityMismatch { .. })),
        "NaN logit must fail closed with CapabilityMismatch"
    );

    // 4. Non-finite logit (+Inf) -> CapabilityMismatch
    let inf_res = route_top2_experts(&[1.0, f32::INFINITY, 2.0]);
    assert!(
        matches!(inf_res, Err(K3Error::CapabilityMismatch { .. })),
        "+Inf logit must fail closed with CapabilityMismatch"
    );

    // 5. Non-finite logit (-Inf) -> CapabilityMismatch
    let neg_inf_res = route_top2_experts(&[f32::NEG_INFINITY, 1.0, 2.0]);
    assert!(
        matches!(neg_inf_res, Err(K3Error::CapabilityMismatch { .. })),
        "-Inf logit must fail closed with CapabilityMismatch"
    );

    // 6. Valid logits: correctly identifies top 2 experts with normalized softmax
    let logits = [0.1, 2.5, 0.4, 1.8, -0.5, 0.0, 0.3, 0.2];
    let top2 = route_top2_experts(&logits).expect("route top 2");
    // Highest is index 1 (2.5), second highest is index 3 (1.8)
    assert_eq!(top2[0].0, 1, "Best expert must be index 1");
    assert_eq!(top2[1].0, 3, "Second expert must be index 3");
    let (w1, w2) = (top2[0].1, top2[1].1);
    assert!(w1 > w2, "Best expert must have higher softmax weight");
    assert!(
        (w1 + w2 - 1.0).abs() < 1e-6,
        "Weights must sum to 1.0: w1={w1}, w2={w2}"
    );
}
