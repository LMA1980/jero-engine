//! Integration regression tests for Wave 1 `jero-engine` tickets:
//! - `#94`: Mixed `Q5_0` (`ggml_type = 6`) & legacy/K-quant dispatch (`qwen2.5-coder:0.5b` layout)
//! - `#95`: `BitNet.cpp` `I2_S` (`ggml_type = 36`) admission & `SinglePromptEngine` execution
//! - `#98` & `#138`: Real `token_embd.weight` / `RoPE` GQA / `output.weight` logits matching scalar reference across `pos = 0..8`
//! - `#99`: Multi-threaded `0xFF00` / `0xFFFF` Rayon `WorkerPool` parity vs single-thread execution
//! - `#112`: `tokenizer.ggml.model = "gpt2"` whitespace (`0x20`), newline (`0x0A`), and `<|eot_id|>` stop decoding without `padding_token_id`

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use jero_core::test_helpers::write_gguf_value;
use jero_core::{
    ArchRecipe, CoordinationProfile, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error,
    ModelWeights, QuantType,
};
use jero_engine::affinity::{ClusterPhasePolicy, MockCpuTopology, WorkerPool};
use jero_engine::capability::verify_model_capabilities;
use jero_engine::engine::{SinglePromptEngine, StreamAction, sample_greedy_argmax};
use jero_engine::kv::SingleSlotKvArena;
use jero_engine::memory::check_memory_admission;
use jero_tokenizer::{DetokenizeStep, GgufTokenizer, Tokenizer};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION_V3: u32 = 3;

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    let len = u64::try_from(s.len()).expect("string len fits u64");
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn align_up(offset: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

/// Serializes a self-contained GGUF v3 binary with arbitrary KV metadata and raw tensor bytes.
fn build_custom_gguf_v3(
    alignment: usize,
    kv_pairs: &[(&str, GgufValue<'_>)],
    tensors: &[(&str, QuantType, Vec<usize>, Vec<u8>)],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());

    let tensor_count = u64::try_from(tensors.len()).expect("tensor count fits u64");
    let kv_count = u64::try_from(kv_pairs.len()).expect("kv count fits u64");
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&kv_count.to_le_bytes());

    for (k, v) in kv_pairs {
        write_gguf_string(&mut buf, k);
        write_gguf_value(&mut buf, v);
    }

    let mut rel_offset = 0usize;
    for (name, quant, shape, data) in tensors {
        let num_elements: usize = shape.iter().product();
        let expected_bytes = quant
            .storage_bytes(name, num_elements)
            .expect("valid quantized element count");
        assert_eq!(
            data.len(),
            expected_bytes,
            "tensor '{name}' payload len mismatch: got {}, expected {expected_bytes}",
            data.len()
        );

        write_gguf_string(&mut buf, name);
        let rank = u32::try_from(shape.len()).expect("rank fits u32");
        buf.extend_from_slice(&rank.to_le_bytes());
        for &dim in shape {
            let dim_u64 = u64::try_from(dim).expect("dim fits u64");
            buf.extend_from_slice(&dim_u64.to_le_bytes());
        }
        buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
        let rel_u64 = u64::try_from(rel_offset).expect("rel_offset fits u64");
        buf.extend_from_slice(&rel_u64.to_le_bytes());
        rel_offset += expected_bytes;
    }

    let aligned_header_len = align_up(buf.len(), alignment);
    buf.resize(aligned_header_len, 0);
    for (_, _, _, data) in tensors {
        buf.extend_from_slice(data);
    }
    buf
}

fn encode_f32_vec(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Fills quantized tensor payloads cleanly with safe non-zero `f16` scales (no NaN/Inf)
/// and deterministic pseudo-random weight bits so forward passes produce finite non-uniform logits.
fn make_valid_quant_bytes(name: &str, quant: QuantType, shape: &[usize], seed: u32) -> Vec<u8> {
    let num_elements: usize = shape.iter().product();
    let total_bytes = quant
        .storage_bytes(name, num_elements)
        .expect("storage_bytes");
    let scale_small: [u8; 2] = 0x2C00_u16.to_le_bytes(); // f16(0.0625)
    let min_small: [u8; 2] = 0x2400_u16.to_le_bytes(); // f16(0.015625)

    match quant {
        QuantType::F32 => {
            let mut vals = Vec::with_capacity(num_elements);
            for idx in 0..num_elements {
                let raw = ((idx as u32)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(seed.wrapping_mul(12345)))
                    % 29;
                #[allow(clippy::cast_precision_loss)]
                let v = (raw as f32 - 14.0) * 0.025;
                vals.push(v);
            }
            encode_f32_vec(&vals)
        }
        QuantType::Q4_0 | QuantType::IQ4_NL => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                for j in 2..block_bytes {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 + j as u32 * 7 + seed) % 251) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q4_1 => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                out[off + 2..off + 4].copy_from_slice(&min_small);
                for j in 4..block_bytes {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 5 + j as u32 * 11 + seed) % 241) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q5_0 => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                for j in 2..block_bytes {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 3 + j as u32 * 13 + seed) % 239) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q5_1 => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                out[off + 2..off + 4].copy_from_slice(&min_small);
                for j in 4..block_bytes {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 9 + j as u32 * 17 + seed) % 233) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q8_0 => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                for j in 2..block_bytes {
                    let q = ((b as i32 * 11 + j as i32 * 5 + seed as i32) % 61) - 30;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        out[off + j] = (q as i8) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q2_K => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                for j in 0..80 {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 + j as u32 * 3 + seed) % 127) as u8 | 0x11;
                    }
                }
                out[off + 80..off + 82].copy_from_slice(&scale_small);
                out[off + 82..off + 84].copy_from_slice(&min_small);
            }
            out
        }
        QuantType::Q3_K => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                for j in 0..108 {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 5 + j as u32 * 9 + seed) % 199) as u8;
                    }
                }
                out[off + 108..off + 110].copy_from_slice(&scale_small);
            }
            out
        }
        QuantType::Q4_K_M => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                out[off..off + 2].copy_from_slice(&scale_small);
                out[off + 2..off + 4].copy_from_slice(&min_small);
                for j in 4..block_bytes {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 7 + j as u32 * 13 + seed) % 181) as u8;
                    }
                }
            }
            out
        }
        QuantType::Q6_K => {
            let block_bytes = quant.block_bytes();
            let blocks = total_bytes / block_bytes;
            let mut out = vec![0u8; total_bytes];
            for b in 0..blocks {
                let off = b * block_bytes;
                for j in 0..208 {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[off + j] = ((b as u32 * 11 + j as u32 * 3 + seed) % 157) as u8;
                    }
                }
                out[off + 208..off + 210].copy_from_slice(&scale_small);
            }
            out
        }
        QuantType::I2_S => {
            let trailer_bytes = quant.scale_trailer_bytes();
            let payload_len = total_bytes - trailer_bytes;
            let mut out = vec![0u8; total_bytes];
            let patterns = [0x2B_u8, 0x9E, 0x64, 0x1B];
            for (idx, b) in out[..payload_len].iter_mut().enumerate() {
                *b = patterns[(idx + seed as usize) % patterns.len()];
            }
            let scale_f32: [u8; 4] = 0.25_f32.to_le_bytes();
            for slot in out[payload_len..].chunks_exact_mut(4) {
                slot.copy_from_slice(&scale_f32);
            }
            out
        }
        other => panic!("unsupported helper quant format {other:?}"),
    }
}

fn default_tokenizer_kvs<'a>(
    tokens: &'a [&'a str],
    extra_eos: u32,
) -> Vec<(&'a str, GgufValue<'a>)> {
    let scores = vec![0.0_f32; tokens.len()];
    let types = vec![1_i32; tokens.len()];
    vec![
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(tokens.to_vec())),
        ),
        (
            "tokenizer.ggml.scores",
            GgufValue::Array(GgufArray::Float32(scores)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(types)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(extra_eos)),
        ("tokenizer.ggml.padding_token_id", GgufValue::Uint32(0)),
    ]
}

#[test]
fn test_issue_94_qwen2_coder_mixed_legacy_and_k_quant_admission_and_forward() {
    let hidden = 256usize;
    let ffn = 256usize;
    let heads_q = 4usize;
    let heads_kv = 2usize;
    let head_dim = hidden / heads_q;
    let vocab = 16usize;

    let tokens = [
        "<pad>", "<s>", "</s>", "fn", " main", "()", " {", " return", " 0", ";", "}", " let", " x",
        " =", " 1", "\n",
    ];
    let mut kvs = vec![
        ("general.architecture", GgufValue::String("qwen2")),
        ("general.alignment", GgufValue::Uint32(32)),
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(hidden as u32)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(ffn as u32)),
        (
            "qwen2.attention.head_count",
            GgufValue::Uint32(heads_q as u32),
        ),
        (
            "qwen2.attention.head_count_kv",
            GgufValue::Uint32(heads_kv as u32),
        ),
        ("qwen2.context_length", GgufValue::Uint32(64)),
        ("qwen2.vocab_size", GgufValue::Uint32(vocab as u32)),
        ("qwen2.rope.freq_base", GgufValue::Float32(1_000_000.0)),
        (
            "qwen2.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(1e-6),
        ),
    ];
    kvs.extend(default_tokenizer_kvs(&tokens, 2));

    let ones = encode_f32_vec(&vec![1.0_f32; hidden]);
    let tensors = vec![
        (
            "token_embd.weight",
            QuantType::Q5_0,
            vec![hidden, vocab],
            make_valid_quant_bytes("token_embd.weight", QuantType::Q5_0, &[hidden, vocab], 1),
        ),
        (
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.attn_q.weight",
            QuantType::Q2_K,
            vec![hidden, heads_q * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_q.weight",
                QuantType::Q2_K,
                &[hidden, heads_q * head_dim],
                2,
            ),
        ),
        (
            "blk.0.attn_k.weight",
            QuantType::Q3_K,
            vec![hidden, heads_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_k.weight",
                QuantType::Q3_K,
                &[hidden, heads_kv * head_dim],
                3,
            ),
        ),
        (
            "blk.0.attn_v.weight",
            QuantType::Q5_0,
            vec![hidden, heads_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_v.weight",
                QuantType::Q5_0,
                &[hidden, heads_kv * head_dim],
                4,
            ),
        ),
        (
            "blk.0.attn_output.weight",
            QuantType::Q4_0,
            vec![heads_q * head_dim, hidden],
            make_valid_quant_bytes(
                "blk.0.attn_output.weight",
                QuantType::Q4_0,
                &[heads_q * head_dim, hidden],
                5,
            ),
        ),
        (
            "blk.0.attn_q.bias",
            QuantType::F32,
            vec![heads_q * head_dim],
            encode_f32_vec(&vec![0.01_f32; heads_q * head_dim]),
        ),
        (
            "blk.0.attn_k.bias",
            QuantType::F32,
            vec![heads_kv * head_dim],
            encode_f32_vec(&vec![-0.01_f32; heads_kv * head_dim]),
        ),
        (
            "blk.0.attn_v.bias",
            QuantType::F32,
            vec![heads_kv * head_dim],
            encode_f32_vec(&vec![0.02_f32; heads_kv * head_dim]),
        ),
        (
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.ffn_gate.weight",
            QuantType::Q5_0,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_gate.weight", QuantType::Q5_0, &[hidden, ffn], 6),
        ),
        (
            "blk.0.ffn_up.weight",
            QuantType::Q5_1,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_up.weight", QuantType::Q5_1, &[hidden, ffn], 7),
        ),
        (
            "blk.0.ffn_down.weight",
            QuantType::Q4_1,
            vec![ffn, hidden],
            make_valid_quant_bytes("blk.0.ffn_down.weight", QuantType::Q4_1, &[ffn, hidden], 8),
        ),
        ("output_norm.weight", QuantType::F32, vec![hidden], ones),
        (
            "output.weight",
            QuantType::Q6_K,
            vec![hidden, vocab],
            make_valid_quant_bytes("output.weight", QuantType::Q6_K, &[hidden, vocab], 9),
        ),
    ];

    let bytes = build_custom_gguf_v3(32, &kvs, &tensors);
    let gguf = GgufFile::parse(&bytes).expect("parse mixed-quant qwen2 gguf");
    assert_eq!(
        gguf.metadata()
            .find_tensor("blk.0.ffn_gate.weight")
            .expect("ffn_gate descriptor")
            .quant(),
        QuantType::Q5_0
    );

    let recipe = verify_model_capabilities(&gguf).expect("admit mixed Q5_0/Q5_1/Q4_1/Q2_K/Q3_K");
    assert_eq!(recipe.architecture_id(), "qwen2");
    check_memory_admission(gguf.metadata(), 64).expect("admission passes");

    let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv arena");
    let mut engine = SinglePromptEngine::new();
    let summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &gguf,
            &mut kv_arena,
            &[1, 3, 4],
            vocab,
            4,
            |_| Ok(StreamAction::Continue),
        )
        .expect("mixed-quant forward executes");
    assert_eq!(summary.generated_tokens.len(), 4);
    assert_eq!(summary.final_position, 6);
}

fn build_bitnet_i2_s_test_gguf() -> Vec<u8> {
    let hidden = 128usize;
    let ffn = 256usize;
    let heads = 4usize;
    let head_dim = hidden / heads;
    let vocab = 16usize;

    let tokens = [
        "<pad>", "<s>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    ];
    let mut kvs = vec![
        ("general.architecture", GgufValue::String("bitnet")),
        ("general.alignment", GgufValue::Uint32(32)),
        ("bitnet.block_count", GgufValue::Uint32(1)),
        ("bitnet.embedding_length", GgufValue::Uint32(hidden as u32)),
        ("bitnet.feed_forward_length", GgufValue::Uint32(ffn as u32)),
        (
            "bitnet.attention.head_count",
            GgufValue::Uint32(heads as u32),
        ),
        (
            "bitnet.attention.head_count_kv",
            GgufValue::Uint32(heads as u32),
        ),
        ("bitnet.context_length", GgufValue::Uint32(64)),
        ("bitnet.vocab_size", GgufValue::Uint32(vocab as u32)),
    ];
    kvs.extend(default_tokenizer_kvs(&tokens, 2));

    let ones_hidden = encode_f32_vec(&vec![1.0_f32; hidden]);
    let ones_ffn = encode_f32_vec(&vec![1.0_f32; ffn]);
    let tensors = vec![
        (
            "token_embd.weight",
            QuantType::Q8_0,
            vec![hidden, vocab],
            make_valid_quant_bytes("token_embd.weight", QuantType::Q8_0, &[hidden, vocab], 11),
        ),
        (
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones_hidden.clone(),
        ),
        (
            "blk.0.attn_q.weight",
            QuantType::I2_S,
            vec![hidden, heads * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_q.weight",
                QuantType::I2_S,
                &[hidden, heads * head_dim],
                12,
            ),
        ),
        (
            "blk.0.attn_k.weight",
            QuantType::I2_S,
            vec![hidden, heads * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_k.weight",
                QuantType::I2_S,
                &[hidden, heads * head_dim],
                13,
            ),
        ),
        (
            "blk.0.attn_v.weight",
            QuantType::I2_S,
            vec![hidden, heads * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_v.weight",
                QuantType::I2_S,
                &[hidden, heads * head_dim],
                14,
            ),
        ),
        (
            "blk.0.attn_sub_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones_hidden.clone(),
        ),
        (
            "blk.0.attn_output.weight",
            QuantType::I2_S,
            vec![heads * head_dim, hidden],
            make_valid_quant_bytes(
                "blk.0.attn_output.weight",
                QuantType::I2_S,
                &[heads * head_dim, hidden],
                15,
            ),
        ),
        (
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones_hidden.clone(),
        ),
        (
            "blk.0.ffn_gate.weight",
            QuantType::I2_S,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_gate.weight", QuantType::I2_S, &[hidden, ffn], 16),
        ),
        (
            "blk.0.ffn_up.weight",
            QuantType::I2_S,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_up.weight", QuantType::I2_S, &[hidden, ffn], 17),
        ),
        (
            "blk.0.ffn_sub_norm.weight",
            QuantType::F32,
            vec![ffn],
            ones_ffn,
        ),
        (
            "blk.0.ffn_down.weight",
            QuantType::I2_S,
            vec![ffn, hidden],
            make_valid_quant_bytes("blk.0.ffn_down.weight", QuantType::I2_S, &[ffn, hidden], 18),
        ),
        (
            "output_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones_hidden,
        ),
        (
            "output.weight",
            QuantType::Q8_0,
            vec![hidden, vocab],
            make_valid_quant_bytes("output.weight", QuantType::Q8_0, &[hidden, vocab], 19),
        ),
    ];

    build_custom_gguf_v3(32, &kvs, &tensors)
}

#[test]
fn test_issue_95_bitnet_i2_s_admission_and_forward() {
    let vocab = 16usize;
    let bytes = build_bitnet_i2_s_test_gguf();
    let gguf = GgufFile::parse(&bytes).expect("parse BitNet I2_S gguf");
    assert_eq!(
        gguf.metadata()
            .find_tensor("blk.0.attn_q.weight")
            .expect("I2_S attn_q tensor")
            .quant()
            .ggml_type_id(),
        36
    );

    let recipe = verify_model_capabilities(&gguf).expect("admit BitNet I2_S");
    assert_eq!(recipe.architecture_id(), "bitnet");
    check_memory_admission(gguf.metadata(), 64).expect("BitNet I2_S admission");

    let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv arena");
    let mut engine = SinglePromptEngine::new();
    let summary = engine
        .evaluate_prompt(
            recipe.as_ref(),
            &gguf,
            &mut kv_arena,
            &[1, 3, 4],
            vocab,
            4,
            |_| Ok(StreamAction::Continue),
        )
        .expect("BitNet I2_S prefill + decode completes");
    assert_eq!(summary.generated_tokens.len(), 4);
    assert_eq!(summary.final_position, 6);
}

fn scalar_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

fn scalar_matvec(weight_row_major: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let row = &weight_row_major[r * cols..(r + 1) * cols];
        out[r] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
    }
    out
}

fn scalar_rope_norm_inplace(vec: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    let n_heads = vec.len() / head_dim;
    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for pair in 0..(head_dim / 2) {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let angle = (pos as f32) / theta.powf(exponent);
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[2 * pair];
            let x1 = head[2 * pair + 1];
            head[2 * pair] = x0 * cos_a - x1 * sin_a;
            head[2 * pair + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

#[derive(Debug)]
struct LogitCaptureRecipe<'a> {
    inner: &'a dyn ArchRecipe,
    captured: Mutex<Vec<Vec<f32>>>,
}

impl ArchRecipe for LogitCaptureRecipe<'_> {
    fn architecture_id(&self) -> &'static str {
        self.inner.architecture_id()
    }

    fn requires_kv_cache(&self) -> bool {
        self.inner.requires_kv_cache()
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        self.inner.supports_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.inner.prefill_step(ctx, weights)?;
        self.captured
            .lock()
            .expect("lock captured")
            .push(ctx.logits().to_vec());
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.inner.decode_step(ctx, weights)?;
        self.captured
            .lock()
            .expect("lock captured")
            .push(ctx.logits().to_vec());
        Ok(())
    }
}

fn f32_tensor_slice(gguf: &GgufFile<'_>, name: &str) -> Vec<f32> {
    let view = gguf.tensor(name).expect("f32 tensor view");
    view.as_bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn build_llama_f32_test_gguf(context_length: u32) -> Vec<u8> {
    let hidden = 8usize;
    let ffn = 16usize;
    let n_q = 2usize;
    let n_kv = 1usize;
    let head_dim = hidden / n_q;
    let vocab = 16usize;
    let rms_eps = 1e-5_f32;
    let rope_theta = 10_000.0_f32;

    let tokens = [
        "<pad>", "<s>", "</s>", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
        "t13", "t14", "t15",
    ];
    let mut kvs = vec![
        ("general.architecture", GgufValue::String("llama")),
        ("general.alignment", GgufValue::Uint32(32)),
        ("llama.block_count", GgufValue::Uint32(1)),
        ("llama.embedding_length", GgufValue::Uint32(hidden as u32)),
        ("llama.feed_forward_length", GgufValue::Uint32(ffn as u32)),
        ("llama.attention.head_count", GgufValue::Uint32(n_q as u32)),
        (
            "llama.attention.head_count_kv",
            GgufValue::Uint32(n_kv as u32),
        ),
        (
            "llama.attention.key_length",
            GgufValue::Uint32(head_dim as u32),
        ),
        ("llama.context_length", GgufValue::Uint32(context_length)),
        ("llama.vocab_size", GgufValue::Uint32(vocab as u32)),
        ("llama.rope.freq_base", GgufValue::Float32(rope_theta)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(rms_eps),
        ),
    ];
    kvs.extend(default_tokenizer_kvs(&tokens, 2));

    let ones = encode_f32_vec(&vec![1.0_f32; hidden]);
    let tensors = vec![
        (
            "token_embd.weight",
            QuantType::F32,
            vec![hidden, vocab],
            make_valid_quant_bytes("token_embd.weight", QuantType::F32, &[hidden, vocab], 31),
        ),
        (
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.attn_q.weight",
            QuantType::F32,
            vec![hidden, n_q * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_q.weight",
                QuantType::F32,
                &[hidden, n_q * head_dim],
                32,
            ),
        ),
        (
            "blk.0.attn_k.weight",
            QuantType::F32,
            vec![hidden, n_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_k.weight",
                QuantType::F32,
                &[hidden, n_kv * head_dim],
                33,
            ),
        ),
        (
            "blk.0.attn_v.weight",
            QuantType::F32,
            vec![hidden, n_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_v.weight",
                QuantType::F32,
                &[hidden, n_kv * head_dim],
                34,
            ),
        ),
        (
            "blk.0.attn_output.weight",
            QuantType::F32,
            vec![n_q * head_dim, hidden],
            make_valid_quant_bytes(
                "blk.0.attn_output.weight",
                QuantType::F32,
                &[n_q * head_dim, hidden],
                35,
            ),
        ),
        (
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.ffn_gate.weight",
            QuantType::F32,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_gate.weight", QuantType::F32, &[hidden, ffn], 36),
        ),
        (
            "blk.0.ffn_up.weight",
            QuantType::F32,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_up.weight", QuantType::F32, &[hidden, ffn], 37),
        ),
        (
            "blk.0.ffn_down.weight",
            QuantType::F32,
            vec![ffn, hidden],
            make_valid_quant_bytes("blk.0.ffn_down.weight", QuantType::F32, &[ffn, hidden], 38),
        ),
        ("output_norm.weight", QuantType::F32, vec![hidden], ones),
        (
            "output.weight",
            QuantType::F32,
            vec![hidden, vocab],
            make_valid_quant_bytes("output.weight", QuantType::F32, &[hidden, vocab], 39),
        ),
    ];

    build_custom_gguf_v3(32, &kvs, &tensors)
}

#[test]
fn test_issue_98_and_138_single_prompt_engine_scalar_parity_pos_0_to_8() {
    let hidden = 8usize;
    let ffn = 16usize;
    let n_q = 2usize;
    let n_kv = 1usize;
    let head_dim = hidden / n_q;
    let vocab = 16usize;
    let rms_eps = 1e-5_f32;
    let rope_theta = 10_000.0_f32;

    let bytes = build_llama_f32_test_gguf(32);
    let gguf = GgufFile::parse(&bytes).expect("parse scalar parity llama gguf");
    let recipe = verify_model_capabilities(&gguf).expect("verify llama recipe");
    let observer = LogitCaptureRecipe {
        inner: recipe.as_ref(),
        captured: Mutex::new(Vec::new()),
    };

    let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
    let mut engine = SinglePromptEngine::new();
    let summary = engine
        .evaluate_prompt(&observer, &gguf, &mut kv_arena, &[3], vocab, 8, |_| {
            Ok(StreamAction::Continue)
        })
        .expect("engine forward pos 0..8 completes");
    assert_eq!(summary.generated_tokens.len(), 8);

    let captured_logits = observer.captured.lock().expect("lock captured");
    assert_eq!(captured_logits.len(), 8);

    let w_emb = f32_tensor_slice(&gguf, "token_embd.weight");
    let w_anorm = f32_tensor_slice(&gguf, "blk.0.attn_norm.weight");
    let w_q = f32_tensor_slice(&gguf, "blk.0.attn_q.weight");
    let w_k = f32_tensor_slice(&gguf, "blk.0.attn_k.weight");
    let w_v = f32_tensor_slice(&gguf, "blk.0.attn_v.weight");
    let w_o = f32_tensor_slice(&gguf, "blk.0.attn_output.weight");
    let w_fnorm = f32_tensor_slice(&gguf, "blk.0.ffn_norm.weight");
    let w_gate = f32_tensor_slice(&gguf, "blk.0.ffn_gate.weight");
    let w_up = f32_tensor_slice(&gguf, "blk.0.ffn_up.weight");
    let w_down = f32_tensor_slice(&gguf, "blk.0.ffn_down.weight");
    let w_onorm = f32_tensor_slice(&gguf, "output_norm.weight");
    let w_out = f32_tensor_slice(&gguf, "output.weight");

    let mut k_cache: Vec<Vec<f32>> = Vec::new();
    let mut v_cache: Vec<Vec<f32>> = Vec::new();
    let mut current_token = 3usize;

    for (pos, eng_logits) in captured_logits.iter().enumerate() {
        let mut x = w_emb[current_token * hidden..(current_token + 1) * hidden].to_vec();
        let x_norm = scalar_rmsnorm(&x, &w_anorm, rms_eps);
        let mut q = scalar_matvec(&w_q, &x_norm, n_q * head_dim, hidden);
        let mut k = scalar_matvec(&w_k, &x_norm, n_kv * head_dim, hidden);
        let v = scalar_matvec(&w_v, &x_norm, n_kv * head_dim, hidden);
        scalar_rope_norm_inplace(&mut q, head_dim, pos, rope_theta);
        scalar_rope_norm_inplace(&mut k, head_dim, pos, rope_theta);
        k_cache.push(k);
        v_cache.push(v);

        let group_size = n_q / n_kv;
        let mut attn_concat = vec![0.0_f32; n_q * head_dim];
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        for h in 0..n_q {
            let kv_h = h / group_size;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(k_cache.len());
            let mut max_s = f32::NEG_INFINITY;
            for kp in &k_cache {
                let kh = &kp[kv_h * head_dim..(kv_h + 1) * head_dim];
                let s = qh.iter().zip(kh.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                max_s = max_s.max(s);
                scores.push(s);
            }
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let out_h = &mut attn_concat[h * head_dim..(h + 1) * head_dim];
            for (t, &w) in exps.iter().enumerate() {
                let p = w / sum_exp;
                let vh = &v_cache[t][kv_h * head_dim..(kv_h + 1) * head_dim];
                for d in 0..head_dim {
                    out_h[d] += p * vh[d];
                }
            }
        }
        let attn_proj = scalar_matvec(&w_o, &attn_concat, hidden, n_q * head_dim);
        for d in 0..hidden {
            x[d] += attn_proj[d];
        }

        let f_norm = scalar_rmsnorm(&x, &w_fnorm, rms_eps);
        let gate = scalar_matvec(&w_gate, &f_norm, ffn, hidden);
        let up = scalar_matvec(&w_up, &f_norm, ffn, hidden);
        let act: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let ffn_proj = scalar_matvec(&w_down, &act, hidden, ffn);
        for d in 0..hidden {
            x[d] += ffn_proj[d];
        }

        let final_norm = scalar_rmsnorm(&x, &w_onorm, rms_eps);
        let ref_logits = scalar_matvec(&w_out, &final_norm, vocab, hidden);

        assert!(
            eng_logits.iter().all(|v| v.is_finite()),
            "pos {pos} produced non-finite logits"
        );
        let min_l = eng_logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max_l = eng_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_l - min_l) > 1e-4,
            "pos {pos} logits are uniform ({min_l}..{max_l})"
        );
        for (tok, (&el, &rl)) in eng_logits.iter().zip(ref_logits.iter()).enumerate() {
            assert!(
                (el - rl).abs() < 1e-4,
                "pos {pos} vocab {tok} logit mismatch: engine={el}, ref={rl}"
            );
        }
        let eng_top1 = sample_greedy_argmax(eng_logits).expect("engine argmax");
        let ref_top1 = sample_greedy_argmax(&ref_logits).expect("ref argmax");
        assert_eq!(eng_top1, ref_top1, "pos {pos} top-1 token mismatch");
        assert_eq!(
            summary.generated_tokens[pos], ref_top1,
            "pos {pos} generated_tokens parity mismatch"
        );
        current_token = ref_top1 as usize;
    }
}

#[test]
fn test_issue_99_single_prompt_engine_multithread_worker_pool_parity() {
    let hidden = 128usize;
    let ffn = 256usize;
    let n_q = 4usize;
    let n_kv = 2usize;
    let head_dim = hidden / n_q;
    let vocab = 128usize;

    let token_strings: Vec<String> = (0..vocab).map(|i| format!("tok{i}")).collect();
    let token_refs: Vec<&str> = token_strings.iter().map(String::as_str).collect();
    let mut kvs = vec![
        ("general.architecture", GgufValue::String("qwen2")),
        ("general.alignment", GgufValue::Uint32(32)),
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(hidden as u32)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(ffn as u32)),
        ("qwen2.attention.head_count", GgufValue::Uint32(n_q as u32)),
        (
            "qwen2.attention.head_count_kv",
            GgufValue::Uint32(n_kv as u32),
        ),
        ("qwen2.context_length", GgufValue::Uint32(64)),
        ("qwen2.vocab_size", GgufValue::Uint32(vocab as u32)),
    ];
    kvs.extend(default_tokenizer_kvs(&token_refs, 2));

    let ones = encode_f32_vec(&vec![1.0_f32; hidden]);
    let tensors = vec![
        (
            "token_embd.weight",
            QuantType::Q8_0,
            vec![hidden, vocab],
            make_valid_quant_bytes("token_embd.weight", QuantType::Q8_0, &[hidden, vocab], 41),
        ),
        (
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.attn_q.weight",
            QuantType::Q8_0,
            vec![hidden, n_q * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_q.weight",
                QuantType::Q8_0,
                &[hidden, n_q * head_dim],
                42,
            ),
        ),
        (
            "blk.0.attn_k.weight",
            QuantType::Q8_0,
            vec![hidden, n_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_k.weight",
                QuantType::Q8_0,
                &[hidden, n_kv * head_dim],
                43,
            ),
        ),
        (
            "blk.0.attn_v.weight",
            QuantType::Q8_0,
            vec![hidden, n_kv * head_dim],
            make_valid_quant_bytes(
                "blk.0.attn_v.weight",
                QuantType::Q8_0,
                &[hidden, n_kv * head_dim],
                44,
            ),
        ),
        (
            "blk.0.attn_output.weight",
            QuantType::Q8_0,
            vec![n_q * head_dim, hidden],
            make_valid_quant_bytes(
                "blk.0.attn_output.weight",
                QuantType::Q8_0,
                &[n_q * head_dim, hidden],
                45,
            ),
        ),
        (
            "blk.0.attn_q.bias",
            QuantType::F32,
            vec![n_q * head_dim],
            encode_f32_vec(&vec![0.01_f32; n_q * head_dim]),
        ),
        (
            "blk.0.attn_k.bias",
            QuantType::F32,
            vec![n_kv * head_dim],
            encode_f32_vec(&vec![-0.01_f32; n_kv * head_dim]),
        ),
        (
            "blk.0.attn_v.bias",
            QuantType::F32,
            vec![n_kv * head_dim],
            encode_f32_vec(&vec![0.02_f32; n_kv * head_dim]),
        ),
        (
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![hidden],
            ones.clone(),
        ),
        (
            "blk.0.ffn_gate.weight",
            QuantType::Q5_0,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_gate.weight", QuantType::Q5_0, &[hidden, ffn], 46),
        ),
        (
            "blk.0.ffn_up.weight",
            QuantType::Q5_0,
            vec![hidden, ffn],
            make_valid_quant_bytes("blk.0.ffn_up.weight", QuantType::Q5_0, &[hidden, ffn], 47),
        ),
        (
            "blk.0.ffn_down.weight",
            QuantType::Q8_0,
            vec![ffn, hidden],
            make_valid_quant_bytes("blk.0.ffn_down.weight", QuantType::Q8_0, &[ffn, hidden], 48),
        ),
        ("output_norm.weight", QuantType::F32, vec![hidden], ones),
        (
            "output.weight",
            QuantType::Q8_0,
            vec![hidden, vocab],
            make_valid_quant_bytes("output.weight", QuantType::Q8_0, &[hidden, vocab], 49),
        ),
    ];

    let bytes = build_custom_gguf_v3(32, &kvs, &tensors);
    let gguf = GgufFile::parse(&bytes).expect("parse multi-thread parity qwen2");
    let recipe = verify_model_capabilities(&gguf).expect("verify qwen2 recipe");

    // Verify 0xFF00 / 0xFFFF simulated SpacemiT K3 topology maps cleanly.
    let topo_16core = MockCpuTopology::spacemit_k3();
    let a100_resolved =
        jero_engine::resolve_affinity_with(Some("a100"), Some(0xFF00), Some(4), &topo_16core)
            .expect("resolve simulated 0xFF00 affinity");
    assert_eq!(a100_resolved.assigned_cores, vec![8, 9, 10, 11]);
    assert_eq!(a100_resolved.effective_mask, 0x0F00);

    let dual_resolved = jero_engine::affinity::resolve_affinity_split_with(
        Some("both"),
        Some("phase"),
        Some(0xFFFF),
        Some(14),
        &topo_16core,
    )
    .expect("resolve simulated 0xFFFF dual-cluster split affinity");
    assert!(dual_resolved.is_dual_cluster());
    assert_eq!(dual_resolved.x100_cores, vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(dual_resolved.a100_cores, vec![8, 9, 10, 11, 12, 13, 14]);

    let host_online_mask =
        jero_engine::CpuTopologyInspector::online_cpu_mask(&jero_engine::SystemCpuTopology::new())
            .expect("read host online CPU mask");
    let host_has_a100 = (host_online_mask & jero_engine::A100_CLUSTER_MASK) != 0;

    // Live OS thread pinning to cores 8..11 succeeds on 16-core K3 and non-Linux simulated hosts,
    // and fails closed with AffinityError on <16-core Linux CI hosts without online A100 cores.
    match WorkerPool::for_phase_policy_with(
        ClusterPhasePolicy::A100Only,
        gguf.metadata(),
        4,
        Some(0xFF00),
        Some(4),
        CoordinationProfile::Rayon,
        &topo_16core,
    ) {
        Ok(a100_pool) => {
            assert_eq!(a100_pool.worker_statuses().len(), 4);
        }
        Err(err) => {
            assert!(
                cfg!(target_os = "linux") && !host_has_a100,
                "unexpected 0xFF00 WorkerPool failure on host with online_mask=0x{host_online_mask:X}: {err}"
            );
            assert!(
                matches!(err, K3Error::AffinityError { .. }),
                "expected AffinityError on <16-core Linux runner, got {err:?}"
            );
        }
    }

    let pool_single = WorkerPool::for_phase_policy(
        ClusterPhasePolicy::X100Only,
        gguf.metadata(),
        4,
        Some(0x0001),
        Some(1),
        CoordinationProfile::Rayon,
    )
    .expect("single-thread pool");
    let x100_online = (host_online_mask & jero_engine::X100_CLUSTER_MASK).count_ones() as usize;
    let multi_threads = x100_online.clamp(1, 4);
    let pool_multi = WorkerPool::for_phase_policy(
        ClusterPhasePolicy::X100Only,
        gguf.metadata(),
        4,
        None,
        Some(multi_threads),
        CoordinationProfile::Rayon,
    )
    .expect("multi-thread X100 pool");
    let active_threads = pool_multi.install(rayon::current_num_threads);
    assert_eq!(active_threads, multi_threads);

    let obs_1 = LogitCaptureRecipe {
        inner: recipe.as_ref(),
        captured: Mutex::new(Vec::new()),
    };
    let obs_4 = LogitCaptureRecipe {
        inner: recipe.as_ref(),
        captured: Mutex::new(Vec::new()),
    };

    let mut kv1 = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv1");
    let mut kv4 = SingleSlotKvArena::new(gguf.metadata(), 64).expect("kv4");
    let mut eng1 = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);
    let mut eng4 = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);

    let sum1 = eng1
        .evaluate_prompt_with_pool(
            &obs_1,
            &gguf,
            &mut kv1,
            Some(&pool_single),
            &[1, 3, 5],
            vocab,
            4,
            |_| Ok(StreamAction::Continue),
        )
        .expect("single-thread run succeeds");
    let sum4 = eng4
        .evaluate_prompt_with_pool(
            &obs_4,
            &gguf,
            &mut kv4,
            Some(&pool_multi),
            &[1, 3, 5],
            vocab,
            4,
            |_| Ok(StreamAction::Continue),
        )
        .expect("multi-thread run succeeds");

    assert_eq!(sum1.generated_tokens, sum4.generated_tokens);
    let l1 = obs_1.captured.lock().expect("l1");
    let l4 = obs_4.captured.lock().expect("l4");
    assert_eq!(l1.len(), l4.len());
    for (step, (a, b)) in l1.iter().zip(l4.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {step} multi-thread logits differed from 1-thread"
        );
    }
}

#[test]
fn test_issue_112_gpt2_whitespace_newline_eot_stop_without_padding() {
    let tokens = [
        "<|begin_of_text|>",
        "<|eot_id|>",
        "hello",
        "\u{0120}world",
        "\u{010A}",
        "foo",
        "\u{0120}",
        "bar",
    ];
    let scores = vec![0.0_f32; tokens.len()];
    let token_types = vec![3_i32, 3, 1, 1, 1, 1, 1, 1];
    let merges = vec!["\u{0120} world", "h ello"];

    let kvs = vec![
        ("general.architecture", GgufValue::String("llama")),
        ("general.alignment", GgufValue::Uint32(32)),
        ("llama.block_count", GgufValue::Uint32(1)),
        ("llama.embedding_length", GgufValue::Uint32(32)),
        ("llama.feed_forward_length", GgufValue::Uint32(64)),
        ("llama.attention.head_count", GgufValue::Uint32(4)),
        ("llama.attention.head_count_kv", GgufValue::Uint32(4)),
        ("llama.context_length", GgufValue::Uint32(32)),
        ("llama.vocab_size", GgufValue::Uint32(tokens.len() as u32)),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        ("tokenizer.ggml.pre", GgufValue::String("llama3")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(tokens.to_vec())),
        ),
        (
            "tokenizer.ggml.scores",
            GgufValue::Array(GgufArray::Float32(scores)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(token_types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(merges)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
    ];

    let bytes = build_custom_gguf_v3(32, &kvs, &[]);
    let gguf = GgufFile::parse(&bytes).expect("parse gpt2 tokenizer gguf without padding_token_id");
    verify_model_capabilities(&gguf).expect("admit llama gpt2 gguf without padding_token_id");

    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load gpt2 tokenizer");
    assert_eq!(tokenizer.model_type(), "gpt2");
    assert_eq!(tokenizer.special_tokens().padding_token_id, None);
    assert!(tokenizer.uses_gpt2_byte_level());

    let prompt_ids = tokenizer
        .tokenize_prompt("hello world\nfoo", false, 32)
        .expect("spaces (0x20) and newlines (0x0A) encode without padding workaround");
    assert_eq!(prompt_ids, vec![2, 3, 4, 5]);

    #[derive(Debug)]
    struct EotSequenceRecipe;
    impl ArchRecipe for EotSequenceRecipe {
        fn architecture_id(&self) -> &'static str {
            "llama"
        }

        fn requires_kv_cache(&self) -> bool {
            true
        }

        fn supports_quant(&self, _quant: QuantType) -> bool {
            true
        }

        fn prefill_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let n = ctx.prompt_tokens().len();
            ctx.advance_position(n)?;
            ctx.logits_mut().fill(-10.0);
            ctx.logits_mut()[6] = 10.0;
            Ok(())
        }

        fn decode_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let pos = ctx.position();
            ctx.advance_position(1)?;
            ctx.logits_mut().fill(-10.0);
            if pos == 4 {
                ctx.logits_mut()[7] = 10.0;
            } else {
                ctx.logits_mut()[1] = 10.0;
            }
            Ok(())
        }
    }

    let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
    let mut engine = SinglePromptEngine::new();
    let mut detok = tokenizer.streaming_detokenizer();
    let mut streamed = String::new();

    let summary = engine
        .evaluate_prompt(
            &EotSequenceRecipe,
            &gguf,
            &mut kv_arena,
            &prompt_ids,
            32,
            8,
            |tok| match detok.push_token(tok)? {
                DetokenizeStep::Emit(s) => {
                    streamed.push_str(&s);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(tail) => {
                    streamed.push_str(&tail);
                    Ok(StreamAction::Stop)
                }
            },
        )
        .expect("streamed generation terminates cleanly on <|eot_id|>");

    streamed.push_str(&detok.finish());
    assert_eq!(streamed, " bar");
    assert_eq!(summary.generated_tokens, vec![6, 7, 1]);
}

#[test]
fn test_issue_133_dory_rope_profile_gguf_binary_admission() {
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_engine::capability::verify_model_capabilities;

    let layer_pattern = "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";
    let base_dory_kvs =
        |extra: Option<(&'static str, GgufValue<'static>)>| -> Vec<(&'static str, GgufValue<'static>)> {
            let mut kvs: Vec<(&'static str, GgufValue<'static>)> = vec![
                ("dory.block_count", GgufValue::Uint32(0)),
                ("dory.input_layer_count", GgufValue::Uint32(16)),
                ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
                ("dory.output_layer_count", GgufValue::Uint32(16)),
                ("dory.num_loops", GgufValue::Uint32(2)),
                ("dory.recurrent_kv_cache_mode", GgufValue::String("per_loop")),
                ("dory.layer_pattern", GgufValue::String(layer_pattern)),
                ("dory.rope.freq_base_2", GgufValue::Float32(10_000.0)),
                ("dory.init_std", GgufValue::Float32(0.02)),
                ("dory.alpha_init", GgufValue::Float32(0.05)),
                ("dory.gate_init", GgufValue::Float32(0.0)),
                ("dory.embedding_length", GgufValue::Uint32(256)),
                ("dory.attention.head_count", GgufValue::Uint32(4)),
                ("dory.attention.head_count_kv", GgufValue::Uint32(4)),
                ("dory.context_length", GgufValue::Uint32(2048)),
            ];
            if let Some(pair) = extra {
                kvs.push(pair);
            }
            kvs
        };

    for (profile_val, label, expected_profile) in [
        (None, "default", "dual_theta"),
        (Some(GgufValue::String("neox")), "string", "neox"),
        (Some(GgufValue::Uint32(1)), "u32", "profile_1"),
        (Some(GgufValue::Int32(2)), "non-negative i32", "profile_2"),
    ] {
        let kvs = base_dory_kvs(profile_val.map(|v| ("dory.rope.profile", v)));
        let bytes = create_synthetic_gguf_v3_with_kvs(
            "dory",
            32,
            &kvs,
            &[
                ("token_embd.weight", QuantType::Q4_K_M, &[256, 16]),
                ("blk.0.attn_q.weight", QuantType::Q5_K, &[256, 256]),
                ("output.weight", QuantType::Q6_K, &[256, 16]),
            ],
        );
        let gguf = GgufFile::parse(&bytes).unwrap_or_else(|e| panic!("parse dory {label}: {e}"));
        assert_eq!(
            gguf.metadata()
                .dory()
                .expect("dory metadata present")
                .rope_profile(),
            expected_profile
        );
        let recipe =
            verify_model_capabilities(&gguf).unwrap_or_else(|e| panic!("verify dory {label}: {e}"));
        assert_eq!(recipe.architecture_id(), "dory");
    }

    for bad_val in [GgufValue::Bool(true), GgufValue::Int32(-1)] {
        let kvs = base_dory_kvs(Some(("dory.rope.profile", bad_val)));
        let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &kvs, &[]);
        assert!(matches!(
            GgufFile::parse(&bytes),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    let empty_kvs = base_dory_kvs(Some(("dory.rope.profile", GgufValue::String("  "))));
    let empty_bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &empty_kvs, &[]);
    let empty_gguf = GgufFile::parse(&empty_bytes).expect("parse dory empty profile");
    assert!(matches!(
        verify_model_capabilities(&empty_gguf),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_issue_120_mixed_quant_gguf_admission_phi3_llama_mistral() {
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use jero_engine::capability::verify_model_capabilities;

    let mixed_tensors = [
        ("token_embd.weight", QuantType::Q6_K, &[256usize, 32][..]),
        ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 256]),
        ("blk.0.attn_k.weight", QuantType::Q4_0, &[256, 256]),
        ("blk.0.ffn_gate.weight", QuantType::Q5_K, &[256, 512]),
        ("output.weight", QuantType::Q8_0, &[256, 32]),
    ];

    for arch in ["phi3", "llama", "mistral"] {
        let bytes = create_synthetic_gguf_v3(arch, 32, &mixed_tensors);
        let model = GgufFile::parse(&bytes).expect("parse mixed quant gguf");
        let recipe = verify_model_capabilities(&model)
            .unwrap_or_else(|e| panic!("admit mixed quant {arch}: {e}"));
        assert_eq!(recipe.architecture_id(), arch);
    }
}

struct TempGgufFile(PathBuf);

impl TempGgufFile {
    fn write(tag: &str, bytes: &[u8]) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "jero_wave1_reg_{tag}_{}_{seq}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write temp gguf");
        Self(path)
    }

    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 temp gguf path")
    }
}

impl Drop for TempGgufFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run_jero_diag_json(args: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_jero-diag"))
        .args(args)
        .env("K3_CPU_ONLINE_MASK", "0xFFFF")
        .env("K3_SIMULATE_A100", "1")
        .output()
        .expect("execute jero-diag");
    assert!(
        output.status.success(),
        "jero-diag {:?} failed (status {:?})\nstdout: {}\nstderr: {}",
        args,
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    serde_json::from_slice(&output.stdout).expect("parse jero-diag --json output")
}

#[test]
fn test_issue_187_real_gguf_ime_on_emits_no_fabricated_comparison() {
    let gguf_file = TempGgufFile::write("issue187_llama", &build_llama_f32_test_gguf(64));
    let report = run_jero_diag_json(&[
        "--model",
        gguf_file.as_str(),
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--prompt",
        "t3 t4",
        "--max-tokens",
        "2",
        "--json",
    ]);

    assert_eq!(report["execution_mode"].as_str(), Some("real_gguf"));
    assert_eq!(report["architecture"].as_str(), Some("llama"));
    assert_eq!(report["ime_enabled"].as_bool(), Some(true));
    assert_eq!(report["ime_vmadot_issued"].as_u64(), Some(0));
    assert!(
        report["ime_comparison"].is_null(),
        "real GGUF --ime on must not fabricate an ime_comparison block: {}",
        report["ime_comparison"]
    );
    assert!(report["ime_delta_prefill_tps"].is_null());
    assert!(report["ime_delta_decode_tps"].is_null());
}

#[test]
fn test_issue_187_eval_suite_real_gguf_ime_on_never_fabricates_speedup() {
    let gguf_file = TempGgufFile::write("issue187_eval_llama", &build_llama_f32_test_gguf(64));
    let report = run_jero_diag_json(&[
        "--model",
        gguf_file.as_str(),
        "--eval-suite",
        "all",
        "--cluster",
        "a100",
        "--ime",
        "on",
        "--n-ctx",
        "64",
        "--max-tokens",
        "4",
        "--json",
    ]);

    assert_eq!(report["architecture"].as_str(), Some("llama"));
    assert_eq!(
        report["runtime_efficiency"]["ime_enabled"].as_bool(),
        Some(true)
    );
    assert_eq!(
        report["runtime_efficiency"]["ime_speedup_prefill_pct"].as_f64(),
        Some(0.0)
    );
    assert_eq!(
        report["runtime_efficiency"]["ime_speedup_decode_pct"].as_f64(),
        Some(0.0)
    );
}

#[test]
fn test_issue_188_phase_placement_reports_concrete_resolved_policy_and_real_cluster_masks() {
    let llama_file = TempGgufFile::write("issue188_llama", &build_llama_f32_test_gguf(64));
    let llama_report = run_jero_diag_json(&[
        "--model",
        llama_file.as_str(),
        "--cluster",
        "both",
        "--split",
        "phase",
        "--prompt",
        "t3 t4 t5 t6 t7",
        "--max-tokens",
        "2",
        "--json",
    ]);

    assert_eq!(
        llama_report["placement_prefill"]["matmul"][0]["cluster"].as_str(),
        Some("x100")
    );
    assert_eq!(
        llama_report["placement_prefill"]["vector"].as_str(),
        Some("x100")
    );
    assert_eq!(
        llama_report["placement_decode"]["matmul"][0]["cluster"].as_str(),
        Some("a100")
    );
    assert_eq!(
        llama_report["placement_decode"]["vector"].as_str(),
        Some("a100")
    );
    let llama_mask = llama_report["matmul_cpu_mask"]
        .as_u64()
        .expect("llama matmul_cpu_mask");
    assert_ne!(
        llama_mask & 0x00FF,
        0,
        "llama phase split must observe X100 prefill core bits (mask={llama_mask:#x})"
    );
    assert_ne!(
        llama_mask & 0xFF00,
        0,
        "llama phase split must observe A100 decode core bits (mask={llama_mask:#x})"
    );

    let bitnet_file = TempGgufFile::write("issue188_bitnet", &build_bitnet_i2_s_test_gguf());
    for mode_args in [
        vec!["--cluster", "both", "--split", "phase"],
        vec!["--phase-policy", "auto"],
    ] {
        let mut args = vec!["--model", bitnet_file.as_str()];
        args.extend(mode_args);
        args.extend(["--prompt", "a b c d e", "--max-tokens", "2", "--json"]);

        let bitnet_report = run_jero_diag_json(&args);
        assert_eq!(
            bitnet_report["placement_prefill"]["matmul"][0]["cluster"].as_str(),
            Some("a100"),
            "bitnet {args:?} must resolve prefill to a100"
        );
        assert_eq!(
            bitnet_report["placement_prefill"]["vector"].as_str(),
            Some("a100")
        );
        assert_eq!(
            bitnet_report["placement_decode"]["matmul"][0]["cluster"].as_str(),
            Some("a100"),
            "bitnet {args:?} must resolve decode to a100"
        );
        assert_eq!(
            bitnet_report["placement_decode"]["vector"].as_str(),
            Some("a100")
        );
        let bitnet_mask = bitnet_report["matmul_cpu_mask"]
            .as_u64()
            .expect("bitnet matmul_cpu_mask");
        assert_eq!(
            bitnet_mask & 0x00FF,
            0,
            "bitnet {args:?} must not execute on X100 cores (mask={bitnet_mask:#x})"
        );
        assert_ne!(
            bitnet_mask & 0xFF00,
            0,
            "bitnet {args:?} must observe A100 core bits (mask={bitnet_mask:#x})"
        );
    }
}
