//! Wave 2 (`LMA1980/jero-engine#149` / `#150` / `#151` / `#152` / `#153` / `#154` / `#155` / `#156`)
//! End-to-End Integration Certification Tests:
//! - `#152`: Instruct chat-template framing across ChatML, Llama-3, Phi-3/4, Gemma 2/3, Mistral,
//!   DeepSeek, placeholder templates, base model raw fallback, single-BOS coordination, and `--raw-prompt`.
//! - `#153`: Unclamped `--eval-suite` `--max-tokens` and `--n-ctx` execution (> 32 tokens) vs.
//!   conservative test defaults (`n_ctx = 64`, `max_tokens = 16`).
//! - `#155`: Multi-EOS/EOT auto-discovery & immediate halt across `SinglePromptEngine`, `jero-diag`,
//!   `--serve-openai`, and `jero-worker`, plus `--ignore-eos`.
//! - `#156`: `StopReason` (`eos_token` / `max_tokens`) telemetry in `PassTiming`, `MultiRunReport`
//!   JSON, OpenAI `finish_reason`, and `OutputSlabHeader` shared-memory IPC headers, plus EOT-free
//!   detokenized output.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use jero_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS, KvBlockMeta, OutputSlabHeader,
    SLAB_STATUS_WRITTEN, TokenInputSlotHeader,
};
use jero_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, IpcMessage, ShutdownCommand, StatusCode, recv_frame,
    recv_msg_frame, send_msg_frame,
};
use jero_core::ipc::shm::SharedMemoryArena;
use jero_core::test_helpers::write_gguf_value;
use jero_core::{GgufArray, GgufFile, GgufValue, QuantType};
use jero_engine::worker::{LockedWeights, WorkerConfig, WorkerEngine, decode_slab_stop_reason};
use jero_engine::{
    SinglePromptEngine, StopReason, discover_stop_token_ids, encode_formatted_prompt_utf8_safe,
    format_single_turn_prompt, tokenize_prompt_utf8_safe,
};
use jero_tokenizer::GgufTokenizer;
use serde_json::Value;
use zerocopy::{FromBytes, IntoBytes};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION_V3: u32 = 3;
static UNIQUE_TMP_ID: AtomicU64 = AtomicU64::new(1);

fn unique_temp_path(prefix: &str, ext: &str) -> std::path::PathBuf {
    let n = UNIQUE_TMP_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    let len = u64::try_from(s.len()).expect("string len fits u64");
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn align_up(offset: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

fn build_custom_gguf_v3(
    alignment: usize,
    kv_pairs: &[(&str, GgufValue<'_>)],
    tensors: &[(&str, QuantType, Vec<usize>, Vec<u8>)],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    buf.extend_from_slice(&u64::try_from(tensors.len()).expect("tc").to_le_bytes());
    buf.extend_from_slice(&u64::try_from(kv_pairs.len()).expect("kc").to_le_bytes());

    for (k, v) in kv_pairs {
        write_gguf_string(&mut buf, k);
        write_gguf_value(&mut buf, v);
    }

    let mut rel_offset = 0usize;
    for (name, quant, shape, data) in tensors {
        let num_elements: usize = shape.iter().product();
        let expected = quant
            .storage_bytes(name, num_elements)
            .expect("valid shape");
        assert_eq!(data.len(), expected, "tensor '{name}' byte size mismatch");

        write_gguf_string(&mut buf, name);
        buf.extend_from_slice(&u32::try_from(shape.len()).expect("rank").to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&u64::try_from(dim).expect("dim").to_le_bytes());
        }
        buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
        buf.extend_from_slice(&u64::try_from(rel_offset).expect("off").to_le_bytes());
        rel_offset += data.len();
    }

    let pad = align_up(buf.len(), alignment) - buf.len();
    buf.resize(buf.len() + pad, 0);
    for (_, _, _, data) in tensors {
        buf.extend_from_slice(data);
    }
    buf
}

fn f32_vec_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Builds a compact runnable 1-layer GGUF whose attention/FFN weights are zeroed so the residual
/// stream passes each step's `token_embd` row straight to `output_norm` + `output.weight`,
/// allowing deterministic scripted token emission across multiple autoregressive decode steps.
fn build_scripted_llama_or_phi3_gguf(
    arch: &str,
    vocab: &[&str],
    token_types: &[i32],
    bos_id: u32,
    eos_id: u32,
    eot_id: Option<u32>,
    add_bos: bool,
    chat_template: Option<&str>,
    n_ctx_train: u32,
    dim: usize,
    scripted_chain: &[u32],
    fallback_tok: u32,
) -> Vec<u8> {
    assert!(!scripted_chain.is_empty());
    assert!(
        scripted_chain.len() < dim,
        "scripted_chain len {} must be < dim {}",
        scripted_chain.len(),
        dim
    );
    let vocab_size = vocab.len();
    let n_heads = 2_u32;
    let n_kv_heads = 2_u32;
    let n_ff = dim;

    // Assign token embeddings to orthogonal coordinate directions cleanly:
    // - All prompt/non-chain tokens -> coordinate 0 -> selects `scripted_chain[0]`
    // - `scripted_chain[i]` -> coordinate (i + 1) -> selects `scripted_chain[i + 1]` (or `fallback_tok`)
    let mut token_embd = vec![0.0_f32; vocab_size * dim];
    for tok in 0..vocab_size {
        let base = tok * dim;
        if let Some(pos) = scripted_chain.iter().position(|&id| id as usize == tok) {
            let slot = (pos + 1).min(dim - 1);
            token_embd[base + slot] = 1.0;
        } else {
            token_embd[base] = 1.0;
        }
    }

    let mut output_weight = vec![0.0_f32; vocab_size * dim];
    // Slot 0 selects `scripted_chain[0]`
    output_weight[(scripted_chain[0] as usize) * dim] = 10.0;
    for (idx, _tok) in scripted_chain.iter().enumerate() {
        let slot = (idx + 1).min(dim - 1);
        let next_tok = scripted_chain.get(idx + 1).copied().unwrap_or(fallback_tok);
        output_weight[(next_tok as usize) * dim + slot] += 10.0;
    }

    let ones = vec![1.0_f32; dim];
    let zero_sq = vec![0.0_f32; dim * dim];

    let prefix = arch;
    let ctx_key = format!("{prefix}.context_length");
    let emb_key = format!("{prefix}.embedding_length");
    let ff_key = format!("{prefix}.feed_forward_length");
    let blk_key = format!("{prefix}.block_count");
    let hd_key = format!("{prefix}.attention.head_count");
    let kv_key = format!("{prefix}.attention.head_count_kv");
    let rms_key = format!("{prefix}.attention.layer_norm_rms_epsilon");

    let mut kvs: Vec<(&str, GgufValue<'_>)> = vec![
        ("general.architecture", GgufValue::String(arch)),
        (&ctx_key, GgufValue::Uint32(n_ctx_train)),
        (
            &emb_key,
            GgufValue::Uint32(u32::try_from(dim).expect("dim fits u32")),
        ),
        (
            &ff_key,
            GgufValue::Uint32(u32::try_from(n_ff).expect("n_ff fits u32")),
        ),
        (&blk_key, GgufValue::Uint32(1)),
        (&hd_key, GgufValue::Uint32(n_heads)),
        (&kv_key, GgufValue::Uint32(n_kv_heads)),
        (&rms_key, GgufValue::Float32(1.0e-5)),
        ("tokenizer.ggml.model", GgufValue::String("gpt2")),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(vocab.to_vec())),
        ),
        (
            "tokenizer.ggml.scores",
            GgufValue::Array(GgufArray::Float32(vec![0.0_f32; vocab_size])),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::Int32(token_types.to_vec())),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(bos_id)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(eos_id)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(add_bos)),
    ];
    if let Some(id) = eot_id {
        kvs.push(("tokenizer.ggml.eot_token_id", GgufValue::Uint32(id)));
    }
    if let Some(tmpl) = chat_template {
        kvs.push(("tokenizer.chat_template", GgufValue::String(tmpl)));
    }

    let tensors = vec![
        (
            "token_embd.weight",
            QuantType::F32,
            vec![dim, vocab_size],
            f32_vec_bytes(&token_embd),
        ),
        (
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![dim],
            f32_vec_bytes(&ones),
        ),
        (
            "blk.0.attn_q.weight",
            QuantType::F32,
            vec![dim, dim],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.attn_k.weight",
            QuantType::F32,
            vec![dim, dim],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.attn_v.weight",
            QuantType::F32,
            vec![dim, dim],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.attn_output.weight",
            QuantType::F32,
            vec![dim, dim],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![dim],
            f32_vec_bytes(&ones),
        ),
        (
            "blk.0.ffn_gate.weight",
            QuantType::F32,
            vec![dim, n_ff],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.ffn_up.weight",
            QuantType::F32,
            vec![dim, n_ff],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "blk.0.ffn_down.weight",
            QuantType::F32,
            vec![n_ff, dim],
            f32_vec_bytes(&zero_sq),
        ),
        (
            "output_norm.weight",
            QuantType::F32,
            vec![dim],
            f32_vec_bytes(&ones),
        ),
        (
            "output.weight",
            QuantType::F32,
            vec![dim, vocab_size],
            f32_vec_bytes(&output_weight),
        ),
    ];

    build_custom_gguf_v3(32, &kvs, &tensors)
}

fn send_http_post_json(addr: &str, path: &str, body: &Value) -> (u16, Value) {
    let body_str = serde_json::to_string(body).expect("serialize request body");
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body_str.len(),
        body_str
    );

    let mut stream = TcpStream::connect(addr).expect("connect to ephemeral openai server");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    stream.write_all(req.as_bytes()).expect("write HTTP POST");
    stream.flush().expect("flush HTTP POST");

    let mut raw_resp = Vec::new();
    stream
        .read_to_end(&mut raw_resp)
        .expect("read full HTTP response");
    let raw_str = String::from_utf8_lossy(&raw_resp);
    let (headers, json_body) = raw_str
        .split_once("\r\n\r\n")
        .expect("valid HTTP response framing");
    let status_line = headers.lines().next().expect("status line");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .expect("HTTP status code")
        .parse()
        .expect("u16 status code");
    let parsed: Value = serde_json::from_str(json_body).expect("valid JSON HTTP response body");
    (code, parsed)
}

#[test]
fn test_wave2_diag_chat_template_framing_single_bos_and_raw_prompt() {
    let vocab = [
        "<|begin_of_text|>", // 0
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "user",
        "assistant",
        "hi",
        "!",
        "hello",
        "world",
        "\n\n",
        " ",
    ];
    let mut types = vec![1_i32; vocab.len()];
    types[0] = 3;
    types[1] = 3;
    types[2] = 3;
    types[3] = 3;

    let l3_tmpl = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";
    let gguf_bytes = build_scripted_llama_or_phi3_gguf(
        "llama",
        &vocab,
        &types,
        0,
        3,
        Some(3),
        true,
        Some(l3_tmpl),
        256,
        8,
        &[8, 3],
        6,
    );

    let gguf = GgufFile::parse(&gguf_bytes).expect("parse scripted llama GGUF");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");

    // 1. Formatted prompt vs `--raw-prompt` token sequence & single-BOS invariant
    let formatted_str = format_single_turn_prompt(&tok, "hi");
    assert!(formatted_str.starts_with("<|begin_of_text|><|start_header_id|>user"));
    let templated_ids =
        tokenize_prompt_utf8_safe(&tok, "hi", true, 64).expect("templated prompt tokenization");
    assert_eq!(templated_ids[0], 0, "leading BOS present");
    assert_ne!(
        templated_ids[1], 0,
        "BOS must NOT be duplicated when add_bos_token=true"
    );
    assert_eq!(templated_ids.iter().filter(|&&id| id == 0).count(), 1);
    assert!(templated_ids.contains(&1));
    assert!(templated_ids.contains(&2));
    assert!(templated_ids.contains(&3));

    let raw_ids =
        tokenize_prompt_utf8_safe(&tok, "hi", false, 64).expect("raw prompt tokenization");
    assert_eq!(raw_ids, vec![0, 6]);
    assert_ne!(raw_ids, templated_ids);

    // 2. CLI subprocess test: default templated vs `--raw-prompt`
    let model_path = unique_temp_path("wave2_llama3_tmpl", "gguf");
    std::fs::write(&model_path, &gguf_bytes).expect("write GGUF");

    let diag_bin = env!("CARGO_BIN_EXE_jero-diag");
    let templated_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "hi",
            "--n-ctx",
            "64",
            "--max-tokens",
            "8",
        ])
        .output()
        .expect("run jero-diag default template");
    assert!(
        templated_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&templated_out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&templated_out.stdout).trim(),
        "hello",
        "must emit scripted token 'hello' and exclude trailing <|eot_id|>"
    );

    let raw_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--prompt",
            "hi",
            "--raw-prompt",
            "--repeat",
            "3",
            "--json",
        ])
        .output()
        .expect("run jero-diag --raw-prompt --json");
    assert!(
        raw_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&raw_out.stderr)
    );
    let report: Value =
        serde_json::from_slice(&raw_out.stdout).expect("parse JSON report for --raw-prompt");
    assert_eq!(report["prompt"], "hi");
    assert_eq!(report["prompt_tokens"], 2);

    // 3. Base model (no chat_template) falls back to raw prompt automatically
    let base_bytes = build_scripted_llama_or_phi3_gguf(
        "llama",
        &vocab,
        &types,
        0,
        3,
        None,
        true,
        None,
        256,
        8,
        &[8, 3],
        6,
    );
    let base_gguf = GgufFile::parse(&base_bytes).expect("parse base GGUF");
    let base_tok = GgufTokenizer::from_gguf(&base_gguf).expect("base tok");
    let base_ids = tokenize_prompt_utf8_safe(&base_tok, "hi", true, 64).expect("base tokenize");
    assert_eq!(
        base_ids,
        vec![0, 6],
        "base model without template must not wrap prompt"
    );

    // Also verify direct encode_formatted_prompt_utf8_safe never injects double BOS
    let explicit = encode_formatted_prompt_utf8_safe(&base_tok, "<|begin_of_text|>hi", true)
        .expect("direct encode");
    assert_eq!(explicit, vec![0, 6]);

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_wave2_diag_multi_eos_halt_ignore_eos_and_stop_reason_json() {
    // Primary EOS = 1 ("</s>"), Secondary EOT = 4 ("<|end|>") stored as UserDefined (type=4).
    // Scripted sequence: 6 ("O") -> 7 ("K") -> 4 ("<|end|>") -> 8 ("X") -> 9 ("Y")
    let vocab = [
        "<s>",
        "</s>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "?",
        "O",
        "K",
        "X",
        "Y",
        "\n",
    ];
    let mut types = vec![1_i32; vocab.len()];
    types[0] = 3;
    types[1] = 3;
    types[2] = 4;
    types[3] = 4;
    types[4] = 4;

    let phi_tmpl = "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}<|assistant|>\n";
    let gguf_bytes = build_scripted_llama_or_phi3_gguf(
        "phi3",
        &vocab,
        &types,
        0,
        1, // primary EOS = </s> (never generated)
        Some(4),
        false,
        Some(phi_tmpl),
        256,
        16,
        &[6, 7, 4, 8, 9],
        9,
    );
    let model_path = unique_temp_path("wave2_phi3_multi_eos", "gguf");
    std::fs::write(&model_path, &gguf_bytes).expect("write GGUF");

    let diag_bin = env!("CARGO_BIN_EXE_jero-diag");

    // 1. Default behavior: halts on secondary EOT (id 4 "<|end|>") BEFORE max_tokens (5),
    // reports `stop_reason: "eos_token"`, `stop_token_id: 4`, and excludes `<|end|>` from output.
    let stop_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--prompt",
            "?",
            "--max-tokens",
            "5",
            "--repeat",
            "3",
            "--json",
        ])
        .output()
        .expect("run jero-diag EOT halt JSON");
    assert!(
        stop_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&stop_out.stderr)
    );
    let stop_json: Value = serde_json::from_slice(&stop_out.stdout).expect("parse EOT JSON");
    assert_eq!(stop_json["stop_reason"], "eos_token");
    assert_eq!(stop_json["stop_token_id"], 4);
    for r in stop_json["runs"].as_array().expect("runs array") {
        assert_eq!(r["stop_reason"], "eos_token");
        assert_eq!(r["stop_token_id"], 4);
        assert_eq!(r["tokens"], serde_json::json!([6, 7, 4]));
    }

    // Stream mode excludes "<|end|>" from stdout completely
    let stream_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "?",
            "--max-tokens",
            "5",
        ])
        .output()
        .expect("run jero-diag stream");
    assert!(stream_out.status.success());
    let stream_text = String::from_utf8_lossy(&stream_out.stdout);
    assert_eq!(stream_text.trim(), "OK");
    assert!(!stream_text.contains("<|end|>"));

    // 2. `--ignore-eos`: bypasses EOT id 4 and runs all 5 requested tokens (`stop_reason: "max_tokens"`)
    let ignore_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--prompt",
            "?",
            "--max-tokens",
            "5",
            "--ignore-eos",
            "--repeat",
            "3",
            "--json",
        ])
        .output()
        .expect("run jero-diag --ignore-eos");
    assert!(
        ignore_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&ignore_out.stderr)
    );
    let ignore_json: Value =
        serde_json::from_slice(&ignore_out.stdout).expect("parse --ignore-eos JSON");
    assert_eq!(ignore_json["stop_reason"], "max_tokens");
    assert!(ignore_json.get("stop_token_id").is_none());
    for r in ignore_json["runs"].as_array().expect("runs array") {
        assert_eq!(r["stop_reason"], "max_tokens");
        assert!(r.get("stop_token_id").is_none());
        assert_eq!(r["tokens"], serde_json::json!([6, 7, 4, 8, 9]));
    }

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_wave2_eval_suite_unclamped_max_tokens_n_ctx_and_clean_eot_grading() {
    // Construct a 64-dim scripted model that emits 38 filler tokens ("a") BEFORE emitting "256" at
    // step 39 and `<|eot_id|>` at step 40.
    // Because 39 > 32, any `.min(32)` cap in `eval.rs` would truncate output before "256" and score 0%!
    let vocab = [
        "<|begin_of_text|>",   // 0
        "<|start_header_id|>", // 1
        "<|end_header_id|>",   // 2
        "<|eot_id|>",          // 3 (stop)
        "user",                // 4
        "assistant",           // 5
        "Q",                   // 6
        "a",                   // 7 (filler, chain steps 0..38)
        "256",                 // 8 (chain step 39)
        "\n\n",                // 9
        " ",                   // 10
    ];
    let mut types = vec![1_i32; vocab.len()];
    types[0] = 3;
    types[1] = 3;
    types[2] = 3;
    types[3] = 3;

    let l3_tmpl = "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";
    let mut chain = vec![7_u32; 38]; // steps 0..37 emit token 7 ("a")
    chain.push(8); // step 38 -> "256" (total emitted before EOT = 39 tokens > 32)
    chain.push(3); // step 39 -> <|eot_id|> (terminates at step 40)

    // Note: in `build_scripted_llama_or_phi3_gguf`, duplicate token IDs in `scripted_chain` would
    // reuse slot positions; give each of the 38 filler tokens its own distinct vocab token `" "` / `"a"`!
    let mut long_vocab: Vec<String> = vec![
        "<|begin_of_text|>".to_owned(),   // 0
        "<|start_header_id|>".to_owned(), // 1
        "<|end_header_id|>".to_owned(),   // 2
        "<|eot_id|>".to_owned(),          // 3
        "user".to_owned(),                // 4
        "assistant".to_owned(),           // 5
        "Q".to_owned(),                   // 6
        "\n\n".to_owned(),                // 7
    ];
    let mut long_chain = Vec::new();
    for _ in 0..36 {
        let id = u32::try_from(long_vocab.len()).expect("id");
        long_vocab.push(" ".to_owned());
        long_chain.push(id);
    }
    let ans_id = u32::try_from(long_vocab.len()).expect("ans id");
    long_vocab.push("256".to_owned());
    long_chain.push(ans_id);
    long_chain.push(3); // <|eot_id|> at step 37 (> 32!)

    let vocab_refs: Vec<&str> = long_vocab.iter().map(String::as_str).collect();
    let mut long_types = vec![1_i32; vocab_refs.len()];
    long_types[0] = 3;
    long_types[1] = 3;
    long_types[2] = 3;
    long_types[3] = 3;

    let _ = chain;
    let gguf_bytes = build_scripted_llama_or_phi3_gguf(
        "llama",
        &vocab_refs,
        &long_types,
        0,
        3,
        Some(3),
        true,
        Some(l3_tmpl),
        2048,
        64,
        &long_chain,
        3,
    );

    let model_path = unique_temp_path("wave2_eval_unclamped", "gguf");
    let dataset_path = unique_temp_path("wave2_eval_ds", "json");
    let export_path = unique_temp_path("wave2_eval_report", "json");
    std::fs::write(&model_path, &gguf_bytes).expect("write GGUF");

    let ds_item = serde_json::json!({
        "id": "gsm8k_unclamped_1",
        "suite": "gsm8k",
        "category": "gsm8k",
        "prompt": "Q",
        "expected_answer": "256",
        "scoring_method": "numerical_extract",
        "source_license": "MIT",
        "contamination_status": "verified_clean"
    });
    std::fs::write(&dataset_path, format!("{}\n", ds_item)).expect("write JSONL dataset");

    let diag_bin = env!("CARGO_BIN_EXE_jero-diag");
    let eval_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--eval-suite",
            "gsm8k",
            "--eval-dataset",
            dataset_path.to_str().unwrap(),
            "--n-ctx",
            "1024",
            "--max-tokens",
            "256",
            "--export-eval",
            export_path.to_str().unwrap(),
        ])
        .output()
        .expect("run jero-diag --eval-suite with --max-tokens 256 --n-ctx 1024");

    assert!(
        eval_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let report_str = std::fs::read_to_string(&export_path).expect("read exported eval report");
    let report: Value = serde_json::from_str(&report_str).expect("parse exported eval JSON");
    assert_eq!(report["categories"][0]["items_passed"], 1);
    assert_eq!(report["categories"][0]["items_evaluated"], 1);
    let score_pct = report["categories"][0]["score_pct"]
        .as_f64()
        .expect("score_pct f64");
    let fp16_ref = report["categories"][0]["fp16_reference_pct"]
        .as_f64()
        .expect("fp16_reference_pct f64");
    assert!(
        score_pct > 50.0 && (score_pct - fp16_ref).abs() < 1.0e-6,
        "expected unclamped 100% pass rate matching fp16_ref ({fp16_ref}) with --max-tokens 256, got {score_pct}"
    );

    // Control check: with --max-tokens 16, generation terminates BEFORE step 37 ("256"), so score == 0%
    let export_clamped_path = unique_temp_path("wave2_eval_clamped_ctrl", "json");
    let ctrl_out = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--eval-suite",
            "gsm8k",
            "--eval-dataset",
            dataset_path.to_str().unwrap(),
            "--n-ctx",
            "1024",
            "--max-tokens",
            "16",
            "--export-eval",
            export_clamped_path.to_str().unwrap(),
        ])
        .output()
        .expect("run control --max-tokens 16");
    assert!(ctrl_out.status.success());
    let ctrl_report: Value =
        serde_json::from_str(&std::fs::read_to_string(&export_clamped_path).unwrap()).unwrap();
    assert_eq!(ctrl_report["categories"][0]["items_passed"], 0);
    let _ = std::fs::remove_file(&export_clamped_path);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&dataset_path);
    let _ = std::fs::remove_file(&export_path);
}

#[test]
fn test_wave2_serve_openai_phi3_chat_template_and_eot_stop_finish_reason() {
    let vocab = [
        "<s>",
        "</s>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "Hello",
        "Yes",
        "!",
        "\n",
    ];
    let mut types = vec![1_i32; vocab.len()];
    types[0] = 3;
    types[1] = 3;
    types[2] = 4;
    types[3] = 4;
    types[4] = 4;

    let phi_tmpl = "<s>{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}<|assistant|>\n";
    let gguf_bytes = build_scripted_llama_or_phi3_gguf(
        "phi3",
        &vocab,
        &types,
        0,
        1,
        Some(4), // secondary EOT <|end|>
        true,
        Some(phi_tmpl),
        256,
        8,
        &[6, 4], // "Yes" -> "<|end|>"
        7,
    );
    let model_path = unique_temp_path("wave2_openai_phi3", "gguf");
    std::fs::write(&model_path, &gguf_bytes).expect("write phi3 GGUF");

    let diag_bin = env!("CARGO_BIN_EXE_jero-diag");
    let mut child = Command::new(diag_bin)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--cluster",
            "x100",
            "--serve-openai",
            "127.0.0.1:0",
            "--max-requests",
            "2",
            "--n-ctx",
            "64",
            "--max-tokens",
            "16",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn --serve-openai");

    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    let addr = loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read listen line");
        if n == 0 || Instant::now() > deadline {
            let _ = child.kill();
            panic!("failed to observe JERO_OPENAI_LISTEN_URL");
        }
        if let Some(url) = line.trim().strip_prefix("JERO_OPENAI_LISTEN_URL=http://") {
            break url.trim_end_matches("/v1").to_owned();
        }
    };

    // 1. /v1/chat/completions formats Phi-3 prompt cleanly and halts on `<|end|>` with finish_reason="stop"
    let (status_chat, resp_chat) = send_http_post_json(
        &addr,
        "/v1/chat/completions",
        &serde_json::json!({
            "model": "phi3-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16
        }),
    );
    assert_eq!(status_chat, 200);
    assert_eq!(resp_chat["choices"][0]["message"]["content"], "Yes");
    assert_eq!(resp_chat["choices"][0]["finish_reason"], "stop");

    // 2. /v1/completions also halts on secondary EOT `<|end|>` with finish_reason="stop"
    let (status_comp, resp_comp) = send_http_post_json(
        &addr,
        "/v1/completions",
        &serde_json::json!({
            "model": "phi3-test",
            "prompt": "Hello",
            "max_tokens": 16
        }),
    );
    assert_eq!(status_comp, 200);
    assert_eq!(resp_comp["choices"][0]["text"], "Yes");
    assert_eq!(resp_comp["choices"][0]["finish_reason"], "stop");

    let status = child.wait().expect("wait openai server exit");
    assert!(status.success());
    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_wave2_worker_ipc_stop_reason_in_output_slab_header() {
    let vocab = [
        "<s>",
        "</s>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "A",
        "B",
    ];
    let mut types = vec![1_i32; vocab.len()];
    types[0] = 3;
    types[1] = 3;
    types[2] = 4;
    types[3] = 4;
    types[4] = 4;

    // Scripted steps: 5 ("A") -> 4 ("<|end|>") -> 6 ("B")
    let gguf_bytes = build_scripted_llama_or_phi3_gguf(
        "phi3",
        &vocab,
        &types,
        0,
        1,
        Some(4),
        false,
        None,
        64,
        8,
        &[5, 4, 6],
        6,
    );
    let model_path = unique_temp_path("wave2_worker_stop", "gguf");
    std::fs::write(&model_path, &gguf_bytes).expect("write worker GGUF");

    let weights = LockedWeights::open(&model_path).expect("open locked weights");
    let worker_cfg = WorkerConfig {
        socket_path: None,
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(1),
        affinity: Some("0".to_owned()),
        verbose: false,
        listen: false,
    };
    let mut engine = WorkerEngine::try_new(worker_cfg, weights).expect("init WorkerEngine");
    assert!(engine.stop_token_ids().contains(&4));

    let (worker_sock, client_sock) = UnixStream::pair().expect("unix socketpair");
    let handle = thread::spawn(move || engine.run_event_loop(worker_sock));

    let arena_name = format!(
        "/k3_wave2_worker_slab_{}_{}",
        std::process::id(),
        UNIQUE_TMP_ID.fetch_add(1, Ordering::Relaxed)
    );
    let arena_len = 65_536_usize;
    let mut arena = SharedMemoryArena::create(&arena_name, arena_len).expect("create SHM arena");

    let meta_table_offset = 128_u64;
    let token_slots_offset = 1024_u64;
    let kv_blocks_offset = 4096_u64;
    let output_slabs_offset = 32_768_u64;

    let hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes: arena_len as u64,
        block_count: 4,
        tokens_per_block: BLOCK_SIZE_TOKENS,
        n_ctx: 4,
        head_dim: 4,
        num_layers: 1,
        head_count_kv: 2,
        vocab_size: 7,
        active_slots: 1,
        meta_table_offset,
        token_slots_offset,
        kv_blocks_offset,
        output_slabs_offset,
        generation_counter: 1,
        swa_window_size: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        derived_weights_offset: 0,
        _reserved: [0; 24],
    };
    arena
        .slice_mut(0, 128)
        .expect("hdr slice")
        .copy_from_slice(hdr.as_bytes());
    for block_idx in 0..4_u32 {
        let meta = KvBlockMeta::new(block_idx, 0);
        let off = (meta_table_offset as usize) + (block_idx as usize) * 64;
        arena
            .slice_mut(off, 64)
            .expect("meta slice")
            .copy_from_slice(meta.as_bytes());
    }

    // Write prefill prompt [0, 5]?; wait: `[0, 2]` (BOS + `<|user|>`, neither in chain) -> prefill hits token 5!
    let slot_hdr = TokenInputSlotHeader {
        slot_seq_id: 1,
        token_len: 4,
        start_position: 0,
    };
    arena
        .slice_mut(token_slots_offset as usize, 16)
        .expect("slot hdr")
        .copy_from_slice(slot_hdr.as_bytes());
    let toks_off = (token_slots_offset as usize) + 16;
    let initial_toks = [0_u32, 2_u32, 0_u32, 0_u32];
    for (i, &t) in initial_toks.iter().enumerate() {
        arena
            .slice_mut(toks_off + i * 4, 4)
            .expect("tok")
            .copy_from_slice(&t.to_le_bytes());
    }

    send_msg_frame(
        &client_sock,
        &IpcMessage::AttachArenaCommand(AttachArenaCommand {
            arena_id: 1,
            arena_bytes: arena_len as u64,
            n_ctx: 4,
            max_blocks: 4,
        }),
        Some(arena.as_raw_fd()),
    )
    .expect("send AttachArenaCommand");
    let (_, ack1, _) = recv_frame(client_sock.as_raw_fd()).expect("recv attach1 ack");
    assert_eq!(ack1.first().copied(), Some(0));

    // Step 1: Prefill tokens 0..2 -> emits 5 ("A"), end_pos = 2 < 4 -> slab stop reason None
    send_msg_frame(
        &client_sock,
        &IpcMessage::ForwardPassCommand(ForwardPassCommand {
            request_id: 10,
            sequence_id: 1,
            token_offset: 0,
            token_count: 2,
            block_indices: vec![0],
            is_prefill: true,
        }),
        None,
    )
    .expect("send prefill");
    let (msg1, _) = recv_msg_frame(&client_sock).expect("recv prefill reply");
    let IpcMessage::ForwardPassCompleted(done1) = msg1 else {
        panic!("expected ForwardPassCompleted, got {msg1:?}");
    };
    assert_eq!(done1.status, StatusCode::Success);
    assert_eq!(done1.next_token, 5);
    let slab1 = OutputSlabHeader::read_from_bytes(
        arena
            .slice(output_slabs_offset as usize, 64)
            .expect("slab1"),
    )
    .expect("parse slab1");
    assert_eq!(slab1.status, SLAB_STATUS_WRITTEN);
    assert_eq!(decode_slab_stop_reason(&slab1), None);

    // Step 2: Decode token 5 ("A") at pos 2 -> emits secondary EOT token 4 ("<|end|>")!
    let slot_dec1 = TokenInputSlotHeader {
        slot_seq_id: 1,
        token_len: 1,
        start_position: 2,
    };
    arena
        .slice_mut(token_slots_offset as usize, 16)
        .expect("slot dec1")
        .copy_from_slice(slot_dec1.as_bytes());
    arena
        .slice_mut(toks_off, 4)
        .expect("tok2")
        .copy_from_slice(&5_u32.to_le_bytes());
    send_msg_frame(
        &client_sock,
        &IpcMessage::ForwardPassCommand(ForwardPassCommand {
            request_id: 11,
            sequence_id: 1,
            token_offset: 0,
            token_count: 1,
            block_indices: vec![0],
            is_prefill: false,
        }),
        None,
    )
    .expect("send decode EOT");
    let (msg2, _) = recv_msg_frame(&client_sock).expect("recv decode reply");
    let IpcMessage::ForwardPassCompleted(done2) = msg2 else {
        panic!("expected ForwardPassCompleted, got {msg2:?}");
    };
    assert_eq!(done2.next_token, 4);
    let slab2 = OutputSlabHeader::read_from_bytes(
        arena
            .slice(output_slabs_offset as usize, 64)
            .expect("slab2"),
    )
    .expect("parse slab2");
    assert_eq!(
        decode_slab_stop_reason(&slab2),
        Some(StopReason::EosToken { token_id: 4 })
    );

    // Step 3: Decode non-stop token 4 -> transitions to 6 ("B") at pos 3 -> end_pos 4 == n_ctx (MaxTokens!)
    let slot_dec2 = TokenInputSlotHeader {
        slot_seq_id: 1,
        token_len: 1,
        start_position: 3,
    };
    arena
        .slice_mut(token_slots_offset as usize, 16)
        .expect("slot dec2")
        .copy_from_slice(slot_dec2.as_bytes());
    arena
        .slice_mut(toks_off, 4)
        .expect("tok3")
        .copy_from_slice(&4_u32.to_le_bytes());
    send_msg_frame(
        &client_sock,
        &IpcMessage::ForwardPassCommand(ForwardPassCommand {
            request_id: 12,
            sequence_id: 1,
            token_offset: 0,
            token_count: 1,
            block_indices: vec![0],
            is_prefill: false,
        }),
        None,
    )
    .expect("send decode MaxTokens");
    let (msg3, _) = recv_msg_frame(&client_sock).expect("recv max_tokens reply");
    let IpcMessage::ForwardPassCompleted(done3) = msg3 else {
        panic!("expected ForwardPassCompleted, got {msg3:?}");
    };
    assert_eq!(done3.next_token, 6);
    let slab3 = OutputSlabHeader::read_from_bytes(
        arena
            .slice(output_slabs_offset as usize, 64)
            .expect("slab3"),
    )
    .expect("parse slab3");
    assert_eq!(decode_slab_stop_reason(&slab3), Some(StopReason::MaxTokens));

    send_msg_frame(
        &client_sock,
        &IpcMessage::ShutdownCommand(ShutdownCommand { graceful: true }),
        None,
    )
    .expect("send shutdown");
    handle
        .join()
        .expect("join worker thread")
        .expect("clean worker exit");

    // Verify SinglePromptEngine unit alignment on exact same GGUF
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse");
    let tok = GgufTokenizer::from_gguf(&gguf).expect("tok");
    let stop_ids = discover_stop_token_ids(&tok);
    assert!(stop_ids.contains(&4));
    let _eng = SinglePromptEngine::new().with_stop_token_ids(stop_ids);

    let _ = std::fs::remove_file(&model_path);
}
