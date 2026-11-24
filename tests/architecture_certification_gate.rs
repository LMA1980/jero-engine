//! Integration tests for the Layered Architecture Certification Gate (ADR-045, US-26).
//!
//! Verifies:
//! 1. `jero-diag --dump-parity` golden logit & greedy-token parity and prompt sensitivity on an
//!    authentic 2-layer `llama` `Q8_0` GGUF fixture (`#179`, `#180`).
//! 2. Immediate fail-closed rejection of broken/stubbed `ArchRecipe` implementations that ignore
//!    `token_embd.weight` / `prompt_tokens` or emit degenerate logits (`#180`).
//! 3. `jero-diag --serve-openai 127.0.0.1:0 --max-requests 5` loopback HTTP wire compatibility
//!    across `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions` (including
//!    `echo: true` + `logprobs`), with bit-identical token output vs in-process `SinglePromptEngine`
//!    (`#176`, `#177`, `#180`).
//! 4. Honest `0.0` capability scoring without `.max(0.95)` floor inflation or substring cheats in
//!    `jero-diag --eval-suite` (`#146`, `#147`, `#148`).

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use jero_core::safetensors::SafeTensorsFile;
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights,
    QuantType, TensorView,
};
use jero_engine::{
    ClusterPhasePolicy, LockedModelWeights, ParityDumpReport, SinglePromptEngine,
    SingleSlotKvArena, StreamAction, SystemMemoryInspector, evaluate_parity_snapshot,
    verify_parity_against_golden, verify_prompt_sensitivity,
};
use jero_tokenizer::{DetokenizeStep, GgufTokenizer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const JERO_DIAG_BIN: &str = env!("CARGO_BIN_EXE_jero-diag");
const GOLDEN_LLAMA_Q8_0_JSON: &str =
    include_str!("fixtures/golden_parity/llama_authentic_q8_0.json");
#[cfg(feature = "jero-arch-gemma")]
const GOLDEN_GEMMA3_Q8_0_JSON: &str =
    include_str!("fixtures/golden_parity/gemma3_authentic_q8_0.json");
#[cfg(feature = "jero-arch-qwen")]
const GOLDEN_QWEN3_ST_GGUF_JSON: &str =
    include_str!("fixtures/golden_parity/qwen3_safetensors_gguf.json");
#[cfg(feature = "jero-arch-qwen")]
const GOLDEN_QWEN3_EMBEDDING_JSON: &str =
    include_str!("fixtures/golden_parity/qwen3_embedding_f32.json");
#[cfg(feature = "jero-arch-encoder")]
const GOLDEN_ENCODER_RERANK_JSON: &str =
    include_str!("fixtures/golden_parity/encoder_rerank_f32.json");
#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
const GOLDEN_VISION_MMPROJ_JSON: &str =
    include_str!("fixtures/golden_parity/vision_mmproj_f32.json");
#[cfg(feature = "jero-arch-whisper")]
const GOLDEN_WHISPER_ASR_JSON: &str = include_str!("fixtures/golden_parity/whisper_asr_f32.json");

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127 + 15;
    let mant = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7BFF
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            sign | ((exp as u16) << 10) | (mant as u16)
        }
    }
}

fn build_q8_0_matrix_bytes(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "Q8_0 requires cols multiple of 32");
    let blocks = rows * (cols / 32);
    let mut out = vec![0_u8; blocks * 34];
    let d = f32_to_f16_bits(0.025).to_le_bytes();
    for b in 0..blocks {
        let off = b * 34;
        out[off..off + 2].copy_from_slice(&d);
        for k in 0..32 {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss
            )]
            {
                let q = (((b * 19 + k * 7 + seed * 13) % 121) as i32 - 60) as i8;
                out[off + 2 + k] = q as u8;
            }
        }
    }
    out
}

fn populate_gguf_q8_0_payloads(raw: &mut [u8], zero_ctrl_output_rows: bool) {
    let descriptors = {
        let parsed = GgufFile::parse(raw).expect("parse template gguf");
        parsed.metadata().tensors().to_vec()
    };

    for (idx, desc) in descriptors.iter().enumerate() {
        let start = desc.absolute_offset();
        let end = start + desc.byte_len();
        if desc.shape().len() == 1 {
            for e in 0..desc.num_elements() {
                #[allow(clippy::cast_precision_loss)]
                let val = 0.85 + ((e % 13) as f32) * 0.02;
                let off = start + e * 4;
                raw[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
        } else {
            let cols = desc.shape()[0];
            let rows = desc.shape()[1];
            let mut payload = build_q8_0_matrix_bytes(rows, cols, idx + 1);
            // Suppress control tokens 0..=4 in `output.weight` so greedy decode emits normal words (5..15)
            if zero_ctrl_output_rows && desc.name() == "output.weight" {
                let row_bytes = (cols / 32) * 34;
                let zero_scale = f32_to_f16_bits(0.0).to_le_bytes();
                for ctrl_row in 0..5 {
                    for blk in 0..(cols / 32) {
                        let off = ctrl_row * row_bytes + blk * 34;
                        payload[off..off + 2].copy_from_slice(&zero_scale);
                        payload[off + 2..off + 34].fill(0);
                    }
                }
            }
            assert_eq!(payload.len(), end - start);
            raw[start..end].copy_from_slice(&payload);
        }
    }
}

/// Constructs a deterministic, compact 2-layer `llama` `Q8_0` GGUF v3 checkpoint with complete
/// transformer weights (`token_embd`, `attn_norm`, `attn_q/k/v/output`, `ffn_norm`,
/// `ffn_gate/up/down`, `output_norm`, `output.weight`).
fn build_authentic_llama_q8_0_gguf() -> Vec<u8> {
    let hidden = 128_usize;
    let ffn = 256_usize;
    let n_q = 4_usize;
    let n_kv = 2_usize;
    let head_dim = 32_usize;
    let vocab = 16_usize;

    let kvs = [
        ("llama.block_count", GgufValue::Uint32(2)),
        ("llama.embedding_length", GgufValue::Uint32(128)),
        ("llama.feed_forward_length", GgufValue::Uint32(256)),
        ("llama.attention.head_count", GgufValue::Uint32(4)),
        ("llama.attention.head_count_kv", GgufValue::Uint32(2)),
        ("llama.attention.key_length", GgufValue::Uint32(32)),
        ("llama.rope.freq_base", GgufValue::Float32(500_000.0)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(1.0e-5),
        ),
        ("llama.context_length", GgufValue::Uint32(128)),
        ("llama.vocab_size", GgufValue::Uint32(16)),
        // Ensure EOS does not prematurely halt 4-token deterministic decode trajectories
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
    ];

    let emb_shape = [hidden, vocab];
    let norm_shape = [hidden];
    let q_shape = [hidden, n_q * head_dim];
    let kv_shape = [hidden, n_kv * head_dim];
    let o_shape = [n_q * head_dim, hidden];
    let gate_up_shape = [hidden, ffn];
    let down_shape = [ffn, hidden];

    let specs: [(&str, QuantType, &[usize]); 21] = [
        ("token_embd.weight", QuantType::Q8_0, &emb_shape),
        ("blk.0.attn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.attn_q.weight", QuantType::Q8_0, &q_shape),
        ("blk.0.attn_k.weight", QuantType::Q8_0, &kv_shape),
        ("blk.0.attn_v.weight", QuantType::Q8_0, &kv_shape),
        ("blk.0.attn_output.weight", QuantType::Q8_0, &o_shape),
        ("blk.0.ffn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.ffn_gate.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.0.ffn_up.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.0.ffn_down.weight", QuantType::Q8_0, &down_shape),
        ("blk.1.attn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.1.attn_q.weight", QuantType::Q8_0, &q_shape),
        ("blk.1.attn_k.weight", QuantType::Q8_0, &kv_shape),
        ("blk.1.attn_v.weight", QuantType::Q8_0, &kv_shape),
        ("blk.1.attn_output.weight", QuantType::Q8_0, &o_shape),
        ("blk.1.ffn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.1.ffn_gate.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.1.ffn_up.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.1.ffn_down.weight", QuantType::Q8_0, &down_shape),
        ("output_norm.weight", QuantType::F32, &norm_shape),
        ("output.weight", QuantType::Q8_0, &emb_shape),
    ];

    let mut raw = create_synthetic_gguf_v3_with_kvs("llama", 32, &kvs, &specs);
    populate_gguf_q8_0_payloads(&mut raw, true);
    raw
}

fn write_temp_gguf(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = env::temp_dir().join(format!(
        "jero_cert_gate_{}_{}.gguf",
        tag,
        std::process::id()
    ));
    let mut f = File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    f.sync_all().expect("sync temp gguf");
    path
}

fn send_http_request(
    addr: &str,
    method: &str,
    path: &str,
    json_body: Option<&Value>,
) -> (u16, Value) {
    let mut stream = TcpStream::connect(addr).expect("connect to ephemeral server");
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .expect("set read timeout");
    let body_bytes = json_body.map_or_else(Vec::new, |v| serde_json::to_vec(v).expect("serialize"));
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(req.as_bytes()).expect("write request");
    if !body_bytes.is_empty() {
        stream.write_all(&body_bytes).expect("write body");
    }
    stream.flush().expect("flush request");

    let mut raw_resp = String::new();
    stream.read_to_string(&mut raw_resp).expect("read response");
    let (head, body) = raw_resp
        .split_once("\r\n\r\n")
        .expect("valid HTTP response delimiter");
    let status_line = head.lines().next().expect("status line");
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .expect("status code token")
        .parse()
        .expect("numeric status code");
    let parsed_body: Value = serde_json::from_str(body).expect("valid JSON response body");
    (status_code, parsed_body)
}

#[cfg(feature = "jero-arch-llama")]
#[test]
fn test_dump_parity_matches_golden_fixture_and_verifies_prompt_sensitivity() {
    let gguf_bytes = build_authentic_llama_q8_0_gguf();
    let model_path = write_temp_gguf("golden_parity", &gguf_bytes);
    let dump_a_path = env::temp_dir().join(format!("jero_parity_a_{}.json", std::process::id()));
    let dump_b_path = env::temp_dir().join(format!("jero_parity_b_{}.json", std::process::id()));

    let out_a = Command::new(JERO_DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "hello world",
            "--threads",
            "1",
            "--n-ctx",
            "32",
            "--max-tokens",
            "4",
            "--dump-parity",
            dump_a_path.to_str().unwrap(),
        ])
        .output()
        .expect("execute jero-diag --dump-parity prompt A");
    assert!(
        out_a.status.success(),
        "dump-parity A failed: {}",
        String::from_utf8_lossy(&out_a.stderr)
    );

    let out_b = Command::new(JERO_DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "Explain RVV",
            "--threads",
            "1",
            "--n-ctx",
            "32",
            "--max-tokens",
            "4",
            "--dump-parity",
            dump_b_path.to_str().unwrap(),
        ])
        .output()
        .expect("execute jero-diag --dump-parity prompt B");
    assert!(
        out_b.status.success(),
        "dump-parity B failed: {}",
        String::from_utf8_lossy(&out_b.stderr)
    );

    let dump_a: ParityDumpReport =
        serde_json::from_str(&std::fs::read_to_string(&dump_a_path).expect("read dump A"))
            .expect("parse dump A");
    let dump_b: ParityDumpReport =
        serde_json::from_str(&std::fs::read_to_string(&dump_b_path).expect("read dump B"))
            .expect("parse dump B");

    // 1. Prompt sensitivity MUST pass on authentic LlamaRecipe
    verify_prompt_sensitivity(&dump_a, &dump_b)
        .expect("authentic LlamaRecipe must produce distinct logits for distinct prompts");

    // Optional helper when refreshing golden fixture intentionally via JERO_UPDATE_GOLDEN=1
    if env::var("JERO_UPDATE_GOLDEN").as_deref() == Ok("1") {
        let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_parity/llama_authentic_q8_0.json");
        std::fs::create_dir_all(fixture_path.parent().unwrap()).unwrap();
        std::fs::write(
            &fixture_path,
            format!("{}\n", serde_json::to_string_pretty(&dump_a).unwrap()),
        )
        .unwrap();
    } else {
        let golden: ParityDumpReport =
            serde_json::from_str(GOLDEN_LLAMA_Q8_0_JSON).expect("parse golden parity JSON");
        verify_parity_against_golden(&dump_a, &golden, 0.9999, 1.0e-4)
            .expect("actual LlamaRecipe parity dump must match git-tracked golden fixture");
    }

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&dump_a_path);
    let _ = std::fs::remove_file(&dump_b_path);
}

/// Simulates a stubbed `ArchRecipe` that ignores `token_embd.weight` / `prompt_tokens`
/// (the exact failure mode that allowed non-working recipes to pass legacy smoke checks).
#[derive(Debug)]
struct BrokenEmbeddingBlindRecipe;

impl ArchRecipe for BrokenEmbeddingBlindRecipe {
    fn architecture_id(&self) -> &'static str {
        "broken-stub"
    }

    fn supports_quant(&self, _quant: QuantType) -> bool {
        true
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let len = ctx.prompt_tokens().len();
        ctx.advance_position(len)?;
        for (i, l) in ctx.logits_mut().iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            {
                *l = (i as f32) * 0.35 - 1.2;
            }
        }
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let tok = ctx.decode_token().unwrap_or(0) as usize;
        ctx.advance_position(1)?;
        let n = ctx.logits().len();
        for (i, l) in ctx.logits_mut().iter_mut().enumerate() {
            *l = if i == (tok + 3) % n { 4.0 } else { -0.5 };
        }
        Ok(())
    }
}

#[derive(Debug)]
struct EmptyWeights;

impl ModelWeights for EmptyWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "empty".to_owned(),
        })
    }

    fn tensor_count(&self) -> usize {
        0
    }

    fn total_weight_bytes(&self) -> usize {
        0
    }
}

#[test]
fn test_broken_embedding_blind_recipes_fail_certification_gate() {
    let stub = BrokenEmbeddingBlindRecipe;
    let weights = EmptyWeights;
    let mut kv_1 = SingleSlotKvArena::with_dimensions(2, 2, 32, 32).expect("kv 1");
    let mut kv_2 = SingleSlotKvArena::with_dimensions(2, 2, 32, 32).expect("kv 2");

    let (dump_1, _) = evaluate_parity_snapshot(
        &stub,
        &weights,
        &mut kv_1,
        None,
        ClusterPhasePolicy::Auto,
        "What is 17 + 25?",
        &[1, 5, 9],
        16,
        4,
        8,
        "Q8_0",
        |_, _| Ok(StreamAction::Continue),
    )
    .expect("stub snapshot 1");

    let (dump_2, _) = evaluate_parity_snapshot(
        &stub,
        &weights,
        &mut kv_2,
        None,
        ClusterPhasePolicy::Auto,
        "Explain SpacemiT K3 RVV",
        &[1, 11, 14, 15, 12],
        16,
        4,
        8,
        "Q8_0",
        |_, _| Ok(StreamAction::Continue),
    )
    .expect("stub snapshot 2");

    let rejection = verify_prompt_sensitivity(&dump_1, &dump_2)
        .expect_err("embedding-blind recipe MUST fail certification gate");
    assert!(rejection.contains("ignores token_embd.weight"));
}

#[cfg(feature = "jero-arch-llama")]
#[test]
#[allow(clippy::too_many_lines)]
fn test_serve_openai_ephemeral_server_matches_single_prompt_engine_bit_for_bit() {
    let gguf_bytes = build_authentic_llama_q8_0_gguf();
    let model_path = write_temp_gguf("serve_openai", &gguf_bytes);

    // Compute ground-truth in-process SinglePromptEngine outputs for chat and raw completion
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _) =
        LockedModelWeights::open_with_admission(&model_path, 64, &inspector).expect("open weights");
    let gguf_view = GgufFile::parse(locked_weights.as_bytes()).expect("parse gguf");
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view).expect("tokenizer");
    let mut engine = SinglePromptEngine::new();

    // Ground-truth chat completion for single user message "hello world"
    let chat_tokens = tokenizer
        .tokenize_prompt("hello world", true, 64)
        .expect("chat tokens");
    let mut kv_chat = SingleSlotKvArena::from_gguf(&gguf_view, 64).expect("kv chat");
    let mut detok_chat = tokenizer.streaming_detokenizer();
    let mut expected_chat_text = String::new();
    engine
        .evaluate_prompt(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_chat,
            &chat_tokens,
            tokenizer.vocab_size(),
            4,
            |tok| match detok_chat.push_token(tok)? {
                DetokenizeStep::Emit(p) => {
                    expected_chat_text.push_str(&p);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(p) => {
                    expected_chat_text.push_str(&p);
                    Ok(StreamAction::Stop)
                }
            },
        )
        .expect("in-process chat eval");
    if !detok_chat.is_stopped() {
        expected_chat_text.push_str(&detok_chat.finish());
    }

    // Ground-truth raw completion for "Explain RVV"
    let raw_tokens = tokenizer
        .tokenize_prompt("Explain RVV", false, 64)
        .expect("raw tokens");
    let mut kv_raw = SingleSlotKvArena::from_gguf(&gguf_view, 64).expect("kv raw");
    let mut detok_raw = tokenizer.streaming_detokenizer();
    let mut expected_raw_text = String::new();
    engine
        .evaluate_prompt(
            recipe.as_ref(),
            &locked_weights,
            &mut kv_raw,
            &raw_tokens,
            tokenizer.vocab_size(),
            4,
            |tok| match detok_raw.push_token(tok)? {
                DetokenizeStep::Emit(p) => {
                    expected_raw_text.push_str(&p);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(p) => {
                    expected_raw_text.push_str(&p);
                    Ok(StreamAction::Stop)
                }
            },
        )
        .expect("in-process raw eval");
    if !detok_raw.is_stopped() {
        expected_raw_text.push_str(&detok_raw.finish());
    }

    drop(kv_chat);
    drop(kv_raw);
    drop(locked_weights);

    // Spawn `jero-diag --serve-openai 127.0.0.1:0 --max-requests 5`
    let mut child = Command::new(JERO_DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--threads",
            "1",
            "--n-ctx",
            "64",
            "--max-tokens",
            "4",
            "--serve-openai",
            "127.0.0.1:0",
            "--max-requests",
            "5",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jero-diag --serve-openai");

    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout);
    let mut listen_line = String::new();
    reader
        .read_line(&mut listen_line)
        .expect("read JERO_OPENAI_LISTEN_URL line");
    let base_url = listen_line
        .trim()
        .strip_prefix("JERO_OPENAI_LISTEN_URL=")
        .expect("stdout must begin with JERO_OPENAI_LISTEN_URL=");
    let host_port = base_url
        .strip_prefix("http://")
        .and_then(|s| s.strip_suffix("/v1"))
        .expect("valid http://<ip:port>/v1 URL");

    // Request 1: GET /health
    let (st_health, health_json) = send_http_request(host_port, "GET", "/health", None);
    assert_eq!(st_health, 200);
    assert_eq!(health_json["status"], "ok");
    assert_eq!(health_json["architecture"], "llama");

    // Request 2: GET /v1/models
    let (st_models, models_json) = send_http_request(host_port, "GET", "/v1/models", None);
    assert_eq!(st_models, 200);
    assert_eq!(models_json["object"], "list");
    assert_eq!(models_json["data"].as_array().unwrap().len(), 1);

    // Request 3: POST /v1/chat/completions
    let chat_req = json!({
        "model": "llama-cert",
        "messages": [{"role": "user", "content": "hello world"}],
        "temperature": 0,
        "max_tokens": 4
    });
    let (st_chat, chat_resp) =
        send_http_request(host_port, "POST", "/v1/chat/completions", Some(&chat_req));
    assert_eq!(st_chat, 200);
    assert_eq!(chat_resp["object"], "chat.completion");
    let actual_chat_text = chat_resp["choices"][0]["message"]["content"]
        .as_str()
        .expect("chat content string");
    assert_ne!(actual_chat_text, "Hello! How can I assist you today?");
    assert_eq!(actual_chat_text, expected_chat_text);

    // Request 4: POST /v1/completions (raw completion)
    let cmpl_req = json!({
        "model": "llama-cert",
        "prompt": "Explain RVV",
        "temperature": 0,
        "max_tokens": 4
    });
    let (st_cmpl, cmpl_resp) =
        send_http_request(host_port, "POST", "/v1/completions", Some(&cmpl_req));
    assert_eq!(st_cmpl, 200);
    assert_eq!(cmpl_resp["object"], "text_completion");
    let actual_raw_text = cmpl_resp["choices"][0]["text"]
        .as_str()
        .expect("completion text string");
    assert_eq!(actual_raw_text, expected_raw_text);

    // Request 5: POST /v1/completions with echo=true & logprobs=3 (lm_eval local-completions contract)
    let lp_req = json!({
        "model": "llama-cert",
        "prompt": "hello world",
        "max_tokens": 0,
        "echo": true,
        "logprobs": 3
    });
    let (st_lp, lp_resp) = send_http_request(host_port, "POST", "/v1/completions", Some(&lp_req));
    assert_eq!(st_lp, 200);
    let lp_obj = &lp_resp["choices"][0]["logprobs"];
    let tok_lps = lp_obj["token_logprobs"].as_array().expect("token_logprobs");
    assert!(tok_lps.len() >= 2);
    assert!(tok_lps[0].is_null());
    for entry in &tok_lps[1..] {
        let v = entry.as_f64().expect("finite logprob f64");
        assert!(v.is_finite() && v <= 0.0);
    }

    // Server must exit cleanly with exit code 0 after handling 5 requests
    let deadline = Instant::now() + Duration::from_secs(5);
    let exit_status = loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "server did not exit after --max-requests 5"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(exit_status.success());

    let _ = std::fs::remove_file(&model_path);
}

#[cfg(feature = "jero-arch-llama")]
#[test]
fn test_eval_suite_reports_honest_zero_score_for_wrong_answers_without_floor_inflation() {
    let gguf_bytes = build_authentic_llama_q8_0_gguf();
    let model_path = write_temp_gguf("honest_zero_eval", &gguf_bytes);

    let output = Command::new(JERO_DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--eval-suite",
            "gsm8k",
            "--threads",
            "1",
            "--n-ctx",
            "64",
            "--max-tokens",
            "4",
            "--json",
        ])
        .output()
        .expect("run jero-diag --eval-suite gsm8k");

    let _ = std::fs::remove_file(&model_path);

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("valid JSON report");
    assert_eq!(report["architecture"], "llama");
    let cat = &report["categories"][0];
    assert_eq!(cat["suite"], "gsm8k");
    assert_eq!(cat["items_passed"], 0);
    assert_eq!(
        cat["score_pct"].as_f64().unwrap(),
        0.0,
        "0/2 passed items must score 0.0%, never >= 0.95 * fp16_ref"
    );
    assert_eq!(report["benchlm_composite_index"].as_f64().unwrap(), 0.0);
    assert!(
        report["runtime_efficiency"]["prefill_tps"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[allow(dead_code)]
fn deterministic_f32_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let v = ((((i * 17 + seed * 13 + 7) % 29) as f32) - 14.0) * scale;
            v
        })
        .collect()
}

#[allow(dead_code)]
fn f32_slice_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(dead_code)]
fn unit_perturbed_norm_vec(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let delta = ((i + seed) % 7) as f32 * 0.03;
            0.9 + delta
        })
        .collect()
}

#[allow(dead_code)]
fn patch_gguf_f32_payloads(raw: &mut [u8], specs: &[(&str, QuantType, Vec<usize>, Vec<u8>)]) {
    let ranges: Vec<(&str, usize, usize, &[u8])> = {
        let parsed = GgufFile::parse(raw).expect("parse template gguf for patching");
        specs
            .iter()
            .map(|(name, _, _, bytes)| {
                let view = parsed.tensor(name).expect("tensor present for patching");
                (*name, view.offset(), view.byte_len(), bytes.as_slice())
            })
            .collect()
    };
    for (name, start, len, bytes) in ranges {
        assert_eq!(bytes.len(), len, "payload len mismatch for {name}");
        raw[start..start + len].copy_from_slice(bytes);
    }
}

#[allow(dead_code)]
fn run_diag_dump_parity(model_path: &Path, prompt: &str, max_tokens: &str) -> ParityDumpReport {
    static NEXT_DUMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = NEXT_DUMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dump_path = env::temp_dir().join(format!(
        "jero_cert_dump_{}_{}_{}.json",
        prompt.len(),
        std::process::id(),
        seq
    ));
    let out = Command::new(JERO_DIAG_BIN)
        .args([
            "--model",
            model_path.to_str().expect("utf-8 model path"),
            "--prompt",
            prompt,
            "--threads",
            "1",
            "--n-ctx",
            "32",
            "--max-tokens",
            max_tokens,
            "--dump-parity",
            dump_path.to_str().expect("utf-8 dump path"),
        ])
        .output()
        .expect("execute jero-diag --dump-parity");
    assert!(
        out.status.success(),
        "jero-diag --dump-parity failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: ParityDumpReport =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).expect("read parity dump json"))
            .expect("parse parity dump json");
    let _ = std::fs::remove_file(&dump_path);
    report
}

#[allow(dead_code)]
fn verify_or_update_golden<T, F>(filename: &str, embedded_json: &str, actual: &T, verify_fn: F)
where
    T: Serialize + for<'de> Deserialize<'de>,
    F: FnOnce(&T, &T),
{
    if env::var("JERO_UPDATE_GOLDEN").as_deref() == Ok("1") {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_parity")
            .join(filename);
        std::fs::create_dir_all(path.parent().expect("fixture parent dir")).expect("mkdir");
        std::fs::write(
            &path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(actual).expect("serialize golden")
            ),
        )
        .expect("write golden");
    } else {
        let golden: T = serde_json::from_str(embedded_json)
            .unwrap_or_else(|e| panic!("parse embedded golden fixture {filename}: {e}"));
        verify_fn(actual, &golden);
    }
}

#[allow(dead_code)]
fn assert_f32_slice_close(lhs: &[f32], rhs: &[f32], tol: f32, label: &str) {
    assert_eq!(lhs.len(), rhs.len(), "{label} length mismatch");
    for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label} non-finite at {i}: {a} vs {b}"
        );
        assert!(
            (a - b).abs() <= tol,
            "{label} drift at {i}: {a} vs {b} (tol={tol})"
        );
    }
}

#[allow(dead_code)]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(feature = "jero-arch-gemma")]
fn build_authentic_gemma3_q8_0_gguf() -> Vec<u8> {
    let hidden = 128_usize;
    let ffn = 256_usize;
    let n_q = 4_usize;
    let n_kv = 2_usize;
    let head_dim = 32_usize;
    let vocab = 16_usize;

    let kvs = [
        ("gemma3.block_count", GgufValue::Uint32(1)),
        ("gemma3.embedding_length", GgufValue::Uint32(128)),
        ("gemma3.feed_forward_length", GgufValue::Uint32(256)),
        ("gemma3.attention.head_count", GgufValue::Uint32(4)),
        ("gemma3.attention.head_count_kv", GgufValue::Uint32(2)),
        ("gemma3.attention.key_length", GgufValue::Uint32(32)),
        ("gemma3.rope.freq_base", GgufValue::Float32(1_000_000.0)),
        ("gemma3.rope.freq_base_local", GgufValue::Float32(10_000.0)),
        ("gemma3.attention.sliding_window", GgufValue::Uint32(512)),
        (
            "gemma3.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(1.0e-6),
        ),
        ("gemma3.context_length", GgufValue::Uint32(128)),
        ("gemma3.vocab_size", GgufValue::Uint32(16)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
    ];

    let emb_shape = [hidden, vocab];
    let norm_shape = [hidden];
    let qk_norm_shape = [head_dim];
    let q_shape = [hidden, n_q * head_dim];
    let kv_shape = [hidden, n_kv * head_dim];
    let o_shape = [n_q * head_dim, hidden];
    let gate_up_shape = [hidden, ffn];
    let down_shape = [ffn, hidden];

    let specs: [(&str, QuantType, &[usize]); 14] = [
        ("token_embd.weight", QuantType::Q8_0, &emb_shape),
        ("blk.0.attn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.attn_q.weight", QuantType::Q8_0, &q_shape),
        ("blk.0.attn_k.weight", QuantType::Q8_0, &kv_shape),
        ("blk.0.attn_v.weight", QuantType::Q8_0, &kv_shape),
        ("blk.0.attn_output.weight", QuantType::Q8_0, &o_shape),
        ("blk.0.attn_q_norm.weight", QuantType::F32, &qk_norm_shape),
        ("blk.0.attn_k_norm.weight", QuantType::F32, &qk_norm_shape),
        ("blk.0.ffn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.ffn_gate.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.0.ffn_up.weight", QuantType::Q8_0, &gate_up_shape),
        ("blk.0.ffn_down.weight", QuantType::Q8_0, &down_shape),
        ("output_norm.weight", QuantType::F32, &norm_shape),
        ("output.weight", QuantType::Q8_0, &emb_shape),
    ];

    let mut raw = create_synthetic_gguf_v3_with_kvs("gemma3", 32, &kvs, &specs);
    populate_gguf_q8_0_payloads(&mut raw, true);
    raw
}

#[cfg(feature = "jero-arch-gemma")]
struct WithoutQNorm<'a>(GgufFile<'a>);

#[cfg(feature = "jero-arch-gemma")]
impl ModelWeights for WithoutQNorm<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        if name.ends_with("q_norm.weight") {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "missing injected test tensor".to_owned(),
            });
        }
        self.0.tensor(name)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        !name.ends_with("q_norm.weight") && self.0.contains_tensor(name)
    }

    fn tensor_count(&self) -> usize {
        self.0.tensor_count().saturating_sub(1)
    }

    fn total_weight_bytes(&self) -> usize {
        self.0.total_weight_bytes()
    }
}

#[cfg(feature = "jero-arch-gemma")]
#[test]
fn test_gemma3_q8_0_golden_parity_and_prompt_sensitivity() {
    let raw = build_authentic_gemma3_q8_0_gguf();
    let model_path = write_temp_gguf("gemma3_q8_0", &raw);
    let dump_a = run_diag_dump_parity(&model_path, "hello world", "4");
    let dump_b = run_diag_dump_parity(&model_path, "Explain RVV", "4");
    let _ = std::fs::remove_file(&model_path);

    assert_eq!(dump_a.architecture, "gemma3");
    verify_prompt_sensitivity(&dump_a, &dump_b)
        .expect("authentic GemmaRecipe must produce distinct logits for distinct prompts");
    verify_or_update_golden(
        "gemma3_authentic_q8_0.json",
        GOLDEN_GEMMA3_Q8_0_JSON,
        &dump_a,
        |actual, golden| {
            verify_parity_against_golden(actual, golden, 0.9999, 1.0e-4)
                .expect("GemmaRecipe parity dump must match golden fixture");
        },
    );

    // ADR-045 fail-closed check: missing blk.0.attn_q_norm.weight rejects prefill
    let parsed = GgufFile::parse(&raw).expect("parse gemma3 gguf");
    let recipe = jero_engine::GemmaRecipe::from_gguf(&parsed).expect("hydrate gemma3");
    let filtered = WithoutQNorm(parsed);
    let mut kv = SingleSlotKvArena::new(filtered.0.metadata(), 32).expect("kv arena");
    let mut logits = vec![0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 9], 32, &mut logits, kv.as_mut_slice()).expect("ctx");
    assert!(recipe.prefill_step(&mut ctx, &filtered).is_err());
}

#[allow(dead_code)]
fn build_dual_f32_containers(
    arch: &str,
    gguf_kvs: &[(&str, GgufValue<'_>)],
    st_meta: &[(&str, &str)],
    specs: &[(&str, &str, Vec<usize>, Vec<f32>)],
) -> (Vec<u8>, Vec<u8>) {
    let raw_bytes: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = specs
        .iter()
        .map(|(gn, sn, shape, vals)| (*gn, *sn, shape.clone(), f32_slice_to_le_bytes(vals)))
        .collect();
    let gguf_shapes: Vec<Vec<usize>> = raw_bytes
        .iter()
        .map(|(_, _, shape, _)| shape.iter().copied().rev().collect())
        .collect();
    let gguf_payloads: Vec<(&str, QuantType, Vec<usize>, Vec<u8>)> = raw_bytes
        .iter()
        .zip(gguf_shapes.iter())
        .map(|((gn, _, _, b), gs)| (*gn, QuantType::F32, gs.clone(), b.clone()))
        .collect();
    let gguf_specs: Vec<(&str, QuantType, &[usize])> = gguf_payloads
        .iter()
        .map(|(n, q, s, _)| (*n, *q, s.as_slice()))
        .collect();
    let mut gguf_bytes = create_synthetic_gguf_v3_with_kvs(arch, 32, gguf_kvs, &gguf_specs);
    patch_gguf_f32_payloads(&mut gguf_bytes, &gguf_payloads);

    let st_specs: Vec<(&str, QuantType, &[usize], &[u8])> = raw_bytes
        .iter()
        .map(|(_, sn, s, b)| (*sn, QuantType::F32, s.as_slice(), b.as_slice()))
        .collect();
    let st_bytes = create_synthetic_safetensors(st_meta, &st_specs);
    (gguf_bytes, st_bytes)
}

#[allow(dead_code)]
fn synthesize_tensor_values(shape: &[usize], seed: usize, scale: f32) -> Vec<f32> {
    let elems: usize = shape.iter().product();
    if scale < -1.5 {
        vec![0.0_f32; elems]
    } else if scale < -0.5 {
        vec![1.0_f32; elems]
    } else if scale <= 0.0 {
        unit_perturbed_norm_vec(elems, seed)
    } else {
        deterministic_f32_vec(elems, seed, scale)
    }
}

#[allow(dead_code)]
fn specs_from_table<'a>(
    rows: &[(&'a str, &'a str, &'a [usize], usize, f32)],
) -> Vec<(&'a str, &'a str, Vec<usize>, Vec<f32>)> {
    rows.iter()
        .map(|&(gn, sn, shape, seed, scale)| {
            (
                gn,
                sn,
                shape.to_vec(),
                synthesize_tensor_values(shape, seed, scale),
            )
        })
        .collect()
}

#[cfg(feature = "jero-arch-qwen")]
const QWEN3_DUAL_ROWS: &[(&str, &str, &[usize], usize, f32)] = &[
    (
        "token_embd.weight",
        "model.embed_tokens.weight",
        &[16, 16],
        1,
        0.08,
    ),
    (
        "blk.0.attn_norm.weight",
        "model.layers.0.input_layernorm.weight",
        &[16],
        2,
        0.0,
    ),
    (
        "blk.0.attn_q.weight",
        "model.layers.0.self_attn.q_proj.weight",
        &[16, 16],
        3,
        0.06,
    ),
    (
        "blk.0.attn_q_norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        &[8],
        4,
        0.0,
    ),
    (
        "blk.0.attn_k.weight",
        "model.layers.0.self_attn.k_proj.weight",
        &[8, 16],
        5,
        0.06,
    ),
    (
        "blk.0.attn_k_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        &[8],
        6,
        0.0,
    ),
    (
        "blk.0.attn_v.weight",
        "model.layers.0.self_attn.v_proj.weight",
        &[8, 16],
        7,
        0.06,
    ),
    (
        "blk.0.attn_output.weight",
        "model.layers.0.self_attn.o_proj.weight",
        &[16, 16],
        8,
        0.06,
    ),
    (
        "blk.0.ffn_norm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        &[16],
        9,
        0.0,
    ),
    (
        "blk.0.ffn_gate.weight",
        "model.layers.0.mlp.gate_proj.weight",
        &[32, 16],
        10,
        0.05,
    ),
    (
        "blk.0.ffn_up.weight",
        "model.layers.0.mlp.up_proj.weight",
        &[32, 16],
        11,
        0.05,
    ),
    (
        "blk.0.ffn_down.weight",
        "model.layers.0.mlp.down_proj.weight",
        &[16, 32],
        12,
        0.05,
    ),
    ("output_norm.weight", "model.norm.weight", &[16], 13, 0.0),
    ("output.weight", "lm_head.weight", &[16, 16], 14, 0.09),
];

#[cfg(feature = "jero-arch-qwen")]
const QWEN3_GGUF_KEY_SUFFIXES: [&str; 10] = [
    "block_count",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.layer_norm_rms_epsilon",
    "rope.freq_base",
    "context_length",
    "vocab_size",
];

#[cfg(feature = "jero-arch-qwen")]
fn build_dual_qwen3_models(arch: &str, zero_ctrl_output_rows: bool) -> (Vec<u8>, Vec<u8>) {
    let mut specs = specs_from_table(QWEN3_DUAL_ROWS);
    if zero_ctrl_output_rows {
        specs[13].3[..5 * 16].fill(0.0);
    }
    let keys: Vec<String> = QWEN3_GGUF_KEY_SUFFIXES
        .iter()
        .map(|suffix| format!("{arch}.{suffix}"))
        .collect();
    let gguf_kvs = [
        (keys[0].as_str(), GgufValue::Uint32(1)),
        (keys[1].as_str(), GgufValue::Uint32(16)),
        (keys[2].as_str(), GgufValue::Uint32(32)),
        (keys[3].as_str(), GgufValue::Uint32(2)),
        (keys[4].as_str(), GgufValue::Uint32(1)),
        (keys[5].as_str(), GgufValue::Uint32(8)),
        (keys[6].as_str(), GgufValue::Float32(1.0e-6)),
        (keys[7].as_str(), GgufValue::Float32(1_000_000.0)),
        (keys[8].as_str(), GgufValue::Uint32(128)),
        (keys[9].as_str(), GgufValue::Uint32(16)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2)),
    ];
    let st_meta = [
        ("general.architecture", arch),
        ("num_hidden_layers", "1"),
        ("hidden_size", "16"),
        ("intermediate_size", "32"),
        ("num_attention_heads", "2"),
        ("num_key_value_heads", "1"),
        ("head_dim", "8"),
        ("vocab_size", "16"),
        ("rms_norm_eps", "1e-6"),
        ("rope_theta", "1000000"),
        ("max_position_embeddings", "128"),
    ];
    build_dual_f32_containers(arch, &gguf_kvs, &st_meta, &specs)
}

#[cfg(feature = "jero-arch-qwen")]
fn write_hf_tokenizer_bundle(dir: &Path) {
    let tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<|im_start|>",
        "<|im_end|>",
        "user",
        "assistant",
        "\n",
        " ",
        "hello",
        "world",
        "Explain",
        "RVV",
        "1.0",
        "SpacemiT",
        "K3",
    ];
    let mut vocab = serde_json::Map::new();
    for (idx, tok) in tokens.iter().enumerate() {
        vocab.insert((*tok).to_owned(), json!(idx));
    }
    let tok_json = json!({
        "model": { "vocab": vocab },
        "added_tokens": [
            { "id": 0, "content": "<pad>", "special": true },
            { "id": 1, "content": "<bos>", "special": true },
            { "id": 2, "content": "<eos>", "special": true }
        ]
    });
    let cfg_json = json!({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "add_bos_token": true
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tok_json).expect("serialize tokenizer.json"),
    )
    .expect("write tokenizer.json");
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&cfg_json).expect("serialize tokenizer_config.json"),
    )
    .expect("write tokenizer_config.json");
}

#[cfg(feature = "jero-arch-qwen")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenEmbeddingGoldenSnapshot {
    prompt_a_embedding_dim8: Vec<f32>,
    prompt_b_embedding_dim8: Vec<f32>,
    cross_prompt_cosine: f32,
}

#[cfg(feature = "jero-arch-qwen")]
fn verify_qwen3_cli_dual_parity() {
    let (gguf_bytes, st_bytes) = build_dual_qwen3_models("qwen3", true);
    let tmp_dir = env::temp_dir().join(format!("jero_cert_qwen3_dual_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("create temp qwen3 dir");
    write_hf_tokenizer_bundle(&tmp_dir);

    let gguf_path = tmp_dir.join("model.gguf");
    let st_path = tmp_dir.join("model.safetensors");
    std::fs::write(&gguf_path, &gguf_bytes).expect("write qwen3 gguf");
    std::fs::write(&st_path, &st_bytes).expect("write qwen3 safetensors");

    let gguf_a = run_diag_dump_parity(&gguf_path, "hello world", "4");
    let gguf_b = run_diag_dump_parity(&gguf_path, "Explain RVV", "4");
    let st_a = run_diag_dump_parity(&st_path, "hello world", "4");
    let st_b = run_diag_dump_parity(&st_path, "Explain RVV", "4");
    let _ = std::fs::remove_dir_all(&tmp_dir);

    verify_prompt_sensitivity(&gguf_a, &gguf_b).expect("GGUF prompt sensitivity");
    verify_prompt_sensitivity(&st_a, &st_b).expect("SafeTensors prompt sensitivity");
    assert_eq!(gguf_a.prompt_token_ids, st_a.prompt_token_ids);
    assert_eq!(gguf_a.greedy_token_ids, st_a.greedy_token_ids);
    assert_eq!(gguf_a.generated_text, st_a.generated_text);
    for (g, s) in gguf_a.first_token_top_k.iter().zip(&st_a.first_token_top_k) {
        assert_eq!(g.token_id, s.token_id);
        assert_eq!(g.logit.to_bits(), s.logit.to_bits(), "bit-identical logits");
    }

    verify_or_update_golden(
        "qwen3_safetensors_gguf.json",
        GOLDEN_QWEN3_ST_GGUF_JSON,
        &st_a,
        |a, g| {
            verify_parity_against_golden(a, g, 0.9999, 1.0e-4).expect("SafeTensors parity");
            verify_parity_against_golden(&gguf_a, g, 0.9999, 1.0e-4).expect("GGUF parity");
        },
    );
}

#[cfg(feature = "jero-arch-qwen")]
fn run_qwen3_embedding_vec(
    recipe: &dyn ArchRecipe,
    w: &dyn ModelWeights,
    toks: &[u32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; 16];
    let mut logits = Vec::new();
    let mut kv = Vec::new();
    let mut ctx = ExecutionContext::new(toks, 32, &mut logits, &mut kv)
        .expect("ctx")
        .with_embedding_output(&mut out, EmbeddingPooling::LastToken, Some(8))
        .expect("embedding output");
    recipe.prefill_step(&mut ctx, w).expect("embed forward");
    let norm_sq: f32 = out[..8].iter().map(|&v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-4,
        "expected unit L2 norm, got {norm_sq}"
    );
    assert!(out[8..].iter().all(|&v| v == 0.0));
    out[..8].to_vec()
}

#[cfg(feature = "jero-arch-qwen")]
#[test]
fn test_qwen3_safetensors_vs_gguf_cli_parity_and_embedding() {
    verify_qwen3_cli_dual_parity();

    let (emb_gguf_bytes, emb_st_bytes) = build_dual_qwen3_models("qwen3-embedding", false);
    let emb_gguf = GgufFile::parse(&emb_gguf_bytes).expect("parse qwen3-embedding gguf");
    let emb_st = SafeTensorsFile::from_bytes(&emb_st_bytes).expect("parse qwen3-embedding st");
    let r_gguf = jero_engine::verify_model_capabilities(&emb_gguf).expect("hydrate gguf emb");
    let r_st = jero_engine::verify_safetensors_capabilities(&emb_st).expect("hydrate st emb");
    assert_eq!(r_gguf.architecture_id(), "qwen3-embedding");
    assert!(!r_gguf.requires_kv_cache());

    let gguf_emb_a = run_qwen3_embedding_vec(r_gguf.as_ref(), &emb_gguf, &[1, 9, 8, 10]);
    let st_emb_a = run_qwen3_embedding_vec(r_st.as_ref(), &emb_st, &[1, 9, 8, 10]);
    let st_emb_b = run_qwen3_embedding_vec(r_st.as_ref(), &emb_st, &[1, 11, 12, 13]);
    assert_f32_slice_close(
        &gguf_emb_a,
        &st_emb_a,
        1e-6,
        "qwen3-embedding GGUF vs SafeTensors",
    );
    let cos = cosine_sim(&st_emb_a, &st_emb_b);
    assert!(
        cos < 0.995,
        "expected prompt-sensitive embeddings, cosine={cos}"
    );

    let snap = QwenEmbeddingGoldenSnapshot {
        prompt_a_embedding_dim8: st_emb_a,
        prompt_b_embedding_dim8: st_emb_b,
        cross_prompt_cosine: cos,
    };
    verify_or_update_golden(
        "qwen3_embedding_f32.json",
        GOLDEN_QWEN3_EMBEDDING_JSON,
        &snap,
        |a, g| {
            assert_f32_slice_close(
                &a.prompt_a_embedding_dim8,
                &g.prompt_a_embedding_dim8,
                1e-4,
                "prompt A",
            );
            assert_f32_slice_close(
                &a.prompt_b_embedding_dim8,
                &g.prompt_b_embedding_dim8,
                1e-4,
                "prompt B",
            );
            assert!((a.cross_prompt_cosine - g.cross_prompt_cosine).abs() < 1e-4);
        },
    );
}

#[cfg(feature = "jero-arch-encoder")]
const ENCODER_DUAL_ROWS: &[(&str, &str, &[usize], usize, f32)] = &[
    ("token_embd.weight", "token_embd.weight", &[16, 8], 21, 0.08),
    (
        "blk.0.attn_norm.weight",
        "blk.0.attn_norm.weight",
        &[8],
        22,
        0.0,
    ),
    (
        "blk.0.attn_q.weight",
        "blk.0.attn_q.weight",
        &[8, 8],
        23,
        0.06,
    ),
    (
        "blk.0.attn_k.weight",
        "blk.0.attn_k.weight",
        &[8, 8],
        24,
        0.06,
    ),
    (
        "blk.0.attn_v.weight",
        "blk.0.attn_v.weight",
        &[8, 8],
        25,
        0.06,
    ),
    (
        "blk.0.attn_output.weight",
        "blk.0.attn_output.weight",
        &[8, 8],
        26,
        0.06,
    ),
    (
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_norm.weight",
        &[8],
        27,
        0.0,
    ),
    (
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_gate.weight",
        &[16, 8],
        28,
        0.05,
    ),
    (
        "blk.0.ffn_up.weight",
        "blk.0.ffn_up.weight",
        &[16, 8],
        29,
        0.05,
    ),
    (
        "blk.0.ffn_down.weight",
        "blk.0.ffn_down.weight",
        &[8, 16],
        30,
        0.05,
    ),
    ("classifier.weight", "classifier.weight", &[2, 8], 31, 0.09),
    ("classifier.bias", "classifier.bias", &[2], 32, 0.04),
];

#[cfg(feature = "jero-arch-encoder")]
fn build_dual_encoder_models(arch: &str, with_rel_emb: bool) -> (Vec<u8>, Vec<u8>) {
    let mut specs = specs_from_table(ENCODER_DUAL_ROWS);
    if with_rel_emb {
        specs.push((
            "rel_embeddings.weight",
            "rel_embeddings.weight",
            vec![16, 8],
            synthesize_tensor_values(&[16, 8], 33, 0.05),
        ));
    }
    let keys: Vec<String> = [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "context_length",
    ]
    .iter()
    .map(|s| format!("{arch}.{s}"))
    .collect();
    let gguf_kvs = [
        (keys[0].as_str(), GgufValue::Uint32(1)),
        (keys[1].as_str(), GgufValue::Uint32(8)),
        (keys[2].as_str(), GgufValue::Uint32(16)),
        (keys[3].as_str(), GgufValue::Uint32(2)),
        (keys[4].as_str(), GgufValue::Uint32(64)),
    ];
    let st_meta = [
        ("general.architecture", arch),
        ("num_hidden_layers", "1"),
        ("hidden_size", "8"),
        ("intermediate_size", "16"),
        ("num_attention_heads", "2"),
        ("max_position_embeddings", "64"),
    ];
    build_dual_f32_containers(arch, &gguf_kvs, &st_meta, &specs)
}

#[cfg(feature = "jero-arch-encoder")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncoderRerankGoldenSnapshot {
    nomic_bert_embedding: Vec<f32>,
    bge_embedding: Vec<f32>,
    jina_bert_embedding: Vec<f32>,
    rerank_doc_scores: Vec<f32>,
    deberta_v3_logits: Vec<f32>,
}

#[cfg(feature = "jero-arch-encoder")]
fn verify_encoder_embedding_arch(arch: &str) -> Vec<f32> {
    let (gguf_bytes, st_bytes) = build_dual_encoder_models(arch, false);
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse encoder gguf");
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse encoder st");
    let recipe = jero_engine::verify_model_capabilities(&gguf).expect("hydrate encoder gguf");
    assert_eq!(recipe.architecture_id(), arch);
    assert!(!recipe.requires_kv_cache());
    let st_recipe = jero_engine::verify_safetensors_capabilities(&st).expect("hydrate st");

    let eval = |r: &dyn ArchRecipe, w: &dyn ModelWeights, toks: &[u32]| -> Vec<f32> {
        let mut emb = vec![0.0_f32; 8];
        let mut logits = Vec::new();
        let mut kv = Vec::new();
        let mut ctx = ExecutionContext::new(toks, 64, &mut logits, &mut kv)
            .expect("ctx")
            .with_embedding_output(&mut emb, EmbeddingPooling::Mean, None)
            .expect("embed output");
        r.prefill_step(&mut ctx, w).expect("encoder embed");
        let norm_sq: f32 = emb.iter().map(|&v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-4, "{arch} norm_sq={norm_sq}");
        emb
    };

    let emb_g = eval(recipe.as_ref(), &gguf, &[1, 9, 8, 10]);
    let emb_s = eval(st_recipe.as_ref(), &st, &[1, 9, 8, 10]);
    let emb_other = eval(st_recipe.as_ref(), &st, &[1, 11, 12, 13]);
    assert_f32_slice_close(&emb_g, &emb_s, 1e-6, &format!("{arch} GGUF vs SafeTensors"));
    assert!(cosine_sim(&emb_s, &emb_other) < 0.995);
    emb_s
}

#[cfg(feature = "jero-arch-encoder")]
fn verify_reranker_and_deberta() -> (Vec<f32>, Vec<f32>) {
    let (rr_gguf_bytes, rr_st_bytes) = build_dual_encoder_models("bge-reranker", false);
    let rr_gguf = GgufFile::parse(&rr_gguf_bytes).expect("parse rerank gguf");
    let rr_st = SafeTensorsFile::from_bytes(&rr_st_bytes).expect("parse rerank st");
    let mut rerank = jero_engine::CrossEncoderRerankRecipe::new_bge_reranker();
    rerank.encoder.num_layers = 1;
    rerank.encoder.hidden_dim = 8;
    rerank.encoder.ffn_dim = 16;
    rerank.encoder.n_heads = 2;
    rerank.encoder.head_dim = 4;
    let docs: [&[u32]; 2] = [&[9, 10], &[11, 12, 13]];
    let scores_gguf = rerank
        .rerank_documents(&[1, 5], &docs, 64, &rr_gguf)
        .expect("rerank gguf");
    let scores_st = rerank
        .rerank_documents(&[1, 5], &docs, 64, &rr_st)
        .expect("rerank st");
    assert_f32_slice_close(&scores_gguf, &scores_st, 1e-6, "rerank GGUF vs SafeTensors");
    assert!(
        (scores_st[0] - scores_st[1]).abs() > 1e-4,
        "docs must score distinctly"
    );

    let (deb_gguf_bytes, deb_st_bytes) = build_dual_encoder_models("deberta-v3", true);
    let deb_gguf = GgufFile::parse(&deb_gguf_bytes).expect("parse deberta gguf");
    let deb_st = SafeTensorsFile::from_bytes(&deb_st_bytes).expect("parse deberta st");
    let mut deberta = jero_engine::DebertaV3Recipe::new();
    deberta.num_layers = 1;
    deberta.hidden_dim = 8;
    deberta.n_heads = 2;
    deberta.head_dim = 4;
    deberta.max_relative_positions = 8;
    deberta.num_labels = 2;
    let mut deb_logits_g = [0.0_f32; 2];
    let mut deb_logits_s = [0.0_f32; 2];
    let mut kv = Vec::new();
    let mut ctx_g =
        ExecutionContext::new(&[1, 9, 10], 64, &mut deb_logits_g, &mut kv).expect("ctx");
    deberta
        .forward_classify(&mut ctx_g, &deb_gguf)
        .expect("deberta gguf");
    let mut ctx_s =
        ExecutionContext::new(&[1, 9, 10], 64, &mut deb_logits_s, &mut kv).expect("ctx");
    deberta
        .forward_classify(&mut ctx_s, &deb_st)
        .expect("deberta st");
    assert_f32_slice_close(
        &deb_logits_g,
        &deb_logits_s,
        1e-6,
        "deberta GGUF vs SafeTensors",
    );
    (scores_st, deb_logits_s.to_vec())
}

#[cfg(feature = "jero-arch-encoder")]
#[test]
fn test_encoder_embeddings_and_rerank_safetensors_gguf_parity() {
    let nomic = verify_encoder_embedding_arch("nomic-bert");
    let bge = verify_encoder_embedding_arch("bge");
    let jina = verify_encoder_embedding_arch("jina-bert");
    let (rerank_doc_scores, deberta_v3_logits) = verify_reranker_and_deberta();
    let snap = EncoderRerankGoldenSnapshot {
        nomic_bert_embedding: nomic,
        bge_embedding: bge,
        jina_bert_embedding: jina,
        rerank_doc_scores,
        deberta_v3_logits,
    };
    verify_or_update_golden(
        "encoder_rerank_f32.json",
        GOLDEN_ENCODER_RERANK_JSON,
        &snap,
        |a, g| {
            assert_f32_slice_close(
                &a.nomic_bert_embedding,
                &g.nomic_bert_embedding,
                1e-4,
                "nomic-bert",
            );
            assert_f32_slice_close(&a.bge_embedding, &g.bge_embedding, 1e-4, "bge");
            assert_f32_slice_close(
                &a.jina_bert_embedding,
                &g.jina_bert_embedding,
                1e-4,
                "jina-bert",
            );
            assert_f32_slice_close(&a.rerank_doc_scores, &g.rerank_doc_scores, 1e-4, "rerank");
            assert_f32_slice_close(
                &a.deberta_v3_logits,
                &g.deberta_v3_logits,
                1e-4,
                "deberta-v3",
            );
        },
    );
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn encode_vision_safetensors_table(
    arch: &str,
    img_size: &str,
    rows: &[(&str, &str, &[usize], usize, f32)],
) -> Vec<u8> {
    let specs = specs_from_table(rows);
    let raw: Vec<(&str, &Vec<usize>, Vec<u8>)> = specs
        .iter()
        .map(|(n, _, s, v)| (*n, s, f32_slice_to_le_bytes(v)))
        .collect();
    let refs: Vec<(&str, QuantType, &[usize], &[u8])> = raw
        .iter()
        .map(|(n, s, b)| (*n, QuantType::F32, s.as_slice(), b.as_slice()))
        .collect();
    create_synthetic_safetensors(
        &[
            ("general.architecture", arch),
            ("clip.vision.image_size", img_size),
            ("clip.vision.patch_size", "2"),
            ("clip.vision.embedding_length", "8"),
        ],
        &refs,
    )
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
const SIGLIP_BACKBONE_ROWS: &[(&str, &str, &[usize], usize, f32)] = &[
    (
        "vision_model.embeddings.patch_embedding.weight",
        "",
        &[8, 3, 2, 2],
        1,
        0.05,
    ),
    (
        "vision_model.embeddings.patch_embedding.bias",
        "",
        &[8],
        2,
        0.02,
    ),
    (
        "vision_model.embeddings.position_embedding.weight",
        "",
        &[16, 8],
        3,
        0.04,
    ),
    (
        "vision_model.encoder.layers.0.self_attn.qkv.weight",
        "",
        &[24, 8],
        4,
        0.05,
    ),
    (
        "vision_model.encoder.layers.0.self_attn.qkv.bias",
        "",
        &[24],
        5,
        0.02,
    ),
    (
        "vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "",
        &[8, 8],
        6,
        0.05,
    ),
    (
        "vision_model.encoder.layers.0.self_attn.out_proj.bias",
        "",
        &[8],
        7,
        0.02,
    ),
    (
        "vision_model.encoder.layers.0.layer_norm1.weight",
        "",
        &[8],
        0,
        -1.0,
    ),
    (
        "vision_model.encoder.layers.0.layer_norm1.bias",
        "",
        &[8],
        0,
        -2.0,
    ),
    (
        "vision_model.encoder.layers.0.mlp.fc1.weight",
        "",
        &[16, 8],
        8,
        0.04,
    ),
    (
        "vision_model.encoder.layers.0.mlp.fc1.bias",
        "",
        &[16],
        9,
        0.02,
    ),
    (
        "vision_model.encoder.layers.0.mlp.fc2.weight",
        "",
        &[8, 16],
        10,
        0.04,
    ),
    (
        "vision_model.encoder.layers.0.mlp.fc2.bias",
        "",
        &[8],
        11,
        0.02,
    ),
    (
        "vision_model.encoder.layers.0.layer_norm2.weight",
        "",
        &[8],
        0,
        -1.0,
    ),
    (
        "vision_model.encoder.layers.0.layer_norm2.bias",
        "",
        &[8],
        0,
        -2.0,
    ),
    ("vision_model.post_layernorm.weight", "", &[8], 0, -1.0),
    ("vision_model.post_layernorm.bias", "", &[8], 0, -2.0),
    ("merger.ln_q.weight", "", &[8], 0, -1.0),
];

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn build_compact_siglip_st(llm_dim: usize) -> Vec<u8> {
    let (merged_in, mlp_hidden) = (32, llm_dim * 2);
    let mut rows = SIGLIP_BACKBONE_ROWS.to_vec();
    let w1_shape = [mlp_hidden, merged_in];
    let b1_shape = [mlp_hidden];
    let w2_shape = [llm_dim, mlp_hidden];
    let b2_shape = [llm_dim];
    rows.extend_from_slice(&[
        (
            "multi_modal_projector.linear_1.weight",
            "",
            &w1_shape,
            12,
            0.04,
        ),
        (
            "multi_modal_projector.linear_1.bias",
            "",
            &b1_shape,
            13,
            0.02,
        ),
        (
            "multi_modal_projector.linear_2.weight",
            "",
            &w2_shape,
            14,
            0.04,
        ),
        (
            "multi_modal_projector.linear_2.bias",
            "",
            &b2_shape,
            15,
            0.02,
        ),
    ]);
    encode_vision_safetensors_table("siglip", "8", &rows)
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn build_compact_clip_st(llm_dim: usize) -> Vec<u8> {
    let (h, ffn, patch, num_pos) = (8, 16, 2, 5);
    let rows: [(&str, &str, &[usize], usize, f32); 28] = [
        ("v.patch_embd.weight", "", &[h, 3, patch, patch], 1, 0.05),
        ("v.patch_embd.bias", "", &[h], 2, 0.02),
        ("v.class_embd", "", &[h], 3, 0.04),
        ("v.position_embd.weight", "", &[num_pos, h], 4, 0.04),
        ("v.pre_ln.weight", "", &[h], 0, -1.0),
        ("v.pre_ln.bias", "", &[h], 0, -2.0),
        ("v.blk.0.attn_q.weight", "", &[h, h], 5, 0.05),
        ("v.blk.0.attn_q.bias", "", &[h], 6, 0.02),
        ("v.blk.0.attn_k.weight", "", &[h, h], 7, 0.05),
        ("v.blk.0.attn_k.bias", "", &[h], 8, 0.02),
        ("v.blk.0.attn_v.weight", "", &[h, h], 9, 0.05),
        ("v.blk.0.attn_v.bias", "", &[h], 10, 0.02),
        ("v.blk.0.attn_out.weight", "", &[h, h], 11, 0.05),
        ("v.blk.0.attn_out.bias", "", &[h], 12, 0.02),
        ("v.blk.0.ln1.weight", "", &[h], 0, -1.0),
        ("v.blk.0.ln1.bias", "", &[h], 0, -2.0),
        ("v.blk.0.ffn_up.weight", "", &[ffn, h], 13, 0.04),
        ("v.blk.0.ffn_up.bias", "", &[ffn], 14, 0.02),
        ("v.blk.0.ffn_down.weight", "", &[h, ffn], 15, 0.04),
        ("v.blk.0.ffn_down.bias", "", &[h], 16, 0.02),
        ("v.blk.0.ln2.weight", "", &[h], 0, -1.0),
        ("v.blk.0.ln2.bias", "", &[h], 0, -2.0),
        ("v.post_ln.weight", "", &[h], 0, -1.0),
        ("v.post_ln.bias", "", &[h], 0, -2.0),
        ("mm.0.weight", "", &[llm_dim, h], 17, 0.05),
        ("mm.0.bias", "", &[llm_dim], 18, 0.02),
        ("mm.2.weight", "", &[llm_dim, llm_dim], 19, 0.05),
        ("mm.2.bias", "", &[llm_dim], 20, 0.02),
    ];
    encode_vision_safetensors_table("clip", "4", &rows)
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
const COMPACT_GEMMA3_ROWS: &[(&str, &str, &[usize], usize, f32)] = &[
    (
        "token_embd.weight",
        "token_embd.weight",
        &[16, 16],
        41,
        0.08,
    ),
    (
        "blk.0.attn_norm.weight",
        "blk.0.attn_norm.weight",
        &[16],
        42,
        0.0,
    ),
    (
        "blk.0.attn_q.weight",
        "blk.0.attn_q.weight",
        &[16, 16],
        43,
        0.06,
    ),
    (
        "blk.0.attn_k.weight",
        "blk.0.attn_k.weight",
        &[8, 16],
        44,
        0.06,
    ),
    (
        "blk.0.attn_v.weight",
        "blk.0.attn_v.weight",
        &[8, 16],
        45,
        0.06,
    ),
    (
        "blk.0.attn_output.weight",
        "blk.0.attn_output.weight",
        &[16, 16],
        46,
        0.06,
    ),
    (
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_q_norm.weight",
        &[8],
        47,
        0.0,
    ),
    (
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_k_norm.weight",
        &[8],
        48,
        0.0,
    ),
    (
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_norm.weight",
        &[16],
        49,
        0.0,
    ),
    (
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_gate.weight",
        &[32, 16],
        50,
        0.05,
    ),
    (
        "blk.0.ffn_up.weight",
        "blk.0.ffn_up.weight",
        &[32, 16],
        51,
        0.05,
    ),
    (
        "blk.0.ffn_down.weight",
        "blk.0.ffn_down.weight",
        &[16, 32],
        52,
        0.05,
    ),
    ("output_norm.weight", "output_norm.weight", &[16], 53, 0.0),
    ("output.weight", "output.weight", &[16, 16], 54, 0.09),
];

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn build_compact_gemma3_f32_gguf() -> Vec<u8> {
    let specs = specs_from_table(COMPACT_GEMMA3_ROWS);
    let kvs = [
        ("gemma3.block_count", GgufValue::Uint32(1)),
        ("gemma3.embedding_length", GgufValue::Uint32(16)),
        ("gemma3.feed_forward_length", GgufValue::Uint32(32)),
        ("gemma3.attention.head_count", GgufValue::Uint32(2)),
        ("gemma3.attention.head_count_kv", GgufValue::Uint32(1)),
        ("gemma3.attention.key_length", GgufValue::Uint32(8)),
        ("gemma3.context_length", GgufValue::Uint32(64)),
        ("gemma3.vocab_size", GgufValue::Uint32(16)),
    ];
    build_dual_f32_containers(
        "gemma3",
        &kvs,
        &[("general.architecture", "gemma3")],
        &specs,
    )
    .0
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisionMmprojGoldenSnapshot {
    clip_gemma3_logits_img_a: Vec<f32>,
    clip_gemma3_logits_img_b: Vec<f32>,
    siglip_qwen3_logits_img_a: Vec<f32>,
    siglip_qwen3_logits_img_b: Vec<f32>,
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn run_vlm_soft_prompt_forward(
    recipe: &dyn ArchRecipe,
    meta: &jero_core::ModelMetadata,
    w: &dyn ModelWeights,
    soft: &[f32],
    n_soft: usize,
    llm_dim: usize,
) -> Vec<f32> {
    let mut arena = SingleSlotKvArena::new(meta, 64).expect("kv arena");
    let mut logits = vec![0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 5], 64, &mut logits, arena.as_mut_slice())
        .expect("ctx")
        .with_soft_prompt_embeds(soft, n_soft, llm_dim)
        .expect("soft prompt");
    recipe.prefill_step(&mut ctx, w).expect("vlm prefill");
    assert_eq!(ctx.position(), n_soft + 2);
    let prefill_logits = ctx.logits().to_vec();
    ctx.prepare_decode(6).expect("prepare decode");
    recipe.decode_step(&mut ctx, w).expect("vlm decode");
    assert_eq!(ctx.position(), n_soft + 3);
    prefill_logits
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn encode_vision_pair(llm_dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let clip_bytes = build_compact_clip_st(llm_dim);
    let sig_bytes = build_compact_siglip_st(llm_dim);
    let clip_st = SafeTensorsFile::from_bytes(&clip_bytes).expect("parse clip st");
    let sig_st = SafeTensorsFile::from_bytes(&sig_bytes).expect("parse siglip st");

    let clip_bridge = jero_engine::VlmMmprojBridge::new_llava_clip(
        jero_engine::ClipVisionRecipe::from_safetensors(&clip_st).with_dims(4, 2, 1, 8, 16, 2),
        llm_dim,
    );
    let sig_bridge = jero_engine::VlmMmprojBridge::new_gemma3_siglip(
        jero_engine::SiglipVisionRecipe::from_safetensors(&sig_st).with_dims(8, 2, 1, 8, 16, 2),
        llm_dim,
    );

    let (ca, na, _) = clip_bridge
        .encode_image_to_soft_prompt_with_dims(
            &clip_st,
            &deterministic_f32_vec(48, 15, 0.25),
            3,
            4,
            4,
        )
        .expect("clip A");
    let (cb, nb, _) = clip_bridge
        .encode_image_to_soft_prompt_with_dims(
            &clip_st,
            &deterministic_f32_vec(48, 97, 0.35),
            3,
            4,
            4,
        )
        .expect("clip B");
    let (sa, nsa, _) = sig_bridge
        .encode_image_to_soft_prompt_with_dims(
            &sig_st,
            &deterministic_f32_vec(192, 25, 0.25),
            3,
            8,
            8,
        )
        .expect("siglip A");
    let (sb, nsb, _) = sig_bridge
        .encode_image_to_soft_prompt_with_dims(
            &sig_st,
            &deterministic_f32_vec(192, 83, 0.35),
            3,
            8,
            8,
        )
        .expect("siglip B");
    assert_eq!((na, nb, nsa, nsb), (4, 4, 4, 4));
    (vec![ca, cb], vec![sa, sb])
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
fn verify_vision_mmproj_golden(snap: &VisionMmprojGoldenSnapshot) {
    verify_or_update_golden(
        "vision_mmproj_f32.json",
        GOLDEN_VISION_MMPROJ_JSON,
        snap,
        |a, g| {
            assert_f32_slice_close(
                &a.clip_gemma3_logits_img_a,
                &g.clip_gemma3_logits_img_a,
                1e-4,
                "clip A",
            );
            assert_f32_slice_close(
                &a.clip_gemma3_logits_img_b,
                &g.clip_gemma3_logits_img_b,
                1e-4,
                "clip B",
            );
            assert_f32_slice_close(
                &a.siglip_qwen3_logits_img_a,
                &g.siglip_qwen3_logits_img_a,
                1e-4,
                "siglip A",
            );
            assert_f32_slice_close(
                &a.siglip_qwen3_logits_img_b,
                &g.siglip_qwen3_logits_img_b,
                1e-4,
                "siglip B",
            );
        },
    );
}

#[cfg(all(
    feature = "jero-arch-vision",
    feature = "jero-arch-gemma",
    feature = "jero-arch-qwen"
))]
#[test]
fn test_vision_mmproj_layer0_soft_prompt_vlm_parity() {
    let llm_dim = 16_usize;
    let (clip_soft, sig_soft) = encode_vision_pair(llm_dim);
    let (soft_clip_a, soft_clip_b) = (&clip_soft[0], &clip_soft[1]);
    let (soft_sig_a, soft_sig_b) = (&sig_soft[0], &sig_soft[1]);

    let gemma_bytes = build_compact_gemma3_f32_gguf();
    let gemma_gguf = GgufFile::parse(&gemma_bytes).expect("parse gemma3 vlm");
    let gemma_recipe = jero_engine::verify_model_capabilities(&gemma_gguf).expect("hydrate gemma3");
    let (_, qwen_st_bytes) = build_dual_qwen3_models("qwen3", false);
    let qwen_st = SafeTensorsFile::from_bytes(&qwen_st_bytes).expect("parse qwen3 vlm");
    let qwen_recipe =
        jero_engine::verify_safetensors_capabilities(&qwen_st).expect("hydrate qwen3");

    let clip_gemma_a = run_vlm_soft_prompt_forward(
        gemma_recipe.as_ref(),
        gemma_gguf.metadata(),
        &gemma_gguf,
        soft_clip_a,
        4,
        llm_dim,
    );
    let clip_gemma_b = run_vlm_soft_prompt_forward(
        gemma_recipe.as_ref(),
        gemma_gguf.metadata(),
        &gemma_gguf,
        soft_clip_b,
        4,
        llm_dim,
    );
    let sig_qwen_a = run_vlm_soft_prompt_forward(
        qwen_recipe.as_ref(),
        qwen_st.metadata(),
        &qwen_st,
        soft_sig_a,
        4,
        llm_dim,
    );
    let sig_qwen_b = run_vlm_soft_prompt_forward(
        qwen_recipe.as_ref(),
        qwen_st.metadata(),
        &qwen_st,
        soft_sig_b,
        4,
        llm_dim,
    );

    let max_delta = |xs: &[f32], ys: &[f32]| -> f32 {
        xs.iter()
            .zip(ys)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    assert!(
        max_delta(&clip_gemma_a, &clip_gemma_b) > 1e-4,
        "CLIP conditions Gemma3"
    );
    assert!(
        max_delta(&sig_qwen_a, &sig_qwen_b) > 1e-4,
        "SigLIP conditions Qwen3"
    );

    let mut bad_arena = SingleSlotKvArena::new(qwen_st.metadata(), 64).expect("kv arena");
    let mut bad_logits = vec![0.0_f32; 16];
    let mut bad_ctx = ExecutionContext::new(&[1], 64, &mut bad_logits, bad_arena.as_mut_slice())
        .expect("ctx")
        .with_soft_prompt_embeds(&[0.1_f32; 24], 2, 12)
        .expect("bad dim soft prompt");
    assert!(qwen_recipe.prefill_step(&mut bad_ctx, &qwen_st).is_err());

    let snap = VisionMmprojGoldenSnapshot {
        clip_gemma3_logits_img_a: clip_gemma_a,
        clip_gemma3_logits_img_b: clip_gemma_b,
        siglip_qwen3_logits_img_a: sig_qwen_a,
        siglip_qwen3_logits_img_b: sig_qwen_b,
    };
    verify_vision_mmproj_golden(&snap);
}

#[cfg(feature = "jero-arch-whisper")]
fn push_table_prefixed(
    prefix: &str,
    entries: &[(&str, &[usize], usize, f32)],
    items: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
) {
    for &(suffix, shape, seed, scale) in entries {
        let name = if prefix.is_empty() {
            suffix.to_owned()
        } else {
            format!("{prefix}.{suffix}")
        };
        items.push((
            name,
            shape.to_vec(),
            synthesize_tensor_values(shape, seed, scale),
        ));
    }
}

#[cfg(feature = "jero-arch-whisper")]
fn append_whisper_encoder_items(
    recipe: &jero_engine::WhisperRecipe,
    items: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
) {
    let (d, ffn, mels) = (recipe.n_audio_state, recipe.audio_ffn_dim, recipe.n_mels);
    push_table_prefixed(
        "",
        &[
            ("encoder.conv1.weight", &[d, mels, 3], 1, 0.04),
            ("encoder.conv1.bias", &[d], 2, 0.02),
            ("encoder.conv2.weight", &[d, d, 3], 3, 0.04),
            ("encoder.conv2.bias", &[d], 4, 0.02),
            (
                "encoder.positional_embedding",
                &[recipe.n_audio_ctx, d],
                5,
                0.03,
            ),
            ("encoder.ln_post.weight", &[d], 6, 0.0),
            ("encoder.ln_post.bias", &[d], 7, 0.01),
        ],
        items,
    );
    for l in 0..recipe.n_audio_layer {
        let s = 20 + l * 30;
        let p = format!("encoder.blocks.{l}");
        push_table_prefixed(
            &p,
            &[
                ("attn_ln.weight", &[d], s + 1, 0.0),
                ("attn_ln.bias", &[d], s + 2, 0.01),
                ("attn.query.weight", &[d, d], s + 3, 0.05),
                ("attn.query.bias", &[d], s + 4, 0.02),
                ("attn.key.weight", &[d, d], s + 5, 0.05),
                ("attn.value.weight", &[d, d], s + 6, 0.05),
                ("attn.value.bias", &[d], s + 7, 0.02),
                ("attn.out.weight", &[d, d], s + 8, 0.05),
                ("attn.out.bias", &[d], s + 9, 0.02),
                ("mlp_ln.weight", &[d], s + 10, 0.0),
                ("mlp_ln.bias", &[d], s + 11, 0.01),
                ("mlp.0.weight", &[ffn, d], s + 12, 0.04),
                ("mlp.0.bias", &[ffn], s + 13, 0.02),
                ("mlp.2.weight", &[d, ffn], s + 14, 0.04),
                ("mlp.2.bias", &[d], s + 15, 0.02),
            ],
            items,
        );
    }
}

#[cfg(feature = "jero-arch-whisper")]
fn append_whisper_decoder_items(
    recipe: &jero_engine::WhisperRecipe,
    items: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
) {
    let (de, dd, ffn) = (
        recipe.n_audio_state,
        recipe.n_text_state,
        recipe.text_ffn_dim,
    );
    push_table_prefixed(
        "",
        &[
            (
                "decoder.token_embedding.weight",
                &[recipe.n_vocab, dd],
                101,
                0.08,
            ),
            (
                "decoder.positional_embedding",
                &[recipe.n_text_ctx, dd],
                102,
                0.03,
            ),
            ("decoder.ln.weight", &[dd], 103, 0.0),
            ("decoder.ln.bias", &[dd], 104, 0.01),
        ],
        items,
    );
    for l in 0..recipe.n_text_layer {
        let s = 200 + l * 40;
        let p = format!("decoder.blocks.{l}");
        push_table_prefixed(
            &p,
            &[
                ("attn_ln.weight", &[dd], s + 1, 0.0),
                ("attn_ln.bias", &[dd], s + 2, 0.01),
                ("attn.query.weight", &[dd, dd], s + 3, 0.05),
                ("attn.query.bias", &[dd], s + 4, 0.02),
                ("attn.key.weight", &[dd, dd], s + 5, 0.05),
                ("attn.value.weight", &[dd, dd], s + 6, 0.05),
                ("attn.value.bias", &[dd], s + 7, 0.02),
                ("attn.out.weight", &[dd, dd], s + 8, 0.05),
                ("attn.out.bias", &[dd], s + 9, 0.02),
                ("cross_attn_ln.weight", &[dd], s + 10, 0.0),
                ("cross_attn_ln.bias", &[dd], s + 11, 0.01),
                ("cross_attn.query.weight", &[dd, dd], s + 12, 0.05),
                ("cross_attn.query.bias", &[dd], s + 13, 0.02),
                ("cross_attn.key.weight", &[dd, de], s + 14, 0.06),
                ("cross_attn.value.weight", &[dd, de], s + 15, 0.06),
                ("cross_attn.value.bias", &[dd], s + 16, 0.02),
                ("cross_attn.out.weight", &[dd, dd], s + 17, 0.05),
                ("cross_attn.out.bias", &[dd], s + 18, 0.02),
                ("mlp_ln.weight", &[dd], s + 19, 0.0),
                ("mlp_ln.bias", &[dd], s + 20, 0.01),
                ("mlp.0.weight", &[ffn, dd], s + 21, 0.04),
                ("mlp.0.bias", &[ffn], s + 22, 0.02),
                ("mlp.2.weight", &[dd, ffn], s + 23, 0.04),
                ("mlp.2.bias", &[dd], s + 24, 0.02),
            ],
            items,
        );
    }
}

#[cfg(feature = "jero-arch-whisper")]
fn build_dual_whisper_models(recipe: &jero_engine::WhisperRecipe) -> (Vec<u8>, Vec<u8>) {
    let mut items = Vec::new();
    append_whisper_encoder_items(recipe, &mut items);
    append_whisper_decoder_items(recipe, &mut items);
    let specs: Vec<(&str, &str, Vec<usize>, Vec<f32>)> = items
        .iter()
        .map(|(n, s, v)| (n.as_str(), n.as_str(), s.clone(), v.clone()))
        .collect();
    let gguf_kvs = [
        ("whisper.n_mels", GgufValue::Uint32(80)),
        ("whisper.encoder.embedding_length", GgufValue::Uint32(16)),
        ("whisper.encoder.attention.head_count", GgufValue::Uint32(2)),
        ("whisper.encoder.layer_count", GgufValue::Uint32(2)),
        ("whisper.encoder.feed_forward_length", GgufValue::Uint32(32)),
        ("whisper.encoder.context_length", GgufValue::Uint32(8)),
        ("whisper.decoder.embedding_length", GgufValue::Uint32(16)),
        ("whisper.decoder.attention.head_count", GgufValue::Uint32(2)),
        ("whisper.decoder.layer_count", GgufValue::Uint32(2)),
        ("whisper.decoder.feed_forward_length", GgufValue::Uint32(32)),
        ("whisper.decoder.context_length", GgufValue::Uint32(12)),
        ("whisper.vocab_size", GgufValue::Uint32(24)),
    ];
    let st_meta = [
        ("general.architecture", "whisper"),
        ("num_mel_bins", "80"),
        ("d_model", "16"),
        ("encoder_attention_heads", "2"),
        ("decoder_attention_heads", "2"),
        ("encoder_ffn_dim", "32"),
        ("decoder_ffn_dim", "32"),
        ("max_source_positions", "8"),
        ("max_target_positions", "12"),
        ("vocab_size", "24"),
    ];
    build_dual_f32_containers("whisper", &gguf_kvs, &st_meta, &specs)
}

#[cfg(feature = "jero-arch-whisper")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhisperGoldenSnapshot {
    prefill_logits_440hz: Vec<f32>,
    decode_logits_440hz: Vec<f32>,
    prefill_logits_2600hz: Vec<f32>,
    greedy_token_ids_440hz: Vec<u32>,
}

#[cfg(feature = "jero-arch-whisper")]
fn run_whisper_asr_step(
    recipe: &jero_engine::WhisperRecipe,
    arch_recipe: &dyn ArchRecipe,
    w: &dyn ModelWeights,
    mel: &[f32],
    n_frames: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, u32, u32) {
    let cross_len = recipe.cross_attn_kv_len(n_frames).expect("cross_kv len");
    let self_len = recipe.self_attn_kv_len(8).expect("self_kv len");
    let mut logits = vec![0.0_f32; recipe.n_vocab];
    let mut self_kv = vec![0.0_f32; self_len];
    let mut cross_kv = vec![0.0_f32; cross_len];
    let mut ctx = ExecutionContext::new(&[1, 2, 3], 8, &mut logits, &mut self_kv)
        .expect("ctx")
        .with_audio_mel(mel, 80, n_frames)
        .expect("audio mel")
        .with_cross_attn_kv(&mut cross_kv)
        .expect("cross kv");
    arch_recipe
        .prefill_step(&mut ctx, w)
        .expect("whisper prefill");
    let prefill_logits = ctx.logits().to_vec();
    let tok0 = jero_engine::sample_greedy_argmax(&prefill_logits).expect("argmax0");
    ctx.prepare_decode(tok0).expect("prepare decode");
    arch_recipe
        .decode_step(&mut ctx, w)
        .expect("whisper decode");
    let decode_logits = ctx.logits().to_vec();
    let tok1 = jero_engine::sample_greedy_argmax(&decode_logits).expect("argmax1");
    let ckv = ctx.cross_attn_kv().expect("cross kv present").to_vec();
    (prefill_logits, decode_logits, ckv, tok0, tok1)
}

#[cfg(feature = "jero-arch-whisper")]
#[test]
fn test_whisper_asr_pcm_mel_cross_attn_safetensors_gguf_parity() {
    let recipe = jero_engine::WhisperRecipe {
        n_mels: 80,
        n_audio_ctx: 8,
        n_audio_state: 16,
        n_audio_head: 2,
        n_audio_layer: 2,
        audio_ffn_dim: 32,
        n_text_ctx: 12,
        n_text_state: 16,
        n_text_head: 2,
        n_text_layer: 2,
        text_ffn_dim: 32,
        n_vocab: 24,
        norm_eps: 1e-5,
    };
    let (gguf_bytes, st_bytes) = build_dual_whisper_models(&recipe);
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse whisper gguf");
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse whisper st");
    let r_gguf = jero_engine::verify_model_capabilities(&gguf).expect("hydrate whisper gguf");
    let r_st = jero_engine::verify_safetensors_capabilities(&st).expect("hydrate whisper st");
    assert_eq!(r_gguf.architecture_id(), "whisper");

    let sine = |freq_hz: f32| -> Vec<f32> {
        (0..640)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = (i as f32) / 16_000.0_f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
            })
            .collect()
    };
    let (mel_440, n_frames) = recipe.pcm_to_log_mel(&sine(440.0), None).expect("mel 440");
    let (mel_2600, _) = recipe
        .pcm_to_log_mel(&sine(2600.0), None)
        .expect("mel 2600");

    let (st_prefill, st_decode, st_ckv, tok0_st, tok1_st) =
        run_whisper_asr_step(&recipe, r_st.as_ref(), &st, &mel_440, n_frames);
    let (gg_prefill, gg_decode, gg_ckv, tok0_gg, tok1_gg) =
        run_whisper_asr_step(&recipe, r_gguf.as_ref(), &gguf, &mel_440, n_frames);
    let (st_prefill_2600, _, _, _, _) =
        run_whisper_asr_step(&recipe, r_st.as_ref(), &st, &mel_2600, n_frames);

    assert_eq!((tok0_st, tok1_st), (tok0_gg, tok1_gg));
    assert_f32_slice_close(&st_ckv, &gg_ckv, 1e-6, "whisper cross_kv parity");
    assert_f32_slice_close(&st_prefill, &gg_prefill, 1e-6, "whisper prefill parity");
    assert_f32_slice_close(&st_decode, &gg_decode, 1e-6, "whisper decode parity");

    let acoustic_delta: f32 = st_prefill
        .iter()
        .zip(&st_prefill_2600)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        acoustic_delta > 1e-4,
        "distinct PCM waves must alter ASR logits"
    );

    let snap = WhisperGoldenSnapshot {
        prefill_logits_440hz: st_prefill,
        decode_logits_440hz: st_decode,
        prefill_logits_2600hz: st_prefill_2600,
        greedy_token_ids_440hz: vec![tok0_st, tok1_st],
    };
    verify_or_update_golden(
        "whisper_asr_f32.json",
        GOLDEN_WHISPER_ASR_JSON,
        &snap,
        |a, g| {
            assert_eq!(a.greedy_token_ids_440hz, g.greedy_token_ids_440hz);
            assert_f32_slice_close(
                &a.prefill_logits_440hz,
                &g.prefill_logits_440hz,
                1e-4,
                "prefill 440Hz",
            );
            assert_f32_slice_close(
                &a.decode_logits_440hz,
                &g.decode_logits_440hz,
                1e-4,
                "decode 440Hz",
            );
            assert_f32_slice_close(
                &a.prefill_logits_2600hz,
                &g.prefill_logits_2600hz,
                1e-4,
                "prefill 2600Hz",
            );
        },
    );
}
