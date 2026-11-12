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

use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType, TensorView,
};
use jero_engine::{
    ClusterPhasePolicy, LockedModelWeights, ParityDumpReport, SinglePromptEngine,
    SingleSlotKvArena, StreamAction, SystemMemoryInspector, evaluate_parity_snapshot,
    verify_parity_against_golden, verify_prompt_sensitivity,
};
use jero_tokenizer::{DetokenizeStep, GgufTokenizer};
use serde_json::{Value, json};

const JERO_DIAG_BIN: &str = env!("CARGO_BIN_EXE_jero-diag");
const GOLDEN_LLAMA_Q8_0_JSON: &str =
    include_str!("fixtures/golden_parity/llama_authentic_q8_0.json");

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
    let parsed = GgufFile::parse(&raw).expect("parse llama template gguf");
    let descriptors = parsed.metadata().tensors().to_vec();

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
            // Suppress control tokens 0..=4 in `output.weight` so greedy decode emits normal words (5..16)
            if desc.name() == "output.weight" {
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
