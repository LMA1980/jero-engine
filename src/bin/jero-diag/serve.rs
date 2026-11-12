//! Ephemeral loopback OpenAI-compatible HTTP server (`jero-diag --serve-openai`, ADR-045).
//!
//! Exposes a zero-daemon, single-slot HTTP/1.1 inference endpoint backed directly by
//! [`LockedModelWeights`], [`GgufTokenizer`], [`SinglePromptEngine`], and [`ArchRecipe`] so
//! unmodified external evaluation harnesses (`promptfoo`, EleutherAI `lm-evaluation-harness`) can
//! certify model answer quality without depending on control-plane daemon IPC.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jero_core::{
    ArchRecipe, CoordinationProfile, ExecutionContext, GgufFile, K3Error, ModelWeights,
    QuantizationInfo,
};
use jero_engine::{
    ClusterPhasePolicy, EnginePhase, LockedModelWeights, SinglePromptEngine, SingleSlotKvArena,
    StreamAction, SystemMemoryInspector, WorkerPool, encode_text_utf8_safe,
    infer_logits_buffer_len, resolve_affinity, sample_greedy_argmax, select_phase_policy_for_arch,
};
use jero_tokenizer::{DetokenizeStep, GgufTokenizer, Tokenizer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{DiagConfig, is_abort_requested};

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Debug, Serialize)]
struct CompletionLogprobs {
    tokens: Vec<String>,
    token_logprobs: Vec<Option<f64>>,
    top_logprobs: Vec<Option<BTreeMap<String, f64>>>,
    text_offset: Vec<usize>,
}

/// Starts the ephemeral OpenAI-compatible HTTP server on `config.serve_openai` (e.g.,
/// `127.0.0.1:0`), prints `JERO_OPENAI_LISTEN_URL=http://<ip>:<port>/v1` immediately after binding,
/// and serves requests sequentially until `--max-requests` is reached or `SIGINT`/`SIGTERM` arrives.
pub fn run_ephemeral_openai_server<W: Write>(
    config: &DiagConfig,
    mut out: W,
) -> Result<(), K3Error> {
    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --model <path>".to_owned(),
        })?;
    let bind_addr = config
        .serve_openai
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --serve-openai <ip:port>".to_owned(),
        })?;

    let n_ctx = config.n_ctx.unwrap_or(512);
    let default_max_tokens = config.max_tokens.unwrap_or(64);
    let profile = config.coord.unwrap_or(if config.phase_policy.is_some() {
        CoordinationProfile::Rayon
    } else {
        CoordinationProfile::Spin
    });

    let inspector = SystemMemoryInspector::new();
    let (worker_pool, locked_weights, recipe, resolved_policy, mut engine) =
        if let Some(policy) = config.phase_policy {
            let (locked_weights, recipe, _footprint) =
                LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
            let worker_pool = WorkerPool::for_phase_policy(
                policy,
                locked_weights.metadata(),
                16,
                config.core_mask,
                config.threads,
                profile,
            )?;
            worker_pool.verify_live_affinities()?;
            let effective = worker_pool.phase_policy().unwrap_or(policy);
            let engine = SinglePromptEngine::new().with_phase_policy(effective);
            (worker_pool, locked_weights, recipe, effective, engine)
        } else {
            let resolved_affinity =
                resolve_affinity(config.cluster.as_deref(), config.core_mask, config.threads)?;
            let worker_pool = WorkerPool::with_coordination(resolved_affinity, profile)?;
            worker_pool.verify_live_affinities()?;
            let (locked_weights, recipe, _footprint) =
                LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
            (
                worker_pool,
                locked_weights,
                recipe,
                ClusterPhasePolicy::Auto,
                SinglePromptEngine::new(),
            )
        };

    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let arch_name = locked_weights.metadata().architecture().to_owned();
    let quant_str = QuantizationInfo::from_descriptors(locked_weights.metadata().tensors()).primary;
    let model_id = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jero-local-model")
        .to_owned();

    let listener = TcpListener::bind(bind_addr).map_err(|e| K3Error::CapabilityMismatch {
        architecture: "serve_openai".to_owned(),
        reason: format!("failed to bind --serve-openai '{bind_addr}': {e}"),
    })?;
    let local_addr = listener
        .local_addr()
        .map_err(|e| K3Error::CapabilityMismatch {
            architecture: "serve_openai".to_owned(),
            reason: format!("failed to query local socket address: {e}"),
        })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| K3Error::CapabilityMismatch {
            architecture: "serve_openai".to_owned(),
            reason: format!("failed to set non-blocking mode on listener: {e}"),
        })?;

    let _ = writeln!(out, "JERO_OPENAI_LISTEN_URL=http://{local_addr}/v1");
    let _ = out.flush();

    let mut handled_requests = 0_usize;
    let mut seq_counter = 0_u64;

    while !is_abort_requested() {
        if let Some(limit) = config.max_requests
            && handled_requests >= limit
        {
            break;
        }

        match listener.accept() {
            Ok((mut stream, _peer)) => {
                // Accepted sockets on BSD/macOS inherit O_NONBLOCK from the listener
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

                let Ok(req) = read_http_request(&mut stream) else {
                    continue;
                };
                handled_requests += 1;
                seq_counter += 1;

                handle_http_request(
                    &mut stream,
                    &req,
                    seq_counter,
                    &model_id,
                    &arch_name,
                    &quant_str,
                    n_ctx,
                    default_max_tokens,
                    &gguf_view,
                    &tokenizer,
                    recipe.as_ref(),
                    &locked_weights,
                    &worker_pool,
                    resolved_policy,
                    &mut engine,
                );
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                if is_abort_requested() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    worker_pool.verify_live_affinities()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_http_request(
    stream: &mut TcpStream,
    req: &HttpRequest,
    seq_counter: u64,
    model_id: &str,
    arch_name: &str,
    quant_str: &str,
    n_ctx: usize,
    default_max_tokens: usize,
    gguf_view: &GgufFile<'_>,
    tokenizer: &GgufTokenizer,
    recipe: &dyn ArchRecipe,
    locked_weights: &LockedModelWeights,
    worker_pool: &WorkerPool,
    resolved_policy: ClusterPhasePolicy,
    engine: &mut SinglePromptEngine,
) {
    let clean_path = req
        .path
        .split_once('?')
        .map_or(req.path.as_str(), |(p, _)| p)
        .trim_end_matches('/');
    let norm_path = if clean_path.is_empty() {
        "/"
    } else {
        clean_path
    };

    if req.method.eq_ignore_ascii_case("OPTIONS") {
        let _ = write_http_json_response(stream, 204, "No Content", &json!({}));
        return;
    }

    if req.method.eq_ignore_ascii_case("GET") && norm_path.ends_with("/health") {
        let body = json!({
            "status": "ok",
            "architecture": arch_name,
            "quantization": quant_str,
            "model": model_id,
            "n_ctx": n_ctx,
        });
        let _ = write_http_json_response(stream, 200, "OK", &body);
        return;
    }

    if req.method.eq_ignore_ascii_case("GET") && norm_path.ends_with("/models") {
        let body = json!({
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": unix_timestamp_secs(),
                    "owned_by": "jero-engine",
                }
            ]
        });
        let _ = write_http_json_response(stream, 200, "OK", &body);
        return;
    }

    if req.method.eq_ignore_ascii_case("POST") && norm_path.ends_with("/chat/completions") {
        match process_chat_completions(
            &req.body,
            seq_counter,
            model_id,
            arch_name,
            n_ctx,
            default_max_tokens,
            gguf_view,
            tokenizer,
            recipe,
            locked_weights,
            worker_pool,
            engine,
        ) {
            Ok(resp_json) => {
                let _ = write_http_json_response(stream, 200, "OK", &resp_json);
            }
            Err((status, message)) => {
                let err_json = openai_error_payload(&message, status);
                let reason = if status == 400 {
                    "Bad Request"
                } else {
                    "Internal Server Error"
                };
                let _ = write_http_json_response(stream, status, reason, &err_json);
            }
        }
        return;
    }

    if req.method.eq_ignore_ascii_case("POST") && norm_path.ends_with("/completions") {
        match process_text_completions(
            &req.body,
            seq_counter,
            model_id,
            n_ctx,
            default_max_tokens,
            gguf_view,
            tokenizer,
            recipe,
            locked_weights,
            worker_pool,
            resolved_policy,
            engine,
        ) {
            Ok(resp_json) => {
                let _ = write_http_json_response(stream, 200, "OK", &resp_json);
            }
            Err((status, message)) => {
                let err_json = openai_error_payload(&message, status);
                let reason = if status == 400 {
                    "Bad Request"
                } else {
                    "Internal Server Error"
                };
                let _ = write_http_json_response(stream, status, reason, &err_json);
            }
        }
        return;
    }

    let not_found = openai_error_payload(
        &format!("unsupported endpoint: {} {}", req.method, req.path),
        404,
    );
    let _ = write_http_json_response(stream, 404, "Not Found", &not_found);
}

#[allow(clippy::too_many_arguments)]
fn process_chat_completions(
    raw_body: &[u8],
    seq_counter: u64,
    default_model_id: &str,
    arch_name: &str,
    n_ctx: usize,
    default_max_tokens: usize,
    gguf_view: &GgufFile<'_>,
    tokenizer: &GgufTokenizer,
    recipe: &dyn ArchRecipe,
    locked_weights: &LockedModelWeights,
    worker_pool: &WorkerPool,
    engine: &mut SinglePromptEngine,
) -> Result<Value, (u16, String)> {
    let payload: Value = serde_json::from_slice(raw_body)
        .map_err(|e| (400_u16, format!("invalid JSON request body: {e}")))?;

    let req_model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(default_model_id);
    let messages_val = payload
        .get("messages")
        .ok_or_else(|| (400_u16, "missing required 'messages' array".to_owned()))?;
    let messages: Vec<ChatMessage> = serde_json::from_value(messages_val.clone())
        .map_err(|e| (400_u16, format!("invalid 'messages' schema: {e}")))?;
    if messages.is_empty() {
        return Err((400_u16, "'messages' array cannot be empty".to_owned()));
    }

    let req_max_tokens = extract_requested_max_tokens(&payload).unwrap_or(default_max_tokens);
    let stop_strings = extract_stop_sequences(&payload);

    let formatted_prompt = format_chat_messages(tokenizer, arch_name, &messages);
    let mut prompt_tokens = encode_text_utf8_safe(tokenizer, &formatted_prompt)
        .map_err(|e| (400_u16, format!("prompt tokenization failed: {e}")))?;
    if prompt_tokens.is_empty() {
        prompt_tokens.push(tokenizer.special_tokens().bos_token_id.unwrap_or(0));
    }
    if prompt_tokens.len() >= n_ctx {
        return Err((
            400_u16,
            format!(
                "prompt token length ({}) meets or exceeds configured n_ctx ({n_ctx})",
                prompt_tokens.len()
            ),
        ));
    }

    let remaining_ctx = n_ctx.saturating_sub(prompt_tokens.len());
    let effective_max_tokens = req_max_tokens.min(remaining_ctx);

    let mut kv_arena = SingleSlotKvArena::from_gguf(gguf_view, n_ctx)
        .map_err(|e| (500_u16, format!("KV arena allocation failed: {e}")))?;
    let mut detok = tokenizer.streaming_detokenizer();
    let mut generated_text = String::new();
    let mut stopped_by_eos_or_stop = false;

    let summary = engine
        .evaluate_prompt_with_pool(
            recipe,
            locked_weights,
            &mut kv_arena,
            Some(worker_pool),
            &prompt_tokens,
            tokenizer.vocab_size(),
            effective_max_tokens,
            |token_id| {
                if is_abort_requested() {
                    return Ok(StreamAction::Stop);
                }
                match detok.push_token(token_id)? {
                    DetokenizeStep::Emit(piece) => {
                        generated_text.push_str(&piece);
                        if trim_matched_stop_sequence(&mut generated_text, &stop_strings) {
                            stopped_by_eos_or_stop = true;
                            return Ok(StreamAction::Stop);
                        }
                        Ok(StreamAction::Continue)
                    }
                    DetokenizeStep::Hold => Ok(StreamAction::Continue),
                    DetokenizeStep::Stop(prefix) => {
                        generated_text.push_str(&prefix);
                        let _ = trim_matched_stop_sequence(&mut generated_text, &stop_strings);
                        stopped_by_eos_or_stop = true;
                        Ok(StreamAction::Stop)
                    }
                }
            },
        )
        .map_err(|e| (500_u16, format!("forward pass failed: {e}")))?;

    if !detok.is_stopped() && !stopped_by_eos_or_stop {
        generated_text.push_str(&detok.finish());
        if trim_matched_stop_sequence(&mut generated_text, &stop_strings) {
            stopped_by_eos_or_stop = true;
        }
    }

    let finish_reason = if stopped_by_eos_or_stop {
        "stop"
    } else {
        "length"
    };
    let prompt_tok_cnt = prompt_tokens.len();
    let comp_tok_cnt = summary.generated_tokens.len();

    Ok(json!({
        "id": format!("chatcmpl-jero-{seq_counter}"),
        "object": "chat.completion",
        "created": unix_timestamp_secs(),
        "model": req_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tok_cnt,
            "completion_tokens": comp_tok_cnt,
            "total_tokens": prompt_tok_cnt + comp_tok_cnt,
        }
    }))
}

#[derive(Debug)]
enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[allow(clippy::too_many_arguments)]
fn process_text_completions(
    raw_body: &[u8],
    seq_counter: u64,
    default_model_id: &str,
    n_ctx: usize,
    default_max_tokens: usize,
    gguf_view: &GgufFile<'_>,
    tokenizer: &GgufTokenizer,
    recipe: &dyn ArchRecipe,
    locked_weights: &LockedModelWeights,
    worker_pool: &WorkerPool,
    resolved_policy: ClusterPhasePolicy,
    engine: &mut SinglePromptEngine,
) -> Result<Value, (u16, String)> {
    let payload: Value = serde_json::from_slice(raw_body)
        .map_err(|e| (400_u16, format!("invalid JSON request body: {e}")))?;

    let req_model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(default_model_id);
    let inputs = parse_completion_prompts(payload.get("prompt"))?;
    let req_max_tokens = extract_requested_max_tokens(&payload).unwrap_or(default_max_tokens);
    let stop_strings = extract_stop_sequences(&payload);
    let echo = payload
        .get("echo")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let logprobs_k = payload
        .get("logprobs")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&k| k > 0)
        .map(|k| k.min(20));

    let mut choices = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0_usize;
    let mut total_completion_tokens = 0_usize;

    for (choice_idx, input) in inputs.iter().enumerate() {
        let (raw_prompt_str, mut prompt_tokens) = match input {
            PromptInput::Text(s) => {
                let toks = encode_text_utf8_safe(tokenizer, s)
                    .map_err(|e| (400_u16, format!("prompt tokenization failed: {e}")))?;
                (s.clone(), toks)
            }
            PromptInput::Tokens(toks) => {
                let decoded = tokenizer.decode(toks).unwrap_or_default();
                (decoded, toks.clone())
            }
        };

        if prompt_tokens.is_empty() {
            prompt_tokens.push(tokenizer.special_tokens().bos_token_id.unwrap_or(0));
        }
        if prompt_tokens.len() > n_ctx || (req_max_tokens > 0 && prompt_tokens.len() >= n_ctx) {
            return Err((
                400_u16,
                format!(
                    "prompt token length ({}) exceeds configured n_ctx ({n_ctx})",
                    prompt_tokens.len()
                ),
            ));
        }

        let remaining_ctx = n_ctx.saturating_sub(prompt_tokens.len());
        let effective_max_tokens = req_max_tokens.min(remaining_ctx);

        if let Some(top_k) = logprobs_k {
            let (out_text, gen_count, finish_reason, lp) = evaluate_completion_with_logprobs(
                &raw_prompt_str,
                &prompt_tokens,
                effective_max_tokens,
                echo,
                top_k,
                &stop_strings,
                n_ctx,
                gguf_view,
                tokenizer,
                recipe,
                locked_weights,
                worker_pool,
                resolved_policy,
            )?;
            total_prompt_tokens += prompt_tokens.len();
            total_completion_tokens += gen_count;
            choices.push(json!({
                "index": choice_idx,
                "text": out_text,
                "logprobs": lp,
                "finish_reason": finish_reason,
            }));
        } else {
            let mut kv_arena = SingleSlotKvArena::from_gguf(gguf_view, n_ctx)
                .map_err(|e| (500_u16, format!("KV arena allocation failed: {e}")))?;
            let mut detok = tokenizer.streaming_detokenizer();
            let mut generated_text = String::new();
            let mut stopped_by_eos = false;

            let summary = engine
                .evaluate_prompt_with_pool(
                    recipe,
                    locked_weights,
                    &mut kv_arena,
                    Some(worker_pool),
                    &prompt_tokens,
                    tokenizer.vocab_size(),
                    effective_max_tokens,
                    |token_id| {
                        if is_abort_requested() {
                            return Ok(StreamAction::Stop);
                        }
                        match detok.push_token(token_id)? {
                            DetokenizeStep::Emit(piece) => {
                                generated_text.push_str(&piece);
                                if trim_matched_stop_sequence(&mut generated_text, &stop_strings) {
                                    stopped_by_eos = true;
                                    return Ok(StreamAction::Stop);
                                }
                                Ok(StreamAction::Continue)
                            }
                            DetokenizeStep::Hold => Ok(StreamAction::Continue),
                            DetokenizeStep::Stop(prefix) => {
                                generated_text.push_str(&prefix);
                                let _ =
                                    trim_matched_stop_sequence(&mut generated_text, &stop_strings);
                                stopped_by_eos = true;
                                Ok(StreamAction::Stop)
                            }
                        }
                    },
                )
                .map_err(|e| (500_u16, format!("forward pass failed: {e}")))?;

            if !detok.is_stopped() && !stopped_by_eos {
                generated_text.push_str(&detok.finish());
                if trim_matched_stop_sequence(&mut generated_text, &stop_strings) {
                    stopped_by_eos = true;
                }
            }

            let final_text = if echo {
                format!("{raw_prompt_str}{generated_text}")
            } else {
                generated_text
            };
            let finish_reason = if stopped_by_eos { "stop" } else { "length" };
            total_prompt_tokens += prompt_tokens.len();
            total_completion_tokens += summary.generated_tokens.len();

            choices.push(json!({
                "index": choice_idx,
                "text": final_text,
                "logprobs": Value::Null,
                "finish_reason": finish_reason,
            }));
        }
    }

    Ok(json!({
        "id": format!("cmpl-jero-{seq_counter}"),
        "object": "text_completion",
        "created": unix_timestamp_secs(),
        "model": req_model,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }
    }))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_completion_with_logprobs(
    raw_prompt_str: &str,
    prompt_tokens: &[u32],
    max_tokens: usize,
    echo: bool,
    top_k: usize,
    stop_strings: &[String],
    n_ctx: usize,
    gguf_view: &GgufFile<'_>,
    tokenizer: &GgufTokenizer,
    recipe: &dyn ArchRecipe,
    locked_weights: &LockedModelWeights,
    worker_pool: &WorkerPool,
    resolved_policy: ClusterPhasePolicy,
) -> Result<(String, usize, &'static str, CompletionLogprobs), (u16, String)> {
    let mut kv_arena = SingleSlotKvArena::from_gguf(gguf_view, n_ctx)
        .map_err(|e| (500_u16, format!("KV arena allocation failed: {e}")))?;
    let vocab_size = tokenizer.vocab_size();
    let logits_len = infer_logits_buffer_len(locked_weights, vocab_size);
    let mut logits = vec![0.0_f32; logits_len];

    let effective_policy = match resolved_policy {
        ClusterPhasePolicy::Auto => worker_pool.phase_policy().unwrap_or_else(|| {
            select_phase_policy_for_arch(
                recipe.architecture_id(),
                prompt_tokens.len(),
                worker_pool.affinity().effective_mask,
            )
        }),
        explicit => explicit,
    };

    let mut tokens_str = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut top_logprobs = Vec::new();
    let mut text_offset = Vec::new();
    let mut running_offset = 0_usize;

    if echo {
        let first_slice = &prompt_tokens[..1];
        let mut ctx =
            ExecutionContext::new(first_slice, n_ctx, &mut logits, kv_arena.as_mut_slice())
                .map_err(|e| (500_u16, format!("ExecutionContext init failed: {e}")))?;
        worker_pool
            .install_for_phase(EnginePhase::Prefill, effective_policy, || {
                recipe.prefill_step(&mut ctx, locked_weights)
            })
            .map_err(|e| (500_u16, format!("prefill_step failed: {e}")))?;

        let first_piece = decode_token_piece_lossy(tokenizer, prompt_tokens[0]);
        text_offset.push(running_offset);
        running_offset += first_piece.len();
        tokens_str.push(first_piece);
        token_logprobs.push(None);
        top_logprobs.push(None);

        for &target_tok in &prompt_tokens[1..] {
            let sample_len = vocab_size.min(ctx.logits().len());
            let (tok_lp, top_map) =
                compute_step_logprobs(&ctx.logits()[..sample_len], target_tok, top_k, tokenizer);
            let piece = decode_token_piece_lossy(tokenizer, target_tok);
            text_offset.push(running_offset);
            running_offset += piece.len();
            tokens_str.push(piece);
            token_logprobs.push(Some(tok_lp));
            top_logprobs.push(Some(top_map));

            ctx.prepare_decode(target_tok)
                .map_err(|e| (500_u16, format!("prepare_decode failed: {e}")))?;
            worker_pool
                .install_for_phase(EnginePhase::Decode, effective_policy, || {
                    recipe.decode_step(&mut ctx, locked_weights)
                })
                .map_err(|e| (500_u16, format!("decode_step failed: {e}")))?;
        }

        let (gen_text, gen_count, finish_reason) = generate_from_ready_context(
            &mut ctx,
            max_tokens,
            n_ctx,
            vocab_size,
            top_k,
            stop_strings,
            tokenizer,
            recipe,
            locked_weights,
            worker_pool,
            effective_policy,
            &mut running_offset,
            &mut tokens_str,
            &mut token_logprobs,
            &mut top_logprobs,
            &mut text_offset,
        )?;
        let echoed_prefix: String = tokens_str[..prompt_tokens.len()].concat();
        let prefix = if echoed_prefix.is_empty() {
            raw_prompt_str.to_owned()
        } else {
            echoed_prefix
        };
        return Ok((
            format!("{prefix}{gen_text}"),
            gen_count,
            finish_reason,
            CompletionLogprobs {
                tokens: tokens_str,
                token_logprobs,
                top_logprobs,
                text_offset,
            },
        ));
    }

    let mut ctx = ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, kv_arena.as_mut_slice())
        .map_err(|e| (500_u16, format!("ExecutionContext init failed: {e}")))?;
    worker_pool
        .install_for_phase(EnginePhase::Prefill, effective_policy, || {
            recipe.prefill_step(&mut ctx, locked_weights)
        })
        .map_err(|e| (500_u16, format!("prefill_step failed: {e}")))?;

    let (gen_text, gen_count, finish_reason) = generate_from_ready_context(
        &mut ctx,
        max_tokens,
        n_ctx,
        vocab_size,
        top_k,
        stop_strings,
        tokenizer,
        recipe,
        locked_weights,
        worker_pool,
        effective_policy,
        &mut running_offset,
        &mut tokens_str,
        &mut token_logprobs,
        &mut top_logprobs,
        &mut text_offset,
    )?;

    Ok((
        gen_text,
        gen_count,
        finish_reason,
        CompletionLogprobs {
            tokens: tokens_str,
            token_logprobs,
            top_logprobs,
            text_offset,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn generate_from_ready_context(
    ctx: &mut ExecutionContext<'_>,
    max_tokens: usize,
    n_ctx: usize,
    vocab_size: usize,
    top_k: usize,
    stop_strings: &[String],
    tokenizer: &GgufTokenizer,
    recipe: &dyn ArchRecipe,
    locked_weights: &dyn ModelWeights,
    worker_pool: &WorkerPool,
    effective_policy: ClusterPhasePolicy,
    running_offset: &mut usize,
    tokens_str: &mut Vec<String>,
    token_logprobs: &mut Vec<Option<f64>>,
    top_logprobs: &mut Vec<Option<BTreeMap<String, f64>>>,
    text_offset: &mut Vec<usize>,
) -> Result<(String, usize, &'static str), (u16, String)> {
    if max_tokens == 0 {
        return Ok((String::new(), 0, "length"));
    }

    let mut generated_text = String::new();
    let mut gen_count = 0_usize;
    let mut stopped = false;

    let sample_len = vocab_size.min(ctx.logits().len());
    let mut current_token = sample_greedy_argmax(&ctx.logits()[..sample_len])
        .ok_or_else(|| (500_u16, "empty logits buffer".to_owned()))?;

    loop {
        let step_len = vocab_size.min(ctx.logits().len());
        let (lp, top_map) =
            compute_step_logprobs(&ctx.logits()[..step_len], current_token, top_k, tokenizer);
        gen_count += 1;

        if tokenizer.special_tokens().is_stop(current_token) {
            stopped = true;
            break;
        }

        let piece = decode_token_piece_lossy(tokenizer, current_token);
        text_offset.push(*running_offset);
        *running_offset += piece.len();
        tokens_str.push(piece.clone());
        token_logprobs.push(Some(lp));
        top_logprobs.push(Some(top_map));
        generated_text.push_str(&piece);

        if trim_matched_stop_sequence(&mut generated_text, stop_strings) {
            stopped = true;
            break;
        }

        if gen_count >= max_tokens || ctx.position() >= n_ctx || is_abort_requested() {
            break;
        }

        ctx.prepare_decode(current_token)
            .map_err(|e| (500_u16, format!("prepare_decode failed: {e}")))?;
        worker_pool
            .install_for_phase(EnginePhase::Decode, effective_policy, || {
                recipe.decode_step(ctx, locked_weights)
            })
            .map_err(|e| (500_u16, format!("decode_step failed: {e}")))?;

        let next_len = vocab_size.min(ctx.logits().len());
        current_token = sample_greedy_argmax(&ctx.logits()[..next_len])
            .ok_or_else(|| (500_u16, "empty decode logits".to_owned()))?;
    }

    let finish_reason = if stopped { "stop" } else { "length" };
    Ok((generated_text, gen_count, finish_reason))
}

fn compute_step_logprobs(
    logits: &[f32],
    chosen_token: u32,
    top_k: usize,
    tokenizer: &GgufTokenizer,
) -> (f64, BTreeMap<String, f64>) {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_f64 = f64::from(max_logit);
    let sum_exp: f64 = logits.iter().map(|&l| (f64::from(l) - max_f64).exp()).sum();
    let log_z = max_f64 + sum_exp.max(1.0e-300).ln();

    let chosen_lp = logits
        .get(chosen_token as usize)
        .map_or(f64::NEG_INFINITY, |&l| f64::from(l) - log_z);

    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(i, &l)| u32::try_from(i).ok().map(|id| (id, l)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut top_map = BTreeMap::new();
    for (tok_id, raw_l) in ranked.into_iter().take(top_k.max(1)) {
        let key = tokenizer
            .id_to_token(tok_id)
            .map_or_else(|| format!("<tok:{tok_id}>"), str::to_owned);
        top_map.entry(key).or_insert(f64::from(raw_l) - log_z);
    }

    (chosen_lp, top_map)
}

fn decode_token_piece_lossy(tokenizer: &GgufTokenizer, token_id: u32) -> String {
    tokenizer
        .decode_piece(token_id)
        .ok()
        .flatten()
        .unwrap_or_default()
}

fn format_chat_messages(
    tokenizer: &GgufTokenizer,
    arch_name: &str,
    messages: &[ChatMessage],
) -> String {
    let extracted: Vec<(&str, String)> = messages
        .iter()
        .map(|m| (m.role.as_str(), extract_message_text(&m.content)))
        .collect();

    // Avoid double-wrapping when a caller (e.g. `lm_eval --apply_chat_template`) already formatted markers
    if extracted.len() == 1 {
        let (_, ref single_text) = extracted[0];
        if single_text.contains("<|im_start|>") || single_text.contains("[INST]") {
            return single_text.clone();
        }
    }

    let tmpl = tokenizer.chat_template().raw_template().unwrap_or_default();
    if tmpl.contains("<|im_start|>") || arch_name == "qwen2" || arch_name == "dory" {
        let mut buf = String::new();
        if arch_name == "dory" {
            buf.push_str("<s>");
        }
        for (role, text) in &extracted {
            buf.push_str("<|im_start|>");
            buf.push_str(role);
            buf.push('\n');
            buf.push_str(text);
            buf.push_str("<|im_end|>\n");
        }
        buf.push_str("<|im_start|>assistant\n");
        if arch_name == "dory" {
            buf.push_str("<think></think>");
        }
        return buf;
    }

    let mut combined = String::new();
    for (role, text) in &extracted {
        if role.eq_ignore_ascii_case("system") {
            combined.push_str(text);
            combined.push_str("\n\n");
        } else if role.eq_ignore_ascii_case("assistant") {
            combined.push_str(text);
            combined.push('\n');
        } else {
            combined.push_str(text);
        }
    }
    tokenizer.chat_template().format_prompt(combined.trim())
}

fn extract_message_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(txt) = p.get("text").and_then(Value::as_str) {
                    out.push_str(txt);
                } else if let Some(s) = p.as_str() {
                    out.push_str(s);
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn parse_completion_prompts(
    prompt_field: Option<&Value>,
) -> Result<Vec<PromptInput>, (u16, String)> {
    let val =
        prompt_field.ok_or_else(|| (400_u16, "missing required 'prompt' field".to_owned()))?;
    match val {
        Value::String(s) => Ok(vec![PromptInput::Text(s.clone())]),
        Value::Array(arr) if !arr.is_empty() => {
            if arr.iter().all(Value::is_string) {
                Ok(arr
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|s| PromptInput::Text(s.to_owned()))
                    .collect())
            } else if arr.iter().all(Value::is_u64) {
                let toks: Result<Vec<u32>, _> = arr
                    .iter()
                    .map(|v| {
                        u32::try_from(v.as_u64().unwrap_or(0))
                            .map_err(|_| (400_u16, "token ID overflows u32".to_owned()))
                    })
                    .collect();
                Ok(vec![PromptInput::Tokens(toks?)])
            } else if arr.iter().all(Value::is_array) {
                let mut batch = Vec::with_capacity(arr.len());
                for sub in arr {
                    let sub_arr = sub.as_array().expect("checked is_array");
                    let toks: Result<Vec<u32>, _> = sub_arr
                        .iter()
                        .map(|v| {
                            let raw = v
                                .as_u64()
                                .ok_or_else(|| (400_u16, "expected integer token ID".to_owned()))?;
                            u32::try_from(raw)
                                .map_err(|_| (400_u16, "token ID overflows u32".to_owned()))
                        })
                        .collect();
                    batch.push(PromptInput::Tokens(toks?));
                }
                Ok(batch)
            } else {
                Err((
                    400_u16,
                    "'prompt' array must contain strings or integer token IDs".to_owned(),
                ))
            }
        }
        _ => Err((
            400_u16,
            "'prompt' must be a non-empty string or array".to_owned(),
        )),
    }
}

fn extract_requested_max_tokens(payload: &Value) -> Option<usize> {
    for key in ["max_tokens", "max_completion_tokens", "max_gen_toks"] {
        if let Some(n) = payload.get(key).and_then(Value::as_u64)
            && let Ok(parsed) = usize::try_from(n)
        {
            return Some(parsed);
        }
    }
    None
}

fn extract_stop_sequences(payload: &Value) -> Vec<String> {
    let mut stops = Vec::new();
    for key in ["stop", "until"] {
        match payload.get(key) {
            Some(Value::String(s)) if !s.is_empty() => stops.push(s.clone()),
            Some(Value::Array(items)) => {
                for item in items {
                    if let Some(s) = item.as_str()
                        && !s.is_empty()
                    {
                        stops.push(s.to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    stops
}

fn trim_matched_stop_sequence(text: &mut String, stop_strings: &[String]) -> bool {
    for stop in stop_strings {
        if !stop.is_empty()
            && let Some(pos) = text.find(stop.as_str())
        {
            text.truncate(pos);
            return true;
        }
    }
    false
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let cloned = stream
        .try_clone()
        .map_err(|e| format!("stream clone failed: {e}"))?;
    let mut reader = BufReader::new(cloned);

    let mut request_line = String::new();
    let n = reader
        .read_line(&mut request_line)
        .map_err(|e| format!("read request line failed: {e}"))?;
    if n == 0 {
        return Err("empty connection".to_owned());
    }

    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "missing HTTP method".to_owned())?
        .to_owned();
    let path = parts
        .next()
        .ok_or_else(|| "missing HTTP path".to_owned())?
        .to_owned();

    let mut content_length: usize = 0;
    let mut is_chunked = false;
    let mut expect_continue = false;

    for _ in 0..128 {
        let mut header_line = String::new();
        let read_bytes = reader
            .read_line(&mut header_line)
            .map_err(|e| format!("header read error: {e}"))?;
        if read_bytes == 0 || header_line == "\r\n" || header_line == "\n" {
            break;
        }
        if let Some((name, val)) = header_line.split_once(':') {
            let k = name.trim();
            let v = val.trim();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid Content-Length '{v}'"))?;
            } else if k.eq_ignore_ascii_case("transfer-encoding")
                && v.to_ascii_lowercase().contains("chunked")
            {
                is_chunked = true;
            } else if k.eq_ignore_ascii_case("expect") && v.eq_ignore_ascii_case("100-continue") {
                expect_continue = true;
            }
        }
    }

    if expect_continue {
        let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        let _ = stream.flush();
    }

    let body = if is_chunked {
        read_chunked_body(&mut reader)?
    } else if content_length > 0 {
        if content_length > (16 << 20) {
            return Err("HTTP body exceeds 16 MiB limit".to_owned());
        }
        let mut buf = vec![0_u8; content_length];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("failed reading HTTP body: {e}"))?;
        buf
    } else {
        Vec::new()
    };

    Ok(HttpRequest { method, path, body })
}

fn read_chunked_body<R: BufRead>(reader: &mut R) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let mut size_line = String::new();
        let n = reader
            .read_line(&mut size_line)
            .map_err(|e| format!("chunk size read error: {e}"))?;
        if n == 0 {
            break;
        }
        let hex_part = size_line.trim().split(';').next().unwrap_or("").trim();
        let chunk_len = usize::from_str_radix(hex_part, 16)
            .map_err(|_| format!("invalid chunk size '{hex_part}'"))?;
        if chunk_len == 0 {
            let mut trailer = String::new();
            let _ = reader.read_line(&mut trailer);
            break;
        }
        if out.len().saturating_add(chunk_len) > (16 << 20) {
            return Err("chunked HTTP body exceeds 16 MiB limit".to_owned());
        }
        let prev_len = out.len();
        out.resize(prev_len + chunk_len, 0);
        reader
            .read_exact(&mut out[prev_len..])
            .map_err(|e| format!("chunk payload read error: {e}"))?;
        let mut crlf = [0_u8; 2];
        let _ = reader.read_exact(&mut crlf);
    }
    Ok(out)
}

fn write_http_json_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason: &str,
    payload: &Value,
) -> std::io::Result<()> {
    let body_bytes = if status_code == 204 {
        Vec::new()
    } else {
        serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec())
    };
    let header = format!(
        "HTTP/1.1 {status_code} {reason}\r\n\
         Content-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(header.as_bytes())?;
    if !body_bytes.is_empty() {
        stream.write_all(&body_bytes)?;
    }
    stream.flush()
}

fn openai_error_payload(message: &str, status_code: u16) -> Value {
    let err_type = if status_code < 500 {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json!({
        "error": {
            "message": message,
            "type": err_type,
            "code": status_code,
        }
    })
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
