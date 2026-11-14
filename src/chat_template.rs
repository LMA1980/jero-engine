//! Model-family chat template framing, single-BOS coordination, multi-EOS/EOT discovery,
//! and EOT-free detokenization helpers (`#152`, `#155`, `#156`).

use jero_core::K3Error;
use jero_tokenizer::{DetokenizeStep, GgufTokenizer, Tokenizer};

use crate::certification::encode_chunk_without_bos;

/// Canonical single-turn chat template family resolved from GGUF `tokenizer.chat_template`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateFamily {
    /// No `tokenizer.chat_template` or unrecognized opaque template: passthrough without framing.
    Plain,
    /// Non-Jinja explicit placeholder template (`{{prompt}}` or `{{content}}`).
    CustomPlaceholder,
    /// Llama 3 / 3.1 / 3.2 / 3.3 (`<|start_header_id|>...<|end_header_id|>`).
    Llama3,
    /// `ChatML` (`Qwen2`, `Qwen2.5`, `Qwen3`, `Yi`, `Dory`, etc. with `<|im_start|>`).
    ChatMl,
    /// Phi-3 / Phi-4 (`<|user|>...<|end|>\n<|assistant|>`).
    Phi3,
    /// Gemma 2 / Gemma 3 (`<start_of_turn>user...<end_of_turn>\n<start_of_turn>model`).
    Gemma,
    /// Mistral / Llama 2 instruct (`[INST] ... [/INST]`).
    Mistral,
    /// `DeepSeek` instruct (`<｜User｜>...<｜Assistant｜>`).
    DeepSeek,
}

/// Known control/special template markers sorted longest-first at scan time.
const CHAT_MARKERS: &[&str] = &[
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<|end_of_sentence|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<start_of_turn>",
    "<|assistant|>",
    "<｜Assistant｜>",
    "<end_of_turn>",
    "<|im_start|>",
    "<|Assistant|>",
    "<|system|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<｜User｜>",
    "<|user|>",
    "<|User|>",
    "[/INST]",
    "[INST]",
    "<|end|>",
    "<|EOT|>",
    "<bos>",
    "<eos>",
    "<s>",
    "</s>",
];

/// Canonical stop token strings across `ChatML`, Llama-3, Phi-3/4, Gemma 2/3, `DeepSeek`, and `SentencePiece`.
const STOP_TOKEN_MARKERS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<｜end▁of▁sentence｜>",
    "<|end_of_sentence|>",
    "</s>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|EOT|>",
];

/// Leading BOS marker strings matched during single-BOS deduplication.
const BOS_MARKERS: &[&str] = &[
    "<|begin_of_text|>",
    "<｜begin▁of▁sentence｜>",
    "<bos>",
    "<s>",
];

/// Detects the canonical chat-template family from a raw `tokenizer.chat_template` string.
#[must_use]
pub fn detect_chat_template_family(raw_template: &str) -> ChatTemplateFamily {
    let trimmed = raw_template.trim();
    if trimmed.is_empty() {
        return ChatTemplateFamily::Plain;
    }

    let is_jinja =
        trimmed.contains("{%") || trimmed.contains("{{-") || trimmed.contains("{{ message");
    if !is_jinja && (trimmed.contains("{{prompt}}") || trimmed.contains("{{content}}")) {
        return ChatTemplateFamily::CustomPlaceholder;
    }

    if trimmed.contains("<|start_header_id|>") {
        return ChatTemplateFamily::Llama3;
    }
    if trimmed.contains("<|im_start|>") {
        return ChatTemplateFamily::ChatMl;
    }
    if (trimmed.contains("<|user|>")
        && (trimmed.contains("<|assistant|>") || trimmed.contains("<|end|>")))
        || (trimmed.contains("<|assistant|>") && trimmed.contains("<|end|>"))
    {
        return ChatTemplateFamily::Phi3;
    }
    if trimmed.contains("<start_of_turn>") {
        return ChatTemplateFamily::Gemma;
    }
    if trimmed.contains("[INST]") {
        return ChatTemplateFamily::Mistral;
    }
    if trimmed.contains("<｜User｜>") || trimmed.contains("<|User|>") {
        return ChatTemplateFamily::DeepSeek;
    }

    ChatTemplateFamily::Plain
}

fn resolve_template_bos_prefix(
    tokenizer: &GgufTokenizer,
    raw_template: &str,
    literal_marker: &str,
) -> String {
    if raw_template.contains(literal_marker) {
        return literal_marker.to_owned();
    }
    if raw_template.contains("bos_token") {
        if let Some(bos_id) = tokenizer.special_tokens().bos_token_id
            && let Some(bos_str) = tokenizer.id_to_token(bos_id)
            && !bos_str.is_empty()
        {
            return bos_str.to_owned();
        }
        if tokenizer.token_to_id(literal_marker).is_some() {
            return literal_marker.to_owned();
        }
    }
    String::new()
}

/// Formats a single-turn user prompt according to the GGUF `tokenizer.chat_template`.
///
/// Base models with no `tokenizer.chat_template` (or unrecognized templates) pass `user_prompt`
/// through unmodified.
#[must_use]
pub fn format_single_turn_prompt(tokenizer: &GgufTokenizer, user_prompt: &str) -> String {
    let Some(raw) = tokenizer.chat_template().raw_template() else {
        return user_prompt.to_owned();
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return user_prompt.to_owned();
    }

    match detect_chat_template_family(trimmed) {
        ChatTemplateFamily::Plain => user_prompt.to_owned(),
        ChatTemplateFamily::CustomPlaceholder => {
            if trimmed.contains("{{prompt}}") {
                trimmed.replace("{{prompt}}", user_prompt)
            } else {
                trimmed.replace("{{content}}", user_prompt)
            }
        }
        ChatTemplateFamily::Llama3 => {
            let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<|begin_of_text|>");
            format!(
                "{bos}<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        }
        ChatTemplateFamily::ChatMl => {
            if trimmed.contains("<think>") && trimmed.contains("reasoning effort") {
                return tokenizer
                    .chat_template()
                    .format_dory_prompt(user_prompt, None, false);
            }
            let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<s>");
            format!("{bos}<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
        }
        ChatTemplateFamily::Phi3 => {
            let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<s>");
            format!("{bos}<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n")
        }
        ChatTemplateFamily::Gemma => {
            let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<bos>");
            format!("{bos}<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n")
        }
        ChatTemplateFamily::Mistral => {
            let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<s>");
            format!("{bos}[INST] {user_prompt} [/INST]")
        }
        ChatTemplateFamily::DeepSeek => {
            if trimmed.contains("<｜User｜>") {
                let bos =
                    resolve_template_bos_prefix(tokenizer, trimmed, "<｜begin▁of▁sentence｜>");
                format!("{bos}<｜User｜>{user_prompt}<｜Assistant｜>")
            } else {
                let bos = resolve_template_bos_prefix(tokenizer, trimmed, "<|begin_of_sentence|>");
                format!("{bos}<|User|>{user_prompt}<|Assistant|>")
            }
        }
    }
}

/// Discovers all deduplicated stop token IDs across GGUF metadata and chat-template EOT markers.
///
/// Covers `tokenizer.ggml.eos_token_id`, `tokenizer.ggml.eot_token_id`,
/// `tokenizer.special_tokens().stop_tokens`, plus vocabulary lookups for `<|eot_id|>`,
/// `<|im_end|>`, `<|end|>`, `<end_of_turn>`, `<｜end▁of▁sentence｜>`, `</s>`, and `<|endoftext|>`.
#[must_use]
pub fn discover_stop_token_ids(tokenizer: &GgufTokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut push_unique = |id: u32| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };

    let special = tokenizer.special_tokens();
    if let Some(eos) = special.eos_token_id {
        push_unique(eos);
    }
    if let Some(eot) = special.eot_token_id {
        push_unique(eot);
    }
    for &id in &special.stop_tokens {
        push_unique(id);
    }
    for marker in STOP_TOKEN_MARKERS {
        if let Some(id) = tokenizer.token_to_id(marker) {
            push_unique(id);
        }
    }
    ids
}

fn matching_leading_bos_prefix<'a>(
    tokenizer: &GgufTokenizer,
    text: &'a str,
) -> Option<(&'a str, u32)> {
    if let Some(bos_id) = tokenizer.special_tokens().bos_token_id
        && let Some(bos_str) = tokenizer.id_to_token(bos_id)
        && !bos_str.is_empty()
        && text.starts_with(bos_str)
    {
        return Some((&text[..bos_str.len()], bos_id));
    }

    for &marker in BOS_MARKERS {
        if text.starts_with(marker)
            && let Some(marker_id) = tokenizer.token_to_id(marker)
        {
            return Some((&text[..marker.len()], marker_id));
        }
    }

    None
}

fn find_next_chat_marker(tokenizer: &GgufTokenizer, text: &str) -> Option<(usize, usize, u32)> {
    let mut best: Option<(usize, usize, u32)> = None;
    for &marker in CHAT_MARKERS {
        let Some(pos) = text.find(marker) else {
            continue;
        };
        let Some(id) = tokenizer.token_to_id(marker) else {
            continue;
        };
        let should_replace = match best {
            None => true,
            Some((best_pos, best_len, _)) => {
                pos < best_pos || (pos == best_pos && marker.len() > best_len)
            }
        };
        if should_replace {
            best = Some((pos, marker.len(), id));
        }
    }
    best
}

/// Encodes `text` with UTF-8/byte fallback while matching special chat markers directly to their
/// vocabulary IDs (even when stored as `UserDefined` or `Normal`) and coordinating `add_bos` so BOS
/// (`<|begin_of_text|>`, `<bos>`, `<s>`, `<｜begin▁of▁sentence｜>`) appears at most once.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if an unmapped byte has neither a `<0xXX>` byte token
/// nor a fallback `padding_token_id`.
pub fn encode_formatted_prompt_utf8_safe(
    tokenizer: &GgufTokenizer,
    text: &str,
    coordinate_single_bos: bool,
) -> Result<Vec<u32>, K3Error> {
    let mut out = Vec::new();
    let add_bos = tokenizer.special_tokens().add_bos_token;
    let bos_id = tokenizer.special_tokens().bos_token_id;

    let mut rest = text;
    if coordinate_single_bos
        && let Some((leading_str, leading_id)) = matching_leading_bos_prefix(tokenizer, rest)
    {
        // Template already supplied an explicit leading BOS token: emit it once (whether add_bos is
        // true or false) and consume the literal prefix before encoding remaining turns.
        out.push(bos_id.unwrap_or(leading_id));
        rest = &rest[leading_str.len()..];
    } else if add_bos && let Some(id) = bos_id {
        out.push(id);
    }

    let mut cursor = 0_usize;
    while cursor < rest.len() {
        let remaining = &rest[cursor..];
        if let Some((offset, marker_len, marker_id)) = find_next_chat_marker(tokenizer, remaining) {
            if offset > 0 {
                encode_chunk_without_bos(tokenizer, &remaining[..offset], &mut out)?;
            }
            out.push(marker_id);
            cursor += offset + marker_len;
        } else {
            encode_chunk_without_bos(tokenizer, remaining, &mut out)?;
            break;
        }
    }

    if coordinate_single_bos
        && let Some(id) = bos_id
        && out.len() >= 2
        && out[0] == id
        && out[1] == id
    {
        out.remove(0);
    }

    Ok(out)
}

/// Detokenizes `tokens` into user-facing text while excluding any terminal or configured
/// EOS/EOT stop tokens (`#156`).
///
/// # Errors
///
/// Propagates vocabulary decoding errors from [`GgufTokenizer`].
pub fn detokenize_excluding_stop_tokens(
    tokenizer: &GgufTokenizer,
    tokens: &[u32],
    extra_stop_ids: &[u32],
) -> Result<String, K3Error> {
    let mut stop_ids = discover_stop_token_ids(tokenizer);
    for &id in extra_stop_ids {
        if !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
    }

    let mut detok = tokenizer.streaming_detokenizer();
    let mut text = String::new();
    for &tok in tokens {
        if stop_ids.contains(&tok) {
            break;
        }
        match detok.push_token(tok)? {
            DetokenizeStep::Emit(piece) => text.push_str(&piece),
            DetokenizeStep::Hold => {}
            DetokenizeStep::Stop(prefix) => {
                text.push_str(&prefix);
                break;
            }
        }
    }
    if !detok.is_stopped() {
        text.push_str(&detok.finish());
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::test_helpers::write_gguf_value;
    use jero_core::{GgufArray, GgufFile, GgufValue};

    const GGUF_MAGIC: u32 = 0x4655_4747;
    const GGUF_VERSION_V3: u32 = 3;

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        let len = u64::try_from(s.len()).expect("len");
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    struct TokGgufSpec<'a> {
        model_kind: &'a str,
        vocab: &'a [&'a str],
        token_types: &'a [i32],
        bos_token: Option<u32>,
        eos_token: Option<u32>,
        eot_token: Option<u32>,
        add_bos: Option<bool>,
        chat_template: Option<&'a str>,
    }

    fn build_tokenizer_gguf(spec: &TokGgufSpec<'_>) -> Vec<u8> {
        let mut kvs: Vec<(&str, GgufValue<'_>)> = vec![
            ("general.architecture", GgufValue::String("llama")),
            ("tokenizer.ggml.model", GgufValue::String(spec.model_kind)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(spec.vocab.to_vec())),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(GgufArray::Int32(spec.token_types.to_vec())),
            ),
        ];
        if let Some(id) = spec.bos_token {
            kvs.push(("tokenizer.ggml.bos_token_id", GgufValue::Uint32(id)));
        }
        if let Some(id) = spec.eos_token {
            kvs.push(("tokenizer.ggml.eos_token_id", GgufValue::Uint32(id)));
        }
        if let Some(id) = spec.eot_token {
            kvs.push(("tokenizer.ggml.eot_token_id", GgufValue::Uint32(id)));
        }
        if let Some(flag) = spec.add_bos {
            kvs.push(("tokenizer.ggml.add_bos_token", GgufValue::Bool(flag)));
        }
        if let Some(tmpl) = spec.chat_template {
            kvs.push(("tokenizer.chat_template", GgufValue::String(tmpl)));
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&u64::try_from(kvs.len()).expect("kv").to_le_bytes());
        for (k, v) in &kvs {
            write_gguf_string(&mut buf, k);
            write_gguf_value(&mut buf, v);
        }
        buf
    }

    const TEST_VOCAB: &[&str] = &[
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<bos>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<s>",
        "</s>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "hi",
        "user",
        "assistant",
        "\n\n",
        "\n",
    ];

    fn default_test_types() -> Vec<i32> {
        let mut types = vec![3_i32; TEST_VOCAB.len()];
        for ty in types.iter_mut().skip(18) {
            *ty = 1;
        }
        types
    }

    #[test]
    fn test_detect_and_format_llama3_chatml_and_phi3() {
        let types = default_test_types();

        // 1. Llama-3 (with bos_token Jinja reference and add_bos=true -> single leading BOS)
        let l3_tmpl = "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";
        let l3_bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "gpt2",
            vocab: TEST_VOCAB,
            token_types: &types,
            bos_token: Some(0),
            eos_token: Some(3),
            eot_token: None,
            add_bos: Some(true),
            chat_template: Some(l3_tmpl),
        });
        let l3_gguf = GgufFile::parse(&l3_bytes).expect("parse");
        let l3_tok = GgufTokenizer::from_gguf(&l3_gguf).expect("tok");
        let formatted_l3 = format_single_turn_prompt(&l3_tok, "hi");
        assert_eq!(
            formatted_l3,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        let toks_l3 = encode_formatted_prompt_utf8_safe(&l3_tok, &formatted_l3, true).expect("enc");
        assert_eq!(toks_l3[0], 0);
        assert_ne!(toks_l3[1], 0, "must emit BOS at most once");
        assert_eq!(toks_l3.iter().filter(|&&id| id == 0).count(), 1);
        assert!(toks_l3.contains(&1));
        assert!(toks_l3.contains(&2));
        assert!(toks_l3.contains(&3));

        // 2. ChatML (Qwen2.5)
        let qwen_tmpl = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
        let q_bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "gpt2",
            vocab: TEST_VOCAB,
            token_types: &types,
            bos_token: None,
            eos_token: Some(5),
            eot_token: None,
            add_bos: Some(false),
            chat_template: Some(qwen_tmpl),
        });
        let q_gguf = GgufFile::parse(&q_bytes).expect("parse");
        let q_tok = GgufTokenizer::from_gguf(&q_gguf).expect("tok");
        assert_eq!(
            format_single_turn_prompt(&q_tok, "hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );

        // 3. Phi-3 / Phi-4 with UserDefined token_type=4
        let mut phi_types = vec![1_i32; TEST_VOCAB.len()];
        phi_types[6] = 4;
        phi_types[7] = 4;
        phi_types[8] = 4;
        phi_types[12] = 3;
        let phi_tmpl = "<s>{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}<|assistant|>\n";
        let phi_bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "llama",
            vocab: TEST_VOCAB,
            token_types: &phi_types,
            bos_token: Some(12),
            eos_token: Some(13),
            eot_token: Some(8),
            add_bos: Some(true),
            chat_template: Some(phi_tmpl),
        });
        let phi_gguf = GgufFile::parse(&phi_bytes).expect("parse");
        let phi_tok = GgufTokenizer::from_gguf(&phi_gguf).expect("tok");
        let formatted_phi = format_single_turn_prompt(&phi_tok, "hi");
        assert_eq!(formatted_phi, "<s><|user|>\nhi<|end|>\n<|assistant|>\n");
        let toks_phi =
            encode_formatted_prompt_utf8_safe(&phi_tok, &formatted_phi, true).expect("enc");
        assert_eq!(toks_phi.iter().filter(|&&id| id == 12).count(), 1);
        assert!(toks_phi.contains(&6), "must encode UserDefined <|user|>");
        assert!(toks_phi.contains(&8), "must encode UserDefined <|end|>");
        assert!(
            toks_phi.contains(&7),
            "must encode UserDefined <|assistant|>"
        );
    }

    #[test]
    fn test_detect_gemma_mistral_deepseek_custom_and_base_passthrough() {
        let types = default_test_types();
        assert_eq!(
            detect_chat_template_family("{{ bos_token }}<start_of_turn>user\n{{ m }}<end_of_turn>"),
            ChatTemplateFamily::Gemma
        );
        assert_eq!(
            detect_chat_template_family("{{ bos_token }}[INST] {{ m }} [/INST]"),
            ChatTemplateFamily::Mistral
        );
        assert_eq!(
            detect_chat_template_family("<｜begin▁of▁sentence｜><｜User｜>{{ m }}<｜Assistant｜>"),
            ChatTemplateFamily::DeepSeek
        );
        let custom_bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "gpt2",
            vocab: TEST_VOCAB,
            token_types: &types,
            bos_token: Some(12),
            eos_token: Some(13),
            eot_token: None,
            add_bos: Some(false),
            chat_template: Some("Ask: {{prompt}}\nAns:"),
        });
        let c_gguf = GgufFile::parse(&custom_bytes).expect("parse");
        let c_tok = GgufTokenizer::from_gguf(&c_gguf).expect("tok");
        assert_eq!(format_single_turn_prompt(&c_tok, "hi"), "Ask: hi\nAns:");

        let base_bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "gpt2",
            vocab: TEST_VOCAB,
            token_types: &types,
            bos_token: Some(12),
            eos_token: Some(13),
            eot_token: None,
            add_bos: Some(true),
            chat_template: None,
        });
        let b_gguf = GgufFile::parse(&base_bytes).expect("parse");
        let b_tok = GgufTokenizer::from_gguf(&b_gguf).expect("tok");
        assert_eq!(format_single_turn_prompt(&b_tok, "hi"), "hi");
    }

    #[test]
    fn test_discover_stop_tokens_and_detokenize_excludes_stop_tokens() {
        let vocab = [
            "<s>",
            "</s>",
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<end_of_turn>",
            "<｜end▁of▁sentence｜>",
            "O",
            "K",
        ];
        let mut types = vec![4_i32; vocab.len()];
        types[0] = 3;
        types[1] = 3;
        types[7] = 1;
        types[8] = 1;
        let bytes = build_tokenizer_gguf(&TokGgufSpec {
            model_kind: "gpt2",
            vocab: &vocab,
            token_types: &types,
            bos_token: Some(0),
            eos_token: Some(1),
            eot_token: Some(4),
            add_bos: None,
            chat_template: None,
        });
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let tok = GgufTokenizer::from_gguf(&gguf).expect("tok");

        let stop_ids = discover_stop_token_ids(&tok);
        for id in [1_u32, 2, 3, 4, 5, 6] {
            assert!(stop_ids.contains(&id), "missing stop token id {id}");
        }

        let clean = detokenize_excluding_stop_tokens(&tok, &[7, 8, 5], &[]).expect("detok");
        assert_eq!(clean, "OK");
    }
}
