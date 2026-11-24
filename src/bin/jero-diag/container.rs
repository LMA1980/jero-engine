//! Container-agnostic tokenizer hydration for `GGUF` and `SafeTensors` checkpoints in `jero-diag`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use jero_core::{GgufFile, K3Error};
use jero_engine::{
    GenerationConfig, LockedModelWeights, ParsedModelContainer, detokenize_excluding_stop_tokens,
    discover_stop_token_ids, encode_text_utf8_safe, format_single_turn_prompt,
    tokenize_prompt_utf8_safe,
};
use jero_tokenizer::{
    DetokenizeStep, GgufTokenizer, SpecialTokens, StreamingDetokenizer, Tokenizer,
};
use serde_json::Value;

/// Unified tokenizer wrapper supporting both embedded `GGUF` tokenizers and Hugging Face
/// `tokenizer.json` / `SafeTensors` metadata vocabularies.
#[derive(Debug, Clone)]
pub enum DiagTokenizer {
    /// Embedded GGUF BPE/SentencePiece/WordPiece tokenizer.
    Gguf(Box<GgufTokenizer>),
    /// Hugging Face `tokenizer.json` or `SafeTensors` metadata vocabulary tokenizer.
    Hf(HfJsonTokenizer),
}

/// Hugging Face JSON / `SafeTensors` vocabulary tokenizer.
#[derive(Debug, Clone)]
pub struct HfJsonTokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    special_tokens: SpecialTokens,
    chat_template: Option<String>,
}

impl HfJsonTokenizer {
    fn from_vocab_map(
        mut id_map: BTreeMap<u32, String>,
        special_tokens: SpecialTokens,
        chat_template: Option<String>,
    ) -> Result<Self, K3Error> {
        if id_map.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "tokenizer".to_owned(),
                reason: "tokenizer vocabulary cannot be empty".to_owned(),
            });
        }
        let max_id = *id_map.keys().max().unwrap_or(&0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        let mut token_to_id = HashMap::with_capacity(id_map.len());
        for (id, tok) in std::mem::take(&mut id_map) {
            if let Some(slot) = id_to_token.get_mut(id as usize) {
                *slot = tok.clone();
            }
            token_to_id.insert(tok, id);
        }
        Ok(Self {
            id_to_token,
            token_to_id,
            special_tokens,
            chat_template,
        })
    }

    fn from_tokenizer_json_str(raw_json: &str, config_json: Option<&str>) -> Result<Self, K3Error> {
        let root: Value =
            serde_json::from_str(raw_json).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "tokenizer".to_owned(),
                reason: format!("malformed tokenizer.json: {e}"),
            })?;

        let mut id_map = BTreeMap::new();
        if let Some(vocab) = root
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(Value::as_object)
        {
            for (tok, id_val) in vocab {
                if let Some(id_u64) = id_val.as_u64()
                    && let Ok(id) = u32::try_from(id_u64)
                {
                    id_map.insert(id, tok.clone());
                }
            }
        }
        if let Some(added) = root.get("added_tokens").and_then(Value::as_array) {
            for entry in added {
                if let (Some(id_u64), Some(content)) = (
                    entry.get("id").and_then(Value::as_u64),
                    entry.get("content").and_then(Value::as_str),
                ) && let Ok(id) = u32::try_from(id_u64)
                {
                    id_map.insert(id, content.to_owned());
                }
            }
        }

        let mut special = SpecialTokens::default();
        let mut chat_template = None;
        if let Some(cfg_str) = config_json
            && let Ok(cfg) = serde_json::from_str::<Value>(cfg_str)
        {
            let resolve_token_str = |key: &str| -> Option<String> {
                match cfg.get(key) {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Object(obj)) => obj
                        .get("content")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    _ => None,
                }
            };
            if let Some(bos_s) = resolve_token_str("bos_token") {
                special.bos_token_id = id_map
                    .iter()
                    .find_map(|(&id, tok)| (*tok == bos_s).then_some(id));
            }
            if let Some(eos_s) = resolve_token_str("eos_token") {
                special.eos_token_id = id_map
                    .iter()
                    .find_map(|(&id, tok)| (*tok == eos_s).then_some(id));
            }
            if let Some(pad_s) = resolve_token_str("pad_token") {
                special.padding_token_id = id_map
                    .iter()
                    .find_map(|(&id, tok)| (*tok == pad_s).then_some(id));
            }
            if let Some(add_bos) = cfg.get("add_bos_token").and_then(Value::as_bool) {
                special.add_bos_token = add_bos;
            }
            if let Some(tmpl) = cfg.get("chat_template").and_then(Value::as_str) {
                chat_template = Some(tmpl.to_owned());
            }
        }

        if special.bos_token_id.is_none() {
            for cand in ["<s>", "<bos>", "<|begin_of_text|>"] {
                if let Some((&id, _)) = id_map.iter().find(|(_, t)| t.as_str() == cand) {
                    special.bos_token_id = Some(id);
                    break;
                }
            }
        }
        if special.eos_token_id.is_none() {
            for cand in ["</s>", "<eos>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"] {
                if let Some((&id, _)) = id_map.iter().find(|(_, t)| t.as_str() == cand) {
                    special.eos_token_id = Some(id);
                    break;
                }
            }
        }
        for eot in ["<|eot_id|>", "<|im_end|>", "<end_of_turn>"] {
            if let Some((&id, _)) = id_map.iter().find(|(_, t)| t.as_str() == eot) {
                special.eot_token_id = Some(id);
                break;
            }
        }

        Self::from_vocab_map(id_map, special, chat_template)
    }

    fn encode_greedy(&self, text: &str, include_bos: bool) -> Result<Vec<u32>, K3Error> {
        let mut out = Vec::new();
        if include_bos
            && self.special_tokens.add_bos_token
            && let Some(bos) = self.special_tokens.bos_token_id
        {
            out.push(bos);
        }

        let mut cursor = 0_usize;
        while cursor < text.len() {
            let rem = &text[cursor..];
            let mut matched = None;
            let max_end = rem.len().min(64);
            for end in (1..=max_end).rev() {
                if !rem.is_char_boundary(end) {
                    continue;
                }
                if let Some(&id) = self.token_to_id.get(&rem[..end]) {
                    matched = Some((end, id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                out.push(id);
                cursor += len;
                continue;
            }

            let ch = rem.chars().next().unwrap_or('\0');
            let char_len = ch.len_utf8().max(1);
            if ch == ' ' {
                let sp = "\u{2581}";
                let gp = "\u{0120}";
                if let Some(&id) = self
                    .token_to_id
                    .get(sp)
                    .or_else(|| self.token_to_id.get(gp))
                {
                    out.push(id);
                    cursor += char_len;
                    continue;
                }
            }

            for &b in &rem.as_bytes()[..char_len] {
                let upper = format!("<0x{b:02X}>");
                let lower = format!("<0x{b:02x}>");
                if let Some(&id) = self
                    .token_to_id
                    .get(&upper)
                    .or_else(|| self.token_to_id.get(&lower))
                    .or(self.special_tokens.padding_token_id.as_ref())
                {
                    out.push(id);
                } else {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "tokenizer".to_owned(),
                        reason: format!("unable to encode character '{ch}' (byte 0x{b:02X})"),
                    });
                }
            }
            cursor += char_len;
        }
        Ok(out)
    }
}

impl Tokenizer for HfJsonTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, K3Error> {
        self.encode_greedy(text, true)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, K3Error> {
        let mut out = String::new();
        for &tok in tokens {
            if let Some(piece) = self.decode_piece(tok)? {
                out.push_str(&piece);
            }
        }
        Ok(out)
    }

    fn decode_piece(&self, token_id: u32) -> Result<Option<String>, K3Error> {
        if self.special_tokens.is_stop(token_id) {
            return Ok(None);
        }
        let raw = self
            .id_to_token
            .get(token_id as usize)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "tokenizer".to_owned(),
                reason: format!("unknown token ID {token_id} in SafeTensors vocabulary"),
            })?;
        Ok(Some(raw.replace(['\u{2581}', '\u{0120}'], " ")))
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

/// Loads a [`DiagTokenizer`] from either an embedded `GGUF` tokenizer or a `SafeTensors`
/// checkpoint (`tokenizer.gguf`, `tokenizer.json`, or `SafeTensors` `__metadata__`).
pub fn load_diag_tokenizer(
    model_path: &Path,
    weights: &LockedModelWeights,
) -> Result<DiagTokenizer, K3Error> {
    match weights.parse_container()? {
        ParsedModelContainer::Gguf(gguf) => Ok(DiagTokenizer::Gguf(Box::new(
            GgufTokenizer::from_gguf(&gguf)?,
        ))),
        ParsedModelContainer::SafeTensors(st) => {
            if let Some(parent) = model_path.parent() {
                let gguf_tok_path = parent.join("tokenizer.gguf");
                if gguf_tok_path.is_file()
                    && let Ok(bytes) = std::fs::read(&gguf_tok_path)
                    && let Ok(gguf_tok) = GgufFile::parse(&bytes)
                    && let Ok(tok) = GgufTokenizer::from_gguf(&gguf_tok)
                {
                    return Ok(DiagTokenizer::Gguf(Box::new(tok)));
                }
                let hf_json_path = parent.join("tokenizer.json");
                if hf_json_path.is_file() {
                    let raw = std::fs::read_to_string(&hf_json_path).map_err(|e| {
                        K3Error::CapabilityMismatch {
                            architecture: st.architecture().to_owned(),
                            reason: format!(
                                "failed to read sibling '{}': {e}",
                                hf_json_path.display()
                            ),
                        }
                    })?;
                    let cfg_str =
                        std::fs::read_to_string(parent.join("tokenizer_config.json")).ok();
                    return HfJsonTokenizer::from_tokenizer_json_str(&raw, cfg_str.as_deref())
                        .map(DiagTokenizer::Hf);
                }
            }

            if let Some(tokens_json) = st.metadata_value("tokenizer.ggml.tokens")
                && let Ok(tokens) = serde_json::from_str::<Vec<String>>(tokens_json)
                && !tokens.is_empty()
            {
                let mut id_map = BTreeMap::new();
                for (idx, tok) in tokens.into_iter().enumerate() {
                    if let Ok(id) = u32::try_from(idx) {
                        id_map.insert(id, tok);
                    }
                }
                let mut special = SpecialTokens::default();
                let parse_u32 = |key: &str| {
                    st.metadata_value(key)
                        .and_then(|s| s.trim().parse::<u32>().ok())
                };
                special.bos_token_id = parse_u32("tokenizer.ggml.bos_token_id");
                special.eos_token_id =
                    parse_u32("tokenizer.ggml.eos_token_id").or_else(|| parse_u32("eos_token_id"));
                special.eot_token_id = parse_u32("tokenizer.ggml.eot_token_id");
                special.padding_token_id = parse_u32("tokenizer.ggml.padding_token_id");
                if let Some(add_bos) = st
                    .metadata_value("tokenizer.ggml.add_bos_token")
                    .and_then(|v| v.trim().parse::<bool>().ok())
                {
                    special.add_bos_token = add_bos;
                }
                let chat_tmpl = st
                    .metadata_value("tokenizer.chat_template")
                    .map(str::to_owned);
                return HfJsonTokenizer::from_vocab_map(id_map, special, chat_tmpl)
                    .map(DiagTokenizer::Hf);
            }

            Err(K3Error::CapabilityMismatch {
                architecture: st.architecture().to_owned(),
                reason: format!(
                    "SafeTensors checkpoint '{}' requires a sibling 'tokenizer.json' or 'tokenizer.gguf' (or 'tokenizer.ggml.tokens' metadata)",
                    model_path.display()
                ),
            })
        }
    }
}

/// Streaming detokenizer abstraction over GGUF and `HfJsonTokenizer` vocabularies.
#[derive(Debug)]
pub enum DiagDetokenizer<'a> {
    /// Native GGUF streaming detokenizer.
    Gguf(StreamingDetokenizer<'a>),
    /// Plain Hugging Face JSON streaming detokenizer.
    Hf {
        tok: &'a HfJsonTokenizer,
        stopped: bool,
    },
}

impl DiagDetokenizer<'_> {
    /// Consumes a generated token ID and emits printable text or stops at special tokens.
    pub fn push_token(&mut self, token_id: u32) -> Result<DetokenizeStep, K3Error> {
        match self {
            Self::Gguf(inner) => inner.push_token(token_id),
            Self::Hf { tok, stopped } => {
                if tok.special_tokens.is_stop(token_id) {
                    *stopped = true;
                    return Ok(DetokenizeStep::Stop(String::new()));
                }
                match tok.decode_piece(token_id)? {
                    Some(piece) => Ok(DetokenizeStep::Emit(piece)),
                    None => {
                        *stopped = true;
                        Ok(DetokenizeStep::Stop(String::new()))
                    }
                }
            }
        }
    }

    /// Flushes any buffered UTF-8 prefix at end of generation.
    pub fn finish(&mut self) -> String {
        match self {
            Self::Gguf(inner) => inner.finish(),
            Self::Hf { .. } => String::new(),
        }
    }

    /// Returns `true` once a stop token has been encountered.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        match self {
            Self::Gguf(inner) => inner.is_stopped(),
            Self::Hf { stopped, .. } => *stopped,
        }
    }
}

impl DiagTokenizer {
    /// Returns vocabulary size in tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Gguf(tok) => tok.vocab_size(),
            Self::Hf(tok) => tok.id_to_token.len(),
        }
    }

    /// Returns special token descriptors.
    #[must_use]
    pub fn special_tokens(&self) -> &SpecialTokens {
        match self {
            Self::Gguf(tok) => tok.special_tokens(),
            Self::Hf(tok) => &tok.special_tokens,
        }
    }

    /// Discovers stop token IDs for the loaded vocabulary.
    #[must_use]
    pub fn stop_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Gguf(tok) => discover_stop_token_ids(tok),
            Self::Hf(tok) => {
                let mut ids = Vec::new();
                for opt in [
                    tok.special_tokens.eos_token_id,
                    tok.special_tokens.eot_token_id,
                ] {
                    if let Some(id) = opt
                        && !ids.contains(&id)
                    {
                        ids.push(id);
                    }
                }
                ids
            }
        }
    }

    /// Builds a [`GenerationConfig`] aligned with the loaded tokenizer.
    #[must_use]
    pub fn generation_config(&self, ignore_eos: bool) -> GenerationConfig {
        match self {
            Self::Gguf(tok) => GenerationConfig::from_tokenizer(tok).with_ignore_eos(ignore_eos),
            Self::Hf(_) => GenerationConfig::new()
                .with_stop_token_ids(self.stop_token_ids())
                .with_ignore_eos(ignore_eos),
        }
    }

    /// Tokenizes a prompt using family chat templates when requested and bounds-checks against `n_ctx`.
    pub fn tokenize_prompt(
        &self,
        prompt: &str,
        apply_chat_template: bool,
        n_ctx: usize,
    ) -> Result<Vec<u32>, K3Error> {
        match self {
            Self::Gguf(tok) => tokenize_prompt_utf8_safe(tok, prompt, apply_chat_template, n_ctx),
            Self::Hf(tok) => {
                let tokens = if apply_chat_template {
                    let formatted = self.format_single_turn_prompt(prompt);
                    self.encode_formatted_prompt(&formatted, true)?
                } else {
                    tok.encode_greedy(prompt, true)?
                };
                if n_ctx == 0 || tokens.len() >= n_ctx {
                    return Err(K3Error::ContextLengthExceeded {
                        requested_tokens: tokens.len(),
                        max_context_tokens: n_ctx,
                    });
                }
                Ok(tokens)
            }
        }
    }

    /// Encodes a pre-formatted prompt string with optional BOS deduplication.
    pub fn encode_formatted_prompt(
        &self,
        text: &str,
        prepend_bos_once: bool,
    ) -> Result<Vec<u32>, K3Error> {
        match self {
            Self::Gguf(tok) => {
                jero_engine::encode_formatted_prompt_utf8_safe(tok, text, prepend_bos_once)
            }
            Self::Hf(tok) => {
                let raw = tok.encode_greedy(text, false)?;
                if prepend_bos_once
                    && tok.special_tokens.add_bos_token
                    && let Some(bos) = tok.special_tokens.bos_token_id
                    && raw.first() != Some(&bos)
                {
                    let mut out = Vec::with_capacity(raw.len() + 1);
                    out.push(bos);
                    out.extend_from_slice(&raw);
                    return Ok(out);
                }
                Ok(raw)
            }
        }
    }

    /// Encodes unformatted text cleanly.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, K3Error> {
        match self {
            Self::Gguf(tok) => encode_text_utf8_safe(tok, text),
            Self::Hf(tok) => tok.encode_greedy(text, true),
        }
    }

    /// Decodes a sequence of tokens excluding stop tokens.
    pub fn decode_excluding_stops(
        &self,
        tokens: &[u32],
        stop_ids: &[u32],
    ) -> Result<String, K3Error> {
        match self {
            Self::Gguf(tok) => detokenize_excluding_stop_tokens(tok, tokens, stop_ids),
            Self::Hf(tok) => {
                let filtered: Vec<u32> = tokens
                    .iter()
                    .copied()
                    .take_while(|id| !stop_ids.contains(id) && !tok.special_tokens.is_stop(*id))
                    .collect();
                tok.decode(&filtered)
            }
        }
    }

    /// Decodes a sequence of tokens via [`Tokenizer::decode`].
    pub fn decode(&self, tokens: &[u32]) -> Result<String, K3Error> {
        match self {
            Self::Gguf(tok) => tok.decode(tokens),
            Self::Hf(tok) => tok.decode(tokens),
        }
    }

    /// Decodes a single token piece.
    pub fn decode_piece(&self, token_id: u32) -> Result<Option<String>, K3Error> {
        match self {
            Self::Gguf(tok) => tok.decode_piece(token_id),
            Self::Hf(tok) => tok.decode_piece(token_id),
        }
    }

    /// Returns raw token string for `token_id`, if present.
    #[must_use]
    pub fn id_to_token(&self, token_id: u32) -> Option<&str> {
        match self {
            Self::Gguf(tok) => tok.id_to_token(token_id),
            Self::Hf(tok) => tok
                .id_to_token
                .get(token_id as usize)
                .filter(|s| !s.is_empty())
                .map(String::as_str),
        }
    }

    /// Returns embedded raw chat template if present.
    #[must_use]
    pub fn raw_chat_template(&self) -> Option<&str> {
        match self {
            Self::Gguf(tok) => tok.chat_template().raw_template(),
            Self::Hf(tok) => tok.chat_template.as_deref(),
        }
    }

    /// Formats a single-turn user prompt.
    #[must_use]
    pub fn format_single_turn_prompt(&self, prompt: &str) -> String {
        match self {
            Self::Gguf(tok) => format_single_turn_prompt(tok, prompt),
            Self::Hf(tok) => {
                if let Some(tmpl) = tok.chat_template.as_deref() {
                    if tmpl.contains("<|im_start|>") {
                        return format!(
                            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        );
                    }
                    if tmpl.contains("<start_of_turn>") {
                        return format!(
                            "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                        );
                    }
                    if tmpl.contains("[INST]") {
                        return format!("[INST] {prompt} [/INST]");
                    }
                }
                prompt.to_owned()
            }
        }
    }

    /// Creates a streaming detokenizer.
    #[must_use]
    pub fn streaming_detokenizer(&self) -> DiagDetokenizer<'_> {
        match self {
            Self::Gguf(tok) => DiagDetokenizer::Gguf(tok.streaming_detokenizer()),
            Self::Hf(tok) => DiagDetokenizer::Hf {
                tok,
                stopped: false,
            },
        }
    }
}
