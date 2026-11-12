//! Architecture integration parity verification and anti-stub certification primitives (ADR-045).
//!
//! Provides deterministic prefill top-$k$ logit extraction, greedy ($T = 0$) token sequence
//! snapshotting, golden reference fixture verification, and intrinsic anti-stub checks that reject:
//! - Prompt-insensitive recipes ignoring `token_embd.weight` or `prompt_tokens`
//! - Non-finite (`NaN` / `±Inf`) or degenerate flat/zero logit distributions
//! - Collapsed constant-token repetition loops

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights};
use jero_tokenizer::{GgufTokenizer, TokenType, Tokenizer};
use serde::{Deserialize, Serialize};

use crate::affinity::{
    ClusterPhasePolicy, WorkerPool, X100_CLUSTER_MASK, select_phase_policy_for_arch,
};
use crate::engine::{EnginePhase, StreamAction, infer_logits_buffer_len, sample_greedy_argmax};
use crate::kv::SingleSlotKvArena;

/// Default number of descending prefill logits recorded in a [`ParityDumpReport`].
pub const DEFAULT_PARITY_TOP_K: usize = 16;

/// Single vocabulary token entry in a sorted descending top-$k$ logit snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopKLogitEntry {
    /// Vocabulary token ID.
    pub token_id: u32,
    /// Raw pre-softmax logit value.
    pub logit: f32,
}

/// Deterministic differential logit and greedy-token parity report exported by `jero-diag --dump-parity`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityDumpReport {
    /// Schema version (`"1.0.0"`).
    pub schema_version: String,
    /// Resolved GGUF architecture identifier (e.g., `"llama"`, `"qwen2"`, `"mistral"`, `"phi3"`).
    pub architecture: String,
    /// Primary weight quantization format (e.g., `"Q8_0"`, `"Q4_K_M"`).
    pub quantization_format: String,
    /// Raw operator prompt evaluated.
    pub prompt: String,
    /// Token IDs produced by tokenizing `prompt` prior to prefill.
    pub prompt_token_ids: Vec<u32>,
    /// Sorted descending top-$k$ logits emitted at the end of `prefill_step` (tie-broken by ascending `token_id`).
    pub first_token_top_k: Vec<TopKLogitEntry>,
    /// Sequence of greedy ($T = 0$) token IDs emitted across prefill + decode.
    pub greedy_token_ids: Vec<u32>,
    /// Detokenized text emitted from `greedy_token_ids`.
    pub generated_text: String,
}

/// Extracts the top-$k$ logits from `logits`, sorted by descending `logit` and tie-broken by
/// ascending `token_id`.
///
/// # Errors
///
/// Returns a descriptive error string if `logits` is empty, `k == 0`, or any element in `logits`
/// is non-finite (`NaN` or `±Inf`).
pub fn extract_top_k_logits(logits: &[f32], k: usize) -> Result<Vec<TopKLogitEntry>, String> {
    if logits.is_empty() {
        return Err("cannot extract top-k from an empty logits slice".to_owned());
    }
    if k == 0 {
        return Err("requested top-k count must be at least 1".to_owned());
    }

    let mut indexed = Vec::with_capacity(logits.len());
    for (idx, &val) in logits.iter().enumerate() {
        if !val.is_finite() {
            return Err(format!("non-finite logit {val} detected at token_id {idx}"));
        }
        let token_id =
            u32::try_from(idx).map_err(|_| format!("token index {idx} overflows u32"))?;
        indexed.push(TopKLogitEntry {
            token_id,
            logit: val,
        });
    }

    indexed.sort_by(|a, b| {
        b.logit
            .total_cmp(&a.logit)
            .then_with(|| a.token_id.cmp(&b.token_id))
    });
    indexed.truncate(k.min(logits.len()));
    Ok(indexed)
}

/// Computes cosine similarity in `f64` precision between two equal-length `f32` slices.
///
/// Returns `0.0` if `vec_a` and `vec_b` differ in length, are empty, or have near-zero $`L_2`$ norm.
#[must_use]
pub fn cosine_similarity_f64(vec_a: &[f32], vec_b: &[f32]) -> f64 {
    if vec_a.is_empty() || vec_a.len() != vec_b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&a, &b) in vec_a.iter().zip(vec_b.iter()) {
        let fa = f64::from(a);
        let fb = f64::from(b);
        if !fa.is_finite() || !fb.is_finite() {
            return 0.0;
        }
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= 1.0e-12 { 0.0 } else { dot / denom }
}

/// Computes sample standard deviation ($N - 1$) across a slice of [`TopKLogitEntry`] values.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn logit_sample_stddev(entries: &[TopKLogitEntry]) -> f64 {
    if entries.len() < 2 {
        return 0.0;
    }
    let n = entries.len() as f64;
    let mean = entries.iter().map(|e| f64::from(e.logit)).sum::<f64>() / n;
    let variance = entries
        .iter()
        .map(|e| {
            let diff = f64::from(e.logit) - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1.0);
    variance.max(0.0).sqrt()
}

/// Verifies intrinsic mathematical sanity invariants on a single [`ParityDumpReport`]:
/// 1. Non-empty, finite `first_token_top_k` logits sorted in descending order.
/// 2. Sample standard deviation across `first_token_top_k` meets `min_logit_stddev` (when $\ge 2$ entries).
/// 3. Non-empty `greedy_token_ids` without degenerate single-token collapse across $\ge 6$ steps.
///
/// # Errors
///
/// Returns a descriptive error string when any intrinsic forward-pass invariant is violated.
pub fn verify_intrinsic_parity(
    dump: &ParityDumpReport,
    min_logit_stddev: f64,
) -> Result<(), String> {
    if dump.first_token_top_k.is_empty() {
        return Err("first_token_top_k cannot be empty".to_owned());
    }
    for (idx, entry) in dump.first_token_top_k.iter().enumerate() {
        if !entry.logit.is_finite() {
            return Err(format!(
                "non-finite logit {} at top-k rank {idx} (token_id {})",
                entry.logit, entry.token_id
            ));
        }
        if idx > 0 && entry.logit > dump.first_token_top_k[idx - 1].logit + 1.0e-6 {
            return Err(format!(
                "first_token_top_k is not sorted descending at rank {idx} ({} > {})",
                entry.logit,
                dump.first_token_top_k[idx - 1].logit
            ));
        }
    }

    if dump.first_token_top_k.len() >= 2 {
        let stddev = logit_sample_stddev(&dump.first_token_top_k);
        if stddev < min_logit_stddev {
            return Err(format!(
                "top-k logits are degenerate/flat (stddev {stddev:.6} < required {min_logit_stddev:.6})"
            ));
        }
    }

    if dump.greedy_token_ids.is_empty() {
        return Err("greedy_token_ids cannot be empty".to_owned());
    }

    if dump.greedy_token_ids.len() >= 6 {
        let first = dump.greedy_token_ids[0];
        if dump.greedy_token_ids.iter().all(|&tok| tok == first) {
            return Err(format!(
                "greedy generation collapsed into repeated single token ID {first} across {} steps",
                dump.greedy_token_ids.len()
            ));
        }
    }

    Ok(())
}

/// Verifies that two [`ParityDumpReport`] snapshots produced from distinct prompts exhibit genuine
/// input sensitivity, catching stubbed `ArchRecipe` implementations that ignore `token_embd.weight`
/// or `prompt_tokens`.
///
/// # Errors
///
/// Returns a descriptive error string if the prompts are identical, either snapshot fails intrinsic
/// sanity checks, or both prompts produce identical top-$k$ logit distributions.
pub fn verify_prompt_sensitivity(
    dump_a: &ParityDumpReport,
    dump_b: &ParityDumpReport,
) -> Result<(), String> {
    if dump_a.prompt == dump_b.prompt && dump_a.prompt_token_ids == dump_b.prompt_token_ids {
        return Err(
            "verify_prompt_sensitivity requires two distinct prompts/token sequences".to_owned(),
        );
    }

    verify_intrinsic_parity(dump_a, 1.0e-4)
        .map_err(|e| format!("prompt A intrinsic check failed: {e}"))?;
    verify_intrinsic_parity(dump_b, 1.0e-4)
        .map_err(|e| format!("prompt B intrinsic check failed: {e}"))?;

    let same_len = dump_a.first_token_top_k.len() == dump_b.first_token_top_k.len();
    let indistinguishable = same_len
        && dump_a
            .first_token_top_k
            .iter()
            .zip(dump_b.first_token_top_k.iter())
            .all(|(a, b)| a.token_id == b.token_id && (a.logit - b.logit).abs() <= 1.0e-5);

    if indistinguishable {
        return Err(
            "distinct prompts produced identical first_token_top_k logits (recipe ignores token_embd.weight / prompt_tokens)".to_owned(),
        );
    }

    Ok(())
}

/// Compares an evaluated [`ParityDumpReport`] against a pinned golden reference fixture.
///
/// Enforces exact architecture, prompt token IDs, top-1 greedy argmax, full greedy token sequence,
/// top-$k$ logit cosine similarity $\ge$ `min_cosine_sim`, and element-wise logit error $\le$
/// `max_abs_logit_err`.
///
/// # Errors
///
/// Returns a descriptive diagnostic string detailing the exact parity divergence if any check fails.
pub fn verify_parity_against_golden(
    actual: &ParityDumpReport,
    golden: &ParityDumpReport,
    min_cosine_sim: f64,
    max_abs_logit_err: f32,
) -> Result<(), String> {
    verify_intrinsic_parity(actual, 1.0e-4)
        .map_err(|e| format!("actual dump failed intrinsic check: {e}"))?;

    if actual.architecture != golden.architecture {
        return Err(format!(
            "architecture mismatch: actual '{}' != golden '{}'",
            actual.architecture, golden.architecture
        ));
    }

    if actual.prompt_token_ids != golden.prompt_token_ids {
        return Err(format!(
            "prompt_token_ids mismatch: actual {:?} != golden {:?}",
            actual.prompt_token_ids, golden.prompt_token_ids
        ));
    }

    if actual.first_token_top_k.len() != golden.first_token_top_k.len() {
        return Err(format!(
            "first_token_top_k length mismatch: actual {} != golden {}",
            actual.first_token_top_k.len(),
            golden.first_token_top_k.len()
        ));
    }

    if actual.first_token_top_k[0].token_id != golden.first_token_top_k[0].token_id {
        return Err(format!(
            "first-token greedy argmax mismatch: actual token_id {} (logit {:.6}) != golden token_id {} (logit {:.6})",
            actual.first_token_top_k[0].token_id,
            actual.first_token_top_k[0].logit,
            golden.first_token_top_k[0].token_id,
            golden.first_token_top_k[0].logit
        ));
    }

    let actual_logits: Vec<f32> = actual.first_token_top_k.iter().map(|e| e.logit).collect();
    let golden_logits: Vec<f32> = golden.first_token_top_k.iter().map(|e| e.logit).collect();
    let cos_sim = cosine_similarity_f64(&actual_logits, &golden_logits);
    if cos_sim < min_cosine_sim {
        return Err(format!(
            "top-k logit cosine similarity {cos_sim:.6} below required floor {min_cosine_sim:.6}"
        ));
    }

    for (rank, (act, gld)) in actual
        .first_token_top_k
        .iter()
        .zip(golden.first_token_top_k.iter())
        .enumerate()
    {
        if act.token_id != gld.token_id {
            return Err(format!(
                "top-k token_id ordering mismatch at rank {rank}: actual {} != golden {}",
                act.token_id, gld.token_id
            ));
        }
        let abs_err = (act.logit - gld.logit).abs();
        if abs_err > max_abs_logit_err {
            return Err(format!(
                "logit error {abs_err:.6} exceeds max_abs_logit_err {max_abs_logit_err:.6} at rank {rank} (token_id {})",
                act.token_id
            ));
        }
    }

    if actual.greedy_token_ids != golden.greedy_token_ids {
        return Err(format!(
            "greedy_token_ids mismatch: actual {:?} != golden {:?}",
            actual.greedy_token_ids, golden.greedy_token_ids
        ));
    }

    Ok(())
}

/// Executes an isolated prefill + greedy ($T = 0$) decode pass and records a [`ParityDumpReport`]
/// containing the exact prefill `first_token_top_k` logits and full greedy token trajectory.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `prompt_tokens` or `vocab_size` is empty, or if
///   any prefill/decode logit is non-finite.
/// - Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > kv_arena.n_ctx()`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn evaluate_parity_snapshot<F>(
    recipe: &dyn ArchRecipe,
    weights: &dyn ModelWeights,
    kv_arena: &mut SingleSlotKvArena,
    worker_pool: Option<&WorkerPool>,
    phase_policy: ClusterPhasePolicy,
    prompt: &str,
    prompt_tokens: &[u32],
    vocab_size: usize,
    max_tokens: usize,
    top_k: usize,
    quantization_format: &str,
    mut on_token: F,
) -> Result<(ParityDumpReport, String), K3Error>
where
    F: FnMut(u32, &mut String) -> Result<StreamAction, K3Error>,
{
    if prompt_tokens.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: recipe.architecture_id().to_owned(),
            reason: "prompt token sequence cannot be empty".to_owned(),
        });
    }
    if vocab_size == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: recipe.architecture_id().to_owned(),
            reason: "vocabulary size must be greater than zero".to_owned(),
        });
    }

    let n_ctx = kv_arena.n_ctx();
    let logits_len = infer_logits_buffer_len(weights, vocab_size);
    let mut logits = vec![0.0_f32; logits_len];
    let mut ctx =
        ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, kv_arena.as_mut_slice())?;

    let effective_policy = match phase_policy {
        ClusterPhasePolicy::Auto => worker_pool
            .and_then(WorkerPool::phase_policy)
            .unwrap_or_else(|| {
                let online_mask =
                    worker_pool.map_or(X100_CLUSTER_MASK, |p| p.affinity().effective_mask);
                select_phase_policy_for_arch(
                    recipe.architecture_id(),
                    prompt_tokens.len(),
                    online_mask,
                )
            }),
        explicit => explicit,
    };

    if let Some(pool) = worker_pool {
        pool.install_for_phase(EnginePhase::Prefill, effective_policy, || {
            recipe.prefill_step(&mut ctx, weights)
        })?;
    } else {
        recipe.prefill_step(&mut ctx, weights)?;
    }

    let sample_len = vocab_size.min(ctx.logits().len());
    let first_token_top_k = extract_top_k_logits(&ctx.logits()[..sample_len], top_k.max(1))
        .map_err(|reason| K3Error::CapabilityMismatch {
            architecture: recipe.architecture_id().to_owned(),
            reason,
        })?;

    let mut greedy_token_ids = Vec::with_capacity(max_tokens);
    let mut generated_text = String::new();

    if max_tokens > 0 {
        let mut current_token =
            sample_greedy_argmax(&ctx.logits()[..sample_len]).ok_or_else(|| {
                K3Error::CapabilityMismatch {
                    architecture: recipe.architecture_id().to_owned(),
                    reason: "empty logits buffer after prefill step".to_owned(),
                }
            })?;
        greedy_token_ids.push(current_token);

        if on_token(current_token, &mut generated_text)? == StreamAction::Continue {
            while greedy_token_ids.len() < max_tokens && ctx.position() < n_ctx {
                ctx.prepare_decode(current_token)?;
                if let Some(pool) = worker_pool {
                    pool.install_for_phase(EnginePhase::Decode, effective_policy, || {
                        recipe.decode_step(&mut ctx, weights)
                    })?;
                } else {
                    recipe.decode_step(&mut ctx, weights)?;
                }

                let step_len = vocab_size.min(ctx.logits().len());
                // Validate finiteness on every decode step as well
                if let Some((bad_idx, bad_val)) = ctx.logits()[..step_len]
                    .iter()
                    .enumerate()
                    .find(|(_, v)| !v.is_finite())
                {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: recipe.architecture_id().to_owned(),
                        reason: format!(
                            "non-finite decode logit {bad_val} at token_id {bad_idx} (step {})",
                            greedy_token_ids.len()
                        ),
                    });
                }

                current_token =
                    sample_greedy_argmax(&ctx.logits()[..step_len]).ok_or_else(|| {
                        K3Error::CapabilityMismatch {
                            architecture: recipe.architecture_id().to_owned(),
                            reason: "empty logits buffer after decode step".to_owned(),
                        }
                    })?;
                greedy_token_ids.push(current_token);
                if on_token(current_token, &mut generated_text)? == StreamAction::Stop {
                    break;
                }
            }
        }
    }

    let report = ParityDumpReport {
        schema_version: "1.0.0".to_owned(),
        architecture: recipe.architecture_id().to_owned(),
        quantization_format: quantization_format.to_owned(),
        prompt: prompt.to_owned(),
        prompt_token_ids: prompt_tokens.to_vec(),
        first_token_top_k,
        greedy_token_ids,
        generated_text: generated_text.clone(),
    };

    Ok((report, generated_text))
}

fn encode_chunk_without_bos(
    tokenizer: &GgufTokenizer,
    chunk: &str,
    out: &mut Vec<u32>,
) -> Result<(), K3Error> {
    if chunk.is_empty() {
        return Ok(());
    }
    let mut sub = tokenizer.encode(chunk)?;
    if tokenizer.special_tokens().add_bos_token
        && let Some(bos) = tokenizer.special_tokens().bos_token_id
        && sub.first() == Some(&bos)
    {
        sub.remove(0);
    }
    out.extend_from_slice(&sub);
    Ok(())
}

fn longest_vocab_prefix_len(
    tokenizer: &GgufTokenizer,
    text: &str,
    require_special: bool,
) -> Option<usize> {
    let mut best = None;
    for (char_idx, ch) in text.char_indices() {
        let end = char_idx + ch.len_utf8();
        if end > 64 {
            break;
        }
        if let Some(id) = tokenizer.token_to_id(&text[..end]) {
            if require_special {
                let is_special = usize::try_from(id)
                    .ok()
                    .and_then(|i| tokenizer.token_types().get(i))
                    .is_some_and(|t| matches!(t, TokenType::Control | TokenType::UserDefined));
                if !is_special {
                    continue;
                }
            }
            best = Some(end);
        }
    }
    best
}

/// Encodes `text` with `tokenizer` while preserving UTF-8 character boundaries across unmapped
/// multi-byte scalars (`'’'`, `'—'`, `'€'`, `'🚀'`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if an unmapped byte has neither a `<0xXX>` byte token
/// nor a fallback `padding_token_id`.
pub fn encode_text_utf8_safe(tokenizer: &GgufTokenizer, text: &str) -> Result<Vec<u32>, K3Error> {
    if text.is_ascii() {
        return tokenizer.encode(text);
    }

    let mut out = Vec::new();
    if tokenizer.special_tokens().add_bos_token
        && let Some(bos) = tokenizer.special_tokens().bos_token_id
    {
        out.push(bos);
    }

    let mut run_start = 0_usize;
    let mut cursor = 0_usize;
    while cursor < text.len() {
        let remaining = &text[cursor..];
        if remaining.starts_with('<')
            && let Some(special_len) = longest_vocab_prefix_len(tokenizer, remaining, true)
        {
            cursor += special_len;
            continue;
        }

        let ch = remaining.chars().next().unwrap_or('\0');
        if ch.is_ascii() {
            cursor += 1;
            continue;
        }

        if let Some(matched_len) = longest_vocab_prefix_len(tokenizer, remaining, false) {
            cursor += matched_len;
            continue;
        }

        encode_chunk_without_bos(tokenizer, &text[run_start..cursor], &mut out)?;
        let char_len = ch.len_utf8().min(remaining.len()).max(1);
        for &b in &remaining.as_bytes()[..char_len] {
            let hex_upper = format!("<0x{b:02X}>");
            let hex_lower = format!("<0x{b:02x}>");
            if let Some(byte_id) = tokenizer
                .token_to_id(&hex_upper)
                .or_else(|| tokenizer.token_to_id(&hex_lower))
            {
                out.push(byte_id);
            } else if let Some(pad_id) = tokenizer.special_tokens().padding_token_id {
                out.push(pad_id);
            } else {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "tokenizer".to_owned(),
                    reason: format!("unable to encode character '{ch}' (byte 0x{b:02X})"),
                });
            }
        }
        cursor += char_len;
        run_start = cursor;
    }

    encode_chunk_without_bos(tokenizer, &text[run_start..], &mut out)?;
    Ok(out)
}

/// Tokenizes `prompt` (with optional `ChatTemplate` formatting) using [`encode_text_utf8_safe`] and
/// enforces `tokens.len() < n_ctx`.
///
/// # Errors
///
/// Returns [`K3Error::ContextLengthExceeded`] if `n_ctx == 0` or `tokens.len() >= n_ctx`, or
/// [`K3Error::CapabilityMismatch`] if `prompt` contains unencodable bytes.
pub fn tokenize_prompt_utf8_safe(
    tokenizer: &GgufTokenizer,
    prompt: &str,
    apply_chat_template: bool,
    n_ctx: usize,
) -> Result<Vec<u32>, K3Error> {
    let formatted = if apply_chat_template {
        tokenizer.chat_template().format_prompt(prompt)
    } else {
        prompt.to_owned()
    };
    let tokens = encode_text_utf8_safe(tokenizer, &formatted)?;
    if n_ctx == 0 || tokens.len() >= n_ctx {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: tokens.len(),
            max_context_tokens: n_ctx,
        });
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::{QuantType, TensorView};
    use std::fmt::Write as _;

    #[derive(Debug)]
    struct DummyWeights;

    impl ModelWeights for DummyWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "unused".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    /// Simulates the broken stub pattern where `token_embd.weight` / `prompt_tokens` are ignored.
    #[derive(Debug)]
    struct PromptBlindStubRecipe;

    impl ArchRecipe for PromptBlindStubRecipe {
        fn architecture_id(&self) -> &'static str {
            "stub-blind"
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
            for (idx, slot) in ctx.logits_mut().iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                {
                    *slot = (16_usize.saturating_sub(idx)) as f32 * 0.5;
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
            let len = ctx.logits().len();
            for (idx, slot) in ctx.logits_mut().iter_mut().enumerate() {
                *slot = if idx == (tok + 1) % len { 5.0 } else { 0.1 };
            }
            Ok(())
        }
    }

    #[test]
    fn extract_top_k_sorts_descending_and_breaks_ties_by_token_id() {
        let logits = [1.0_f32, 4.5, 4.5, -2.0, 3.2];
        let top = extract_top_k_logits(&logits, 3).expect("valid top-k");
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].token_id, 1);
        assert!((top[0].logit - 4.5).abs() < f32::EPSILON);
        assert_eq!(top[1].token_id, 2);
        assert!((top[1].logit - 4.5).abs() < f32::EPSILON);
        assert_eq!(top[2].token_id, 4);
        assert!((top[2].logit - 3.2).abs() < f32::EPSILON);
    }

    #[test]
    fn extract_top_k_rejects_nan_and_infinity() {
        let with_nan = [0.1_f32, f32::NAN, 0.3];
        assert!(extract_top_k_logits(&with_nan, 2).is_err());

        let with_inf = [0.1_f32, f32::INFINITY, 0.3];
        assert!(extract_top_k_logits(&with_inf, 2).is_err());
    }

    #[test]
    fn intrinsic_parity_rejects_flat_logits_and_collapsed_repetition() {
        let flat = ParityDumpReport {
            schema_version: "1.0.0".to_owned(),
            architecture: "llama".to_owned(),
            quantization_format: "Q8_0".to_owned(),
            prompt: "test".to_owned(),
            prompt_token_ids: vec![1, 2],
            first_token_top_k: vec![
                TopKLogitEntry {
                    token_id: 0,
                    logit: 0.0,
                },
                TopKLogitEntry {
                    token_id: 1,
                    logit: 0.0,
                },
            ],
            greedy_token_ids: vec![1, 2, 3],
            generated_text: "abc".to_owned(),
        };
        let err_flat = verify_intrinsic_parity(&flat, 0.05).unwrap_err();
        assert!(err_flat.contains("degenerate/flat"));

        let collapsed = ParityDumpReport {
            schema_version: "1.0.0".to_owned(),
            architecture: "llama".to_owned(),
            quantization_format: "Q8_0".to_owned(),
            prompt: "test".to_owned(),
            prompt_token_ids: vec![1, 2],
            first_token_top_k: vec![
                TopKLogitEntry {
                    token_id: 7,
                    logit: 9.0,
                },
                TopKLogitEntry {
                    token_id: 2,
                    logit: 1.0,
                },
            ],
            greedy_token_ids: vec![7, 7, 7, 7, 7, 7],
            generated_text: "777777".to_owned(),
        };
        let err_col = verify_intrinsic_parity(&collapsed, 0.05).unwrap_err();
        assert!(err_col.contains("collapsed"));
    }

    #[test]
    fn prompt_blind_stub_recipe_is_rejected_by_sensitivity_gate() {
        let recipe = PromptBlindStubRecipe;
        let weights = DummyWeights;
        let mut kv_a = SingleSlotKvArena::with_dimensions(1, 2, 16, 16).expect("kv arena");
        let mut kv_b = SingleSlotKvArena::with_dimensions(1, 2, 16, 16).expect("kv arena");

        let (dump_a, _) = evaluate_parity_snapshot(
            &recipe,
            &weights,
            &mut kv_a,
            None,
            ClusterPhasePolicy::Auto,
            "What is 2 + 2?",
            &[1, 4, 5],
            16,
            4,
            8,
            "Q8_0",
            |tok, buf| {
                let _ = write!(buf, "{tok} ");
                Ok(StreamAction::Continue)
            },
        )
        .expect("snapshot A");

        let (dump_b, _) = evaluate_parity_snapshot(
            &recipe,
            &weights,
            &mut kv_b,
            None,
            ClusterPhasePolicy::Auto,
            "Capital of France?",
            &[1, 9, 11, 12],
            16,
            4,
            8,
            "Q8_0",
            |tok, buf| {
                let _ = write!(buf, "{tok} ");
                Ok(StreamAction::Continue)
            },
        )
        .expect("snapshot B");

        let err = verify_prompt_sensitivity(&dump_a, &dump_b)
            .expect_err("prompt-blind stub must be rejected");
        assert!(err.contains("ignores token_embd.weight"));
    }

    #[test]
    fn golden_parity_verifier_accepts_matching_and_rejects_drifted_snapshots() {
        let golden = ParityDumpReport {
            schema_version: "1.0.0".to_owned(),
            architecture: "llama".to_owned(),
            quantization_format: "Q8_0".to_owned(),
            prompt: "hello".to_owned(),
            prompt_token_ids: vec![1, 9],
            first_token_top_k: vec![
                TopKLogitEntry {
                    token_id: 10,
                    logit: 6.25,
                },
                TopKLogitEntry {
                    token_id: 12,
                    logit: 3.50,
                },
                TopKLogitEntry {
                    token_id: 15,
                    logit: 1.25,
                },
            ],
            greedy_token_ids: vec![10, 11, 12],
            generated_text: "worldExplainRVV".to_owned(),
        };

        let mut slightly_perturbed = golden.clone();
        slightly_perturbed.first_token_top_k[0].logit = 6.2505;
        assert!(verify_parity_against_golden(&slightly_perturbed, &golden, 0.9999, 1.0e-2).is_ok());

        let mut wrong_greedy = golden.clone();
        wrong_greedy.greedy_token_ids = vec![10, 11, 14];
        assert!(verify_parity_against_golden(&wrong_greedy, &golden, 0.99, 1.0).is_err());

        let mut drifted_logit = golden.clone();
        drifted_logit.first_token_top_k[0].logit = 9.0;
        assert!(verify_parity_against_golden(&drifted_logit, &golden, 0.90, 0.05).is_err());
    }

    #[test]
    fn encode_text_utf8_safe_handles_multibyte_punctuation_and_emoji_without_panic() {
        let gguf_bytes = jero_core::test_helpers::create_synthetic_gguf_v3("synthetic", 32, &[]);
        let gguf = jero_core::GgufFile::parse(&gguf_bytes).expect("parse synthetic gguf");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");

        let tokens = encode_text_utf8_safe(&tokenizer, "Janet’s ducks — €100 🚀")
            .expect("UTF-8 safe encode must succeed without char-boundary panic");
        assert_eq!(tokens.first(), Some(&1), "BOS must be emitted once");
        assert!(tokens.len() > 5);
    }
}
