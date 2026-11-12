//! Differential logit and greedy-token parity exporter (`jero-diag --dump-parity`, ADR-045).
//!
//! Executes an isolated forward pass on a real or fixture GGUF checkpoint via [`LockedModelWeights`],
//! [`GgufTokenizer`], and [`ArchRecipe`], writing a machine-readable [`ParityDumpReport`] JSON file
//! for differential comparison against `llama.cpp` (`llama-cli`) or git-tracked golden fixtures.

use std::io::Write;
use std::path::Path;

use jero_core::{CoordinationProfile, GgufFile, K3Error, QuantizationInfo};
use jero_engine::{
    ClusterPhasePolicy, DEFAULT_PARITY_TOP_K, LockedModelWeights, ParityDumpReport,
    SingleSlotKvArena, StreamAction, SystemMemoryInspector, WorkerPool, evaluate_parity_snapshot,
    resolve_affinity, tokenize_prompt_utf8_safe,
};
use jero_tokenizer::{DetokenizeStep, GgufTokenizer};

use crate::{DiagConfig, is_abort_requested};

/// Executes a single deterministic greedy ($T = 0$) forward pass and writes a [`ParityDumpReport`]
/// JSON snapshot to `--dump-parity <path>`.
pub fn run_parity_dump<W: Write>(
    config: &DiagConfig,
    mut out: W,
) -> Result<ParityDumpReport, K3Error> {
    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --model <path>".to_owned(),
        })?;
    let dump_path = config
        .dump_parity
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --dump-parity <path>".to_owned(),
        })?;

    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(64);
    let max_tokens = config.max_tokens.unwrap_or(16);
    let profile = config.coord.unwrap_or(if config.phase_policy.is_some() {
        CoordinationProfile::Rayon
    } else {
        CoordinationProfile::Spin
    });

    let inspector = SystemMemoryInspector::new();
    let (worker_pool, locked_weights, recipe, resolved_policy) = if let Some(policy) =
        config.phase_policy
    {
        let (locked_weights, recipe, _footprint) =
            LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;
        let gguf_probe = GgufFile::parse(locked_weights.as_bytes())?;
        let tokenizer_probe = GgufTokenizer::from_gguf(&gguf_probe)?;
        let probe_tokens = tokenize_prompt_utf8_safe(&tokenizer_probe, prompt_text, false, n_ctx)?;
        let worker_pool = WorkerPool::for_phase_policy(
            policy,
            locked_weights.metadata(),
            probe_tokens.len(),
            config.core_mask,
            config.threads,
            profile,
        )?;
        worker_pool.verify_live_affinities()?;
        let effective = worker_pool.phase_policy().unwrap_or(policy);
        (worker_pool, locked_weights, recipe, effective)
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
        )
    };

    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let mut kv_arena = SingleSlotKvArena::from_gguf(&gguf_view, n_ctx)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens = tokenize_prompt_utf8_safe(&tokenizer, prompt_text, false, n_ctx)?;
    let quant_str = QuantizationInfo::from_descriptors(locked_weights.metadata().tensors()).primary;

    let mut detokenizer = tokenizer.streaming_detokenizer();
    let bounded_max = max_tokens.min(n_ctx.saturating_sub(prompt_tokens.len()));
    let (mut report, _) = evaluate_parity_snapshot(
        recipe.as_ref(),
        &locked_weights,
        &mut kv_arena,
        Some(&worker_pool),
        resolved_policy,
        prompt_text,
        &prompt_tokens,
        tokenizer.vocab_size(),
        bounded_max,
        DEFAULT_PARITY_TOP_K,
        &quant_str,
        |token_id, text_acc| {
            if is_abort_requested() {
                return Ok(StreamAction::Stop);
            }
            match detokenizer.push_token(token_id)? {
                DetokenizeStep::Emit(piece) => {
                    text_acc.push_str(&piece);
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(prefix) => {
                    text_acc.push_str(&prefix);
                    Ok(StreamAction::Stop)
                }
            }
        },
    )?;

    if !detokenizer.is_stopped() && !is_abort_requested() {
        let trailing = detokenizer.finish();
        if !trailing.is_empty() {
            report.generated_text.push_str(&trailing);
        }
    }

    worker_pool.verify_live_affinities()?;

    let serialized =
        serde_json::to_string_pretty(&report).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "dump_parity".to_owned(),
            reason: e.to_string(),
        })?;

    if dump_path != "-" {
        if let Some(parent) = Path::new(dump_path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "dump_parity".to_owned(),
                reason: format!("failed to create parent directory for '{dump_path}': {e}"),
            })?;
        }
        std::fs::write(dump_path, format!("{serialized}\n")).map_err(|e| {
            K3Error::CapabilityMismatch {
                architecture: "dump_parity".to_owned(),
                reason: format!("failed to write --dump-parity '{dump_path}': {e}"),
            }
        })?;
    }

    if config.json || dump_path == "-" {
        let _ = writeln!(out, "{serialized}");
    } else {
        let _ = writeln!(
            out,
            "[parity] exported {} ({}) top-{} logits and {} greedy tokens -> {}",
            report.architecture,
            report.quantization_format,
            report.first_token_top_k.len(),
            report.greedy_token_ids.len(),
            dump_path
        );
    }
    let _ = out.flush();

    Ok(report)
}
