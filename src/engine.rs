//! Cycle-exclusive single-prompt forward-pass engine loop (US-01.6.1, ADR-005, ADR-027).
//!
//! Enforces strict single-prompt cycle exclusivity (zero cross-prompt batching),
//! transitions deterministically from prompt prefill to token-by-token autoregressive decode,
//! and invokes an unbuffered streaming callback on every generated token.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights};

use crate::affinity::WorkerPool;
use crate::kv::SingleSlotKvArena;

/// Execution phase of the single-prompt forward pass state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnginePhase {
    /// Engine is idle and ready to accept a single prompt.
    Idle,
    /// Executing prompt prefill across all input tokens.
    Prefill,
    /// Executing autoregressive single-token decode steps.
    Decode,
    /// Forward pass completed cleanly.
    Completed,
}

/// Flow-control action returned by the streaming token callback after each generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAction {
    /// Continue generating the next token.
    Continue,
    /// Stop generation immediately (e.g., EOS/stop sequence reached or signal abort).
    Stop,
}

/// Execution timing, throughput, and synchronization barrier wait metrics for a single forward pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PassTiming {
    /// Time to First Token (TTFT) in milliseconds: duration from prompt prefill start
    /// until the greedy argmax sampling of the first generated token.
    pub ttft_ms: f64,
    /// Wall-clock duration of the prompt prefill phase in milliseconds.
    pub prefill_ms: f64,
    /// Wall-clock duration of the autoregressive decode phase in milliseconds.
    pub decode_ms: f64,
    /// Decode generation throughput in tokens per second.
    /// Calculated as `decode_steps / (decode_ms / 1000.0)`.
    /// Evaluates to `0.0` if no decode tokens were generated (`decode_steps == 0`).
    pub decode_tokens_per_sec: f64,
    /// Total wall-clock execution duration of the forward pass in milliseconds (`prefill_ms + decode_ms`).
    pub total_wall_time_ms: f64,
    /// Cumulative inter-worker barrier wait overhead in milliseconds, aggregated across all assigned worker threads.
    pub cumulative_barrier_wait_ms: f64,
    /// Ratio of cumulative barrier wait overhead to total available worker CPU time (`cumulative_barrier_wait_ms / (total_wall_time_ms * num_workers)`).
    /// Clamped to `[0.0, 1.0]`.
    pub barrier_wait_ratio: f64,
    /// Ratio of active compute utilization across assigned cores (`1.0 - barrier_wait_ratio`).
    /// Clamped to `[0.0, 1.0]`.
    pub compute_utilization_ratio: f64,
    /// Number of decode steps executed.
    pub decode_steps: usize,
}

impl Default for PassTiming {
    fn default() -> Self {
        Self {
            ttft_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            decode_tokens_per_sec: 0.0,
            total_wall_time_ms: 0.0,
            cumulative_barrier_wait_ms: 0.0,
            barrier_wait_ratio: 0.0,
            compute_utilization_ratio: 1.0,
            decode_steps: 0,
        }
    }
}

impl PassTiming {
    /// Creates a zeroed [`PassTiming`] instance.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            ttft_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            decode_tokens_per_sec: 0.0,
            total_wall_time_ms: 0.0,
            cumulative_barrier_wait_ms: 0.0,
            barrier_wait_ratio: 0.0,
            compute_utilization_ratio: 1.0,
            decode_steps: 0,
        }
    }
}

/// Summary statistics and generated token sequence from a completed single-prompt evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationSummary {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Sequence of generated token IDs produced during prefill + decode.
    pub generated_tokens: Vec<u32>,
    /// Final KV cache sequence cursor position (`prompt_tokens + decode_steps`).
    pub final_position: usize,
    /// Final state machine phase (`EnginePhase::Completed`).
    pub final_phase: EnginePhase,
    /// Phase-level execution timing and synchronization barrier wait metrics.
    pub timing: PassTiming,
}

/// Single-slot forward-pass orchestrator enforcing strict cycle exclusivity (ADR-005, US-01.6.1).
#[derive(Debug)]
pub struct SinglePromptEngine {
    active_cycle: AtomicBool,
    phase: EnginePhase,
}

impl Default for SinglePromptEngine {
    fn default() -> Self {
        Self::new()
    }
}

struct CycleGuard<'a> {
    flag: &'a AtomicBool,
}

impl<'a> CycleGuard<'a> {
    fn acquire(flag: &'a AtomicBool, arch: &str) -> Result<Self, K3Error> {
        if flag
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "cycle exclusivity violation (ADR-005): another prompt is already executing on this engine instance".to_owned(),
            });
        }
        Ok(Self { flag })
    }
}

impl Drop for CycleGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

impl SinglePromptEngine {
    /// Creates a new idle [`SinglePromptEngine`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_cycle: AtomicBool::new(false),
            phase: EnginePhase::Idle,
        }
    }

    /// Returns the current phase of the engine state machine.
    #[must_use]
    pub const fn phase(&self) -> EnginePhase {
        self.phase
    }

    /// Returns `true` if a prompt is currently executing on this engine instance.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.active_cycle.load(Ordering::SeqCst)
    }

    /// Executes an isolated single-prompt forward pass (prefill + autoregressive decode)
    /// without a multi-thread worker pool (single-thread baseline).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if another prompt is already active (ADR-005)
    ///   or if `prompt_tokens` / `vocab_size` is empty.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > kv_arena.n_ctx()`.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_prompt<F>(
        &mut self,
        recipe: &dyn ArchRecipe,
        weights: &dyn ModelWeights,
        kv_arena: &mut SingleSlotKvArena,
        prompt_tokens: &[u32],
        vocab_size: usize,
        max_tokens: usize,
        on_token: F,
    ) -> Result<EvaluationSummary, K3Error>
    where
        F: FnMut(u32) -> Result<StreamAction, K3Error>,
    {
        self.evaluate_prompt_with_pool(
            recipe,
            weights,
            kv_arena,
            None,
            prompt_tokens,
            vocab_size,
            max_tokens,
            on_token,
        )
    }

    /// Executes an isolated single-prompt forward pass dispatching across an optional pinned
    /// [`WorkerPool`] (US-01.5.2, US-01.6.1, US-01 Scenario 1 & 2).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if another prompt is already active (ADR-005)
    ///   or if `prompt_tokens` / `vocab_size` is empty.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > kv_arena.n_ctx()`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn evaluate_prompt_with_pool<F>(
        &mut self,
        recipe: &dyn ArchRecipe,
        weights: &dyn ModelWeights,
        kv_arena: &mut SingleSlotKvArena,
        worker_pool: Option<&WorkerPool>,
        prompt_tokens: &[u32],
        vocab_size: usize,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<EvaluationSummary, K3Error>
    where
        F: FnMut(u32) -> Result<StreamAction, K3Error>,
    {
        let _guard = CycleGuard::acquire(&self.active_cycle, recipe.architecture_id())?;

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
        let mut logits = vec![0.0_f32; vocab_size];
        let mut ctx =
            ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, kv_arena.as_mut_slice())?;

        let pass_start = Instant::now();
        if let Some(pool) = worker_pool {
            pool.reset_barrier_wait();
        }

        // 1. Transition Idle -> Prefill
        self.phase = EnginePhase::Prefill;
        let prefill_start = Instant::now();
        if let Some(pool) = worker_pool {
            pool.execute(|_worker_idx, _total_workers| {
                // Synchronize pinned worker threads on prefill barrier
            });
        }
        recipe.prefill_step(&mut ctx, weights)?;

        let mut generated_tokens = Vec::with_capacity(max_tokens);
        if max_tokens == 0 {
            let prefill_elapsed = prefill_start.elapsed();
            let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;
            let prefill_ms = ttft_ms;
            self.phase = EnginePhase::Completed;
            return Ok(EvaluationSummary {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens,
                final_position: ctx.position(),
                final_phase: self.phase,
                timing: compute_pass_timing(
                    pass_start,
                    ttft_ms,
                    prefill_ms,
                    0.0,
                    0,
                    0.0,
                    worker_pool,
                ),
            });
        }

        // Sample first token produced by prefill logits (TTFT ends here)
        let mut current_token =
            sample_greedy_argmax(ctx.logits()).ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "empty logits buffer after prefill step".to_owned(),
            })?;
        let prefill_elapsed = prefill_start.elapsed();
        let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        let prefill_ms = ttft_ms;

        generated_tokens.push(current_token);
        if on_token(current_token)? == StreamAction::Stop {
            self.phase = EnginePhase::Completed;
            return Ok(EvaluationSummary {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens,
                final_position: ctx.position(),
                final_phase: self.phase,
                timing: compute_pass_timing(
                    pass_start,
                    ttft_ms,
                    prefill_ms,
                    0.0,
                    0,
                    0.0,
                    worker_pool,
                ),
            });
        }

        // 2. Transition Prefill -> Decode
        self.phase = EnginePhase::Decode;
        let decode_start = Instant::now();
        let mut decode_steps = 0usize;

        while generated_tokens.len() < max_tokens && ctx.position() < n_ctx {
            ctx.prepare_decode(current_token)?;
            if let Some(pool) = worker_pool {
                pool.execute(|_worker_idx, _total_workers| {
                    // Synchronize pinned worker threads on decode step
                });
            }
            recipe.decode_step(&mut ctx, weights)?;

            current_token =
                sample_greedy_argmax(ctx.logits()).ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: recipe.architecture_id().to_owned(),
                    reason: "empty logits buffer after decode step".to_owned(),
                })?;

            generated_tokens.push(current_token);
            decode_steps += 1;
            if on_token(current_token)? == StreamAction::Stop {
                break;
            }
        }

        let decode_duration = decode_start.elapsed();
        let decode_ms = decode_duration.as_secs_f64() * 1000.0;
        let decode_duration_secs = decode_duration.as_secs_f64();
        let decode_tokens_per_sec = if decode_duration_secs > 0.0 && decode_steps > 0 {
            decode_steps as f64 / decode_duration_secs
        } else {
            0.0
        };

        self.phase = EnginePhase::Completed;
        Ok(EvaluationSummary {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens,
            final_position: ctx.position(),
            final_phase: self.phase,
            timing: compute_pass_timing(
                pass_start,
                ttft_ms,
                prefill_ms,
                decode_ms,
                decode_steps,
                decode_tokens_per_sec,
                worker_pool,
            ),
        })
    }
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn compute_pass_timing(
    pass_start: Instant,
    ttft_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    decode_steps: usize,
    decode_tokens_per_sec: f64,
    worker_pool: Option<&WorkerPool>,
) -> PassTiming {
    let total_wall_time_ms = pass_start.elapsed().as_secs_f64() * 1000.0;
    let cumulative_barrier_wait_ms =
        worker_pool.map_or(0.0, WorkerPool::cumulative_barrier_wait_ms);
    let worker_count = worker_pool.map_or(1, WorkerPool::thread_count);
    let total_core_time_ms = total_wall_time_ms * (worker_count as f64);
    let barrier_wait_ratio = if total_core_time_ms > 0.0 {
        (cumulative_barrier_wait_ms / total_core_time_ms).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let compute_utilization_ratio = (1.0 - barrier_wait_ratio).clamp(0.0, 1.0);

    PassTiming {
        ttft_ms,
        prefill_ms,
        decode_ms,
        decode_tokens_per_sec,
        total_wall_time_ms,
        cumulative_barrier_wait_ms,
        barrier_wait_ratio,
        compute_utilization_ratio,
        decode_steps,
    }
}

/// Deterministic greedy (temperature `0.0`) argmax token sampler with lowest-token-ID tie-breaking.
#[must_use]
pub fn sample_greedy_argmax(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    let mut best_id = 0_usize;
    let mut best_score = logits[0];

    for (i, &score) in logits.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_id = i;
        }
    }

    u32::try_from(best_id).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "arch-synthetic")]
    use k3_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

    #[cfg(feature = "arch-synthetic")]
    #[test]
    fn test_single_prompt_engine_prefill_and_decode_loop_us01_6_1() {
        use arch_synthetic::SyntheticRecipe;

        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("allocate kv arena");
        let recipe = SyntheticRecipe::new();

        let mut engine = SinglePromptEngine::new();
        assert_eq!(engine.phase(), EnginePhase::Idle);

        let mut streamed = Vec::new();
        let summary = engine
            .evaluate_prompt(&recipe, &gguf, &mut kv_arena, &[1, 9, 10], 16, 4, |tok| {
                streamed.push(tok);
                Ok(StreamAction::Continue)
            })
            .expect("evaluate prompt");

        assert_eq!(summary.prompt_tokens, 3);
        assert_eq!(summary.generated_tokens, vec![9, 10, 11, 12]);
        assert_eq!(streamed, vec![9, 10, 11, 12]);
        assert_eq!(summary.final_position, 3 + 3); // 3 prefill + 3 decode steps = 6
        assert_eq!(summary.final_phase, EnginePhase::Completed);
        assert!(!engine.is_busy());

        // Phase timing instrumentation validation
        assert!(summary.timing.ttft_ms >= 0.0);
        assert!(summary.timing.total_wall_time_ms >= summary.timing.ttft_ms);
        assert_eq!(summary.timing.decode_steps, 3);
        assert!(summary.timing.decode_tokens_per_sec >= 0.0);
        assert!(summary.timing.cumulative_barrier_wait_ms.abs() <= f64::EPSILON);
        assert!(summary.timing.barrier_wait_ratio.abs() <= f64::EPSILON);
        assert!((summary.timing.compute_utilization_ratio - 1.0).abs() <= f64::EPSILON);
    }

    #[cfg(feature = "arch-synthetic")]
    #[test]
    fn test_phase_timing_metrics_in_single_prompt_engine() {
        use arch_synthetic::SyntheticRecipe;

        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let recipe = SyntheticRecipe::new();

        // 1. max_tokens = 0 (prefill only)
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
        let mut engine = SinglePromptEngine::new();
        let summary_0 = engine
            .evaluate_prompt(&recipe, &gguf, &mut kv_arena, &[1, 9, 10], 16, 0, |_| {
                Ok(StreamAction::Continue)
            })
            .expect("eval 0");
        assert_eq!(summary_0.generated_tokens.len(), 0);
        assert_eq!(summary_0.timing.decode_steps, 0);
        assert!(summary_0.timing.decode_tokens_per_sec.abs() <= f64::EPSILON);
        assert!(summary_0.timing.ttft_ms >= 0.0);

        // 2. max_tokens = 1 (prefill + first token sample, zero decode steps)
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
        let summary_1 = engine
            .evaluate_prompt(&recipe, &gguf, &mut kv_arena, &[1, 9, 10], 16, 1, |_| {
                Ok(StreamAction::Continue)
            })
            .expect("eval 1");
        assert_eq!(summary_1.generated_tokens.len(), 1);
        assert_eq!(summary_1.timing.decode_steps, 0);
        assert!(summary_1.timing.decode_tokens_per_sec.abs() <= f64::EPSILON);
        assert!(summary_1.timing.ttft_ms >= 0.0);

        // 3. PassTiming zero and default
        let zero_timing = PassTiming::zero();
        assert_eq!(zero_timing, PassTiming::default());
        assert!((zero_timing.compute_utilization_ratio - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn test_cycle_exclusivity_guard_rejects_concurrent_prompt_adr005() {
        let flag = AtomicBool::new(false);
        let _guard1 = CycleGuard::acquire(&flag, "synthetic").expect("first guard acquires slot");
        let err = CycleGuard::acquire(&flag, "synthetic")
            .err()
            .expect("second concurrent prompt must be rejected");
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        assert!(err.to_string().contains("cycle exclusivity violation"));
    }
}
