//! Cycle-exclusive single-prompt forward-pass engine loop (US-01.6.1, ADR-005, ADR-027).
//!
//! Enforces strict single-prompt cycle exclusivity (zero cross-prompt batching),
//! transitions deterministically from prompt prefill to token-by-token autoregressive decode,
//! and invokes an unbuffered streaming callback on every generated token.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights};

use crate::affinity::{
    ClusterPhasePolicy, WorkerPool, X100_CLUSTER_MASK, select_phase_policy_for_arch,
};
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
    phase_policy: ClusterPhasePolicy,
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
    /// Creates a new idle [`SinglePromptEngine`] with [`ClusterPhasePolicy::Auto`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_cycle: AtomicBool::new(false),
            phase: EnginePhase::Idle,
            phase_policy: ClusterPhasePolicy::Auto,
        }
    }

    /// Configures the execution-phase cluster scheduling policy for this engine instance.
    #[must_use]
    pub const fn with_phase_policy(mut self, phase_policy: ClusterPhasePolicy) -> Self {
        self.phase_policy = phase_policy;
        self
    }

    /// Updates the execution-phase cluster scheduling policy for subsequent forward passes.
    pub fn set_phase_policy(&mut self, phase_policy: ClusterPhasePolicy) {
        self.phase_policy = phase_policy;
    }

    /// Returns the configured [`ClusterPhasePolicy`] for this engine instance.
    #[must_use]
    pub const fn phase_policy(&self) -> ClusterPhasePolicy {
        self.phase_policy
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
        let logits_len = infer_logits_buffer_len(weights, vocab_size);
        let mut logits = vec![0.0_f32; logits_len];
        let mut ctx =
            ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, kv_arena.as_mut_slice())?;

        let effective_policy = match self.phase_policy {
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

        let pass_start = Instant::now();
        if let Some(pool) = worker_pool {
            pool.reset_barrier_wait();
        }

        // 1. Transition Idle -> Prefill
        self.phase = EnginePhase::Prefill;
        let prefill_start = Instant::now();
        let requires_kv = recipe.requires_kv_cache();
        if let Some(pool) = worker_pool {
            pool.install_for_phase(EnginePhase::Prefill, effective_policy, || {
                if requires_kv {
                    recipe.prefill_step(&mut ctx, weights)
                } else {
                    recipe.forward_classify(&mut ctx, weights)
                }
            })?;
        } else if requires_kv {
            recipe.prefill_step(&mut ctx, weights)?;
        } else {
            recipe.forward_classify(&mut ctx, weights)?;
        }

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
        let sample_len = vocab_size.min(ctx.logits().len());
        let mut current_token =
            sample_greedy_argmax(&ctx.logits()[..sample_len]).ok_or_else(|| {
                K3Error::CapabilityMismatch {
                    architecture: recipe.architecture_id().to_owned(),
                    reason: "empty logits buffer after prefill step".to_owned(),
                }
            })?;
        let prefill_elapsed = prefill_start.elapsed();
        let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        let prefill_ms = ttft_ms;

        generated_tokens.push(current_token);
        if on_token(current_token)? == StreamAction::Stop || !requires_kv {
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
                pool.install_for_phase(EnginePhase::Decode, effective_policy, || {
                    recipe.decode_step(&mut ctx, weights)
                })?;
            } else {
                recipe.decode_step(&mut ctx, weights)?;
            }

            let step_sample_len = vocab_size.min(ctx.logits().len());
            current_token =
                sample_greedy_argmax(&ctx.logits()[..step_sample_len]).ok_or_else(|| {
                    K3Error::CapabilityMismatch {
                        architecture: recipe.architecture_id().to_owned(),
                        reason: "empty logits buffer after decode step".to_owned(),
                    }
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

    /// Executes a zero-KV single-pass classification or logit-harvest evaluation (`0` decode steps)
    /// using [`ArchRecipe::forward_classify`] without allocating KV cache pages.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if another prompt is already executing, or if
    /// `prompt_tokens` or `num_labels` is empty/zero; returns [`K3Error::ContextLengthExceeded`] if
    /// `prompt_tokens.len() > n_ctx`.
    pub fn evaluate_classify(
        &mut self,
        recipe: &dyn ArchRecipe,
        weights: &dyn ModelWeights,
        prompt_tokens: &[u32],
        n_ctx: usize,
        num_labels: usize,
    ) -> Result<(Vec<f32>, EvaluationSummary), K3Error> {
        let _guard = CycleGuard::acquire(&self.active_cycle, recipe.architecture_id())?;

        if prompt_tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "classification prompt sequence cannot be empty".to_owned(),
            });
        }
        if num_labels == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "classification num_labels must be greater than zero".to_owned(),
            });
        }

        let mut logits = vec![0.0_f32; num_labels];
        let mut ctx = ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, &mut [])?;
        let pass_start = Instant::now();
        self.phase = EnginePhase::Prefill;
        recipe.forward_classify(&mut ctx, weights)?;

        let prefill_ms = pass_start.elapsed().as_secs_f64() * 1000.0;
        let winning_label = sample_greedy_argmax(ctx.logits()).unwrap_or(0);
        let final_position = ctx.position();
        self.phase = EnginePhase::Completed;

        Ok((
            logits,
            EvaluationSummary {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: vec![winning_label],
                final_position,
                final_phase: self.phase,
                timing: compute_pass_timing(pass_start, prefill_ms, prefill_ms, 0.0, 0, 0.0, None),
            },
        ))
    }
}

/// Computes log-softmax probabilities over `candidate_token_ids` from raw prefill `logits`.
///
/// Used immediately after [`ArchRecipe::prefill_step`] for zero-decode decision classification and
/// action routing, guaranteeing normalized probability mass across candidate action tokens.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if inputs are empty or any candidate logit is non-finite,
/// or [`K3Error::TensorBoundsError`] if any candidate token ID exceeds `logits.len()`.
pub fn harvest_decision_logprobs(
    logits: &[f32],
    candidate_token_ids: &[u32],
) -> Result<Vec<(u32, f32)>, K3Error> {
    if logits.is_empty() || candidate_token_ids.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "decision_logprobs".to_owned(),
            reason: "logits and candidate_token_ids cannot be empty".to_owned(),
        });
    }

    let mut seen = HashSet::with_capacity(candidate_token_ids.len());
    let mut unique_ids = Vec::with_capacity(candidate_token_ids.len());
    let mut max_logit = f32::NEG_INFINITY;

    for &tok_id in candidate_token_ids {
        let idx = usize::try_from(tok_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "decision_logits".to_owned(),
            offset: usize::MAX,
            byte_len: size_of::<f32>(),
            buffer_len: size_of_val(logits),
            reason: format!("candidate token id {tok_id} overflows usize"),
        })?;
        if idx >= logits.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "decision_logits".to_owned(),
                offset: idx,
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(logits),
                reason: format!(
                    "candidate token id {tok_id} out of bounds for logits len {}",
                    logits.len()
                ),
            });
        }
        let val = logits[idx];
        if !val.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "decision_logprobs".to_owned(),
                reason: format!("non-finite logit ({val}) at candidate token id {tok_id}"),
            });
        }
        if seen.insert(tok_id) {
            unique_ids.push(tok_id);
            if val > max_logit {
                max_logit = val;
            }
        }
    }

    let mut sum_exp = 0.0_f64;
    for &tok_id in &unique_ids {
        let val = f64::from(logits[tok_id as usize]);
        sum_exp += (val - f64::from(max_logit)).exp();
    }
    let log_z = f64::from(max_logit) + sum_exp.ln();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "log-softmax difference converges cleanly in f32 precision"
    )]
    let out = unique_ids
        .into_iter()
        .map(|tok_id| {
            let logp = (f64::from(logits[tok_id as usize]) - log_z) as f32;
            (tok_id, logp)
        })
        .collect();

    Ok(out)
}

/// Applies a deterministic Finite State Machine (FSM) allowed-token mask directly to `logits`.
///
/// Sets every disallowed vocabulary slot to [`f32::NEG_INFINITY`] and selects the winning token ID
/// from `allowed_token_ids` with lowest-token-ID tie breaking.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if inputs are empty or any allowed logit is `NaN`, or
/// [`K3Error::TensorBoundsError`] if any allowed token ID is outside `0..logits.len()`.
pub fn apply_fsm_logit_mask(logits: &mut [f32], allowed_token_ids: &[u32]) -> Result<u32, K3Error> {
    if logits.is_empty() || allowed_token_ids.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "fsm_mask".to_owned(),
            reason: "logits and allowed_token_ids cannot be empty".to_owned(),
        });
    }

    let mut allowed = vec![false; logits.len()];
    for &tok_id in allowed_token_ids {
        let idx = usize::try_from(tok_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "fsm_logits".to_owned(),
            offset: usize::MAX,
            byte_len: size_of::<f32>(),
            buffer_len: size_of_val(logits),
            reason: format!("allowed token id {tok_id} overflows usize"),
        })?;
        if idx >= logits.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "fsm_logits".to_owned(),
                offset: idx,
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(logits),
                reason: format!(
                    "allowed token id {tok_id} out of bounds for logits len {}",
                    logits.len()
                ),
            });
        }
        if logits[idx].is_nan() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "fsm_mask".to_owned(),
                reason: format!("NaN logit at allowed token id {tok_id}"),
            });
        }
        allowed[idx] = true;
    }

    let mut best: Option<(u32, f32)> = None;
    for (idx, (&is_allowed, slot)) in allowed.iter().zip(logits.iter_mut()).enumerate() {
        if !is_allowed {
            *slot = f32::NEG_INFINITY;
            continue;
        }
        let Ok(tok_id) = u32::try_from(idx) else {
            continue;
        };
        if best.is_none_or(|(_, best_score)| *slot > best_score) {
            best = Some((tok_id, *slot));
        }
    }

    best.map(|(id, _)| id)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "fsm_mask".to_owned(),
            reason: "no allowed vocabulary token fits in u32 index space".to_owned(),
        })
}

/// Prefix-compiled Finite State Machine decision constraint with unary jump-forward chains and
/// terminal branch early-exit detection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FsmDecisionConstraint {
    /// Allowed outgoing vocabulary token IDs at each FSM state.
    pub allowed_tokens_at_state: HashMap<u32, Vec<u32>>,
    /// Deterministic unary token bursts starting at a state before the next branch or terminal leaf.
    pub forced_jump_forward: HashMap<u32, Vec<u32>>,
    /// Accepting states where the unique decision branch has completed (early exit).
    pub terminal_states: HashSet<u32>,
}

#[derive(Debug, Default)]
struct TrieNode {
    children: BTreeMap<u32, usize>,
    is_end: bool,
}

impl FsmDecisionConstraint {
    /// Compiles a set of valid token ID sequences into a deterministic FSM constraint table,
    /// extracting forced single-choice jump-forward runs and terminal early-exit states.
    #[must_use]
    pub fn from_allowed_sequences(sequences: &[Vec<u32>]) -> Self {
        if sequences.is_empty() {
            return Self::default();
        }

        let mut trie = vec![TrieNode::default()];
        for seq in sequences {
            let mut node_idx = 0_usize;
            for &tok in seq {
                if let Some(&child_idx) = trie[node_idx].children.get(&tok) {
                    node_idx = child_idx;
                } else {
                    let next_idx = trie.len();
                    trie.push(TrieNode::default());
                    trie[node_idx].children.insert(tok, next_idx);
                    node_idx = next_idx;
                }
            }
            trie[node_idx].is_end = true;
        }

        // BFS contiguous numbering so state 0 is root and level order is deterministic.
        let mut node_to_state = vec![None::<u32>; trie.len()];
        let mut bfs_order = Vec::with_capacity(trie.len());
        let mut queue = VecDeque::new();
        node_to_state[0] = Some(0);
        bfs_order.push(0_usize);
        queue.push_back(0_usize);

        while let Some(node_idx) = queue.pop_front() {
            let child_ids: Vec<usize> = trie[node_idx].children.values().copied().collect();
            for child_idx in child_ids {
                if node_to_state[child_idx].is_none()
                    && let Ok(state_id) = u32::try_from(bfs_order.len())
                {
                    node_to_state[child_idx] = Some(state_id);
                    bfs_order.push(child_idx);
                    queue.push_back(child_idx);
                }
            }
        }

        let mut allowed_tokens_at_state = HashMap::new();
        let mut forced_jump_forward = HashMap::new();
        let mut terminal_states = HashSet::new();

        for &node_idx in &bfs_order {
            let Some(state_id) = node_to_state[node_idx] else {
                continue;
            };
            let node = &trie[node_idx];
            if node.is_end {
                terminal_states.insert(state_id);
            }
            if node.children.is_empty() {
                continue;
            }

            let allowed: Vec<u32> = node.children.keys().copied().collect();
            allowed_tokens_at_state.insert(state_id, allowed);

            if !node.is_end && node.children.len() == 1 {
                let mut chain = Vec::new();
                let mut cur = node_idx;
                while !trie[cur].is_end && trie[cur].children.len() == 1 {
                    let Some((&tok, &next_node)) = trie[cur].children.iter().next() else {
                        break;
                    };
                    chain.push(tok);
                    cur = next_node;
                }
                if !chain.is_empty() {
                    forced_jump_forward.insert(state_id, chain);
                }
            }
        }

        Self {
            allowed_tokens_at_state,
            forced_jump_forward,
            terminal_states,
        }
    }

    /// Returns allowed outgoing token IDs for `state`, if any transitions exist.
    #[must_use]
    pub fn allowed_tokens(&self, state: u32) -> Option<&[u32]> {
        self.allowed_tokens_at_state.get(&state).map(Vec::as_slice)
    }

    /// Returns any deterministic forced jump-forward token burst available at `state`.
    #[must_use]
    pub fn forced_continuation(&self, state: u32) -> Option<&[u32]> {
        self.forced_jump_forward.get(&state).map(Vec::as_slice)
    }

    /// Returns `true` when `state` is an accepting terminal state with no required decode steps
    /// remaining (unique-branch early exit).
    #[must_use]
    pub fn is_terminal_state(&self, state: u32) -> bool {
        self.terminal_states.contains(&state)
    }
}

/// Executes a deterministic FSM jump-forward structural token burst in a single `prefill_step`
/// and atomically advances `ctx` position upon success (ADR-005, ADR-026).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `forced_tokens` is empty,
/// [`K3Error::ContextLengthExceeded`] if `ctx.position() + forced_tokens.len() > ctx.n_ctx()`, or
/// any execution error returned by [`ArchRecipe::prefill_step`] without mutating `ctx.position()`.
pub fn execute_jump_forward_prefill(
    recipe: &dyn ArchRecipe,
    ctx: &mut ExecutionContext<'_>,
    weights: &dyn ModelWeights,
    forced_tokens: &[u32],
) -> Result<usize, K3Error> {
    if forced_tokens.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: recipe.architecture_id().to_owned(),
            reason: "jump-forward forced_tokens cannot be empty".to_owned(),
        });
    }

    let start_pos = ctx.position();
    let n_ctx = ctx.n_ctx();
    let target_pos =
        start_pos
            .checked_add(forced_tokens.len())
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: n_ctx,
            })?;
    if target_pos > n_ctx {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: target_pos,
            max_context_tokens: n_ctx,
        });
    }

    let (logits_buf, kv_buf) = ctx.buffers_mut();
    let mut burst_ctx = ExecutionContext::new(forced_tokens, n_ctx, logits_buf, kv_buf)?
        .with_position(start_pos)?;
    burst_ctx.prepare_jump_forward(forced_tokens)?;
    recipe.prefill_step(&mut burst_ctx, weights)?;
    burst_ctx.clear_jump_forward();

    ctx.advance_position(forced_tokens.len())
}

/// Computes the safe logit scratch buffer length for `weights` and `vocab_size`,
/// accounting for 64-element GGUF row-alignment padding on `output.weight`.
#[must_use]
pub fn infer_logits_buffer_len(weights: &dyn ModelWeights, vocab_size: usize) -> usize {
    if weights.contains_tensor("output.weight")
        && let Ok(view) = weights.tensor("output.weight")
        && view.shape().len() == 2
    {
        let rows = view.shape()[1];
        if rows > vocab_size && rows % 64 == 0 && (rows - vocab_size) <= 1024 {
            return rows;
        }
    }
    vocab_size
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
    #[cfg(feature = "jero-arch-synthetic")]
    use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_single_prompt_engine_prefill_and_decode_loop_us01_6_1() {
        use jero_arch_synthetic::SyntheticRecipe;

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

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_phase_timing_metrics_in_single_prompt_engine() {
        use jero_arch_synthetic::SyntheticRecipe;

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

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::too_many_lines)]
    fn test_engine_dispatches_prefill_and_decode_inside_phase_scoped_worker_pool() {
        use crate::affinity::{MockCpuTopology, WorkerPool, get_current_thread_core};
        use jero_core::{
            CoordinationProfile, GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3,
        };
        use std::sync::Mutex;

        #[derive(Debug)]
        struct PhaseProbeRecipe {
            prefill_cores: Mutex<Vec<Option<usize>>>,
            decode_cores: Mutex<Vec<Option<usize>>>,
        }

        impl ArchRecipe for PhaseProbeRecipe {
            fn architecture_id(&self) -> &'static str {
                "llama"
            }

            fn recipe_version(&self) -> &'static str {
                "1.0.0"
            }

            fn target_layouts(&self) -> &'static [&'static str] {
                &["native"]
            }

            fn supports_quant(&self, _quant: QuantType) -> bool {
                true
            }

            fn prefill_step(
                &self,
                ctx: &mut ExecutionContext<'_>,
                _weights: &dyn ModelWeights,
            ) -> Result<(), K3Error> {
                self.prefill_cores
                    .lock()
                    .expect("lock prefill cores")
                    .push(get_current_thread_core());
                let n = ctx.prompt_tokens().len();
                ctx.advance_position(n)?;
                ctx.logits_mut().fill(0.0);
                ctx.logits_mut()[1] = 1.0;
                Ok(())
            }

            fn decode_step(
                &self,
                ctx: &mut ExecutionContext<'_>,
                _weights: &dyn ModelWeights,
            ) -> Result<(), K3Error> {
                self.decode_cores
                    .lock()
                    .expect("lock decode cores")
                    .push(get_current_thread_core());
                ctx.advance_position(1)?;
                ctx.logits_mut().fill(0.0);
                ctx.logits_mut()[2] = 1.0;
                Ok(())
            }
        }

        let bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");

        // 1. Always verify single-cluster X100Only execution on core 0
        let x100_pool = WorkerPool::for_phase_policy(
            ClusterPhasePolicy::X100Only,
            gguf.metadata(),
            4,
            Some(0x0001),
            Some(1),
            CoordinationProfile::Rayon,
        )
        .expect("spawn X100Only pool");

        let probe_single = PhaseProbeRecipe {
            prefill_cores: Mutex::new(Vec::new()),
            decode_cores: Mutex::new(Vec::new()),
        };
        let mut engine = SinglePromptEngine::new().with_phase_policy(ClusterPhasePolicy::X100Only);
        assert_eq!(engine.phase_policy(), ClusterPhasePolicy::X100Only);

        let summary = engine
            .evaluate_prompt_with_pool(
                &probe_single,
                &gguf,
                &mut kv_arena,
                Some(&x100_pool),
                &[1, 2, 3, 4],
                8,
                3,
                |_| Ok(StreamAction::Continue),
            )
            .expect("evaluate prompt with X100Only pool");
        assert_eq!(summary.generated_tokens, vec![1, 2, 2]);
        assert_eq!(*probe_single.prefill_cores.lock().unwrap(), vec![Some(0)]);
        assert_eq!(
            *probe_single.decode_cores.lock().unwrap(),
            vec![Some(0), Some(0)]
        );

        // 2. When dual-cluster pinning succeeds (16-core K3 or non-Linux), verify Prefill runs
        // on X100 (0..8) and Decode runs on A100 (8..16).
        let k3_topo = MockCpuTopology::spacemit_k3();
        if let Ok(dual_pool) = WorkerPool::for_phase_policy_with(
            ClusterPhasePolicy::X100PrefillA100Decode,
            gguf.metadata(),
            4,
            None,
            Some(4),
            CoordinationProfile::Rayon,
            &k3_topo,
        ) {
            let probe_dual = PhaseProbeRecipe {
                prefill_cores: Mutex::new(Vec::new()),
                decode_cores: Mutex::new(Vec::new()),
            };
            engine.set_phase_policy(ClusterPhasePolicy::Auto);
            assert_eq!(engine.phase_policy(), ClusterPhasePolicy::Auto);

            let dual_summary = engine
                .evaluate_prompt_with_pool(
                    &probe_dual,
                    &gguf,
                    &mut kv_arena,
                    Some(&dual_pool),
                    &[1, 2, 3, 4],
                    8,
                    3,
                    |_| Ok(StreamAction::Continue),
                )
                .expect("evaluate prompt on dual pool");
            assert_eq!(dual_summary.generated_tokens, vec![1, 2, 2]);

            let prefill_cores = probe_dual.prefill_cores.lock().unwrap().clone();
            let decode_cores = probe_dual.decode_cores.lock().unwrap().clone();
            assert_eq!(prefill_cores.len(), 1);
            assert_eq!(decode_cores.len(), 2);
            assert!(
                prefill_cores[0].is_some_and(|c| c < 8),
                "Prefill must execute on X100 cluster (0..7), got {prefill_cores:?}"
            );
            assert!(
                decode_cores
                    .iter()
                    .all(|c| c.is_some_and(|id| (8..16).contains(&id))),
                "Decode must execute on A100 cluster (8..15), got {decode_cores:?}"
            );
        }
    }

    #[test]
    fn test_harvest_decision_logprobs() {
        let logits = [1.0_f32, 3.0, 2.0, -4.0, 3.0];
        let probs = harvest_decision_logprobs(&logits, &[1, 2, 0]).expect("logprob harvest");
        assert_eq!(probs.len(), 3);
        assert_eq!(probs[0].0, 1);
        assert_eq!(probs[1].0, 2);
        assert_eq!(probs[2].0, 0);

        // Softmax over restricted set [1, 2, 0] must normalize to 1.0 and preserve ranking.
        let sum_probs: f32 = probs.iter().map(|(_, lp)| lp.exp()).sum();
        assert!((sum_probs - 1.0).abs() < 1e-5, "sum={sum_probs}");
        assert!(probs[0].1 > probs[1].1 && probs[1].1 > probs[2].1);

        // Out-of-bounds candidate rejected cleanly
        let oob = harvest_decision_logprobs(&logits, &[99]).expect_err("oob candidate fails");
        assert!(matches!(oob, K3Error::TensorBoundsError { .. }));

        // Non-finite candidate logit rejected cleanly
        let nan_logits = [1.0_f32, f32::NAN, 0.5];
        let nan_err =
            harvest_decision_logprobs(&nan_logits, &[1]).expect_err("NaN candidate fails");
        assert!(matches!(nan_err, K3Error::CapabilityMismatch { .. }));
    }

    #[cfg(feature = "jero-arch-modernbert")]
    fn overwrite_f32_tensor_tail(bytes: &mut [u8], tensor_specs: &[(&str, QuantType, &[usize])]) {
        let total_elems: usize = tensor_specs
            .iter()
            .map(|(_, _, shape)| shape.iter().product::<usize>())
            .sum();
        let data_start = bytes.len() - total_elems * 4;
        for idx in 0..total_elems {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small bounded test index fits f32"
            )]
            let val = (((idx * 7 + 3) % 19) as f32 - 9.0) * 0.05;
            let offset = data_start + idx * 4;
            bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    #[cfg(feature = "jero-arch-modernbert")]
    #[test]
    fn test_modernbert_recipe_zero_kv_classify() {
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[8, 16][..]),
            ("blk.0.attn_q.weight", QuantType::F32, &[8, 8][..]),
            ("classifier.weight", QuantType::F32, &[8, 3][..]),
        ];
        let mut bytes = create_synthetic_gguf_v3("modernbert", 64, &tensor_specs);
        overwrite_f32_tensor_tail(&mut bytes, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse modernbert gguf");

        let mut recipe = crate::modernbert::ModernBertRecipe::new_modernbert();
        recipe.num_layers = 2;
        recipe.hidden_dim = 8;
        recipe.ffn_dim = 16;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.num_labels = 3;
        assert!(!recipe.requires_kv_cache());

        let mut engine = SinglePromptEngine::new();
        let (logits, summary) = engine
            .evaluate_classify(&recipe, &gguf, &[2, 4, 6], 64, 3)
            .expect("zero-KV classification pass");
        assert_eq!(logits.len(), 3);
        assert_eq!(summary.timing.decode_steps, 0);
        assert_eq!(summary.final_position, 3);
        assert_eq!(summary.final_phase, EnginePhase::Completed);

        let harvested =
            harvest_decision_logprobs(&logits, &[0, 1, 2]).expect("harvest class logprobs");
        let total_prob: f32 = harvested.iter().map(|(_, lp)| lp.exp()).sum();
        assert!((total_prob - 1.0).abs() < 1e-5);
    }

    #[cfg(feature = "jero-arch-modernbert")]
    #[test]
    fn test_zero_kv_admission_and_logit_harvest() {
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[8, 16][..]),
            ("classifier.weight", QuantType::F32, &[8, 2][..]),
        ];
        let mut bytes = create_synthetic_gguf_v3("modernbert", 128, &tensor_specs);
        overwrite_f32_tensor_tail(&mut bytes, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");

        // Tight RLIMIT_MEMLOCK barely above weight allocation must still admit zero-KV encoders.
        let footprint =
            crate::memory::calculate_footprint(gguf.metadata(), 128).expect("zero-kv footprint");
        assert_eq!(footprint.kv_bytes, 0);
        let kv_estimate = crate::memory::estimate_kv_cache_bytes(gguf.metadata(), 128, false)
            .expect("estimate kv cache");
        assert_eq!(kv_estimate, 0);

        let tight_mem = crate::memory::MockMemoryInspector::new(
            footprint.weight_bytes + 4096,
            footprint.weight_bytes + 4096,
        );
        crate::memory::check_memory_admission_with(gguf.metadata(), 128, &tight_mem)
            .expect("zero-kv encoder must pass tight memlock admission");

        let mut recipe = crate::modernbert::ModernBertRecipe::new_modernbert();
        recipe.num_layers = 1;
        recipe.hidden_dim = 8;
        recipe.ffn_dim = 16;
        recipe.n_heads = 2;
        recipe.head_dim = 4;
        recipe.num_labels = 2;

        let mut arena = SingleSlotKvArena::new(gguf.metadata(), 128).expect("zero-byte kv arena");
        assert_eq!(arena.aligned_bytes(), 0);
        let mut engine = SinglePromptEngine::new();
        let mut emitted = Vec::new();
        let summary = engine
            .evaluate_prompt(&recipe, &gguf, &mut arena, &[1, 3, 5], 2, 4, |tok| {
                emitted.push(tok);
                Ok(StreamAction::Continue)
            })
            .expect("evaluate_prompt with zero-KV encoder exits after 0 decode steps");
        assert_eq!(emitted.len(), 1);
        assert_eq!(summary.timing.decode_steps, 0);
    }

    #[test]
    fn test_fsm_logit_mask_enforces_allowed_set() {
        // Unallowed slot 0 has much higher raw logit than allowed slots [2, 4, 6].
        // Tied top allowed scores at indices 2 and 6 must break toward lowest ID (2).
        let mut logits = [100.0_f32, 50.0, 5.0, 80.0, 3.0, 90.0, 5.0];
        let winner = apply_fsm_logit_mask(&mut logits, &[2, 4, 6]).expect("apply fsm mask");
        assert_eq!(winner, 2);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert!((logits[2] - 5.0).abs() < f32::EPSILON);
        assert!((logits[4] - 3.0).abs() < f32::EPSILON);
        assert!((logits[6] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fsm_from_allowed_sequences_early_exit() {
        let sequences = vec![vec![10, 20, 30, 40], vec![10, 50, 60]];
        let fsm = FsmDecisionConstraint::from_allowed_sequences(&sequences);

        // State 0 (root): single forced prefix token [10]
        assert_eq!(fsm.allowed_tokens(0), Some([10_u32].as_slice()));
        assert_eq!(fsm.forced_continuation(0), Some([10_u32].as_slice()));

        // Branch state 1 (after [10]): branching between 20 and 50 (no forced burst)
        assert_eq!(fsm.allowed_tokens(1), Some([20_u32, 50].as_slice()));
        assert_eq!(fsm.forced_continuation(1), None);

        // State 2 (after [10, 20]): forced jump-forward burst [30, 40] directly to leaf
        assert_eq!(fsm.forced_continuation(2), Some([30_u32, 40].as_slice()));
        // State 3 (after [10, 50]): forced jump-forward burst [60] directly to leaf
        assert_eq!(fsm.forced_continuation(3), Some([60_u32].as_slice()));

        // Terminal leaves exist cleanly
        assert!(!fsm.terminal_states.is_empty());
        assert!(
            fsm.terminal_states
                .iter()
                .any(|&s| fsm.is_terminal_state(s))
        );
    }

    #[test]
    fn test_apply_fsm_logit_mask_and_early_exit() {
        let fsm = FsmDecisionConstraint::from_allowed_sequences(&[vec![1, 3, 4], vec![1, 2, 5]]);
        let allowed_branch = fsm.allowed_tokens(1).expect("branch state");
        assert_eq!(allowed_branch, &[2, 3]);

        let mut logits = [99.0_f32, 88.0, 2.5, 4.0, 77.0, 66.0];
        let picked = apply_fsm_logit_mask(&mut logits, allowed_branch).expect("mask branch");
        assert_eq!(picked, 3);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);

        // Once branch 3 (state 3) is entered, only forced continuation [4] remains before early exit
        assert_eq!(fsm.forced_continuation(3), Some([4_u32].as_slice()));
        assert!(fsm.terminal_states.len() >= 2);
    }

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_jump_forward_prefill_atomic_advance() {
        use jero_arch_synthetic::SyntheticRecipe;
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let bytes = create_synthetic_gguf_v3(
            "synthetic",
            32,
            &[("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..])],
        );
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic");
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
        let recipe = SyntheticRecipe::new();

        let prompt = [1_u32, 2, 3];
        let mut logits = [0.0_f32; 16];
        let mut ctx =
            ExecutionContext::new(&prompt, 32, &mut logits, kv_arena.as_mut_slice()).expect("ctx");
        recipe
            .prefill_step(&mut ctx, &gguf)
            .expect("initial prefill");
        assert_eq!(ctx.position(), 3);

        let forced = [10_u32, 11, 12, 13];
        let new_pos = execute_jump_forward_prefill(&recipe, &mut ctx, &gguf, &forced)
            .expect("atomic jump-forward burst");
        assert_eq!(new_pos, 7);
        assert_eq!(ctx.position(), 7);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "jero-arch-synthetic")]
    #[test]
    fn test_execute_jump_forward_prefill_bounds() {
        use jero_arch_synthetic::SyntheticRecipe;
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let bytes = create_synthetic_gguf_v3(
            "synthetic",
            8,
            &[("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..])],
        );
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic");
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 8).expect("kv arena");
        let recipe = SyntheticRecipe::new();

        let prompt = [1_u32, 2, 3, 4, 5];
        let mut logits = [0.0_f32; 16];
        let mut ctx =
            ExecutionContext::new(&prompt, 8, &mut logits, kv_arena.as_mut_slice()).expect("ctx");
        recipe
            .prefill_step(&mut ctx, &gguf)
            .expect("initial prefill");
        assert_eq!(ctx.position(), 5);

        // 4 tokens exceed remaining n_ctx=8 capacity -> must fail without advancing position.
        let err = execute_jump_forward_prefill(&recipe, &mut ctx, &gguf, &[6, 7, 8, 9])
            .expect_err("oversized jump-forward burst must fail");
        assert_eq!(
            err,
            K3Error::ContextLengthExceeded {
                requested_tokens: 9,
                max_context_tokens: 8,
            }
        );
        assert_eq!(ctx.position(), 5);

        // Empty forced_tokens rejected cleanly
        let empty_err = execute_jump_forward_prefill(&recipe, &mut ctx, &gguf, &[])
            .expect_err("empty forced_tokens must fail");
        assert!(matches!(empty_err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(ctx.position(), 5);
    }

    #[cfg(feature = "jero-arch-gemma")]
    #[test]
    fn test_fsm_logit_mask_and_jump_forward_checkpoint() {
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let bytes = create_synthetic_gguf_v3("functiongemma", 32, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse functiongemma");
        let mut recipe = crate::gemma::GemmaRecipe::new_functiongemma();
        recipe.num_layers = 1;
        recipe.hidden_dim = 8;
        recipe.ffn_dim = 16;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 4;
        recipe.vocab_size = 8;

        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("kv arena");
        let fsm =
            FsmDecisionConstraint::from_allowed_sequences(&[vec![1, 2, 4, 5], vec![1, 2, 6, 7]]);
        let prefix = fsm.forced_continuation(0).expect("forced JSON prefix");
        assert_eq!(prefix, &[1, 2]);

        let mut logits = [0.0_f32; 8];
        let mut ctx =
            ExecutionContext::new(&[0_u32], 32, &mut logits, kv_arena.as_mut_slice()).expect("ctx");
        recipe
            .prefill_step(&mut ctx, &gguf)
            .expect("prompt prefill");
        assert_eq!(ctx.position(), 1);

        let pos_after_prefix = execute_jump_forward_prefill(&recipe, &mut ctx, &gguf, prefix)
            .expect("jump-forward prefix burst");
        assert_eq!(pos_after_prefix, 3);

        // Mask logits to valid decision branch tokens [4, 6]
        let chosen = apply_fsm_logit_mask(&mut logits, &[4, 6]).expect("mask decision");
        assert!(chosen == 4 || chosen == 6);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);

        let _ = QuantType::F32;
    }
}
