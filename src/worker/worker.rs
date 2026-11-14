//! Worker execution engine, IPC event loop, and zero-copy forward pass dispatch.

use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::Instant;

use jero_core::ipc::arrow::wrap_tokens_as_uint32_array;
use jero_core::ipc::layout::{
    BLOCK_SIZE_TOKENS, KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN, TokenInputSlotHeader,
};
use jero_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, ForwardPassCompleted, IpcMessage, StatusCode,
    WorkerHeartbeatRequest, WorkerHeartbeatResponse, global_fory, recv_frame, send_frame,
    send_msg_frame,
};
use jero_core::ipc::shm::SharedMemoryArena;
use jero_core::{GgufFile, GgufValue, K3Error};
use jero_tokenizer::GgufTokenizer;
use zerocopy::FromBytes;

use crate::affinity::{
    ClusterPhasePolicy, ClusterPreset, ClusterSelection, CpuTopologyInspector, ResolvedAffinity,
    SystemCpuTopology, VlenDomain, WorkerPool,
};
use crate::chat_template::discover_stop_token_ids;
use crate::engine::{GenerationConfig, SinglePromptEngine, StopReason, sample_greedy_argmax};
use crate::kv::SingleSlotKvArena;

use super::affinity::{
    A100_CLUSTER_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, cores_from_mask, parse_affinity_spec,
    verify_cluster_capabilities_full,
};
use super::cli::WorkerConfig;
use super::error::WorkerError;
use super::weights::LockedWeights;

const SLAB_STOP_MAGIC: [u8; 4] = *b"K3S1";
const SLAB_STOP_KIND_NONE: u8 = 0;
const SLAB_STOP_KIND_EOS_TOKEN: u8 = 1;
const SLAB_STOP_KIND_MAX_TOKENS: u8 = 2;

/// Encodes an optional [`StopReason`] into [`OutputSlabHeader::_reserved`] without altering the
/// 64-byte shared-memory ABI (`#156`).
pub fn encode_slab_stop_reason(hdr: &mut OutputSlabHeader, stop_reason: Option<StopReason>) {
    hdr._reserved[0..4].copy_from_slice(&SLAB_STOP_MAGIC);
    match stop_reason {
        None => {
            hdr._reserved[4] = SLAB_STOP_KIND_NONE;
            hdr._reserved[8..12].copy_from_slice(&0_u32.to_le_bytes());
        }
        Some(StopReason::EosToken { token_id }) => {
            hdr._reserved[4] = SLAB_STOP_KIND_EOS_TOKEN;
            hdr._reserved[8..12].copy_from_slice(&token_id.to_le_bytes());
        }
        Some(StopReason::MaxTokens) => {
            hdr._reserved[4] = SLAB_STOP_KIND_MAX_TOKENS;
            hdr._reserved[8..12].copy_from_slice(&0_u32.to_le_bytes());
        }
    }
}

/// Decodes a [`StopReason`] previously written to [`OutputSlabHeader::_reserved`] by `jero-worker`.
#[must_use]
pub fn decode_slab_stop_reason(hdr: &OutputSlabHeader) -> Option<StopReason> {
    if hdr._reserved[0..4] != SLAB_STOP_MAGIC {
        return None;
    }
    match hdr._reserved[4] {
        SLAB_STOP_KIND_EOS_TOKEN => {
            let token_id = u32::from_le_bytes([
                hdr._reserved[8],
                hdr._reserved[9],
                hdr._reserved[10],
                hdr._reserved[11],
            ]);
            Some(StopReason::EosToken { token_id })
        }
        SLAB_STOP_KIND_MAX_TOKENS => Some(StopReason::MaxTokens),
        _ => None,
    }
}

fn discover_worker_stop_tokens(weights: &LockedWeights) -> Vec<u32> {
    if let Ok(gguf) = GgufFile::parse(weights.as_slice()) {
        if let Ok(tok) = GgufTokenizer::from_gguf(&gguf) {
            return discover_stop_token_ids(&tok);
        }
        let mut ids = Vec::new();
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let Some(id) = gguf.metadata_value(key).and_then(GgufValue::as_u32)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        return ids;
    }
    Vec::new()
}

/// Isolated compute worker execution engine.
#[derive(Debug)]
pub struct WorkerEngine {
    config: WorkerConfig,
    weights: LockedWeights,
    arena: Option<SharedMemoryArena>,
    kv_arena: Option<SingleSlotKvArena>,
    prompt_engine: SinglePromptEngine,
    pool: Option<WorkerPool>,
    stop_token_ids: Vec<u32>,
    attach_n_ctx: usize,
    last_stop_reason: Option<StopReason>,
    completed_passes: u64,
    is_busy: bool,
    compute_mask: u64,
    assigned_cores: Vec<usize>,
}

fn build_worker_pool(
    cluster: &str,
    compute_mask: u64,
    assigned_cores: &[usize],
) -> (ClusterPhasePolicy, Option<WorkerPool>) {
    let preset = if cluster.eq_ignore_ascii_case("a100") {
        ClusterPreset::A100
    } else {
        ClusterPreset::X100
    };
    let (vlen_domain, phase_policy, x100_cores, a100_cores) = match preset {
        ClusterPreset::X100 => (
            VlenDomain::Zvl256b,
            ClusterPhasePolicy::X100Only,
            assigned_cores.to_vec(),
            Vec::new(),
        ),
        ClusterPreset::A100 => (
            VlenDomain::Zvl1024b,
            ClusterPhasePolicy::A100Only,
            Vec::new(),
            assigned_cores.to_vec(),
        ),
    };

    let resolved = ResolvedAffinity {
        cluster: Some(preset),
        selection: Some(ClusterSelection::Single(preset)),
        vlen_domain,
        effective_mask: compute_mask,
        threads: assigned_cores.len().max(1),
        assigned_cores: assigned_cores.to_vec(),
        x100_cores,
        a100_cores,
    };

    if let Ok(pool) = WorkerPool::new(resolved) {
        return (phase_policy, Some(pool));
    }

    // If the host OS exposes fewer online CPUs than the requested default cluster mask
    // (e.g. a 4-core gateway node running with default X100 6-core mask), clamp pool threads
    // to currently online CPUs in that cluster while preserving logical assigned_cores reporting.
    let online_mask = SystemCpuTopology::new().online_cpu_mask().unwrap_or(0) & compute_mask;
    if online_mask != 0 {
        let online_cores = cores_from_mask(online_mask);
        let fallback_threads = assigned_cores.len().min(online_cores.len()).max(1);
        let fallback_cores = online_cores[..fallback_threads].to_vec();
        let (fx, fa) = match preset {
            ClusterPreset::X100 => (fallback_cores.clone(), Vec::new()),
            ClusterPreset::A100 => (Vec::new(), fallback_cores.clone()),
        };
        let fallback_affinity = ResolvedAffinity {
            cluster: Some(preset),
            selection: Some(ClusterSelection::Single(preset)),
            vlen_domain,
            effective_mask: online_mask,
            threads: fallback_threads,
            assigned_cores: fallback_cores,
            x100_cores: fx,
            a100_cores: fa,
        };
        if let Ok(pool) = WorkerPool::new(fallback_affinity) {
            return (phase_policy, Some(pool));
        }
    }

    (phase_policy, None)
}

impl WorkerEngine {
    /// Attempts to construct a new [`WorkerEngine`] with the given configuration and loaded weights.
    ///
    /// Validates cluster capabilities, thread bounds, affinity specifications, and initial KV arena
    /// capacity upfront (failing closed).
    ///
    /// # Errors
    /// Returns [`WorkerError`] if configuration, affinity, or hardware verification fails.
    pub fn try_new(config: WorkerConfig, weights: LockedWeights) -> Result<Self, WorkerError> {
        verify_cluster_capabilities_full(
            &config.cluster,
            config.affinity.as_deref(),
            config.threads,
            std::env::var("K3_SIMULATE_A100").ok().as_deref(),
        )?;

        let compute_mask = if let Some(spec) = &config.affinity {
            parse_affinity_spec(spec)?.1
        } else if config.cluster.eq_ignore_ascii_case("a100") {
            A100_CLUSTER_MASK
        } else {
            DEFAULT_X100_COMPUTE_CAP_MASK
        };

        let available_cores = cores_from_mask(compute_mask);
        let thread_count = config.threads.unwrap_or(available_cores.len()).max(1);
        let assigned_cores: Vec<usize> = if available_cores.is_empty() {
            vec![0]
        } else {
            (0..thread_count)
                .map(|i| available_cores[i % available_cores.len()])
                .collect()
        };

        let meta_ctx = weights.metadata().context_length();
        let initial_n_ctx = if meta_ctx > 0 { meta_ctx.min(64) } else { 64 };
        let initial_kv = weights.instantiate_kv_arena(initial_n_ctx)?;

        let stop_token_ids = discover_worker_stop_tokens(&weights);
        let (phase_policy, pool) =
            build_worker_pool(&config.cluster, compute_mask, &assigned_cores);
        let prompt_engine = SinglePromptEngine::new()
            .with_phase_policy(phase_policy)
            .with_generation_config(
                GenerationConfig::new().with_stop_token_ids(stop_token_ids.iter().copied()),
            );

        Ok(Self {
            config,
            weights,
            arena: None,
            kv_arena: Some(initial_kv),
            prompt_engine,
            pool,
            stop_token_ids,
            attach_n_ctx: initial_n_ctx,
            last_stop_reason: None,
            completed_passes: 0,
            is_busy: false,
            compute_mask,
            assigned_cores,
        })
    }

    /// Constructs a new [`WorkerEngine`] with the given configuration and loaded weights,
    /// spawning and pinning compute worker threads according to the assigned compute core mask.
    ///
    /// # Panics
    /// Panics if configuration or affinity validation fails.
    #[must_use]
    pub fn new(config: WorkerConfig, weights: LockedWeights) -> Self {
        Self::try_new(config, weights).expect("failed to construct WorkerEngine from config")
    }

    /// Returns the resolved compute core bitmask.
    #[must_use]
    pub const fn compute_mask(&self) -> u64 {
        self.compute_mask
    }

    /// Returns the list of assigned cores for all compute worker threads.
    #[must_use]
    pub fn assigned_cores(&self) -> &[usize] {
        &self.assigned_cores
    }

    /// Returns the number of compute worker threads.
    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.assigned_cores.len()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Returns a reference to the loaded weights.
    #[must_use]
    pub const fn weights(&self) -> &LockedWeights {
        &self.weights
    }

    /// Returns `true` if an arena is currently attached.
    #[must_use]
    pub const fn has_arena(&self) -> bool {
        self.arena.is_some()
    }

    /// Returns the discovered EOS/EOT stop token IDs enforced for IPC passes.
    #[must_use]
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    /// Replaces the stop token IDs enforced on subsequent IPC passes.
    pub fn set_stop_token_ids<I>(&mut self, stop_token_ids: I)
    where
        I: IntoIterator<Item = u32>,
    {
        let gen_cfg = GenerationConfig::new().with_stop_token_ids(stop_token_ids);
        self.stop_token_ids = gen_cfg.stop_token_ids.clone();
        self.prompt_engine.set_generation_config(gen_cfg);
    }

    /// Returns the [`StopReason`] recorded for the most recent `ForwardPassCommand`, if any.
    #[must_use]
    pub const fn last_stop_reason(&self) -> Option<StopReason> {
        self.last_stop_reason
    }

    /// Runs the worker event loop on the connected Unix domain stream socket.
    ///
    /// # Errors
    /// Returns [`WorkerError`] if a protocol error occurs.
    pub fn run_event_loop(&mut self, stream: UnixStream) -> Result<(), WorkerError> {
        let sock = stream.as_raw_fd();
        let fory = global_fory();

        loop {
            // Receive fixed 12-byte header + payload + optional SCM_RIGHTS FD
            let (header, payload, opt_fd) = match recv_frame(sock) {
                Ok(res) => res,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if self.config.verbose {
                        eprintln!("[k3-worker] supervisor disconnected (EOF), exiting cleanly");
                    }
                    return Ok(());
                }
                Err(e) => {
                    return Err(WorkerError::ProtocolError(format!(
                        "recv_frame failed: {e}"
                    )));
                }
            };

            match header.type_id {
                // Type ID 201: AttachArenaCommand
                201 => {
                    let cmd: AttachArenaCommand = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode AttachArenaCommand: {e}"
                        ))
                    })?;

                    let fd = opt_fd.ok_or_else(|| {
                        WorkerError::ProtocolError(
                            "AttachArenaCommand specified FLAG_FD_ATTACHED but no SCM_RIGHTS FD was delivered".to_owned(),
                        )
                    })?;

                    if self.config.verbose {
                        eprintln!(
                            "[k3-worker] received AttachArenaCommand: id={}, bytes={}, n_ctx={}, blocks={}",
                            cmd.arena_id, cmd.arena_bytes, cmd.n_ctx, cmd.max_blocks
                        );
                    }

                    // Map shared memory arena with read/write permissions
                    let len = usize::try_from(cmd.arena_bytes).map_err(|_| {
                        WorkerError::ResourceExhausted("arena_bytes overflowed usize".to_owned())
                    })?;

                    let arena =
                        SharedMemoryArena::from_existing_fd(fd, len, "worker_arena", false)?;
                    let hdr = *arena.header()?;
                    let attach_n_ctx = usize::try_from(cmd.n_ctx.max(hdr.n_ctx))
                        .unwrap_or(64)
                        .max(1);
                    let kv_arena = self.weights.instantiate_kv_arena(attach_n_ctx)?;
                    self.kv_arena = Some(kv_arena);
                    self.attach_n_ctx = attach_n_ctx;
                    self.arena = Some(arena);

                    // Send response frame (Type ID 206) with StatusCode::Success
                    let status_payload = (StatusCode::Success as u32).to_le_bytes();
                    send_frame(sock, 206, &status_payload, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send AttachArena response: {e}"
                        ))
                    })?;
                }

                // Type ID 202: ForwardPassCommand
                202 => {
                    self.is_busy = true;
                    let start_time = Instant::now();

                    let cmd: ForwardPassCommand = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode ForwardPassCommand: {e}"
                        ))
                    })?;

                    if self.config.verbose {
                        eprintln!(
                            "[k3-worker] executing ForwardPassCommand: req={}, seq={}, offset={}, count={}, prefill={}",
                            cmd.request_id,
                            cmd.sequence_id,
                            cmd.token_offset,
                            cmd.token_count,
                            cmd.is_prefill
                        );
                    }

                    let arena = self.arena.as_mut().ok_or_else(|| {
                        WorkerError::ProtocolError(
                            "received ForwardPassCommand before AttachArenaCommand".to_owned(),
                        )
                    })?;
                    let kv_arena = self.kv_arena.as_mut().ok_or_else(|| {
                        WorkerError::Internal(
                            "worker KV arena uninitialized despite attached shared arena"
                                .to_owned(),
                        )
                    })?;

                    let hdr = *arena.header()?;
                    let model_vocab = self.weights.metadata().vocab_size().unwrap_or(0);
                    let slab_vocab_size = if hdr.vocab_size > 0 {
                        usize::try_from(hdr.vocab_size).unwrap_or(model_vocab.max(1))
                    } else {
                        model_vocab.max(1)
                    };
                    let eval_vocab = model_vocab.max(slab_vocab_size).max(1);

                    // Zero-copy read tokens from TokenInputSlot using Apache Arrow UInt32Array view
                    let tokens_slot_offset =
                        usize::try_from(hdr.token_slots_offset).map_err(|e| {
                            WorkerError::Internal(format!("invalid token_slots_offset: {e}"))
                        })?;

                    // Decode slot header (16 bytes)
                    let slot_hdr_slice = arena.slice(tokens_slot_offset, 16)?;
                    let slot_hdr =
                        TokenInputSlotHeader::ref_from_bytes(slot_hdr_slice).map_err(|e| {
                            WorkerError::ProtocolError(format!(
                                "failed to decode TokenInputSlotHeader: {e}"
                            ))
                        })?;

                    let token_count = if cmd.token_count > 0 {
                        cmd.token_count as usize
                    } else {
                        slot_hdr.token_len as usize
                    };
                    if token_count == 0 {
                        return Err(WorkerError::TensorBoundsError(
                            "ForwardPassCommand received zero input tokens".to_owned(),
                        ));
                    }

                    let tokens_data_offset = tokens_slot_offset + 16;
                    let tokens_slice = arena.slice(tokens_data_offset, token_count * 4)?;

                    // Wrap into Apache Arrow UInt32Array zero-copy view
                    // SAFETY: tokens_slice is verified within arena bounds and memory is valid.
                    let arrow_tokens = unsafe {
                        wrap_tokens_as_uint32_array(
                            Arc::new(()),
                            tokens_slice.as_ptr().cast::<u32>(),
                            token_count,
                        )?
                    };

                    let input_tokens: Vec<u32> = arrow_tokens.values().to_vec();
                    let raw_start_pos = if slot_hdr.start_position != 0 {
                        slot_hdr.start_position
                    } else {
                        cmd.token_offset
                    };
                    let start_pos = usize::try_from(raw_start_pos).unwrap_or(usize::MAX);

                    let step_result = if cmd.is_prefill {
                        self.prompt_engine.prefill_step_with_pool(
                            self.weights.recipe(),
                            &self.weights,
                            kv_arena,
                            self.pool.as_ref(),
                            &input_tokens,
                            start_pos,
                            eval_vocab,
                        )
                    } else {
                        let mut last_step = None;
                        let mut step_err = None;
                        for (idx, &tok) in input_tokens.iter().enumerate() {
                            match self.prompt_engine.decode_step_with_pool(
                                self.weights.recipe(),
                                &self.weights,
                                kv_arena,
                                self.pool.as_ref(),
                                tok,
                                start_pos.saturating_add(idx),
                                eval_vocab,
                            ) {
                                Ok(res) => last_step = Some(res),
                                Err(e) => {
                                    step_err = Some(e);
                                    break;
                                }
                            }
                        }
                        match step_err {
                            Some(e) => Err(e),
                            None => last_step.ok_or_else(|| K3Error::TensorBoundsError {
                                tensor_name: "token_input_slot".to_owned(),
                                offset: start_pos,
                                byte_len: 0,
                                buffer_len: 0,
                                reason: "empty token input in decode ForwardPassCommand".to_owned(),
                            }),
                        }
                    };

                    let (logits, end_pos) = match step_result {
                        Ok((logits, _engine_next_tok, end_pos)) => (logits, end_pos),
                        Err(k3_err) => {
                            let reply = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
                                request_id: cmd.request_id,
                                sequence_id: cmd.sequence_id,
                                committed_tokens: 0,
                                prefill_latency_ms: 0.0,
                                decode_latency_ms: 0.0,
                                next_token: 0,
                                status: StatusCode::from_k3_error(&k3_err),
                                error_message: Some(k3_err.to_string()),
                            });
                            send_msg_frame(&stream, &reply, None).map_err(|e| {
                                WorkerError::ProtocolError(format!(
                                    "failed to send error ForwardPassCompleted: {e}"
                                ))
                            })?;
                            self.is_busy = false;
                            continue;
                        }
                    };

                    let sample_len = if model_vocab > 0 {
                        model_vocab.min(slab_vocab_size).min(logits.len())
                    } else {
                        slab_vocab_size.min(logits.len())
                    };
                    let next_token = sample_greedy_argmax(&logits[..sample_len]).unwrap_or(0);

                    // Deterministic in-place KvBlockMeta checkpointing per forward pass
                    let blocks_to_update = if cmd.block_indices.is_empty() {
                        vec![0_u32]
                    } else {
                        cmd.block_indices.clone()
                    };

                    let total_tokens = cmd.token_offset + cmd.token_count;
                    let rng_state = if cmd.is_prefill {
                        [987654321, 0, 0, 0]
                    } else {
                        let mut st = {
                            let first_block = arena.block_meta(blocks_to_update[0] as usize)?;
                            if st_is_zero(&first_block.sampler_rng_state) {
                                [42, 0, 0, 0]
                            } else {
                                first_block.sampler_rng_state
                            }
                        };
                        step_xoshiro256ss(&mut st);
                        st
                    };

                    for (seq_idx, &b_idx) in blocks_to_update.iter().enumerate() {
                        let block = arena.block_meta_mut(b_idx as usize)?;
                        let committed = if cmd.is_prefill && blocks_to_update.len() <= 1 {
                            cmd.token_count.min(BLOCK_SIZE_TOKENS)
                        } else {
                            let block_start = if blocks_to_update.len() <= 1 {
                                (cmd.token_offset / BLOCK_SIZE_TOKENS) * BLOCK_SIZE_TOKENS
                            } else {
                                (seq_idx as u32) * BLOCK_SIZE_TOKENS
                            };
                            if total_tokens >= block_start + BLOCK_SIZE_TOKENS {
                                BLOCK_SIZE_TOKENS
                            } else {
                                total_tokens
                                    .saturating_sub(block_start)
                                    .min(BLOCK_SIZE_TOKENS)
                            }
                        };
                        block.set_committed_tokens(committed);
                        block.set_tag(KvBlockTag::Persist);
                        block.sampler_rng_state = rng_state;
                    }

                    // Write output logits directly in shared memory OutputSlab
                    let slab_offset = usize::try_from(hdr.output_slabs_offset).map_err(|e| {
                        WorkerError::Internal(format!("invalid output_slabs_offset: {e}"))
                    })?;

                    // OutputSlabHeader (64 bytes aligned)
                    let slab_slice = arena.slice_mut(slab_offset, 64)?;
                    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to obtain mutable OutputSlabHeader: {e}"
                        ))
                    })?;
                    slab_hdr.request_id = cmd.request_id;
                    slab_hdr.vocab_size = slab_vocab_size as u32;
                    slab_hdr.position =
                        u32::try_from(end_pos).unwrap_or(cmd.token_offset + cmd.token_count);
                    slab_hdr.status = SLAB_STATUS_WRITTEN;
                    let step_stop_reason = if self.stop_token_ids.contains(&next_token) {
                        Some(StopReason::EosToken {
                            token_id: next_token,
                        })
                    } else if end_pos >= self.attach_n_ctx {
                        Some(StopReason::MaxTokens)
                    } else {
                        None
                    };
                    self.last_stop_reason = step_stop_reason;
                    encode_slab_stop_reason(slab_hdr, step_stop_reason);

                    // Logits float array starting at slab_offset + 64
                    let logits_data_offset = slab_offset + 64;
                    let logits_slice = arena.slice_mut(logits_data_offset, slab_vocab_size * 4)?;
                    for i in 0..slab_vocab_size {
                        let val = logits.get(i).copied().unwrap_or(0.0);
                        logits_slice[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
                    }

                    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                    let (prefill_ms, decode_ms) = if cmd.is_prefill {
                        (elapsed_ms.max(0.01), 0.0)
                    } else {
                        (0.0, elapsed_ms.max(0.01))
                    };

                    // Send ForwardPassCompleted (Type ID 203) back over UDS
                    // Zero bulk tensor payload bytes transferred over socket!
                    let reply = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
                        request_id: cmd.request_id,
                        sequence_id: cmd.sequence_id,
                        committed_tokens: cmd.token_count,
                        prefill_latency_ms: prefill_ms,
                        decode_latency_ms: decode_ms,
                        next_token,
                        status: StatusCode::Success,
                        error_message: None,
                    });

                    send_msg_frame(&stream, &reply, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send ForwardPassCompleted: {e}"
                        ))
                    })?;

                    self.completed_passes += 1;
                    self.is_busy = false;
                }

                // Type ID 204: WorkerHeartbeatRequest
                204 => {
                    let req: WorkerHeartbeatRequest = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode WorkerHeartbeatRequest: {e}"
                        ))
                    })?;

                    let resp = IpcMessage::WorkerHeartbeatResponse(WorkerHeartbeatResponse {
                        timestamp_ns: req.timestamp_ns,
                        is_busy: self.is_busy,
                        completed_passes: self.completed_passes,
                    });

                    send_msg_frame(&stream, &resp, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send WorkerHeartbeatResponse: {e}"
                        ))
                    })?;
                }

                // Type ID 206: ShutdownCommand
                206 => {
                    if self.config.verbose {
                        eprintln!("[k3-worker] received ShutdownCommand, terminating cleanly");
                    }
                    return Ok(());
                }

                other => {
                    return Err(WorkerError::ProtocolError(format!(
                        "unexpected IPC message type ID: {other}"
                    )));
                }
            }
        }
    }
}

#[inline]
fn step_xoshiro256ss(state: &mut [u64; 4]) -> u64 {
    let result = (state[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
    let t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = state[3].rotate_left(45);
    result
}

#[inline]
fn st_is_zero(state: &[u64; 4]) -> bool {
    state.iter().all(|&x| x == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_slab_stop_reason_roundtrip_preserves_abi() {
        #[repr(C, align(64))]
        struct AlignedSlab([u8; 64]);

        let mut buf = AlignedSlab([0_u8; 64]);
        let hdr = OutputSlabHeader::mut_from_bytes(&mut buf.0).expect("64B slab header");
        assert_eq!(decode_slab_stop_reason(hdr), None);

        encode_slab_stop_reason(hdr, Some(StopReason::EosToken { token_id: 128_009 }));
        assert_eq!(
            decode_slab_stop_reason(hdr),
            Some(StopReason::EosToken { token_id: 128_009 })
        );

        encode_slab_stop_reason(hdr, Some(StopReason::MaxTokens));
        assert_eq!(decode_slab_stop_reason(hdr), Some(StopReason::MaxTokens));

        encode_slab_stop_reason(hdr, None);
        assert_eq!(decode_slab_stop_reason(hdr), None);
    }
}
