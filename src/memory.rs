//! Pre-flight memory footprint calculation and physical RAM admission checks.
//!
//! Enforces deterministic upfront admission checking against available physical memory
//! (`MemAvailable` in `/proc/meminfo`) and process lockable limits (`RLIMIT_MEMLOCK`),
//! refusing runs before forward passes begin (ADR-027, ADR-037, US-01 Scenario 3).

use std::fs;
use std::io::{BufRead, BufReader};

use jero_core::{DoryKvCacheMode, K3Error, ModelMetadata, align_up};

/// Default sliding-window attention span (`dory.attention.sliding_window`) for `Dory1.0-4B`.
pub const DORY_DEFAULT_SLIDING_WINDOW: usize = 1024;

/// Estimates the raw unaligned single-slot KV cache footprint in bytes for a Dory looped nGPT model,
/// accounting for disaggregated sub-layers (`'-'` MLP = 0 bytes, `'S'` SWA bounded by `min(n_ctx, 1024)`,
/// `'*'` full attention = `n_ctx`) and `recurrent_kv_cache_mode` (`"per_loop"` vs `"last_loop"`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx == 0`, `loop_override` is outside `1..=4`,
/// `mode_override` is invalid, or `dory.layer_pattern` contains unrecognized characters.
/// Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds `metadata.context_length()`.
#[expect(
    clippy::too_many_lines,
    reason = "computes disaggregated SWA/Full/MLP KV cache bytes across 3 Dory loop segments"
)]
pub fn estimate_dory_kv_cache_bytes(
    metadata: &ModelMetadata,
    n_ctx: usize,
    loop_override: Option<u32>,
    mode_override: Option<&str>,
) -> Result<usize, K3Error> {
    if n_ctx == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: metadata.architecture().to_owned(),
            reason: "requested context length n_ctx must be greater than zero".to_owned(),
        });
    }
    if metadata.context_length() > 0 && n_ctx > metadata.context_length() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: n_ctx,
            max_context_tokens: metadata.context_length(),
        });
    }

    let (n_input, n_recurrent, n_output, default_loops, default_mode, pattern) =
        if let Some(dory) = metadata.dory() {
            (
                dory.input_layer_count(),
                dory.recurrent_layer_count(),
                dory.output_layer_count(),
                dory.num_loops(),
                dory.recurrent_kv_cache_mode(),
                dory.layer_pattern(),
            )
        } else {
            (
                16_u32,
                40_u32,
                16_u32,
                2_u32,
                DoryKvCacheMode::PerLoop,
                "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-",
            )
        };

    let n_loops = loop_override.unwrap_or(default_loops);
    if n_loops == 0 || n_loops > 4 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: format!("dory recurrent loop count must be in 1..=4, got {n_loops}"),
        });
    }

    let kv_mode = match mode_override {
        Some(raw) => DoryKvCacheMode::parse_mode(raw)?,
        None => default_mode,
    };

    let n_physical = (n_input + n_recurrent + n_output) as usize;
    if pattern.len() < n_physical {
        return Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: format!(
                "dory.layer_pattern length ({}) is shorter than physical layer count ({n_physical})",
                pattern.len()
            ),
        });
    }

    let head_count_kv = metadata.head_count_kv().max(1);
    let head_dim = if metadata.embedding_dim() == 2560 && metadata.head_count() == 16 {
        256_usize
    } else if metadata.head_count() > 0 {
        (metadata.embedding_dim() / metadata.head_count()).max(1)
    } else {
        256_usize
    };

    let elements_per_kv_token = 2_usize
        .checked_mul(head_count_kv)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "dory kv elements per token overflowed usize".to_owned(),
        })?;

    let full_tokens = n_ctx.div_ceil(KV_BLOCK_SIZE) * KV_BLOCK_SIZE;
    let swa_tokens = n_ctx
        .min(DORY_DEFAULT_SLIDING_WINDOW)
        .div_ceil(KV_BLOCK_SIZE)
        * KV_BLOCK_SIZE;

    let mut total_elements = 0_usize;
    for (p, ch) in pattern.chars().take(n_physical).enumerate() {
        let p_u32 = u32::try_from(p).unwrap_or(0);
        let is_recurrent = p_u32 >= n_input && p_u32 < (n_input + n_recurrent);
        let loop_multiplier = if is_recurrent && kv_mode == DoryKvCacheMode::PerLoop {
            n_loops as usize
        } else {
            1_usize
        };

        let layer_tokens = match ch {
            '-' => 0_usize,
            'S' => swa_tokens,
            '*' => full_tokens,
            other => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: format!(
                        "invalid sub-layer character '{other}' at physical layer {p} in dory.layer_pattern"
                    ),
                });
            }
        };

        let layer_elements = loop_multiplier
            .checked_mul(layer_tokens)
            .and_then(|v| v.checked_mul(elements_per_kv_token))
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "dory layer kv elements overflowed usize".to_owned(),
            })?;
        total_elements = total_elements.checked_add(layer_elements).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "dory total kv elements overflowed usize".to_owned(),
            }
        })?;
    }

    total_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "dory raw kv bytes overflowed usize".to_owned(),
        })
}

/// Per-layer KV cache retention policy for hybrid local sliding-window and global full-context
/// architectures (such as `gemma2`, `gemma3`, and `gemma4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKvPolicy {
    /// Retains all tokens up to `n_ctx`.
    GlobalFullContext,
    /// Retains a bounded cyclic ring buffer of at most `min(n_ctx, window_tokens)` tokens.
    SlidingWindow {
        /// Sliding window token capacity for this layer.
        window_tokens: usize,
    },
}

/// Builds the canonical per-layer KV retention policies for Gemma architectures:
/// - `gemma2`: 1:1 alternating (`layer_idx % 2 == 0` sliding window, odd global)
/// - `gemma3` / `gemma4`: 5:1 interleaved (`(layer_idx + 1) % 6 == 0` global, all others local)
#[must_use]
pub fn gemma_layer_kv_policies(
    architecture: &str,
    layer_count: usize,
    sliding_window: Option<usize>,
) -> Vec<LayerKvPolicy> {
    let default_window = match architecture {
        "gemma2" => 4096,
        _ => 1024,
    };
    let window_tokens = sliding_window.unwrap_or(default_window).max(1);

    (0..layer_count)
        .map(|layer_idx| {
            let is_global = if architecture == "gemma2" {
                layer_idx % 2 == 1
            } else {
                (layer_idx + 1) % 6 == 0
            };
            if is_global {
                LayerKvPolicy::GlobalFullContext
            } else {
                LayerKvPolicy::SlidingWindow { window_tokens }
            }
        })
        .collect()
}

/// Disambiguates per-layer state backing across hybrid SSM/short-conv/attention architectures (`#59`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    /// Standard transformer GQA attention layer backed by paged KV cache blocks.
    GqaAttention {
        /// Compact attention-only KV layer slot index (`0..num_attention_layers`).
        kv_layer_slot: usize,
    },
    /// Mamba-2 Selective State-Space Duality (SSD) layer backed by O(1) ping-pong recurrent slots.
    Mamba2Ssd {
        /// Compact recurrent layer slot index (`0..num_recurrent_layers`).
        recurrent_layer_slot: usize,
    },
    /// `LiquidAI` LFM2 gated causal short-convolution layer backed by O(1) ping-pong recurrent slots.
    Lfm2ShortConv {
        /// Compact recurrent layer slot index (`0..num_recurrent_layers`).
        recurrent_layer_slot: usize,
    },
}

/// Computes the combined paged-KV + O(1) double-buffered ping-pong recurrent arena byte budget
/// for a hybrid [`HybridLayerKind`] schedule (`#62`).
///
/// Only [`HybridLayerKind::GqaAttention`] layers scale with `kv_bytes_per_attn_layer`, while
/// [`HybridLayerKind::Mamba2Ssd`] and [`HybridLayerKind::Lfm2ShortConv`] layers reserve fixed O(1)
/// dual-bank ping-pong slots sized by `state_bytes_per_ssm_layer` plus the checkpoint header.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx == 0`, or [`K3Error::TensorBoundsError`] if
/// byte accumulation overflows `usize`.
pub fn estimate_hybrid_ssm_arena_bytes(
    schedule: &[HybridLayerKind],
    n_ctx: usize,
    kv_bytes_per_attn_layer: usize,
    state_bytes_per_ssm_layer: usize,
) -> Result<usize, K3Error> {
    if n_ctx == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "hybrid".to_owned(),
            reason: "requested context length n_ctx must be greater than zero".to_owned(),
        });
    }

    let mut attn_layers = 0_usize;
    let mut recurrent_layers = 0_usize;
    for kind in schedule {
        match kind {
            HybridLayerKind::GqaAttention { .. } => attn_layers += 1,
            HybridLayerKind::Mamba2Ssd { .. } | HybridLayerKind::Lfm2ShortConv { .. } => {
                recurrent_layers += 1;
            }
        }
    }

    let overflow = |reason: &str| K3Error::TensorBoundsError {
        tensor_name: "hybrid_ssm_arena".to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: usize::MAX,
        reason: reason.to_owned(),
    };

    let kv_total = attn_layers
        .checked_mul(kv_bytes_per_attn_layer)
        .ok_or_else(|| overflow("hybrid KV bytes overflowed usize"))?;

    let recurrent_total = if recurrent_layers == 0 {
        0
    } else {
        let header_bytes = size_of::<jero_core::RecurrentStateSlotHeader>();
        let banks_bytes = recurrent_layers
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(state_bytes_per_ssm_layer))
            .ok_or_else(|| overflow("hybrid dual-bank recurrent bytes overflowed usize"))?;
        header_bytes
            .checked_add(banks_bytes)
            .ok_or_else(|| overflow("hybrid recurrent region bytes overflowed usize"))?
    };

    kv_total
        .checked_add(recurrent_total)
        .ok_or_else(|| overflow("combined hybrid arena bytes overflowed usize"))
}

/// Returns `true` when `arch` is a pure linear-time SSM/RNN architecture (`mamba`, `mamba2`,
/// `rwkv6`, `rwkv7`) whose persistent state size is strictly O(1) independent of `n_ctx`.
#[must_use]
pub(crate) const fn is_pure_recurrent_arch(arch: &str) -> bool {
    matches!(arch.as_bytes(), b"mamba" | b"mamba2" | b"rwkv6" | b"rwkv7")
}

/// Returns `true` when `arch` interleaves attention layers with SSM/short-conv layers (`jamba`,
/// `zamba2`, `granitehybrid`, `lfm2`, `lfm2moe`).
#[must_use]
pub(crate) const fn is_hybrid_ssm_arch(arch: &str) -> bool {
    matches!(
        arch.as_bytes(),
        b"jamba" | b"zamba2" | b"granitehybrid" | b"lfm2" | b"lfm2moe"
    )
}

/// Computes single-slot O(1) dual-bank ping-pong recurrent state bytes for pure SSM/RWKV checkpoints.
pub(crate) fn estimate_pure_recurrent_raw_bytes_per_slot(
    metadata: &ModelMetadata,
) -> Result<usize, K3Error> {
    let arch = metadata.architecture();
    let num_layers = metadata.layer_count().max(1);
    let overflow = |reason: &str| K3Error::TensorBoundsError {
        tensor_name: "recurrent_state_arena".to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: usize::MAX,
        reason: reason.to_owned(),
    };

    let (conv_bytes_per_layer, ssm_bytes_per_layer) = if matches!(arch, "rwkv6" | "rwkv7") {
        let dim = metadata.embedding_dim().max(1);
        let head_size = metadata.wkv_head_size().unwrap_or(64).min(dim).max(1);
        let n_heads = (dim / head_size).max(1);
        let conv_bytes = 2_usize
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("rwkv token-shift state bytes overflowed usize"))?;
        let ssm_bytes = n_heads
            .checked_mul(head_size)
            .and_then(|v| v.checked_mul(head_size))
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("rwkv wkv state bytes overflowed usize"))?;
        (conv_bytes, ssm_bytes)
    } else {
        let d_inner = metadata
            .ssm_inner_size()
            .unwrap_or_else(|| metadata.embedding_dim().saturating_mul(2))
            .max(1);
        let d_conv = metadata.ssm_conv_kernel().unwrap_or(4).max(1);
        let d_state = metadata.ssm_state_size().unwrap_or(16).max(1);
        let conv_bytes = d_inner
            .checked_mul(d_conv)
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("mamba2 conv state bytes overflowed usize"))?;
        let ssm_bytes = d_inner
            .checked_mul(d_state)
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("mamba2 ssm state bytes overflowed usize"))?;
        (conv_bytes, ssm_bytes)
    };

    let per_layer = conv_bytes_per_layer
        .checked_add(ssm_bytes_per_layer)
        .ok_or_else(|| overflow("recurrent per-layer state bytes overflowed usize"))?;
    let bank_bytes = num_layers
        .checked_mul(per_layer)
        .ok_or_else(|| overflow("recurrent bank bytes overflowed usize"))?;
    let header_bytes = size_of::<jero_core::RecurrentStateSlotHeader>();
    header_bytes
        .checked_add(
            bank_bytes
                .checked_mul(2)
                .ok_or_else(|| overflow("dual-bank recurrent bytes overflowed usize"))?,
        )
        .ok_or_else(|| overflow("total recurrent state bytes overflowed usize"))
}

/// Counts `(attention_layers, recurrent_layers)` across a hybrid checkpoint using GGUF per-layer
/// `head_count_kv` arrays or layer tensor names (`blk.{i}.attn_q.weight` vs SSM/shortconv).
#[must_use]
pub(crate) fn count_hybrid_schedule_layers(metadata: &ModelMetadata) -> (usize, usize) {
    let total_layers = metadata.layer_count().max(1);
    let per_layer_kv = metadata.head_count_kv_per_layer();
    if !per_layer_kv.is_empty() {
        let mut attn = 0_usize;
        let mut rec = 0_usize;
        for idx in 0..total_layers {
            let kv_heads = per_layer_kv.get(idx).copied().unwrap_or(0);
            if kv_heads > 0 {
                attn += 1;
            } else {
                rec += 1;
            }
        }
        return (attn, rec);
    }

    let tensors = metadata.tensors();
    if !tensors.is_empty() {
        let mut attn = 0_usize;
        let mut rec = 0_usize;
        let mut classified = false;
        for idx in 0..total_layers {
            let has_attn = metadata
                .find_tensor(&format!("blk.{idx}.attn_q.weight"))
                .is_some()
                || metadata
                    .find_tensor(&format!("blk.{idx}.attn_k.weight"))
                    .is_some();
            let has_rec = metadata
                .find_tensor(&format!("blk.{idx}.ssm_in.weight"))
                .is_some()
                || metadata
                    .find_tensor(&format!("blk.{idx}.ssm_out.weight"))
                    .is_some()
                || metadata
                    .find_tensor(&format!("blk.{idx}.shortconv.conv.weight"))
                    .is_some()
                || metadata
                    .find_tensor(&format!("blk.{idx}.shortconv.in_proj.weight"))
                    .is_some()
                || metadata
                    .find_tensor(&format!("blk.{idx}.shortconv_conv.weight"))
                    .is_some();
            if has_attn {
                attn += 1;
                classified = true;
            } else if has_rec {
                rec += 1;
                classified = true;
            }
        }
        if classified {
            return (attn, rec);
        }
    }

    (total_layers, 0)
}

/// Estimates raw per-layer recurrent state bytes for a hybrid model's SSM or shortconv layers.
pub(crate) fn estimate_hybrid_recurrent_bytes_per_layer(
    metadata: &ModelMetadata,
) -> Result<usize, K3Error> {
    let overflow = |reason: &str| K3Error::TensorBoundsError {
        tensor_name: "hybrid_recurrent_state".to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: usize::MAX,
        reason: reason.to_owned(),
    };
    if matches!(metadata.architecture(), "lfm2" | "lfm2moe") {
        let dim = metadata.embedding_dim().max(1);
        let l_cache = metadata.shortconv_l_cache().unwrap_or(3).max(1);
        dim.checked_mul(l_cache)
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("lfm2 shortconv state bytes overflowed usize"))
    } else {
        let d_inner = metadata
            .ssm_inner_size()
            .unwrap_or_else(|| metadata.embedding_dim().saturating_mul(2))
            .max(1);
        let d_conv = metadata.ssm_conv_kernel().unwrap_or(4).max(1);
        let d_state = metadata.ssm_state_size().unwrap_or(16).max(1);
        d_inner
            .checked_mul(d_conv.saturating_add(d_state))
            .and_then(|v| v.checked_mul(size_of::<f32>()))
            .ok_or_else(|| overflow("hybrid ssm state bytes overflowed usize"))
    }
}

/// Resolves effective `(head_count_kv, head_dim)` for standard/hybrid KV arena sizing.
#[must_use]
pub(crate) fn infer_kv_head_geom(metadata: &ModelMetadata) -> (usize, usize) {
    let head_count_kv = metadata
        .head_count_kv_per_layer()
        .iter()
        .copied()
        .filter(|&v| v > 0)
        .max()
        .unwrap_or_else(|| metadata.head_count_kv().max(1));
    let head_dim = metadata
        .key_length()
        .unwrap_or_else(|| {
            if metadata.head_count() > 0 {
                metadata.embedding_dim() / metadata.head_count()
            } else {
                metadata.embedding_dim()
            }
        })
        .max(1);
    (head_count_kv, head_dim)
}

/// Estimates raw KV + O(1) recurrent arena bytes for a single slot of a hybrid SSM/attention model.
pub(crate) fn estimate_hybrid_raw_bytes_per_slot(
    metadata: &ModelMetadata,
    n_ctx: usize,
    blocks_per_slot: usize,
) -> Result<usize, K3Error> {
    let (attn_layers, rec_layers) = count_hybrid_schedule_layers(metadata);
    let (head_count_kv, head_dim) = infer_kv_head_geom(metadata);
    let kv_tokens =
        blocks_per_slot
            .checked_mul(KV_BLOCK_SIZE)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "hybrid kv tokens overflowed usize".to_owned(),
            })?;
    let kv_bytes_per_attn_layer = 2_usize
        .checked_mul(head_count_kv)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(kv_tokens))
        .and_then(|v| v.checked_mul(size_of::<f32>()))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "hybrid kv per-layer bytes overflowed usize".to_owned(),
        })?;
    let state_bytes_per_ssm = if rec_layers == 0 {
        0
    } else {
        estimate_hybrid_recurrent_bytes_per_layer(metadata)?
    };
    let mut synthetic_schedule = Vec::with_capacity(attn_layers + rec_layers);
    for slot in 0..attn_layers {
        synthetic_schedule.push(HybridLayerKind::GqaAttention {
            kv_layer_slot: slot,
        });
    }
    for slot in 0..rec_layers {
        if matches!(metadata.architecture(), "lfm2" | "lfm2moe") {
            synthetic_schedule.push(HybridLayerKind::Lfm2ShortConv {
                recurrent_layer_slot: slot,
            });
        } else {
            synthetic_schedule.push(HybridLayerKind::Mamba2Ssd {
                recurrent_layer_slot: slot,
            });
        }
    }
    estimate_hybrid_ssm_arena_bytes(
        &synthetic_schedule,
        n_ctx,
        kv_bytes_per_attn_layer,
        state_bytes_per_ssm,
    )
}

/// Estimates exact raw KV cache byte requirements across heterogeneous [`LayerKvPolicy`] layers,
/// bounding each local sliding-window layer to `min(n_ctx, window_tokens)` tokens.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx == 0`, `head_count_kv == 0`, `head_dim == 0`,
/// `policies` is empty, or any `SlidingWindow { window_tokens }` has zero window capacity.
/// Returns [`K3Error::TensorBoundsError`] on `usize` overflow.
pub fn estimate_hybrid_kv_cache_bytes(
    n_ctx: usize,
    head_count_kv: usize,
    head_dim: usize,
    policies: &[LayerKvPolicy],
) -> Result<usize, K3Error> {
    if n_ctx == 0 || head_count_kv == 0 || head_dim == 0 || policies.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gemma".to_owned(),
            reason: format!(
                "invalid hybrid KV geometry: n_ctx={n_ctx}, head_count_kv={head_count_kv}, head_dim={head_dim}, layers={}",
                policies.len()
            ),
        });
    }

    let kv_bytes_per_token_per_layer = 2usize
        .checked_mul(head_count_kv)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(size_of::<f32>()))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "hybrid KV per-layer token byte stride overflowed usize".to_owned(),
        })?;

    let mut total_bytes = 0usize;
    for policy in policies {
        let cap_tokens = match *policy {
            LayerKvPolicy::GlobalFullContext => n_ctx,
            LayerKvPolicy::SlidingWindow { window_tokens } => {
                if window_tokens == 0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "gemma".to_owned(),
                        reason: "sliding-window layer policy must specify window_tokens > 0"
                            .to_owned(),
                    });
                }
                n_ctx.min(window_tokens)
            }
        };
        let layer_bytes = cap_tokens
            .checked_mul(kv_bytes_per_token_per_layer)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "hybrid KV layer byte capacity overflowed usize".to_owned(),
            })?;
        total_bytes =
            total_bytes
                .checked_add(layer_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: "kv_cache".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: usize::MAX,
                    reason: "hybrid KV total byte capacity overflowed usize".to_owned(),
                })?;
    }

    Ok(total_bytes)
}

/// Default system page size fallback in bytes if `sysconf` query fails.
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Tokens per KV block in the paged KV cache arena (ADR-004, `kv-allocator.md`).
pub const KV_BLOCK_SIZE: usize = 16;

/// Returns the system memory page size in bytes.
#[must_use]
pub fn system_page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is a standard POSIX query with no side effects.
    let val = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if val > 0 {
        usize::try_from(val).unwrap_or(DEFAULT_PAGE_SIZE)
    } else {
        DEFAULT_PAGE_SIZE
    }
}

/// Structured page-aligned memory footprint for a model placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryFootprint {
    /// Exact page-aligned bytes required for mapped model weights.
    pub weight_bytes: u64,
    /// Exact page-aligned bytes required for the single-slot KV cache arena.
    pub kv_bytes: u64,
    /// Total page-aligned bytes required (`weight_bytes + kv_bytes`).
    pub total_required_bytes: u64,
    /// Number of 16-token blocks allocated for the KV cache.
    pub kv_block_count: usize,
    /// Total token capacity allocated for the KV cache (`kv_block_count * 16`).
    pub kv_allocated_tokens: usize,
}

/// Computes the exact page-aligned memory footprint required for model weights
/// and a standalone single-slot KV cache arena sized to `n_ctx` tokens (ADR-027, ADR-037).
///
/// # Errors
///
/// Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds the model's trained context limit.
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx` is zero or if required architecture dimensions are zero.
/// Computes the exact page-aligned memory footprint required for model weights
/// and a standalone single-slot KV cache arena sized to `n_ctx` tokens (ADR-027, ADR-037).
///
/// # Errors
///
/// Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds the model's trained context limit.
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx` is zero or if required architecture dimensions are zero.
pub fn calculate_footprint(
    metadata: &ModelMetadata,
    n_ctx: usize,
) -> Result<MemoryFootprint, K3Error> {
    calculate_footprint_with_slots(metadata, n_ctx, 1)
}

/// Returns `true` if `arch` requires an autoregressive KV cache allocation (`false` for
/// single-pass bidirectional encoders such as `modernbert`, `bert`, and `deberta-v3`).
#[must_use]
pub const fn architecture_requires_kv_cache(arch: &str) -> bool {
    !matches!(arch.as_bytes(), b"modernbert" | b"bert" | b"deberta-v3")
}

/// Computes the raw unaligned KV cache size in bytes across `slots` concurrency slots.
fn estimate_gemma_raw_kv_bytes(
    metadata: &ModelMetadata,
    blocks_per_slot: usize,
    slots: usize,
) -> Result<usize, K3Error> {
    let per_slot_tokens =
        blocks_per_slot
            .checked_mul(KV_BLOCK_SIZE)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "per-slot kv allocated tokens overflow usize".to_owned(),
            })?;
    let layer_count = metadata.layer_count().max(1);
    let head_count_kv = metadata.head_count_kv().max(1);
    let head_dim = if metadata.head_count() > 0 {
        metadata.embedding_dim() / metadata.head_count()
    } else {
        metadata.embedding_dim()
    }
    .max(1);
    let policy_arch = if metadata.architecture() == "functiongemma" {
        "gemma3"
    } else {
        metadata.architecture()
    };
    let policies = gemma_layer_kv_policies(policy_arch, layer_count, None);
    let single_slot_gemma_bytes =
        estimate_hybrid_kv_cache_bytes(per_slot_tokens, head_count_kv, head_dim, &policies)?;
    single_slot_gemma_bytes
        .checked_mul(slots)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "gemma multi-slot kv bytes calculation overflowed usize".to_owned(),
        })
}

fn estimate_mla_raw_kv_bytes(
    metadata: &ModelMetadata,
    blocks_per_slot: usize,
    slots: usize,
) -> Result<usize, K3Error> {
    let kv_allocated_tokens = blocks_per_slot
        .checked_mul(slots)
        .and_then(|blocks| blocks.checked_mul(KV_BLOCK_SIZE))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "mla kv allocated tokens calculation overflowed usize".to_owned(),
        })?;
    let layer_count = metadata.layer_count().max(1);
    let kv_lora_rank = metadata.kv_lora_rank().unwrap_or(512).max(1);
    let qk_rope_head_dim = metadata.qk_rope_head_dim().unwrap_or(64).max(1);
    let kv_latent_dim =
        kv_lora_rank
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "mla kv_latent_dim calculation overflowed usize".to_owned(),
            })?;
    let total_elements = kv_allocated_tokens
        .checked_mul(layer_count)
        .and_then(|v| v.checked_mul(kv_latent_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "mla total kv elements calculation overflowed usize".to_owned(),
        })?;
    total_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "mla raw kv bytes calculation overflowed usize".to_owned(),
        })
}

fn estimate_raw_kv_bytes_for_slots(
    metadata: &ModelMetadata,
    n_ctx: usize,
    blocks_per_slot: usize,
    slots: usize,
) -> Result<usize, K3Error> {
    if metadata.architecture() == "dory" {
        let single_slot_dory_bytes = estimate_dory_kv_cache_bytes(metadata, n_ctx, None, None)?;
        return single_slot_dory_bytes.checked_mul(slots).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "dory multi-slot kv bytes calculation overflowed usize".to_owned(),
            }
        });
    }

    if matches!(
        metadata.architecture(),
        "gemma2" | "gemma3" | "gemma4" | "functiongemma"
    ) {
        return estimate_gemma_raw_kv_bytes(metadata, blocks_per_slot, slots);
    }

    if metadata.architecture() == "deepseek2" || metadata.kv_lora_rank().is_some() {
        return estimate_mla_raw_kv_bytes(metadata, blocks_per_slot, slots);
    }

    if is_pure_recurrent_arch(metadata.architecture()) {
        let single_slot = estimate_pure_recurrent_raw_bytes_per_slot(metadata)?;
        return single_slot
            .checked_mul(slots)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "recurrent_state_arena".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "pure recurrent multi-slot bytes overflowed usize".to_owned(),
            });
    }

    if is_hybrid_ssm_arch(metadata.architecture()) {
        let single_slot = estimate_hybrid_raw_bytes_per_slot(metadata, n_ctx, blocks_per_slot)?;
        return single_slot
            .checked_mul(slots)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "hybrid_ssm_arena".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "hybrid multi-slot bytes overflowed usize".to_owned(),
            });
    }

    let kv_allocated_tokens = blocks_per_slot
        .checked_mul(slots)
        .and_then(|blocks| blocks.checked_mul(KV_BLOCK_SIZE))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "kv allocated tokens calculation overflowed usize".to_owned(),
        })?;
    let layer_count = metadata.layer_count().max(1);
    let head_count_kv = metadata.head_count_kv().max(1);
    let head_dim = if metadata.head_count() > 0 {
        metadata.embedding_dim() / metadata.head_count()
    } else {
        metadata.embedding_dim()
    }
    .max(1);

    let elements_per_token = 2usize
        .checked_mul(layer_count)
        .and_then(|v| v.checked_mul(head_count_kv))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "kv elements per token calculation overflowed usize".to_owned(),
        })?;

    let total_elements = kv_allocated_tokens
        .checked_mul(elements_per_token)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "total kv elements calculation overflowed usize".to_owned(),
        })?;

    total_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "raw kv bytes calculation overflowed usize".to_owned(),
        })
}

/// Estimates page-aligned single-slot `Multi-Head Latent Attention` (`MLA`) KV cache bytes
/// for `deepseek2` checkpoints caching compressed latent vectors (`kv_lora_rank + qk_rope_head_dim`)
/// per token per layer (Decision 1).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if any dimension argument is zero, or
/// [`K3Error::TensorBoundsError`] if the byte calculation overflows `usize`.
pub fn estimate_mla_kv_cache_bytes(
    n_ctx: usize,
    num_layers: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
) -> Result<usize, K3Error> {
    if n_ctx == 0 || num_layers == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deepseek2".to_owned(),
            reason: format!(
                "MLA KV cache dimensions must all be > 0 (got n_ctx={n_ctx}, num_layers={num_layers}, kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim})"
            ),
        });
    }
    let allocated_tokens = n_ctx
        .div_ceil(KV_BLOCK_SIZE)
        .checked_mul(KV_BLOCK_SIZE)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "mla kv allocated tokens overflowed usize".to_owned(),
        })?;
    let kv_latent_dim =
        kv_lora_rank
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "mla kv_latent_dim overflowed usize".to_owned(),
            })?;
    let raw_bytes = allocated_tokens
        .checked_mul(num_layers)
        .and_then(|v| v.checked_mul(kv_latent_dim))
        .and_then(|v| v.checked_mul(size_of::<f32>()))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "mla raw kv bytes overflowed usize".to_owned(),
        })?;
    Ok(align_up(raw_bytes, system_page_size()))
}

/// Estimates page-aligned single-slot KV cache bytes for `metadata` and `n_ctx`.
///
/// Returns `Ok(0)` without reserving KV cache pages when `requires_kv_cache` is `false`
/// or when `metadata.architecture()` is a zero-KV encoder (`modernbert`, `bert`, `deberta-v3`).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx == 0`, [`K3Error::ContextLengthExceeded`]
/// if `n_ctx > metadata.context_length()`, or [`K3Error::TensorBoundsError`] on size overflow.
pub fn estimate_kv_cache_bytes(
    metadata: &ModelMetadata,
    n_ctx: usize,
    requires_kv_cache: bool,
) -> Result<usize, K3Error> {
    if n_ctx == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: metadata.architecture().to_owned(),
            reason: "requested context length n_ctx must be greater than zero".to_owned(),
        });
    }
    if metadata.context_length() > 0 && n_ctx > metadata.context_length() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: n_ctx,
            max_context_tokens: metadata.context_length(),
        });
    }
    if !requires_kv_cache || !architecture_requires_kv_cache(metadata.architecture()) {
        return Ok(0);
    }
    let blocks_per_slot = n_ctx.div_ceil(KV_BLOCK_SIZE);
    let raw_bytes = estimate_raw_kv_bytes_for_slots(metadata, n_ctx, blocks_per_slot, 1)?;
    Ok(align_up(raw_bytes, system_page_size()))
}

/// Computes the exact page-aligned memory footprint required for model weights
/// and a standalone KV cache arena sized to `n_ctx` tokens across `active_slots` concurrency slots (ADR-027, ADR-037, ADR-043).
///
/// # Errors
///
/// Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds the model's trained context limit.
/// Returns [`K3Error::CapabilityMismatch`] if `n_ctx` is zero or if required architecture dimensions are zero.
pub fn calculate_footprint_with_slots(
    metadata: &ModelMetadata,
    n_ctx: usize,
    active_slots: usize,
) -> Result<MemoryFootprint, K3Error> {
    calculate_footprint_inner(metadata, metadata.total_weight_bytes(), n_ctx, active_slots)
}

fn calculate_footprint_inner(
    metadata: &ModelMetadata,
    weight_bytes_raw: usize,
    n_ctx: usize,
    active_slots: usize,
) -> Result<MemoryFootprint, K3Error> {
    if n_ctx == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: metadata.architecture().to_owned(),
            reason: "requested context length n_ctx must be greater than zero".to_owned(),
        });
    }

    if metadata.context_length() > 0 && n_ctx > metadata.context_length() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: n_ctx,
            max_context_tokens: metadata.context_length(),
        });
    }

    let page_size = system_page_size();

    // 1. Model weights footprint (page-aligned)
    let weight_bytes_aligned = align_up(weight_bytes_raw, page_size);
    let weight_bytes =
        u64::try_from(weight_bytes_aligned).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "weights".to_owned(),
            offset: 0,
            byte_len: weight_bytes_aligned,
            buffer_len: usize::MAX,
            reason: "aligned weight bytes overflow u64".to_owned(),
        })?;

    if !architecture_requires_kv_cache(metadata.architecture()) {
        return Ok(MemoryFootprint {
            weight_bytes,
            kv_bytes: 0,
            total_required_bytes: weight_bytes,
            kv_block_count: 0,
            kv_allocated_tokens: 0,
        });
    }

    // 2. Standalone KV cache / O(1) recurrent state footprint
    let blocks_per_slot = n_ctx.div_ceil(KV_BLOCK_SIZE);
    let slots = active_slots.max(1);
    let (kv_block_count, kv_allocated_tokens) = if is_pure_recurrent_arch(metadata.architecture()) {
        (0, 0)
    } else {
        let kv_block_count =
            blocks_per_slot
                .checked_mul(slots)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: "kv_cache".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: usize::MAX,
                    reason: "kv block count calculation overflowed usize".to_owned(),
                })?;
        let kv_allocated_tokens = kv_block_count.checked_mul(KV_BLOCK_SIZE).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "kv allocated tokens overflow usize".to_owned(),
            }
        })?;
        (kv_block_count, kv_allocated_tokens)
    };

    let raw_kv_bytes = estimate_raw_kv_bytes_for_slots(metadata, n_ctx, blocks_per_slot, slots)?;

    let kv_bytes_aligned = align_up(raw_kv_bytes, page_size);
    let kv_bytes = u64::try_from(kv_bytes_aligned).map_err(|_| K3Error::TensorBoundsError {
        tensor_name: "kv_cache".to_owned(),
        offset: 0,
        byte_len: kv_bytes_aligned,
        buffer_len: usize::MAX,
        reason: "aligned kv bytes overflow u64".to_owned(),
    })?;

    let total_required_bytes =
        weight_bytes
            .checked_add(kv_bytes)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "memory_footprint".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "total required bytes overflowed u64".to_owned(),
            })?;

    Ok(MemoryFootprint {
        weight_bytes,
        kv_bytes,
        total_required_bytes,
        kv_block_count,
        kv_allocated_tokens,
    })
}

/// Trait abstracting system physical RAM and lockable memory limits inspection.
pub trait MemoryInspector {
    /// Returns currently available physical memory in bytes (from `/proc/meminfo` `MemAvailable`).
    ///
    /// # Errors
    /// Returns [`K3Error::InsufficientLockableMemory`] if inspection fails.
    fn available_memory_bytes(&self) -> Result<u64, K3Error>;

    /// Returns the process lockable memory limit (`RLIMIT_MEMLOCK`) in bytes.
    /// Returns `u64::MAX` if unlimited.
    ///
    /// # Errors
    /// Returns [`K3Error::InsufficientLockableMemory`] if the limit query fails.
    fn rlimit_memlock_bytes(&self) -> Result<u64, K3Error>;
}

/// Standard system memory inspector reading from Linux `/proc/meminfo` and `libc::getrlimit`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemMemoryInspector;

impl SystemMemoryInspector {
    /// Creates a new system memory inspector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl MemoryInspector for SystemMemoryInspector {
    fn available_memory_bytes(&self) -> Result<u64, K3Error> {
        if let Ok(val) = std::env::var("K3_AVAILABLE_RAM_BYTES") {
            if let Ok(parsed) = val.trim().parse::<u64>() {
                return Ok(parsed);
            }
        }

        if let Ok(file) = fs::File::open("/proc/meminfo") {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let Ok(entry) = line else { continue };
                if let Some(rest) = entry.strip_prefix("MemAvailable:") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if let Some(kb_str) = parts.first() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb.checked_mul(1024).ok_or(
                                K3Error::InsufficientLockableMemory {
                                    required_bytes: 0,
                                    weight_bytes: 0,
                                    kv_bytes: 0,
                                    available_bytes: 0,
                                    rlimit_memlock_bytes: 0,
                                },
                            );
                        }
                    }
                }
            }
        }

        // Fallback: query sysconf if /proc/meminfo parsing did not find MemAvailable
        // SAFETY: sysconf is a POSIX query with no side effects.
        #[cfg(target_os = "linux")]
        let pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
        // SAFETY: sysconf is a POSIX query with no side effects.
        #[cfg(not(target_os = "linux"))]
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        // SAFETY: sysconf(_SC_PAGESIZE) is a POSIX query with no side effects.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            if let (Ok(p), Ok(ps)) = (u64::try_from(pages), u64::try_from(page_size)) {
                return Ok(p.saturating_mul(ps));
            }
        }

        Err(K3Error::InsufficientLockableMemory {
            required_bytes: 0,
            weight_bytes: 0,
            kv_bytes: 0,
            available_bytes: 0,
            rlimit_memlock_bytes: 0,
        })
    }

    fn rlimit_memlock_bytes(&self) -> Result<u64, K3Error> {
        if let Ok(val) = std::env::var("K3_RLIMIT_MEMLOCK_BYTES") {
            if let Ok(parsed) = val.trim().parse::<u64>() {
                return Ok(parsed);
            }
        }

        let mut rlim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };

        // SAFETY: getrlimit with RLIMIT_MEMLOCK passes a valid pointer to libc::rlimit.
        let ret = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &raw mut rlim) };
        if ret != 0 {
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: 0,
                weight_bytes: 0,
                kv_bytes: 0,
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            });
        }

        #[allow(clippy::useless_conversion)]
        if rlim.rlim_cur == libc::RLIM_INFINITY {
            Ok(u64::MAX)
        } else {
            Ok(rlim.rlim_cur)
        }
    }
}

/// Deterministic mock memory inspector for testing admission boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockMemoryInspector {
    /// Mocked available RAM in bytes.
    pub available_bytes: u64,
    /// Mocked process `RLIMIT_MEMLOCK` in bytes.
    pub rlimit_memlock_bytes: u64,
}

impl MockMemoryInspector {
    /// Creates a mock memory inspector with explicit available and lockable memory bounds.
    #[must_use]
    pub const fn new(available_bytes: u64, rlimit_memlock_bytes: u64) -> Self {
        Self {
            available_bytes,
            rlimit_memlock_bytes,
        }
    }
}

impl MemoryInspector for MockMemoryInspector {
    fn available_memory_bytes(&self) -> Result<u64, K3Error> {
        Ok(self.available_bytes)
    }

    fn rlimit_memlock_bytes(&self) -> Result<u64, K3Error> {
        Ok(self.rlimit_memlock_bytes)
    }
}

/// Verifies that required model weights and the single-slot KV cache arena do not exceed
/// currently available physical memory or the process lockable limit (`RLIMIT_MEMLOCK`).
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if total required bytes exceed available RAM
/// or `RLIMIT_MEMLOCK` (ADR-037, US-01 Scenario 3).
pub fn check_memory_admission(
    metadata: &ModelMetadata,
    n_ctx: usize,
) -> Result<MemoryFootprint, K3Error> {
    check_memory_admission_with(metadata, n_ctx, &SystemMemoryInspector::new())
}

/// Verifies pre-flight memory admission using a customized or mocked memory inspector.
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if required footprint exceeds lockable limits.
pub fn check_memory_admission_with<I: MemoryInspector>(
    metadata: &ModelMetadata,
    n_ctx: usize,
    inspector: &I,
) -> Result<MemoryFootprint, K3Error> {
    check_memory_admission_with_slots(metadata, n_ctx, 1, inspector)
}

fn check_footprint_against_inspector<I: MemoryInspector>(
    footprint: MemoryFootprint,
    inspector: &I,
) -> Result<MemoryFootprint, K3Error> {
    let available_bytes = inspector.available_memory_bytes()?;
    let rlimit_memlock_bytes = inspector.rlimit_memlock_bytes()?;

    if footprint.total_required_bytes > available_bytes
        || footprint.total_required_bytes > rlimit_memlock_bytes
    {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: footprint.total_required_bytes,
            weight_bytes: footprint.weight_bytes,
            kv_bytes: footprint.kv_bytes,
            available_bytes,
            rlimit_memlock_bytes,
        });
    }

    Ok(footprint)
}

/// Verifies pre-flight memory admission accounting for active concurrency slots using a customized inspector.
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if required footprint exceeds lockable limits.
pub fn check_memory_admission_with_slots<I: MemoryInspector>(
    metadata: &ModelMetadata,
    n_ctx: usize,
    active_slots: usize,
    inspector: &I,
) -> Result<MemoryFootprint, K3Error> {
    let footprint = calculate_footprint_with_slots(metadata, n_ctx, active_slots)?;
    check_footprint_against_inspector(footprint, inspector)
}

/// Verifies pre-flight single-slot memory admission using an explicit raw weight size override
/// (for checking large production profiles such as `Qwen3-30B-A3B` without materializing multi-GiB
/// files in constrained test environments).
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if the footprint exceeds lockable limits, or
/// [`K3Error::ContextLengthExceeded`] / [`K3Error::CapabilityMismatch`] on invalid context bounds.
pub fn check_memory_admission_for_weight_bytes<I: MemoryInspector>(
    metadata: &ModelMetadata,
    weight_bytes_raw: usize,
    n_ctx: usize,
    inspector: &I,
) -> Result<MemoryFootprint, K3Error> {
    let footprint = calculate_footprint_inner(metadata, weight_bytes_raw, n_ctx, 1)?;
    check_footprint_against_inspector(footprint, inspector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::GgufFile;
    use jero_core::QuantType;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    #[test]
    fn test_footprint_calculation_page_aligned() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // 32 tokens requested -> 2 blocks of 16 tokens -> 32 allocated tokens
        let footprint = calculate_footprint(meta, 32).expect("footprint calculation succeeds");
        let page = system_page_size();

        assert_eq!(footprint.kv_block_count, 2);
        assert_eq!(footprint.kv_allocated_tokens, 32);
        assert_eq!(footprint.weight_bytes % page as u64, 0);
        assert_eq!(footprint.kv_bytes % page as u64, 0);
        assert_eq!(
            footprint.total_required_bytes,
            footprint.weight_bytes + footprint.kv_bytes
        );
    }

    #[test]
    fn test_memory_admission_succeeds_when_capacity_sufficient() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        let inspector = MockMemoryInspector::new(100_000_000, 100_000_000);
        let admission = check_memory_admission_with(meta, 32, &inspector);
        assert!(admission.is_ok());
    }

    #[test]
    fn test_memory_admission_refuses_when_exceeding_available_ram() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // Available RAM is only 100 bytes (far less than page-aligned footprint)
        let inspector = MockMemoryInspector::new(100, 100_000_000);
        let err = check_memory_admission_with(meta, 32, &inspector)
            .expect_err("must refuse upfront when available RAM is insufficient");

        match err {
            K3Error::InsufficientLockableMemory {
                required_bytes,
                weight_bytes,
                kv_bytes,
                available_bytes,
                rlimit_memlock_bytes,
            } => {
                assert!(required_bytes > available_bytes);
                assert_eq!(available_bytes, 100);
                assert_eq!(rlimit_memlock_bytes, 100_000_000);
                assert_eq!(required_bytes, weight_bytes + kv_bytes);
            }
            other => panic!("expected InsufficientLockableMemory, got: {other:?}"),
        }
    }

    #[test]
    fn test_memory_admission_refuses_when_exceeding_rlimit_memlock() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // RLIMIT_MEMLOCK is only 500 bytes
        let inspector = MockMemoryInspector::new(100_000_000, 500);
        let err = check_memory_admission_with(meta, 32, &inspector)
            .expect_err("must refuse upfront when RLIMIT_MEMLOCK is exceeded");

        match err {
            K3Error::InsufficientLockableMemory {
                required_bytes,
                available_bytes,
                rlimit_memlock_bytes,
                ..
            } => {
                assert!(required_bytes > rlimit_memlock_bytes);
                assert_eq!(available_bytes, 100_000_000);
                assert_eq!(rlimit_memlock_bytes, 500);
            }
            other => panic!("expected InsufficientLockableMemory, got: {other:?}"),
        }
    }

    #[test]
    fn test_calculate_footprint_rejects_zero_n_ctx() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        let err = calculate_footprint(meta, 0).expect_err("must reject zero n_ctx");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    #[test]
    fn test_calculate_footprint_rejects_n_ctx_exceeding_trained_limit() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // synthetic model trained context length in test helper is 2048
        let err = calculate_footprint(meta, 4096).expect_err("must reject n_ctx > 2048");
        assert_eq!(
            err,
            K3Error::ContextLengthExceeded {
                requested_tokens: 4096,
                max_context_tokens: 2048,
            }
        );
    }

    #[test]
    fn test_calculate_footprint_scales_with_active_slots() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // 32 tokens = 2 blocks of 16
        let fp1 = calculate_footprint_with_slots(meta, 32, 1).expect("footprint 1 slot");
        assert_eq!(fp1.kv_block_count, 2);
        assert_eq!(fp1.kv_allocated_tokens, 32);

        // 4 slots: 2 * 4 = 8 blocks = 128 tokens
        let fp4 = calculate_footprint_with_slots(meta, 32, 4).expect("footprint 4 slots");
        assert_eq!(fp4.kv_block_count, 8);
        assert_eq!(fp4.kv_allocated_tokens, 128);
        assert!(fp4.kv_bytes > fp1.kv_bytes);
        assert_eq!(fp4.weight_bytes, fp1.weight_bytes);
    }

    #[test]
    fn test_dory_kv_admission_per_loop_and_last_loop() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        let canonical_pattern =
            "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";
        let extra_kvs = [
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.input_layer_count", GgufValue::Uint32(16)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
            ("dory.output_layer_count", GgufValue::Uint32(16)),
            ("dory.num_loops", GgufValue::Uint32(2)),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("per_loop"),
            ),
            ("dory.layer_pattern", GgufValue::String(canonical_pattern)),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
        let model = GgufFile::parse(&bytes).expect("parse synthetic dory gguf");
        let meta = model.metadata();

        // At n_ctx = 2048 (> sliding_window = 1024):
        // 'S' sub-layers cap at 1024 tokens, '*' sub-layers use 2048 tokens, '-' MLP sub-layers allocate 0 bytes.
        let per_loop_k1 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(1), Some("per_loop")).unwrap();
        let per_loop_k2 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(2), Some("per_loop")).unwrap();
        let per_loop_k4 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(4), Some("per_loop")).unwrap();

        let last_loop_k1 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(1), Some("last_loop")).unwrap();
        let last_loop_k2 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(2), Some("last_loop")).unwrap();
        let last_loop_k4 =
            estimate_dory_kv_cache_bytes(meta, 2048, Some(4), Some("last_loop")).unwrap();

        // In last_loop mode, KV bytes remain strictly invariant across K=1, 2, 4 and equal K=1 per_loop
        assert_eq!(last_loop_k1, per_loop_k1);
        assert_eq!(last_loop_k2, last_loop_k1);
        assert_eq!(last_loop_k4, last_loop_k1);

        // In per_loop mode, KV bytes grow monotonically with K
        assert!(per_loop_k2 > per_loop_k1);
        assert!(per_loop_k4 > per_loop_k2);

        // Default from GGUF metadata (num_loops=2, per_loop) matches per_loop_k2
        let default_bytes = estimate_dory_kv_cache_bytes(meta, 2048, None, None).unwrap();
        assert_eq!(default_bytes, per_loop_k2);

        // Verify pure MLP ('-') sub-layers allocate 0 KV bytes
        let mlp_only_kvs = [
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.input_layer_count", GgufValue::Uint32(2)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(2)),
            ("dory.output_layer_count", GgufValue::Uint32(2)),
            ("dory.num_loops", GgufValue::Uint32(2)),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("per_loop"),
            ),
            ("dory.layer_pattern", GgufValue::String("------")),
        ];
        let mlp_bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &mlp_only_kvs, &[]);
        let mlp_model = GgufFile::parse(&mlp_bytes).expect("parse mlp-only dory gguf");
        assert_eq!(
            estimate_dory_kv_cache_bytes(mlp_model.metadata(), 512, None, None).unwrap(),
            0
        );
    }

    #[test]
    fn test_tied_embedding_footprint_admission() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        let qwen3_kvs = [
            ("qwen3.block_count", GgufValue::Uint32(4)),
            ("qwen3.embedding_length", GgufValue::Uint32(64)),
            ("qwen3.attention.head_count", GgufValue::Uint32(4)),
            ("qwen3.attention.head_count_kv", GgufValue::Uint32(2)),
        ];
        // Tied checkpoint omits output.weight and carries only token_embd.weight.
        let tied_specs = [("token_embd.weight", QuantType::F32, &[64, 16][..])];
        let tied_bytes = create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &tied_specs);
        let tied_model = GgufFile::parse(&tied_bytes).expect("parse tied qwen3 gguf");
        assert!(tied_model.metadata().has_tied_embeddings());

        // Untied checkpoint duplicates output.weight with the same shape.
        let untied_specs = [
            ("token_embd.weight", QuantType::F32, &[64, 16][..]),
            ("output.weight", QuantType::F32, &[64, 16][..]),
        ];
        let untied_bytes =
            create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &untied_specs);
        let untied_model = GgufFile::parse(&untied_bytes).expect("parse untied qwen3 gguf");
        assert!(!untied_model.metadata().has_tied_embeddings());

        let page_size = system_page_size();
        let expected_tied_weights = u64::try_from(align_up(64 * 16 * 4, page_size)).unwrap();
        let tied_fp = calculate_footprint(tied_model.metadata(), 32).expect("tied footprint");
        let untied_fp = calculate_footprint(untied_model.metadata(), 32).expect("untied footprint");

        assert_eq!(tied_fp.weight_bytes, expected_tied_weights);
        assert_eq!(
            tied_model.metadata().total_weight_bytes(),
            64 * 16 * size_of::<f32>()
        );
        assert_eq!(
            untied_model.metadata().total_weight_bytes(),
            2 * 64 * 16 * size_of::<f32>()
        );
        assert_eq!(tied_fp.kv_bytes, untied_fp.kv_bytes);
        assert!(tied_fp.total_required_bytes <= untied_fp.total_required_bytes);
    }

    #[test]
    fn test_deepseek2_mla_kv_admission_reduction() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // DeepSeek-V2-Lite profile: 27 layers, 16 heads, head_dim=128 uncompressed vs
        // MLA latent dim = kv_lora_rank(512) + qk_rope_head_dim(64) = 576 floats/token/layer.
        let mla_bytes = estimate_mla_kv_cache_bytes(4096, 27, 512, 64).expect("mla kv bytes");

        let ds_kvs = [
            ("deepseek2.block_count", GgufValue::Uint32(27)),
            ("deepseek2.embedding_length", GgufValue::Uint32(2048)),
            ("deepseek2.attention.head_count", GgufValue::Uint32(16)),
            ("deepseek2.attention.head_count_kv", GgufValue::Uint32(16)),
            ("deepseek2.attention.kv_lora_rank", GgufValue::Uint32(512)),
            ("deepseek2.rope.dimension_count", GgufValue::Uint32(64)),
            ("deepseek2.context_length", GgufValue::Uint32(4096)),
        ];
        let ds_bytes = create_synthetic_gguf_v3_with_kvs("deepseek2", 4096, &ds_kvs, &[]);
        let ds_model = GgufFile::parse(&ds_bytes).expect("parse deepseek2 gguf");

        let meta_kv_bytes =
            estimate_kv_cache_bytes(ds_model.metadata(), 4096, true).expect("meta mla kv bytes");
        assert_eq!(meta_kv_bytes, mla_bytes);

        let gqa_kvs = [
            ("llama.block_count", GgufValue::Uint32(27)),
            ("llama.embedding_length", GgufValue::Uint32(2048)),
            ("llama.attention.head_count", GgufValue::Uint32(16)),
            ("llama.attention.head_count_kv", GgufValue::Uint32(16)),
            ("llama.attention.key_length", GgufValue::Uint32(128)),
            ("llama.context_length", GgufValue::Uint32(4096)),
        ];
        let gqa_bytes = create_synthetic_gguf_v3_with_kvs("llama", 4096, &gqa_kvs, &[]);
        let gqa_model = GgufFile::parse(&gqa_bytes).expect("parse uncompressed baseline");
        let uncompressed_bytes =
            estimate_kv_cache_bytes(gqa_model.metadata(), 4096, true).expect("gqa kv bytes");

        // 576 / (2 * 16 * 128 = 4096) = 14.0625%, i.e., ~85.9% reduction (>85% saved).
        assert!(mla_bytes * 100 <= uncompressed_bytes * 15);
        assert!(
            estimate_mla_kv_cache_bytes(0, 27, 512, 64).is_err(),
            "zero n_ctx must fail closed"
        );
    }

    #[test]
    fn test_hybrid_ssm_arena_footprint() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // 1:7 hybrid model (32 layers: 4 GqaAttention + 28 Mamba2Ssd => ~87.5% KV reduction).
        let mut schedule_hybrid = Vec::with_capacity(32);
        let mut kv_slot = 0_usize;
        let mut ssm_slot = 0_usize;
        for layer in 0..32_usize {
            if layer % 8 == 0 {
                schedule_hybrid.push(HybridLayerKind::GqaAttention {
                    kv_layer_slot: kv_slot,
                });
                kv_slot += 1;
            } else {
                schedule_hybrid.push(HybridLayerKind::Mamba2Ssd {
                    recurrent_layer_slot: ssm_slot,
                });
                ssm_slot += 1;
            }
        }
        let schedule_dense: Vec<HybridLayerKind> = (0..32)
            .map(|slot| HybridLayerKind::GqaAttention {
                kv_layer_slot: slot,
            })
            .collect();

        let kv_bytes_per_attn = 4 * 1024 * 1024;
        let state_bytes_per_ssm = 8 * 1024;
        let hybrid_bytes = estimate_hybrid_ssm_arena_bytes(
            &schedule_hybrid,
            4096,
            kv_bytes_per_attn,
            state_bytes_per_ssm,
        )
        .expect("hybrid arena bytes");
        let dense_bytes = estimate_hybrid_ssm_arena_bytes(
            &schedule_dense,
            4096,
            kv_bytes_per_attn,
            state_bytes_per_ssm,
        )
        .expect("dense arena bytes");

        // Hybrid uses ~12.5-15% of all-attention KV footprint (>=85% saved).
        assert!(
            hybrid_bytes * 100 <= dense_bytes * 15,
            "expected ~85% KV reduction on 1:8 hybrid, got hybrid={hybrid_bytes} dense={dense_bytes}"
        );
        assert!(
            estimate_hybrid_ssm_arena_bytes(
                &schedule_hybrid,
                0,
                kv_bytes_per_attn,
                state_bytes_per_ssm
            )
            .is_err(),
            "zero n_ctx must fail closed"
        );

        // Verify pure Mamba-2 & RWKV-7 footprints remain strictly O(1) independent of n_ctx.
        for (arch, extra_kvs) in [
            (
                "mamba2",
                vec![
                    ("mamba2.block_count", GgufValue::Uint32(4)),
                    ("mamba2.embedding_length", GgufValue::Uint32(64)),
                    ("mamba2.ssm.inner_size", GgufValue::Uint32(128)),
                    ("mamba2.ssm.conv_kernel", GgufValue::Uint32(4)),
                    ("mamba2.ssm.state_size", GgufValue::Uint32(16)),
                    ("mamba2.context_length", GgufValue::Uint32(32768)),
                ],
            ),
            (
                "rwkv7",
                vec![
                    ("rwkv7.block_count", GgufValue::Uint32(4)),
                    ("rwkv7.embedding_length", GgufValue::Uint32(64)),
                    ("rwkv7.wkv.head_size", GgufValue::Uint32(16)),
                    ("rwkv7.context_length", GgufValue::Uint32(32768)),
                ],
            ),
        ] {
            let bytes = create_synthetic_gguf_v3_with_kvs(arch, 32, &extra_kvs, &[]);
            let gguf = GgufFile::parse(&bytes).expect("parse pure recurrent gguf");
            let fp_small = calculate_footprint(gguf.metadata(), 64).expect("footprint n_ctx=64");
            let fp_large =
                calculate_footprint(gguf.metadata(), 16_384).expect("footprint n_ctx=16384");
            assert_eq!(fp_small.kv_block_count, 0);
            assert_eq!(fp_small.kv_bytes, fp_large.kv_bytes);
            assert_eq!(fp_small.total_required_bytes, fp_large.total_required_bytes);
            assert!(fp_small.kv_bytes > 0);
        }
    }
}
