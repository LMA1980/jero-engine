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
        let per_slot_tokens = blocks_per_slot.checked_mul(KV_BLOCK_SIZE).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "per-slot kv allocated tokens overflow usize".to_owned(),
            }
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
        return single_slot_gemma_bytes.checked_mul(slots).ok_or_else(|| {
            K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "gemma multi-slot kv bytes calculation overflowed usize".to_owned(),
            }
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
    let weight_bytes_raw = metadata.total_weight_bytes();
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

    // 2. Standalone KV cache footprint (16-token blocks scaled by active_slots, page-aligned)
    let blocks_per_slot = n_ctx.div_ceil(KV_BLOCK_SIZE);
    let slots = active_slots.max(1);
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
    let kv_allocated_tokens =
        kv_block_count
            .checked_mul(KV_BLOCK_SIZE)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "kv allocated tokens overflow usize".to_owned(),
            })?;

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
}
