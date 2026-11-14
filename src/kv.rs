//! Standalone in-process locked KV cache arena.
//!
//! Organizes KV cache memory into fixed 16-token blocks sized strictly for 1 active slot × `n_ctx`
//! tokens, locked in physical memory via `mlock()` and indexed via block tables (ADR-027, ADR-037, `kv-allocator.md`).

use std::ptr;

use jero_core::{GgufFile, K3Error, ModelMetadata, align_up};

use crate::capability::GgufHyperparams;
use crate::memory::{KV_BLOCK_SIZE, system_page_size};

/// In-process single-slot KV cache arena organized in 16-token blocks and locked in RAM.
#[derive(Debug)]
pub struct SingleSlotKvArena {
    ptr: *mut f32,
    total_elements: usize,
    aligned_bytes: usize,
    n_ctx: usize,
    num_blocks: usize,
    layer_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    elements_per_token: usize,
    elements_per_block: usize,
}

// SAFETY: SingleSlotKvArena owns an mmap/mlocked buffer of floats. Concurrent immutable access is safe.
unsafe impl Send for SingleSlotKvArena {}
// SAFETY: SingleSlotKvArena access adheres to Rust borrow rules (& vs &mut).
unsafe impl Sync for SingleSlotKvArena {}

#[derive(Debug, Clone, Copy)]
struct ArenaLayout {
    num_blocks: usize,
    elements_per_token: usize,
    elements_per_block: usize,
    total_elements: usize,
    aligned_bytes: usize,
}

fn calculate_arena_layout(
    layer_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    n_ctx: usize,
) -> Result<ArenaLayout, K3Error> {
    let num_blocks = n_ctx.div_ceil(KV_BLOCK_SIZE);
    let allocated_tokens =
        num_blocks
            .checked_mul(KV_BLOCK_SIZE)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_arena".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "allocated tokens overflowed usize".to_owned(),
            })?;

    // 2 vectors (Key and Value) per layer * head_count_kv * head_dim
    let elements_per_token = 2usize
        .checked_mul(layer_count)
        .and_then(|v| v.checked_mul(head_count_kv))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_arena".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "elements per token overflowed usize".to_owned(),
        })?;

    let elements_per_block = KV_BLOCK_SIZE
        .checked_mul(elements_per_token)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_arena".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "elements per block overflowed usize".to_owned(),
        })?;

    let total_elements = allocated_tokens
        .checked_mul(elements_per_token)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_arena".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "total elements overflowed usize".to_owned(),
        })?;

    let raw_bytes = total_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "kv_arena".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "raw bytes overflowed usize".to_owned(),
        })?;

    let page_size = system_page_size();
    let aligned_bytes = align_up(raw_bytes, page_size);

    Ok(ArenaLayout {
        num_blocks,
        elements_per_token,
        elements_per_block,
        total_elements,
        aligned_bytes,
    })
}

impl SingleSlotKvArena {
    /// Allocates and locks an in-process single-slot KV cache arena sized for `n_ctx` tokens
    /// matching the given model metadata parameters (ADR-027, ADR-037, `kv-allocator.md`).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if `n_ctx` is zero.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds `metadata.context_length()`.
    /// - Returns [`K3Error::InsufficientLockableMemory`] if `mmap` or `mlock` allocation fails.
    pub fn new(metadata: &ModelMetadata, n_ctx: usize) -> Result<Self, K3Error> {
        if n_ctx == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: "KV cache context length n_ctx must be greater than zero".to_owned(),
            });
        }

        if metadata.context_length() > 0 && n_ctx > metadata.context_length() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: n_ctx,
                max_context_tokens: metadata.context_length(),
            });
        }

        if !crate::memory::architecture_requires_kv_cache(metadata.architecture()) {
            return Ok(Self::empty(n_ctx));
        }

        let params = GgufHyperparams::from_metadata(metadata);
        let layer_count = params.block_count.max(1);
        let head_count_kv = params.head_count_kv.max(1);
        let head_dim = params
            .key_length
            .max(if metadata.head_count() == 0 {
                metadata.embedding_dim()
            } else {
                0
            })
            .max(1);

        Self::with_dimensions(layer_count, head_count_kv, head_dim, n_ctx)
    }

    /// Allocates and locks an in-process single-slot KV cache arena from a parsed [`GgufFile`],
    /// honoring explicit `{arch}.attention.key_length` when decoupled from
    /// `embedding_length / head_count`.
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if `n_ctx` is zero.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds `gguf.metadata().context_length()`.
    /// - Returns [`K3Error::InsufficientLockableMemory`] if `mmap` or `mlock` allocation fails.
    pub fn from_gguf(gguf: &GgufFile<'_>, n_ctx: usize) -> Result<Self, K3Error> {
        let metadata = gguf.metadata();
        if n_ctx == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: "KV cache context length n_ctx must be greater than zero".to_owned(),
            });
        }

        if metadata.context_length() > 0 && n_ctx > metadata.context_length() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: n_ctx,
                max_context_tokens: metadata.context_length(),
            });
        }

        if !crate::memory::architecture_requires_kv_cache(metadata.architecture()) {
            return Ok(Self::empty(n_ctx));
        }

        if !crate::memory::architecture_requires_kv_cache(metadata.architecture()) {
            return Ok(Self::empty(n_ctx));
        }

        let params = GgufHyperparams::from_gguf(gguf);
        let layer_count = if let Some(dory) = metadata.dory() {
            let virtual_layers = dory
                .input_layer_count()
                .saturating_add(
                    dory.recurrent_layer_count()
                        .saturating_mul(dory.num_loops()),
                )
                .saturating_add(dory.output_layer_count());
            usize::try_from(virtual_layers)
                .unwrap_or(params.block_count)
                .max(1)
        } else {
            params.block_count.max(1)
        };
        let head_count_kv = params.head_count_kv.max(1);
        let head_dim = params
            .key_length
            .max(if metadata.head_count() == 0 {
                metadata.embedding_dim()
            } else {
                0
            })
            .max(1);

        Self::with_dimensions(layer_count, head_count_kv, head_dim, n_ctx)
    }

    /// Allocates and locks an arena with explicitly specified model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::InsufficientLockableMemory`] if allocation or `mlock` fails.
    pub fn with_dimensions(
        layer_count: usize,
        head_count_kv: usize,
        head_dim: usize,
        n_ctx: usize,
    ) -> Result<Self, K3Error> {
        let layout = calculate_arena_layout(layer_count, head_count_kv, head_dim, n_ctx)?;

        // 1. Allocate anonymous memory pages
        // SAFETY: mmap is called with valid length, anonymous mapping, and read/write protection.
        let raw_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                layout.aligned_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if raw_ptr == libc::MAP_FAILED {
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: u64::try_from(layout.aligned_bytes).unwrap_or(u64::MAX),
                weight_bytes: 0,
                kv_bytes: u64::try_from(layout.aligned_bytes).unwrap_or(u64::MAX),
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            });
        }

        // 2. Best-effort huge page hint (ADR-037)
        // SAFETY: madvise is called on the valid anonymous memory mapping.
        #[cfg(all(target_os = "linux", not(miri)))]
        unsafe {
            libc::madvise(raw_ptr, layout.aligned_bytes, libc::MADV_HUGEPAGE);
        }

        // 3. Mandatory mlock() (ADR-037)
        // SAFETY: mlock is called on the valid anonymous memory mapping.
        #[cfg(not(miri))]
        let mlock_res = unsafe { libc::mlock(raw_ptr, layout.aligned_bytes) };
        #[cfg(miri)]
        let mlock_res = 0;
        if mlock_res != 0 {
            // SAFETY: Clean up mmap on mlock failure.
            unsafe {
                libc::munmap(raw_ptr, layout.aligned_bytes);
            }
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: u64::try_from(layout.aligned_bytes).unwrap_or(u64::MAX),
                weight_bytes: 0,
                kv_bytes: u64::try_from(layout.aligned_bytes).unwrap_or(u64::MAX),
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            });
        }

        let ptr = raw_ptr.cast::<f32>();

        // Zero-initialize memory
        // SAFETY: ptr is valid for writes of total_elements f32 elements.
        unsafe {
            ptr::write_bytes(ptr, 0, layout.total_elements);
        }

        Ok(Self {
            ptr,
            total_elements: layout.total_elements,
            aligned_bytes: layout.aligned_bytes,
            n_ctx,
            num_blocks: layout.num_blocks,
            layer_count,
            head_count_kv,
            head_dim,
            elements_per_token: layout.elements_per_token,
            elements_per_block: layout.elements_per_block,
        })
    }

    /// Creates a zero-allocation arena handle for non-autoregressive architectures that do not
    /// require a persistent KV cache (e.g., `modernbert`, `bert`, `deberta-v3`).
    #[must_use]
    pub const fn empty(n_ctx: usize) -> Self {
        Self {
            ptr: ptr::NonNull::<f32>::dangling().as_ptr(),
            total_elements: 0,
            aligned_bytes: 0,
            n_ctx,
            num_blocks: 0,
            layer_count: 0,
            head_count_kv: 0,
            head_dim: 0,
            elements_per_token: 0,
            elements_per_block: 0,
        }
    }

    /// Returns the maximum context sequence length configured for this arena.
    #[must_use]
    pub const fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Returns the number of allocated 16-token blocks.
    #[must_use]
    pub const fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Returns the layer count configured for this arena.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns the grouped-query K/V head count configured for this arena.
    #[must_use]
    pub const fn head_count_kv(&self) -> usize {
        self.head_count_kv
    }

    /// Returns the per-head K/V dimension (`head_dim`) configured for this arena.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the total token capacity (`num_blocks * 16`).
    #[must_use]
    pub const fn allocated_tokens(&self) -> usize {
        self.num_blocks * KV_BLOCK_SIZE
    }

    /// Returns the total count of `f32` elements allocated in the arena.
    #[must_use]
    pub const fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Returns the number of `f32` elements per token across all layers.
    #[must_use]
    pub const fn elements_per_token(&self) -> usize {
        self.elements_per_token
    }

    /// Returns the number of `f32` elements per 16-token block.
    #[must_use]
    pub const fn elements_per_block(&self) -> usize {
        self.elements_per_block
    }

    /// Returns the total page-aligned bytes allocated and locked.
    #[must_use]
    pub const fn aligned_bytes(&self) -> usize {
        self.aligned_bytes
    }

    /// Returns the full KV cache buffer as a flat immutable slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: self.ptr is valid for reads of self.total_elements f32 elements.
        unsafe { std::slice::from_raw_parts(self.ptr, self.total_elements) }
    }

    /// Returns the full KV cache buffer as a flat mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: self.ptr is valid for reads and writes of self.total_elements f32 elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.total_elements) }
    }

    /// Returns an immutable slice for a specific 16-token block.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `block_idx >= self.num_blocks`.
    pub fn block_slice(&self, block_idx: usize) -> Result<&[f32], K3Error> {
        if block_idx >= self.num_blocks {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_block".to_owned(),
                offset: block_idx,
                byte_len: self.elements_per_block * size_of::<f32>(),
                buffer_len: self.total_elements * size_of::<f32>(),
                reason: format!(
                    "block index {block_idx} exceeds allocated block count {}",
                    self.num_blocks
                ),
            });
        }

        let start = block_idx * self.elements_per_block;
        let end = start + self.elements_per_block;
        Ok(&self.as_slice()[start..end])
    }

    /// Returns a mutable slice for a specific 16-token block.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `block_idx >= self.num_blocks`.
    pub fn block_slice_mut(&mut self, block_idx: usize) -> Result<&mut [f32], K3Error> {
        if block_idx >= self.num_blocks {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_block".to_owned(),
                offset: block_idx,
                byte_len: self.elements_per_block * size_of::<f32>(),
                buffer_len: self.total_elements * size_of::<f32>(),
                reason: format!(
                    "block index {block_idx} exceeds allocated block count {}",
                    self.num_blocks
                ),
            });
        }

        let start = block_idx * self.elements_per_block;
        let end = start + self.elements_per_block;
        Ok(&mut self.as_mut_slice()[start..end])
    }

    /// Returns `(key_slice, value_slice)` views for a specific logical token position and layer.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::ContextLengthExceeded`] if `token_pos >= self.n_ctx`.
    /// Returns [`K3Error::TensorBoundsError`] if `layer >= self.layer_count`.
    pub fn token_kv_slice(
        &self,
        token_pos: usize,
        layer: usize,
    ) -> Result<(&[f32], &[f32]), K3Error> {
        if token_pos >= self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: token_pos + 1,
                max_context_tokens: self.n_ctx,
            });
        }

        if layer >= self.layer_count {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_layer".to_owned(),
                offset: layer,
                byte_len: 0,
                buffer_len: self.layer_count,
                reason: format!(
                    "layer index {layer} exceeds model layer count {}",
                    self.layer_count
                ),
            });
        }

        let block_idx = token_pos / KV_BLOCK_SIZE;
        let token_in_block = token_pos % KV_BLOCK_SIZE;
        let head_kv_elements = self.head_count_kv * self.head_dim;

        let token_start =
            (block_idx * self.elements_per_block) + (token_in_block * self.elements_per_token);
        let layer_start = token_start + (layer * 2 * head_kv_elements);
        let key_start = layer_start;
        let key_end = key_start + head_kv_elements;
        let val_start = key_end;
        let val_end = val_start + head_kv_elements;

        let full = self.as_slice();
        Ok((&full[key_start..key_end], &full[val_start..val_end]))
    }

    /// Returns mutable `(key_slice, value_slice)` views for a specific logical token position and layer.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::ContextLengthExceeded`] if `token_pos >= self.n_ctx`.
    /// Returns [`K3Error::TensorBoundsError`] if `layer >= self.layer_count`.
    pub fn token_kv_slice_mut(
        &mut self,
        token_pos: usize,
        layer: usize,
    ) -> Result<(&mut [f32], &mut [f32]), K3Error> {
        if token_pos >= self.n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: token_pos + 1,
                max_context_tokens: self.n_ctx,
            });
        }

        if layer >= self.layer_count {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_layer".to_owned(),
                offset: layer,
                byte_len: 0,
                buffer_len: self.layer_count,
                reason: format!(
                    "layer index {layer} exceeds model layer count {}",
                    self.layer_count
                ),
            });
        }

        let block_idx = token_pos / KV_BLOCK_SIZE;
        let token_in_block = token_pos % KV_BLOCK_SIZE;
        let head_kv_elements = self.head_count_kv * self.head_dim;

        let token_start =
            (block_idx * self.elements_per_block) + (token_in_block * self.elements_per_token);
        let layer_start = token_start + (layer * 2 * head_kv_elements);
        let key_start = layer_start;
        let key_end = key_start + head_kv_elements;
        let val_start = key_end;

        let full = self.as_mut_slice();
        let (left, right) = full.split_at_mut(val_start);
        let key_slice = &mut left[key_start..key_end];
        let val_slice = &mut right[..head_kv_elements];

        Ok((key_slice, val_slice))
    }
}

impl Drop for SingleSlotKvArena {
    fn drop(&mut self) {
        if self.aligned_bytes == 0 {
            return;
        }
        let ptr = self.ptr.cast::<libc::c_void>();
        let bytes = self.aligned_bytes;

        // SAFETY: self.ptr was mapped by mmap and locked by mlock with size aligned_bytes.
        unsafe {
            #[cfg(not(miri))]
            libc::munlock(ptr, bytes);
            libc::munmap(ptr, bytes);
        }
    }
}

/// Borrowed cyclic ring view returned after writing a token KV pair to a local sliding-window layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlidingWindowKvView<'a> {
    /// Active key vectors slice (`active_len * kv_dim`).
    pub keys: &'a [f32],
    /// Active value vectors slice (`active_len * kv_dim`).
    pub values: &'a [f32],
    /// Number of valid cached causal positions (`min(pos + 1, window_size)`).
    pub active_len: usize,
    /// Cyclic ring buffer slot index written for `pos` (`pos % window_size`).
    pub slot_idx: usize,
}

/// Writes `(k_in, v_in)` at cyclic ring slot `pos % window_size` into a local sliding-window
/// layer buffer `[k_ring (window_size * kv_dim), v_ring (window_size * kv_dim)]` and returns
/// `(k_slice, v_slice, active_len)` bounded to `min(pos + 1, window_size)`.
///
/// # Panics
///
/// Panics if `window_size == 0`, `kv_dim == 0`, `k_in.len() != kv_dim`, `v_in.len() != kv_dim`,
/// or `layer_kv.len() < 2 * window_size * kv_dim`.
pub fn write_and_slice_sliding_kv<'a>(
    layer_kv: &'a mut [f32],
    pos: usize,
    window_size: usize,
    kv_dim: usize,
    k_in: &[f32],
    v_in: &[f32],
) -> (&'a [f32], &'a [f32], usize) {
    assert!(window_size > 0, "sliding window_size must be > 0");
    assert!(kv_dim > 0, "kv_dim must be > 0");
    assert_eq!(k_in.len(), kv_dim, "k_in length must match kv_dim");
    assert_eq!(v_in.len(), kv_dim, "v_in length must match kv_dim");

    let ring_elems = window_size
        .checked_mul(kv_dim)
        .expect("window_size * kv_dim overflowed");
    let required_elems = ring_elems
        .checked_mul(2)
        .expect("2 * window_size * kv_dim overflowed");
    assert!(
        layer_kv.len() >= required_elems,
        "layer_kv len {} < required {required_elems}",
        layer_kv.len()
    );

    let slot = pos % window_size;
    let active_len = (pos + 1).min(window_size);
    let slot_start = slot * kv_dim;
    let slot_end = slot_start + kv_dim;

    let (k_ring, rest) = layer_kv.split_at_mut(ring_elems);
    let v_ring = &mut rest[..ring_elems];

    k_ring[slot_start..slot_end].copy_from_slice(k_in);
    v_ring[slot_start..slot_end].copy_from_slice(v_in);

    let active_elems = active_len * kv_dim;
    (&k_ring[..active_elems], &v_ring[..active_elems], active_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::GgufFile;
    use jero_core::QuantType;
    use jero_core::test_helpers::create_synthetic_gguf_v3;

    #[test]
    fn test_kv_arena_allocation_and_indexing() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let model = GgufFile::parse(&bytes).expect("parse model");
        let meta = model.metadata();

        // 32 tokens -> 2 blocks of 16 tokens
        let mut arena = SingleSlotKvArena::new(meta, 32).expect("allocate kv arena");
        assert_eq!(arena.n_ctx(), 32);
        assert_eq!(arena.num_blocks(), 2);
        assert_eq!(arena.allocated_tokens(), 32);
        assert!(arena.total_elements() > 0);
        assert!(arena.aligned_bytes() > 0);

        // Test block slicing
        let block0 = arena.block_slice(0).expect("block 0 slice");
        assert_eq!(block0.len(), arena.elements_per_block());

        let block1 = arena.block_slice(1).expect("block 1 slice");
        assert_eq!(block1.len(), arena.elements_per_block());

        let block2_err = arena.block_slice(2).expect_err("block 2 out of bounds");
        assert!(matches!(block2_err, K3Error::TensorBoundsError { .. }));

        // Test token KV slice access
        {
            let (k_mut, v_mut) = arena
                .token_kv_slice_mut(5, 0)
                .expect("token 5 layer 0 mutable slice");
            assert_eq!(k_mut.len(), v_mut.len());
            k_mut[0] = 3.5;
            v_mut[0] = 2.5;
        }

        let (k_ref, v_ref) = arena
            .token_kv_slice(5, 0)
            .expect("token 5 layer 0 immutable slice");
        assert!((k_ref[0] - 3.5).abs() < f32::EPSILON);
        assert!((v_ref[0] - 2.5).abs() < f32::EPSILON);

        // Token position out of n_ctx bound returns ContextLengthExceeded (ADR-027)
        let oob_err = arena
            .token_kv_slice(32, 0)
            .expect_err("token 32 exceeds n_ctx=32");
        assert_eq!(
            oob_err,
            K3Error::ContextLengthExceeded {
                requested_tokens: 33,
                max_context_tokens: 32,
            }
        );
    }

    #[test]
    fn test_kv_arena_from_gguf_honors_explicit_key_length() {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        // embedding_length = 256, head_count = 8 (256/8 = 32), but explicit key_length = 64
        let extra_kvs = [
            ("llama.block_count", GgufValue::Uint32(2)),
            ("llama.embedding_length", GgufValue::Uint32(256)),
            ("llama.attention.head_count", GgufValue::Uint32(8)),
            ("llama.attention.head_count_kv", GgufValue::Uint32(2)),
            ("llama.attention.key_length", GgufValue::Uint32(64)),
        ];
        let bytes = create_synthetic_gguf_v3_with_kvs("llama", 32, &extra_kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");

        let arena = SingleSlotKvArena::from_gguf(&gguf, 16).expect("allocate kv arena from gguf");
        assert_eq!(arena.layer_count(), 2);
        assert_eq!(arena.head_count_kv(), 2);
        assert_eq!(arena.head_dim(), 64);
        assert_eq!(arena.elements_per_token(), 2 * 2 * 2 * 64);
    }

    #[test]
    fn test_sliding_window_kv_ring_wraparound() {
        let window_size = 4;
        let kv_dim = 3;
        let mut layer_kv = vec![0.0_f32; 2 * window_size * kv_dim];

        // Positions 0..4 fill slots 0..3 with growing active_len 1..=4
        for pos in 0..4 {
            let marker = f32::from(u16::try_from(pos + 1).expect("pos fits u16"));
            let k = [marker, marker * 2.0, marker * 3.0];
            let v = [-marker, -marker * 2.0, -marker * 3.0];
            let (k_slice, v_slice, active_len) =
                write_and_slice_sliding_kv(&mut layer_kv, pos, window_size, kv_dim, &k, &v);
            assert_eq!(active_len, pos + 1);
            assert_eq!(k_slice.len(), active_len * kv_dim);
            assert_eq!(v_slice.len(), active_len * kv_dim);
        }

        // Positions 4..10 wrap around mod 4 while clamping active_len to window_size=4
        for pos in 4..10 {
            let marker = f32::from(u16::try_from(pos + 1).expect("pos fits u16"));
            let k = [marker, marker * 2.0, marker * 3.0];
            let v = [-marker, -marker * 2.0, -marker * 3.0];
            let (k_slice, v_slice, active_len) =
                write_and_slice_sliding_kv(&mut layer_kv, pos, window_size, kv_dim, &k, &v);
            assert_eq!(active_len, window_size);
            assert_eq!(k_slice.len(), window_size * kv_dim);
            assert_eq!(v_slice.len(), window_size * kv_dim);

            let slot = pos % window_size;
            assert!((k_slice[slot * kv_dim] - marker).abs() < f32::EPSILON);
            assert!((v_slice[slot * kv_dim] + marker).abs() < f32::EPSILON);
        }

        // After pos=9 (marker=10 -> slot 1), ring slots hold tokens:
        // slot 0 <- pos=8 (9.0), slot 1 <- pos=9 (10.0), slot 2 <- pos=6 (7.0), slot 3 <- pos=7 (8.0)
        assert!((layer_kv[0] - 9.0).abs() < f32::EPSILON);
        assert!((layer_kv[kv_dim] - 10.0).abs() < f32::EPSILON);
        assert!((layer_kv[2 * kv_dim] - 7.0).abs() < f32::EPSILON);
        assert!((layer_kv[3 * kv_dim] - 8.0).abs() < f32::EPSILON);

        let view = SlidingWindowKvView {
            keys: &layer_kv[..window_size * kv_dim],
            values: &layer_kv[window_size * kv_dim..],
            active_len: window_size,
            slot_idx: 9 % window_size,
        };
        assert_eq!(view.slot_idx, 1);
        assert_eq!(view.active_len, 4);
    }
}
