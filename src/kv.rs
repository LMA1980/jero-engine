//! Standalone in-process locked KV cache arena.
//!
//! Organizes KV cache memory into fixed 16-token blocks sized strictly for 1 active slot × `n_ctx`
//! tokens, locked in physical memory via `mlock()` and indexed via block tables (ADR-027, ADR-037, `kv-allocator.md`).

use std::ptr;

use k3_core::{K3Error, ModelMetadata, align_up};

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

        let layer_count = metadata.layer_count().max(1);
        let head_count_kv = metadata.head_count_kv().max(1);
        let head_dim = if metadata.head_count() > 0 {
            metadata.embedding_dim() / metadata.head_count()
        } else {
            metadata.embedding_dim()
        }
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
        let mlock_res = unsafe { libc::mlock(raw_ptr, layout.aligned_bytes) };
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
        let ptr = self.ptr.cast::<libc::c_void>();
        let bytes = self.aligned_bytes;

        // SAFETY: self.ptr was mapped by mmap and locked by mlock with size aligned_bytes.
        unsafe {
            libc::munlock(ptr, bytes);
            libc::munmap(ptr, bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::GgufFile;
    use k3_core::QuantType;
    use k3_core::test_helpers::create_synthetic_gguf_v3;

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
}
