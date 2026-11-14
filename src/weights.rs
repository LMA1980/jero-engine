//! Locked read-only weight residency mapping and shared file advisory locking.
//!
//! Maps model files read-only (`mmap`), applies best-effort huge page hints (`madvise`),
//! locks weights permanently into physical RAM (`mlock`), and holds shared advisory locks
//! (`flock(LOCK_SH)`) to prevent concurrent garbage collection deletions (ADR-037, ADR-040).

use std::collections::BTreeMap;
use std::fs::File;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::ptr;

use jero_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;
use jero_core::{
    ArchRecipe, GgufFile, K3Error, ModelMetadata, ModelWeights, TensorDescriptor, TensorView,
};

use crate::capability::verify_model_capabilities;
use crate::memory::{
    MemoryFootprint, MemoryInspector, SystemMemoryInspector, check_memory_admission_with,
};

/// Locked in-memory resident model weights backed by a read-only `mmap` mapping
/// with an active shared file lock (`flock(LOCK_SH)`).
#[derive(Debug)]
pub struct LockedModelWeights {
    file: File,
    ptr: *mut u8,
    len: usize,
    is_locked: bool,
    metadata: ModelMetadata,
    tensor_index: BTreeMap<String, usize>,
}

// SAFETY: LockedModelWeights holds a read-only mapped buffer (PROT_READ) locked in physical memory.
// Reading from the buffer is entirely thread-safe and immutable.
unsafe impl Send for LockedModelWeights {}
// SAFETY: Concurrent immutable access across threads to read-only mmap memory is safe.
unsafe impl Sync for LockedModelWeights {}

type OpenedWeights = (
    LockedModelWeights,
    Option<Box<dyn ArchRecipe>>,
    Option<MemoryFootprint>,
);

type ParsedWeights = (
    ModelMetadata,
    Option<Box<dyn ArchRecipe>>,
    Option<MemoryFootprint>,
);

#[allow(clippy::too_many_lines)]
fn open_from_file_inner<I: MemoryInspector>(
    file: File,
    label: &str,
    admission: Option<(Option<usize>, &I)>,
    allow_unprivileged_mlock_eperm: bool,
) -> Result<OpenedWeights, K3Error> {
    let fd = file.as_raw_fd();

    // 1. Acquire shared advisory lock flock(LOCK_SH | LOCK_NB) per ADR-040
    // SAFETY: flock on a valid open file descriptor with LOCK_SH | LOCK_NB is non-blocking.
    let flock_res = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
    if flock_res != 0 {
        let last_err = std::io::Error::last_os_error();
        let raw_code = last_err.raw_os_error().unwrap_or(0);
        return Err(K3Error::CapabilityMismatch {
            architecture: "unknown".to_owned(),
            reason: format!(
                "model file '{label}' is locked exclusively by another process (flock error: {last_err} [errno: {raw_code}])"
            ),
        });
    }

    // 2. Query file length
    let metadata_fs = file.metadata().map_err(|err| {
        // SAFETY: Releasing the shared lock acquired above.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: format!("failed to read file metadata: {err}"),
        }
    })?;

    let file_len = usize::try_from(metadata_fs.len()).map_err(|_| {
        // SAFETY: Releasing the shared lock.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "file length exceeds usize capacity".to_owned(),
        }
    })?;

    if file_len == 0 {
        // SAFETY: Releasing the shared lock.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return Err(K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "model file is empty (0 bytes)".to_owned(),
        });
    }

    let file_len_u64 = u64::try_from(file_len).unwrap_or(u64::MAX);
    if file_len_u64 > DEFAULT_LIMIT_MEMLOCK_BYTES {
        // SAFETY: Releasing the shared lock before returning LimitMEMLOCK refusal.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: file_len_u64,
            weight_bytes: file_len_u64,
            kv_bytes: 0,
            available_bytes: DEFAULT_LIMIT_MEMLOCK_BYTES,
            rlimit_memlock_bytes: DEFAULT_LIMIT_MEMLOCK_BYTES,
        });
    }

    // 3. Map read-only without mlock first so pre-flight checks fail before locking RAM
    // SAFETY: mmap is called with valid length > 0, valid fd, PROT_READ, MAP_PRIVATE.
    let raw_void_ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            file_len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            fd,
            0,
        )
    };
    if raw_void_ptr == libc::MAP_FAILED {
        // SAFETY: Releasing flock upon mapping failure.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: file_len_u64,
            weight_bytes: file_len_u64,
            kv_bytes: 0,
            available_bytes: 0,
            rlimit_memlock_bytes: 0,
        });
    }
    let raw_ptr = raw_void_ptr.cast::<u8>();

    // 4. Parse GGUF v3 header & verify capabilities and memory admission before mlock
    // SAFETY: raw_ptr is valid for reads up to file_len.
    let slice = unsafe { std::slice::from_raw_parts(raw_ptr, file_len) };
    let parsed = (|| -> Result<ParsedWeights, K3Error> {
        let gguf = GgufFile::parse(slice)?;
        if let Some((opt_n_ctx, inspector)) = admission {
            let recipe = verify_model_capabilities(&gguf)?;
            let effective_n_ctx = if let Some(n) = opt_n_ctx {
                n
            } else {
                let meta_ctx = gguf.metadata().context_length();
                if meta_ctx > 0 { meta_ctx.min(64) } else { 64 }
            };
            let footprint =
                check_memory_admission_with(gguf.metadata(), effective_n_ctx, inspector)?;
            Ok((gguf.metadata().clone(), Some(recipe), Some(footprint)))
        } else {
            Ok((gguf.metadata().clone(), None, None))
        }
    })();

    let (metadata, recipe_opt, footprint_opt) = match parsed {
        Ok(res) => res,
        Err(err) => {
            // SAFETY: Unmapping and releasing flock on pre-flight rejection before any mlock.
            unsafe {
                libc::munmap(raw_void_ptr, file_len);
                libc::flock(fd, libc::LOCK_UN);
            }
            return Err(err);
        }
    };

    // 5. Best-effort huge page hint and mlock() (ADR-037)
    // SAFETY: raw_void_ptr is a valid mapped memory region of length file_len.
    #[cfg(all(target_os = "linux", not(miri)))]
    unsafe {
        libc::madvise(raw_void_ptr, file_len, libc::MADV_HUGEPAGE);
    }
    // SAFETY: mlock is called on the valid mapped range [raw_void_ptr .. raw_void_ptr + file_len].
    #[cfg(not(miri))]
    let mlock_res = unsafe { libc::mlock(raw_void_ptr, file_len) };
    #[cfg(miri)]
    let mlock_res = 0;

    let is_locked = if mlock_res == 0 {
        true
    } else {
        let last_err = std::io::Error::last_os_error();
        if allow_unprivileged_mlock_eperm && last_err.raw_os_error() == Some(libc::EPERM) {
            false
        } else {
            let (avail, rlim) = admission.map_or((0, 0), |(_, insp)| {
                (
                    insp.available_memory_bytes().unwrap_or(0),
                    insp.rlimit_memlock_bytes().unwrap_or(0),
                )
            });
            // SAFETY: Unmapping and releasing flock on mlock failure.
            unsafe {
                libc::munmap(raw_void_ptr, file_len);
                libc::flock(fd, libc::LOCK_UN);
            }
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: footprint_opt
                    .as_ref()
                    .map_or(file_len_u64, |fp| fp.total_required_bytes),
                weight_bytes: footprint_opt
                    .as_ref()
                    .map_or(file_len_u64, |fp| fp.weight_bytes),
                kv_bytes: footprint_opt.as_ref().map_or(0, |fp| fp.kv_bytes),
                available_bytes: avail,
                rlimit_memlock_bytes: rlim,
            });
        }
    };

    let mut tensor_index = BTreeMap::new();
    for (i, desc) in metadata.tensors().iter().enumerate() {
        tensor_index.insert(desc.name().to_owned(), i);
    }

    Ok((
        LockedModelWeights {
            file,
            ptr: raw_ptr,
            len: file_len,
            is_locked,
            metadata,
            tensor_index,
        },
        recipe_opt,
        footprint_opt,
    ))
}

impl LockedModelWeights {
    /// Opens the GGUF model file at `path`, acquires an advisory shared lock (`LOCK_SH`),
    /// maps the file read-only, parses GGUF v3 metadata/tensor descriptors, and locks
    /// pages into physical memory via `mlock` (ADR-037, ADR-040).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if the file cannot be opened or if another process
    ///   holds an exclusive lock (`LOCK_EX`).
    /// - Returns [`K3Error::InsufficientLockableMemory`] if `mlock` fails or size exceeds `DEFAULT_LIMIT_MEMLOCK_BYTES`.
    /// - Returns [`K3Error::TensorBoundsError`] if GGUF v3 parsing fails or the file is truncated.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, K3Error> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|err| K3Error::CapabilityMismatch {
            architecture: "unknown".to_owned(),
            reason: format!("failed to open model file '{}': {err}", path_ref.display()),
        })?;
        let label = path_ref.display().to_string();
        let (weights, _, _) =
            open_from_file_inner::<SystemMemoryInspector>(file, &label, None, false)?;
        Ok(weights)
    }

    /// Wraps an already-opened [`File`] (e.g., from `memfd_create` / `SCM_RIGHTS`), acquires `LOCK_SH`,
    /// maps read-only, parses GGUF v3 metadata/tensor bounds, and locks pages via `mlock`.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn from_file(file: File) -> Result<Self, K3Error> {
        let (weights, _, _) =
            open_from_file_inner::<SystemMemoryInspector>(file, "<fd>", None, false)?;
        Ok(weights)
    }

    /// Wraps an [`OwnedFd`] (such as an `SCM_RIGHTS` or `memfd_create` descriptor) via [`Self::from_file`].
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn from_fd(fd: OwnedFd) -> Result<Self, K3Error> {
        Self::from_file(File::from(fd))
    }

    /// Opens the GGUF model file at `path`, acquires `flock(LOCK_SH | LOCK_NB)`, parses the
    /// GGUF v3 header, verifies architecture and quantization capabilities, performs pre-flight
    /// physical memory admission against `inspector`, and only then locks pages into physical RAM
    /// via `mlock` (US-01.2, US-01.3, US-01.7.2, ADR-023, ADR-037, ADR-040).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if the file cannot be opened, is locked
    ///   exclusively, or uses an unsupported architecture or quantization format.
    /// - Returns [`K3Error::InsufficientLockableMemory`] if pre-flight RAM admission or `mlock` fails.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `n_ctx` exceeds the model's context window.
    /// - Returns [`K3Error::TensorBoundsError`] if the GGUF v3 binary header or tensor bounds are invalid.
    pub fn open_with_admission<P: AsRef<Path>, I: MemoryInspector>(
        path: P,
        n_ctx: usize,
        inspector: &I,
    ) -> Result<(Self, Box<dyn ArchRecipe>, MemoryFootprint), K3Error> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|err| K3Error::CapabilityMismatch {
            architecture: "unknown".to_owned(),
            reason: format!("failed to open model file '{}': {err}", path_ref.display()),
        })?;
        let label = path_ref.display().to_string();
        Self::from_file_with_admission_labeled(file, &label, Some(n_ctx), inspector, false)
    }

    /// Wraps an already-opened [`File`] with full GGUF capability verification and bounded memory
    /// admission against `inspector`. When `n_ctx` is `None`, uses `min(metadata.context_length(), 64)`.
    ///
    /// # Errors
    /// Same as [`Self::open_with_admission`].
    pub fn from_file_with_admission<I: MemoryInspector>(
        file: File,
        n_ctx: Option<usize>,
        inspector: &I,
        allow_unprivileged_mlock_eperm: bool,
    ) -> Result<(Self, Box<dyn ArchRecipe>, MemoryFootprint), K3Error> {
        Self::from_file_with_admission_labeled(
            file,
            "<fd>",
            n_ctx,
            inspector,
            allow_unprivileged_mlock_eperm,
        )
    }

    /// Wraps an [`OwnedFd`] (`SCM_RIGHTS` / `memfd_create`) with full GGUF capability verification
    /// and memory admission against `inspector`.
    ///
    /// # Errors
    /// Same as [`Self::open_with_admission`].
    pub fn from_fd_with_admission<I: MemoryInspector>(
        fd: OwnedFd,
        n_ctx: Option<usize>,
        inspector: &I,
        allow_unprivileged_mlock_eperm: bool,
    ) -> Result<(Self, Box<dyn ArchRecipe>, MemoryFootprint), K3Error> {
        Self::from_file_with_admission(
            File::from(fd),
            n_ctx,
            inspector,
            allow_unprivileged_mlock_eperm,
        )
    }

    /// Labeled internal constructor shared by path-based and FD-based admission paths.
    #[doc(hidden)]
    pub fn from_file_with_admission_labeled<I: MemoryInspector>(
        file: File,
        label: &str,
        n_ctx: Option<usize>,
        inspector: &I,
        allow_unprivileged_mlock_eperm: bool,
    ) -> Result<(Self, Box<dyn ArchRecipe>, MemoryFootprint), K3Error> {
        let (weights, recipe_opt, footprint_opt) = open_from_file_inner(
            file,
            label,
            Some((n_ctx, inspector)),
            allow_unprivileged_mlock_eperm,
        )?;
        let recipe = recipe_opt.ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: weights.metadata.architecture().to_owned(),
            reason: "internal admission error: missing ArchRecipe".to_owned(),
        })?;
        let footprint = footprint_opt.ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: weights.metadata.architecture().to_owned(),
            reason: "internal admission error: missing MemoryFootprint".to_owned(),
        })?;
        Ok((weights, recipe, footprint))
    }

    /// Returns `true` if pages are locked into physical RAM via `mlock`.
    #[must_use]
    pub const fn is_locked(&self) -> bool {
        self.is_locked
    }

    /// Returns the parsed [`ModelMetadata`] for the locked model.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Returns the mapped file length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the mapped file length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw contiguous byte slice of the locked model file.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: self.ptr is valid for reads up to self.len while this struct is live.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns the underlying file reference.
    #[must_use]
    pub const fn file(&self) -> &File {
        &self.file
    }
}

impl ModelWeights for LockedModelWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        if let Some(&idx) = self.tensor_index.get(name) {
            let desc: &TensorDescriptor = &self.metadata.tensors()[idx];
            return TensorView::from_bytes(
                desc.name(),
                desc.quant(),
                desc.shape(),
                desc.absolute_offset(),
                self.as_bytes(),
            );
        }

        // Tied-embedding fallback (ADR-023 / Qwen2 / Qwen3 / Qwen3.5): when `output.weight`
        // is absent from an autoregressive (non-BitNet) GGUF, alias `token_embd.weight`
        // zero-copy over the already-locked byte range without doubling mlock residency.
        if name == "output.weight"
            && crate::memory::architecture_requires_kv_cache(self.metadata.architecture())
            && !self.metadata.architecture().starts_with("bitnet")
            && let Some(&emb_idx) = self.tensor_index.get("token_embd.weight")
        {
            let desc: &TensorDescriptor = &self.metadata.tensors()[emb_idx];
            return TensorView::from_bytes(
                "output.weight",
                desc.quant(),
                desc.shape(),
                desc.absolute_offset(),
                self.as_bytes(),
            );
        }

        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: self.len,
            reason: format!("tensor '{name}' not found in locked model weights"),
        })
    }

    fn tensor_count(&self) -> usize {
        self.metadata.tensor_count()
    }

    fn total_weight_bytes(&self) -> usize {
        self.metadata.total_weight_bytes()
    }
}

impl Drop for LockedModelWeights {
    fn drop(&mut self) {
        let fd = self.file.as_raw_fd();
        let ptr = self.ptr.cast::<libc::c_void>();
        let len = self.len;

        // SAFETY: self.ptr and self.len were mapped by mmap and conditionally locked with mlock.
        // We munlock (when locked), munmap, and release the flock advisory lock.
        unsafe {
            #[cfg(not(miri))]
            if self.is_locked {
                libc::munlock(ptr, len);
            }
            libc::munmap(ptr, len);
            libc::flock(fd, libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::QuantType;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use std::io::Write;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_locked_model_flock_shared_blocks_exclusive() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test_model_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&file_path).expect("create temp model");
            f.write_all(&bytes).expect("write bytes");
            f.sync_all().expect("sync temp model");
        }

        // Open locked model
        let locked_model = LockedModelWeights::open(&file_path).expect("open locked model");
        assert_eq!(locked_model.metadata().architecture(), "synthetic");
        assert_eq!(locked_model.tensor_count(), 2);
        assert!(locked_model.contains_tensor("blk.0.attn_q.weight"));
        assert!(locked_model.contains_tensor("output_norm.weight"));

        let q_tensor = locked_model
            .tensor("blk.0.attn_q.weight")
            .expect("find tensor");
        assert_eq!(q_tensor.quant(), QuantType::Q4_K_M);
        assert_eq!(q_tensor.byte_len(), 288);

        // Verify that another process/fd attempting flock(LOCK_EX | LOCK_NB) is blocked with EWOULDBLOCK
        let second_file = File::open(&file_path).expect("open second handle");
        // SAFETY: Attempting non-blocking exclusive lock on file already held with LOCK_SH.
        let ex_lock =
            unsafe { libc::flock(second_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_ne!(ex_lock, 0, "LOCK_EX must fail when LOCK_SH is held");
        let last_err = std::io::Error::last_os_error();
        assert_eq!(
            last_err.raw_os_error(),
            Some(libc::EWOULDBLOCK),
            "error must be EWOULDBLOCK (EAGAIN)"
        );

        // Clean up
        drop(locked_model);
        drop(second_file);
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_locked_model_vmlck_reflected() {
        let tensor_specs = [("weight", QuantType::F32, &[16][..])];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!("test_vmlck_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&file_path).expect("create temp model");
            f.write_all(&bytes).expect("write bytes");
            f.sync_all().expect("sync temp model");
        }

        let locked = LockedModelWeights::open(&file_path).expect("open locked model");
        assert!(!locked.is_empty());

        // Read /proc/self/status to check VmLck
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let vmlck_line = status.lines().find(|l| l.starts_with("VmLck:"));
            assert!(
                vmlck_line.is_some(),
                "/proc/self/status should contain VmLck"
            );
        }

        drop(locked);
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_weights_tied_embedding_lookup() {
        let tensor_specs = [("token_embd.weight", QuantType::F32, &[4, 8][..])];
        let qwen2_bytes = create_synthetic_gguf_v3("qwen2", 32, &tensor_specs);
        let bitnet_bytes = create_synthetic_gguf_v3("bitnet", 32, &tensor_specs);

        let temp_dir = std::env::temp_dir();
        let qwen2_path = temp_dir.join(format!("test_tied_qwen2_{}.gguf", std::process::id()));
        let bitnet_path = temp_dir.join(format!("test_tied_bitnet_{}.gguf", std::process::id()));
        std::fs::write(&qwen2_path, &qwen2_bytes).expect("write qwen2 temp model");
        std::fs::write(&bitnet_path, &bitnet_bytes).expect("write bitnet temp model");

        let locked_qwen2 = LockedModelWeights::open(&qwen2_path).expect("open tied qwen2 model");
        let emb = locked_qwen2
            .tensor("token_embd.weight")
            .expect("token_embd.weight");
        let out = locked_qwen2
            .tensor("output.weight")
            .expect("tied output.weight should alias token_embd.weight");
        assert!(locked_qwen2.contains_tensor("output.weight"));
        assert_eq!(out.name(), "output.weight");
        assert_eq!(out.quant(), emb.quant());
        assert_eq!(out.shape(), emb.shape());
        assert_eq!(out.offset(), emb.offset());
        assert_eq!(out.byte_len(), emb.byte_len());
        assert_eq!(locked_qwen2.total_weight_bytes(), emb.byte_len());

        let locked_bitnet = LockedModelWeights::open(&bitnet_path).expect("open bitnet model");
        assert!(
            locked_bitnet.tensor("output.weight").is_err(),
            "bitnet must not alias output.weight to token_embd.weight"
        );

        drop(locked_qwen2);
        drop(locked_bitnet);
        let _ = std::fs::remove_file(&qwen2_path);
        let _ = std::fs::remove_file(&bitnet_path);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tied_output_weight_zero_copy_alias() {
        let tied_specs = [("token_embd.weight", QuantType::F32, &[16, 4][..])];
        let tied_bytes = create_synthetic_gguf_v3("qwen3", 32, &tied_specs);

        let untied_specs = [
            ("token_embd.weight", QuantType::F32, &[16, 4][..]),
            ("output.weight", QuantType::F32, &[16, 4][..]),
        ];
        let untied_bytes = create_synthetic_gguf_v3("qwen3", 32, &untied_specs);

        let temp_dir = std::env::temp_dir();
        let tied_path = temp_dir.join(format!("test_tied_alias_{}.gguf", std::process::id()));
        let untied_path = temp_dir.join(format!("test_untied_alias_{}.gguf", std::process::id()));
        std::fs::write(&tied_path, &tied_bytes).expect("write tied model");
        std::fs::write(&untied_path, &untied_bytes).expect("write untied model");

        let tied = LockedModelWeights::open(&tied_path).expect("open tied model");
        let tied_emb = tied.tensor("token_embd.weight").expect("token_embd.weight");
        let tied_out = tied.tensor("output.weight").expect("aliased output.weight");
        assert_eq!(tied_out.as_bytes().as_ptr(), tied_emb.as_bytes().as_ptr());
        assert_eq!(tied_out.as_bytes(), tied_emb.as_bytes());
        assert!(tied.metadata().has_tied_embeddings());

        let untied = LockedModelWeights::open(&untied_path).expect("open untied model");
        let untied_emb = untied
            .tensor("token_embd.weight")
            .expect("token_embd.weight");
        let untied_out = untied
            .tensor("output.weight")
            .expect("explicit output.weight");
        assert_ne!(
            untied_out.as_bytes().as_ptr(),
            untied_emb.as_bytes().as_ptr()
        );
        assert_ne!(untied_out.offset(), untied_emb.offset());
        assert!(!untied.metadata().has_tied_embeddings());

        drop(tied);
        drop(untied);
        let _ = std::fs::remove_file(&tied_path);
        let _ = std::fs::remove_file(&untied_path);
    }
}
