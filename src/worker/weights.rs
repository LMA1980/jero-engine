//! Read-only memory-mapped, GGUF-validated, and `mlock`-backed model weights management
//! for `jero-engine`.
//!
//! Unifies worker weight admission with [`LockedModelWeights`], `CapabilityVerifier`
//! (`verify_model_capabilities`), bounded `RLIMIT_MEMLOCK` admission checks
//! (`check_memory_admission_with`), and `ArchRecipe` + `SingleSlotKvArena` instantiation
//! across both filesystem paths and `SCM_RIGHTS` / `memfd_create` file descriptors.

use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::Path;

use jero_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;
use jero_core::{ArchRecipe, K3Error, ModelMetadata, ModelWeights, TensorView};

use crate::affinity::WorkerPool;
use crate::kv::SingleSlotKvArena;
use crate::memory::{
    MemoryFootprint, MemoryInspector, SystemMemoryInspector, check_memory_admission_with,
};
use crate::weights::{LockedModelWeights, ModelView, RepackSummary};

use super::error::WorkerError;

/// Admitted, read-only memory-mapped, and locked GGUF model weights for `jero-engine`.
#[derive(Debug)]
pub struct LockedWeights {
    inner: LockedModelWeights,
    recipe: Box<dyn ArchRecipe>,
    footprint: MemoryFootprint,
}

impl LockedWeights {
    /// Opens a GGUF model file at `path`, acquires `flock(LOCK_SH | LOCK_NB)`,
    /// validates GGUF headers/tensors and `CapabilityVerifier`, checks bounded memory
    /// admission, and locks pages in RAM via `mlock`.
    ///
    /// # Errors
    /// - Returns [`WorkerError::ResourceExhausted`] (exit 12) if file does not exist or has 0 bytes.
    /// - Returns [`WorkerError::CapabilityMismatch`] (exit 10) if locked exclusively or unsupported arch/quant.
    /// - Returns [`WorkerError::InsufficientLockableMemory`] (exit 11) if file exceeds 28 GiB ceiling,
    ///   admitted footprint exceeds RAM/RLIMIT_MEMLOCK headroom, or `mlock` fails with `ENOMEM`.
    /// - Returns [`WorkerError::TensorBoundsError`] (exit 14) if GGUF header/tensor descriptors are malformed.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WorkerError> {
        Self::open_with_admission(path, None, &SystemMemoryInspector::new())
    }

    /// Opens a GGUF model file at `path` with explicit context window `n_ctx` and memory `inspector`.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_admission<P: AsRef<Path>, I: MemoryInspector>(
        path: P,
        n_ctx: Option<usize>,
        inspector: &I,
    ) -> Result<Self, WorkerError> {
        let p = path.as_ref();
        if !p.exists() {
            return Err(WorkerError::ResourceExhausted(format!(
                "model file does not exist: {}",
                p.display()
            )));
        }

        let file = File::open(p).map_err(|err| {
            WorkerError::ResourceExhausted(format!(
                "failed to open model file '{}': {err}",
                p.display()
            ))
        })?;

        Self::from_file_with_label(file, &p.display().to_string(), n_ctx, inspector)
    }

    /// Admits an `SCM_RIGHTS` / `memfd_create` [`OwnedFd`] through GGUF header/tensor validation,
    /// `CapabilityVerifier`, bounded memory admission, and `mlock`.
    ///
    /// # Errors
    /// Returns structured admission errors (`CapabilityMismatch`, `InsufficientLockableMemory`,
    /// `TensorBoundsError`, or `ResourceExhausted`).
    pub fn from_fd(fd: OwnedFd) -> Result<Self, WorkerError> {
        Self::from_fd_with_admission(fd, None, &SystemMemoryInspector::new())
    }

    /// Admits an `SCM_RIGHTS` / `memfd_create` [`OwnedFd`] using explicit `n_ctx` and `inspector`.
    ///
    /// # Errors
    /// Same as [`Self::from_fd`].
    pub fn from_fd_with_admission<I: MemoryInspector>(
        fd: OwnedFd,
        n_ctx: Option<usize>,
        inspector: &I,
    ) -> Result<Self, WorkerError> {
        Self::from_file_with_label(File::from(fd), "<scm_rights_fd>", n_ctx, inspector)
    }

    /// Admits an `SCM_RIGHTS` / `memfd_create` [`OwnedFd`] and instantiates the corresponding
    /// [`SingleSlotKvArena`] sized for `n_ctx` in one unified step.
    ///
    /// # Errors
    /// Same as [`Self::from_fd`] plus [`SingleSlotKvArena`] bounds errors.
    pub fn from_fd_and_instantiate<I: MemoryInspector>(
        fd: OwnedFd,
        n_ctx: usize,
        inspector: &I,
    ) -> Result<(Self, SingleSlotKvArena), WorkerError> {
        let weights = Self::from_fd_with_admission(fd, Some(n_ctx), inspector)?;
        let kv_arena = weights.instantiate_kv_arena_with(n_ctx, inspector)?;
        Ok((weights, kv_arena))
    }

    fn from_file_with_label<I: MemoryInspector>(
        file: File,
        label: &str,
        n_ctx: Option<usize>,
        inspector: &I,
    ) -> Result<Self, WorkerError> {
        let metadata = file.metadata().map_err(|e| {
            WorkerError::ResourceExhausted(format!(
                "failed to stat model descriptor '{label}': {e}"
            ))
        })?;

        let file_len = metadata.len();
        if file_len == 0 {
            return Err(WorkerError::ResourceExhausted(format!(
                "model file '{label}' is empty (0 bytes)"
            )));
        }

        if file_len > DEFAULT_LIMIT_MEMLOCK_BYTES {
            return Err(WorkerError::InsufficientLockableMemory(format!(
                "model weight size ({file_len} bytes) exceeds operator LimitMEMLOCK bound ({DEFAULT_LIMIT_MEMLOCK_BYTES} bytes)"
            )));
        }

        let (inner, recipe, footprint) = LockedModelWeights::from_file_with_admission_labeled(
            file, label, n_ctx, inspector, true,
        )?;

        Ok(Self {
            inner,
            recipe,
            footprint,
        })
    }

    /// Validates memory admission for `n_ctx` against [`SystemMemoryInspector`] and instantiates
    /// a [`SingleSlotKvArena`] matching this model's GGUF metadata.
    ///
    /// # Errors
    /// Returns [`WorkerError::InsufficientLockableMemory`], [`WorkerError::TensorBoundsError`], or
    /// [`WorkerError::CapabilityMismatch`] if `n_ctx` or KV footprint fails admission.
    pub fn instantiate_kv_arena(&self, n_ctx: usize) -> Result<SingleSlotKvArena, WorkerError> {
        self.instantiate_kv_arena_with(n_ctx, &SystemMemoryInspector::new())
    }

    /// Validates memory admission for `n_ctx` against `inspector` and instantiates
    /// a [`SingleSlotKvArena`] matching this model's GGUF metadata.
    ///
    /// # Errors
    /// Same as [`Self::instantiate_kv_arena`].
    pub fn instantiate_kv_arena_with<I: MemoryInspector>(
        &self,
        n_ctx: usize,
        inspector: &I,
    ) -> Result<SingleSlotKvArena, WorkerError> {
        let _ = check_memory_admission_with(self.inner.metadata(), n_ctx, inspector)?;
        Ok(SingleSlotKvArena::new(self.inner.metadata(), n_ctx)?)
    }

    /// Returns mapped length of model weights in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if weights file length is 0.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if physical pages are locked in RAM via `mlock`.
    #[must_use]
    pub const fn is_locked(&self) -> bool {
        self.inner.is_locked()
    }

    /// Returns byte slice view of mapped weights.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Returns the underlying [`LockedModelWeights`].
    #[must_use]
    pub const fn inner(&self) -> &LockedModelWeights {
        &self.inner
    }

    /// Returns parsed GGUF [`ModelMetadata`].
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        self.inner.metadata()
    }

    /// Returns the hydrated [`ArchRecipe`] admitted for this model.
    #[must_use]
    pub fn recipe(&self) -> &dyn ArchRecipe {
        &*self.recipe
    }

    /// Returns the admitted [`MemoryFootprint`] computed at load time.
    #[must_use]
    pub const fn footprint(&self) -> &MemoryFootprint {
        &self.footprint
    }

    /// Returns a lifetime-bound [`ModelView`] pairing these locked weights with their repacked store.
    #[must_use]
    pub const fn model_view(&self) -> ModelView<'_> {
        self.inner.model_view()
    }

    /// Eagerly builds and caches repacked 2D weight tensors in parallel across `pool` once per model.
    ///
    /// # Errors
    /// Returns [`K3Error`] converted into [`WorkerError`] if repacking fails bounds verification.
    pub fn prepare_repacked_weights(
        &self,
        pool: Option<&WorkerPool>,
    ) -> Result<RepackSummary, WorkerError> {
        Ok(self.inner.prepare_repacked_weights(pool)?)
    }

    /// Returns the current [`RepackSummary`] residency and reuse telemetry.
    #[must_use]
    pub fn repack_summary(&self) -> RepackSummary {
        self.inner.repack_summary()
    }
}

impl ModelWeights for LockedWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        self.inner.tensor(name)
    }

    fn tensor_count(&self) -> usize {
        self.inner.tensor_count()
    }

    fn total_weight_bytes(&self) -> usize {
        self.inner.total_weight_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "jero-arch-synthetic")]
    use crate::memory::MockMemoryInspector;
    use jero_core::ipc::protocol::StatusCode;
    use jero_core::test_helpers::create_synthetic_gguf_v3;
    use std::io::Write;
    #[cfg(feature = "jero-arch-synthetic")]
    use std::os::fd::AsRawFd;

    fn create_anonymous_fd_with_bytes(name: &str, bytes: &[u8]) -> OwnedFd {
        #[cfg(target_os = "linux")]
        {
            let cname = std::ffi::CString::new(name).expect("valid CString");
            // SAFETY: memfd_create allocates an in-memory anonymous file descriptor with MFD_CLOEXEC.
            let raw_fd = unsafe { libc::syscall(libc::SYS_memfd_create, cname.as_ptr(), 1) as i32 };
            if raw_fd >= 0 {
                // SAFETY: raw_fd is newly returned by memfd_create.
                let mut file = unsafe { <File as std::os::fd::FromRawFd>::from_raw_fd(raw_fd) };
                file.write_all(bytes).expect("write memfd bytes");
                return file.into();
            }
        }

        let unique = format!("k3_worker_memfd_{name}_{}.gguf", std::process::id());
        let temp = std::env::temp_dir().join(unique);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)
            .expect("create backing file");
        let _ = std::fs::remove_file(&temp);
        (&file).write_all(bytes).expect("write fd bytes");
        file.into()
    }

    #[test]
    fn test_nonexistent_model() {
        let res = LockedWeights::open("/nonexistent/path/to/model.gguf");
        assert!(matches!(res, Err(WorkerError::ResourceExhausted(_))));
        if let Err(e) = res {
            assert_eq!(e.exit_code(), 12);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_empty_model() {
        let temp = std::env::temp_dir().join("k3_worker_empty_model.gguf");
        File::create(&temp).unwrap();
        let res = LockedWeights::open(&temp);
        let _ = std::fs::remove_file(&temp);
        assert!(matches!(res, Err(WorkerError::ResourceExhausted(_))));
        if let Err(e) = res {
            assert_eq!(e.exit_code(), 12);
        }
    }

    #[test]
    #[cfg(feature = "jero-arch-synthetic")]
    #[cfg_attr(miri, ignore)]
    fn test_valid_model_mapping() {
        let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let temp = std::env::temp_dir().join("k3_worker_valid_model.gguf");
        {
            let mut f = File::create(&temp).unwrap();
            f.write_all(&gguf_bytes).unwrap();
        }
        let weights = LockedWeights::open(&temp).expect("open valid miniature GGUF");
        assert_eq!(weights.len(), gguf_bytes.len());
        assert_eq!(weights.as_slice(), gguf_bytes.as_slice());
        assert_eq!(weights.recipe().architecture_id(), "synthetic");
        assert_eq!(weights.metadata().architecture(), "synthetic");
        drop(weights);
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    #[cfg(feature = "jero-arch-synthetic")]
    #[cfg_attr(miri, ignore)]
    fn test_exclusive_lock_conflict() {
        let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let temp = std::env::temp_dir().join("k3_worker_locked_model.gguf");
        {
            let mut f = File::create(&temp).unwrap();
            f.write_all(&gguf_bytes).unwrap();
        }
        let f_ex = File::open(&temp).unwrap();
        // SAFETY: Test acquires exclusive flock to test worker error handling.
        let flock_res = unsafe { libc::flock(f_ex.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(flock_res, 0);

        let res = LockedWeights::open(&temp);
        assert!(matches!(res, Err(WorkerError::CapabilityMismatch(_))));
        if let Err(e) = res {
            assert_eq!(e.exit_code(), 10);
        }

        // SAFETY: Test unlocks flock after verification.
        unsafe { libc::flock(f_ex.as_raw_fd(), libc::LOCK_UN) };
        drop(f_ex);
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    #[cfg(feature = "jero-arch-synthetic")]
    #[cfg_attr(miri, ignore)]
    fn test_memfd_admission_valid_miniature_gguf() {
        let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let fd = create_anonymous_fd_with_bytes("valid_synthetic", &gguf_bytes);

        let (weights, kv_arena) =
            LockedWeights::from_fd_and_instantiate(fd, 32, &SystemMemoryInspector::new())
                .expect("memfd miniature GGUF must be admitted cleanly");
        assert_eq!(weights.recipe().architecture_id(), "synthetic");
        assert_eq!(kv_arena.n_ctx(), 32);
        assert_eq!(weights.len(), gguf_bytes.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_memfd_admission_rejects_truncated_gguf_with_tensor_bounds_error() {
        let mut gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        gguf_bytes.truncate(20);
        let fd = create_anonymous_fd_with_bytes("truncated_gguf", &gguf_bytes);

        let err = LockedWeights::from_fd(fd).expect_err("truncated GGUF must fail closed");
        assert!(
            matches!(err, WorkerError::TensorBoundsError(_)),
            "expected TensorBoundsError, got {err:?}"
        );
        assert_eq!(err.status_code(), StatusCode::TensorBoundsError);
        assert_eq!(err.exit_code(), 14);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_memfd_admission_rejects_unsupported_arch_with_capability_mismatch() {
        let gguf_bytes = create_synthetic_gguf_v3("unsupported_arch_v99", 32, &[]);
        let fd = create_anonymous_fd_with_bytes("unsupported_arch", &gguf_bytes);

        let err = LockedWeights::from_fd(fd).expect_err("unsupported arch must fail closed");
        assert!(
            matches!(err, WorkerError::CapabilityMismatch(_)),
            "expected CapabilityMismatch, got {err:?}"
        );
        assert_eq!(err.status_code(), StatusCode::CapabilityMismatch);
        assert_eq!(err.exit_code(), 10);
    }

    #[test]
    #[cfg(feature = "jero-arch-synthetic")]
    #[cfg_attr(miri, ignore)]
    fn test_memfd_admission_rejects_insufficient_memlock_headroom() {
        let gguf_bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let fd = create_anonymous_fd_with_bytes("low_memlock", &gguf_bytes);
        let constrained = MockMemoryInspector::new(128, 128);

        let err = LockedWeights::from_fd_with_admission(fd, Some(32), &constrained)
            .expect_err("constrained RLIMIT_MEMLOCK must fail closed");
        assert!(
            matches!(err, WorkerError::InsufficientLockableMemory(_)),
            "expected InsufficientLockableMemory, got {err:?}"
        );
        assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
        assert_eq!(err.exit_code(), 11);
    }
}
