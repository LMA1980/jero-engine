//! Locked read-only weight residency mapping and shared file advisory locking.
//!
//! Maps model files read-only (`mmap`), applies best-effort huge page hints (`madvise`),
//! locks weights permanently into physical RAM (`mlock`), and holds shared advisory locks
//! (`flock(LOCK_SH)`) to prevent concurrent garbage collection deletions (ADR-037, ADR-040).

use std::collections::BTreeMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;

use k3_core::{
    ArchRecipe, GgufFile, K3Error, ModelMetadata, ModelWeights, TensorDescriptor, TensorView,
};

use crate::capability::verify_model_capabilities;
use crate::memory::{MemoryFootprint, MemoryInspector, check_memory_admission_with};

/// Locked in-memory resident model weights backed by a read-only `mmap` mapping
/// with an active shared file lock (`flock(LOCK_SH)`).
#[derive(Debug)]
pub struct LockedModelWeights {
    file: File,
    ptr: *mut u8,
    len: usize,
    metadata: ModelMetadata,
    tensor_index: BTreeMap<String, usize>,
}

// SAFETY: LockedModelWeights holds a read-only mapped buffer (PROT_READ) locked in physical memory.
// Reading from the buffer is entirely thread-safe and immutable.
unsafe impl Send for LockedModelWeights {}
// SAFETY: Concurrent immutable access across threads to read-only mmap memory is safe.
unsafe impl Sync for LockedModelWeights {}

fn map_and_lock_file(fd: std::os::fd::RawFd, file_len: usize) -> Result<*mut u8, K3Error> {
    #[cfg(target_os = "linux")]
    let map_flags = libc::MAP_PRIVATE | libc::MAP_POPULATE;
    #[cfg(not(target_os = "linux"))]
    let map_flags = libc::MAP_PRIVATE;

    // SAFETY: mmap is called with valid arguments: length > 0, valid fd, PROT_READ, MAP_PRIVATE.
    let ptr = unsafe { libc::mmap(ptr::null_mut(), file_len, libc::PROT_READ, map_flags, fd, 0) };

    if ptr == libc::MAP_FAILED {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
            weight_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
            kv_bytes: 0,
            available_bytes: 0,
            rlimit_memlock_bytes: 0,
        });
    }

    // 4. Best-effort huge page hint (ADR-037)
    // SAFETY: madvise is called on valid mapped range [ptr .. ptr + file_len].
    #[cfg(all(target_os = "linux", not(miri)))]
    unsafe {
        libc::madvise(ptr, file_len, libc::MADV_HUGEPAGE);
    }

    // 5. Mandatory mlock() (ADR-037)
    // SAFETY: mlock is called on valid mapped memory [ptr .. ptr + file_len].
    let mlock_res = unsafe { libc::mlock(ptr, file_len) };
    if mlock_res != 0 {
        // SAFETY: Unmapping memory upon mlock failure.
        unsafe {
            libc::munmap(ptr, file_len);
        }
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
            weight_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
            kv_bytes: 0,
            available_bytes: 0,
            rlimit_memlock_bytes: 0,
        });
    }

    Ok(ptr.cast::<u8>())
}

impl LockedModelWeights {
    /// Opens the GGUF model file at `path`, acquires an advisory shared lock (`LOCK_SH`),
    /// maps the file read-only with `MAP_POPULATE`, applies huge page hints, and locks
    /// pages into physical memory via `mlock` (ADR-037, ADR-040).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if the file cannot be opened or if another process
    ///   holds an exclusive lock (`LOCK_EX`).
    /// - Returns [`K3Error::InsufficientLockableMemory`] if `mlock` fails due to memory limits.
    /// - Returns [`K3Error::TensorBoundsError`] if GGUF v3 parsing fails or the file is truncated.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, K3Error> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|err| K3Error::CapabilityMismatch {
            architecture: "unknown".to_owned(),
            reason: format!("failed to open model file '{}': {err}", path_ref.display()),
        })?;

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
                    "model file '{}' is locked exclusively by another process (flock error: {last_err} [errno: {raw_code}])",
                    path_ref.display()
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

        let raw_ptr = match map_and_lock_file(fd, file_len) {
            Ok(p) => p,
            Err(err) => {
                // SAFETY: Releasing flock upon mapping failure.
                unsafe { libc::flock(fd, libc::LOCK_UN) };
                return Err(err);
            }
        };

        // 6. Parse GGUF v3 from mapped buffer
        // SAFETY: ptr is valid for reads up to file_len for the lifetime of this struct.
        let slice = unsafe { std::slice::from_raw_parts(raw_ptr, file_len) };
        let gguf = match GgufFile::parse(slice) {
            Ok(parsed) => parsed,
            Err(err) => {
                // Clean up mlock, mmap, and flock on parse error
                // SAFETY: Cleaning up locked memory and lock on parse error.
                unsafe {
                    libc::munlock(raw_ptr.cast(), file_len);
                    libc::munmap(raw_ptr.cast(), file_len);
                    libc::flock(fd, libc::LOCK_UN);
                }
                return Err(err);
            }
        };

        let metadata = gguf.metadata().clone();
        let mut tensor_index = BTreeMap::new();
        for (i, desc) in metadata.tensors().iter().enumerate() {
            tensor_index.insert(desc.name().to_owned(), i);
        }

        Ok(Self {
            file,
            ptr: raw_ptr,
            len: file_len,
            metadata,
            tensor_index,
        })
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
    #[allow(clippy::too_many_lines)]
    pub fn open_with_admission<P: AsRef<Path>, I: MemoryInspector>(
        path: P,
        n_ctx: usize,
        inspector: &I,
    ) -> Result<(Self, &'static dyn ArchRecipe, MemoryFootprint), K3Error> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|err| K3Error::CapabilityMismatch {
            architecture: "unknown".to_owned(),
            reason: format!("failed to open model file '{}': {err}", path_ref.display()),
        })?;

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
                    "model file '{}' is locked exclusively by another process (flock error: {last_err} [errno: {raw_code}])",
                    path_ref.display()
                ),
            });
        }

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

        // 2. Map read-only without mlock first to inspect GGUF metadata and run admission checks
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
            // SAFETY: Releasing flock on mmap failure.
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
                weight_bytes: u64::try_from(file_len).unwrap_or(u64::MAX),
                kv_bytes: 0,
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            });
        }
        let raw_ptr = raw_void_ptr.cast::<u8>();

        // 3. Parse GGUF v3 header & verify capabilities and memory admission before mlock
        // SAFETY: raw_ptr is valid for reads up to file_len.
        let slice = unsafe { std::slice::from_raw_parts(raw_ptr, file_len) };
        let preflight =
            (|| -> Result<(ModelMetadata, &'static dyn ArchRecipe, MemoryFootprint), K3Error> {
                let gguf = GgufFile::parse(slice)?;
                let recipe = verify_model_capabilities(&gguf)?;
                let footprint = check_memory_admission_with(gguf.metadata(), n_ctx, inspector)?;
                Ok((gguf.metadata().clone(), recipe, footprint))
            })();

        let (metadata, recipe, footprint) = match preflight {
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

        // 4. Best-effort huge page hint and mandatory mlock() (ADR-037)
        // SAFETY: raw_void_ptr is a valid mapped memory region of length file_len.
        #[cfg(all(target_os = "linux", not(miri)))]
        unsafe {
            libc::madvise(raw_void_ptr, file_len, libc::MADV_HUGEPAGE);
        }
        // SAFETY: mlock is called on the valid mapped range [raw_void_ptr .. raw_void_ptr + file_len].
        let mlock_res = unsafe { libc::mlock(raw_void_ptr, file_len) };
        if mlock_res != 0 {
            let avail = inspector.available_memory_bytes().unwrap_or(0);
            let rlim = inspector.rlimit_memlock_bytes().unwrap_or(0);
            // SAFETY: Unmapping and releasing flock on mlock failure.
            unsafe {
                libc::munmap(raw_void_ptr, file_len);
                libc::flock(fd, libc::LOCK_UN);
            }
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: footprint.total_required_bytes,
                weight_bytes: footprint.weight_bytes,
                kv_bytes: footprint.kv_bytes,
                available_bytes: avail,
                rlimit_memlock_bytes: rlim,
            });
        }

        let mut tensor_index = BTreeMap::new();
        for (i, desc) in metadata.tensors().iter().enumerate() {
            tensor_index.insert(desc.name().to_owned(), i);
        }

        Ok((
            Self {
                file,
                ptr: raw_ptr,
                len: file_len,
                metadata,
                tensor_index,
            },
            recipe,
            footprint,
        ))
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
        let &idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.len,
                reason: format!("tensor '{name}' not found in locked model weights"),
            })?;

        let desc: &TensorDescriptor = &self.metadata.tensors()[idx];
        TensorView::from_bytes(
            desc.name(),
            desc.quant(),
            desc.shape(),
            desc.absolute_offset(),
            self.as_bytes(),
        )
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

        // SAFETY: self.ptr and self.len were mapped by mmap and locked with mlock.
        // We munlock, munmap, and release the flock advisory lock.
        unsafe {
            libc::munlock(ptr, len);
            libc::munmap(ptr, len);
            libc::flock(fd, libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::QuantType;
    use k3_core::test_helpers::create_synthetic_gguf_v3;
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
}
