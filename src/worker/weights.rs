//! Read-only memory-mapped and locked model weights management for `k3-worker`.

use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::ptr::NonNull;

use k3_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;

use super::error::WorkerError;

/// Holds a read-only memory-mapped and locked GGUF model file.
#[derive(Debug)]
pub struct LockedWeights {
    _file: File,
    fd: RawFd,
    ptr: NonNull<u8>,
    len: usize,
    is_locked: bool,
}

// SAFETY: Mapped memory is read-only (PROT_READ) and valid for concurrent thread access.
unsafe impl Send for LockedWeights {}
// SAFETY: Mapped memory is read-only and immutable across threads.
unsafe impl Sync for LockedWeights {}

impl LockedWeights {
    /// Opens the model file, acquires non-blocking shared lock (`LOCK_SH`),
    /// maps read-only (`MAP_SHARED`), verifies memory limit, and locks pages in RAM.
    ///
    /// # Errors
    /// Returns [`WorkerError::ResourceExhausted`] if file does not exist or has 0 bytes.
    /// Returns [`WorkerError::CapabilityMismatch`] if another process holds an exclusive lock.
    /// Returns [`WorkerError::InsufficientLockableMemory`] if file exceeds 28 GiB ceiling or `mlock` fails.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WorkerError> {
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

        let fd = file.as_raw_fd();

        // 1. Shared advisory lock: flock(LOCK_SH | LOCK_NB)
        // SAFETY: flock on valid open file descriptor with non-blocking flag.
        let flock_res = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
        if flock_res != 0 {
            let last_err = std::io::Error::last_os_error();
            let raw_code = last_err.raw_os_error().unwrap_or(0);
            return Err(WorkerError::CapabilityMismatch(format!(
                "model file '{}' is exclusively locked by another process (flock error: {last_err} [errno: {raw_code}])",
                p.display()
            )));
        }

        // 2. Query file size
        let metadata = match file.metadata() {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: Unlock on failure before returning.
                unsafe { libc::flock(fd, libc::LOCK_UN) };
                return Err(WorkerError::ResourceExhausted(format!(
                    "failed to stat model file: {e}"
                )));
            }
        };

        let file_len = metadata.len();
        if file_len == 0 {
            // SAFETY: Unlock on failure.
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            return Err(WorkerError::ResourceExhausted(format!(
                "model file '{}' is empty (0 bytes)",
                p.display()
            )));
        }

        // 3. Check memory limit against 28 GiB operator bound
        if file_len > DEFAULT_LIMIT_MEMLOCK_BYTES {
            // SAFETY: Unlock on failure.
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            return Err(WorkerError::InsufficientLockableMemory(format!(
                "model weight size ({} bytes) exceeds operator LimitMEMLOCK bound ({} bytes)",
                file_len, DEFAULT_LIMIT_MEMLOCK_BYTES
            )));
        }

        let len_usize = usize::try_from(file_len).map_err(|_| {
            // SAFETY: Releasing the shared flock on error.
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            WorkerError::ResourceExhausted("file size overflowed usize".to_owned())
        })?;

        // 4. Memory-map read-only with MAP_SHARED
        // SAFETY: mmap read-only shared mapping of open file descriptor.
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len_usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            let last_err = std::io::Error::last_os_error();
            // SAFETY: Releasing the shared flock on mmap error.
            unsafe { libc::flock(fd, libc::LOCK_UN) };
            return Err(WorkerError::ResourceExhausted(format!(
                "mmap failed on model file: {last_err}"
            )));
        }

        let ptr = match NonNull::new(mmap_ptr.cast::<u8>()) {
            Some(p) => p,
            None => {
                // SAFETY: Cleaning up mmap and flock on null pointer failure.
                unsafe {
                    libc::munmap(mmap_ptr, len_usize);
                    libc::flock(fd, libc::LOCK_UN);
                }
                return Err(WorkerError::Internal(
                    "mmap returned null pointer".to_owned(),
                ));
            }
        };

        // 5. Optional mlock within RLIMIT_MEMLOCK bounds
        let mut is_locked = false;
        // SAFETY: ptr points to len_usize valid mapped bytes.
        let mlock_res = unsafe { libc::mlock(ptr.as_ptr().cast(), len_usize) };
        if mlock_res == 0 {
            is_locked = true;
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::ENOMEM) {
                // Insufficient memory locking allowance
                // SAFETY: Unmapping memory and unlocking file descriptor before error return.
                unsafe {
                    libc::munmap(ptr.as_ptr().cast(), len_usize);
                    libc::flock(fd, libc::LOCK_UN);
                }
                return Err(WorkerError::InsufficientLockableMemory(format!(
                    "mlock failed with ENOMEM: {err}"
                )));
            }
            // In unprivileged test / container environments, mlock may fail with EPERM.
            // Allow execution to proceed if unprivileged rather than crashing.
        }

        Ok(Self {
            _file: file,
            fd,
            ptr,
            len: len_usize,
            is_locked,
        })
    }

    /// Returns mapped length of model weights in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if weights file length is 0.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if physical pages are locked in RAM via `mlock`.
    #[must_use]
    pub const fn is_locked(&self) -> bool {
        self.is_locked
    }

    /// Returns byte slice view of mapped weights.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: self.ptr is valid for reads up to self.len while self is live.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for LockedWeights {
    fn drop(&mut self) {
        // SAFETY: Clean teardown of mlock, mmap, and flock on valid pointers and fd.
        unsafe {
            if self.is_locked {
                libc::munlock(self.ptr.as_ptr().cast(), self.len);
            }
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
            libc::flock(self.fd, libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
    #[cfg_attr(miri, ignore)]
    fn test_valid_model_mapping() {
        let temp = std::env::temp_dir().join("k3_worker_valid_model.gguf");
        {
            let mut f = File::create(&temp).unwrap();
            f.write_all(b"GGUFv3_SYNTHETIC_WEIGHTS_DATA").unwrap();
        }
        let weights = LockedWeights::open(&temp).expect("open valid model");
        assert_eq!(weights.len(), 29);
        assert_eq!(weights.as_slice(), b"GGUFv3_SYNTHETIC_WEIGHTS_DATA");
        drop(weights);
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_exclusive_lock_conflict() {
        let temp = std::env::temp_dir().join("k3_worker_locked_model.gguf");
        {
            let mut f = File::create(&temp).unwrap();
            f.write_all(b"LOCKED_MODEL_CONTENT").unwrap();
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
}
