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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use jero_core::QuantType;
use jero_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;
use jero_core::{
    ArchRecipe, GGUF_MAGIC, GgufFile, K3Error, ModelMetadata, ModelWeights, SafeTensorsFile,
    TensorDescriptor, TensorView,
};
#[cfg(feature = "jero-kernels-rvv")]
use jero_kernels_rvv::repack::{
    Aligned128, BlockScaledTileLayout, PackedNibbleMatrix, RepackedWeight, repack_into_tile_range,
    repack_nibble_into_tile_range,
};

use crate::affinity::WorkerPool;
use crate::capability::{verify_model_capabilities, verify_safetensors_capabilities};
use crate::memory::{
    MemoryFootprint, MemoryInspector, SystemMemoryInspector, check_memory_admission_with,
};

/// Detected binary container format backing [`LockedModelWeights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelContainerFormat {
    /// GGUF v3 binary container (`GGUF` magic `0x4655_4747`).
    Gguf,
    /// Hugging Face `SafeTensors` binary container.
    SafeTensors,
}

impl ModelContainerFormat {
    /// Detects the binary container format from raw leading file bytes (`GGUF` magic vs `SafeTensors`).
    #[must_use]
    pub fn detect(bytes: &[u8]) -> Self {
        if bytes.len() >= 4
            && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == GGUF_MAGIC
        {
            Self::Gguf
        } else {
            Self::SafeTensors
        }
    }
}

/// Zero-copy parsed container view over either a [`GgufFile`] or a [`SafeTensorsFile`].
#[derive(Debug)]
pub enum ParsedModelContainer<'a> {
    /// Parsed GGUF v3 checkpoint.
    Gguf(GgufFile<'a>),
    /// Parsed `SafeTensors` checkpoint.
    SafeTensors(SafeTensorsFile<'a>),
}

impl<'a> ParsedModelContainer<'a> {
    /// Parses a memory-mapped checkpoint slice by inspecting magic bytes (`GGUF` vs `SafeTensors`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] on malformed input.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, K3Error> {
        match ModelContainerFormat::detect(bytes) {
            ModelContainerFormat::Gguf => Ok(Self::Gguf(GgufFile::parse(bytes)?)),
            ModelContainerFormat::SafeTensors => {
                Ok(Self::SafeTensors(SafeTensorsFile::from_bytes(bytes)?))
            }
        }
    }

    /// Returns the detected container format.
    #[must_use]
    pub const fn format(&self) -> ModelContainerFormat {
        match self {
            Self::Gguf(_) => ModelContainerFormat::Gguf,
            Self::SafeTensors(_) => ModelContainerFormat::SafeTensors,
        }
    }

    /// Returns the extracted [`ModelMetadata`] regardless of container format.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        match self {
            Self::Gguf(gguf) => gguf.metadata(),
            Self::SafeTensors(st) => st.metadata(),
        }
    }

    /// Returns the inner [`GgufFile`] when this container is GGUF.
    #[must_use]
    pub const fn as_gguf(&self) -> Option<&GgufFile<'a>> {
        match self {
            Self::Gguf(gguf) => Some(gguf),
            Self::SafeTensors(_) => None,
        }
    }

    /// Returns the inner [`SafeTensorsFile`] when this container is `SafeTensors`.
    #[must_use]
    pub const fn as_safetensors(&self) -> Option<&SafeTensorsFile<'a>> {
        match self {
            Self::Gguf(_) => None,
            Self::SafeTensors(st) => Some(st),
        }
    }
}

impl ModelWeights for ParsedModelContainer<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        match self {
            Self::Gguf(gguf) => gguf.tensor(name),
            Self::SafeTensors(st) => st.tensor(name),
        }
    }

    fn tensor_count(&self) -> usize {
        match self {
            Self::Gguf(gguf) => gguf.tensor_count(),
            Self::SafeTensors(st) => st.tensor_count(),
        }
    }

    fn total_weight_bytes(&self) -> usize {
        match self {
            Self::Gguf(gguf) => gguf.total_weight_bytes(),
            Self::SafeTensors(st) => st.total_weight_bytes(),
        }
    }
}

static ACTIVE_LOCKED_MODEL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Telemetry summary describing model-scoped repacked weight residency and reuse counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepackSummary {
    /// Number of 2D linear matrices currently retained in the model-scoped repack store.
    pub tensors_repacked: usize,
    /// Total resident heap bytes held by repacked matrix buffers (`0` when cleared or downgraded).
    pub resident_bytes: usize,
    /// Number of full load/prepare repack passes executed for this model instance (`<= 1` across repeated prompts).
    pub build_count: usize,
    /// Number of lookup/prepare calls satisfied from already-resident repacked tensors.
    pub hit_count: usize,
    /// Number of lookup/prepare calls that missed before population.
    pub miss_count: usize,
    /// `true` when eager repacking was disabled under memory/memlock pressure (`eager_repack_enabled == false`).
    pub downgraded: bool,
}

/// Model-scoped repacked weight representation retained for a single [`LockedModelWeights`] instance.
#[cfg(feature = "jero-kernels-rvv")]
#[derive(Debug)]
pub enum RepackedTensorEntry {
    /// Compact 4-bit nibble tile layout (`Q4_0`, `IQ4_NL`, `Q4_1`, `Q4_K_M`) for `XSMTVDotII` / `smt.vpack.vv`.
    Nibble(PackedNibbleMatrix),
    /// Block-scaled `i8` tile layout (`Q8_0`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`, etc.).
    BlockScaled(RepackedWeight),
}

/// Returns whether a `BlockScaled` repacked representation for `quant` allocates
/// an asymmetric block-`mins` `f32` buffer alongside `scales`.
#[must_use]
pub(crate) const fn block_scaled_quant_has_mins(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q4_1 | QuantType::Q5_1 | QuantType::Q2_K
    )
}

/// Disambiguates `(rows, cols)` for a 2D weight tensor using GGUF `[cols, rows]` layout rules.
#[must_use]
pub(crate) fn matrix_dims_for_repack(shape: &[usize], quant: QuantType) -> Option<(usize, usize)> {
    if shape.len() != 2 {
        return None;
    }
    let (d0, d1) = (shape[0], shape[1]);
    let blk = quant.block_elements().max(1);
    if d1 % blk == 0 && d0 % blk != 0 {
        Some((d0, d1))
    } else {
        Some((d1, d0))
    }
}

/// Returns whether a tensor descriptor should be included in model repack estimation/preparation.
#[cfg(feature = "jero-kernels-rvv")]
#[must_use]
pub(crate) fn is_repackable_tensor_spec(name: &str, shape: &[usize], quant: QuantType) -> bool {
    if shape.len() != 2 {
        return false;
    }
    if name.contains("token_embd")
        || name.contains("position_embd")
        || name.ends_with("norm.weight")
    {
        return false;
    }
    let supported_quant = matches!(
        quant,
        QuantType::Q4_0
            | QuantType::IQ4_NL
            | QuantType::Q4_1
            | QuantType::Q4_K_M
            | QuantType::Q8_0
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::TQ1_0
            | QuantType::TQ2_0
            | QuantType::I2_S
    );
    if !supported_quant {
        return false;
    }
    let blk = quant.block_elements().max(1);
    shape[0] % blk == 0 || shape[1] % blk == 0
}

#[cfg(feature = "jero-kernels-rvv")]
impl RepackedTensorEntry {
    /// Returns the approximate resident heap payload size in bytes for this entry.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Nibble(m) => m.resident_bytes(),
            Self::BlockScaled(w) => {
                let group = w.group_size();
                if w.n_tiles() == 0 || group == 0 {
                    return w.aligned_bytes();
                }
                let meta_scales = w.n_tiles() * (w.k_padded() / group) * 8 * size_of::<f32>();
                let meta_mins = if block_scaled_quant_has_mins(w.quant()) {
                    meta_scales
                } else {
                    0
                };
                w.aligned_bytes()
                    .saturating_add(meta_scales)
                    .saturating_add(meta_mins)
            }
        }
    }

    /// Returns a reference to the compact [`PackedNibbleMatrix`] if present.
    #[must_use]
    pub const fn as_nibble(&self) -> Option<&PackedNibbleMatrix> {
        match self {
            Self::Nibble(m) => Some(m),
            Self::BlockScaled(_) => None,
        }
    }

    /// Returns a reference to the [`RepackedWeight`] if present.
    #[must_use]
    pub const fn as_block_scaled(&self) -> Option<&RepackedWeight> {
        match self {
            Self::Nibble(_) => None,
            Self::BlockScaled(w) => Some(w),
        }
    }
}

/// Placeholder entry when the crate is built with `--no-default-features`.
#[cfg(not(feature = "jero-kernels-rvv"))]
#[derive(Debug)]
pub struct RepackedTensorEntry {
    bytes: usize,
}

#[cfg(not(feature = "jero-kernels-rvv"))]
impl RepackedTensorEntry {
    /// Returns the resident byte count (`0`).
    #[must_use]
    pub const fn resident_bytes(&self) -> usize {
        self.bytes
    }
}

/// Thread-safe, lifetime-bound repacked tensor store owned by a single [`LockedModelWeights`].
///
/// Keyed by both tensor index (`Vec<Option<Arc<RepackedTensorEntry>>>`) and tensor name (`tensor_index`),
/// allowing multiple prompt passes (`--repeat N` or worker requests) to reuse load-time repacked weights
/// without redundant rebuilds and releasing 100% of repacked memory when dropped.
#[derive(Debug)]
pub struct RepackedTensorStore {
    slots: RwLock<Vec<Option<Arc<RepackedTensorEntry>>>>,
    build_lock: Mutex<()>,
    populated: AtomicBool,
    eager_admitted: bool,
    budget_downgraded: AtomicBool,
    repack_arena_budget: Option<usize>,
    build_count: AtomicUsize,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
    resident_bytes: AtomicUsize,
    tensors_repacked: AtomicUsize,
}

impl RepackedTensorStore {
    fn new(tensor_count: usize, eager_admitted: bool, repack_arena_budget: Option<usize>) -> Self {
        Self {
            slots: RwLock::new(vec![None; tensor_count]),
            build_lock: Mutex::new(()),
            populated: AtomicBool::new(false),
            eager_admitted,
            budget_downgraded: AtomicBool::new(false),
            repack_arena_budget,
            build_count: AtomicUsize::new(0),
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
            resident_bytes: AtomicUsize::new(0),
            tensors_repacked: AtomicUsize::new(0),
        }
    }

    /// Returns whether eager repacking was admitted by pre-flight memory/memlock checks.
    #[must_use]
    pub const fn eager_admitted(&self) -> bool {
        self.eager_admitted
    }

    /// Returns the optional admitted repack arena ceiling in bytes.
    #[must_use]
    pub const fn repack_arena_budget(&self) -> Option<usize> {
        self.repack_arena_budget
    }

    /// Returns a snapshot of residency and cache reuse telemetry.
    #[must_use]
    pub fn summary(&self) -> RepackSummary {
        RepackSummary {
            tensors_repacked: self.tensors_repacked.load(Ordering::Acquire),
            resident_bytes: self.resident_bytes.load(Ordering::Acquire),
            build_count: self.build_count.load(Ordering::Acquire),
            hit_count: self.hit_count.load(Ordering::Acquire),
            miss_count: self.miss_count.load(Ordering::Acquire),
            downgraded: !self.eager_admitted || self.budget_downgraded.load(Ordering::Acquire),
        }
    }

    /// Retrieves a repacked tensor entry by canonical GGUF tensor index.
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<Arc<RepackedTensorEntry>> {
        let guard = self
            .slots
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let found = guard
            .get(index)
            .and_then(|slot| slot.as_ref().map(Arc::clone));
        if found.is_some() {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    /// Clears all retained repacked tensor buffers immediately and resets resident byte accounting to `0`.
    pub fn clear(&self) {
        let mut guard = self
            .slots
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for slot in guard.iter_mut() {
            *slot = None;
        }
        self.populated.store(false, Ordering::Release);
        self.budget_downgraded.store(false, Ordering::Release);
        self.resident_bytes.store(0, Ordering::Release);
        self.tensors_repacked.store(0, Ordering::Release);
    }
}

/// Lifetime-bound model view pairing [`LockedModelWeights`] with its [`RepackedTensorStore`].
///
/// Implements [`ModelWeights`] so recipes and evaluation loops share a single model-scoped
/// cache across repeated prompts and release all cached allocations atomically when the owning
/// [`LockedModelWeights`] drops.
#[derive(Debug, Clone, Copy)]
pub struct ModelView<'a> {
    weights: &'a LockedModelWeights,
}

impl<'a> ModelView<'a> {
    /// Returns the backing [`LockedModelWeights`].
    #[must_use]
    pub const fn weights(&self) -> &'a LockedModelWeights {
        self.weights
    }

    /// Returns parsed GGUF [`ModelMetadata`].
    #[must_use]
    pub const fn metadata(&self) -> &'a ModelMetadata {
        self.weights.metadata()
    }

    /// Returns the model-scoped [`RepackedTensorStore`].
    #[must_use]
    pub const fn repack_store(&self) -> &'a RepackedTensorStore {
        self.weights.repack_store()
    }

    /// Looks up a repacked tensor entry by canonical tensor name, incrementing hit/miss telemetry.
    #[must_use]
    pub fn repacked_tensor(&self, name: &str) -> Option<Arc<RepackedTensorEntry>> {
        self.weights.repacked_tensor(name)
    }

    /// Returns the current [`RepackSummary`] snapshot.
    #[must_use]
    pub fn repack_summary(&self) -> RepackSummary {
        self.weights.repack_summary()
    }
}

impl ModelWeights for ModelView<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        self.weights.tensor(name)
    }

    fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    fn total_weight_bytes(&self) -> usize {
        self.weights.total_weight_bytes()
    }
}

/// Locked in-memory resident model weights backed by a read-only `mmap` mapping
/// with an active shared file lock (`flock(LOCK_SH)`).
#[derive(Debug)]
pub struct LockedModelWeights {
    file: File,
    ptr: *mut u8,
    len: usize,
    is_locked: bool,
    format: ModelContainerFormat,
    metadata: ModelMetadata,
    tensor_index: BTreeMap<String, usize>,
    repack_store: RepackedTensorStore,
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
    ModelContainerFormat,
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

    // 4. Detect GGUF vs SafeTensors header, parse container & verify capabilities/admission before mlock
    // SAFETY: raw_ptr is valid for reads up to file_len.
    let slice = unsafe { std::slice::from_raw_parts(raw_ptr, file_len) };
    let parsed = (|| -> Result<ParsedWeights, K3Error> {
        let container = ParsedModelContainer::parse(slice)?;
        let format = container.format();
        let metadata = container.metadata().clone();
        if let Some((opt_n_ctx, inspector)) = admission {
            let recipe = match &container {
                ParsedModelContainer::Gguf(gguf) => verify_model_capabilities(gguf)?,
                ParsedModelContainer::SafeTensors(st) => verify_safetensors_capabilities(st)?,
            };
            let effective_n_ctx = if let Some(n) = opt_n_ctx {
                n
            } else {
                let meta_ctx = metadata.context_length();
                if meta_ctx > 0 { meta_ctx.min(64) } else { 64 }
            };
            let footprint = check_memory_admission_with(&metadata, effective_n_ctx, inspector)?;
            Ok((format, metadata, Some(recipe), Some(footprint)))
        } else {
            Ok((format, metadata, None, None))
        }
    })();

    let (format, metadata, recipe_opt, footprint_opt) = match parsed {
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

    let eager_admitted = footprint_opt
        .as_ref()
        .is_none_or(|fp| fp.eager_repack_enabled);
    let repack_arena_budget = footprint_opt.as_ref().map(|fp| fp.repack_arena_bytes);
    let repack_store = RepackedTensorStore::new(
        metadata.tensors().len(),
        eager_admitted,
        repack_arena_budget,
    );
    ACTIVE_LOCKED_MODEL_COUNT.fetch_add(1, Ordering::Relaxed);

    Ok((
        LockedModelWeights {
            file,
            ptr: raw_ptr,
            len: file_len,
            is_locked,
            format,
            metadata,
            tensor_index,
            repack_store,
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

    /// Returns the detected binary container format (`GGUF` or `SafeTensors`).
    #[must_use]
    pub const fn container_format(&self) -> ModelContainerFormat {
        self.format
    }

    /// Parses a zero-copy [`ParsedModelContainer`] view over the locked memory-mapped bytes.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if header parsing fails.
    pub fn parse_container(&self) -> Result<ParsedModelContainer<'_>, K3Error> {
        ParsedModelContainer::parse(self.as_bytes())
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

    /// Returns a reference to the model-scoped [`RepackedTensorStore`].
    #[must_use]
    pub const fn repack_store(&self) -> &RepackedTensorStore {
        &self.repack_store
    }

    /// Returns a lifetime-bound [`ModelView`] pairing these weights with their repacked store.
    #[must_use]
    pub const fn model_view(&self) -> ModelView<'_> {
        ModelView { weights: self }
    }

    /// Returns the current [`RepackSummary`] residency and reuse snapshot.
    #[must_use]
    pub fn repack_summary(&self) -> RepackSummary {
        self.repack_store.summary()
    }

    /// Looks up a repacked tensor by canonical tensor name, incrementing hit/miss telemetry.
    #[must_use]
    pub fn repacked_tensor(&self, name: &str) -> Option<Arc<RepackedTensorEntry>> {
        let idx = self.tensor_index.get(name).copied()?;
        self.repack_store.get_by_index(idx)
    }

    /// Eagerly repacks eligible 2D quantized weight matrices in parallel across `pool` once per
    /// model instance, respecting `#163` memory admission (`eager_repack_enabled`).
    ///
    /// Subsequent calls reuse already-resident tensors and increment `hit_count` with zero
    /// redundant rebuilds (`build_count == 1`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if any eligible weight tensor fails bounds validation or repacking.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_repacked_weights(
        &self,
        pool: Option<&WorkerPool>,
    ) -> Result<RepackSummary, K3Error> {
        if !self.repack_store.eager_admitted
            || self.repack_store.budget_downgraded.load(Ordering::Acquire)
        {
            return Ok(self.repack_store.summary());
        }
        if self.repack_store.populated.load(Ordering::Acquire) {
            self.repack_store.hit_count.fetch_add(1, Ordering::Relaxed);
            return Ok(self.repack_store.summary());
        }

        let _build_guard = self
            .repack_store
            .build_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.repack_store.budget_downgraded.load(Ordering::Acquire) {
            return Ok(self.repack_store.summary());
        }
        if self.repack_store.populated.load(Ordering::Acquire) {
            self.repack_store.hit_count.fetch_add(1, Ordering::Relaxed);
            return Ok(self.repack_store.summary());
        }

        self.repack_store.miss_count.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "jero-kernels-rvv")]
        {
            let budget = self.repack_store.repack_arena_budget;
            let mut built: Vec<(usize, Arc<RepackedTensorEntry>)> = Vec::new();
            let mut total_bytes = 0_usize;
            let mut budget_exceeded = false;

            for (idx, desc) in self.metadata.tensors().iter().enumerate() {
                if !is_repackable_matrix_descriptor(desc) {
                    continue;
                }
                if budget == Some(0) {
                    budget_exceeded = true;
                    break;
                }
                let view = self.tensor(desc.name())?;
                let entry = Arc::new(repack_tensor_parallel(&view, pool)?);
                total_bytes = total_bytes.saturating_add(entry.resident_bytes());
                if budget.is_some_and(|cap| total_bytes > cap) {
                    budget_exceeded = true;
                    break;
                }
                built.push((idx, entry));
            }

            if budget_exceeded {
                drop(built);
                self.repack_store.clear();
                self.repack_store
                    .budget_downgraded
                    .store(true, Ordering::Release);
                self.repack_store.populated.store(true, Ordering::Release);
                return Ok(self.repack_store.summary());
            }

            let count = built.len();
            let mut slots = self
                .repack_store
                .slots
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (idx, entry) in built {
                if let Some(slot) = slots.get_mut(idx) {
                    *slot = Some(entry);
                }
            }
            self.repack_store
                .tensors_repacked
                .store(count, Ordering::Release);
            self.repack_store
                .resident_bytes
                .store(total_bytes, Ordering::Release);
        }

        #[cfg(not(feature = "jero-kernels-rvv"))]
        let _ = pool;

        self.repack_store
            .build_count
            .fetch_add(1, Ordering::Relaxed);
        self.repack_store.populated.store(true, Ordering::Release);
        Ok(self.repack_store.summary())
    }
}

#[cfg(feature = "jero-kernels-rvv")]
fn infer_matrix_rows_cols(view: &TensorView<'_>) -> Result<(usize, usize), K3Error> {
    matrix_dims_for_repack(view.shape(), view.quant()).ok_or_else(|| K3Error::TensorBoundsError {
        tensor_name: view.name().to_owned(),
        offset: view.offset(),
        byte_len: view.byte_len(),
        buffer_len: view.as_bytes().len(),
        reason: format!(
            "expected 2D tensor for parallel repack, got {}D",
            view.shape().len()
        ),
    })
}

#[cfg(feature = "jero-kernels-rvv")]
fn is_repackable_matrix_descriptor(desc: &TensorDescriptor) -> bool {
    is_repackable_tensor_spec(desc.name(), desc.shape(), desc.quant())
}

#[cfg(feature = "jero-kernels-rvv")]
fn try_for_each_work<W, F>(pool: Option<&WorkerPool>, work: &mut [W], f: F) -> Result<(), K3Error>
where
    W: Send,
    F: Fn(&mut W) -> Result<(), K3Error> + Sync,
{
    let target_pool = pool.map(|p| p.a100_pool().unwrap_or(p));
    match target_pool {
        Some(p) if p.thread_count() > 1 && work.len() > 1 => {
            p.try_par_chunks_mut(work, 1, |_, chunk| f(&mut chunk[0]))
        }
        Some(p) => p.install(|| work.iter_mut().try_for_each(&f)),
        None => work.iter_mut().try_for_each(&f),
    }
}

/// Repacks an eligible 2D quantized tensor into a compact nibble or block-scaled `i8` entry using
/// disjoint tile-range slices across `pool` worker threads.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor is not 2D or fails bounds validation, or
/// [`K3Error::CapabilityMismatch`] if the quantization format does not support tile repacking.
#[cfg(feature = "jero-kernels-rvv")]
pub fn repack_tensor_parallel(
    view: &TensorView<'_>,
    pool: Option<&WorkerPool>,
) -> Result<RepackedTensorEntry, K3Error> {
    let (rows, cols) = infer_matrix_rows_cols(view)?;
    let worker_threads = pool
        .map_or(1, |p| {
            p.a100_pool()
                .map_or_else(|| p.thread_count(), WorkerPool::thread_count)
        })
        .max(1);

    match view.quant() {
        QuantType::Q4_0 | QuantType::IQ4_NL => {
            repack_nibble_scale_f16_parallel(view, rows, cols, worker_threads, pool)
        }
        QuantType::Q4_K_M => repack_nibble_q4_k_m_parallel(view, rows, cols, worker_threads, pool),
        QuantType::Q4_1 => {
            PackedNibbleMatrix::from_tensor(view, rows, cols).map(RepackedTensorEntry::Nibble)
        }
        QuantType::Q8_0
        | QuantType::Q5_0
        | QuantType::Q5_1
        | QuantType::Q2_K
        | QuantType::Q3_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::TQ1_0
        | QuantType::TQ2_0
        | QuantType::I2_S => repack_block_scaled_parallel(view, rows, cols, worker_threads, pool),
        QuantType::F32 | QuantType::F16 | QuantType::BF16 | QuantType::Q8_1 | QuantType::MXFP4 => {
            Err(K3Error::CapabilityMismatch {
                architecture: view.quant().as_str().to_owned(),
                reason: format!(
                    "tensor '{}' quant {:?} does not support tiled repacking",
                    view.name(),
                    view.quant()
                ),
            })
        }
    }
}

#[cfg(feature = "jero-kernels-rvv")]
fn repack_nibble_scale_f16_parallel(
    view: &TensorView<'_>,
    rows: usize,
    cols: usize,
    worker_threads: usize,
    pool: Option<&WorkerPool>,
) -> Result<RepackedTensorEntry, K3Error> {
    const TILE_N: usize = 8;
    const GROUP_ELEMS: usize = 32;
    if rows == 0 || cols == 0 || cols % view.quant().block_elements() != 0 {
        return PackedNibbleMatrix::from_tensor(view, rows, cols).map(RepackedTensorEntry::Nibble);
    }
    let n_tiles = rows.div_ceil(TILE_N);
    let k_blocks32 = cols / GROUP_ELEMS;
    let chunk_tiles = n_tiles.div_ceil(worker_threads).max(1);
    let mut nibble_tiles = vec![Aligned128::zeroed(); n_tiles * k_blocks32];
    let mut scales_f16 = vec![0_u16; n_tiles * k_blocks32 * TILE_N];

    let mut work: Vec<(usize, usize, &mut [Aligned128], &mut [u16])> = nibble_tiles
        .chunks_mut(chunk_tiles * k_blocks32)
        .zip(scales_f16.chunks_mut(chunk_tiles * k_blocks32 * TILE_N))
        .enumerate()
        .map(|(idx, (tile_sub, scale_sub))| {
            let t0 = idx * chunk_tiles;
            let t1 = (t0 + chunk_tiles).min(n_tiles);
            (t0, t1, tile_sub, scale_sub)
        })
        .collect();

    try_for_each_work(pool, &mut work, |(t0, t1, tile_sub, scale_sub)| {
        repack_nibble_into_tile_range(view, rows, cols, *t0..*t1, tile_sub, scale_sub, None, None)
    })?;

    PackedNibbleMatrix::from_scale_f16_buffers(view.quant(), rows, cols, nibble_tiles, scales_f16)
        .map(RepackedTensorEntry::Nibble)
}

#[cfg(feature = "jero-kernels-rvv")]
fn repack_nibble_q4_k_m_parallel(
    view: &TensorView<'_>,
    rows: usize,
    cols: usize,
    worker_threads: usize,
    pool: Option<&WorkerPool>,
) -> Result<RepackedTensorEntry, K3Error> {
    type NibbleChunkWork<'a> = (usize, usize, &'a mut [Aligned128], &'a mut [[u8; 16]]);
    const TILE_N: usize = 8;
    const GROUP_ELEMS: usize = 32;
    const Q4_K_ELEMS: usize = 256;
    if rows == 0 || cols == 0 || cols % Q4_K_ELEMS != 0 {
        return PackedNibbleMatrix::from_tensor(view, rows, cols).map(RepackedTensorEntry::Nibble);
    }
    let n_tiles = rows.div_ceil(TILE_N);
    let k_blocks32 = cols / GROUP_ELEMS;
    let n_sblks = cols / Q4_K_ELEMS;
    let chunk_tiles = n_tiles.div_ceil(worker_threads).max(1);
    let mut nibble_tiles = vec![Aligned128::zeroed(); n_tiles * k_blocks32];
    let mut headers = vec![[0_u8; 16]; n_tiles * n_sblks * TILE_N];

    let mut work: Vec<NibbleChunkWork<'_>> = nibble_tiles
        .chunks_mut(chunk_tiles * k_blocks32)
        .zip(headers.chunks_mut(chunk_tiles * n_sblks * TILE_N))
        .enumerate()
        .map(|(idx, (tile_sub, hdr_sub))| {
            let t0 = idx * chunk_tiles;
            let t1 = (t0 + chunk_tiles).min(n_tiles);
            (t0, t1, tile_sub, hdr_sub)
        })
        .collect();

    try_for_each_work(pool, &mut work, |(t0, t1, tile_sub, hdr_sub)| {
        repack_nibble_into_tile_range(
            view,
            rows,
            cols,
            *t0..*t1,
            tile_sub,
            &mut [],
            None,
            Some(hdr_sub),
        )
    })?;

    PackedNibbleMatrix::from_q4_k_buffers(rows, cols, nibble_tiles, headers)
        .map(RepackedTensorEntry::Nibble)
}

#[cfg(feature = "jero-kernels-rvv")]
fn repack_block_scaled_parallel(
    view: &TensorView<'_>,
    rows: usize,
    cols: usize,
    worker_threads: usize,
    pool: Option<&WorkerPool>,
) -> Result<RepackedTensorEntry, K3Error> {
    const TILE_N: usize = 8;
    let layout = BlockScaledTileLayout::for_tensor(view, rows, cols)?;
    let n_tiles = layout.n_tiles();
    let chunk_tiles = n_tiles.div_ceil(worker_threads).max(1);
    let mut tiles = vec![Aligned128::zeroed(); layout.total_tiles()];
    let mut scales = vec![0.0_f32; layout.total_scales()];
    let mut mins = if layout.has_mins() {
        Some(vec![0.0_f32; layout.total_mins()])
    } else {
        None
    };

    let mut tile_chunks = tiles.chunks_mut(chunk_tiles * layout.k_chunks());
    let mut scale_chunks = scales.chunks_mut(chunk_tiles * layout.scale_groups() * TILE_N);
    let mut min_chunks = mins
        .as_deref_mut()
        .map(|m| m.chunks_mut(chunk_tiles * layout.scale_groups() * TILE_N));

    let mut work = Vec::new();
    let mut idx = 0_usize;
    while let (Some(tile_sub), Some(scale_sub)) = (tile_chunks.next(), scale_chunks.next()) {
        let min_sub = min_chunks.as_mut().and_then(Iterator::next);
        let t0 = idx * chunk_tiles;
        let t1 = (t0 + chunk_tiles).min(n_tiles);
        work.push((t0, t1, tile_sub, scale_sub, min_sub));
        idx += 1;
    }

    try_for_each_work(pool, &mut work, |(t0, t1, tile_sub, scale_sub, min_sub)| {
        repack_into_tile_range(
            view,
            rows,
            cols,
            *t0..*t1,
            tile_sub,
            scale_sub,
            min_sub.as_deref_mut(),
        )
    })?;

    RepackedWeight::from_tile_buffers(layout, tiles, scales, mins)
        .map(RepackedTensorEntry::BlockScaled)
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
        self.repack_store.clear();
        if ACTIVE_LOCKED_MODEL_COUNT.fetch_sub(1, Ordering::AcqRel) == 1 {
            #[cfg(feature = "jero-arch-llama")]
            jero_arch_llama::clear_repack_cache();
            #[cfg(feature = "jero-arch-mistral")]
            jero_arch_mistral::clear_global_layer_cache();
        }

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

    #[test]
    fn test_repacked_tensor_store_lifecycle_and_clear() {
        let store = RepackedTensorStore::new(4, true, Some(4096));
        assert!(store.eager_admitted());
        assert_eq!(store.repack_arena_budget(), Some(4096));
        assert!(store.get_by_index(0).is_none());
        let snap = store.summary();
        assert_eq!(snap.tensors_repacked, 0);
        assert_eq!(snap.resident_bytes, 0);
        assert_eq!(snap.miss_count, 1);
        assert!(!snap.downgraded);

        store.clear();
        let after_clear = store.summary();
        assert_eq!(after_clear.tensors_repacked, 0);
        assert_eq!(after_clear.resident_bytes, 0);
    }

    #[test]
    #[cfg(all(feature = "jero-kernels-rvv", not(miri)))]
    fn test_parallel_slice_repack_equivalence_across_worker_pool() {
        use crate::affinity::resolve_affinity;
        use jero_core::{CoordinationProfile, GgufFile};

        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 16][..]),
            ("blk.0.attn_k.weight", QuantType::Q4_0, &[32, 16][..]),
            ("blk.0.attn_v.weight", QuantType::Q8_0, &[32, 16][..]),
        ];
        let mut bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        // Deterministic non-zero block pattern in payload tail to exercise every tile/scale path.
        let tail = bytes.len();
        for (i, byte) in bytes[tail.saturating_sub(2048)..].iter_mut().enumerate() {
            *byte ^= u8::try_from((i * 17 + 3) % 251).unwrap_or(0);
        }
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(2));
        let affinity = resolve_affinity(None, None, Some(threads)).expect("host affinity");
        let pool = WorkerPool::with_coordination(affinity, CoordinationProfile::Rayon)
            .expect("spawn worker pool");

        // 1. Q4_K_M compact nibble parallel vs sequential reference + i8 parallel slices check
        let q4k = gguf.tensor("blk.0.attn_q.weight").expect("q4_k tensor");
        let par_q4k = repack_tensor_parallel(&q4k, Some(&pool)).expect("parallel q4_k nibble");
        let seq_q4k = PackedNibbleMatrix::repack(&q4k).expect("seq q4_k nibble");
        let par_nibble = par_q4k.as_nibble().expect("nibble variant");
        assert_eq!(par_nibble.rows(), seq_q4k.rows());
        assert_eq!(par_nibble.cols(), seq_q4k.cols());
        assert_eq!(par_nibble.packed_bytes(), seq_q4k.packed_bytes());

        let mut i8_tiles = vec![0_i8; 2 * 16 * 128];
        let mut i8_scales = vec![0.0_f32; 2 * 8 * 8];
        let mut i8_mins = vec![0.0_f32; 2 * 8 * 8];
        jero_kernels_rvv::repack::repack_q4_k_m_parallel_slices(
            &q4k,
            16,
            256,
            0..2,
            &mut i8_tiles,
            &mut i8_scales,
            &mut i8_mins,
        )
        .expect("repack_q4_k_m_parallel_slices succeeds");
        assert!(i8_scales.iter().any(|&v| v != 0.0));

        // 2. Q4_0 compact nibble parallel vs sequential reference via QuantizedActQ8::quantize
        let q4_0 = gguf.tensor("blk.0.attn_k.weight").expect("q4_0 tensor");
        let par_q4_0 = repack_tensor_parallel(&q4_0, Some(&pool)).expect("parallel q4_0 nibble");
        let seq_q4_0 = PackedNibbleMatrix::repack(&q4_0).expect("seq q4_0 nibble");
        let par_q4_0_nib = par_q4_0.as_nibble().expect("q4_0 nibble");
        let acts = vec![0.25_f32; 2 * seq_q4_0.cols()];
        let q_act = jero_kernels_rvv::repack::QuantizedActQ8::quantize(&acts, 2, seq_q4_0.cols())
            .expect("quantize acts");
        let mut out_par = vec![0.0_f32; 2 * par_q4_0_nib.rows()];
        let mut out_seq = vec![0.0_f32; 2 * seq_q4_0.rows()];
        par_q4_0_nib
            .gemm(&mut out_par, &q_act)
            .expect("par nibble gemm");
        seq_q4_0
            .gemm(&mut out_seq, &q_act)
            .expect("seq nibble gemm");
        assert_eq!(out_par, out_seq);

        // 3. Q8_0 block-scaled i8 parallel vs sequential row dequantization
        let q8_0 = gguf.tensor("blk.0.attn_v.weight").expect("q8_0 tensor");
        let par_q8 = repack_tensor_parallel(&q8_0, Some(&pool)).expect("parallel q8_0");
        let seq_q8 = RepackedWeight::repack(&q8_0).expect("seq q8_0");
        let par_bs = par_q8.as_block_scaled().expect("block scaled variant");
        assert_eq!(par_bs.rows(), seq_q8.rows());
        assert_eq!(par_bs.cols(), seq_q8.cols());
        for row in 0..seq_q8.rows() {
            let mut r_par = vec![0.0_f32; seq_q8.cols()];
            let mut r_seq = vec![0.0_f32; seq_q8.cols()];
            par_bs.dequantize_row(row, &mut r_par).expect("dequant par");
            seq_q8.dequantize_row(row, &mut r_seq).expect("dequant seq");
            assert_eq!(r_par, r_seq, "row {row} mismatch");
        }
    }

    #[test]
    #[cfg(feature = "jero-arch-synthetic")]
    #[cfg_attr(miri, ignore)]
    fn test_locked_weights_repeat_repack_reuse_and_drop_cleanup() {
        use crate::affinity::resolve_affinity;
        use crate::memory::MockMemoryInspector;
        use jero_core::CoordinationProfile;

        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 16][..]),
            ("blk.0.attn_v.weight", QuantType::Q4_K_M, &[256, 16][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let temp_path =
            std::env::temp_dir().join(format!("test_repack_reuse_{}.gguf", std::process::id()));
        std::fs::write(&temp_path, &bytes).expect("write model");

        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(2));
        let affinity = resolve_affinity(None, None, Some(threads)).expect("host affinity");
        let pool = WorkerPool::with_coordination(affinity, CoordinationProfile::Rayon)
            .expect("worker pool");

        let inspector = MockMemoryInspector::new(256 * 1024 * 1024, 256 * 1024 * 1024);
        let (locked, _recipe, footprint) =
            LockedModelWeights::open_with_admission(&temp_path, 64, &inspector)
                .expect("open with generous admission");
        assert!(footprint.eager_repack_enabled);

        let before = ACTIVE_LOCKED_MODEL_COUNT.load(Ordering::Acquire);
        assert!(before >= 1);

        // Repeat 3 prompt passes reuse the same load-time repacked matrices without rebuilds
        let first = locked
            .prepare_repacked_weights(Some(&pool))
            .expect("initial repack");
        let second = locked
            .prepare_repacked_weights(Some(&pool))
            .expect("second pass reuse");
        let third = locked
            .prepare_repacked_weights(Some(&pool))
            .expect("third pass reuse");

        assert_eq!(first.build_count, 1);
        assert_eq!(second.build_count, 1);
        assert_eq!(third.build_count, 1);
        assert_eq!(third.miss_count, 1);
        assert_eq!(third.hit_count, 2);
        #[cfg(feature = "jero-kernels-rvv")]
        {
            assert_eq!(first.tensors_repacked, 2);
            assert!(first.resident_bytes > 0);
            let view = locked.model_view();
            assert!(view.repacked_tensor("blk.0.attn_q.weight").is_some());
            assert!(view.repacked_tensor("blk.0.attn_v.weight").is_some());
            assert_eq!(view.repack_summary().hit_count, 4);
        }

        // Verify automatic Drop frees 100% of repacked store allocations without calling clear()
        #[cfg(feature = "jero-kernels-rvv")]
        let weak_entry = {
            let entry = locked
                .repacked_tensor("blk.0.attn_q.weight")
                .expect("repacked entry exists before drop");
            let weak = Arc::downgrade(&entry);
            assert!(weak.upgrade().is_some());
            weak
        };
        drop(locked);
        #[cfg(feature = "jero-kernels-rvv")]
        assert!(
            weak_entry.upgrade().is_none(),
            "dropping LockedModelWeights must automatically free all repacked entries"
        );

        // Tight budget that fits base but downgrades optional repack arena keeps eager_admitted = false
        let exact_base = footprint.base_required_bytes();
        let tight = MockMemoryInspector::new(exact_base, exact_base);
        let (downgraded_weights, _, downgraded_fp) =
            LockedModelWeights::open_with_admission(&temp_path, 64, &tight)
                .expect("admit downgraded footprint");
        assert!(!downgraded_fp.eager_repack_enabled);
        let downgraded_summary = downgraded_weights
            .prepare_repacked_weights(Some(&pool))
            .expect("downgraded prepare succeeds as no-op");
        assert!(downgraded_summary.downgraded);
        assert_eq!(downgraded_summary.tensors_repacked, 0);
        assert_eq!(downgraded_summary.resident_bytes, 0);
        drop(downgraded_weights);

        let _ = std::fs::remove_file(&temp_path);
    }

    #[cfg(all(feature = "jero-kernels-rvv", not(miri)))]
    fn build_synthetic_llama_repack_gguf_bytes() -> Vec<u8> {
        use jero_core::GgufValue;
        use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;

        let extra_kvs = [
            ("llama.block_count", GgufValue::Uint32(1)),
            ("llama.embedding_length", GgufValue::Uint32(256)),
            ("llama.feed_forward_length", GgufValue::Uint32(512)),
            ("llama.attention.head_count", GgufValue::Uint32(4)),
            ("llama.attention.head_count_kv", GgufValue::Uint32(4)),
        ];
        let tensor_specs = [
            ("token_embd.weight", QuantType::F32, &[256usize, 16][..]),
            ("blk.0.attn_norm.weight", QuantType::F32, &[256][..]),
            ("blk.0.attn_q.weight", QuantType::Q5_K, &[256, 256][..]),
            ("blk.0.attn_k.weight", QuantType::Q4_K_M, &[256, 256][..]),
            ("blk.0.attn_v.weight", QuantType::Q8_0, &[256, 256][..]),
            ("blk.0.attn_output.weight", QuantType::F32, &[256, 256][..]),
            ("blk.0.ffn_norm.weight", QuantType::F32, &[256][..]),
            ("blk.0.ffn_gate.weight", QuantType::F32, &[256, 512][..]),
            ("blk.0.ffn_up.weight", QuantType::F32, &[256, 512][..]),
            ("blk.0.ffn_down.weight", QuantType::F32, &[512, 256][..]),
            ("output_norm.weight", QuantType::F32, &[256][..]),
            ("output.weight", QuantType::F32, &[256, 16][..]),
        ];
        create_synthetic_gguf_v3_with_kvs("llama", 32, &extra_kvs, &tensor_specs)
    }

    #[test]
    #[cfg(all(feature = "jero-kernels-rvv", not(miri)))]
    fn test_repack_arena_budget_enforced_and_block_scaled_mins_accounted() {
        use crate::memory::{SystemMemoryInspector, calculate_footprint_with_slots};
        use std::io::Write;

        let bytes = build_synthetic_llama_repack_gguf_bytes();

        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("jero_repack_budget_test_{pid}_{ts}.gguf"));
        {
            let mut file = File::create(&path).expect("create temp gguf");
            file.write_all(&bytes).expect("write temp gguf");
            file.sync_all().expect("sync temp gguf");
        }

        let file = File::open(&path).expect("open temp gguf");
        let (mut locked, _recipe, footprint) = LockedModelWeights::from_file_with_admission(
            file,
            Some(64),
            &SystemMemoryInspector,
            true,
        )
        .expect("admit synthetic gguf");

        let q5k_view = locked.tensor("blk.0.attn_q.weight").expect("q5_k view");
        let q5k_entry = repack_tensor_parallel(&q5k_view, None).expect("repack q5_k");
        let q5k_bs = q5k_entry.as_block_scaled().expect("q5_k is block scaled");
        let expected_meta =
            q5k_bs.n_tiles() * (q5k_bs.k_padded() / q5k_bs.group_size()) * 8 * size_of::<f32>();
        assert_eq!(
            q5k_entry.resident_bytes(),
            q5k_bs.aligned_bytes() + expected_meta * 2,
            "Q5_K BlockScaled resident_bytes must include both scales and mins"
        );

        let full_summary = locked.prepare_repacked_weights(None).expect("prepare full");
        assert!(!full_summary.downgraded);
        assert_eq!(full_summary.tensors_repacked, 3);
        assert!(
            full_summary.resident_bytes <= footprint.repack_arena_bytes,
            "actual resident bytes ({}) must not exceed admitted repack arena ({})",
            full_summary.resident_bytes,
            footprint.repack_arena_bytes
        );

        // Verify unpadded TILE_N=8 + Q4_1 scales_f16/mins_f16 estimate bounds exact PackedNibbleMatrix residency
        let q4_1_specs = [("blk.0.attn_k.weight", QuantType::Q4_1, &[32, 5][..])];
        let q4_1_bytes = create_synthetic_gguf_v3("qwen2", 32, &q4_1_specs);
        let q4_1_path = dir.join(format!("jero_repack_q4_1_test_{pid}_{ts}.gguf"));
        {
            let mut file = File::create(&q4_1_path).expect("create q4_1 temp gguf");
            file.write_all(&q4_1_bytes).expect("write q4_1 temp gguf");
            file.sync_all().expect("sync q4_1 temp gguf");
        }
        let q4_1_locked = LockedModelWeights::open(&q4_1_path).expect("open q4_1 gguf");
        let q4_1_view = q4_1_locked
            .tensor("blk.0.attn_k.weight")
            .expect("q4_1 view");
        let q4_1_entry = repack_tensor_parallel(&q4_1_view, None).expect("repack q4_1");
        assert_eq!(q4_1_entry.resident_bytes(), 160);
        let q4_1_fp =
            calculate_footprint_with_slots(q4_1_locked.metadata(), 64, 1).expect("q4_1 footprint");
        assert!(
            q4_1_fp.repack_arena_bytes >= q4_1_entry.resident_bytes(),
            "Q4_1 unpadded TILE_N=8 estimate ({}) must cover exact residency ({})",
            q4_1_fp.repack_arena_bytes,
            q4_1_entry.resident_bytes()
        );
        drop(q4_1_locked);
        let _ = std::fs::remove_file(&q4_1_path);

        // Enforce tight budget cap smaller than total_bytes -> reject & downgrade without over-budget retention
        locked.repack_store().clear();
        locked.repack_store = RepackedTensorStore::new(
            locked.metadata().tensors().len(),
            true,
            Some(full_summary.resident_bytes.saturating_sub(1)),
        );
        let rejected = locked
            .prepare_repacked_weights(None)
            .expect("over-budget repack downgrades cleanly without error");
        assert!(rejected.downgraded);
        assert_eq!(rejected.tensors_repacked, 0);
        assert_eq!(rejected.resident_bytes, 0);
        assert!(locked.repacked_tensor("blk.0.attn_q.weight").is_none());
        assert!(locked.repacked_tensor("blk.0.attn_k.weight").is_none());
        let repeat_rejected = locked
            .prepare_repacked_weights(None)
            .expect("repeat over-budget stays downgraded");
        assert!(repeat_rejected.downgraded);
        assert_eq!(repeat_rejected.resident_bytes, 0);

        drop(locked);
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(all(feature = "jero-arch-qwen", not(miri)))]
    fn build_qwen3_test_safetensors_bytes(vocab: usize) -> Vec<u8> {
        use jero_core::safetensors::test_helpers::create_synthetic_safetensors;

        type Qwen3Row<'a> = (&'a str, &'a [usize], Option<(usize, f32)>);

        let (hidden, ffn, head_dim, kv_dim) = (16_usize, 32_usize, 8_usize, 8_usize);
        let table: [Qwen3Row<'_>; 14] = [
            (
                "model.embed_tokens.weight",
                &[vocab, hidden],
                Some((1, 0.05)),
            ),
            ("model.layers.0.input_layernorm.weight", &[hidden], None),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[hidden, hidden],
                Some((2, 0.04)),
            ),
            ("model.layers.0.self_attn.q_norm.weight", &[head_dim], None),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[kv_dim, hidden],
                Some((3, 0.04)),
            ),
            ("model.layers.0.self_attn.k_norm.weight", &[head_dim], None),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &[kv_dim, hidden],
                Some((4, 0.04)),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                &[hidden, hidden],
                Some((5, 0.04)),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                &[hidden],
                None,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                &[ffn, hidden],
                Some((6, 0.03)),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                &[ffn, hidden],
                Some((7, 0.03)),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                &[hidden, ffn],
                Some((8, 0.03)),
            ),
            ("model.norm.weight", &[hidden], None),
            ("lm_head.weight", &[vocab, hidden], Some((9, 0.05))),
        ];
        let owned: Vec<(&str, &[usize], Vec<u8>)> = table
            .iter()
            .map(|&(name, shape, spec)| {
                let len: usize = shape.iter().product();
                let bytes = (0..len)
                    .flat_map(|i| {
                        let val = match spec {
                            None => 1.0_f32,
                            Some((seed, scale)) => {
                                #[expect(
                                    clippy::cast_precision_loss,
                                    reason = "bounded test index fits f32"
                                )]
                                let d = ((((i * 11 + seed * 5 + 3) % 19) as f32) - 9.0) * scale;
                                d
                            }
                        };
                        val.to_le_bytes()
                    })
                    .collect();
                (name, shape, bytes)
            })
            .collect();
        let specs: Vec<(&str, QuantType, &[usize], &[u8])> = owned
            .iter()
            .map(|(n, s, b)| (*n, QuantType::F32, *s, b.as_slice()))
            .collect();
        create_synthetic_safetensors(
            &[
                ("general.architecture", "qwen3"),
                ("num_hidden_layers", "1"),
                ("hidden_size", "16"),
                ("intermediate_size", "32"),
                ("num_attention_heads", "2"),
                ("num_key_value_heads", "1"),
                ("head_dim", "8"),
                ("vocab_size", "12"),
                ("rms_norm_eps", "1e-6"),
                ("rope_theta", "1000000"),
            ],
            &specs,
        )
    }

    #[test]
    #[cfg(all(feature = "jero-arch-qwen", not(miri)))]
    fn test_locked_model_weights_safetensors_admission_and_forward() {
        use crate::engine::SinglePromptEngine;
        use crate::kv::SingleSlotKvArena;
        use crate::memory::SystemMemoryInspector;
        use std::io::Write;

        let vocab = 12_usize;
        let st_bytes = build_qwen3_test_safetensors_bytes(vocab);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "jero_st_admission_{}_{ts}.safetensors",
            std::process::id()
        ));
        {
            let mut f = File::create(&path).expect("create temp safetensors");
            f.write_all(&st_bytes).expect("write temp safetensors");
            f.sync_all().expect("sync temp safetensors");
        }

        let (locked, recipe, footprint) =
            LockedModelWeights::open_with_admission(&path, 16, &SystemMemoryInspector)
                .expect("admit safetensors qwen3");
        assert_eq!(locked.container_format(), ModelContainerFormat::SafeTensors);
        assert!(matches!(
            locked.parse_container().expect("parse container"),
            ParsedModelContainer::SafeTensors(_)
        ));
        assert_eq!(recipe.architecture_id(), "qwen3");
        assert!(footprint.weight_bytes > 0);
        assert!(footprint.kv_bytes > 0);

        let mut arena = SingleSlotKvArena::new(locked.metadata(), 16).expect("kv arena");
        let mut engine = SinglePromptEngine::new();
        let summary = engine
            .evaluate_prompt(
                recipe.as_ref(),
                &locked,
                &mut arena,
                &[1, 3, 5],
                vocab,
                2,
                |_| Ok(crate::engine::StreamAction::Continue),
            )
            .expect("qwen3 safetensors forward");
        assert_eq!(summary.generated_tokens.len(), 2);

        drop(locked);
        let _ = std::fs::remove_file(&path);
    }
}
