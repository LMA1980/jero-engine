//! `SpacemiT` K3 heterogeneous cluster topology, core mask resolution, and pinned worker pool.
//!
//! Restricts standalone model evaluation strictly to the operator-requested processor cluster
//! (`x100` or `a100`), core bitmask, and worker concurrency bound while preventing cross-cluster
//! VLEN migration (ADR-007, ADR-008, `core-affinity.md`, US-01 Scenario 2).

use std::cell::{Cell, UnsafeCell};
use std::fs;
use std::marker::PhantomData;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
#[cfg(all(
    target_os = "linux",
    target_arch = "riscv64",
    not(miri),
    feature = "jero-kernels-rvv"
))]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use jero_core::{ComputePool, CoordinationProfile, K3Error, ModelMetadata};
use serde::{Deserialize, Serialize};

thread_local! {
    static CURRENT_ASSIGNED_CORE: Cell<Option<usize>> = const { Cell::new(None) };
    static CURRENT_WORKER_POOL: Cell<Option<*const WorkerPool>> = const { Cell::new(None) };
    static CURRENT_DEDICATED_WORKER: Cell<Option<DedicatedWorkerCtx>> = const { Cell::new(None) };
}

#[derive(Clone, Copy)]
struct DedicatedWorkerCtx {
    shared_ptr: *const DedicatedShared,
    worker_index: usize,
    in_nested: bool,
}

struct TlsPoolGuard<'a> {
    prev: Option<*const WorkerPool>,
    _marker: PhantomData<&'a WorkerPool>,
}

impl<'a> TlsPoolGuard<'a> {
    fn enter(pool: &'a WorkerPool) -> Self {
        let prev = CURRENT_WORKER_POOL.with(|c| c.replace(Some(std::ptr::from_ref(pool))));
        Self {
            prev,
            _marker: PhantomData,
        }
    }
}

impl Drop for TlsPoolGuard<'_> {
    fn drop(&mut self) {
        CURRENT_WORKER_POOL.with(|c| c.set(self.prev));
    }
}

struct TlsDedicatedWorkerGuard {
    prev: Option<DedicatedWorkerCtx>,
}

impl TlsDedicatedWorkerGuard {
    fn enter(shared_ptr: *const DedicatedShared, worker_index: usize, in_nested: bool) -> Self {
        let ctx = DedicatedWorkerCtx {
            shared_ptr,
            worker_index,
            in_nested,
        };
        let prev = CURRENT_DEDICATED_WORKER.with(|c| c.replace(Some(ctx)));
        Self { prev }
    }
}

impl Drop for TlsDedicatedWorkerGuard {
    fn drop(&mut self) {
        CURRENT_DEDICATED_WORKER.with(|c| c.set(self.prev));
    }
}

#[allow(dead_code)]
pub(crate) fn set_current_thread_core(core: usize) {
    CURRENT_ASSIGNED_CORE.with(|c| c.set(Some(core)));
}

#[allow(dead_code)]
pub(crate) fn get_current_thread_core() -> Option<usize> {
    CURRENT_ASSIGNED_CORE.with(Cell::get)
}

/// Returns a single-bit CPU mask (`1 << core`) for the thread currently executing work,
/// matching `sched_getcpu()` on physical Linux nodes and thread-local pinned core tracking
/// under simulated `K3_CPU_ONLINE_MASK` / `K3_SIMULATE_A100`, non-Linux hosts, and Miri.
#[must_use]
pub(crate) fn current_execution_cpu_mask() -> u64 {
    #[cfg(miri)]
    {
        return get_current_thread_core()
            .filter(|&c| c < 64)
            .map_or(0, |c| 1_u64 << c);
    }
    #[cfg(all(not(miri), target_os = "linux"))]
    {
        if std::env::var_os("K3_CPU_ONLINE_MASK").is_some()
            || std::env::var_os("K3_SIMULATE_A100").is_some()
        {
            if let Some(core) = get_current_thread_core() {
                if core < 64 {
                    return 1_u64 << core;
                }
            }
        }
        // SAFETY: `sched_getcpu` has no preconditions.
        let cpu = unsafe { libc::sched_getcpu() };
        u64::try_from(cpu)
            .ok()
            .filter(|&c| c < 64)
            .map_or(0, |c| 1_u64 << c)
    }
    #[cfg(all(not(miri), not(target_os = "linux")))]
    {
        if let Some(core) = get_current_thread_core() {
            if core < 64 {
                return 1_u64 << core;
            }
        }
        0
    }
}

/// Core bitmask for Cluster 0 (`X100` `OoO` cores 0–7, `zvl256b`).
pub const X100_CLUSTER_MASK: u64 = 0x00FF;

/// Core bitmask for Cluster 1 (`A100` In-Order cores 8–15, `zvl1024b`).
pub const A100_CLUSTER_MASK: u64 = 0xFF00;

/// Default maximum compute threads on `X100` to preserve control-plane cores per ADR-007.
pub const X100_DEFAULT_MAX_COMPUTE_THREADS: usize = 6;

/// Default maximum compute threads on `A100` (all 8 dedicated compute cores).
pub const A100_DEFAULT_MAX_COMPUTE_THREADS: usize = 8;

/// Hardware RVV 1.0 vector register length (`VLEN`) domain on `SpacemiT` K3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlenDomain {
    /// 256-bit vector registers (`zvl256b`) on `X100` cores 0–7.
    Zvl256b,
    /// 1024-bit vector registers (`zvl1024b`) on `A100` cores 8–15.
    Zvl1024b,
    /// Heterogeneous dual-cluster configuration with partitioned VLEN domains.
    Heterogeneous,
}

impl VlenDomain {
    /// Returns the canonical ISA extension name for this VLEN domain.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Zvl256b => "zvl256b",
            Self::Zvl1024b => "zvl1024b",
            Self::Heterogeneous => "heterogeneous",
        }
    }
}

/// Named processor cluster presets on the `SpacemiT` K3 `SoC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterPreset {
    /// Cluster 0: `X100` cores 0–7 (`0x00FF`, `zvl256b`, default max 6 compute threads per ADR-007).
    X100,
    /// Cluster 1: `A100` cores 8–15 (`0xFF00`, `zvl1024b`, up to 8 compute threads).
    A100,
}

impl ClusterPreset {
    /// Parses a cluster preset string (`"x100"` or `"a100"`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if `name` is not `"x100"` or `"a100"`.
    pub fn parse(name: &str, requested_threads: usize) -> Result<Self, K3Error> {
        match name {
            "x100" => Ok(Self::X100),
            "a100" => Ok(Self::A100),
            other => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: format!("unknown cluster preset '{other}'; expected 'x100' or 'a100'"),
            }),
        }
    }

    /// Returns the canonical cluster name (`"x100"` or `"a100"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::X100 => "x100",
            Self::A100 => "a100",
        }
    }

    /// Returns the 64-bit physical core bitmask for this cluster.
    #[must_use]
    pub const fn core_mask(&self) -> u64 {
        match self {
            Self::X100 => X100_CLUSTER_MASK,
            Self::A100 => A100_CLUSTER_MASK,
        }
    }

    /// Returns the hardware [`VlenDomain`] for this cluster.
    #[must_use]
    pub const fn vlen_domain(&self) -> VlenDomain {
        match self {
            Self::X100 => VlenDomain::Zvl256b,
            Self::A100 => VlenDomain::Zvl1024b,
        }
    }

    /// Returns the default maximum compute thread count for this cluster (ADR-007).
    #[must_use]
    pub const fn default_max_threads(&self) -> usize {
        match self {
            Self::X100 => X100_DEFAULT_MAX_COMPUTE_THREADS,
            Self::A100 => A100_DEFAULT_MAX_COMPUTE_THREADS,
        }
    }

    /// Converts a [`ComputePool`] to an optional [`ClusterPreset`].
    ///
    /// Returns `Some(ClusterPreset::X100)` for [`ComputePool::X100`],
    /// `Some(ClusterPreset::A100)` for [`ComputePool::A100`], and `None` for
    /// [`ComputePool::Both`].
    #[must_use]
    pub const fn from_compute_pool(pool: ComputePool) -> Option<Self> {
        match pool {
            ComputePool::X100 => Some(Self::X100),
            ComputePool::A100 => Some(Self::A100),
            ComputePool::Both => None,
        }
    }

    /// Converts this cluster preset to its corresponding [`ComputePool`].
    #[must_use]
    pub const fn to_compute_pool(&self) -> ComputePool {
        match self {
            Self::X100 => ComputePool::X100,
            Self::A100 => ComputePool::A100,
        }
    }
}

impl TryFrom<ComputePool> for ClusterPreset {
    type Error = K3Error;

    fn try_from(pool: ComputePool) -> Result<Self, Self::Error> {
        match pool {
            ComputePool::X100 => Ok(Self::X100),
            ComputePool::A100 => Ok(Self::A100),
            ComputePool::Both => Err(K3Error::CapabilityMismatch {
                architecture: "cluster".to_owned(),
                reason: "ComputePool::Both cannot be converted to a single ClusterPreset"
                    .to_owned(),
            }),
        }
    }
}

/// Extension trait converting a [`ComputePool`] into an [`Option<ClusterPreset>`].
pub trait IntoClusterPreset {
    /// Converts this compute pool into an optional cluster preset.
    ///
    /// Returns `Some(ClusterPreset::X100)` for [`ComputePool::X100`],
    /// `Some(ClusterPreset::A100)` for [`ComputePool::A100`], and `None` for
    /// [`ComputePool::Both`].
    #[must_use]
    fn into_cluster_preset(self) -> Option<ClusterPreset>;
}

impl IntoClusterPreset for ComputePool {
    fn into_cluster_preset(self) -> Option<ClusterPreset> {
        ClusterPreset::from_compute_pool(self)
    }
}

/// Cross-cluster workload distribution strategy (US-03 R1, ADR-009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitStrategy {
    /// Prefill on one cluster, token decode on another.
    Phase,
    /// Matrix operations on A100 (IME), Attention/Norm on X100 (`OoO`).
    Op,
    /// Concurrent row partitioning proportional to measured cluster throughput.
    Row,
}

impl SplitStrategy {
    /// Parses a split strategy string ("phase", "op", "row").
    ///
    /// # Errors
    /// Returns [`K3Error::AffinityError`] if `name` is unrecognized.
    pub fn parse(name: &str, requested_threads: usize) -> Result<Self, K3Error> {
        match name.trim().to_ascii_lowercase().as_str() {
            "phase" => Ok(Self::Phase),
            "op" => Ok(Self::Op),
            "row" => Ok(Self::Row),
            other => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: format!(
                    "unknown split strategy '{other}'; expected 'phase', 'op', or 'row'"
                ),
            }),
        }
    }

    /// Returns canonical string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Phase => "phase",
            Self::Op => "op",
            Self::Row => "row",
        }
    }
}

/// Target processor cluster placement and split strategy configuration (US-03 R1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClusterSelection {
    /// Dedicated placement on a single cluster (`x100` or `a100`).
    Single(ClusterPreset),
    /// Dual-cluster allocation partitioned across both `X100` and `A100` with an explicit split policy.
    Both { split: SplitStrategy },
}

impl ClusterSelection {
    /// Parses cluster selection from cluster name and optional split strategy.
    ///
    /// # Errors
    /// Returns [`K3Error::AffinityError`] if:
    /// - `cluster` is "both" without an explicit `split`.
    /// - `cluster` is "x100" or "a100" with an unexpected `split`.
    /// - `cluster` or `split` strings are unknown.
    pub fn parse(
        cluster: Option<&str>,
        split: Option<&str>,
        requested_threads: usize,
    ) -> Result<Option<Self>, K3Error> {
        match (cluster, split) {
            (None, None) => Ok(None),
            (Some("x100"), None) => Ok(Some(Self::Single(ClusterPreset::X100))),
            (Some("a100"), None) => Ok(Some(Self::Single(ClusterPreset::A100))),
            (Some("both"), Some(s)) => {
                let strategy = SplitStrategy::parse(s, requested_threads)?;
                Ok(Some(Self::Both { split: strategy }))
            }
            (Some("both"), None) => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: "--cluster both requires an explicit --split <phase|op|row> policy"
                    .to_owned(),
            }),
            (Some(c), Some(_)) if c == "x100" || c == "a100" => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: format!(
                    "--split policy is only valid with '--cluster both', but '--cluster {c}' was requested"
                ),
            }),
            (None, Some(_)) => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: "--split policy requires '--cluster both'".to_owned(),
            }),
            (Some(other), _) => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: format!(
                    "unknown cluster preset '{other}'; expected 'both', 'x100', or 'a100'"
                ),
            }),
        }
    }

    /// Returns `true` if this selection is dual-cluster.
    #[must_use]
    pub const fn is_both(&self) -> bool {
        matches!(self, Self::Both { .. })
    }

    /// Returns split strategy if dual-cluster.
    #[must_use]
    pub const fn split_strategy(&self) -> Option<SplitStrategy> {
        match self {
            Self::Both { split } => Some(*split),
            Self::Single(_) => None,
        }
    }
}

/// Minimum prompt token count at which compute-bound batched prefill saturates `A100` `VLEN=1024`
/// `XSMTVDotII` matrix acceleration on heterogeneous `SpacemiT` K3 hardware (`#160`).
pub const PHASE_SPLIT_MIN_PROMPT_TOKENS: usize = 8;

/// Execution-phase cluster scheduling policy for heterogeneous `SpacemiT` K3 hardware.
///
/// Partitions multi-token prompt prefill and single-token autoregressive decode across
/// `X100` (`0x00FF`, `OoO`, `zvl256b`) and `A100` (`0xFF00`, In-Order, `zvl1024b` + `IME`)
/// clusters without mixing asymmetric `VLEN` domains inside a single Rayon thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClusterPhasePolicy {
    /// Automatically select the cluster phase policy from model architecture, prompt length,
    /// online CPU topology, and `XSMTVDotII` (`IME`) availability (`A100Only` when `A100`
    /// `VLEN=1024` + `IME` is online, or `X100Only` fallback when `A100`/`IME` is unavailable).
    #[default]
    Auto,
    /// Execute prompt prefill on `X100` (`0x00FF`) and single-token decode on `A100` (`0xFF00`).
    X100PrefillA100Decode,
    /// Execute prompt prefill on `A100` (`0xFF00`) and single-token decode on `X100` (`0x00FF`).
    A100PrefillX100Decode,
    /// Execute prompt prefill on `A100` (`0xFF00`) and single-token decode on `A100` (`0xFF00`).
    A100PrefillA100Decode,
    /// Execute both prompt prefill and decode exclusively on `A100` (`0xFF00`).
    A100Only,
    /// Execute both prompt prefill and decode exclusively on `X100` (`0x00FF`).
    X100Only,
    /// Dual-cluster row-split configuration (`A100` prefill/decode when `Dual` pool is used in
    /// `SinglePromptEngine`).
    RowSplit,
}

impl ClusterPhasePolicy {
    /// Returns all supported [`ClusterPhasePolicy`] variants in canonical order.
    #[must_use]
    pub const fn all() -> [Self; 7] {
        [
            Self::Auto,
            Self::X100PrefillA100Decode,
            Self::A100PrefillX100Decode,
            Self::A100PrefillA100Decode,
            Self::A100Only,
            Self::X100Only,
            Self::RowSplit,
        ]
    }

    /// Returns the canonical kebab-case CLI / serde string for this policy.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::X100PrefillA100Decode => "x100-prefill-a100-decode",
            Self::A100PrefillX100Decode => "a100-prefill-x100-decode",
            Self::A100PrefillA100Decode => "a100-prefill-a100-decode",
            Self::A100Only => "a100-only",
            Self::X100Only => "x100-only",
            Self::RowSplit => "row-split",
        }
    }

    /// Parses a [`ClusterPhasePolicy`] from its canonical kebab-case identifier.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if `name` does not match one of the supported policy names.
    pub fn parse(name: &str) -> Result<Self, K3Error> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "x100-prefill-a100-decode" => Ok(Self::X100PrefillA100Decode),
            "a100-prefill-x100-decode" => Ok(Self::A100PrefillX100Decode),
            "a100-prefill-a100-decode" => Ok(Self::A100PrefillA100Decode),
            "a100-only" => Ok(Self::A100Only),
            "x100-only" => Ok(Self::X100Only),
            "row-split" => Ok(Self::RowSplit),
            other => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: format!(
                    "unknown cluster phase policy '{other}'; expected 'auto', 'x100-prefill-a100-decode', 'a100-prefill-x100-decode', 'a100-prefill-a100-decode', 'a100-only', 'x100-only', or 'row-split'"
                ),
            }),
        }
    }
}

impl std::fmt::Display for ClusterPhasePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ClusterPhasePolicy {
    type Err = K3Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Reports whether `A100` cores (`0xFF00`) in `online_cpu_mask` support `SpacemiT` `XSMTVDotII`
/// (`IME`) matrix execution.
#[must_use]
pub fn has_a100_ime_for_mask(online_cpu_mask: u64) -> bool {
    let a100_mask = online_cpu_mask & A100_CLUSTER_MASK;
    if a100_mask == 0 {
        return false;
    }
    #[cfg(all(
        target_os = "linux",
        target_arch = "riscv64",
        not(miri),
        feature = "jero-kernels-rvv"
    ))]
    {
        if std::env::var_os("K3_CPU_ONLINE_MASK").is_some()
            || std::env::var_os("K3_SIMULATE_A100").is_some()
        {
            return true;
        }
        static CACHED_A100_IME: OnceLock<bool> = OnceLock::new();
        *CACHED_A100_IME.get_or_init(|| {
            let core = a100_mask.trailing_zeros() as usize;
            crate::forward::probe_ime_on_core(core)
        })
    }
    #[cfg(not(all(
        target_os = "linux",
        target_arch = "riscv64",
        not(miri),
        feature = "jero-kernels-rvv"
    )))]
    {
        true
    }
}

/// Selects the deterministic cluster phase scheduling policy for `metadata`, `prompt_tokens`,
/// and `online_cpu_mask`.
#[must_use]
pub fn select_phase_policy(
    metadata: &ModelMetadata,
    prompt_tokens: usize,
    online_cpu_mask: u64,
) -> ClusterPhasePolicy {
    select_phase_policy_for_arch(metadata.architecture(), prompt_tokens, online_cpu_mask)
}

/// Selects the deterministic cluster phase scheduling policy for `metadata`, `prompt_tokens`,
/// `online_cpu_mask`, and explicit `has_ime` capability flag (`#160`).
#[must_use]
pub fn select_phase_policy_with_ime(
    metadata: &ModelMetadata,
    prompt_tokens: usize,
    online_cpu_mask: u64,
    has_ime: bool,
) -> ClusterPhasePolicy {
    select_phase_policy_for_arch_with_ime(
        metadata.architecture(),
        prompt_tokens,
        online_cpu_mask,
        has_ime,
    )
}

/// Selects the deterministic cluster phase scheduling policy for `architecture`, `prompt_tokens`,
/// and `online_cpu_mask`.
#[must_use]
pub fn select_phase_policy_for_arch(
    architecture: &str,
    prompt_tokens: usize,
    online_cpu_mask: u64,
) -> ClusterPhasePolicy {
    let has_ime = has_a100_ime_for_mask(online_cpu_mask);
    select_phase_policy_for_arch_with_ime(architecture, prompt_tokens, online_cpu_mask, has_ime)
}

/// Selects the deterministic cluster phase scheduling policy for `architecture`, `prompt_tokens`,
/// `online_cpu_mask`, and explicit `XSMTVDotII` (`has_ime`) availability (`#160`).
#[must_use]
pub fn select_phase_policy_for_arch_with_ime(
    architecture: &str,
    prompt_tokens: usize,
    online_cpu_mask: u64,
    has_ime: bool,
) -> ClusterPhasePolicy {
    let x100_present = (online_cpu_mask & X100_CLUSTER_MASK) != 0;
    let matrix_present = (online_cpu_mask & A100_CLUSTER_MASK) != 0;
    if !matrix_present {
        return ClusterPhasePolicy::X100Only;
    }
    if !x100_present {
        return ClusterPhasePolicy::A100Only;
    }
    if matches!(
        architecture,
        "deberta-v3" | "shieldgemma" | "granite" | "llama-guard"
    ) {
        return ClusterPhasePolicy::X100Only;
    }
    if !has_ime {
        // Fall back cleanly to the OoO X100 cluster when XSMTVDotII matrix acceleration is absent.
        return ClusterPhasePolicy::X100Only;
    }
    // Both micro-prompts (`< PHASE_SPLIT_MIN_PROMPT_TOKENS`) and compute-bound batched prefill
    // (`>= PHASE_SPLIT_MIN_PROMPT_TOKENS`) as well as BitNet models execute on A100 + XSMTVDotII
    // whenever A100 IME cores are online.
    let _ = (prompt_tokens, PHASE_SPLIT_MIN_PROMPT_TOKENS);
    ClusterPhasePolicy::A100Only
}

/// Resolves `policy` into a concrete (non-[`ClusterPhasePolicy::Auto`]) policy for `architecture`,
/// `prompt_tokens`, and `online_cpu_mask`.
#[must_use]
pub fn resolve_phase_policy(
    policy: ClusterPhasePolicy,
    architecture: &str,
    prompt_tokens: usize,
    online_cpu_mask: u64,
) -> ClusterPhasePolicy {
    let has_ime = has_a100_ime_for_mask(online_cpu_mask);
    resolve_phase_policy_with_ime(
        policy,
        architecture,
        prompt_tokens,
        online_cpu_mask,
        has_ime,
    )
}

/// Resolves `policy` into a concrete (non-[`ClusterPhasePolicy::Auto`]) policy with explicit
/// `XSMTVDotII` (`has_ime`) availability.
#[must_use]
pub fn resolve_phase_policy_with_ime(
    policy: ClusterPhasePolicy,
    architecture: &str,
    prompt_tokens: usize,
    online_cpu_mask: u64,
    has_ime: bool,
) -> ClusterPhasePolicy {
    match policy {
        ClusterPhasePolicy::Auto => select_phase_policy_for_arch_with_ime(
            architecture,
            prompt_tokens,
            online_cpu_mask,
            has_ime,
        ),
        concrete => concrete,
    }
}

/// Trait abstracting host online CPU topology discovery.
pub trait CpuTopologyInspector {
    /// Returns a 64-bit bitmask of currently online host CPUs.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if online CPU topology cannot be determined.
    fn online_cpu_mask(&self) -> Result<u64, K3Error>;
}

/// Reads online CPU topology from `/sys/devices/system/cpu/online` (or `K3_CPU_ONLINE_MASK` override).
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemCpuTopology;

impl SystemCpuTopology {
    /// Creates a new [`SystemCpuTopology`] inspector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl CpuTopologyInspector for SystemCpuTopology {
    fn online_cpu_mask(&self) -> Result<u64, K3Error> {
        if let Ok(override_str) = std::env::var("K3_CPU_ONLINE_MASK") {
            let trimmed = override_str.trim();
            let parsed = if let Some(hex) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16)
            } else {
                trimmed.parse::<u64>()
            };
            if let Ok(mask) = parsed {
                if mask != 0 {
                    return Ok(mask);
                }
            }
        }

        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/online") {
            let mask = parse_cpu_list_mask(&content)?;
            if mask != 0 {
                return Ok(mask);
            }
        }

        // Fallback to libc::_SC_NPROCESSORS_ONLN
        // SAFETY: sysconf(_SC_NPROCESSORS_ONLN) is a standard POSIX query with no side effects.
        let nprocs = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if nprocs > 0 {
            let count = u32::try_from(nprocs).unwrap_or(1).min(64);
            let mask = if count == 64 {
                u64::MAX
            } else {
                (1u64 << count) - 1
            };
            return Ok(mask);
        }

        Ok(0xFFFF)
    }
}

/// Deterministic mock CPU topology for unit testing 16-core `SpacemiT` K3 masks on any host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockCpuTopology {
    /// Bitmask of simulated online CPUs.
    pub online_mask: u64,
}

impl MockCpuTopology {
    /// Creates a mock topology with the specified online CPU bitmask.
    #[must_use]
    pub const fn new(online_mask: u64) -> Self {
        Self { online_mask }
    }

    /// Creates a mock 16-core `SpacemiT` K3 topology (`0xFFFF`: cores 0–15 online).
    #[must_use]
    pub const fn spacemit_k3() -> Self {
        Self {
            online_mask: 0xFFFF,
        }
    }
}

impl CpuTopologyInspector for MockCpuTopology {
    fn online_cpu_mask(&self) -> Result<u64, K3Error> {
        Ok(self.online_mask)
    }
}

/// Parses a Linux CPU range list string (e.g. `"0-7,12-15"` or `"0-15\n"`) into a 64-bit bitmask.
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if the format is malformed.
pub fn parse_cpu_list_mask(list_str: &str) -> Result<u64, K3Error> {
    let trimmed = list_str.trim();
    if trimmed.is_empty() {
        return Err(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: 0,
            reason: "empty CPU list string".to_owned(),
        });
    }

    let mut mask = 0u64;
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start = start_s
                .trim()
                .parse::<u32>()
                .map_err(|_| K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid CPU range start in '{part}'"),
                })?;
            let end = end_s
                .trim()
                .parse::<u32>()
                .map_err(|_| K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid CPU range end in '{part}'"),
                })?;
            if start > end {
                return Err(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("inverted CPU range '{part}'"),
                });
            }
            for cpu in start..=end.min(63) {
                mask |= 1u64 << cpu;
            }
        } else {
            let cpu = part.parse::<u32>().map_err(|_| K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: format!("invalid CPU number '{part}'"),
            })?;
            if cpu < 64 {
                mask |= 1u64 << cpu;
            }
        }
    }

    Ok(mask)
}

/// Validated compute cluster, core bitmask, VLEN domain, and ordered core assignments (US-01.5.1, US-03 R1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAffinity {
    /// Target cluster preset, if single-cluster selected or inferred.
    pub cluster: Option<ClusterPreset>,
    /// Target cluster selection (Single preset or Both with split strategy).
    pub selection: Option<ClusterSelection>,
    /// Active hardware VLEN domain (`zvl256b`, `zvl1024b`, or `Heterogeneous`).
    pub vlen_domain: VlenDomain,
    /// Effective core bitmask intersected with online CPUs.
    pub effective_mask: u64,
    /// Number of worker threads to spawn.
    pub threads: usize,
    /// Exact 1-to-1 physical core IDs assigned to each worker thread (`len == threads`).
    pub assigned_cores: Vec<usize>,
    /// Exact core IDs assigned to X100 workers (cores 0-7).
    pub x100_cores: Vec<usize>,
    /// Exact core IDs assigned to A100 workers (cores 8-15).
    pub a100_cores: Vec<usize>,
}

impl ResolvedAffinity {
    /// Returns `true` if this resolved configuration spans both clusters.
    #[must_use]
    pub fn is_dual_cluster(&self) -> bool {
        self.selection
            .as_ref()
            .is_some_and(ClusterSelection::is_both)
    }

    /// Returns the split strategy if this configuration is dual-cluster.
    #[must_use]
    pub fn split_strategy(&self) -> Option<SplitStrategy> {
        self.selection
            .as_ref()
            .and_then(ClusterSelection::split_strategy)
    }
}

/// Resolves and validates requested `--cluster`, `--cores`, and `--threads` against the live system
/// CPU topology (US-01.5.1, US-01 Scenario 2).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if the configuration is invalid or exceeds available cores.
pub fn resolve_affinity(
    cluster: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
) -> Result<ResolvedAffinity, K3Error> {
    resolve_affinity_with(
        cluster,
        explicit_mask,
        requested_threads,
        &SystemCpuTopology::new(),
    )
}

/// Resolves and validates requested `--cluster`, `--cores`, and `--threads` against a provided
/// [`CpuTopologyInspector`] (US-01.5.1, ADR-007, ADR-008).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if configuration is invalid or constraints are violated.
pub fn resolve_affinity_with<T: CpuTopologyInspector>(
    cluster: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
    topology: &T,
) -> Result<ResolvedAffinity, K3Error> {
    let req_threads = requested_threads.unwrap_or(1);
    let selection = ClusterSelection::parse(cluster, None, req_threads)?;
    resolve_cluster_affinity_with(selection, explicit_mask, requested_threads, topology)
}

/// Resolves and validates requested `--cluster`, `--split`, `--cores`, and `--threads` against the
/// live system CPU topology (US-03 R1).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if configuration is invalid or constraints are violated.
pub fn resolve_affinity_split(
    cluster: Option<&str>,
    split: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
) -> Result<ResolvedAffinity, K3Error> {
    resolve_affinity_split_with(
        cluster,
        split,
        explicit_mask,
        requested_threads,
        &SystemCpuTopology::new(),
    )
}

/// Resolves and validates requested `--cluster`, `--split`, `--cores`, and `--threads` against a
/// provided [`CpuTopologyInspector`] (US-03 R1).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if configuration is invalid or constraints are violated.
pub fn resolve_affinity_split_with<T: CpuTopologyInspector>(
    cluster: Option<&str>,
    split: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
    topology: &T,
) -> Result<ResolvedAffinity, K3Error> {
    let req_threads = requested_threads.unwrap_or(1);
    let selection = ClusterSelection::parse(cluster, split, req_threads)?;
    resolve_cluster_affinity_with(selection, explicit_mask, requested_threads, topology)
}

/// Resolves and validates a [`ClusterSelection`] against live system CPU topology.
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if configuration is invalid or constraints are violated.
pub fn resolve_cluster_affinity(
    selection: Option<ClusterSelection>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
) -> Result<ResolvedAffinity, K3Error> {
    resolve_cluster_affinity_with(
        selection,
        explicit_mask,
        requested_threads,
        &SystemCpuTopology::new(),
    )
}

/// Resolves and validates a [`ClusterSelection`], core bitmask, and thread count against a
/// [`CpuTopologyInspector`] (US-01, US-02, US-03).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if configuration is invalid, cores are offline, or
/// cluster isolation is violated.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn resolve_cluster_affinity_with<T: CpuTopologyInspector>(
    selection: Option<ClusterSelection>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
    topology: &T,
) -> Result<ResolvedAffinity, K3Error> {
    let online_mask = topology.online_cpu_mask()?;
    let req_threads_for_err = requested_threads.unwrap_or(1);

    if requested_threads == Some(0) {
        return Err(K3Error::AffinityError {
            requested_mask: explicit_mask.unwrap_or(0),
            requested_threads: 0,
            reason: "requested worker thread count must be at least 1".to_owned(),
        });
    }

    if explicit_mask == Some(0) {
        return Err(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: req_threads_for_err,
            reason: "core affinity mask cannot be 0x0000".to_owned(),
        });
    }

    let is_dual = selection.as_ref().is_some_and(ClusterSelection::is_both);

    let candidate_mask = match (&selection, explicit_mask) {
        (Some(ClusterSelection::Single(p)), Some(mask)) => {
            if (mask & !p.core_mask()) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: mask,
                    requested_threads: req_threads_for_err,
                    reason: format!(
                        "explicit core mask 0x{mask:04X} contains cores outside cluster '{}' (0x{:04X})",
                        p.as_str(),
                        p.core_mask()
                    ),
                });
            }
            mask
        }
        (Some(ClusterSelection::Single(p)), None) => p.core_mask(),
        (Some(ClusterSelection::Both { split }), Some(mask)) => {
            if (mask & !(X100_CLUSTER_MASK | A100_CLUSTER_MASK)) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: mask,
                    requested_threads: req_threads_for_err,
                    reason: format!(
                        "explicit core mask 0x{mask:04X} contains cores outside valid clusters 0..15"
                    ),
                });
            }
            let has_x100 = (mask & X100_CLUSTER_MASK) != 0;
            let has_a100 = (mask & A100_CLUSTER_MASK) != 0;
            if !has_x100 || !has_a100 {
                return Err(K3Error::AffinityError {
                    requested_mask: mask,
                    requested_threads: req_threads_for_err,
                    reason: format!(
                        "dual-cluster split policy '{}' requires cores on both X100 (0x00FF) and A100 (0xFF00), but mask 0x{mask:04X} does not cover both clusters",
                        split.as_str()
                    ),
                });
            }
            mask
        }
        (Some(ClusterSelection::Both { .. }), None) => X100_CLUSTER_MASK | A100_CLUSTER_MASK,
        (None, Some(mask)) => mask,
        (None, None) => X100_CLUSTER_MASK & online_mask,
    };

    // Enforce VLEN domain isolation guard: single-cluster pool must not mix X100 (0x00FF) and A100 (0xFF00)
    let has_x100_cores = (candidate_mask & X100_CLUSTER_MASK) != 0;
    let has_a100_cores = (candidate_mask & A100_CLUSTER_MASK) != 0;
    let has_non_x100_cores = (candidate_mask & !X100_CLUSTER_MASK) != 0;
    if !is_dual && has_x100_cores && has_non_x100_cores {
        return Err(K3Error::AffinityError {
            requested_mask: candidate_mask,
            requested_threads: req_threads_for_err,
            reason: format!(
                "core mask 0x{candidate_mask:04X} mixes asymmetric X100 (0x00FF, zvl256b) and A100 (0xFF00, zvl1024b) VLEN domains without an explicit cross-cluster split policy"
            ),
        });
    }

    let effective_mask = candidate_mask & online_mask;
    if effective_mask == 0 {
        return Err(K3Error::AffinityError {
            requested_mask: candidate_mask,
            requested_threads: req_threads_for_err,
            reason: format!(
                "no online CPUs in requested mask 0x{candidate_mask:04X} (online mask: 0x{online_mask:04X})"
            ),
        });
    }

    if is_dual {
        let x100_online: Vec<usize> = (0..8)
            .filter(|&c| (effective_mask & (1 << c)) != 0)
            .collect();
        let a100_online: Vec<usize> = (8..16)
            .filter(|&c| (effective_mask & (1 << c)) != 0)
            .collect();

        if x100_online.is_empty() || a100_online.is_empty() {
            return Err(K3Error::AffinityError {
                requested_mask: candidate_mask,
                requested_threads: req_threads_for_err,
                reason: format!(
                    "no online CPUs for both clusters in mask 0x{candidate_mask:04X} (x100 online: {}, a100 online: {})",
                    x100_online.len(),
                    a100_online.len()
                ),
            });
        }

        let max_x100 = if explicit_mask.is_some() {
            x100_online.len()
        } else {
            x100_online.len().min(X100_DEFAULT_MAX_COMPUTE_THREADS)
        };
        let max_a100 = a100_online.len().min(A100_DEFAULT_MAX_COMPUTE_THREADS);
        let total_cap = max_x100 + max_a100;

        let (nx, na) = match requested_threads {
            Some(n) => {
                if n < 2 {
                    return Err(K3Error::AffinityError {
                        requested_mask: effective_mask,
                        requested_threads: n,
                        reason: "dual-cluster execution requires at least 2 worker threads (at least 1 per cluster)".to_owned(),
                    });
                }
                if n > total_cap {
                    return Err(K3Error::AffinityError {
                        requested_mask: candidate_mask,
                        requested_threads: n,
                        reason: format!(
                            "requested {n} threads exceeds available capacity of {total_cap} cores across both clusters (max X100: {max_x100}, max A100: {max_a100})"
                        ),
                    });
                }
                let half = n / 2;
                let mut nx_calc = half.min(max_x100).max(1);
                let mut na_calc = n - nx_calc;
                if na_calc > max_a100 {
                    na_calc = max_a100;
                    nx_calc = n - na_calc;
                }
                if nx_calc > max_x100 {
                    nx_calc = max_x100;
                    na_calc = n - nx_calc;
                }
                (nx_calc, na_calc)
            }
            None => (max_x100, max_a100),
        };

        let x100_assigned: Vec<usize> = x100_online.into_iter().take(nx).collect();
        let a100_assigned: Vec<usize> = a100_online.into_iter().take(na).collect();

        let mut assigned_cores = Vec::with_capacity(nx + na);
        assigned_cores.extend_from_slice(&x100_assigned);
        assigned_cores.extend_from_slice(&a100_assigned);

        let active_mask = assigned_cores
            .iter()
            .fold(0u64, |acc, &c| acc | (1u64 << c));

        Ok(ResolvedAffinity {
            cluster: None,
            selection,
            vlen_domain: VlenDomain::Heterogeneous,
            effective_mask: active_mask,
            threads: assigned_cores.len(),
            assigned_cores,
            x100_cores: x100_assigned,
            a100_cores: a100_assigned,
        })
    } else {
        // Single-cluster resolution (preserves US-01 logic exactly)
        let preset = match &selection {
            Some(ClusterSelection::Single(p)) => Some(*p),
            _ => None,
        };
        let available_cores: Vec<usize> = (0..64usize)
            .filter(|&bit| (effective_mask & (1u64 << bit)) != 0)
            .collect();

        let threads = match (requested_threads, preset, explicit_mask) {
            (Some(n), Some(ClusterPreset::X100), None) => {
                if n > X100_DEFAULT_MAX_COMPUTE_THREADS {
                    return Err(K3Error::AffinityError {
                        requested_mask: effective_mask,
                        requested_threads: n,
                        reason: format!(
                            "requested {n} compute threads on cluster 'x100' exceeds ADR-007 default cap of {X100_DEFAULT_MAX_COMPUTE_THREADS} (2 active + 1 spare reserved for control plane)"
                        ),
                    });
                }
                n
            }
            (Some(n), _, _) => n,
            (None, Some(p), None) => available_cores.len().min(p.default_max_threads()),
            (None, _, Some(_)) => available_cores.len(),
            (None, None, None) => 1,
        };

        if threads > available_cores.len() {
            return Err(K3Error::AffinityError {
                requested_mask: candidate_mask,
                requested_threads: threads,
                reason: format!(
                    "requested {threads} threads exceeds {} available cores in mask 0x{effective_mask:04X}",
                    available_cores.len()
                ),
            });
        }

        let assigned_cores: Vec<usize> = available_cores.into_iter().take(threads).collect();
        let active_mask = assigned_cores
            .iter()
            .fold(0u64, |acc, &core| acc | (1u64 << core));

        let inferred_cluster = preset.or({
            if has_a100_cores && !has_x100_cores {
                Some(ClusterPreset::A100)
            } else if has_x100_cores && !has_a100_cores {
                Some(ClusterPreset::X100)
            } else {
                None
            }
        });

        let vlen_domain = match inferred_cluster {
            Some(ClusterPreset::A100) => VlenDomain::Zvl1024b,
            Some(ClusterPreset::X100) | None => VlenDomain::Zvl256b,
        };

        let mut x100_cores = Vec::new();
        let mut a100_cores = Vec::new();
        for &core in &assigned_cores {
            if core < 8 {
                x100_cores.push(core);
            } else {
                a100_cores.push(core);
            }
        }

        Ok(ResolvedAffinity {
            cluster: inferred_cluster,
            selection,
            vlen_domain,
            effective_mask: active_mask,
            threads,
            assigned_cores,
            x100_cores,
            a100_cores,
        })
    }
}

/// Runtime affinity verification report for a single pinned compute worker thread (US-01.5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerAffinityStatus {
    /// Worker index within the pool (`0..threads`).
    pub worker_index: usize,
    /// Linux kernel thread ID (`TID`).
    pub tid: i64,
    /// Assigned physical core ID (`0..63`).
    pub assigned_core: usize,
    /// Verified CPU affinity bitmask queried from `sched_getaffinity`.
    pub sched_affinity_mask: u64,
    /// Verified `Cpus_allowed_list` read from `/proc/self/task/<tid>/status`.
    pub proc_cpus_allowed_list: String,
}

pub type JobFn = Arc<dyn Fn(usize, usize) + Send + Sync>;

#[derive(Clone, Copy)]
struct SyncConstPtr<T>(*const T);

impl<T> SyncConstPtr<T> {
    const fn from_ref(r: &T) -> Self {
        Self(std::ptr::from_ref(r))
    }

    // SAFETY: Caller ensures pointee outlives dereference and has valid provenance for `&T`.
    unsafe fn as_ref(&self) -> &T {
        // SAFETY: Upheld by caller contract documented on `as_ref`.
        unsafe { &*self.0 }
    }
}

// SAFETY: Only constructed when caller synchronizes worker completion before `&T` ends and `T: Sync`.
unsafe impl<T> Send for SyncConstPtr<T> {}
// SAFETY: Read-only pointer wrapper for `T: Sync`.
unsafe impl<T> Sync for SyncConstPtr<T> {}

#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);

impl<T> SyncMutPtr<T> {
    const fn as_ptr(&self) -> *mut T {
        self.0
    }
}

// SAFETY: Workers strictly access mutually disjoint index ranges bounded by barrier synchronization.
unsafe impl<T: Send> Send for SyncMutPtr<T> {}
// SAFETY: Workers strictly access mutually disjoint index ranges within caller-owned allocation.
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}

#[derive(Clone, Copy)]
struct RawWorkerTask {
    invoke: unsafe fn(*const (), usize, usize),
    data: *const (),
}

impl RawWorkerTask {
    fn from_fn<F: Fn(usize, usize) + Sync>(f: &F) -> Self {
        unsafe fn call<F: Fn(usize, usize) + Sync>(ptr: *const (), worker_idx: usize, n: usize) {
            // SAFETY: `ptr` borrows caller's `F` until barrier wait completes.
            let f = unsafe { &*ptr.cast::<F>() };
            f(worker_idx, n);
        }
        Self {
            invoke: call::<F>,
            data: std::ptr::from_ref(f).cast(),
        }
    }

    // SAFETY: Caller ensures task target remains live until worker completion barrier exits.
    unsafe fn call(&self, worker_idx: usize, num_workers: usize) {
        // SAFETY: Upheld by caller contract documented on `call`.
        unsafe { (self.invoke)(self.data, worker_idx, num_workers) };
    }
}

// SAFETY: Borrowed closure lifetime is strictly bounded by synchronous barrier completion.
unsafe impl Send for RawWorkerTask {}
// SAFETY: Shared access to `F: Sync` across workers for barrier-bounded execution.
unsafe impl Sync for RawWorkerTask {}

#[derive(Clone, Copy)]
struct RawInstallReq {
    invoke: unsafe fn(*const ()),
    slot: *const (),
    pool: *const WorkerPool,
}

// SAFETY: Caller waits on `host_done` before popping `slot` or `&WorkerPool`.
unsafe impl Send for RawInstallReq {}

#[derive(Clone, Copy)]
struct ParallelDispatch {
    task: RawWorkerTask,
    workers_only: bool,
}

struct PoolSharedState {
    parallel_gen: usize,
    parallel_task: Option<ParallelDispatch>,
    install_gen: usize,
    install_req: Option<RawInstallReq>,
    shutdown: bool,
}

struct DedicatedShared {
    profile: CoordinationProfile,
    num_workers: usize,
    exec_mutex: Mutex<()>,
    state: Mutex<PoolSharedState>,
    worker0_cvar: Condvar,
    workers_rest_cvar: Condvar,
    host_done: (Mutex<bool>, Condvar),
    #[cfg(not(miri))]
    session_shared: SessionShared,
    shutdown: Arc<AtomicBool>,
    wait_slots: Arc<[WorkerWaitSlot]>,
    spin_barrier: Option<Arc<SpinBarrier>>,
    futex_barrier: Option<Arc<FutexBarrier>>,
    workers_spin_barrier: Option<Arc<SpinBarrier>>,
    workers_futex_barrier: Option<Arc<FutexBarrier>>,
    coordinator_sense: AtomicBool,
    worker0_seen_parallel_gen: AtomicUsize,
    worker0_workers_sense: AtomicBool,
}

impl DedicatedShared {
    fn wait_all_worker(&self, worker_index: usize, local_sense: &mut bool) {
        if let Some(barrier) = &self.spin_barrier {
            let wait_time = barrier.wait(local_sense, &self.shutdown);
            self.wait_slots[worker_index].record_wait(wait_time);
        } else if let Some(barrier) = &self.futex_barrier {
            let wait_time = barrier.wait(&self.shutdown);
            self.wait_slots[worker_index].record_wait(wait_time);
        }
    }

    fn wait_nested_worker(&self, worker_index: usize, local_sense: &mut bool) {
        if let Some(barrier) = &self.workers_spin_barrier {
            let wait_time = barrier.wait(local_sense, &self.shutdown);
            self.wait_slots[worker_index].record_wait(wait_time);
        } else if let Some(barrier) = &self.workers_futex_barrier {
            let wait_time = barrier.wait(&self.shutdown);
            self.wait_slots[worker_index].record_wait(wait_time);
        }
    }

    fn wait_coordinator_all(&self) {
        if let Some(barrier) = &self.spin_barrier {
            let mut local_sense = self.coordinator_sense.load(Ordering::Relaxed);
            barrier.wait(&mut local_sense, &self.shutdown);
            self.coordinator_sense.store(local_sense, Ordering::Relaxed);
        } else if let Some(barrier) = &self.futex_barrier {
            barrier.wait(&self.shutdown);
        }
    }

    fn clear_parallel_task(&self) {
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.parallel_task = None;
    }
}

struct DedicatedAsyncWaiter<'a> {
    shared: &'a DedicatedShared,
    _exec_guard: MutexGuard<'a, ()>,
}

impl Drop for DedicatedAsyncWaiter<'_> {
    fn drop(&mut self) {
        self.shared.wait_coordinator_all();
        self.shared.clear_parallel_task();
    }
}

enum Worker0Action {
    Install(Option<RawInstallReq>),
    Parallel(Option<ParallelDispatch>),
}

fn run_worker0_loop(shared: &Arc<DedicatedShared>) {
    let mut external_sense = false;
    let mut seen_install_gen = 0usize;
    let num_workers = shared.num_workers;
    let shared_ptr = Arc::as_ptr(shared);

    loop {
        let action = {
            let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
            while !guard.shutdown
                && !shared.shutdown.load(Ordering::Relaxed)
                && guard.install_gen == seen_install_gen
                && guard.parallel_gen == shared.worker0_seen_parallel_gen.load(Ordering::Relaxed)
            {
                guard = shared
                    .worker0_cvar
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if guard.shutdown || shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if guard.install_gen == seen_install_gen {
                shared
                    .worker0_seen_parallel_gen
                    .store(guard.parallel_gen, Ordering::Relaxed);
                Worker0Action::Parallel(guard.parallel_task)
            } else {
                seen_install_gen = guard.install_gen;
                Worker0Action::Install(guard.install_req.take())
            }
        };

        match action {
            Worker0Action::Install(Some(req)) => {
                let w_guard = TlsDedicatedWorkerGuard::enter(shared_ptr, 0, false);
                // SAFETY: Caller keeps `&WorkerPool` alive while blocked waiting on `host_done`.
                let pool_guard = TlsPoolGuard::enter(unsafe { &*req.pool });
                // SAFETY: `invoke` catches panics and writes the result before returning.
                unsafe { (req.invoke)(req.slot) };
                drop(pool_guard);
                drop(w_guard);

                let (done_lock, done_cvar) = &shared.host_done;
                let mut done = done_lock.lock().unwrap_or_else(PoisonError::into_inner);
                *done = true;
                done_cvar.notify_all();
            }
            Worker0Action::Install(None) => {
                let (done_lock, done_cvar) = &shared.host_done;
                let mut done = done_lock.lock().unwrap_or_else(PoisonError::into_inner);
                *done = true;
                done_cvar.notify_all();
            }
            Worker0Action::Parallel(Some(dispatch)) => {
                let _w_guard = TlsDedicatedWorkerGuard::enter(shared_ptr, 0, true);
                // SAFETY: Borrowed task remains valid until barrier synchronization completes.
                unsafe { (dispatch.task.invoke)(dispatch.task.data, 0, num_workers) };
                shared.wait_all_worker(0, &mut external_sense);
            }
            Worker0Action::Parallel(None) => {
                shared.wait_all_worker(0, &mut external_sense);
            }
        }

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
    }
}

fn run_subworker_loop(shared: &Arc<DedicatedShared>, worker_index: usize) {
    let mut external_sense = false;
    let mut workers_sense = false;
    let mut seen_parallel_gen = 0usize;
    let num_workers = shared.num_workers;
    let shared_ptr = Arc::as_ptr(shared);

    loop {
        let dispatch = {
            let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
            while !guard.shutdown
                && !shared.shutdown.load(Ordering::Relaxed)
                && guard.parallel_gen == seen_parallel_gen
            {
                guard = shared
                    .workers_rest_cvar
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if guard.shutdown || shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            seen_parallel_gen = guard.parallel_gen;
            guard.parallel_task
        };

        if let Some(d) = dispatch {
            let _w_guard = TlsDedicatedWorkerGuard::enter(shared_ptr, worker_index, true);
            // SAFETY: Borrowed task remains valid until barrier synchronization completes.
            unsafe { (d.task.invoke)(d.task.data, worker_index, num_workers) };
            if d.workers_only {
                shared.wait_nested_worker(worker_index, &mut workers_sense);
            } else {
                shared.wait_all_worker(worker_index, &mut external_sense);
            }
        }

        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
    }
}

fn dedicated_worker_entry(
    shared: &Arc<DedicatedShared>,
    worker_index: usize,
    assigned_core: usize,
    req_mask: u64,
    tx: &mpsc::Sender<Result<WorkerAffinityStatus, K3Error>>,
) {
    set_current_thread_core(assigned_core);
    // SAFETY: `raw_pin_thread` uses only fixed-frame integer syscalls to pin the worker.
    let pin_res = unsafe { raw_pin_thread(assigned_core) };
    let pin_result = verify_pinned_thread(
        worker_index,
        assigned_core,
        req_mask,
        shared.num_workers,
        pin_res,
    );
    let is_ok = pin_result.is_ok();
    let _ = tx.send(pin_result);
    if !is_ok {
        return;
    }
    if worker_index == 0 {
        run_worker0_loop(shared);
    } else {
        run_subworker_loop(shared, worker_index);
    }
}

#[cfg(not(miri))]
struct RayonSignal {
    done: Mutex<bool>,
    cvar: Condvar,
}

#[cfg(not(miri))]
impl RayonSignal {
    const fn new() -> Self {
        Self {
            done: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    fn mark_done(&self) {
        let mut g = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        *g = true;
        self.cvar.notify_all();
    }

    fn wait(&self) {
        let mut g = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        while !*g {
            g = self.cvar.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

#[cfg(not(miri))]
struct RayonFinishNotifier<'a>(&'a RayonSignal);

#[cfg(not(miri))]
impl Drop for RayonFinishNotifier<'_> {
    fn drop(&mut self) {
        self.0.mark_done();
    }
}

#[cfg(not(miri))]
struct RayonSignalWaiter<'a>(&'a RayonSignal);

#[cfg(not(miri))]
impl Drop for RayonSignalWaiter<'_> {
    fn drop(&mut self) {
        self.0.wait();
    }
}

#[cfg(not(miri))]
#[derive(Clone, Copy)]
struct SessionStep {
    invoke: unsafe fn(*const ()),
    slot: *const (),
}

#[cfg(not(miri))]
// SAFETY: Caller blocks inside `PhaseSession::run` until worker marks `SessionCmd::Done`.
unsafe impl Send for SessionStep {}

#[cfg(not(miri))]
enum SessionCmd {
    Idle,
    Run(SessionStep),
    Done,
    Close,
}

#[cfg(not(miri))]
fn pre_touch_sync_pair<T>(mutex: &Mutex<T>, cvar: &Condvar) {
    let guard = mutex.lock().unwrap_or_else(PoisonError::into_inner);
    let (guard, _) = cvar
        .wait_timeout(guard, Duration::ZERO)
        .unwrap_or_else(PoisonError::into_inner);
    cvar.notify_all();
    drop(guard);
}

#[cfg(not(miri))]
struct SessionShared {
    mutex: Mutex<SessionCmd>,
    cvar: Condvar,
}

#[cfg(not(miri))]
impl SessionShared {
    fn new() -> Self {
        let shared = Self {
            mutex: Mutex::new(SessionCmd::Idle),
            cvar: Condvar::new(),
        };
        pre_touch_sync_pair(&shared.mutex, &shared.cvar);
        shared
    }
}

#[cfg(not(miri))]
fn run_session_worker_loop(shared: &SessionShared) {
    loop {
        let step = {
            let mut guard = shared.mutex.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                match *guard {
                    SessionCmd::Idle | SessionCmd::Done => {
                        guard = shared
                            .cvar
                            .wait(guard)
                            .unwrap_or_else(PoisonError::into_inner);
                    }
                    SessionCmd::Run(step) => {
                        *guard = SessionCmd::Idle;
                        break Some(step);
                    }
                    SessionCmd::Close => break None,
                }
            }
        };
        let Some(step) = step else {
            return;
        };
        // SAFETY: Caller's `SessionStep` cell remains valid until we publish `SessionCmd::Done`.
        unsafe { (step.invoke)(step.slot) };
        let mut guard = shared.mutex.lock().unwrap_or_else(PoisonError::into_inner);
        *guard = SessionCmd::Done;
        shared.cvar.notify_all();
    }
}

/// Borrowed handle for running multiple sequential operations inside a single hoisted
/// [`WorkerPool::install_for_phase`] entry on the target cluster thread.
#[derive(Debug)]
pub struct PhaseSession<'a> {
    mode: PhaseSessionMode<'a>,
}

#[derive(Debug)]
enum PhaseSessionMode<'a> {
    Inline {
        _marker: PhantomData<&'a WorkerPool>,
    },
    #[cfg(not(miri))]
    WorkerLoop { ctrl: &'a SessionShared },
}

#[cfg(not(miri))]
impl std::fmt::Debug for SessionShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionShared").finish_non_exhaustive()
    }
}

impl PhaseSession<'_> {
    /// Executes `op` on the target cluster worker thread owned by this active session.
    ///
    /// # Panics
    ///
    /// Propagates any panic raised by `op` on the worker thread.
    pub fn run<OP, T>(&mut self, op: OP) -> T
    where
        OP: FnOnce() -> T + Send,
        T: Send,
    {
        match &self.mode {
            PhaseSessionMode::Inline { .. } => op(),
            #[cfg(not(miri))]
            PhaseSessionMode::WorkerLoop { ctrl } => {
                unsafe fn invoke_step<OP, T>(slot_ptr: *const ())
                where
                    OP: FnOnce() -> T + Send,
                    T: Send,
                {
                    // SAFETY: Caller passes `&UnsafeCell` alive until `SessionCmd::Done` is signaled.
                    let slot = unsafe {
                        &*slot_ptr.cast::<UnsafeCell<(Option<OP>, Option<thread::Result<T>>)>>()
                    };
                    // SAFETY: Only the active session worker accesses `slot` between `Run` and `Done`.
                    let (op_opt, res_opt) = unsafe { &mut *slot.get() };
                    let op = op_opt.take().expect("phase session step missing closure");
                    *res_opt = Some(catch_unwind(AssertUnwindSafe(op)));
                }

                let cell: UnsafeCell<(Option<OP>, Option<thread::Result<T>>)> =
                    UnsafeCell::new((Some(op), None));
                let step = SessionStep {
                    invoke: invoke_step::<OP, T>,
                    slot: std::ptr::from_ref(&cell).cast(),
                };
                {
                    let mut guard = ctrl.mutex.lock().unwrap_or_else(PoisonError::into_inner);
                    *guard = SessionCmd::Run(step);
                    ctrl.cvar.notify_all();
                    while !matches!(*guard, SessionCmd::Done) {
                        guard = ctrl
                            .cvar
                            .wait(guard)
                            .unwrap_or_else(PoisonError::into_inner);
                    }
                    *guard = SessionCmd::Idle;
                }
                // SAFETY: Worker stored `SessionCmd::Done` and no longer accesses `cell`.
                let res = unsafe { &mut *cell.get() }
                    .1
                    .take()
                    .expect("phase session step missing result");
                match res {
                    Ok(val) => val,
                    Err(payload) => resume_unwind(payload),
                }
            }
        }
    }
}

/// Cache-line aligned (64-byte) per-worker barrier wait accumulator (US-02 R1, ADR-006).
///
/// Prevents cross-core cache line false sharing on `SpacemiT` K3 L1/L2 caches when multiple
/// pinned worker threads concurrently accumulate barrier wait times.
#[repr(align(64))]
#[derive(Debug, Default)]
pub struct WorkerWaitSlot {
    /// Cumulative barrier wait duration in nanoseconds.
    accumulated_nanos: AtomicU64,
    /// Number of barrier synchronization events recorded.
    barrier_count: AtomicU64,
}

impl WorkerWaitSlot {
    /// Creates a new zeroed wait slot.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            accumulated_nanos: AtomicU64::new(0),
            barrier_count: AtomicU64::new(0),
        }
    }

    /// Accumulates wait duration into this slot.
    pub fn record_wait(&self, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.accumulated_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.barrier_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates wait duration in nanoseconds into this slot.
    pub fn record_wait_nanos(&self, delta_nanos: u64) {
        self.accumulated_nanos
            .fetch_add(delta_nanos, Ordering::Relaxed);
        self.barrier_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns accumulated wait duration in nanoseconds.
    #[must_use]
    pub fn wait_nanos(&self) -> u64 {
        self.accumulated_nanos.load(Ordering::Relaxed)
    }

    /// Returns accumulated wait duration in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wait_ms(&self) -> f64 {
        self.wait_nanos() as f64 / 1_000_000.0
    }

    /// Returns accumulated wait duration as a [`Duration`].
    #[must_use]
    pub fn wait_duration(&self) -> Duration {
        Duration::from_nanos(self.wait_nanos())
    }

    /// Returns number of barrier events recorded.
    #[must_use]
    pub fn barrier_count(&self) -> u64 {
        self.barrier_count.load(Ordering::Relaxed)
    }

    /// Resets accumulator and count to zero.
    pub fn reset(&self) {
        self.accumulated_nanos.store(0, Ordering::Relaxed);
        self.barrier_count.store(0, Ordering::Relaxed);
    }
}

/// Sense-reversing atomic busy-polling barrier (US-02 R1).
///
/// Uses `core::hint::spin_loop()` for sub-microsecond synchronization between pinned worker cores.
pub struct SpinBarrier {
    num_threads: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
    sense: AtomicBool,
}

impl std::fmt::Debug for SpinBarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpinBarrier")
            .field("num_threads", &self.num_threads)
            .field("count", &self.count.load(Ordering::Relaxed))
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .field("sense", &self.sense.load(Ordering::Relaxed))
            .finish()
    }
}

impl SpinBarrier {
    /// Creates a new spin barrier for `num_threads` participants.
    #[must_use]
    pub const fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
        }
    }

    /// Synchronizes calling thread. Returns elapsed wait duration spent spinning.
    pub fn wait(&self, local_sense: &mut bool, shutdown: &AtomicBool) -> Duration {
        if self.num_threads <= 1 {
            return Duration::ZERO;
        }

        *local_sense = !*local_sense;
        let expected_sense = *local_sense;
        let start = Instant::now();

        let arrival = self.count.fetch_add(1, Ordering::AcqRel);
        if arrival + 1 == self.num_threads {
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            self.sense.store(expected_sense, Ordering::Release);
            Duration::ZERO
        } else {
            while self.sense.load(Ordering::Acquire) != expected_sense {
                if shutdown.load(Ordering::Relaxed) {
                    return Duration::ZERO;
                }
                core::hint::spin_loop();
            }
            start.elapsed()
        }
    }
}

/// Thread-sleep yielding barrier via [`Mutex`] and [`Condvar`] (US-02 R1).
///
/// Yields CPU cycles to the Linux kernel scheduler during inter-worker wait intervals.
pub struct FutexBarrier {
    num_threads: usize,
    state: Mutex<FutexBarrierState>,
    cvar: Condvar,
}

struct FutexBarrierState {
    count: usize,
    generation: usize,
}

impl std::fmt::Debug for FutexBarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FutexBarrier")
            .field("num_threads", &self.num_threads)
            .finish_non_exhaustive()
    }
}

impl FutexBarrier {
    /// Creates a new futex barrier for `num_threads` participants.
    #[must_use]
    pub const fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            state: Mutex::new(FutexBarrierState {
                count: 0,
                generation: 0,
            }),
            cvar: Condvar::new(),
        }
    }

    /// Synchronizes calling thread yielding CPU cycles. Returns elapsed wait duration.
    pub fn wait(&self, shutdown: &AtomicBool) -> Duration {
        if self.num_threads <= 1 {
            return Duration::ZERO;
        }

        let start = Instant::now();
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let current_gen = guard.generation;
        guard.count += 1;
        if guard.count == self.num_threads {
            guard.count = 0;
            guard.generation = guard.generation.wrapping_add(1);
            self.cvar.notify_all();
            Duration::ZERO
        } else {
            while guard.generation == current_gen && !shutdown.load(Ordering::Relaxed) {
                guard = self
                    .cvar
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            start.elapsed()
        }
    }

    /// Notifies all waiting threads to wake immediately for shutdown.
    pub fn notify_shutdown(&self) {
        self.cvar.notify_all();
    }
}

enum PoolBackend {
    DedicatedThreads {
        shared: Arc<DedicatedShared>,
        handles: Vec<JoinHandle<()>>,
    },
    Rayon {
        pool: rayon::ThreadPool,
    },
    Dual {
        x100_pool: Box<WorkerPool>,
        a100_pool: Box<WorkerPool>,
    },
}

impl std::fmt::Debug for PoolBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DedicatedThreads { shared, .. } => f
                .debug_struct("DedicatedThreads")
                .field("profile", &shared.profile)
                .finish_non_exhaustive(),
            Self::Rayon { .. } => f.debug_struct("Rayon").finish_non_exhaustive(),
            Self::Dual {
                x100_pool,
                a100_pool,
            } => f
                .debug_struct("Dual")
                .field("x100_pool", x100_pool)
                .field("a100_pool", a100_pool)
                .finish(),
        }
    }
}

/// Affinity-pinned multi-core compute worker pool (US-01.5.2, US-02 R1, US-03 R1).
///
/// Supports three multi-core coordination profiles (`spin`, `futex`, `rayon`),
/// dual-cluster composite backends, pins workers 1-to-1 to physical cores,
/// and instruments cumulative barrier wait times.
pub struct WorkerPool {
    affinity: ResolvedAffinity,
    profile: CoordinationProfile,
    phase_policy: Option<ClusterPhasePolicy>,
    worker_statuses: Vec<WorkerAffinityStatus>,
    wait_slots: Arc<[WorkerWaitSlot]>,
    backend: PoolBackend,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("affinity", &self.affinity)
            .field("profile", &self.profile)
            .field("phase_policy", &self.phase_policy)
            .field("worker_statuses", &self.worker_statuses)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl WorkerPool {
    /// Spawns a new [`WorkerPool`] using the default coordination profile ([`CoordinationProfile::Spin`]).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if thread creation, `sched_setaffinity`, or runtime
    /// affinity verification fails.
    pub fn new(affinity: ResolvedAffinity) -> Result<Self, K3Error> {
        Self::new_with_profile(affinity, CoordinationProfile::Spin)
    }

    /// Spawns a new [`WorkerPool`] configured with a specific [`CoordinationProfile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if thread creation, `sched_setaffinity`, or runtime
    /// affinity verification fails.
    #[allow(clippy::too_many_lines)]
    pub fn new_with_profile(
        affinity: ResolvedAffinity,
        profile: CoordinationProfile,
    ) -> Result<Self, K3Error> {
        if affinity.is_dual_cluster() {
            let x100_mask = affinity
                .x100_cores
                .iter()
                .fold(0u64, |acc, &c| acc | (1u64 << c));
            let a100_mask = affinity
                .a100_cores
                .iter()
                .fold(0u64, |acc, &c| acc | (1u64 << c));

            let x100_aff = ResolvedAffinity {
                cluster: Some(ClusterPreset::X100),
                selection: Some(ClusterSelection::Single(ClusterPreset::X100)),
                vlen_domain: VlenDomain::Zvl256b,
                effective_mask: x100_mask,
                threads: affinity.x100_cores.len(),
                assigned_cores: affinity.x100_cores.clone(),
                x100_cores: affinity.x100_cores.clone(),
                a100_cores: Vec::new(),
            };
            let a100_aff = ResolvedAffinity {
                cluster: Some(ClusterPreset::A100),
                selection: Some(ClusterSelection::Single(ClusterPreset::A100)),
                vlen_domain: VlenDomain::Zvl1024b,
                effective_mask: a100_mask,
                threads: affinity.a100_cores.len(),
                assigned_cores: affinity.a100_cores.clone(),
                x100_cores: Vec::new(),
                a100_cores: affinity.a100_cores.clone(),
            };

            let p1 = Self::new_with_profile(x100_aff, profile)?;
            let p2 = Self::new_with_profile(a100_aff, profile)?;

            let mut combined_statuses = p1.worker_statuses().to_vec();
            let nx = combined_statuses.len();
            for mut st in p2.worker_statuses().to_vec() {
                st.worker_index += nx;
                combined_statuses.push(st);
            }

            let mut combined_slots: Vec<WorkerWaitSlot> = Vec::with_capacity(affinity.threads);
            for _ in 0..affinity.threads {
                combined_slots.push(WorkerWaitSlot::new());
            }

            return Ok(Self {
                affinity,
                profile,
                phase_policy: None,
                worker_statuses: combined_statuses,
                wait_slots: combined_slots.into(),
                backend: PoolBackend::Dual {
                    x100_pool: Box::new(p1),
                    a100_pool: Box::new(p2),
                },
            });
        }

        let num_workers = affinity.threads;
        if num_workers == 0 || affinity.assigned_cores.len() != num_workers {
            return Err(K3Error::AffinityError {
                requested_mask: affinity.effective_mask,
                requested_threads: num_workers,
                reason: "worker pool requires at least 1 assigned core matching thread count"
                    .to_owned(),
            });
        }

        let mut slots = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            slots.push(WorkerWaitSlot::new());
        }
        let wait_slots: Arc<[WorkerWaitSlot]> = slots.into();

        match profile {
            CoordinationProfile::Spin | CoordinationProfile::Futex => {
                let spin_barrier = (profile == CoordinationProfile::Spin)
                    .then(|| Arc::new(SpinBarrier::new(num_workers + 1)));
                let futex_barrier = (profile == CoordinationProfile::Futex)
                    .then(|| Arc::new(FutexBarrier::new(num_workers + 1)));
                let workers_spin_barrier = (profile == CoordinationProfile::Spin
                    && num_workers > 1)
                    .then(|| Arc::new(SpinBarrier::new(num_workers)));
                let workers_futex_barrier = (profile == CoordinationProfile::Futex
                    && num_workers > 1)
                    .then(|| Arc::new(FutexBarrier::new(num_workers)));

                let shared = Arc::new(DedicatedShared {
                    profile,
                    num_workers,
                    exec_mutex: Mutex::new(()),
                    state: Mutex::new(PoolSharedState {
                        parallel_gen: 0,
                        parallel_task: None,
                        install_gen: 0,
                        install_req: None,
                        shutdown: false,
                    }),
                    worker0_cvar: Condvar::new(),
                    workers_rest_cvar: Condvar::new(),
                    host_done: (Mutex::new(false), Condvar::new()),
                    #[cfg(not(miri))]
                    session_shared: SessionShared::new(),
                    shutdown: Arc::new(AtomicBool::new(false)),
                    wait_slots: Arc::clone(&wait_slots),
                    spin_barrier,
                    futex_barrier,
                    workers_spin_barrier,
                    workers_futex_barrier,
                    coordinator_sense: AtomicBool::new(false),
                    worker0_seen_parallel_gen: AtomicUsize::new(0),
                    worker0_workers_sense: AtomicBool::new(false),
                });
                #[cfg(not(miri))]
                {
                    drop(shared.exec_mutex.lock());
                    pre_touch_sync_pair(&shared.state, &shared.worker0_cvar);
                    pre_touch_sync_pair(&shared.state, &shared.workers_rest_cvar);
                    pre_touch_sync_pair(&shared.host_done.0, &shared.host_done.1);
                }

                let (init_tx, init_rx) = mpsc::channel::<Result<WorkerAffinityStatus, K3Error>>();
                let mut handles = Vec::with_capacity(num_workers);

                for (worker_index, &assigned_core) in affinity.assigned_cores.iter().enumerate() {
                    let shared_clone = Arc::clone(&shared);
                    let tx = init_tx.clone();
                    let req_mask = affinity.effective_mask;

                    let handle = thread::Builder::new()
                        .name(format!("k3-worker-{worker_index}"))
                        .spawn(move || {
                            dedicated_worker_entry(
                                &shared_clone,
                                worker_index,
                                assigned_core,
                                req_mask,
                                &tx,
                            );
                        })
                        .map_err(|err| {
                            shutdown_dedicated(&shared, &mut handles);
                            K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!(
                                    "failed to spawn worker thread {worker_index}: {err}"
                                ),
                            }
                        })?;

                    handles.push(handle);
                }
                drop(init_tx);

                let mut worker_statuses = Vec::with_capacity(num_workers);
                for _ in 0..num_workers {
                    match init_rx.recv() {
                        Ok(Ok(status)) => {
                            worker_statuses.push(status);
                        }
                        Ok(Err(err)) => {
                            shutdown_dedicated(&shared, &mut handles);
                            return Err(err);
                        }
                        Err(recv_err) => {
                            shutdown_dedicated(&shared, &mut handles);
                            return Err(K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!(
                                    "worker thread terminated before reporting affinity: {recv_err}"
                                ),
                            });
                        }
                    }
                }
                worker_statuses.sort_by_key(|s| s.worker_index);

                Ok(Self {
                    affinity,
                    profile,
                    phase_policy: None,
                    worker_statuses,
                    wait_slots,
                    backend: PoolBackend::DedicatedThreads { shared, handles },
                })
            }
            CoordinationProfile::Rayon => {
                let (init_tx, init_rx) = mpsc::channel::<Result<WorkerAffinityStatus, K3Error>>();
                let assigned_cores = affinity.assigned_cores.clone();
                let req_mask = affinity.effective_mask;

                let rayon_pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_workers)
                    .thread_name(|i| format!("k3-rayon-worker-{i}"))
                    .start_handler(move |worker_idx| {
                        let assigned_core = assigned_cores[worker_idx];
                        set_current_thread_core(assigned_core);
                        // SAFETY: `raw_pin_thread` uses only fixed-frame integer syscalls to pin the worker.
                        let pin_res = unsafe { raw_pin_thread(assigned_core) };
                        let res = verify_pinned_thread(
                            worker_idx,
                            assigned_core,
                            req_mask,
                            num_workers,
                            pin_res,
                        );
                        let _ = init_tx.send(res);
                    })
                    .build()
                    .map_err(|err| K3Error::AffinityError {
                        requested_mask: affinity.effective_mask,
                        requested_threads: num_workers,
                        reason: format!("failed to initialize rayon thread pool: {err}"),
                    })?;

                let mut worker_statuses = Vec::with_capacity(num_workers);
                for _ in 0..num_workers {
                    match init_rx.recv() {
                        Ok(Ok(status)) => worker_statuses.push(status),
                        Ok(Err(err)) => return Err(err),
                        Err(recv_err) => {
                            return Err(K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!("rayon worker terminated prematurely: {recv_err}"),
                            });
                        }
                    }
                }
                worker_statuses.sort_by_key(|s| s.worker_index);

                Ok(Self {
                    affinity,
                    profile,
                    phase_policy: None,
                    worker_statuses,
                    wait_slots,
                    backend: PoolBackend::Rayon { pool: rayon_pool },
                })
            }
        }
    }

    /// Spawns a new [`WorkerPool`] using the specified [`CoordinationProfile`] (alias for [`Self::new_with_profile`]).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if configuration or worker initialization fails.
    pub fn with_coordination(
        affinity: ResolvedAffinity,
        coordination: CoordinationProfile,
    ) -> Result<Self, K3Error> {
        Self::new_with_profile(affinity, coordination)
    }

    /// Spawns a phase-aware [`WorkerPool`] for `policy` against the live system CPU topology.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if the requested policy cannot be satisfied on the
    /// active CPU topology or if thread pinning fails.
    pub fn for_phase_policy(
        policy: ClusterPhasePolicy,
        metadata: &ModelMetadata,
        prompt_tokens: usize,
        explicit_mask: Option<u64>,
        requested_threads: Option<usize>,
        profile: CoordinationProfile,
    ) -> Result<Self, K3Error> {
        Self::for_phase_policy_with(
            policy,
            metadata,
            prompt_tokens,
            explicit_mask,
            requested_threads,
            profile,
            &SystemCpuTopology::new(),
        )
    }

    /// Spawns a phase-aware [`WorkerPool`] for `policy` against a provided [`CpuTopologyInspector`].
    ///
    /// Under [`ClusterPhasePolicy::Auto`], selects [`ClusterPhasePolicy::A100Only`] when `A100`
    /// (`VLEN=1024`) with `XSMTVDotII` matrix support is online (or [`ClusterPhasePolicy::X100Only`]
    /// for guardrail architectures / non-IME or single-cluster `cpuset`-restricted CI hosts).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if an explicit policy cannot be satisfied on the
    /// inspected CPU topology or if thread pinning fails.
    pub fn for_phase_policy_with<T: CpuTopologyInspector>(
        policy: ClusterPhasePolicy,
        metadata: &ModelMetadata,
        prompt_tokens: usize,
        explicit_mask: Option<u64>,
        requested_threads: Option<usize>,
        profile: CoordinationProfile,
        topology: &T,
    ) -> Result<Self, K3Error> {
        let online_mask = topology.online_cpu_mask()?;
        let mask_for_selection = explicit_mask.map_or(online_mask, |m| online_mask & m);
        let has_ime = has_a100_ime_for_mask(mask_for_selection);
        let mut resolved_policy = resolve_phase_policy_with_ime(
            policy,
            metadata.architecture(),
            prompt_tokens,
            mask_for_selection,
            has_ime,
        );

        if policy == ClusterPhasePolicy::Auto
            && requested_threads == Some(1)
            && matches!(
                resolved_policy,
                ClusterPhasePolicy::X100PrefillA100Decode
                    | ClusterPhasePolicy::A100PrefillX100Decode
                    | ClusterPhasePolicy::RowSplit
            )
        {
            resolved_policy = if (mask_for_selection & A100_CLUSTER_MASK) != 0 && has_ime {
                ClusterPhasePolicy::A100Only
            } else {
                ClusterPhasePolicy::X100Only
            };
        }

        let build_for_concrete = |concrete: ClusterPhasePolicy| -> Result<Self, K3Error> {
            let selection = match concrete {
                ClusterPhasePolicy::X100PrefillA100Decode
                | ClusterPhasePolicy::A100PrefillX100Decode => Some(ClusterSelection::Both {
                    split: SplitStrategy::Phase,
                }),
                ClusterPhasePolicy::RowSplit => Some(ClusterSelection::Both {
                    split: SplitStrategy::Row,
                }),
                ClusterPhasePolicy::A100Only | ClusterPhasePolicy::A100PrefillA100Decode => {
                    Some(ClusterSelection::Single(ClusterPreset::A100))
                }
                ClusterPhasePolicy::X100Only | ClusterPhasePolicy::Auto => {
                    Some(ClusterSelection::Single(ClusterPreset::X100))
                }
            };
            let affinity = resolve_cluster_affinity_with(
                selection,
                explicit_mask,
                requested_threads,
                topology,
            )?;
            let mut pool = Self::new_with_profile(affinity, profile)?;
            pool.phase_policy = Some(concrete);
            Ok(pool)
        };

        match build_for_concrete(resolved_policy) {
            Ok(pool) => Ok(pool),
            Err(_err)
                if policy == ClusterPhasePolicy::Auto
                    && resolved_policy != ClusterPhasePolicy::X100Only
                    && (mask_for_selection & X100_CLUSTER_MASK) != 0 =>
            {
                build_for_concrete(ClusterPhasePolicy::X100Only)
            }
            Err(err) => Err(err),
        }
    }

    /// Returns the concrete [`ClusterPhasePolicy`] resolved for this pool, if constructed via
    /// [`Self::for_phase_policy`] or [`Self::for_phase_policy_with`].
    #[must_use]
    pub const fn phase_policy(&self) -> Option<ClusterPhasePolicy> {
        self.phase_policy
    }

    /// Returns the active coordination profile for this pool.
    #[must_use]
    pub const fn profile(&self) -> CoordinationProfile {
        self.profile
    }

    /// Returns the active coordination profile for this pool (alias for [`Self::profile`]).
    #[must_use]
    pub const fn coordination_profile(&self) -> CoordinationProfile {
        self.profile
    }

    /// Returns the resolved affinity configuration for this pool.
    #[must_use]
    pub const fn affinity(&self) -> &ResolvedAffinity {
        &self.affinity
    }

    /// Returns the number of pinned worker threads in this pool.
    #[must_use]
    pub const fn thread_count(&self) -> usize {
        self.affinity.threads
    }

    /// Returns the startup affinity verification reports for all worker threads.
    #[must_use]
    pub fn worker_statuses(&self) -> &[WorkerAffinityStatus] {
        &self.worker_statuses
    }

    /// Returns reference to per-worker wait slots.
    #[must_use]
    pub fn wait_slots(&self) -> &[WorkerWaitSlot] {
        &self.wait_slots
    }

    /// Returns the wait slot reference for a given worker index.
    #[must_use]
    pub fn wait_slot(&self, worker_index: usize) -> &WorkerWaitSlot {
        &self.wait_slots[worker_index]
    }

    /// Returns cumulative inter-worker barrier wait time across all workers in milliseconds.
    #[must_use]
    pub fn cumulative_barrier_wait_ms(&self) -> f64 {
        if self.thread_count() <= 1 {
            return 0.0;
        }
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => x100_pool.cumulative_barrier_wait_ms() + a100_pool.cumulative_barrier_wait_ms(),
            _ => self.wait_slots.iter().map(WorkerWaitSlot::wait_ms).sum(),
        }
    }

    /// Returns cumulative inter-worker barrier wait time across all workers in nanoseconds.
    #[must_use]
    pub fn cumulative_barrier_wait_nanos(&self) -> u64 {
        if self.thread_count() <= 1 {
            return 0;
        }
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                x100_pool.cumulative_barrier_wait_nanos()
                    + a100_pool.cumulative_barrier_wait_nanos()
            }
            _ => self.wait_slots.iter().map(WorkerWaitSlot::wait_nanos).sum(),
        }
    }

    /// Returns cumulative inter-worker barrier wait duration across all workers.
    #[must_use]
    pub fn cumulative_barrier_wait(&self) -> Duration {
        Duration::from_nanos(self.cumulative_barrier_wait_nanos())
    }

    /// Resets all per-worker barrier wait time accumulators.
    pub fn reset_barrier_wait(&self) {
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                x100_pool.reset_barrier_wait();
                a100_pool.reset_barrier_wait();
                for slot in self.wait_slots.iter() {
                    slot.reset();
                }
            }
            _ => {
                for slot in self.wait_slots.iter() {
                    slot.reset();
                }
            }
        }
    }

    /// Resets all per-worker barrier wait time accumulators (alias for [`Self::reset_barrier_wait`]).
    pub fn reset_wait_metrics(&self) {
        self.reset_barrier_wait();
    }

    /// Re-reads `/proc/self/task/<tid>/status` (`Cpus_allowed_list`) for every active worker thread
    /// to confirm strict 1-to-1 core binding and zero cross-cluster migration (US-01.5.2, US-03 R3).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if any worker thread has migrated outside its assigned core
    /// or crossed the cluster boundary.
    pub fn verify_live_affinities(&self) -> Result<Vec<WorkerAffinityStatus>, K3Error> {
        if let PoolBackend::Dual {
            x100_pool,
            a100_pool,
        } = &self.backend
        {
            let mut x100_live = x100_pool.verify_live_affinities()?;
            let a100_live = a100_pool.verify_live_affinities()?;
            let nx = x100_live.len();
            for mut status in a100_live {
                status.worker_index += nx;
                x100_live.push(status);
            }
            Ok(x100_live)
        } else {
            let mut live = Vec::with_capacity(self.worker_statuses.len());
            for status in &self.worker_statuses {
                let allowed_list = read_proc_task_cpus_allowed_list(status.tid)
                    .unwrap_or_else(|| status.proc_cpus_allowed_list.clone());
                let parsed_mask = parse_cpu_list_mask(&allowed_list)?;
                let expected_mask = 1u64 << status.assigned_core;
                if parsed_mask != expected_mask {
                    return Err(K3Error::AffinityError {
                        requested_mask: self.affinity.effective_mask,
                        requested_threads: self.affinity.threads,
                        reason: format!(
                            "worker {} (tid {}) affinity drifted: expected core {} (0x{expected_mask:04X}), found Cpus_allowed_list='{allowed_list}' (0x{parsed_mask:04X})",
                            status.worker_index, status.tid, status.assigned_core
                        ),
                    });
                }

                // Enforce VLEN cluster boundary isolation:
                // X100 worker (cores 0-7) must not touch A100 cores (8-15)
                if status.assigned_core < 8 && (parsed_mask & A100_CLUSTER_MASK) != 0 {
                    return Err(K3Error::AffinityError {
                        requested_mask: self.affinity.effective_mask,
                        requested_threads: self.affinity.threads,
                        reason: format!(
                            "worker {} (tid {}) assigned to X100 core {} migrated into A100 cluster mask (0x{parsed_mask:04X})",
                            status.worker_index, status.tid, status.assigned_core
                        ),
                    });
                }
                // A100 worker (cores 8-15) must not touch X100 cores (0-7)
                if status.assigned_core >= 8 && (parsed_mask & X100_CLUSTER_MASK) != 0 {
                    return Err(K3Error::AffinityError {
                        requested_mask: self.affinity.effective_mask,
                        requested_threads: self.affinity.threads,
                        reason: format!(
                            "worker {} (tid {}) assigned to A100 core {} migrated into X100 cluster mask (0x{parsed_mask:04X})",
                            status.worker_index, status.tid, status.assigned_core
                        ),
                    });
                }

                live.push(WorkerAffinityStatus {
                    worker_index: status.worker_index,
                    tid: status.tid,
                    assigned_core: status.assigned_core,
                    sched_affinity_mask: parsed_mask,
                    proc_cpus_allowed_list: allowed_list,
                });
            }
            Ok(live)
        }
    }

    /// Invokes `f` with a reference to the [`WorkerPool`] currently installed on this thread via
    /// [`Self::install`], [`Self::install_for_phase`], or [`Self::with_phase_session`], if any.
    pub fn with_current<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&WorkerPool) -> R,
    {
        CURRENT_WORKER_POOL.with(|cell| {
            let ptr = cell.get()?;
            // SAFETY: `TlsPoolGuard` ensures `ptr` remains live and shared for the exact scope of
            // `install` / `with_phase_session` and restores the previous value on exit.
            let pool = unsafe { &*ptr };
            Some(f(pool))
        })
    }

    fn is_current_worker_on_pool(&self) -> bool {
        match &self.backend {
            PoolBackend::DedicatedThreads { shared, .. } => {
                let shared_ptr = Arc::as_ptr(shared);
                CURRENT_DEDICATED_WORKER.with(|c| {
                    c.get()
                        .is_some_and(|ctx| std::ptr::eq(ctx.shared_ptr, shared_ptr))
                })
            }
            PoolBackend::Rayon { .. } => CURRENT_WORKER_POOL.with(|c| {
                c.get()
                    .is_some_and(|ptr| std::ptr::eq(ptr, std::ptr::from_ref(self)))
            }),
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => x100_pool.is_current_worker_on_pool() || a100_pool.is_current_worker_on_pool(),
        }
    }

    /// Executes `task(worker_index, total_workers)` in parallel across all pinned worker threads
    /// without requiring `'static` closures or heap-allocated `Arc` wrappers.
    pub fn execute_scoped<F>(&self, task: &F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let raw = RawWorkerTask::from_fn(task);
        self.execute_raw_task(raw);
    }

    fn execute_raw_task(&self, task: RawWorkerTask) {
        match &self.backend {
            PoolBackend::Dual { .. } => {
                let total_workers = self.affinity.threads;
                let nx = self.x100_pool().map_or(0, Self::thread_count);
                self.execute_split_scoped(
                    &|w_idx, _| {
                        // SAFETY: Caller waits for both lanes to finish before stack closure exits.
                        unsafe { task.call(w_idx, total_workers) };
                    },
                    &|w_idx, _| {
                        // SAFETY: Caller waits for both lanes to finish before stack closure exits.
                        unsafe { task.call(nx + w_idx, total_workers) };
                    },
                );
            }
            PoolBackend::DedicatedThreads { shared, .. } => {
                dispatch_dedicated_scoped(shared, task);
            }
            PoolBackend::Rayon { pool } => {
                let num_workers = self.affinity.threads;
                let wait_slots = &self.wait_slots;
                let t_start = Instant::now();

                if num_workers <= 16 {
                    let finish_times: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
                    pool.install(|| {
                        let _tls = TlsPoolGuard::enter(self);
                        rayon::scope(|s| {
                            for (worker_idx, finish_ref) in
                                finish_times.iter().take(num_workers).enumerate()
                            {
                                s.spawn(move |_| {
                                    // SAFETY: `rayon::scope` joins all spawned closures before exit.
                                    unsafe { task.call(worker_idx, num_workers) };
                                    let elapsed_ns = u64::try_from(t_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                    finish_ref.store(elapsed_ns, Ordering::Release);
                                });
                            }
                        });
                    });
                    let t_total_ns =
                        u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    for (worker_idx, finish_atomic) in
                        finish_times.iter().take(num_workers).enumerate()
                    {
                        let finish_ns = finish_atomic.load(Ordering::Acquire);
                        let wait_ns = t_total_ns.saturating_sub(finish_ns);
                        wait_slots[worker_idx].record_wait_nanos(wait_ns);
                    }
                } else {
                    let finish_times: Vec<AtomicU64> =
                        (0..num_workers).map(|_| AtomicU64::new(0)).collect();
                    pool.install(|| {
                        let _tls = TlsPoolGuard::enter(self);
                        rayon::scope(|s| {
                            for (worker_idx, finish_ref) in finish_times.iter().enumerate() {
                                s.spawn(move |_| {
                                    // SAFETY: `rayon::scope` joins all spawned closures before exit.
                                    unsafe { task.call(worker_idx, num_workers) };
                                    let elapsed_ns = u64::try_from(t_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                    finish_ref.store(elapsed_ns, Ordering::Release);
                                });
                            }
                        });
                    });
                    let t_total_ns =
                        u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    for (worker_idx, finish_atomic) in finish_times.iter().enumerate() {
                        let finish_ns = finish_atomic.load(Ordering::Acquire);
                        let wait_ns = t_total_ns.saturating_sub(finish_ns);
                        wait_slots[worker_idx].record_wait_nanos(wait_ns);
                    }
                }
            }
        }
    }

    /// Executes `task(worker_index, total_workers)` in parallel across all pinned worker threads
    /// and waits for completion.
    pub fn execute<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.execute_scoped(&task);
    }

    /// Internal non-generic job dispatcher kept compatible with `JobFn` callers.
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute_job(&self, job: JobFn) {
        self.execute_scoped(&|w_idx, total| job(w_idx, total));
    }

    /// Executes `f(range, worker_index)` across balanced non-empty sub-ranges partitioning `0..n_items`.
    pub fn dispatch_indexed<F>(&self, n_items: usize, f: F)
    where
        F: Fn(Range<usize>, usize) + Sync,
    {
        if n_items == 0 {
            return;
        }
        let n_workers = self.thread_count().max(1);
        if n_workers == 1 && self.is_current_worker_on_pool() {
            f(0..n_items, 0);
            return;
        }
        self.execute_scoped(&|w_idx, total| {
            let start = usize::try_from((w_idx as u128 * n_items as u128) / (total.max(1) as u128))
                .unwrap_or(n_items);
            let end =
                usize::try_from(((w_idx + 1) as u128 * n_items as u128) / (total.max(1) as u128))
                    .unwrap_or(n_items);
            if start < end {
                f(start..end, w_idx);
            }
        });
    }

    /// Executes `f(index)` across all `n_items` indices in `0..n_items` on this pool's workers.
    pub fn par_for_each_index<F>(&self, n_items: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        if n_items == 0 {
            return;
        }
        if let PoolBackend::Rayon { pool } = &self.backend {
            use rayon::prelude::{IntoParallelIterator, ParallelIterator};
            if self.is_current_worker_on_pool() {
                (0..n_items).into_par_iter().for_each(&f);
            } else {
                pool.install(|| (0..n_items).into_par_iter().for_each(&f));
            }
            return;
        }
        self.dispatch_indexed(n_items, |range, _| {
            for idx in range {
                f(idx);
            }
        });
    }

    /// Processes disjoint mutable chunks of `slice` in parallel across the pool's pinned workers.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size == 0`.
    pub fn par_chunks_mut<T, F>(&self, slice: &mut [T], chunk_size: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        assert!(chunk_size > 0, "chunk_size must be non-zero");
        if slice.is_empty() {
            return;
        }
        let num_chunks = slice.len().div_ceil(chunk_size);
        if (self.thread_count() <= 1 || num_chunks <= 1) && self.is_current_worker_on_pool() {
            for (c_idx, chunk) in slice.chunks_mut(chunk_size).enumerate() {
                f(c_idx, chunk);
            }
            return;
        }
        if let PoolBackend::Rayon { pool } = &self.backend {
            use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            if self.is_current_worker_on_pool() {
                slice
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(idx, ch)| f(idx, ch));
            } else {
                pool.install(|| {
                    slice
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(idx, ch)| f(idx, ch));
                });
            }
            return;
        }
        let len = slice.len();
        let ptr = SyncMutPtr(slice.as_mut_ptr());
        self.execute_scoped(&|w_idx, n_workers| {
            let c_start =
                usize::try_from((w_idx as u128 * num_chunks as u128) / (n_workers.max(1) as u128))
                    .unwrap_or(num_chunks);
            let c_end = usize::try_from(
                ((w_idx + 1) as u128 * num_chunks as u128) / (n_workers.max(1) as u128),
            )
            .unwrap_or(num_chunks);
            for c_idx in c_start..c_end {
                let elem_start = c_idx * chunk_size;
                let elem_end = (elem_start + chunk_size).min(len);
                // SAFETY: Distinct `c_idx` ranges map to strictly disjoint sub-slices within
                // `slice`, bounded by synchronous worker completion before `slice` is accessed again.
                let chunk = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.as_ptr().add(elem_start),
                        elem_end - elem_start,
                    )
                };
                f(c_idx, chunk);
            }
        });
    }

    /// Fallible parallel counterpart to [`Self::par_chunks_mut`], returning the first error emitted.
    ///
    /// # Errors
    ///
    /// Propagates the first `Err(E)` returned by `f` across any worker.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size == 0`.
    pub fn try_par_chunks_mut<T, E, F>(
        &self,
        slice: &mut [T],
        chunk_size: usize,
        f: F,
    ) -> Result<(), E>
    where
        T: Send,
        E: Send,
        F: Fn(usize, &mut [T]) -> Result<(), E> + Sync,
    {
        assert!(chunk_size > 0, "chunk_size must be non-zero");
        if slice.is_empty() {
            return Ok(());
        }
        let num_chunks = slice.len().div_ceil(chunk_size);
        if (self.thread_count() <= 1 || num_chunks <= 1) && self.is_current_worker_on_pool() {
            for (c_idx, chunk) in slice.chunks_mut(chunk_size).enumerate() {
                f(c_idx, chunk)?;
            }
            return Ok(());
        }
        if let PoolBackend::Rayon { pool } = &self.backend {
            use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
            return if self.is_current_worker_on_pool() {
                slice
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .try_for_each(|(idx, ch)| f(idx, ch))
            } else {
                pool.install(|| {
                    slice
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .try_for_each(|(idx, ch)| f(idx, ch))
                })
            };
        }
        let len = slice.len();
        let ptr = SyncMutPtr(slice.as_mut_ptr());
        let err_flag = AtomicBool::new(false);
        let first_err: Mutex<Option<E>> = Mutex::new(None);
        self.execute_scoped(&|w_idx, n_workers| {
            let c_start =
                usize::try_from((w_idx as u128 * num_chunks as u128) / (n_workers.max(1) as u128))
                    .unwrap_or(num_chunks);
            let c_end = usize::try_from(
                ((w_idx + 1) as u128 * num_chunks as u128) / (n_workers.max(1) as u128),
            )
            .unwrap_or(num_chunks);
            for c_idx in c_start..c_end {
                if err_flag.load(Ordering::Relaxed) {
                    return;
                }
                let elem_start = c_idx * chunk_size;
                let elem_end = (elem_start + chunk_size).min(len);
                // SAFETY: Distinct `c_idx` ranges map to strictly disjoint sub-slices within
                // `slice`, bounded by synchronous worker completion before `slice` is accessed again.
                let chunk = unsafe {
                    std::slice::from_raw_parts_mut(
                        ptr.as_ptr().add(elem_start),
                        elem_end - elem_start,
                    )
                };
                if let Err(err) = f(c_idx, chunk) {
                    err_flag.store(true, Ordering::Relaxed);
                    let mut guard = first_err.lock().unwrap_or_else(PoisonError::into_inner);
                    if guard.is_none() {
                        *guard = Some(err);
                    }
                    return;
                }
            }
        });
        if let Some(err) = first_err
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Executes a task exclusively across X100 worker threads.
    pub fn execute_x100<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        match &self.backend {
            PoolBackend::Dual { x100_pool, .. } => x100_pool.execute_scoped(&task),
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::X100)
                    || (!self.affinity.x100_cores.is_empty() && self.affinity.a100_cores.is_empty())
                {
                    self.execute_scoped(&task);
                }
            }
        }
    }

    /// Executes a task exclusively across A100 worker threads.
    pub fn execute_a100<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        match &self.backend {
            PoolBackend::Dual { a100_pool, .. } => a100_pool.execute_scoped(&task),
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::A100)
                    || (!self.affinity.a100_cores.is_empty() && self.affinity.x100_cores.is_empty())
                {
                    self.execute_scoped(&task);
                }
            }
        }
    }

    /// Executes independent scoped tasks concurrently across persistent X100 and A100 worker
    /// pools without spawning transient OS threads or heap-allocating synchronization state.
    pub fn execute_split_scoped<FX, FA>(&self, x100_task: &FX, a100_task: &FA)
    where
        FX: Fn(usize, usize) + Sync,
        FA: Fn(usize, usize) + Sync,
    {
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                if x100_pool.is_current_worker_on_pool() {
                    dispatch_dual_async_then_sync(a100_pool, a100_task, x100_pool, x100_task);
                } else {
                    dispatch_dual_async_then_sync(x100_pool, x100_task, a100_pool, a100_task);
                }
            }
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::A100) {
                    self.execute_scoped(a100_task);
                } else {
                    self.execute_scoped(x100_task);
                }
            }
        }
    }

    /// Executes independent tasks concurrently across X100 and A100 worker pools.
    pub fn execute_split<FX, FA>(&self, x100_task: FX, a100_task: FA)
    where
        FX: Fn(usize, usize) + Sync,
        FA: Fn(usize, usize) + Sync,
    {
        self.execute_split_scoped(&x100_task, &a100_task);
    }

    /// Returns a reference to the X100 worker pool if operating in dual-cluster mode.
    #[must_use]
    pub fn x100_pool(&self) -> Option<&WorkerPool> {
        match &self.backend {
            PoolBackend::Dual { x100_pool, .. } => Some(x100_pool.as_ref()),
            _ => None,
        }
    }

    /// Returns a reference to the A100 worker pool if operating in dual-cluster mode.
    #[must_use]
    pub fn a100_pool(&self) -> Option<&WorkerPool> {
        match &self.backend {
            PoolBackend::Dual { a100_pool, .. } => Some(a100_pool.as_ref()),
            _ => None,
        }
    }

    /// Returns `true` if this worker pool is backed by a dual-cluster configuration.
    #[must_use]
    pub fn is_dual(&self) -> bool {
        matches!(self.backend, PoolBackend::Dual { .. })
    }

    fn resolve_phase_pool(
        &self,
        phase: crate::engine::EnginePhase,
        policy: ClusterPhasePolicy,
    ) -> &WorkerPool {
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                let a100_has_ime = if policy == ClusterPhasePolicy::Auto {
                    has_a100_ime_for_mask(a100_pool.affinity().effective_mask)
                } else {
                    false
                };
                if select_dual_cluster_a100(policy, phase, a100_has_ime) {
                    a100_pool.as_ref()
                } else {
                    x100_pool.as_ref()
                }
            }
            _ => self,
        }
    }

    /// Executes `op` directly on this pool's pinned worker context (`worker 0` on
    /// [`CoordinationProfile::Spin`]/[`CoordinationProfile::Futex`] or a pool worker on
    /// [`CoordinationProfile::Rayon`]) without creating a secondary Rayon thread pool.
    ///
    /// # Panics
    ///
    /// Propagates any panic raised inside `op`.
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        #[cfg(miri)]
        {
            let _tls = TlsPoolGuard::enter(self);
            op()
        }
        #[cfg(not(miri))]
        match &self.backend {
            PoolBackend::Rayon { pool } => {
                if self.is_current_worker_on_pool() {
                    let _tls = TlsPoolGuard::enter(self);
                    op()
                } else {
                    pool.install(|| {
                        let _tls = TlsPoolGuard::enter(self);
                        op()
                    })
                }
            }
            PoolBackend::DedicatedThreads { shared, .. } => {
                if self.is_current_worker_on_pool() {
                    let _tls = TlsPoolGuard::enter(self);
                    return op();
                }

                let cell: UnsafeCell<(Option<OP>, Option<thread::Result<R>>)> =
                    UnsafeCell::new((Some(op), None));
                let req = RawInstallReq {
                    invoke: invoke_dedicated_install::<OP, R>,
                    slot: std::ptr::from_ref(&cell).cast(),
                    pool: std::ptr::from_ref(self),
                };
                {
                    let exec_guard = start_dedicated_install(shared.as_ref(), req);
                    let _waiter = DedicatedInstallWaiter {
                        shared: shared.as_ref(),
                        _exec_guard: exec_guard,
                    };
                }
                // SAFETY: `DedicatedInstallWaiter` synchronized on `host_done` before returning.
                let res = unsafe { &mut *cell.get() }
                    .1
                    .take()
                    .expect("install slot missing result");
                match res {
                    Ok(val) => val,
                    Err(payload) => resume_unwind(payload),
                }
            }
            PoolBackend::Dual { x100_pool, .. } => x100_pool.install(op),
        }
    }

    /// Executes `op` inside the cluster sub-pool dictated by `(phase, policy)`.
    ///
    /// When `self` is a dual-cluster composite pool (`X100` + `A100`):
    /// - [`ClusterPhasePolicy::A100Only`], [`ClusterPhasePolicy::A100PrefillA100Decode`], and
    ///   [`ClusterPhasePolicy::RowSplit`] execute prefill and decode on `a100_pool`
    ///   (`zvl1024b` + `XSMTVDotII`);
    /// - [`ClusterPhasePolicy::Auto`] executes prefill and decode on `a100_pool` whenever
    ///   `A100` `XSMTVDotII` is online, falling back to `x100_pool` when unavailable;
    /// - [`ClusterPhasePolicy::A100PrefillX100Decode`] executes prefill on `a100_pool` and decode
    ///   on `x100_pool`;
    /// - [`ClusterPhasePolicy::X100PrefillA100Decode`] executes prefill on `x100_pool` and decode
    ///   on `a100_pool`;
    /// - [`ClusterPhasePolicy::X100Only`] executes both phases on `x100_pool`.
    ///
    /// Single-cluster pools execute `op` on their own threads.
    pub fn install_for_phase<OP, R>(
        &self,
        phase: crate::engine::EnginePhase,
        policy: ClusterPhasePolicy,
        op: OP,
    ) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.resolve_phase_pool(phase, policy).install(op)
    }

    /// Enters the target cluster pool for `(phase, policy)` once and exposes a [`PhaseSession`]
    /// to `f` on the caller thread, allowing `!Send` streaming callbacks to interleave across
    /// multiple pinned worker steps without re-entering pool installation per token.
    pub fn with_phase_session<F, R>(
        &self,
        phase: crate::engine::EnginePhase,
        policy: ClusterPhasePolicy,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut PhaseSession<'_>) -> R,
    {
        let target_pool = self.resolve_phase_pool(phase, policy);
        #[cfg(miri)]
        {
            let _tls = TlsPoolGuard::enter(target_pool);
            let mut session = PhaseSession {
                mode: PhaseSessionMode::Inline {
                    _marker: PhantomData,
                },
            };
            f(&mut session)
        }
        #[cfg(not(miri))]
        {
            if target_pool.is_current_worker_on_pool() {
                let _tls = TlsPoolGuard::enter(target_pool);
                let mut session = PhaseSession {
                    mode: PhaseSessionMode::Inline {
                        _marker: PhantomData,
                    },
                };
                return f(&mut session);
            }
            match &target_pool.backend {
                PoolBackend::DedicatedThreads { shared, .. } => {
                    run_dedicated_phase_session(target_pool, shared, f)
                }
                PoolBackend::Rayon { pool } => run_rayon_phase_session(target_pool, pool, f),
                PoolBackend::Dual { .. } => unreachable!("resolve_phase_pool returned dual pool"),
            }
        }
    }
}

#[cfg(not(miri))]
unsafe fn invoke_dedicated_install<OP, R>(slot_ptr: *const ())
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    // SAFETY: Caller keeps `&UnsafeCell` alive until `wait_host_done` completes.
    let slot = unsafe { &*slot_ptr.cast::<UnsafeCell<(Option<OP>, Option<thread::Result<R>>)>>() };
    // SAFETY: Only worker 0 accesses `slot` between submission and `host_done`.
    let (op_opt, res_opt) = unsafe { &mut *slot.get() };
    let op = op_opt.take().expect("install slot missing closure");
    *res_opt = Some(catch_unwind(AssertUnwindSafe(op)));
}

#[cfg(not(miri))]
struct DedicatedInstallWaiter<'a> {
    shared: &'a DedicatedShared,
    _exec_guard: MutexGuard<'a, ()>,
}

#[cfg(not(miri))]
impl Drop for DedicatedInstallWaiter<'_> {
    fn drop(&mut self) {
        wait_host_done(self.shared);
    }
}

#[cfg(not(miri))]
unsafe fn invoke_session_loop(slot_ptr: *const ()) {
    // SAFETY: Caller keeps `SessionShared` alive until `DedicatedSessionGuard` drops.
    let ctrl = unsafe { &*slot_ptr.cast::<SessionShared>() };
    let _ = catch_unwind(AssertUnwindSafe(|| run_session_worker_loop(ctrl)));
}

#[cfg(not(miri))]
struct DedicatedSessionGuard<'a> {
    ctrl: &'a SessionShared,
    shared: &'a DedicatedShared,
    _exec_guard: MutexGuard<'a, ()>,
}

#[cfg(not(miri))]
impl Drop for DedicatedSessionGuard<'_> {
    fn drop(&mut self) {
        {
            let mut g = self
                .ctrl
                .mutex
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            *g = SessionCmd::Close;
            self.ctrl.cvar.notify_all();
        }
        wait_host_done(self.shared);
        let mut g = self
            .ctrl
            .mutex
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *g = SessionCmd::Idle;
    }
}

#[cfg(not(miri))]
struct RayonSessionGuard<'a> {
    ctrl: &'a SessionShared,
    finished: &'a RayonSignal,
}

#[cfg(not(miri))]
impl Drop for RayonSessionGuard<'_> {
    fn drop(&mut self) {
        {
            let mut g = self
                .ctrl
                .mutex
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            *g = SessionCmd::Close;
            self.ctrl.cvar.notify_all();
        }
        self.finished.wait();
    }
}

#[cfg(not(miri))]
fn run_dedicated_phase_session<F, R>(
    target_pool: &WorkerPool,
    shared: &Arc<DedicatedShared>,
    f: F,
) -> R
where
    F: FnOnce(&mut PhaseSession<'_>) -> R,
{
    let session_shared = &shared.session_shared;
    let req = RawInstallReq {
        invoke: invoke_session_loop,
        slot: std::ptr::from_ref(session_shared).cast(),
        pool: std::ptr::from_ref(target_pool),
    };
    let exec_guard = start_dedicated_install(shared.as_ref(), req);
    let _session_guard = DedicatedSessionGuard {
        ctrl: session_shared,
        shared: shared.as_ref(),
        _exec_guard: exec_guard,
    };
    let mut session = PhaseSession {
        mode: PhaseSessionMode::WorkerLoop {
            ctrl: session_shared,
        },
    };
    f(&mut session)
}

#[cfg(not(miri))]
fn run_rayon_phase_session<F, R>(target_pool: &WorkerPool, pool: &rayon::ThreadPool, f: F) -> R
where
    F: FnOnce(&mut PhaseSession<'_>) -> R,
{
    let session_shared = SessionShared::new();
    let finished = RayonSignal::new();
    let ctrl_ptr = SyncConstPtr::from_ref(&session_shared);
    let fin_ptr = SyncConstPtr::from_ref(&finished);
    let pool_ptr = SyncConstPtr::from_ref(target_pool);

    pool.spawn(move || {
        // SAFETY: `RayonSessionGuard` keeps `finished` live until `_fin` drops on exit.
        let _fin = RayonFinishNotifier(unsafe { fin_ptr.as_ref() });
        // SAFETY: `RayonSessionGuard` keeps `target_pool` live until `_fin` drops on exit.
        let _tls = TlsPoolGuard::enter(unsafe { pool_ptr.as_ref() });
        // SAFETY: `RayonSessionGuard` keeps `session_shared` live until `_fin` drops on exit.
        run_session_worker_loop(unsafe { ctrl_ptr.as_ref() });
    });

    let _session_guard = RayonSessionGuard {
        ctrl: &session_shared,
        finished: &finished,
    };
    let mut session = PhaseSession {
        mode: PhaseSessionMode::WorkerLoop {
            ctrl: &session_shared,
        },
    };
    f(&mut session)
}

fn dispatch_dedicated_scoped(shared: &Arc<DedicatedShared>, task: RawWorkerTask) {
    let shared_ptr = Arc::as_ptr(shared);
    if let Some(ctx) = CURRENT_DEDICATED_WORKER.with(Cell::get) {
        if std::ptr::eq(ctx.shared_ptr, shared_ptr) {
            if ctx.worker_index == 0 && !ctx.in_nested {
                if shared.num_workers <= 1 {
                    let _nested = TlsDedicatedWorkerGuard::enter(shared_ptr, 0, true);
                    // SAFETY: Caller closure stays alive on current frame for inline call.
                    unsafe { (task.invoke)(task.data, 0, 1) };
                    return;
                }
                let _nested = TlsDedicatedWorkerGuard::enter(shared_ptr, 0, true);
                {
                    let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
                    guard.parallel_task = Some(ParallelDispatch {
                        task,
                        workers_only: true,
                    });
                    guard.parallel_gen = guard.parallel_gen.wrapping_add(1);
                    shared
                        .worker0_seen_parallel_gen
                        .store(guard.parallel_gen, Ordering::Relaxed);
                    shared.workers_rest_cvar.notify_all();
                }
                // SAFETY: Borrowed task stays valid until `wait_nested_worker` completes.
                unsafe { (task.invoke)(task.data, 0, shared.num_workers) };
                let mut sense = shared.worker0_workers_sense.load(Ordering::Relaxed);
                shared.wait_nested_worker(0, &mut sense);
                shared.worker0_workers_sense.store(sense, Ordering::Relaxed);
                shared.clear_parallel_task();
            } else {
                for w_idx in 0..shared.num_workers {
                    // SAFETY: Recursive fallback runs sequentially inside current frame.
                    unsafe { (task.invoke)(task.data, w_idx, shared.num_workers) };
                }
            }
            return;
        }
    }

    let _waiter = start_dedicated_scoped_async(shared, task);
}

fn start_dedicated_scoped_async(
    shared: &Arc<DedicatedShared>,
    task: RawWorkerTask,
) -> DedicatedAsyncWaiter<'_> {
    let exec_guard = shared
        .exec_mutex
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    {
        let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.parallel_task = Some(ParallelDispatch {
            task,
            workers_only: false,
        });
        guard.parallel_gen = guard.parallel_gen.wrapping_add(1);
        shared.worker0_cvar.notify_one();
        shared.workers_rest_cvar.notify_all();
    }
    DedicatedAsyncWaiter {
        shared: shared.as_ref(),
        _exec_guard: exec_guard,
    }
}

fn dispatch_dual_async_then_sync<FA, FS>(
    async_pool: &WorkerPool,
    async_task: &FA,
    sync_pool: &WorkerPool,
    sync_task: &FS,
) where
    FA: Fn(usize, usize) + Sync,
    FS: Fn(usize, usize) + Sync,
{
    let bg_task = RawWorkerTask::from_fn(async_task);
    let fg_task = RawWorkerTask::from_fn(sync_task);
    match &async_pool.backend {
        PoolBackend::DedicatedThreads { shared, .. } => {
            let _waiter = start_dedicated_scoped_async(shared, bg_task);
            sync_pool.execute_raw_task(fg_task);
        }
        PoolBackend::Rayon { pool } => {
            #[cfg(not(miri))]
            {
                let signal = RayonSignal::new();
                let sig_ptr = SyncConstPtr::from_ref(&signal);
                let pool_ptr = SyncConstPtr::from_ref(async_pool);
                let na = async_pool.thread_count();
                pool.spawn(move || {
                    // SAFETY: `RayonSignalWaiter` guarantees `signal` remains valid until `_fin` drops.
                    let _fin = RayonFinishNotifier(unsafe { sig_ptr.as_ref() });
                    // SAFETY: `RayonSignalWaiter` guarantees `async_pool` remains valid until `_fin` drops.
                    let _tls = TlsPoolGuard::enter(unsafe { pool_ptr.as_ref() });
                    rayon::scope(|s| {
                        for w in 0..na {
                            s.spawn(move |_| {
                                // SAFETY: `RayonSignalWaiter` guarantees `async_task` outlives `scope`.
                                unsafe { bg_task.call(w, na) };
                            });
                        }
                    });
                });
                let _waiter = RayonSignalWaiter(&signal);
                sync_pool.execute_raw_task(fg_task);
            }
            #[cfg(miri)]
            {
                let _ = pool;
                async_pool.execute_raw_task(bg_task);
                sync_pool.execute_raw_task(fg_task);
            }
        }
        PoolBackend::Dual { .. } => {
            async_pool.execute_raw_task(bg_task);
            sync_pool.execute_raw_task(fg_task);
        }
    }
}

fn start_dedicated_install(shared: &DedicatedShared, req: RawInstallReq) -> MutexGuard<'_, ()> {
    let exec_guard = shared
        .exec_mutex
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    {
        let (done_lock, _) = &shared.host_done;
        let mut done = done_lock.lock().unwrap_or_else(PoisonError::into_inner);
        *done = false;
    }
    {
        let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.install_req = Some(req);
        guard.install_gen = guard.install_gen.wrapping_add(1);
        shared.worker0_cvar.notify_one();
    }
    exec_guard
}

fn wait_host_done(shared: &DedicatedShared) {
    let (done_lock, done_cvar) = &shared.host_done;
    let mut done = done_lock.lock().unwrap_or_else(PoisonError::into_inner);
    while !*done {
        done = done_cvar.wait(done).unwrap_or_else(PoisonError::into_inner);
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        if let PoolBackend::DedicatedThreads { shared, handles } = &mut self.backend {
            shutdown_dedicated(shared, handles);
        }
    }
}

fn shutdown_dedicated(shared: &Arc<DedicatedShared>, handles: &mut Vec<JoinHandle<()>>) {
    shared.shutdown.store(true, Ordering::Release);
    {
        let mut guard = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.shutdown = true;
        shared.worker0_cvar.notify_all();
        shared.workers_rest_cvar.notify_all();
    }
    if let Some(fb) = &shared.futex_barrier {
        fb.notify_shutdown();
    }
    if let Some(wfb) = &shared.workers_futex_barrier {
        wfb.notify_shutdown();
    }
    for handle in handles.drain(..) {
        let _ = handle.join();
    }
}

/// Low-level thread pinning helper that performs only integer operations and direct POSIX syscalls.
///
/// CRITICAL ARCHITECTURAL REQUIREMENT:
/// On the `SpacemiT` K3 heterogeneous `SoC`, Cluster 0 (X100) has VLEN = 256 bits (`vlenb = 32`),
/// while Cluster 1 (A100) has VLEN = 1024 bits (`vlenb = 128`).
/// Newly spawned threads originate on Cluster 0. When migrating to Cluster 1 (`assigned_core >= 8`),
/// the thread moves across VLEN domains.
/// Any function compiled with RVV enabled (`+v`) that allocates scalable vector registers or vector spill
/// slots adjusts its stack frame via `csrr vlenb; sub sp, sp, ...` in its prologue and `add sp, sp, ...`
/// in its epilogue. If the thread migrates inside such a function, the prologue executes with `vlenb = 32`
/// while the epilogue executes with `vlenb = 128`, shifting `sp` and corrupting the return address (`ra`).
///
/// To guarantee 100% stack frame integrity across the heterogeneous VLEN boundary:
/// 1. `raw_pin_thread` performs no heap allocations, uses no `cpu_set_t` array or vector types,
///    ensuring LLVM allocates a fixed, constant-size stack frame unaffected by `vlenb`.
/// 2. It registers with `/proc/set_ai_thread` if `assigned_core >= 8` (vendor HMP requirement).
/// 3. It executes `sched_setaffinity` and `sched_getaffinity` via raw syscalls.
/// 4. When it returns, the thread is permanently on `assigned_core`, so all subsequent function calls
///    have matching prologues and epilogues in that cluster's native VLEN domain.
#[inline(never)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unnecessary_wraps
)]
unsafe fn raw_pin_thread(assigned_core: usize) -> Result<u64, i32> {
    #[cfg(any(not(target_os = "linux"), miri))]
    {
        Ok(1u64 << assigned_core)
    }
    #[cfg(all(target_os = "linux", not(miri)))]
    // SAFETY: Raw Linux syscalls and libc file operations to pin the thread
    // without triggering scalable vector spill allocation in function frame.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid);
        if assigned_core >= 8 {
            let path = b"/proc/set_ai_thread\0";
            let fd = libc::open(path.as_ptr().cast::<libc::c_char>(), libc::O_WRONLY);
            if fd >= 0 {
                let mut buf = [0u8; 32];
                let mut val = tid;
                let mut tmp = [0u8; 32];
                let mut i = 0;
                if val == 0 {
                    buf[0] = b'0';
                    i = 1;
                } else {
                    while val > 0 {
                        tmp[i] = b'0' + (val % 10) as u8;
                        val /= 10;
                        i += 1;
                    }
                    for j in 0..i {
                        buf[j] = tmp[i - 1 - j];
                    }
                }
                buf[i] = b'\n';
                libc::write(fd, buf.as_ptr().cast::<libc::c_void>(), i + 1);
                libc::close(fd);
            }
        }

        let mask = 1u64 << assigned_core;
        let set_ret = libc::syscall(
            libc::SYS_sched_setaffinity,
            0i32,
            size_of::<u64>(),
            (&raw const mask).cast::<libc::c_void>(),
        ) as i32;
        if set_ret < 0 {
            let k3_mask = libc::getenv(c"K3_CPU_ONLINE_MASK".as_ptr());
            let k3_sim = libc::getenv(c"K3_SIMULATE_A100".as_ptr());
            if !k3_mask.is_null() || !k3_sim.is_null() {
                return Ok(1u64 << assigned_core);
            }
            return Err(set_ret);
        }

        let mut get_mask = 0u64;
        let get_ret = libc::syscall(
            libc::SYS_sched_getaffinity,
            0i32,
            size_of::<u64>(),
            (&raw mut get_mask).cast::<libc::c_void>(),
        ) as i32;
        if get_ret < 0 {
            let k3_mask = libc::getenv(c"K3_CPU_ONLINE_MASK".as_ptr());
            let k3_sim = libc::getenv(c"K3_SIMULATE_A100".as_ptr());
            if !k3_mask.is_null() || !k3_sim.is_null() {
                return Ok(1u64 << assigned_core);
            }
            return Err(get_ret);
        }

        Ok(get_mask)
    }
}

fn verify_pinned_thread(
    worker_index: usize,
    assigned_core: usize,
    requested_mask: u64,
    requested_threads: usize,
    pin_res: Result<u64, i32>,
) -> Result<WorkerAffinityStatus, K3Error> {
    let observed_mask = match pin_res {
        Ok(mask) => mask,
        Err(err) => {
            return Err(K3Error::AffinityError {
                requested_mask,
                requested_threads,
                reason: format!(
                    "sched_setaffinity/getaffinity failed for worker {worker_index} on core {assigned_core} (errno {err})"
                ),
            });
        }
    };

    let expected_mask = 1u64 << assigned_core;
    if observed_mask != expected_mask {
        return Err(K3Error::AffinityError {
            requested_mask,
            requested_threads,
            reason: format!(
                "worker {worker_index} sched_getaffinity mismatch: expected 0x{expected_mask:04X}, got 0x{observed_mask:04X}"
            ),
        });
    }

    // SAFETY: SYS_gettid is a read-only syscall taking no arguments.
    #[cfg(all(target_os = "linux", not(miri)))]
    let tid: i64 = unsafe { libc::syscall(libc::SYS_gettid) };
    #[cfg(any(not(target_os = "linux"), miri))]
    let tid: i64 = 0;
    let proc_cpus_allowed_list =
        read_proc_task_cpus_allowed_list(tid).unwrap_or_else(|| assigned_core.to_string());

    Ok(WorkerAffinityStatus {
        worker_index,
        tid,
        assigned_core,
        sched_affinity_mask: observed_mask,
        proc_cpus_allowed_list,
    })
}

#[allow(dead_code)]
fn pin_current_thread_and_verify(
    worker_index: usize,
    assigned_core: usize,
    requested_mask: u64,
    requested_threads: usize,
) -> Result<WorkerAffinityStatus, K3Error> {
    // SAFETY: `raw_pin_thread` uses only fixed-frame integer syscalls to pin the calling thread.
    let pin_res = unsafe { raw_pin_thread(assigned_core) };
    verify_pinned_thread(
        worker_index,
        assigned_core,
        requested_mask,
        requested_threads,
        pin_res,
    )
}

fn read_proc_task_cpus_allowed_list(tid: i64) -> Option<String> {
    if std::env::var_os("K3_CPU_ONLINE_MASK").is_some()
        || std::env::var_os("K3_SIMULATE_A100").is_some()
    {
        return None;
    }
    let status_path = format!("/proc/self/task/{tid}/status");
    let content = fs::read_to_string(status_path).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

const fn select_dual_cluster_a100(
    policy: ClusterPhasePolicy,
    phase: crate::engine::EnginePhase,
    a100_has_ime: bool,
) -> bool {
    use crate::engine::EnginePhase;

    match policy {
        ClusterPhasePolicy::A100Only
        | ClusterPhasePolicy::A100PrefillA100Decode
        | ClusterPhasePolicy::RowSplit => true,
        ClusterPhasePolicy::X100Only => false,
        ClusterPhasePolicy::A100PrefillX100Decode => matches!(phase, EnginePhase::Prefill),
        ClusterPhasePolicy::X100PrefillA100Decode => !matches!(phase, EnginePhase::Prefill),
        ClusterPhasePolicy::Auto => a100_has_ime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_select_dual_cluster_a100_gates_auto_decode_on_ime() {
        use crate::engine::EnginePhase;

        for phase in [EnginePhase::Prefill, EnginePhase::Decode] {
            assert!(!select_dual_cluster_a100(
                ClusterPhasePolicy::Auto,
                phase,
                false
            ));
            assert!(select_dual_cluster_a100(
                ClusterPhasePolicy::Auto,
                phase,
                true
            ));
        }
        assert!(select_dual_cluster_a100(
            ClusterPhasePolicy::X100PrefillA100Decode,
            EnginePhase::Decode,
            false
        ));
        assert!(!select_dual_cluster_a100(
            ClusterPhasePolicy::X100PrefillA100Decode,
            EnginePhase::Prefill,
            true
        ));
        assert!(select_dual_cluster_a100(
            ClusterPhasePolicy::A100PrefillX100Decode,
            EnginePhase::Prefill,
            false
        ));
        assert!(!select_dual_cluster_a100(
            ClusterPhasePolicy::A100PrefillX100Decode,
            EnginePhase::Decode,
            true
        ));
    }

    #[test]
    #[cfg(not(miri))]
    fn test_dual_pool_install_for_phase_auto_decode_without_ime_routes_to_x100() {
        use crate::engine::EnginePhase;

        let x100_aff = ResolvedAffinity {
            cluster: Some(ClusterPreset::X100),
            selection: Some(ClusterSelection::Single(ClusterPreset::X100)),
            vlen_domain: VlenDomain::Zvl256b,
            effective_mask: 1,
            threads: 1,
            assigned_cores: vec![0],
            x100_cores: vec![0],
            a100_cores: Vec::new(),
        };
        let a100_aff = ResolvedAffinity {
            cluster: Some(ClusterPreset::A100),
            selection: Some(ClusterSelection::Single(ClusterPreset::A100)),
            vlen_domain: VlenDomain::Zvl1024b,
            effective_mask: 1,
            threads: 1,
            assigned_cores: vec![0],
            x100_cores: Vec::new(),
            a100_cores: vec![0],
        };
        let dual_aff = ResolvedAffinity {
            cluster: None,
            selection: Some(ClusterSelection::Both {
                split: SplitStrategy::Phase,
            }),
            vlen_domain: VlenDomain::Heterogeneous,
            effective_mask: 1,
            threads: 2,
            assigned_cores: vec![0, 0],
            x100_cores: vec![0],
            a100_cores: vec![0],
        };
        let mut dual = WorkerPool::new_with_profile(dual_aff, CoordinationProfile::Rayon)
            .expect("spawn dual pool");
        let x100_pool = WorkerPool::new_with_profile(x100_aff, CoordinationProfile::Rayon)
            .expect("spawn single-thread x100 pool");
        let a100_pool = WorkerPool::new_with_profile(a100_aff, CoordinationProfile::Rayon)
            .expect("spawn single-thread a100 pool");
        dual.backend = PoolBackend::Dual {
            x100_pool: Box::new(x100_pool),
            a100_pool: Box::new(a100_pool),
        };

        let x100_tid = dual
            .x100_pool()
            .expect("x100 subpool")
            .install(|| thread::current().id());
        let a100_tid = dual
            .a100_pool()
            .expect("a100 subpool")
            .install(|| thread::current().id());
        assert_ne!(x100_tid, a100_tid);
        assert!(!has_a100_ime_for_mask(
            dual.a100_pool().expect("a100").affinity().effective_mask
        ));

        let prefill_tid =
            dual.install_for_phase(EnginePhase::Prefill, ClusterPhasePolicy::Auto, || {
                thread::current().id()
            });
        let decode_tid =
            dual.install_for_phase(EnginePhase::Decode, ClusterPhasePolicy::Auto, || {
                thread::current().id()
            });
        let explicit_decode_tid = dual.install_for_phase(
            EnginePhase::Decode,
            ClusterPhasePolicy::X100PrefillA100Decode,
            || thread::current().id(),
        );

        assert_eq!(prefill_tid, x100_tid);
        assert_eq!(decode_tid, x100_tid);
        assert_eq!(explicit_decode_tid, a100_tid);
    }

    #[test]
    fn test_cluster_preset_resolution_x100_and_a100_us01_5_1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. --cluster x100 defaults to 6 compute threads on cores 0..5 (ADR-007)
        let x100 = resolve_affinity_with(Some("x100"), None, None, &k3_topo)
            .expect("resolve x100 default");
        assert_eq!(x100.cluster, Some(ClusterPreset::X100));
        assert_eq!(x100.vlen_domain, VlenDomain::Zvl256b);
        assert_eq!(x100.threads, 6);
        assert_eq!(x100.assigned_cores, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(x100.effective_mask, 0x003F);

        // 2. --cluster a100 defaults to 8 compute threads on cores 8..15 (0xFF00, zvl1024b)
        let a100 = resolve_affinity_with(Some("a100"), None, None, &k3_topo)
            .expect("resolve a100 default");
        assert_eq!(a100.cluster, Some(ClusterPreset::A100));
        assert_eq!(a100.vlen_domain, VlenDomain::Zvl1024b);
        assert_eq!(a100.threads, 8);
        assert_eq!(a100.assigned_cores, vec![8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(a100.effective_mask, 0xFF00);

        // 3. Explicit --cores 0x0F00 --threads 4 on A100
        let custom_a100 = resolve_affinity_with(Some("a100"), Some(0x0F00), Some(4), &k3_topo)
            .expect("resolve custom a100 mask");
        assert_eq!(custom_a100.assigned_cores, vec![8, 9, 10, 11]);
        assert_eq!(custom_a100.effective_mask, 0x0F00);
    }

    #[test]
    fn test_invalid_affinity_combinations_rejected_us01_5_1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. Mixing X100 (0x00FF) and A100 (0xFF00) VLEN domains without split policy is rejected
        let cross_vlen = resolve_affinity_with(None, Some(0x0101), Some(2), &k3_topo)
            .expect_err("must reject cross-cluster VLEN mixing");
        assert_eq!(cross_vlen.diagnostic_code(), "affinity_error");
        assert!(cross_vlen.to_string().contains("mixes asymmetric X100"));

        // 2. Threads exceeding cores in mask is rejected
        let too_many_threads = resolve_affinity_with(None, Some(0x000F), Some(6), &k3_topo)
            .expect_err("must reject threads > mask cores");
        assert_eq!(too_many_threads.diagnostic_code(), "affinity_error");
        assert!(
            too_many_threads
                .to_string()
                .contains("exceeds 4 available cores")
        );

        // 3. --cluster x100 with > 6 threads without explicit --cores override is rejected per ADR-007
        let x100_cap = resolve_affinity_with(Some("x100"), None, Some(8), &k3_topo)
            .expect_err("must enforce ADR-007 6-thread cap on x100 preset");
        assert_eq!(x100_cap.diagnostic_code(), "affinity_error");
        assert!(x100_cap.to_string().contains("ADR-007"));

        // 4. --cluster x100 with A100 core mask is rejected
        let mismatch = resolve_affinity_with(Some("x100"), Some(0xFF00), Some(4), &k3_topo)
            .expect_err("must reject A100 mask on x100 preset");
        assert_eq!(mismatch.diagnostic_code(), "affinity_error");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_worker_pool_pinning_and_proc_status_verification_us01_5_2() {
        let online_mask = SystemCpuTopology::new()
            .online_cpu_mask()
            .expect("read online CPUs");
        let x100_online = (online_mask & X100_CLUSTER_MASK).count_ones() as usize;
        let target_threads = x100_online.clamp(1, 2);

        let resolved = resolve_affinity(
            None,
            Some(online_mask & X100_CLUSTER_MASK),
            Some(target_threads),
        )
        .expect("resolve host online cores");

        let pool = WorkerPool::new(resolved.clone()).expect("spawn pinned worker pool");
        assert_eq!(pool.thread_count(), target_threads);

        let statuses = pool.worker_statuses();
        assert_eq!(statuses.len(), target_threads);
        for (i, status) in statuses.iter().enumerate() {
            assert_eq!(status.worker_index, i);
            assert_eq!(status.assigned_core, resolved.assigned_cores[i]);
            assert_eq!(status.sched_affinity_mask, 1u64 << status.assigned_core);
            assert_eq!(
                status.proc_cpus_allowed_list,
                status.assigned_core.to_string()
            );
        }

        // Execute parallel work across pinned workers and verify live affinity during/after execution
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        pool.execute(move |worker_idx, total| {
            assert!(worker_idx < total);
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), target_threads);

        let live = pool
            .verify_live_affinities()
            .expect("live affinity check succeeds");
        assert_eq!(live.len(), target_threads);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_worker_pool_on_a100() {
        let topo = SystemCpuTopology::new();
        let Ok(online_mask) = topo.online_cpu_mask() else {
            return; // gracefully skip if unable to inspect topology
        };

        // Guard against non-K3 CI hosts (e.g. 4-core / 8-core x86_64 runners) where cores 8..15 (0xFF00)
        // are not online. Only execute pool affinity when A100 cluster cores are present.
        if (online_mask & A100_CLUSTER_MASK) == 0 {
            eprintln!(
                "Skipping test_worker_pool_on_a100: no online cores in A100 cluster mask 0x{A100_CLUSTER_MASK:04X} (host online mask: 0x{online_mask:04X})"
            );
            return;
        }

        let a100 = resolve_affinity_with(Some("a100"), None, Some(1), &topo).unwrap();
        let pool = WorkerPool::new(a100).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        pool.execute(move |_worker_idx, _total| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_worker_wait_slot_alignment_and_stride_adr006() {
        assert_eq!(align_of::<WorkerWaitSlot>(), 64);
        assert_eq!(size_of::<WorkerWaitSlot>(), 64);

        let slots: Vec<WorkerWaitSlot> = (0..8).map(|_| WorkerWaitSlot::new()).collect();
        for i in 0..7 {
            let p1 = (&raw const slots[i]) as usize;
            let p2 = (&raw const slots[i + 1]) as usize;
            assert_eq!(p1 % 64, 0, "slot {i} must be 64-byte aligned");
            assert_eq!(p2 - p1, 64, "stride between slots must equal 64 bytes");
        }

        // Test recording and reset
        let slot = WorkerWaitSlot::new();
        assert_eq!(slot.wait_nanos(), 0);
        assert_eq!(slot.barrier_count(), 0);
        assert!((slot.wait_ms() - 0.0).abs() < f64::EPSILON);

        slot.record_wait(Duration::from_millis(5));
        assert_eq!(slot.barrier_count(), 1);
        assert!(slot.wait_nanos() >= 5_000_000);
        assert!(slot.wait_ms() >= 4.9);

        slot.record_wait_nanos(2_000_000);
        assert_eq!(slot.barrier_count(), 2);
        assert!(slot.wait_nanos() >= 7_000_000);

        slot.reset();
        assert_eq!(slot.wait_nanos(), 0);
        assert_eq!(slot.barrier_count(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_spin_barrier_synchronization_and_timing() {
        let barrier = Arc::new(SpinBarrier::new(4));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&shutdown);
            handles.push(thread::spawn(move || {
                let mut local_sense = false;
                let mut total_wait = Duration::ZERO;
                for _ in 0..10 {
                    total_wait += b.wait(&mut local_sense, &s);
                }
                total_wait
            }));
        }

        for h in handles {
            let _ = h.join().expect("barrier thread completes");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_futex_barrier_synchronization_and_timing() {
        let barrier = Arc::new(FutexBarrier::new(4));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&shutdown);
            handles.push(thread::spawn(move || {
                let mut total_wait = Duration::ZERO;
                for _ in 0..10 {
                    total_wait += b.wait(&s);
                }
                total_wait
            }));
        }

        for h in handles {
            let _ = h.join().expect("futex barrier thread completes");
        }
    }

    #[test]
    fn test_dual_cluster_split_resolution_us03_m1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. Default dual-cluster resolution: 14 threads (6 on X100, 8 on A100)
        let resolved =
            resolve_affinity_split_with(Some("both"), Some("phase"), None, None, &k3_topo)
                .expect("resolve dual-cluster phase split");

        assert!(resolved.is_dual_cluster());
        assert_eq!(resolved.split_strategy(), Some(SplitStrategy::Phase));
        assert_eq!(resolved.vlen_domain, VlenDomain::Heterogeneous);
        assert_eq!(resolved.threads, 14);
        assert_eq!(resolved.x100_cores, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(resolved.a100_cores, vec![8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(
            resolved.assigned_cores,
            vec![0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        assert_eq!(resolved.effective_mask, 0xFF3F);

        // 2. Op and Row split strategies
        let op_res = resolve_affinity_split_with(Some("both"), Some("op"), None, None, &k3_topo)
            .expect("resolve dual-cluster op split");
        assert_eq!(op_res.split_strategy(), Some(SplitStrategy::Op));

        let row_res = resolve_affinity_split_with(Some("both"), Some("row"), None, None, &k3_topo)
            .expect("resolve dual-cluster row split");
        assert_eq!(row_res.split_strategy(), Some(SplitStrategy::Row));

        // 3. Custom thread count balanced across clusters
        let balanced =
            resolve_affinity_split_with(Some("both"), Some("row"), None, Some(8), &k3_topo)
                .expect("resolve balanced 8 threads");
        assert_eq!(balanced.threads, 8);
        assert_eq!(balanced.x100_cores, vec![0, 1, 2, 3]);
        assert_eq!(balanced.a100_cores, vec![8, 9, 10, 11]);

        // 4. Custom core mask spanning both clusters
        let custom_mask = resolve_affinity_split_with(
            Some("both"),
            Some("phase"),
            Some(0x0303),
            Some(4),
            &k3_topo,
        )
        .expect("resolve custom dual mask");
        assert_eq!(custom_mask.threads, 4);
        assert_eq!(custom_mask.x100_cores, vec![0, 1]);
        assert_eq!(custom_mask.a100_cores, vec![8, 9]);
    }

    #[test]
    fn test_invalid_dual_cluster_combinations_rejected_us03_m1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. --cluster both without --split is rejected
        let no_split = resolve_affinity_with(Some("both"), None, None, &k3_topo)
            .expect_err("must reject both without split");
        assert_eq!(no_split.diagnostic_code(), "affinity_error");
        assert!(
            no_split
                .to_string()
                .contains("requires an explicit --split")
        );

        // 2. --split with single cluster is rejected
        let split_x100 =
            resolve_affinity_split_with(Some("x100"), Some("phase"), None, None, &k3_topo)
                .expect_err("must reject split with x100");
        assert_eq!(split_x100.diagnostic_code(), "affinity_error");
        assert!(
            split_x100
                .to_string()
                .contains("only valid with '--cluster both'")
        );

        // 3. Unknown split strategy is rejected
        let bad_split =
            resolve_affinity_split_with(Some("both"), Some("invalid"), None, None, &k3_topo)
                .expect_err("must reject unknown split strategy");
        assert_eq!(bad_split.diagnostic_code(), "affinity_error");
        assert!(bad_split.to_string().contains("unknown split strategy"));

        // 4. Threads < 2 on dual-cluster is rejected
        let threads_1 =
            resolve_affinity_split_with(Some("both"), Some("row"), None, Some(1), &k3_topo)
                .expect_err("must reject 1 thread on dual cluster");
        assert_eq!(threads_1.diagnostic_code(), "affinity_error");
        assert!(threads_1.to_string().contains("at least 2 worker threads"));

        // 5. Explicit mask not covering both clusters with --cluster both is rejected
        let x100_only_mask = resolve_affinity_split_with(
            Some("both"),
            Some("phase"),
            Some(0x00FF),
            Some(2),
            &k3_topo,
        )
        .expect_err("must reject single-cluster mask with both");
        assert_eq!(x100_only_mask.diagnostic_code(), "affinity_error");
        assert!(
            x100_only_mask
                .to_string()
                .contains("does not cover both clusters")
        );
    }

    #[test]
    fn test_compute_pool_cluster_preset_conversions() {
        assert_eq!(
            ClusterPreset::from_compute_pool(ComputePool::X100),
            Some(ClusterPreset::X100)
        );
        assert_eq!(
            ClusterPreset::from_compute_pool(ComputePool::A100),
            Some(ClusterPreset::A100)
        );
        assert_eq!(ClusterPreset::from_compute_pool(ComputePool::Both), None);

        assert_eq!(
            ComputePool::X100.into_cluster_preset(),
            Some(ClusterPreset::X100)
        );
        assert_eq!(
            ComputePool::A100.into_cluster_preset(),
            Some(ClusterPreset::A100)
        );
        assert_eq!(ComputePool::Both.into_cluster_preset(), None);

        assert_eq!(
            ClusterPreset::try_from(ComputePool::X100).unwrap(),
            ClusterPreset::X100
        );
        assert_eq!(
            ClusterPreset::try_from(ComputePool::A100).unwrap(),
            ClusterPreset::A100
        );
        assert!(ClusterPreset::try_from(ComputePool::Both).is_err());

        assert_eq!(ClusterPreset::X100.to_compute_pool(), ComputePool::X100);
        assert_eq!(ClusterPreset::A100.to_compute_pool(), ComputePool::A100);
    }

    #[test]
    fn test_cluster_phase_policy_parse_display_and_serde() {
        assert_eq!(ClusterPhasePolicy::default(), ClusterPhasePolicy::Auto);
        let expected = [
            (ClusterPhasePolicy::Auto, "auto"),
            (
                ClusterPhasePolicy::X100PrefillA100Decode,
                "x100-prefill-a100-decode",
            ),
            (
                ClusterPhasePolicy::A100PrefillX100Decode,
                "a100-prefill-x100-decode",
            ),
            (
                ClusterPhasePolicy::A100PrefillA100Decode,
                "a100-prefill-a100-decode",
            ),
            (ClusterPhasePolicy::A100Only, "a100-only"),
            (ClusterPhasePolicy::X100Only, "x100-only"),
            (ClusterPhasePolicy::RowSplit, "row-split"),
        ];
        assert_eq!(ClusterPhasePolicy::all().len(), expected.len());

        for (variant, text) in expected {
            assert_eq!(variant.as_str(), text);
            assert_eq!(variant.to_string(), text);
            assert_eq!(ClusterPhasePolicy::parse(text).unwrap(), variant);
            assert_eq!(
                text.parse::<ClusterPhasePolicy>().unwrap(),
                variant,
                "FromStr roundtrip failed for {text}"
            );
            let json = serde_json::to_string(&variant).expect("serialize policy");
            assert_eq!(json, format!("\"{text}\""));
            let roundtrip: ClusterPhasePolicy =
                serde_json::from_str(&json).expect("deserialize policy");
            assert_eq!(roundtrip, variant);
        }

        let err = ClusterPhasePolicy::parse("bogus-policy").expect_err("reject invalid policy");
        assert_eq!(err.diagnostic_code(), "affinity_error");
    }

    #[test]
    fn test_select_phase_policy_topology_and_architecture_matrix() {
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let llama_bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let llama_gguf = GgufFile::parse(&llama_bytes).expect("parse llama gguf");

        let bitnet_bytes = create_synthetic_gguf_v3(
            "bitnet-b1.58",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let bitnet_gguf = GgufFile::parse(&bitnet_bytes).expect("parse bitnet gguf");

        // 1. Dual-cluster SpacemiT K3 (0xFFFF) with A100 IME online routes dense/MoE
        // models to A100Only (A100 prefill + A100 decode) regardless of prompt length >= 8 or < 8.
        assert_eq!(PHASE_SPLIT_MIN_PROMPT_TOKENS, 8);
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 8, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_for_arch_with_ime("qwen2", 32, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_for_arch_with_ime("mixtral", 16, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 7, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 1, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(bitnet_gguf.metadata(), 64, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_for_arch_with_ime("bitnet", 128, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );

        // 2. Dual-cluster (0xFFFF) when A100 IME is unavailable falls back to X100Only,
        // and guardrail architectures always remain pinned to X100Only.
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 8, 0xFFFF, false),
            ClusterPhasePolicy::X100Only
        );
        assert_eq!(
            select_phase_policy_for_arch_with_ime("qwen2", 32, 0xFFFF, false),
            ClusterPhasePolicy::X100Only
        );
        for guardrail_arch in ["deberta-v3", "shieldgemma", "granite", "llama-guard"] {
            assert_eq!(
                select_phase_policy_for_arch_with_ime(guardrail_arch, 32, 0xFFFF, true),
                ClusterPhasePolicy::X100Only,
                "guardrail architecture {guardrail_arch} must stay pinned to X100Only on 0xFFFF"
            );
            assert_eq!(
                select_phase_policy_for_arch_with_ime(guardrail_arch, 32, 0xFFFF, false),
                ClusterPhasePolicy::X100Only,
                "guardrail architecture {guardrail_arch} must stay pinned to X100Only without IME"
            );
        }

        // 3. Single-cluster fallbacks (CI / symmetric hosts with 0x00FF / 0x000F, or A100-only 0xFF00)
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 32, 0x00FF, true),
            ClusterPhasePolicy::X100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(bitnet_gguf.metadata(), 32, 0x000F, true),
            ClusterPhasePolicy::X100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 32, 0xFF00, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            select_phase_policy_with_ime(llama_gguf.metadata(), 32, 0xFF00, false),
            ClusterPhasePolicy::A100Only
        );
    }

    #[test]
    fn test_resolve_phase_policy_with_ime_explicit_and_auto() {
        assert_eq!(
            resolve_phase_policy_with_ime(ClusterPhasePolicy::RowSplit, "llama", 32, 0x00FF, false),
            ClusterPhasePolicy::RowSplit
        );
        assert_eq!(
            resolve_phase_policy_with_ime(
                ClusterPhasePolicy::A100PrefillX100Decode,
                "llama",
                32,
                0xFFFF,
                true
            ),
            ClusterPhasePolicy::A100PrefillX100Decode
        );
        assert_eq!(
            resolve_phase_policy_with_ime(
                ClusterPhasePolicy::A100PrefillA100Decode,
                "llama",
                32,
                0xFFFF,
                true
            ),
            ClusterPhasePolicy::A100PrefillA100Decode
        );
        assert_eq!(
            resolve_phase_policy_with_ime(ClusterPhasePolicy::Auto, "llama", 32, 0xFFFF, true),
            ClusterPhasePolicy::A100Only
        );
        assert_eq!(
            resolve_phase_policy_with_ime(ClusterPhasePolicy::Auto, "llama", 32, 0xFFFF, false),
            ClusterPhasePolicy::X100Only
        );
    }

    fn verify_dual_pool_phase_cluster_isolation(dual_pool: &WorkerPool) {
        use crate::engine::EnginePhase;
        assert!(dual_pool.is_dual());

        let a100_prefill_core = dual_pool
            .install_for_phase(
                EnginePhase::Prefill,
                ClusterPhasePolicy::A100PrefillX100Decode,
                get_current_thread_core,
            )
            .expect("A100PrefillX100Decode prefill thread core");
        assert!(
            (8..16).contains(&a100_prefill_core),
            "A100PrefillX100Decode Prefill must execute on A100 (8..15), got {a100_prefill_core}"
        );

        let x100_decode_core = dual_pool
            .install_for_phase(
                EnginePhase::Decode,
                ClusterPhasePolicy::A100PrefillX100Decode,
                get_current_thread_core,
            )
            .expect("A100PrefillX100Decode decode thread core");
        assert!(
            x100_decode_core < 8,
            "A100PrefillX100Decode Decode must execute on X100 (0..7), got {x100_decode_core}"
        );

        let a100_both_prefill = dual_pool
            .install_for_phase(
                EnginePhase::Prefill,
                ClusterPhasePolicy::A100PrefillA100Decode,
                get_current_thread_core,
            )
            .expect("A100PrefillA100Decode prefill core");
        assert!((8..16).contains(&a100_both_prefill));

        let a100_both_decode = dual_pool
            .install_for_phase(
                EnginePhase::Decode,
                ClusterPhasePolicy::A100PrefillA100Decode,
                get_current_thread_core,
            )
            .expect("A100PrefillA100Decode decode core");
        assert!((8..16).contains(&a100_both_decode));

        let x100_prefill_core = dual_pool
            .install_for_phase(
                EnginePhase::Prefill,
                ClusterPhasePolicy::X100PrefillA100Decode,
                get_current_thread_core,
            )
            .expect("X100PrefillA100Decode prefill thread core");
        assert!(
            x100_prefill_core < 8,
            "X100PrefillA100Decode Prefill must execute on X100 (0..7), got core {x100_prefill_core}"
        );

        let a100_decode_core = dual_pool
            .install_for_phase(
                EnginePhase::Decode,
                ClusterPhasePolicy::X100PrefillA100Decode,
                get_current_thread_core,
            )
            .expect("X100PrefillA100Decode decode thread core");
        assert!(
            (8..16).contains(&a100_decode_core),
            "X100PrefillA100Decode Decode must execute on A100 (8..15), got core {a100_decode_core}"
        );

        let row_split_prefill = dual_pool
            .install_for_phase(
                EnginePhase::Prefill,
                ClusterPhasePolicy::RowSplit,
                get_current_thread_core,
            )
            .expect("RowSplit prefill thread core");
        assert!(
            (8..16).contains(&row_split_prefill),
            "RowSplit Prefill must execute on A100 (8..15), got core {row_split_prefill}"
        );

        let pinned_matrix_core = dual_pool
            .install_for_phase(
                EnginePhase::Prefill,
                ClusterPhasePolicy::A100Only,
                get_current_thread_core,
            )
            .expect("A100Only core");
        assert!((8..16).contains(&pinned_matrix_core));

        let pinned_ooo_core = dual_pool
            .install_for_phase(
                EnginePhase::Decode,
                ClusterPhasePolicy::X100Only,
                get_current_thread_core,
            )
            .expect("X100Only core");
        assert!(pinned_ooo_core < 8);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_worker_pool_install_for_phase_cluster_isolation() {
        use crate::engine::EnginePhase;
        use jero_core::{GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3};

        let llama_bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let llama_gguf = GgufFile::parse(&llama_bytes).expect("parse llama gguf");
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. Always verify single-cluster X100Only pinned Rayon execution on core 0
        let x100_pool = WorkerPool::for_phase_policy(
            ClusterPhasePolicy::X100Only,
            llama_gguf.metadata(),
            8,
            Some(0x0001),
            Some(1),
            CoordinationProfile::Rayon,
        )
        .expect("spawn X100Only pool on core 0");
        assert_eq!(x100_pool.phase_policy(), Some(ClusterPhasePolicy::X100Only));
        let core_in_x100 = x100_pool.install_for_phase(
            EnginePhase::Prefill,
            ClusterPhasePolicy::X100Only,
            get_current_thread_core,
        );
        assert_eq!(core_in_x100, Some(0));

        // 2. Verify Auto policy succeeds on simulated 0xFFFF topology (spawning A100Only on IME
        // K3 systems or cleanly falling back to X100Only on <16-core / non-IME CI hosts).
        let auto_pool = WorkerPool::for_phase_policy_with(
            ClusterPhasePolicy::Auto,
            llama_gguf.metadata(),
            8,
            None,
            Some(2),
            CoordinationProfile::Rayon,
            &k3_topo,
        )
        .expect("Auto phase policy must succeed on all hosts");
        assert!(matches!(
            auto_pool.phase_policy(),
            Some(ClusterPhasePolicy::A100Only | ClusterPhasePolicy::X100Only)
        ));

        // 3. When dual-cluster pinning is permitted by the kernel (16-core K3 or non-Linux),
        // verify explicit A100PrefillX100Decode, A100PrefillA100Decode, X100PrefillA100Decode,
        // and RowSplit phase-to-cluster routing.
        if let Ok(dual_pool) = WorkerPool::for_phase_policy_with(
            ClusterPhasePolicy::A100PrefillX100Decode,
            llama_gguf.metadata(),
            8,
            None,
            Some(4),
            CoordinationProfile::Rayon,
            &k3_topo,
        ) {
            verify_dual_pool_phase_cluster_isolation(&dual_pool);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dedicated_threads_single_pool_no_extra_threads_and_disjoint_fanout() {
        use jero_core::{
            CoordinationProfile, GgufFile, QuantType, test_helpers::create_synthetic_gguf_v3,
        };
        use std::cell::Cell;
        use std::collections::HashSet;
        use std::rc::Rc;
        use std::sync::Mutex;

        let bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        for profile in [
            CoordinationProfile::Spin,
            CoordinationProfile::Futex,
            CoordinationProfile::Rayon,
        ] {
            let pool = WorkerPool::for_phase_policy(
                ClusterPhasePolicy::X100Only,
                gguf.metadata(),
                8,
                None,
                Some(4),
                profile,
            )
            .expect("create worker pool");
            assert!(pool.thread_count() <= 4);

            let seen_tids: Mutex<HashSet<thread::ThreadId>> = Mutex::new(HashSet::new());
            let record_tid = || {
                seen_tids
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(thread::current().id());
            };

            // Verify `install` + nested `par_chunks_mut` / `par_for_each_index` / `dispatch_indexed`
            // / `execute_scoped` never spawns an extra thread pool or leaks outside N workers.
            let mut buf = vec![0_usize; 257];
            pool.install(|| {
                assert!(WorkerPool::with_current(|_| ()).is_some());
                record_tid();

                pool.par_chunks_mut(&mut buf, 32, |chunk_idx, chunk| {
                    record_tid();
                    for (offset, slot) in chunk.iter_mut().enumerate() {
                        *slot = chunk_idx * 32 + offset + 1;
                    }
                });

                pool.dispatch_indexed(64, |range, _worker_idx| {
                    record_tid();
                    assert!(!range.is_empty());
                });

                let sum_check = AtomicUsize::new(0);
                pool.par_for_each_index(64, |idx| {
                    record_tid();
                    sum_check.fetch_add(idx + 1, Ordering::Relaxed);
                });
                assert_eq!(sum_check.load(Ordering::Acquire), (64 * 65) / 2);

                pool.execute_scoped(&|_worker_idx, _num_workers| {
                    record_tid();
                });
            });

            for (i, &val) in buf.iter().enumerate() {
                assert_eq!(val, i + 1, "par_chunks_mut disjoint write mismatch at {i}");
            }

            let err = pool
                .try_par_chunks_mut(
                    &mut buf,
                    16,
                    |chunk_idx, _chunk| -> Result<(), &'static str> {
                        if chunk_idx == 3 { Err("boom") } else { Ok(()) }
                    },
                )
                .expect_err("try_par_chunks_mut must surface worker error");
            assert_eq!(err, "boom");

            let distinct_count = seen_tids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .len();
            assert!(
                distinct_count <= pool.thread_count(),
                "{profile:?} observed {distinct_count} worker threads for pool of size {}",
                pool.thread_count()
            );

            // Verify `with_phase_session` retains `!Send` state on caller thread across iterations.
            let local_counter = Rc::new(Cell::new(0_usize));
            let local_ref = Rc::clone(&local_counter);
            let total = pool.with_phase_session(
                crate::engine::EnginePhase::Decode,
                ClusterPhasePolicy::Auto,
                |session| {
                    for step in 0..8 {
                        let out = session.run(|| step * 2);
                        assert_eq!(out, step * 2);
                        local_ref.set(local_ref.get() + 1);
                    }
                    local_ref.get()
                },
            );
            assert_eq!(total, 8);
            assert_eq!(local_counter.get(), 8);
        }
    }
}
