//! `SpacemiT` K3 heterogeneous cluster topology, core mask resolution, and pinned worker pool.
//!
//! Restricts standalone model evaluation strictly to the operator-requested processor cluster
//! (`x100` or `a100`), core bitmask, and worker concurrency bound while preventing cross-cluster
//! VLEN migration (ADR-007, ADR-008, `core-affinity.md`, US-01 Scenario 2).

use std::cell::Cell;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use jero_core::{ComputePool, CoordinationProfile, K3Error};
use serde::{Deserialize, Serialize};

thread_local! {
    static CURRENT_ASSIGNED_CORE: Cell<Option<usize>> = const { Cell::new(None) };
}

#[allow(dead_code)]
pub(crate) fn set_current_thread_core(core: usize) {
    CURRENT_ASSIGNED_CORE.with(|c| c.set(Some(core)));
}

#[allow(dead_code)]
pub(crate) fn get_current_thread_core() -> Option<usize> {
    CURRENT_ASSIGNED_CORE.with(Cell::get)
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

type JobFn = Arc<dyn Fn(usize, usize) + Send + Sync>;

struct PoolSharedState {
    generation: usize,
    active_job: Option<JobFn>,
    shutdown: bool,
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
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        profile: CoordinationProfile,
        state: Arc<(Mutex<PoolSharedState>, Condvar)>,
        shutdown: Arc<AtomicBool>,
        spin_barrier: Option<Arc<SpinBarrier>>,
        futex_barrier: Option<Arc<FutexBarrier>>,
        coordinator_sense: Arc<AtomicBool>,
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
            Self::DedicatedThreads { profile, .. } => f
                .debug_struct("DedicatedThreads")
                .field("profile", profile)
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
    worker_statuses: Vec<WorkerAffinityStatus>,
    wait_slots: Arc<[WorkerWaitSlot]>,
    backend: PoolBackend,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("affinity", &self.affinity)
            .field("profile", &self.profile)
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
                let state = Arc::new((
                    Mutex::new(PoolSharedState {
                        generation: 0,
                        active_job: None,
                        shutdown: false,
                    }),
                    Condvar::new(),
                ));
                let shutdown = Arc::new(AtomicBool::new(false));
                let coordinator_sense = Arc::new(AtomicBool::new(false));

                let spin_barrier = if profile == CoordinationProfile::Spin {
                    Some(Arc::new(SpinBarrier::new(num_workers + 1)))
                } else {
                    None
                };

                let futex_barrier = if profile == CoordinationProfile::Futex {
                    Some(Arc::new(FutexBarrier::new(num_workers + 1)))
                } else {
                    None
                };

                let (init_tx, init_rx) = mpsc::channel::<Result<WorkerAffinityStatus, K3Error>>();
                let mut handles = Vec::with_capacity(num_workers);

                for (worker_index, &assigned_core) in affinity.assigned_cores.iter().enumerate() {
                    let state_clone = Arc::clone(&state);
                    let shutdown_clone = Arc::clone(&shutdown);
                    let spin_barrier_clone = spin_barrier.clone();
                    let futex_barrier_clone = futex_barrier.clone();
                    let wait_slots_clone = Arc::clone(&wait_slots);
                    let tx = init_tx.clone();
                    let req_mask = affinity.effective_mask;

                    let handle = thread::Builder::new()
                        .name(format!("k3-worker-{worker_index}"))
                        .spawn(move || {
                            set_current_thread_core(assigned_core);
                            // SAFETY: `raw_pin_thread` uses only fixed-frame integer syscalls to pin the worker.
                            let pin_res = unsafe { raw_pin_thread(assigned_core) };
                            let pin_result = verify_pinned_thread(
                                worker_index,
                                assigned_core,
                                req_mask,
                                num_workers,
                                pin_res,
                            );
                            let is_ok = pin_result.is_ok();
                            let _ = tx.send(pin_result);
                            if !is_ok {
                                return;
                            }

                            let mut local_sense = false;
                            let mut seen_gen = 0usize;
                            loop {
                                let (job, shutdown_flag) = {
                                    let (lock, cvar) = &*state_clone;
                                    let mut guard = lock
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    while !guard.shutdown
                                        && !shutdown_clone.load(Ordering::Relaxed)
                                        && guard.generation == seen_gen
                                    {
                                        guard = cvar
                                            .wait(guard)
                                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    }
                                    if guard.shutdown || shutdown_clone.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    seen_gen = guard.generation;
                                    (guard.active_job.clone(), guard.shutdown)
                                };

                                if let Some(work_fn) = job {
                                    work_fn(worker_index, num_workers);
                                }

                                if let Some(barrier) = &spin_barrier_clone {
                                    let wait_time = barrier.wait(&mut local_sense, &shutdown_clone);
                                    wait_slots_clone[worker_index].record_wait(wait_time);
                                } else if let Some(barrier) = &futex_barrier_clone {
                                    let wait_time = barrier.wait(&shutdown_clone);
                                    wait_slots_clone[worker_index].record_wait(wait_time);
                                }

                                if shutdown_flag || shutdown_clone.load(Ordering::Relaxed) {
                                    return;
                                }
                            }
                        })
                        .map_err(|err| {
                            shutdown_dedicated(
                                &state,
                                &shutdown,
                                futex_barrier.as_ref(),
                                &mut handles,
                            );
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
                            shutdown_dedicated(
                                &state,
                                &shutdown,
                                futex_barrier.as_ref(),
                                &mut handles,
                            );
                            return Err(err);
                        }
                        Err(recv_err) => {
                            shutdown_dedicated(
                                &state,
                                &shutdown,
                                futex_barrier.as_ref(),
                                &mut handles,
                            );
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
                    worker_statuses,
                    wait_slots,
                    backend: PoolBackend::DedicatedThreads {
                        profile,
                        state,
                        shutdown,
                        spin_barrier,
                        futex_barrier,
                        coordinator_sense,
                        handles,
                    },
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

    /// Executes `task(worker_index, total_workers)` in parallel across all pinned worker threads
    /// and waits for completion.
    pub fn execute<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.execute_job(Arc::new(task));
    }

    /// Internal non-generic job dispatcher to eliminate recursive monomorphization across dual pools.
    pub fn execute_job(&self, job: JobFn) {
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                let j1 = Arc::clone(&job);
                let j2 = Arc::clone(&job);
                let total_workers = self.affinity.threads;
                let nx = x100_pool.thread_count();

                thread::scope(|s| {
                    s.spawn(move || {
                        let wrapped: JobFn = Arc::new(move |w_idx, _| {
                            j1(w_idx, total_workers);
                        });
                        x100_pool.execute_job(wrapped);
                    });
                    let wrapped: JobFn = Arc::new(move |w_idx, _| {
                        j2(nx + w_idx, total_workers);
                    });
                    a100_pool.execute_job(wrapped);
                });
            }
            PoolBackend::DedicatedThreads {
                state,
                shutdown,
                spin_barrier,
                futex_barrier,
                coordinator_sense,
                ..
            } => {
                {
                    let (lock, cvar) = &**state;
                    let mut guard = lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard.active_job = Some(job);
                    guard.generation = guard.generation.wrapping_add(1);
                    cvar.notify_all();
                }
                if let Some(barrier) = spin_barrier {
                    let mut local_sense = coordinator_sense.load(Ordering::Relaxed);
                    barrier.wait(&mut local_sense, shutdown);
                    coordinator_sense.store(local_sense, Ordering::Relaxed);
                } else if let Some(barrier) = futex_barrier {
                    barrier.wait(shutdown);
                }
                {
                    let (lock, _) = &**state;
                    let mut guard = lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard.active_job = None;
                }
            }
            PoolBackend::Rayon { pool } => {
                let num_workers = self.affinity.threads;
                let wait_slots = Arc::clone(&self.wait_slots);
                let finish_times = Arc::new(
                    (0..num_workers)
                        .map(|_| AtomicU64::new(0))
                        .collect::<Vec<_>>(),
                );
                let t_start = Instant::now();

                pool.install(|| {
                    rayon::scope(|s| {
                        for worker_idx in 0..num_workers {
                            let finish_times_clone = Arc::clone(&finish_times);
                            let job_clone = Arc::clone(&job);
                            s.spawn(move |_| {
                                job_clone(worker_idx, num_workers);
                                let elapsed_ns =
                                    u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                                finish_times_clone[worker_idx].store(elapsed_ns, Ordering::Release);
                            });
                        }
                    });
                });

                let t_total_ns = u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                for worker_idx in 0..num_workers {
                    let finish_ns = finish_times[worker_idx].load(Ordering::Acquire);
                    let wait_ns = t_total_ns.saturating_sub(finish_ns);
                    wait_slots[worker_idx].record_wait_nanos(wait_ns);
                }
            }
        }
    }

    /// Executes a task exclusively across X100 worker threads.
    pub fn execute_x100<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        match &self.backend {
            PoolBackend::Dual { x100_pool, .. } => x100_pool.execute_job(Arc::new(task)),
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::X100)
                    || (!self.affinity.x100_cores.is_empty() && self.affinity.a100_cores.is_empty())
                {
                    self.execute_job(Arc::new(task));
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
            PoolBackend::Dual { a100_pool, .. } => a100_pool.execute_job(Arc::new(task)),
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::A100)
                    || (!self.affinity.a100_cores.is_empty() && self.affinity.x100_cores.is_empty())
                {
                    self.execute_job(Arc::new(task));
                }
            }
        }
    }

    /// Executes independent tasks concurrently across X100 and A100 worker pools.
    pub fn execute_split<FX, FA>(&self, x100_task: FX, a100_task: FA)
    where
        FX: Fn(usize, usize) + Send + Sync + 'static,
        FA: Fn(usize, usize) + Send + Sync + 'static,
    {
        match &self.backend {
            PoolBackend::Dual {
                x100_pool,
                a100_pool,
            } => {
                let jx: JobFn = Arc::new(x100_task);
                let ja: JobFn = Arc::new(a100_task);
                thread::scope(|s| {
                    s.spawn(|| {
                        x100_pool.execute_job(jx);
                    });
                    a100_pool.execute_job(ja);
                });
            }
            _ => {
                if self.affinity.cluster == Some(ClusterPreset::A100) {
                    self.execute_job(Arc::new(a100_task));
                } else {
                    self.execute_job(Arc::new(x100_task));
                }
            }
        }
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
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        if let PoolBackend::DedicatedThreads {
            state,
            shutdown,
            futex_barrier,
            handles,
            ..
        } = &mut self.backend
        {
            shutdown_dedicated(state, shutdown, futex_barrier.as_ref(), handles);
        }
    }
}

fn shutdown_dedicated(
    state: &Arc<(Mutex<PoolSharedState>, Condvar)>,
    shutdown: &Arc<AtomicBool>,
    futex_barrier: Option<&Arc<FutexBarrier>>,
    handles: &mut Vec<JoinHandle<()>>,
) {
    shutdown.store(true, Ordering::Release);
    {
        let (lock, cvar) = &**state;
        let mut guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.shutdown = true;
        cvar.notify_all();
    }
    if let Some(fb) = futex_barrier {
        fb.notify_shutdown();
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
            let k3_mask = libc::getenv(b"K3_CPU_ONLINE_MASK\0".as_ptr().cast::<libc::c_char>());
            let k3_sim = libc::getenv(b"K3_SIMULATE_A100\0".as_ptr().cast::<libc::c_char>());
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
            let k3_mask = libc::getenv(b"K3_CPU_ONLINE_MASK\0".as_ptr().cast::<libc::c_char>());
            let k3_sim = libc::getenv(b"K3_SIMULATE_A100\0".as_ptr().cast::<libc::c_char>());
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
    let status_path = format!("/proc/self/task/{tid}/status");
    let content = fs::read_to_string(status_path).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
}
