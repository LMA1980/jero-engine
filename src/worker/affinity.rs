//! Hardware cluster affinity, capability probing, and conditional `/proc/set_ai_thread` registration.

#[cfg(all(target_os = "linux", not(miri)))]
use core::mem::size_of;
#[cfg(target_arch = "riscv64")]
use std::path::Path;

use super::error::WorkerError;

/// Core bitmask for Cluster 0 (X100 Out-of-Order cores 0–7, VLEN = 256 bits).
pub const X100_CLUSTER_MASK: u64 = 0x00FF;

/// Core bitmask for Cluster 1 (A100 In-Order compute cores 8–15, VLEN = 1024 bits, XSMTVDotII).
pub const A100_CLUSTER_MASK: u64 = 0xFF00;

/// Permitted X100 compute core bitmask (cores 0..=5).
pub const DEFAULT_X100_COMPUTE_CAP_MASK: u64 = 0x003F;

/// Dedicated control-plane core bitmask on X100 (cores 6..=7).
pub const DEFAULT_ACTIVE_CONTROL_MASK: u64 = 0x00C0;

/// Default maximum compute threads on X100 preserving control-plane reservation per ADR-007.
pub const X100_DEFAULT_MAX_COMPUTE_THREADS: usize = 6;

/// Default maximum compute threads on A100 across all dedicated compute cores.
pub const A100_DEFAULT_MAX_COMPUTE_THREADS: usize = 8;

/// Extracts the list of active core indices from a bitmask in ascending order.
#[must_use]
pub fn cores_from_mask(mask: u64) -> Vec<usize> {
    let mut cores = Vec::new();
    for i in 0..64 {
        if (mask & (1_u64 << i)) != 0 {
            cores.push(i);
        }
    }
    cores
}

/// Parses an affinity core index, cpuset range/list, or hex/decimal mask string into a single core ID and mask.
///
/// # Errors
/// Returns [`WorkerError::InvalidArguments`] if affinity string cannot be parsed or is out of bounds.
pub fn parse_affinity_spec(spec: &str) -> Result<(usize, u64), WorkerError> {
    let s = spec.trim();
    if s.is_empty() {
        return Err(WorkerError::InvalidArguments(
            "affinity spec cannot be empty".to_owned(),
        ));
    }
    if s.starts_with('-') {
        return Err(WorkerError::InvalidArguments(format!(
            "invalid affinity spec '{s}': negative values are not allowed"
        )));
    }

    // Try parsing as hex bitmask (e.g. "0xFF00", "0x00FF", "0x003F")
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let mask = u64::from_str_radix(hex, 16).map_err(|e| {
            WorkerError::InvalidArguments(format!("invalid hex affinity mask '{s}': {e}"))
        })?;
        if mask == 0 {
            return Err(WorkerError::InvalidArguments(
                "affinity mask cannot be 0".to_owned(),
            ));
        }
        let primary_core = mask.trailing_zeros() as usize;
        return Ok((primary_core, mask));
    }

    // Try parsing cpuset range or comma list (e.g. "0-5", "8-15", "0,1,2")
    if s.contains('-') || s.contains(',') {
        let mut mask = 0_u64;
        for token in s.split(',') {
            let t = token.trim();
            if t.is_empty() {
                return Err(WorkerError::InvalidArguments(format!(
                    "malformed affinity spec '{s}': empty entry between commas"
                )));
            }
            if let Some((start_str, end_str)) = t.split_once('-') {
                let start: usize = start_str.trim().parse().map_err(|e| {
                    WorkerError::InvalidArguments(format!(
                        "invalid range start '{start_str}' in '{s}': {e}"
                    ))
                })?;
                let end: usize = end_str.trim().parse().map_err(|e| {
                    WorkerError::InvalidArguments(format!(
                        "invalid range end '{end_str}' in '{s}': {e}"
                    ))
                })?;
                if start > end {
                    return Err(WorkerError::InvalidArguments(format!(
                        "inverted affinity range '{t}' in '{s}': start {start} > end {end}"
                    )));
                }
                if end >= 64 {
                    return Err(WorkerError::InvalidArguments(format!(
                        "core index {end} in '{t}' exceeds 63"
                    )));
                }
                for c in start..=end {
                    mask |= 1_u64 << c;
                }
            } else {
                let c: usize = t.parse().map_err(|e| {
                    WorkerError::InvalidArguments(format!("invalid core '{t}' in '{s}': {e}"))
                })?;
                if c >= 64 {
                    return Err(WorkerError::InvalidArguments(format!(
                        "core index {c} exceeds 63"
                    )));
                }
                mask |= 1_u64 << c;
            }
        }
        if mask == 0 {
            return Err(WorkerError::InvalidArguments(
                "affinity mask cannot be 0".to_owned(),
            ));
        }
        return Ok((mask.trailing_zeros() as usize, mask));
    }

    // Try parsing as single core number first (e.g. "8")
    if let Ok(core) = s.parse::<usize>() {
        if core >= 64 {
            return Err(WorkerError::InvalidArguments(format!(
                "core index {core} exceeds 63"
            )));
        }
        return Ok((core, 1_u64 << core));
    }

    // Decimal bitmask fallback
    let mask = s
        .parse::<u64>()
        .map_err(|e| WorkerError::InvalidArguments(format!("invalid affinity mask '{s}': {e}")))?;

    if mask == 0 {
        return Err(WorkerError::InvalidArguments(
            "affinity mask cannot be 0".to_owned(),
        ));
    }

    // Lowest set bit represents the primary core
    let primary_core = mask.trailing_zeros() as usize;
    Ok((primary_core, mask))
}

/// Verifies whether the host platform supports the requested cluster and affinity settings.
///
/// Returns `Ok(())` if supported.
/// Returns `Err(WorkerError::CapabilityMismatch)` (exit code 10) if requested cluster or core is unavailable.
///
/// # Errors
/// Returns [`WorkerError`] on capability mismatch.
pub fn verify_cluster_capabilities(
    cluster: &str,
    affinity_spec: Option<&str>,
) -> Result<(), WorkerError> {
    let sim = std::env::var("K3_SIMULATE_A100").ok();
    verify_cluster_capabilities_full(cluster, affinity_spec, None, sim.as_deref())
}

/// Verifies whether the host platform supports the requested cluster and affinity settings,
/// with an explicit simulation flag parameter.
///
/// # Errors
/// Returns [`WorkerError`] on capability mismatch.
pub fn verify_cluster_capabilities_with_sim(
    cluster: &str,
    affinity_spec: Option<&str>,
    sim: Option<&str>,
) -> Result<(), WorkerError> {
    verify_cluster_capabilities_full(cluster, affinity_spec, None, sim)
}

/// Verifies cluster, affinity mask, thread bounds, and hardware capabilities.
///
/// # Errors
/// Returns [`WorkerError::CapabilityMismatch`] on capacity or hardware violation.
#[allow(clippy::too_many_lines)]
pub fn verify_cluster_capabilities_full(
    cluster: &str,
    affinity_spec: Option<&str>,
    threads: Option<usize>,
    sim: Option<&str>,
) -> Result<(), WorkerError> {
    let cluster_lower = cluster.to_ascii_lowercase();
    let is_x100 = cluster_lower == "x100";
    let is_a100 = cluster_lower == "a100";

    if !is_x100 && !is_a100 {
        return Err(WorkerError::InvalidArguments(format!(
            "unknown cluster '{cluster}'; expected 'x100' or 'a100'"
        )));
    }

    let parsed_aff = if let Some(spec) = affinity_spec {
        Some(parse_affinity_spec(spec)?)
    } else {
        None
    };

    if is_a100 {
        if let Some((_core, mask)) = parsed_aff {
            // Check active control core encroachment (0x00C0)
            if (mask & DEFAULT_ACTIVE_CONTROL_MASK) != 0 {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'a100' encroaches on dedicated active control cores 0x{DEFAULT_ACTIVE_CONTROL_MASK:04X}"
                )));
            }
            // Check cores outside A100
            if (mask & !A100_CLUSTER_MASK) != 0 {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'a100' specifies cores outside A100 mask 0x{A100_CLUSTER_MASK:04X}"
                )));
            }
        }
        if let Some(t) = threads {
            if t > A100_DEFAULT_MAX_COMPUTE_THREADS {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'a100' requests {t} threads, exceeding A100 max compute limit {A100_DEFAULT_MAX_COMPUTE_THREADS}"
                )));
            }
        }
    } else {
        if let Some((_core, mask)) = parsed_aff {
            // Check active control core encroachment (0x00C0)
            if (mask & DEFAULT_ACTIVE_CONTROL_MASK) != 0 {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'x100' encroaches on dedicated active control cores 0x{DEFAULT_ACTIVE_CONTROL_MASK:04X} (intersection: 0x{:04X})",
                    mask & DEFAULT_ACTIVE_CONTROL_MASK
                )));
            }
            // Check X100 compute cap mask (0x003F)
            if (mask & !DEFAULT_X100_COMPUTE_CAP_MASK & X100_CLUSTER_MASK) != 0 {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'x100' exceeds X100 compute cap mask 0x{DEFAULT_X100_COMPUTE_CAP_MASK:04X}"
                )));
            }
            // Check cores outside X100
            if (mask & !X100_CLUSTER_MASK) != 0 {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'x100' specifies cores outside X100 cluster mask 0x{X100_CLUSTER_MASK:04X}"
                )));
            }
        }
        if let Some(t) = threads {
            if t > X100_DEFAULT_MAX_COMPUTE_THREADS {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "worker placement on 'x100' requests {t} threads, exceeding X100 compute cap of {X100_DEFAULT_MAX_COMPUTE_THREADS} threads"
                )));
            }
        }
    }

    if let Some((_core, mask)) = parsed_aff {
        if let Some(t) = threads {
            if t > mask.count_ones() as usize {
                return Err(WorkerError::CapabilityMismatch(format!(
                    "requested threads {t} exceeds {0} available cores in affinity mask 0x{mask:04X}",
                    mask.count_ones()
                )));
            }
        }
    }

    if !is_a100 {
        return Ok(());
    }

    // 1. Check test simulation environment override
    if sim == Some("1") {
        return Ok(());
    }

    // 2. Non-riscv64 platforms (macOS, x86_64, aarch64) cannot execute A100 kernels or matrix ops
    #[cfg(not(target_arch = "riscv64"))]
    {
        Err(WorkerError::CapabilityMismatch(format!(
            "host architecture is not riscv64; SpacemiT A100 compute cores (cpu8-15) are unavailable for cluster '{cluster}'"
        )))
    }

    // 3. On riscv64, probe device-tree / sysfs for physical A100 presence
    #[cfg(target_arch = "riscv64")]
    {
        let is_a100_present = Path::new("/sys/devices/system/cpu/cpu8/of_node/cpu-ai").exists()
            || std::fs::read("/sys/devices/system/cpu/cpu8/of_node/compatible")
                .map(|b| String::from_utf8_lossy(&b).contains("spacemit,a100"))
                .unwrap_or(false);

        if !is_a100_present {
            return Err(WorkerError::CapabilityMismatch(
                "host does not possess physical SpacemiT K3 A100 compute cores (cpu8-15)"
                    .to_owned(),
            ));
        }

        Ok(())
    }
}

/// Pins the current worker thread to the assigned physical core.
///
/// # Safety
/// Invokes low-level OS threading syscalls. Marked `#[inline(never)]` and keeps a scalar-only
/// stack frame to prevent scalable vector spill corruption during cross-cluster migrations.
#[inline(never)]
pub unsafe fn pin_worker_core(assigned_core: usize) -> Result<(), WorkerError> {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // SAFETY: Direct Linux syscall without vector registers in stack frame.
        unsafe {
            let tid = libc::syscall(libc::SYS_gettid);
            if assigned_core >= 8 && tid > 0 {
                let path = b"/proc/set_ai_thread\0";
                let fd = libc::open(path.as_ptr().cast::<libc::c_char>(), libc::O_WRONLY);
                if fd >= 0 {
                    let mut buf = [0u8; 32];
                    let mut val = tid;
                    let mut tmp = [0u8; 32];
                    let mut i = 0;
                    while val > 0 {
                        tmp[i] = b'0' + (val % 10) as u8;
                        val /= 10;
                        i += 1;
                    }
                    for j in 0..i {
                        buf[j] = tmp[i - 1 - j];
                    }
                    buf[i] = b'\n';
                    let _ = libc::write(fd, buf.as_ptr().cast::<libc::c_void>(), i + 1);
                    let _ = libc::close(fd);
                }
            }

            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(assigned_core, &mut set);
            let ret = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(WorkerError::Internal(format!(
                    "sched_setaffinity failed for core {assigned_core}: {err}"
                )));
            }
        }
    }

    #[cfg(any(not(target_os = "linux"), miri))]
    {
        // Non-Linux hosts (e.g. macOS host development): no-op
        let _ = assigned_core;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_affinity_spec() {
        assert_eq!(parse_affinity_spec("0").unwrap(), (0, 0x0001));
        assert_eq!(parse_affinity_spec("8").unwrap(), (8, 0x0100));
        assert_eq!(parse_affinity_spec("0x00FF").unwrap(), (0, 0x00FF));
        assert_eq!(parse_affinity_spec("0xFF00").unwrap(), (8, 0xFF00));
    }

    #[test]
    fn test_capability_check_x100() {
        assert!(verify_cluster_capabilities("x100", Some("2")).is_ok());
    }

    #[test]
    fn test_capability_check_a100_non_riscv() {
        #[cfg(not(target_arch = "riscv64"))]
        {
            let res = verify_cluster_capabilities_with_sim("a100", None, None);
            assert!(matches!(res, Err(WorkerError::CapabilityMismatch(_))));
            if let Err(e) = res {
                assert_eq!(e.exit_code(), 10);
            }

            let res_core8 = verify_cluster_capabilities_with_sim("x100", Some("8"), None);
            assert!(matches!(res_core8, Err(WorkerError::CapabilityMismatch(_))));
        }
    }

    #[test]
    fn test_capability_check_a100_simulated() {
        assert!(verify_cluster_capabilities_with_sim("a100", None, Some("1")).is_ok());
        assert!(verify_cluster_capabilities_with_sim("a100", Some("8"), Some("1")).is_ok());
        assert!(verify_cluster_capabilities_with_sim("x100", Some("8"), Some("1")).is_err());
    }
}
