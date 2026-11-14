//! `k3-worker`: Isolated compute worker library and runtime for the `SpacemiT` K3 platform.
//!
//! Provides CLI parsing, memory-locked read-only weight mapping, cluster affinity pinning
//! with conditional `/proc/set_ai_thread` registration, and zero-copy forward pass dispatch
//! over Unix Domain Sockets using Apache Fory and Apache Arrow zero-copy views.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_inception
)]

pub mod affinity;
pub mod cli;
pub mod error;
pub mod weights;
pub mod worker;

pub use affinity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, DEFAULT_ACTIVE_CONTROL_MASK,
    DEFAULT_X100_COMPUTE_CAP_MASK, X100_CLUSTER_MASK, X100_DEFAULT_MAX_COMPUTE_THREADS,
    cores_from_mask, parse_affinity_spec, pin_worker_core, verify_cluster_capabilities,
    verify_cluster_capabilities_full,
};
pub use cli::{WorkerConfig, print_usage};
pub use error::{
    EXIT_CAPABILITY_MISMATCH, EXIT_INSUFFICIENT_MEMLOCK, EXIT_INVALID_ARGS, EXIT_PROTOCOL_ERROR,
    EXIT_RESOURCE_EXHAUSTED, EXIT_SUCCESS, WorkerError,
};
pub use weights::LockedWeights;
pub use worker::{WorkerEngine, decode_slab_stop_reason, encode_slab_stop_reason};
