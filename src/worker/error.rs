//! Standardized error types and process exit code mappings for `k3-worker`.

use std::fmt;
use std::io;

use jero_core::error::K3Error;
use jero_core::ipc::protocol::StatusCode;
use jero_core::ipc::shm::IpcError;

/// Process exit codes per K3 architecture specifications.
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_INVALID_ARGS: i32 = 2;
pub const EXIT_CAPABILITY_MISMATCH: i32 = 10;
pub const EXIT_INSUFFICIENT_MEMLOCK: i32 = 11;
pub const EXIT_RESOURCE_EXHAUSTED: i32 = 12;
pub const EXIT_PROTOCOL_ERROR: i32 = 13;

/// Errors produced during worker initialization and execution.
#[derive(Debug)]
pub enum WorkerError {
    /// Command line arguments invalid or missing (exit code 2).
    InvalidArguments(String),
    /// Host capabilities or quantization incompatible (exit code 10).
    CapabilityMismatch(String),
    /// Memory footprint exceeds RLIMIT_MEMLOCK bounds or mlock failure (exit code 11).
    InsufficientLockableMemory(String),
    /// Model file missing, empty, or resource exhausted (exit code 12).
    ResourceExhausted(String),
    /// UDS framing, magic, or sequence protocol violation (exit code 13).
    ProtocolError(String),
    /// Internal unexpected execution failure (exit code 13).
    Internal(String),
}

impl WorkerError {
    /// Returns the standardized process exit code for this error.
    #[must_use]
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidArguments(_) => EXIT_INVALID_ARGS,
            Self::CapabilityMismatch(_) => EXIT_CAPABILITY_MISMATCH,
            Self::InsufficientLockableMemory(_) => EXIT_INSUFFICIENT_MEMLOCK,
            Self::ResourceExhausted(_) => EXIT_RESOURCE_EXHAUSTED,
            Self::ProtocolError(_) | Self::Internal(_) => EXIT_PROTOCOL_ERROR,
        }
    }

    /// Returns corresponding [`StatusCode`] for IPC responses.
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidArguments(_) => StatusCode::InternalError,
            Self::CapabilityMismatch(_) => StatusCode::CapabilityMismatch,
            Self::InsufficientLockableMemory(_) => StatusCode::InsufficientLockableMemory,
            Self::ResourceExhausted(_) => StatusCode::InternalError,
            Self::ProtocolError(_) => StatusCode::ProtocolError,
            Self::Internal(_) => StatusCode::InternalError,
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArguments(msg) => write!(f, "invalid arguments: {msg}"),
            Self::CapabilityMismatch(msg) => write!(f, "capability mismatch: {msg}"),
            Self::InsufficientLockableMemory(msg) => {
                write!(f, "insufficient lockable memory: {msg}")
            }
            Self::ResourceExhausted(msg) => write!(f, "resource exhausted: {msg}"),
            Self::ProtocolError(msg) => write!(f, "protocol error: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<io::Error> for WorkerError {
    fn from(err: io::Error) -> Self {
        if err.kind() == io::ErrorKind::InvalidData || err.kind() == io::ErrorKind::UnexpectedEof {
            Self::ProtocolError(err.to_string())
        } else {
            Self::Internal(err.to_string())
        }
    }
}

impl From<K3Error> for WorkerError {
    fn from(err: K3Error) -> Self {
        match err {
            K3Error::CapabilityMismatch { reason, .. } => Self::CapabilityMismatch(reason),
            K3Error::InsufficientLockableMemory { .. } => {
                Self::InsufficientLockableMemory(err.to_string())
            }
            K3Error::ContextLengthExceeded { .. } => Self::ResourceExhausted(err.to_string()),
            K3Error::AffinityError { reason, .. } => Self::CapabilityMismatch(reason),
            K3Error::TensorBoundsError { reason, .. } => Self::ProtocolError(reason),
            K3Error::RoutingUnavailable { reason } => Self::ProtocolError(reason),
            K3Error::QueueSaturated { .. } => Self::ResourceExhausted(err.to_string()),
        }
    }
}

impl From<IpcError> for WorkerError {
    fn from(err: IpcError) -> Self {
        match err {
            IpcError::InsufficientLockableMemory(k3) => Self::from(k3),
            IpcError::TensorBoundsError(k3) => Self::from(k3),
            IpcError::ProtocolError(msg) => Self::ProtocolError(msg),
            IpcError::IoError { reason, .. } => Self::Internal(reason),
        }
    }
}
