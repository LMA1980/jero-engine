//! CLI parsing and configuration for `jero-engine`.

use std::os::fd::RawFd;
use std::path::PathBuf;

use super::error::WorkerError;

/// Configuration parsed from command-line arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    /// Path to Unix Domain Socket for manager communication.
    pub socket_path: Option<PathBuf>,
    /// Pre-connected socket file descriptor (e.g. from supervisor `socketpair`).
    pub socket_fd: Option<RawFd>,
    /// Path to local GGUF model weights file.
    pub model_path: PathBuf,
    /// Compute cluster target (`x100` or `a100`).
    pub cluster: String,
    /// Number of worker threads for parallel compute execution.
    pub threads: Option<usize>,
    /// Explicit core pinning or affinity mask (e.g. `8` or `0x00FF`).
    pub affinity: Option<String>,
    /// Enable verbose diagnostic logging.
    pub verbose: bool,
    /// Whether worker acts as socket listener rather than client.
    pub listen: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            socket_path: None,
            socket_fd: None,
            model_path: PathBuf::new(),
            cluster: "x100".to_owned(),
            threads: None,
            affinity: None,
            verbose: false,
            listen: false,
        }
    }
}

impl WorkerConfig {
    /// Parses command-line arguments.
    ///
    /// Returns `Ok(Some(config))` on successful parse.
    /// Returns `Ok(None)` if `--help` or `--version` was handled (clean exit 0).
    /// Returns `Err(WorkerError::InvalidArguments)` on invalid or missing options.
    ///
    /// # Errors
    /// Returns [`WorkerError`] if required arguments are missing or malformed.
    pub fn parse_args<I>(mut args: I) -> Result<Option<Self>, WorkerError>
    where
        I: Iterator<Item = String>,
    {
        let mut config = Self::default();

        // Skip binary name if present
        let _ = args.next();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    return Ok(None);
                }
                "-V" | "--version" => {
                    println!("jero-engine {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                "--socket" | "--socket-path" => {
                    let path = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(format!("{arg} requires a path"))
                    })?;
                    if path.trim().is_empty() {
                        return Err(WorkerError::InvalidArguments(format!(
                            "{arg} path cannot be empty"
                        )));
                    }
                    config.socket_path = Some(PathBuf::from(path));
                }
                "--socket-fd" => {
                    let fd_str = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--socket-fd requires an integer fd".to_owned(),
                        )
                    })?;
                    let fd = fd_str.parse::<RawFd>().map_err(|e| {
                        WorkerError::InvalidArguments(format!(
                            "invalid --socket-fd '{fd_str}': {e}"
                        ))
                    })?;
                    if fd < 0 {
                        return Err(WorkerError::InvalidArguments(
                            "--socket-fd must be non-negative".to_owned(),
                        ));
                    }
                    config.socket_fd = Some(fd);
                }
                "--model" | "--weights-path" => {
                    let path = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(format!("{arg} requires a path"))
                    })?;
                    if path.trim().is_empty() {
                        return Err(WorkerError::InvalidArguments(format!(
                            "{arg} path cannot be empty"
                        )));
                    }
                    config.model_path = PathBuf::from(path);
                }
                "--model-id" => {
                    let _ = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--model-id requires an identifier".to_owned(),
                        )
                    })?;
                }
                "--threads" => {
                    let threads_str = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--threads requires an integer count".to_owned(),
                        )
                    })?;
                    let threads = threads_str.parse::<usize>().map_err(|e| {
                        WorkerError::InvalidArguments(format!(
                            "invalid --threads count '{threads_str}': {e}"
                        ))
                    })?;
                    if threads == 0 {
                        return Err(WorkerError::InvalidArguments(
                            "--threads must be greater than 0".to_owned(),
                        ));
                    }
                    config.threads = Some(threads);
                }
                "--memlock-limit" => {
                    let _ = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--memlock-limit requires a byte ceiling".to_owned(),
                        )
                    })?;
                }
                "--cluster" => {
                    let cluster = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--cluster requires a target (x100 or a100)".to_owned(),
                        )
                    })?;
                    let cluster_lower = cluster.to_ascii_lowercase();
                    if cluster_lower != "x100" && cluster_lower != "a100" {
                        return Err(WorkerError::InvalidArguments(format!(
                            "unsupported cluster '{cluster}'; must be 'x100' or 'a100'"
                        )));
                    }
                    config.cluster = cluster_lower;
                }
                "--affinity" => {
                    let aff = args.next().ok_or_else(|| {
                        WorkerError::InvalidArguments(
                            "--affinity requires a core or mask".to_owned(),
                        )
                    })?;
                    if aff.trim().is_empty() {
                        return Err(WorkerError::InvalidArguments(
                            "--affinity cannot be empty".to_owned(),
                        ));
                    }
                    config.affinity = Some(aff);
                }
                "--verbose" => {
                    config.verbose = true;
                }
                "--listen" => {
                    config.listen = true;
                }
                unknown => {
                    return Err(WorkerError::InvalidArguments(format!(
                        "unknown option '{unknown}'"
                    )));
                }
            }
        }

        // Validate required parameters
        if config.model_path.as_os_str().is_empty() {
            return Err(WorkerError::InvalidArguments(
                "missing required argument --model <path>".to_owned(),
            ));
        }

        if config.socket_path.is_none() && config.socket_fd.is_none() {
            return Err(WorkerError::InvalidArguments(
                "either --socket <path> or --socket-fd <fd> is required".to_owned(),
            ));
        }

        Ok(Some(config))
    }
}

/// Prints CLI usage help.
pub fn print_usage() {
    println!(
        r"jero-engine: Isolated compute worker binary for the SpacemiT K3 platform.

USAGE:
    jero-engine --model <path> (--socket <path> | --socket-fd <fd>) [OPTIONS]

REQUIRED:
    --model <path>         Path to local GGUF model weights file
    --socket <path>        Unix domain socket control endpoint
    --socket-fd <fd>       Pre-connected socket file descriptor (alternative to --socket)

OPTIONS:
    --cluster <x100|a100>  Target compute cluster [default: x100]
    --threads <count>      Number of compute threads (max 6 on x100, max 8 on a100)
    --affinity <core|mask> Explicit core pinning (e.g. 8, 0x003F, 0-5, 0xFF00)
    --listen               Listen on socket rather than connecting to supervisor
    --verbose              Enable verbose diagnostic output
    -h, --help             Print help information
    -V, --version          Print version information

EXIT CODES:
    0    Success (graceful shutdown, SIGINT/SIGTERM, EOF, --help, --version)
    2    Invalid CLI arguments
    10   CapabilityMismatch (unsupported cluster, A100 requested on non-A100 host, exclusive file lock)
    11   InsufficientLockableMemory (memory footprint exceeds 28 GiB or mlock failure)
    12   ResourceExhausted (model file not found, empty file, or context exceeded)
    13   ProtocolError (framing or protocol violation, UDS communication error)
"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_help_and_version() {
        let help_args = vec!["jero-engine".to_owned(), "--help".to_owned()];
        assert_eq!(
            WorkerConfig::parse_args(help_args.into_iter()).unwrap(),
            None
        );

        let ver_args = vec!["jero-engine".to_owned(), "-V".to_owned()];
        assert_eq!(
            WorkerConfig::parse_args(ver_args.into_iter()).unwrap(),
            None
        );
    }

    #[test]
    fn test_missing_required_args() {
        // Missing --model
        let args = vec![
            "jero-engine".to_owned(),
            "--socket".to_owned(),
            "/tmp/test.sock".to_owned(),
        ];
        assert!(WorkerConfig::parse_args(args.into_iter()).is_err());

        // Missing --socket and --socket-fd
        let args = vec![
            "jero-engine".to_owned(),
            "--model".to_owned(),
            "/tmp/model.gguf".to_owned(),
        ];
        assert!(WorkerConfig::parse_args(args.into_iter()).is_err());
    }

    #[test]
    fn test_valid_args() {
        let args = vec![
            "jero-engine".to_owned(),
            "--socket".to_owned(),
            "/tmp/test.sock".to_owned(),
            "--model".to_owned(),
            "/models/test.gguf".to_owned(),
            "--cluster".to_owned(),
            "x100".to_owned(),
            "--affinity".to_owned(),
            "2".to_owned(),
            "--verbose".to_owned(),
        ];
        let cfg = WorkerConfig::parse_args(args.into_iter()).unwrap().unwrap();
        assert_eq!(cfg.socket_path.unwrap(), PathBuf::from("/tmp/test.sock"));
        assert_eq!(cfg.model_path, PathBuf::from("/models/test.gguf"));
        assert_eq!(cfg.cluster, "x100");
        assert_eq!(cfg.affinity.as_deref(), Some("2"));
        assert!(cfg.verbose);
    }

    #[test]
    fn test_invalid_cluster() {
        let args = vec![
            "jero-engine".to_owned(),
            "--socket".to_owned(),
            "/tmp/test.sock".to_owned(),
            "--model".to_owned(),
            "/models/test.gguf".to_owned(),
            "--cluster".to_owned(),
            "gpu99".to_owned(),
        ];
        assert!(WorkerConfig::parse_args(args.into_iter()).is_err());
    }
}
