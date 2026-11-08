#![allow(clippy::pedantic, clippy::undocumented_unsafe_blocks)]

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const BIN_PATH: &str = env!("CARGO_BIN_EXE_jero-worker");

#[test]
fn test_cli_help_and_version_exit_0() {
    // --help
    let output = Command::new(BIN_PATH)
        .arg("--help")
        .output()
        .expect("run k3-worker --help");
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("USAGE:"));
    assert!(stdout.contains("--model"));

    // -h
    let output_h = Command::new(BIN_PATH)
        .arg("-h")
        .output()
        .expect("run k3-worker -h");
    assert_eq!(output_h.status.code(), Some(0));

    // --version
    let output_v = Command::new(BIN_PATH)
        .arg("--version")
        .output()
        .expect("run k3-worker --version");
    assert_eq!(output_v.status.code(), Some(0));
    let stdout_v = String::from_utf8_lossy(&output_v.stdout);
    assert!(stdout_v.contains("k3-worker"));

    // -V
    let output_short_v = Command::new(BIN_PATH)
        .arg("-V")
        .output()
        .expect("run k3-worker -V");
    assert_eq!(output_short_v.status.code(), Some(0));
}

#[test]
fn test_cli_invalid_args_exit_2() {
    // Missing all arguments
    let output = Command::new(BIN_PATH).output().expect("run k3-worker");
    assert_eq!(output.status.code(), Some(2));

    // Missing --socket/--socket-fd
    let output = Command::new(BIN_PATH)
        .args(["--model", "/tmp/fake.gguf"])
        .output()
        .expect("run k3-worker missing socket");
    assert_eq!(output.status.code(), Some(2));

    // Missing --model
    let output = Command::new(BIN_PATH)
        .args(["--socket", "/tmp/test.sock"])
        .output()
        .expect("run k3-worker missing model");
    assert_eq!(output.status.code(), Some(2));

    // Invalid cluster target
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/tmp/fake.gguf",
            "--socket",
            "/tmp/test.sock",
            "--cluster",
            "cuda",
        ])
        .output()
        .expect("run k3-worker invalid cluster");
    assert_eq!(output.status.code(), Some(2));

    // Unknown option
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/tmp/fake.gguf",
            "--socket",
            "/tmp/test.sock",
            "--bogus-flag",
        ])
        .output()
        .expect("run k3-worker unknown flag");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn test_model_not_found_or_empty_exit_12() {
    // Non-existent model file -> exit code 12
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/path/to/nonexistent/weights.gguf",
            "--socket",
            "/tmp/fake.sock",
        ])
        .output()
        .expect("run k3-worker nonexistent model");
    assert_eq!(output.status.code(), Some(12));

    // Empty model file -> exit code 12
    let empty_path = std::env::temp_dir().join("k3_worker_test_empty.gguf");
    File::create(&empty_path).unwrap();
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            empty_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
        ])
        .output()
        .expect("run k3-worker empty model");
    let _ = std::fs::remove_file(&empty_path);
    assert_eq!(output.status.code(), Some(12));
}

#[test]
fn test_capability_mismatch_exit_10() {
    // Create a valid dummy model file
    let model_path = std::env::temp_dir().join("k3_worker_cap_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_V3_MODEL_HEADER").unwrap();
    }

    // 1. Requesting A100 cluster on non-riscv64 host without K3_SIMULATE_A100 -> exit code 10
    #[cfg(not(target_arch = "riscv64"))]
    {
        let output = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                "/tmp/fake.sock",
                "--cluster",
                "a100",
            ])
            .output()
            .expect("run k3-worker a100 without sim");
        assert_eq!(output.status.code(), Some(10));

        // 2. Requesting core 8 (A100 cluster) on non-riscv64 host -> exit code 10
        let output = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                "/tmp/fake.sock",
                "--affinity",
                "8",
            ])
            .output()
            .expect("run k3-worker core 8 without sim");
        assert_eq!(output.status.code(), Some(10));
    }

    // 3. Exclusive lock held on model file -> exit code 10
    let lock_file = File::open(&model_path).unwrap();
    let flock_ret = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(flock_ret, 0, "Exclusive lock acquired by test process");

    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
        ])
        .output()
        .expect("run k3-worker with exclusively locked model");
    assert_eq!(output.status.code(), Some(10));

    unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    drop(lock_file);
    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_invalid_affinity_out_of_bounds_exit_2() {
    let model_path = std::env::temp_dir().join("k3_worker_aff64_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--affinity",
            "64",
        ])
        .output()
        .expect("run k3-worker --affinity 64");

    let _ = std::fs::remove_file(&model_path);
    assert_eq!(
        output.status.code(),
        Some(2),
        "Invalid core index 64 must return exit code 2 (InvalidArguments)"
    );
}

#[test]
fn test_signal_termination_sigint_and_sigterm_exit_0() {
    let model_path = std::env::temp_dir().join("k3_worker_sig_test_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    for &sig in &[libc::SIGINT, libc::SIGTERM] {
        let sig_name = if sig == libc::SIGINT {
            "SIGINT"
        } else {
            "SIGTERM"
        };
        let sock_path = std::env::temp_dir().join(format!("k3_worker_sig_{sig_name}.sock"));
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }

        let mut child = Command::new(BIN_PATH)
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                sock_path.to_str().unwrap(),
                "--listen",
                "--cluster",
                "x100",
            ])
            .spawn()
            .expect("spawn worker");

        // Wait for socket to become available and connect to unblock listener.accept()
        let mut connected = None;
        let connect_deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < connect_deadline {
            if sock_path.exists() {
                if let Ok(stream) = UnixStream::connect(&sock_path) {
                    connected = Some(stream);
                    break;
                }
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _stream = connected.expect("failed to connect to worker socket");

        // Short sleep to allow worker to register signal handlers and enter select!
        thread::sleep(Duration::from_millis(50));

        // Deliver OS signal
        unsafe {
            libc::kill(child.id() as i32, sig);
        }

        // Wait for exit, asserting it exits with code 0 within 2 seconds
        let wait_start = Instant::now();
        let mut exited = false;
        while wait_start.elapsed() < Duration::from_secs(2) {
            if let Ok(Some(_)) = child.try_wait() {
                exited = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }

        if !exited {
            let _ = child.kill();
        }
        let status = child.wait().expect("wait for child");
        assert_eq!(
            status.code(),
            Some(0),
            "{sig_name} must exit cleanly with code 0"
        );
        assert!(
            exited,
            "{sig_name} termination timed out (> 2 seconds), deadlock detected"
        );

        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }
    }

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_threads_rejections() {
    let model_path = std::env::temp_dir().join("k3_worker_threads_test_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    // --threads 0 -> exit code 2 (InvalidArguments)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--threads",
            "0",
        ])
        .output()
        .expect("run k3-worker --threads 0");
    assert_eq!(output.status.code(), Some(2));

    // --threads 7 on x100 -> exit code 10 (CapabilityMismatch: exceeds X100 cap of 6)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--threads",
            "7",
        ])
        .output()
        .expect("run k3-worker --threads 7 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0x00C0 on x100 encroaches on control cores -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0x00C0",
        ])
        .output()
        .expect("run k3-worker --affinity 0x00C0 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0x00FF on x100 encroaches on control cores 6-7 -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0x00FF",
        ])
        .output()
        .expect("run k3-worker --affinity 0x00FF on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0-6 on x100 encroaches on control core 6 -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0-6",
        ])
        .output()
        .expect("run k3-worker --affinity 0-6 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0-3 with --threads 5 -> exit code 10 (threads exceed available cores)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0-3",
            "--threads",
            "5",
        ])
        .output()
        .expect("run k3-worker --threads 5 on 4 cores");
    assert_eq!(output.status.code(), Some(10));

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_worker_engine_multi_thread_affinity_and_no_control_encroachment() {
    let model_path = std::env::temp_dir().join("k3_worker_engine_threads_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    let config = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_engine.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(4),
        affinity: Some("0-3".to_owned()),
        verbose: false,
        listen: false,
    };
    let weights = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let engine = jero_engine::worker::WorkerEngine::new(config, weights);

    assert_eq!(engine.thread_count(), 4);
    assert_eq!(engine.compute_mask(), 0x000F);
    assert_eq!(engine.assigned_cores(), &[0, 1, 2, 3]);
    // Assert zero compute threads ever run on dedicated active control cores 6 or 7
    for &core in engine.assigned_cores() {
        assert!(
            core < 6,
            "core {core} must not encroach on active control cores 6-7"
        );
        assert_eq!((1_u64 << core) & 0x00C0, 0);
    }

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_worker_engine_non_contiguous_affinity_multi_thread() {
    let model_path = std::env::temp_dir().join("k3_worker_engine_noncontig_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    let config = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_noncontig_engine.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(2),
        affinity: Some("0,2".to_owned()),
        verbose: false,
        listen: false,
    };
    let weights = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let engine = jero_engine::worker::WorkerEngine::new(config, weights);

    assert_eq!(engine.thread_count(), 2);
    assert_eq!(engine.compute_mask(), 0x0005);
    assert_eq!(engine.assigned_cores(), &[0, 2]);

    for &core in engine.assigned_cores() {
        assert!(
            core < 6,
            "core {core} must not encroach on active control cores 6-7"
        );
        assert_eq!((1_u64 << core) & 0x00C0, 0);
    }

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_non_contiguous_and_cross_cluster_rejections() {
    let model_path = std::env::temp_dir().join("k3_worker_noncontig_rejections.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    // --affinity 0,2 with --threads 3 -> exit code 10 (exceeds available cores in mask)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0,2",
            "--threads",
            "3",
        ])
        .output()
        .expect("run k3-worker --threads 3 on 2 cores");
    assert_eq!(output.status.code(), Some(10));

    // --cluster x100 with --affinity 8-15 (A100 cores) -> exit code 10
    let output_x100_a100 = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "8-15",
        ])
        .output()
        .expect("run k3-worker x100 with A100 cores");
    assert_eq!(output_x100_a100.status.code(), Some(10));

    // --cluster a100 with --affinity 8-9 and --threads 3 (exceeds 2 available cores) -> exit code 10
    let output_a100_sub3 = Command::new(BIN_PATH)
        .env("K3_SIMULATE_A100", "1")
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "a100",
            "--affinity",
            "8-9",
            "--threads",
            "3",
        ])
        .output()
        .expect("run k3-worker a100 with threads 3 on 2 cores");
    assert_eq!(output_a100_sub3.status.code(), Some(10));

    // --cluster x100 with --threads 7 (exceeds 6-thread compute cap) -> exit code 10
    let output_x100_t7 = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--threads",
            "7",
        ])
        .output()
        .expect("run k3-worker x100 with 7 threads");
    assert_eq!(output_x100_t7.status.code(), Some(10));

    // --cluster a100 with --threads 9 (exceeds 8-thread compute cap) -> exit code 10
    let output_a100_t9 = Command::new(BIN_PATH)
        .env("K3_SIMULATE_A100", "1")
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "a100",
            "--threads",
            "9",
        ])
        .output()
        .expect("run k3-worker a100 with 9 threads");
    assert_eq!(output_a100_t9.status.code(), Some(10));

    // --cluster x100 with whitespace cpuset encroaching on control core 6 ("  0 - 6  ") -> exit code 10
    let output_ws_encroach = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "  0 - 6  ",
        ])
        .output()
        .expect("run k3-worker x100 with whitespace 0-6");
    assert_eq!(output_ws_encroach.status.code(), Some(10));

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_worker_engine_try_new_fail_closed_validation() {
    let model_path = std::env::temp_dir().join("k3_worker_engine_try_new_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    // 1. Encroaching on active control cores 6-7 (0x00C0)
    let config_encroach = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(1),
        affinity: Some("0x00C0".to_owned()),
        verbose: false,
        listen: false,
    };
    let w1 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let err_encroach = jero_engine::worker::WorkerEngine::try_new(config_encroach, w1).unwrap_err();
    assert_eq!(err_encroach.exit_code(), 10);
    assert!(
        err_encroach
            .to_string()
            .contains("encroaches on dedicated active control cores")
    );

    // 2. Unparseable affinity spec
    let config_unparseable = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(1),
        affinity: Some("not_a_valid_mask".to_owned()),
        verbose: false,
        listen: false,
    };
    let w2 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let err_unparseable =
        jero_engine::worker::WorkerEngine::try_new(config_unparseable, w2).unwrap_err();
    assert_eq!(err_unparseable.exit_code(), 2);

    // 3. Threads exceeding available cores
    let config_excess_threads = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(3),
        affinity: Some("0,2".to_owned()),
        verbose: false,
        listen: false,
    };
    let w3 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let err_excess =
        jero_engine::worker::WorkerEngine::try_new(config_excess_threads, w3).unwrap_err();
    assert_eq!(err_excess.exit_code(), 10);
    assert!(err_excess.to_string().contains("exceeds 2 available cores"));

    // 4. X100 threads exceeding 6-thread cap (no affinity)
    let config_x100_t7 = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(7),
        affinity: None,
        verbose: false,
        listen: false,
    };
    let w4 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let err_x100_t7 = jero_engine::worker::WorkerEngine::try_new(config_x100_t7, w4).unwrap_err();
    assert_eq!(err_x100_t7.exit_code(), 10);
    assert!(
        err_x100_t7
            .to_string()
            .contains("exceeding X100 compute cap of 6 threads")
    );

    // 5. A100 sub-mask thread bounding (threads 3 > cores 8-9) under simulation
    unsafe {
        std::env::set_var("K3_SIMULATE_A100", "1");
    }
    let config_a100_sub3 = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "a100".to_owned(),
        threads: Some(3),
        affinity: Some("8-9".to_owned()),
        verbose: false,
        listen: false,
    };
    let w5 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let err_a100_sub3 =
        jero_engine::worker::WorkerEngine::try_new(config_a100_sub3, w5).unwrap_err();
    assert_eq!(err_a100_sub3.exit_code(), 10);
    assert!(
        err_a100_sub3
            .to_string()
            .contains("exceeds 2 available cores")
    );

    // 6. Valid threads without affinity on X100 (threads = 4)
    let config_x100_t4 = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(4),
        affinity: None,
        verbose: false,
        listen: false,
    };
    let w6 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let engine_x100_t4 =
        jero_engine::worker::WorkerEngine::try_new(config_x100_t4, w6).expect("engine t4");
    assert_eq!(engine_x100_t4.thread_count(), 4);
    assert_eq!(engine_x100_t4.compute_mask(), 0x003F);
    assert_eq!(engine_x100_t4.assigned_cores(), &[0, 1, 2, 3]);

    // 7. Whitespace in affinity spec ("  0 - 3  ")
    let config_ws = jero_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_try_new.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(4),
        affinity: Some("  0 - 3  ".to_owned()),
        verbose: false,
        listen: false,
    };
    let w7 = jero_engine::worker::LockedWeights::open(&model_path).unwrap();
    let engine_ws = jero_engine::worker::WorkerEngine::try_new(config_ws, w7).expect("engine ws");
    assert_eq!(engine_ws.thread_count(), 4);
    assert_eq!(engine_ws.compute_mask(), 0x000F);
    assert_eq!(engine_ws.assigned_cores(), &[0, 1, 2, 3]);

    unsafe {
        std::env::remove_var("K3_SIMULATE_A100");
    }

    let _ = std::fs::remove_file(&model_path);
}
