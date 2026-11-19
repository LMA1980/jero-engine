# Incident Report: `main` Branch CI Failure in `jero-engine` (`clippy::needless_bool`)

- **Date**: 2026-11-19
- **Repository**: `LMA1980/jero-engine`
- **Affected Commit**: `5b74f50c395ae8fe5914ba37c2264dd1cca8532e`
- **Tracking Issue**: [#205](https://github.com/LMA1980/jero-engine/issues/205)
- **Resolution Branch**: `feat/fix-main-ci-clippy-needless-bool`

______________________________________________________________________

## 1. Summary & Impact

The `main` branch CI pipeline (`job-1795073570-5b74f50c`) on `LMA1980/jero-engine` failed `clippy-check` (`exit_code = 101`) under Rust `1.100.0`:

```text
error: this `if` guard returns a bool literal and is followed by another
   --> src/affinity.rs:511:5
    |
    = note: `-D clippy::needless-bool` implied by `-D warnings`
```

______________________________________________________________________

## 2. Root Cause Analysis

1. **Conditional Compilation Exposed `clippy::needless_bool` on `x86_64`**:
   `has_a100_ime_for_mask` in `src/affinity.rs` checked `if a100_mask == 0 { return false; }` before a `#[cfg(all(target_os = "linux", target_arch = "riscv64", ...))]` hardware probe block and a fallback `#[cfg(not(...))] { true }` block. On non-`riscv64` targets (`x86_64` CI runners), the function expanded to `if a100_mask == 0 { return false; } true`, which Rust `1.100.0` Clippy rejects under `-D warnings`.
1. **How the Regression Reached `main`**:
   Prior PRs were verified on earlier Clippy versions (`1.99.x`) before CI workers updated to Rust `1.100.0`, so the expanded `x86_64` lint did not trigger until post-merge `main` execution on `1.100.0`.

______________________________________________________________________

## 3. Remediation & Preventive Measures

### Remediation

1. Refactored `has_a100_ime_for_mask` in `src/affinity.rs` to return `a100_mask != 0` directly in the `#[cfg(not(...))]` branch while retaining the `if a100_mask == 0 { return false; }` early return inside the `#[cfg(all(...))]` `riscv64` probe branch.
1. Verified the full Pre-PR Local CI Gate Sequence (`cargo fmt`, `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and `cargo +nightly miri test --workspace --lib`).

### Preventive Measures

1. Validate `cargo clippy --workspace --all-targets -- -D warnings` with the active CI toolchain (`1.100.0`) across `#[cfg]` target branches.
1. Prefer direct boolean expressions (`a100_mask != 0`) over `if cond { return false; } true` in fallback `#[cfg]` blocks.
