---
name: k3-engine-runtime
description: "Configure and run the k3-engine compute runtime, RecipeRegistry capability verification, LockedModelWeights mlock bounds, and k3-diag diagnostics."
---

# `k3-engine` Compute Runtime, Capability Verification & `k3-diag` Invariants

## 1. `RecipeRegistry` & Capability Verification

- Register architecture recipes (`arch-synthetic`, `k3-arch-llama`, `k3-arch-mistral`, `k3-arch-phi3`, `k3-arch-bitnet`, `k3-arch-flint`) in `RecipeRegistry` and validate GGUF metadata via `verify_metadata_capabilities` / `verify_model_capabilities` before locking weights.

## 2. Conservative Diagnostic Context Bounds (`n_ctx: 64`)

- Diagnostic utilities (`k3-diag`) and default integration test harnesses must use conservative context bounds (e.g., `n_ctx: 64`, requiring ~64 KiB KV cache) so tests run cleanly out-of-the-box in unprivileged environments without `InsufficientLockableMemory` (exit code `11`) admission refusal.
- Pre-validate compilation locally (`cargo check --target riscv64gc-unknown-linux-gnu --workspace --tests`) before dispatching runs to target hardware.
