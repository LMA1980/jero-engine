---
name: jero-engine
description: "Configure and run the jero-engine compute runtime, RecipeRegistry capability verification, LockedModelWeights mlock bounds, jero-diag diagnostics, and jero-worker process engine."
---

# `jero-engine` Compute Runtime, Capability Verification, `jero-diag` & `jero-worker` Invariants

## 1. `RecipeRegistry` & Capability Verification

- Register architecture recipes (`jero-arch-synthetic`, `jero-arch-llama`, `jero-arch-mistral`, `jero-arch-phi3`, `jero-arch-bitnet`, `jero-arch-dory`) in `RecipeRegistry` and validate GGUF metadata via `verify_metadata_capabilities` / `verify_model_capabilities` before locking weights.

## 2. Conservative Diagnostic Context Bounds (`n_ctx: 64`)

- Diagnostic utilities (`jero-diag`) and default integration test harnesses must use conservative context bounds (e.g., `n_ctx: 64`, requiring ~64 KiB KV cache) so tests run cleanly out-of-the-box in unprivileged environments without `InsufficientLockableMemory` (exit code `11`) admission refusal.
- Pre-validate compilation locally (`cargo check --target riscv64gc-unknown-linux-gnu --workspace --tests`) before dispatching runs to target hardware.

## 3. Worker Execution (`jero-worker`)

- Multi-process worker instances (`jero-worker`) handle decoupled matrix and attention evaluation over Unix domain sockets with shared memory weight arenas.
