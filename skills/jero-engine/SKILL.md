---
name: jero-engine
description: "Configure and run the jero-engine compute runtime, RecipeRegistry capability verification, LockedModelWeights mlock bounds, jero-diag diagnostics, and jero-worker process engine."
---

# `jero-engine` Compute Runtime, Capability Verification, `jero-diag` & `jero-worker` Invariants

## 1. `RecipeRegistry` & Capability Verification

- Register architecture recipes (`jero-arch-synthetic`, `jero-arch-llama`, `jero-arch-mistral`, `jero-arch-phi3`, `jero-arch-bitnet`, `jero-arch-dory`) in `RecipeRegistry` and validate GGUF metadata via `verify_metadata_capabilities` / `verify_model_capabilities` before locking weights.

## 2. Conservative Diagnostic Context Bounds (`n_ctx: 64`), Chat Templates & Stop Termination

- Diagnostic utilities (`jero-diag`) and default integration test harnesses must use conservative context bounds (e.g., `n_ctx: 64`, requiring ~64 KiB KV cache) so tests run cleanly out-of-the-box in unprivileged environments without `InsufficientLockableMemory` (exit code `11`) admission refusal.
- By default, single-prompt diagnostics (`jero-diag`), OpenAI serving (`--serve-openai`), and the 5-pillar open-source evaluation suite (`--eval-suite`) read `tokenizer.chat_template` and `tokenizer.ggml.add_bos_token` to format prompts across ChatML, Llama-3, Phi-3/4, Gemma 2/3, Mistral, DeepSeek, and placeholder templates while emitting BOS at most once. Base models with no template fall back to raw prompt text, and `--raw-prompt` bypasses chat-template formatting explicitly.
- `--eval-suite` honors caller-supplied `--n-ctx` and `--max-tokens` without artificial caps, reserving conservative defaults (`n_ctx: 64`, `max_tokens: 16`) only when omitted.
- `SinglePromptEngine`, `WorkerEngine`, and `jero-diag` automatically halt generation upon emitting primary EOS or family stop/EOT tokens (`discover_stop_token_ids`), exclude terminal stop tokens from detokenized output, and report `stop_reason` (`eos_token` / `max_tokens`) in `PassTiming`, JSON benchmark reports, OpenAI `finish_reason`, and worker output slab headers unless `--ignore-eos` is supplied.
- Passing `--model <path.gguf>` always executes real GGUF weights through the hydrated `ArchRecipe` and reports `"execution_mode": "real_gguf"` with `architecture`, `model_filename`, and `quant_summary`. Synthetic `ForwardModel` hardware calibration requires the explicit `--synthetic-matrix` flag (mutually exclusive with `--model`), emitting `"execution_mode": "synthetic_matrix"`.
- Pre-validate compilation locally (`cargo check --target riscv64gc-unknown-linux-gnu --workspace --tests`) before dispatching runs to target hardware.

## 3. Worker Execution (`jero-worker`)

- Multi-process worker instances (`jero-worker`) handle decoupled matrix and attention evaluation over Unix domain sockets with shared memory weight arenas.
