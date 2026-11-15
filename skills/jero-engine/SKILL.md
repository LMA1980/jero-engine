---
name: jero-engine
description: "Configure and run the jero-engine compute runtime, RecipeRegistry capability verification, LockedModelWeights mlock bounds, jero-diag diagnostics, and jero-worker process engine."
---

# `jero-engine` Compute Runtime, Capability Verification, `jero-diag` & `jero-worker` Invariants

## 1. `RecipeRegistry` & Capability Verification

- Register architecture recipes (`jero-arch-synthetic`, `jero-arch-llama`, `jero-arch-mistral`, `jero-arch-phi3`, `jero-arch-bitnet`, `jero-arch-dory`) and built-in dense/hybrid/SSM/recurrent recipes (`Mamba2Recipe` for `mamba`/`mamba2`, `Rwkv7Recipe` for `rwkv6`/`rwkv7`, `HybridSsmTransformerRecipe` for `jamba`/`zamba2`/`granitehybrid`/`lfm2`, `Lfm2MoeRecipe` for `lfm2moe`, and multiplier-aware `GraniteRecipe` for `granite`) in `RecipeRegistry` and validate GGUF metadata via `verify_metadata_capabilities` / `verify_model_capabilities` before locking weights.
- Pure linear-time SSM/RWKV architectures (`mamba`, `mamba2`, `rwkv6`, `rwkv7`) allocate O(1) context-invariant dual-bank ping-pong state arenas (`kv_block_count = 0`), while hybrid architectures (`jamba`, `zamba2`, `granitehybrid`, `lfm2`, `lfm2moe`) pair compact KV cache blocks strictly for attention layers with dual-bank recurrent state (`estimate_hybrid_ssm_arena_bytes`).

## 2. Conservative Diagnostic Context Bounds (`n_ctx: 64`), Chat Templates & Stop Termination

- Diagnostic utilities (`jero-diag`) and default integration test harnesses must use conservative context bounds (e.g., `n_ctx: 64`, requiring ~64 KiB KV cache) so tests run cleanly out-of-the-box in unprivileged environments without `InsufficientLockableMemory` (exit code `11`) admission refusal.
- By default, single-prompt diagnostics (`jero-diag`), OpenAI serving (`--serve-openai`), and the 5-pillar open-source evaluation suite (`--eval-suite`) read `tokenizer.chat_template` and `tokenizer.ggml.add_bos_token` to format prompts across ChatML (including `qwen*`, `dory`, `lfm2`, `lfm2moe`), Llama-3, Phi-3/4, Gemma 2/3, Mistral, DeepSeek, and placeholder templates while emitting BOS at most once. Base models with no template fall back to raw prompt text, and `--raw-prompt` bypasses chat-template formatting explicitly.
- `--eval-suite` honors caller-supplied `--n-ctx` and `--max-tokens` without artificial caps, reserving conservative defaults (`n_ctx: 64`, `max_tokens: 16`) only when omitted.
- `SinglePromptEngine`, `WorkerEngine`, and `jero-diag` automatically halt generation upon emitting primary EOS or family stop/EOT tokens (`discover_stop_token_ids`), exclude terminal stop tokens from detokenized output, and report `stop_reason` (`eos_token` / `max_tokens`) in `PassTiming`, JSON benchmark reports, OpenAI `finish_reason`, and worker output slab headers unless `--ignore-eos` is supplied.
- Passing `--model <path.gguf>` always executes real GGUF weights through the hydrated `ArchRecipe` and reports `"execution_mode": "real_gguf"` with `architecture`, `model_filename`, and `quant_summary`. Synthetic `ForwardModel` hardware calibration requires the explicit `--synthetic-matrix` flag (mutually exclusive with `--model`), emitting `"execution_mode": "synthetic_matrix"`.
- Pre-validate compilation locally (`cargo check --target riscv64gc-unknown-linux-gnu --workspace --tests`) before dispatching runs to target hardware.

## 3. Worker Execution (`jero-worker`), Phase Scheduling & Repack Residency

- Multi-process worker instances (`jero-worker`) handle decoupled matrix and attention evaluation over Unix domain sockets with shared memory weight arenas.
- Under `ClusterPhasePolicy::Auto` on heterogeneous clusters with `A100` (`VLEN=1024`) and `XSMTVDotII` (`has_ime`), non-guardrail dense, MoE, and BitNet models resolve to `A100Only` (`A100` prefill + `A100` decode); explicit `A100PrefillX100Decode`, `A100PrefillA100Decode`, and `RowSplit` prefill also target `a100_pool`, while guardrails and non-IME / single-cluster hosts stay on `X100Only`.
- `PassTiming`, `RunMetric`, `StatisticsSummary` (`mean`, `median`, `min`, `max`, `stddev`), and `jero-diag` CLI tables report `prefill_tokens_per_sec` (`pp tok/s`, returning `0.0` when `prefill_ns == 0`) alongside `prefill_ms` (`TTFT`) and `decode_tokens_per_sec`.
- `estimate_footprint` / `calculate_footprint` tracks `repack_arena_bytes` (compact nibble `smt.vpack.vv` vs `i8` tiles), bounded `prefill_scratch_bytes`, and `eager_repack_enabled`; `check_memory_admission*` fails closed on `weight_bytes + kv_bytes + prefill_scratch_bytes > budget`, and downgrades (`eager_repack_enabled = false`, `repack_arena_bytes = 0`) when only the optional repack arena exceeds headroom.
- `LockedModelWeights::prepare_repacked_weights` populates the model-scoped `RepackedTensorStore` once in parallel across active `WorkerPool` threads, reuses `ModelView` across repeated prompt runs (`--repeat N`), and clears resident repack allocations and fallback caches cleanly on drop.
