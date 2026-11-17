# jero-engine

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-engine/card.svg" alt="jero-engine Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-engine/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-engine/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-engine/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-engine/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Low-level compute runtime for the SpacemiT K3 AI platform (16 RISC-V cores, 32GB Unified RAM).

## Overview

`jero-engine` implements the high-performance execution engine targeted at `riscv64`:

- **Binaries**: `jero-worker` (isolated inference worker) and `jero-diag` (hardware/kernel diagnostic harness, gated on `jero-kernels-rvv`).
- **Compute Kernels**: Unified VLA RVV 1.0 loops with A100 IME acceleration via `jero-kernels-rvv`.
- **Architecture Recipes**: Wired `jero-arch-*` crates (`synthetic`, `llama`, `mistral`, `phi3`, `bitnet`, `dory`) alongside built-in `gemma` (`gemma2`, `gemma3`, `gemma4`, `functiongemma`), `modernbert`, `qwen_deepseek` (`qwen2`, `qwen3`, `qwen3moe`, `deepseek2`, `qwen35`, `qwen35moe`), `deberta-v3`, `ssm`, and `machine_models` (`ttm`, `ft-transformer`, `tabpfn-v2`, `decision-transformer`) recipes.
- **Synchronization**: Thread-group scheduling and diagnostic barriers (`SpinBarrier`, `FutexBarrier`, `WorkerPool`).
- **Memory**: Single-slot KV arena views, sliding-window KV slicing, and `RLIMIT_MEMLOCK` admission checks.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependencies**: [`jero-core`](https://github.com/LMA1980/jero-core), [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv), [`jero-tokenizer`](https://github.com/LMA1980/jero-tokenizer), [`jero-arch-synthetic`](https://github.com/LMA1980/jero-arch-synthetic), [`jero-arch-llama`](https://github.com/LMA1980/jero-arch-llama), [`jero-arch-mistral`](https://github.com/LMA1980/jero-arch-mistral), [`jero-arch-phi3`](https://github.com/LMA1980/jero-arch-phi3), [`jero-arch-bitnet`](https://github.com/LMA1980/jero-arch-bitnet), and [`jero-arch-dory`](https://github.com/LMA1980/jero-arch-dory)
- **Companion Runtime Crates**: [`jero-kv-cache`](https://github.com/LMA1980/jero-kv-cache) and [`jero-manager`](https://github.com/LMA1980/jero-manager)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
