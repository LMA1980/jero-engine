# k3-engine

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/k3-engine/card.svg" alt="k3-engine Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-engine/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-engine/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-engine/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-engine/benchmark.svg" alt="Benchmark"/></a>
</p>

---

Low-level compute runtime for the SpacemiT K3 AI platform (16 RISC-V cores, 32GB Unified RAM).

## Overview
`k3-engine` implements the high-performance execution engine targeted at `riscv64`:
- **Compute Kernels**: Unified VLA RVV 1.0 loops with A100 IME acceleration.
- **Quantization**: Qwen/LLaMA Q4_K_M first.
- **Synchronization**: Thread group scheduling and diagnostic spikes (`spin`, `futex`, `rayon`).
- **Memory**: Paged KV cache allocator and eviction engine.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/runtime/](https://github.com/LMA1980/model-runner/tree/main/specs/runtime)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
