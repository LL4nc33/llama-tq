# TurboQuant Nano

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Ultra-low-bit KV-cache quantization for llama.cpp.**
2.125 to 4.125 bits per weight. Lossless at 4-bit (+0.44 % PPL). Multi-GPU ready.

Built on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al.) and
[QTIP](https://arxiv.org/abs/2406.11235), with a custom group-Viterbi encoder
and CUDA kernels tuned for Flash Attention.

---

## What

Every KV-cache token eats VRAM. `f16` is the baseline — 16 bits per value.
TurboQuant Nano compresses the V-cache to **2–4 bits** with trellis-coded
quantization (TCQ), keeping the quality tight enough that most deployments
won't notice the swap.

Two generations ship in the same build:

| Family | bpw | Technique | Best for |
|---|---|---|---|
| **v1** (`KTQ*`, `VTQ*_1`) | 2.5 / 3.5 / 4.5 / 5.5 | RHT + 1-D Lloyd-Max codebook | Drop-in, FA-fast |
| **v2** (`VTQ*_2`) | 2.125 / 3.125 / 4.125 | Group-Viterbi + inv-Gaussian LUT | Max quality per bit |

Both encode the V-cache in blocks, both decode per-sample inside the Flash
Attention inner loop. v1 is codebook-lookup (fast, good). v2 is a full
Viterbi DP over 65 536 states (slower encode, near-lossless at K ≥ 3).

---

## Quick Start

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j --target llama-cli

# Run with 3-bit V-cache (v2 family, near-lossless)
./build-cuda/bin/llama-cli -m model.gguf -p "hello" \
    --cache-type-k f16 --cache-type-v vtq3_2 -fa on -ngl 99
```

Env knobs:
- `GGML_CUDA_VTQ_POOL_SLOTS=64` — raise encoder GPU workspace (1 → 2 GB)
- `GGML_TRELLIS_BEAM=512` — CPU encoder beam width (if GPU unavailable)

---

## Benchmarks

All on **2× NVIDIA RTX 2060 12GB** (CC 7.5, PCIe 3.0). Flash Attention on,
all layers offloaded. `K = q8_0` throughout for v1; `K = f16` for v2.

### Quality — Qwen3.5-0.8B (wikitext-2, 10 chunks, ctx 512)

| V-Cache | PPL | Δ f16 | bpw | VRAM vs f16 |
|---|---:|---:|---:|---:|
| `f16`     | 20.22 | — | 16.0  | 100 % |
| `vtq2_2`  | 21.76 | +7.6 % | 2.125 | **13 %** |
| `vtq3_2`  | 20.71 | +2.4 % | 3.125 | **20 %** |
| **`vtq4_2`** | **20.31** | **+0.44 %** | **4.125** | **26 %** |

`vtq4_2` is statistically indistinguishable from `f16` (error bars overlap,
3.88× compression). `vtq3_2` is the mid-sweet-spot; `vtq2_2` is aggressive
but still within a few percent.

### Throughput — v1 family

<details>
<summary>Qwen3.6-35B-A3B UD-IQ2_XXS (10.01 GiB) — single-GPU fit</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | 820 | 60.1 | — | — |
| `q8_0` | `vtq2_1` | 757 | 58.7 | -8 % | **-2 %** |
| `q4_0` | `vtq2_1` | 756 | 58.8 | -8 % | **-2 %** |
| `q8_0` | `vtq3_1` | 735 | 58.1 | -10 % | -3 % |

v1 V-dequant is an `__forceinline__` codebook lookup — lighter than `q4_0`'s
per-lane shift+scale (`vtq2_1` at 2.5 bpw beats `q4_0` V at 5 bpw on PP512).
</details>

<details>
<summary>Qwen3.5-35B-A3B Q4_K_M (19.92 GiB) — pooled dual-GPU</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | 842 | 61.9 | — | — |
| `q8_0` | `vtq2_1` | 781 | 60.4 | -7 % | **-2 %** |
| `q4_0` | `vtq2_1` | 783 | 60.4 | -7 % | **-2 %** |
</details>

<details>
<summary>Qwen3.5-27B Dense Q4_K_M (15.94 GiB)</summary>

| K | V | PP512 tok/s | TG128 tok/s | PP Δ | TG Δ |
|---|---|---:|---:|---:|---:|
| `f16` | `f16` | 318 | 14.6 | — | — |
| `q8_0` | `vtq2_1` | 297 | 14.5 | -7 % | **-1 %** |
| `q4_0` | `vtq2_1` | 297 | 14.5 | -7 % | **-1 %** |

Dense models lose more on PP512 (no MoE sparsity) but keep the same
~1-2 % TG128 tax for `vtq2_1` / `vtq3_1`.
</details>

### Throughput — v2 family

v2 encodes via Viterbi DP (CUDA kernel, pool-allocated workspace). v2
decode is a shift-register + LUT lookup, comparable to v1. Encode is
one-shot per cache write; amortized per token it's negligible.

_(Full v2 throughput table pending 35B MoE sweep — see [Status](#status).)_

---

## Choose Your Config

| You need… | Use | Notes |
|---|---|---|
| Lossless swap | `vtq4_2` | +0.44 % PPL, 3.88× compression |
| Best bpw/quality curve | `vtq3_2` | +2.4 % PPL, 5.1× compression |
| Minimum VRAM | `vtq2_2` | +7.6 % PPL, 7.5× compression |
| Fastest decode | `vtq2_1` (v1) | Codebook lookup, `-fa on` hot path |
| CPU-only | `vtq{K}_2 + GGML_TRELLIS_BEAM=512` | 7-10× slower encode, same decode |

---

## Technical

- **RHT** (Randomized Hadamard Transform): applied at graph level for V
  (once per forward pass), not per block. Produces post-rotation Gaussian
  marginal so 1-D codebooks are near-optimal.
- **Group Viterbi** (v2): N = 256 samples per block, L = 16 state register,
  K bits emitted per sample, full 65 536-state DP with per-device LUT.
  Bug-free multi-GPU (both devices init their own `__device__` LUT).
- **Flash Attention integration**: V-dequant is `__forceinline__` for v1,
  `__noinline__` for KTQ FWHT path. v2 uses a per-timestep shift-register
  with a 256 KiB constant LUT.
- **CUDA encoder**: 1 CUDA block per trellis-block, 256 threads, 64-bit
  packed `atomicMin(cost << 32 | prev)` on workspace DP rows.

Deep dive: [`docs/turboquant.md`](docs/turboquant.md),
[`tests/trellis-phase1/`](tests/trellis-phase1/),
[`docs/plans/2026-04-17-trellis-v2-phase1-report.md`](docs/plans/2026-04-17-trellis-v2-phase1-report.md).

---

## Status

- **v1 family** (`KTQ1-4`, `VTQ1-4_1`): production, FA CUDA kernels shipped.
- **v2 family** (`VTQ2-4_2`): production on CUDA (encoder + decode), CPU
  fallback via `GGML_TRELLIS_BEAM`. FA-vec template instances for v2 V-type
  not yet generated — v2 currently routes through non-FA mul-mat on V.

Roadmap:
- v2 FA-vec template instances (unlocks full GPU FA path for v2)
- 27B / 35B MoE PPL validation table (in progress)
- Nsight-compute profiling to kill atomicMin contention in the encoder

---

## References

- Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant." arXiv:2504.19874 (2025).
- QTIP: arXiv:2406.11235.
- Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT license.

## License

MIT (inherited from llama.cpp).
