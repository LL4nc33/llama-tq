# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Asymmetric KV-Cache Quantization for llama.cpp** -- separate K and V compression paths, each optimized for their role in Flash Attention.

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) based on [PolarQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

> **Key findings:**
> - VTQ V-cache at 2.5 bpw is **faster than q4_0** (+40% prompt throughput) while using less memory
> - VTQ3_1 (4.0 bpw) is **perplexity-neutral** on both Qwen3.5 and Qwen3.6 (+1%)
> - `q8_0` K + `vtq2_1` V: only **+6-7% PPL** at 65% VRAM savings

---

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Recommended: high-quality K + compressed V
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq2_1 \
    -fa on -ngl 99
```

## Recommended Configurations

| Config | K | V | Avg bpw | PPL Impact | Best For |
|--------|---|---|:---:|---|---|
| **Safe** | `q8_0` | `vtq3_1` | 6.25 | +1% | production, quality-first |
| **Balanced** | `q4_0` | `vtq3_1` | 4.25 | ~1% | general use |
| **Recommended** | `q8_0` | `vtq2_1` | 5.5 | +6-7% | best VRAM/quality tradeoff |
| Aggressive | `q4_0` | `vtq2_1` | 3.5 | +8% | long context, VRAM-limited |

---

## Benchmarks

All benchmarks on **2x NVIDIA RTX 2060 12GB** (CC 7.5, PCIe 3.0), Flash Attention on, all layers offloaded.

### Throughput (llama-bench, PP512/TG128)

#### Qwen3.5-35B-A3B (IQ2_XS, 10.16 GiB)

| K-Cache | V-Cache | PP512 tok/s | TG128 tok/s | PP vs f16 | TG vs f16 |
|---------|---------|:---:|:---:|:---:|:---:|
| f16 | f16 | **731** | **58.8** | baseline | baseline |
| q8_0 | vtq2_1 | 684 | 57.5 | -6% | **-2%** |
| q4_0 | vtq2_1 | 682 | 57.4 | -7% | **-2%** |
| q8_0 | vtq3_1 | 664 | 56.9 | -9% | -3% |
| q4_0 | f16 | 531 | 49.1 | -27% | -17% |
| q8_0 | q4_0 | 485 | 50.6 | -34% | -14% |
| f16 | q4_0 | 483 | 49.3 | -34% | -16% |

#### Qwen3.6-35B-A3B (UD-IQ2_XXS, 10.01 GiB)

| K-Cache | V-Cache | PP512 tok/s | TG128 tok/s | PP vs f16 | TG vs f16 |
|---------|---------|:---:|:---:|:---:|:---:|
| f16 | f16 | **820** | **60.1** | baseline | baseline |
| q8_0 | vtq2_1 | 757 | 58.7 | -8% | **-2%** |
| q4_0 | vtq2_1 | 756 | 58.8 | -8% | **-2%** |
| q8_0 | vtq3_1 | 735 | 58.1 | -10% | -3% |
| f16 | vtq2_1 | 760 | 59.5 | -7% | **-1%** |
| f16 | q4_0 | 508 | 49.3 | -38% | -18% |
| q4_0 | f16 | 571 | 48.9 | -30% | -19% |

**VTQ is faster than q4_0 for V-cache.** The `__forceinline__` codebook dequant (~8 registers) outperforms q4_0's shift+scale dequant in the FA inner loop. At the same time, VTQ uses less memory (2.5 bpw vs 4.5 bpw).

### Perplexity (wikitext-2, 512 ctx, 3 chunks)

#### Qwen3.5-35B-A3B (IQ2_XS)

| K-Cache | V-Cache | KV bpw | PPL | vs baseline |
|---------|---------|:---:|:---:|:---:|
| f16 | f16 | 16.0 | **6.598** | -- |
| q8_0 | q8_0 | 8.5 | 6.600 | +0.03% |
| q4_0 | q4_0 | 4.5 | 6.619 | +0.32% |
| f16 | vtq3_1 | 10.0 | 6.716 | +1.8% |
| q8_0 | vtq2_1 | 5.5 | **7.072** | **+7.2%** |
| f16 | vtq2_1 | 9.3 | 7.115 | +7.8% |
| ktq2_1 | f16 | 9.8 | 7.246 | +9.8% |

#### Qwen3.6-35B-A3B (UD-IQ2_XXS)

| K-Cache | V-Cache | KV bpw | PPL | vs baseline |
|---------|---------|:---:|:---:|:---:|
| f16 | f16 | 16.0 | **5.967** | -- |
| q8_0 | q8_0 | 8.5 | 6.006 | +0.65% |
| q4_0 | q4_0 | 4.5 | 6.001 | +0.57% |
| f16 | vtq3_1 | 10.0 | 6.030 | **+1.05%** |
| q8_0 | vtq2_1 | 5.5 | **6.361** | **+6.6%** |
| f16 | vtq2_1 | 9.3 | 6.378 | +6.9% |

VTQ3_1 is **perplexity-neutral** on both models. VTQ2_1 with q8_0 K costs only +6-7% PPL.

### KV-Cache Memory (4096 ctx)

| Config | KV Size | Savings vs f16 |
|--------|:---:|:---:|
| f16 / f16 | 40.0 MiB | -- |
| q8_0 / vtq2_1 | 13.8 MiB | **65%** |
| q4_0 / vtq3_1 | 10.6 MiB | **73%** |
| q4_0 / vtq2_1 | 8.7 MiB | **78%** |
| ktq2_1 / vtq2_1 | 7.5 MiB | **81%** |

### Comparison with Other Approaches

PPL delta vs f16 baseline (lower is better). Different hardware, so absolute tok/s are not comparable — **relative deltas are**.

| Approach | Type | bpw | PPL Delta (Qwen ~35B) | Decode Delta | Hardware |
|----------|------|:---:|:---:|:---:|---|
| **llama-tq vtq3_1** | V-only | 4.0 | **+1.0%** | **-2%** | 2x RTX 2060 |
| TheTom turbo4 | K+V sym | 4.25 | +0.23% | -7% | M5 Max |
| **llama-tq vtq2_1** | V-only | 2.5 | **+6.6%** | **-2%** | 2x RTX 2060 |
| TheTom turbo3 | K+V sym | 3.5 | +1.06% | -10% | M5 Max |
| TheTom turbo2 | K+V sym | 2.5 | +6.48% | -22%* | M5 Max |
| q4_0 | K+V sym | 4.5 | +0.3-0.6% | -16% | 2x RTX 2060 |
| **llama-tq q8_0+vtq2_1** | asymmetric | 5.5 | **+6.6%** | **-2%** | 2x RTX 2060 |
| TheTom q8_0+turbo3 | asymmetric | ~5.5 | +2.0% | ~-10% | M5 Max |

*TheTom turbo2 decode varies: -22% on MoE short context, but **+33.9%** on M1 Max with turbo4.

**Key differences:**
- **llama-tq VTQ** separates K and V quantization paths — V dequant is `__forceinline__` (~8 registers) vs TheTom's shared TQ dequant
- **VTQ decode overhead is minimal** (-2% TG) because the V-dequant is a trivial codebook lookup. TheTom's symmetric approach has higher decode cost (-7 to -10%) because both K and V share the full TQ dequant path
- **PPL at 2-bit**: Both achieve ~+6.5% at 2.5 bpw V-cache, but llama-tq uses D\*H\*D rotation (Laplace codebook) while TheTom uses standard RHT
- **TheTom has better PPL at 3-4 bit** because symmetric K+V quantization avoids the K-V error amplification issue

---

## Available Cache Types

<details>
<summary><strong>KTQ (K-Cache TurboQuant)</strong></summary>

Per-block Randomized Hadamard Transform + Lloyd-Max codebook. FA kernel uses Hadamard-domain dot product (FWHT on Q, not inverse-FWHT on K).

| Type | bpw | Block | Notes |
|------|:---:|:---:|---|
| `ktq1_1` | 2.5 | 10B | 1-bit, extreme compression |
| `ktq2_1` | 3.5 | 14B | 2-bit, good quality |
| `ktq3_1` | 4.5 | 18B | 3-bit, near-lossless |
| `ktq4_1` | 5.5 | 22B | 4-bit, best KTQ quality |

**Note:** Combining KTQ K + VTQ V at low bit-widths causes super-additive PPL degradation through softmax error amplification. Use `q8_0`/`q4_0` for K when using VTQ for V.

</details>

<details>
<summary><strong>VTQ (V-Cache TurboQuant)</strong></summary>

Graph-level D\*H\*D rotation (randomized Hadamard). FA dequant is `codebook[idx] * scale` -- `__forceinline__`, ~8 registers, no spilling. Laplace-optimized codebooks for 1-2 bit.

| Type | bpw | Block | Notes |
|------|:---:|:---:|---|
| `vtq1_1` | 1.5 | 6B | 1-bit, maximum compression |
| `vtq2_1` | 2.5 | 10B | 2-bit, Laplace codebook, **recommended** |
| `vtq3_1` | 4.0 | 16B | 3-bit, **near-lossless** |
| `vtq4_1` | 4.5 | 18B | 4-bit, near-lossless |

</details>

---

## How It Works

### The Problem

PolarQuant's V-dequant requires a 32-element FWHT butterfly (~40 float registers) inside the FA kernel's inner loop. This causes register spilling and corruption on CUDA.

### The Solution: Split K and V

```
KTQ K-dequant (FA inner loop):     VTQ V-dequant (FA inner loop):
  float buf[32];                      float val = codebook[idx] * scale;
  fwht_32_serial(buf);                // done — that's it
  sign_flip(buf, sb[4]);
  // ~40 registers, __noinline__      // ~8 registers, __forceinline__
```

**KTQ** keeps per-block RHT for K (the Hadamard-domain dot product avoids the dequant entirely).
**VTQ** replaces per-block RHT with a fixed D\*H\*D rotation at graph level. The D\*H\*D randomization makes coordinates i.i.d., critical for 2-bit codebook quality.

---

## Build

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# For all KTQ x VTQ combinations:
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

CUDA CC 6.1+. CPU fallback available.

## Known Limitations

- **KTQ + VTQ at low bits**: Combined K+V quantization at 2-bit has super-additive PPL degradation. Use standard types (`q8_0`, `q4_0`) for K with VTQ V.
- **Metal/CPU**: CUDA-optimized. CPU fallback exists but is slow.
- **MoE models**: Models with few attention layers (10/40) amplify per-layer quantization impact.

## Documentation

| Doc | Content |
|-----|---------|
| [docs/turboquant.md](docs/turboquant.md) | Architecture, CUDA kernels, codebooks |
| [docs/plans/2026-04-16-vtq-design.md](docs/plans/2026-04-16-vtq-design.md) | VTQ design spec, math proofs |

## Related Projects

| Project | Focus |
|---------|-------|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Community TQ, Metal + CUDA, extensive benchmarks |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ (Trellis-Coded Quantization) |
| [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | TurboQuant community thread |

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Google Research, ICLR 2026 -- [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT (inherited from [llama.cpp](https://github.com/ggml-org/llama.cpp))
