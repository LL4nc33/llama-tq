# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Asymmetric KV-Cache Quantization for llama.cpp** -- separate K and V compression paths, each optimized for their role in Flash Attention.

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) based on [PolarQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

> **Key finding:** V-cache can be quantized to 2.5 bpw with <6% PPL impact using graph-level rotation + lightweight codebook dequant. K-cache compression works independently via Hadamard-domain dot product.

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq2_1 \
    -fa on -ngl 99
```

## Recommended Configurations

Tested on Qwen3.5-35B-A3B IQ2_XS, wikitext-2:

| Config | K | V | Avg bpw | PPL Impact | Use Case |
|--------|---|---|:---:|---|---|
| **Safe** | `q8_0` | `vtq3_1` | 6.25 | <0.5% | production, quality-first |
| **Balanced** | `q4_0` | `vtq3_1` | 4.25 | ~1% | production, good compression |
| **Recommended** | `q8_0` | `vtq2_1` | 5.5 | +5.9% | best VRAM/quality tradeoff |
| Aggressive | `ktq2_1` | `vtq3_1` | 3.75 | ~5% | long context, VRAM-limited |

<details>
<summary><strong>All available cache types</strong></summary>

### KTQ (K-Cache TurboQuant)

Per-block Randomized Hadamard Transform + Lloyd-Max codebook. FA kernel uses Hadamard-domain dot product (FWHT applied to Q, not inverse-FWHT to K).

| Type | bpw | Block | Notes |
|------|:---:|:---:|---|
| `ktq1_1` | 2.5 | 10B | 1-bit, extreme compression |
| `ktq2_1` | 3.5 | 14B | 2-bit, good quality |
| `ktq3_1` | 4.5 | 18B | 3-bit, near-lossless |
| `ktq4_1` | 5.5 | 22B | 4-bit, best KTQ quality |

### VTQ (V-Cache TurboQuant)

Graph-level D\*H\*D rotation (no per-block FWHT). FA kernel dequant is `codebook[idx] * scale` -- `__forceinline__`, ~8 registers, no spilling.

| Type | bpw | Block | Notes |
|------|:---:|:---:|---|
| `vtq1_1` | 1.5 | 6B | 1-bit, maximum compression |
| `vtq2_1` | 2.5 | 10B | 2-bit, Laplace-optimized codebook |
| `vtq3_1` | 4.0 | 16B | 3-bit, **near-lossless** |
| `vtq4_1` | 4.5 | 18B | 4-bit, near-lossless |

</details>

## Benchmarks

### Perplexity (wikitext-2, 512 ctx)

Model: Qwen3.5-35B-A3B IQ2_XS -- 2x RTX 2060 12GB

| K | V | PPL | vs f16 baseline |
|---|---|:---:|:---:|
| f16 | f16 | 6.60 | -- |
| q8_0 | q8_0 | 6.60 | +0.04% |
| q4_0 | q4_0 | 6.62 | +0.3% |
| f16 | vtq3_1 | 6.59 | **-0.1%** |
| f16 | vtq4_1 | 6.64 | +0.6% |
| **q8_0** | **vtq2_1** | **6.99** | **+5.9%** |
| ktq2_1 | f16 | 6.77 | +2.6% |

VTQ3_1 and VTQ4_1 are **perplexity-neutral** -- within noise of f16 baseline.

### Throughput (`llama-bench`, PP512/TG128)

| K | V | PP tok/s | TG tok/s | KV VRAM vs f16 |
|---|---|:---:|:---:|:---:|
| f16 | f16 | 730 | 58.9 | 100% |
| f16 | vtq2_1 | 691 | 58.1 | 58% |
| ktq2_1 | vtq2_1 | 567 | 55.0 | **19%** |
| ktq3_1 | vtq2_1 | 556 | 54.3 | 22% |

VTQ V-dequant overhead: **<2% decode**. The `__forceinline__` codebook lookup is nearly free.

<details>
<summary><strong>KV-Cache memory examples</strong></summary>

4096 context, Qwen3.5-35B-A3B (10 attention layers):

| Config | KV Size | Savings |
|--------|:---:|:---:|
| f16 / f16 | 40.0 MiB | -- |
| q8_0 / vtq2_1 | 13.8 MiB | 65% |
| ktq2_1 / vtq2_1 | 7.5 MiB | 81% |
| ktq2_1 / vtq1_1 | 6.3 MiB | 84% |

</details>

## How It Works

### The Problem

Standard KV-cache quantization (q4_0, q8_0) uses uniform scalar quantization. PolarQuant improves on this using Hadamard rotation + Lloyd-Max codebooks, but the original TQ implementation requires a 32-element FWHT butterfly transform (~40 float registers) inside the FA kernel's V-dequant inner loop. This causes **register spilling and corruption** on CUDA -- the V-dequant produces garbage output.

### The Solution: Split K and V

**KTQ** for K-cache: Keep the per-block RHT (random signs + FWHT). The K dot-product trick applies FWHT to Q instead of inverse-FWHT to K, avoiding the register pressure problem entirely.

**VTQ** for V-cache: Replace per-block RHT with a **fixed D\*H\*D rotation** applied at graph level (before cache write, after FA). The V-dequant in the FA kernel reduces to a trivial codebook lookup:

```
// KTQ V-dequant: ~40 registers, __noinline__, register spilling
float buf[32];
fwht_32_serial(buf);           // 160 FMA butterfly ops
sign_flip(buf, sb[4]);         // per-block random signs
// ... extract ne elements

// VTQ V-dequant: ~8 registers, __forceinline__, no spilling
float val = codebook[idx] * scale;  // that's it
```

### D\*H\*D Rotation

Fixed Hadamard H produces non-i.i.d. coordinates (the DC row captures the mean, creating systematic outliers). D\*H\*D applies deterministic diagonal sign flips that distribute the mean uniformly across all coordinates. This is critical for 2-bit quality -- without D\*H\*D, VTQ2_1 PPL is 24+ (unusable). With D\*H\*D, it drops to 6.97.

D\*H\*D is self-transpose (`(DHD)^T = D^T H^T D^T = DHD`), so the existing `self_v_rot` infrastructure works unchanged for both pre-rotation and post-FA inverse.

### Shared Codebooks (PolarQuant)

Both families use Lloyd-Max codebooks optimized for their respective distributions:
- **KTQ**: Beta(15.5, 15.5) marginal (from per-block RHT)
- **VTQ**: Laplace marginal (from fixed D\*H\*D rotation, heavier tails)

The 3-bit and 4-bit codebooks are shared (identical distributions at higher bit-widths). Only 2-bit and 1-bit differ.

## Build

```bash
# CUDA (recommended)
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# For all KTQ x VTQ type combinations in FA:
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

Requires CUDA toolkit (CC 6.1+). CPU fallback available for all types.

## Known Limitations

- **KTQ + VTQ at low bit-widths**: Combining KTQ K-quant with VTQ V-quant at 2-bit causes super-additive PPL degradation through softmax error amplification. Use `q8_0` or `q4_0` for K when using `vtq2_1`/`vtq1_1` for V.
- **Metal/CPU**: KTQ/VTQ types are CUDA-optimized. CPU fallback exists but is slow. No Metal implementation yet.
- **MoE models**: Models with few attention layers (e.g., Qwen3.5-35B-A3B has 10/40) amplify quantization impact per-layer.

## Documentation

| Doc | Content |
|-----|---------|
| [docs/turboquant.md](docs/turboquant.md) | Technical deep-dive: architecture, CUDA kernels, codebooks |
| [docs/plans/2026-04-16-vtq-design.md](docs/plans/2026-04-16-vtq-design.md) | VTQ design spec, math proofs, performance analysis |

## Related Projects

| Project | Focus |
|---------|-------|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Community TQ workspace, Metal + CUDA, extensive benchmarks |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ (Trellis-Coded Quantization) fork |
| [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | TurboQuant community thread |

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Google Research, ICLR 2026 -- [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT (inherited from [llama.cpp](https://github.com/ggml-org/llama.cpp))
