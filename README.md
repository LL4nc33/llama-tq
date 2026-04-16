# llama-tq

**KTQ + VTQ KV-Cache Quantization for llama.cpp** -- PolarQuant-based, up to 84% VRAM savings

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) implementing asymmetric KV-cache quantization based on [TurboQuant/PolarQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

Two separate type families optimized for their role:
- **KTQ** (K-Cache TurboQuant) -- per-block RHT + Hadamard-domain dot product in FA kernel
- **VTQ** (V-Cache TurboQuant) -- graph-level D\*H\*D rotation, lightweight `__forceinline__` dequant (~8 registers)

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Recommended: q8_0 K + vtq2_1 V (near-lossless, 65% VRAM savings)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq2_1 \
    -fa on -ngl 99

# Balanced: q4_0 K + vtq3_1 V (73% VRAM savings)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q4_0 --cache-type-v vtq3_1 \
    -fa on -ngl 99
```

## Available Types

### KTQ (K-Cache) -- Hadamard-domain dot product, per-block RHT

| Type | bpw | Block Size | Description |
|------|:---:|:---:|---|
| `ktq1_1` | 2.5 | 10B | 1-bit PolarQuant, extreme compression |
| `ktq2_1` | 3.5 | 14B | 2-bit PolarQuant, good quality |
| `ktq3_1` | 4.5 | 18B | 3-bit PolarQuant, near-lossless |
| `ktq4_1` | 5.5 | 22B | 4-bit PolarQuant, best quality |

### VTQ (V-Cache) -- Graph-level D\*H\*D rotation, no FWHT in FA kernel

| Type | bpw | Block Size | Description |
|------|:---:|:---:|---|
| `vtq1_1` | 1.5 | 6B | 1-bit, maximum compression |
| `vtq2_1` | 2.5 | 10B | 2-bit, Laplace-optimized codebook |
| `vtq3_1` | 4.0 | 16B | 3-bit, near-lossless |
| `vtq4_1` | 4.5 | 18B | 4-bit, near-lossless |

## Perplexity Benchmarks

Qwen3.5-35B-A3B IQ2_XS, wikitext-2, 512 ctx -- 2x RTX 2060 12GB:

| K-Cache | V-Cache | Avg bpw | PPL | Delta | Status |
|---------|---------|:---:|:---:|:---:|---|
| f16 | f16 | 16.0 | **6.60** | baseline | -- |
| q8_0 | q8_0 | 8.5 | 6.60 | +0.04% | lossless |
| q4_0 | q4_0 | 4.5 | 6.62 | +0.3% | near-lossless |
| f16 | vtq3_1 | 10.0 | 6.59 | -0.1% | **near-lossless** |
| f16 | vtq4_1 | 10.3 | 6.64 | +0.6% | near-lossless |
| **q8_0** | **vtq2_1** | **5.5** | **6.99** | **+5.9%** | **recommended** |
| f16 | vtq2_1 | 9.3 | 6.97 | +5.6% | good |
| ktq2_1 | f16 | 9.8 | 6.77 | +2.6% | good |

## Throughput Benchmarks

llama-bench, Qwen3.5-35B-A3B IQ2_XS, 2x RTX 2060 12GB:

| K-Cache | V-Cache | PP512 (tok/s) | TG128 (tok/s) | KV VRAM |
|---------|---------|:---:|:---:|:---:|
| f16 | f16 | **730** | **58.9** | 100% |
| f16 | vtq2_1 | 691 (-5%) | 58.1 (-1%) | 58% |
| f16 | vtq1_1 | 693 (-5%) | 58.0 (-2%) | 55% |
| ktq2_1 | vtq2_1 | 567 (-22%) | 55.0 (-7%) | 19% |
| ktq3_1 | vtq2_1 | 556 (-24%) | 54.3 (-8%) | 22% |

VTQ V-dequant overhead: **<2% TG** (the `__forceinline__` codebook lookup is nearly free).

## Key Technical Innovations

### KTQ: Hadamard-Domain K Dot Product
Instead of inverse-FWHT on K (per-block, expensive), FWHT is applied to Q once and dotted directly against codebook values. 39% fewer warp shuffles per vec_dot call.

### VTQ: Graph-Level Rotation + Lightweight FA Dequant
V values are pre-rotated via `self_v_rot` (D\*H\*D randomized Hadamard) before cache write. The FA kernel dequant reduces to `codebook[idx] * scale` -- only ~8 float registers vs ~40 for KTQ. Post-FA inverse rotation via existing `ggml_mul_mat_aux`.

### D\*H\*D Randomized Rotation
Fixed Hadamard H produces non-i.i.d. coordinates (DC component captures mean). D\*H\*D with deterministic per-head diagonal signs distributes the DC component uniformly. This reduced VTQ2_1 PPL from 24.19 to 6.97.

### Separate Codebooks (PolarQuant)
KTQ uses Beta(15.5,15.5)-optimal Lloyd-Max codebooks (for per-block RHT distribution).
VTQ uses Laplace-optimal codebooks (for D\*H\*D fixed rotation distribution).
Shared scale factor: `1/sqrt(32)`.

## Recommended Configurations

| Use Case | K-Cache | V-Cache | Avg bpw | PPL Impact |
|----------|---------|---------|:---:|---|
| **Production (quality)** | q8_0 | vtq2_1 | 5.5 | +5.9% |
| **Production (balanced)** | q4_0 | vtq3_1 | 4.25 | ~+1% |
| **Production (safe)** | q8_0 | vtq3_1 | 6.25 | <0.5% |
| Maximum compression | ktq2_1 | vtq3_1 | 3.75 | ~+5% |
| Long context (400K+) | ktq2_1 | vtq2_1 | 3.0 | higher |

**Note:** Combining KTQ K + VTQ V at low bit-widths (both <= 2-bit) causes super-additive PPL degradation through softmax error amplification. Use q8_0/q4_0 for K when using vtq2_1/vtq1_1 for V.

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/turboquant.md](docs/turboquant.md) | Technical deep-dive: KTQ/VTQ architecture, CUDA kernels, codebooks |
| [docs/plans/2026-04-16-vtq-design.md](docs/plans/2026-04-16-vtq-design.md) | VTQ design spec and mathematical proofs |

## Related Projects

| Project | Focus | Hardware |
|---------|-------|----------|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Community TQ fork | Metal + CUDA + CPU |
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | Trellis-constrained TQ | CUDA |
| [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | Community coordination | -- |

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Google Research, ICLR 2026 -- https://arxiv.org/abs/2504.19874

## License

MIT (inherited from llama.cpp)
