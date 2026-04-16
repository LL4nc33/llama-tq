# TurboQuant v7 -- KTQ/VTQ KV Cache Quantization for CUDA

## Overview

TurboQuant implements KV cache quantization for GGML, inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., arXiv preprint April 2025). It compresses the KV cache via Randomized Hadamard Transform + Lloyd-Max scalar quantization, enabling significantly longer context windows on limited VRAM.

Two type families exist, split by cache role:

- **KTQ** (K-Cache TurboQuant) -- Hadamard rotation (RHT) with per-block sign bits. The FA kernel uses a Hadamard-domain dot product (FWHT on Q, not inverse-FWHT on K), eliminating gather shuffles and branch divergence. 39% fewer warp shuffles per vec_dot call.
- **VTQ** (V-Cache TurboQuant) -- codebook-only quantization without FWHT or sign bits. V values are pre-rotated via `self_v_rot` (graph-level Hadamard). FA V-dequant is `codebook[idx] * scale` -- `__forceinline__`, ~8 registers (vs ~40 for KTQ). Inverse rotation applied as a single post-FA matmul.

The KTQ/VTQ split was motivated by a V-dequant register spilling bug in the FA kernel: KTQ's full dequant path (32-element FWHT butterfly + sign bits) requires ~40 float registers, forcing `__noinline__` and causing LMEM spills that corrupt the FA accumulator state. VTQ eliminates this by moving the rotation out of the FA hot loop.

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Maximum compression (2.5 bpw avg)
./build/bin/llama-server -m model.gguf \
    --cache-type-k ktq2_1 --cache-type-v vtq1_1 -fa on -ngl 99

# Balanced (3.5 bpw avg)
./build/bin/llama-server -m model.gguf \
    --cache-type-k ktq3_1 --cache-type-v vtq2_1 -fa on -ngl 99

# Quality (4.75 bpw avg)
./build/bin/llama-server -m model.gguf \
    --cache-type-k ktq4_1 --cache-type-v vtq3_1 -fa on -ngl 99
```

## Available Types

### KTQ (K-Cache) -- Hadamard rotation (RHT) with sign bits

| Type | bpw | Block Size | vs f16 | Use Case |
|------|-----|------------|--------|----------|
| `ktq1_1` | 2.5 | 10 bytes | **-84%** | Extreme K compression |
| `ktq2_1` | 3.5 | 14 bytes | -78% | Maximum compression, long context |
| `ktq3_1` | 4.5 | 18 bytes | -72% | Balanced quality/compression |
| `ktq4_1` | 5.5 | 22 bytes | -66% | Best KTQ quality |

### VTQ (V-Cache) -- no FWHT/sign bits, pre-rotated via self_v_rot

| Type | bpw | Block Size | vs f16 | Use Case |
|------|-----|------------|--------|----------|
| `vtq1_1` | 1.5 | 6 bytes | **-91%** | Extreme V compression |
| `vtq2_1` | 2.5 | 10 bytes | -84% | Best V compression with 2-bit quality |
| `vtq3_1` | 4.0 | 16 bytes | -75% | Balanced V quality |
| `vtq4_1` | 4.5 | 18 bytes | -72% | Best VTQ quality |

### Combined Reference

| Type | Family | bpw | Block | d (norm) | qs (indices) | sb (signs) |
|------|--------|-----|-------|----------|-------------|------------|
| `ktq1_1` | KTQ | 2.5 | 10B | 2B | 4B (1-bit) | 4B |
| `ktq2_1` | KTQ | 3.5 | 14B | 2B | 8B (2-bit) | 4B |
| `ktq3_1` | KTQ | 4.5 | 18B | 2B | 12B (3-bit) | 4B |
| `ktq4_1` | KTQ | 5.5 | 22B | 2B | 16B (4-bit) | 4B |
| `vtq1_1` | VTQ | 1.5 | 6B | 2B | 4B (1-bit) | -- |
| `vtq2_1` | VTQ | 2.5 | 10B | 2B | 8B (2-bit) | -- |
| `vtq3_1` | VTQ | 4.0 | 16B | 2B | 14B (3-bit) | -- |
| `vtq4_1` | VTQ | 4.5 | 18B | 2B | 16B (4-bit) | -- |

VTQ saves 4 bytes per block (the `sb[4]` sign bits) because the rotation is fixed and position-independent. The V data arrives pre-rotated; no per-block state is needed.

## Recommended Configurations

| Use Case | K-Cache | V-Cache | Avg bpw | Notes |
|----------|---------|---------|---------|-------|
| Maximum compression | `ktq2_1` (3.5) | `vtq1_1` (1.5) | **2.5** | Best VRAM, extreme V compression |
| Balanced | `ktq3_1` (4.5) | `vtq2_1` (2.5) | **3.5** | Good quality, long context |
| Quality | `ktq4_1` (5.5) | `vtq3_1` (4.0) | **4.75** | Better PPL than q4_0/q4_0 |
| Conservative | `q8_0` (8.5) | `vtq2_1` (2.5) | **5.5** | Best K quality + max V compression |

## Memory Savings (Ministral 3B, 26 layers, 8 KV-heads)

| Context | f16 | q8_0 | q4_0 | ktq3_1+vtq2_1 | ktq2_1+vtq1_1 |
|---------|------|------|------|----------------|----------------|
| 32K | 1.6 GB | 0.8 GB | 0.5 GB | 0.3 GB | 0.2 GB |
| 128K | 6.4 GB | 3.4 GB | 1.8 GB | 1.4 GB | 1.0 GB |
| 384K | 19.2 GB | 10.2 GB | 5.4 GB | 4.2 GB | 3.0 GB |

## Benchmarks

### CC 7.5 (12 GB) -- Qwen3.5-35B-A3B IQ2_XS

| KV Cache | bpw | pp512 (tok/s) | tg32 (tok/s) | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| q4_0/q4_0 | 4.5 | 838 | 69.5 | baseline |
| **ktq2_1/vtq2_1 (v7)** | 3.0 | **634** | **63.4** | **-8.8%** |

v7 TG: **+65% faster** than v6 (63.4 vs 38.4 tok/s).
78% less KV-Cache VRAM vs f16 -- fit 4x more context in the same memory.

### CC 6.1 (6 GB) -- Ministral 3B IQ2_M

| KV Cache | bpw | pp128 (tok/s) | tg32 (tok/s) | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| f16/f16 | 16.0 | 824 | 45.8 | +6.5% |
| q4_0/q4_0 | 4.5 | 798 | 43.0 | baseline |
| **ktq2_1/vtq2_1 (v7)** | 3.0 | **211** | **33.3** | **-22.6%** |

### Example Configurations

**Qwen3.5-35B-A3B IQ2_XS on CC 7.5 (12 GB):**

```bash
llama-server -m Qwen3.5-35B-A3B-IQ2_XS.gguf \
    --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
    -c 400000 -ngl 99 --parallel 2
```

| Detail | Value |
|--------|-------|
| Context | 400K tokens |
| Parallel Slots | 2 |
| KV-Cache (ktq2_1+vtq2_1) | ~1,500 MB (vs ~10,400 MB with q4_0) |
| Total VRAM | 9.0 GB / 12 GB |

**Gemma4 26B on CC 7.5 (12 GB):**

| Metric | q4_0 KV | ktq2_1+vtq2_1 KV | Delta |
|--------|:---:|:---:|:---:|
| KV-Cache (200K ctx) | ~1.7 GB | ~1.1 GB | -600 MB |
| Speed | 36.6 tok/s | 38.6 tok/s | **+2 tok/s** |
| GPU Layers | 16/30 | 17/30 | **+1 layer** |

Saved VRAM enables +1 GPU layer, netting +2 tok/s. Speed AND compression win.

## How It Works

### KTQ Quantization Pipeline (K-Cache)

```
float[32] -> normalize -> random signs -> FWHT -> codebook -> norm correction -> pack
                              |                                      |
                        [store in sb[4]]                    [store corrected d]
```

1. **Normalize:** `x_hat = x / ||x||`
2. **Random Signs:** Apply deterministic +/-1 signs (Philox 6-round PRNG from block-index seed)
3. **FWHT:** Fast Walsh-Hadamard Transform, scaled by `1/sqrt(32)`
4. **Lloyd-Max Quantization:** Nearest centroid from Beta(15.5, 15.5)-approximate codebook (d=32 approximation of the TurboQuant marginal distribution)
5. **Norm Correction:** Reconstruct, measure `||recon||`, store `norm / ||recon||` -- compensates codebook error (~1.2% PPL improvement)
6. **Sign Bits:** Store precomputed signs in `sb[4]` (32 bits = 32 signs)

### KTQ Dequantization (K-Cache, Hadamard-domain dot product)

Instead of inverse-FWHT on K, the FA kernel applies FWHT to Q once per block and dots directly against codebook values:
```
score = norm * dot(FWHT(sign * Q), codebook_values)
```
Mathematically exact (FWHT is orthogonal). No gathers, no branch divergence.

### VTQ Quantization Pipeline (V-Cache)

```
float[d_head] -> graph-level R * v -> [cache write] -> per-block: normalize -> codebook -> norm correction -> pack
                     (self_v_rot)                                                                              |
                                                                                          [NO signs, NO FWHT]
```

1. **Pre-rotation:** `v_rot = R * v` applied at graph level via `self_v_rot` before cache write
2. **Normalize:** `x_hat = block / ||block||`
3. **Lloyd-Max Quantization:** Same shared codebooks as KTQ
4. **Norm Correction:** Same approach as KTQ
5. **Pack:** `d` + `qs` only -- NO `sb[4]` sign bits

### VTQ Dequantization (V-Cache, in FA kernel)

```cuda
v_approx[j] = CB[q_j] * scale    // codebook lookup + scale multiply
                                  // NO FWHT, NO sign flip, ~8 registers
```

Values are in the rotated (Hadamard) domain. After the FA kernel accumulates the weighted sum, a post-FA matmul applies the inverse rotation:
```
VKQ_final = R^T * VKQ_rotated    // graph-level matmul via self_v_rot
```

### Shared Codebooks (PQ_CODEBOOK_*)

Both KTQ and VTQ use the same Lloyd-Max codebooks, optimal for the TurboQuant marginal distribution ≈ Beta(15.5, 15.5) at d=32:

- **1-bit (2 centroids):** `{-0.7979, 0.7979}` (= sqrt(2/pi))
- **2-bit (4 centroids):** `{-1.4896, -0.4514, 0.4514, 1.4896}`
- **3-bit (8 centroids):** `{-2.0719, -1.3150, -0.7453, -0.2424, 0.2424, 0.7453, 1.3150, 2.0719}`
- **4-bit (16 centroids):** `{-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326}`

All scaled by `1/sqrt(32)`. CPU constants: `PQ_CODEBOOK_*BIT`. CUDA constants: `PQ_CUDA_CB_*BIT`.

### Block Layouts

**KTQ (K-Cache) -- includes sb[4] sign bits:**
```
KTQ1_1 (10 bytes, 2.5 bpw):  [d:2B] [qs:4B 1-bit] [sb:4B signs]
KTQ2_1 (14 bytes, 3.5 bpw):  [d:2B] [qs:8B 2-bit] [sb:4B signs]
KTQ3_1 (18 bytes, 4.5 bpw):  [d:2B] [qs:12B 3-bit] [sb:4B signs]
KTQ4_1 (22 bytes, 5.5 bpw):  [d:2B] [qs:16B 4-bit] [sb:4B signs]
```

**VTQ (V-Cache) -- NO sign bits, data pre-rotated:**
```
VTQ1_1 (6 bytes, 1.5 bpw):   [d:2B] [qs:4B 1-bit]
VTQ2_1 (10 bytes, 2.5 bpw):  [d:2B] [qs:8B 2-bit]
VTQ3_1 (16 bytes, 4.0 bpw):  [d:2B] [qs:14B 3-bit]  (12B indices + 2B padding)
VTQ4_1 (18 bytes, 4.5 bpw):  [d:2B] [qs:16B 4-bit]
```

## KTQ vs VTQ V-Dequant Comparison

| Operation | KTQ V-dequant (old) | VTQ V-dequant |
|-----------|---------------------|---------------|
| Load block data | qs + sb + d | qs + d |
| Codebook lookup | 32 lookups | ne lookups (4 typical) |
| Serial FWHT | **32-element butterfly (160 FMA)** | **NONE** |
| Sign flip | 32 branchless mul | **NONE** |
| Scale multiply | 32 mul | ne mul |
| Post-FA rotation | None | R^T matmul (once per layer) |
| Registers | **~40 floats** | **~8 floats** |
| Can `__forceinline__` | No (`__noinline__` required) | **Yes** |
| Register spilling | Severe (corruption root cause) | **None** |

## v7 Optimizations

### 1. Hadamard-Domain KQ Dot Product (v7)

Since FWHT is orthogonal, `<K_dequant, Q> = norm * sum_i(cb[idx_i] * FWHT(sign * Q)[i])`.

Instead of inverse-FWHT on K (5 shuffles per block + 4 gather shuffles), apply FWHT to Q once per block and dot directly against codebook values.

| Metric | v6 | v7 |
|--------|:--:|:--:|
| Shuffles per vec_dot | 41 | **25** |
| Gather shuffles | 16 | **0** |
| Branch divergence | Yes | **No** |

### 2. Precomputed Sign Bits (`sb[4]`) in KTQ

Signs are computed once during quantization and stored as 32 bits in `sb[4]`. All KTQ dequant paths read signs directly -- **zero Philox calls at dequant time**. Eliminates ~320 multiply-XOR operations per block.

### 3. VTQ: Register-Light V-Dequant

VTQ eliminates the 32-element FWHT butterfly and sign bit reads from the FA inner loop. V-dequant reduces to `codebook[idx] * scale` -- inlineable, no LMEM spills, no FA accumulator corruption.

### 4. Warp-Cooperative FWHT in Flash Attention (v6)

Each warp lane holds one element; 5 `__shfl_xor_sync` rounds perform the full 32-point transform. Eliminates the 80 serial butterfly ops that bottlenecked CC 6.1.

### 5. Norm Correction

Stores `||x|| / ||reconstruction||` instead of raw `||x||`. Compensates systematic magnitude loss from codebook quantization. ~1.2% PPL improvement at zero dequant cost. Used by both KTQ and VTQ.

### 6. Sparse V Dequant

In Flash Attention V-accumulation: skip dequant for positions where attention weight < 1e-6. At 32K+ context, >90% of positions are skipped. +22% decode speedup. Works with both KTQ and VTQ V types.

### 7. Branchless Sign x Norm Fusion (v7)

`(1.0f - 2.0f * bit) * norm` replaces ternary `(bit ? 1.0f : -1.0f) * ... * norm` across KTQ dequant paths. Eliminates warp divergence.

## CUDA Implementation Details

| Feature | Description |
|---------|-------------|
| Hadamard-domain KQ dot | FWHT on Q, dot against codebook values. 39% fewer shuffles (KTQ) |
| Warp-parallel FWHT | 32 threads, `__shfl_xor_sync` butterfly -- zero shared memory (KTQ) |
| Branchless sign x norm | `(1 - 2*bit) * norm` replaces ternary branch (KTQ) |
| Sign lookup | Single bit-extract from `sb[]` per thread (KTQ) |
| VTQ V-dequant | `codebook[idx] * scale`, `__forceinline__`, ~8 registers |
| VTQ pre-rotation | `self_v_rot` matmul before cache write + inverse after FA |
| Shared codebooks | `PQ_CUDA_CB_*BIT` constants used by both KTQ and VTQ |
| FA Dispatch | KTQ1_1..KTQ4_1 (K+V), VTQ1_1..VTQ4_1 (V-only) |
| SET_ROWS | `k_set_rows_ktq*` + `k_set_rows_vtq*` kernels |
| Compute Capability | CC 6.1+ tested (CC 6.1, CC 7.5) |

## Source Files

| File | Description |
|------|-------------|
| `ggml/include/ggml.h` | Type enums: KTQ1_1=45, KTQ2_1=42, KTQ3_1=43, KTQ4_1=44, VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49 |
| `ggml/src/ggml-common.h` | Block structs: `block_ktq*` (with sb[4]) + `block_vtq*` (without sb) |
| `ggml/src/ggml-cuda/turboquant.cuh` | CUDA kernels: KTQ (Philox, FWHT, quantize, dequant) + VTQ (quantize, dequant -- no FWHT) |
| `ggml/src/ggml-cuda/fattn-common.cuh` | FA: `vec_dot_KQ_ktq*`, `dequantize_V_ktq*`, `dequantize_V_vtq*`, Sparse V guard |
| `ggml/src/ggml-cuda/convert.cu` | CUDA dequant dispatch (contiguous + NC) for KTQ + VTQ |
| `ggml/src/ggml-quants.c` | CPU quantize/dequantize for KTQ + VTQ, shared `PQ_CODEBOOK_*` |
| `common/arg.cpp` | CLI: `--cache-type-k ktq2_1 --cache-type-v vtq2_1` etc. |

## Roadmap

### Deferred K-Cache Quantization (HIGH)
K-Cache stays f16 during prefill, quantized to KTQ only at decode time. 3x better PPL, eliminates dequant overhead during prefill entirely.

### Boundary Layer Protection (MEDIUM)
First 2 + last 2 transformer layers use q8_0 instead of KTQ/VTQ for K/V cache. Recovers 37-91% of quality gap with zero speed penalty.

### VTQ-Specific Codebooks (LOW)
The fixed Hadamard rotation (without per-block random signs) produces slightly different marginal distributions than the full RHT. Re-optimized Lloyd-Max codebooks for fixed-Hadamard marginals could improve VTQ quality by ~0.5-1% PPL.

### Block Size 128 Rotation (LOW)
WHT rotation over 128 elements (full head_dim) instead of 32. Better decorrelation (5.12x vs 4.57x compression). Significant architecture change.

## References

This implementation is inspired by but deviates from the TurboQuant paper. KTQ uses Hadamard (FWHT) + random signs instead of QR rotation; VTQ uses a fixed D\*H\*D rotation (our own design). Neither uses QJL (removed in v5).

| Paper | Authors | arXiv | Relevance |
|-------|---------|-------|-----------|
| **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** | Zandieh, Daliri, Hadian, Mirrokni | [2504.19874](https://arxiv.org/abs/2504.19874) (April 2025) | Primary inspiration: random rotation + Lloyd-Max codebooks |
| **PolarQuant: Quantizing KV Cache via Polar Coordinate Transformation** | Han, Kacham, Karbasi, Mirrokni, Zandieh | [2502.02617](https://arxiv.org/abs/2502.02617) (Feb 2025) | Different method (polar coordinates), not used in our implementation |
| **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization** | Zandieh, Daliri, Han | [2406.03482](https://arxiv.org/abs/2406.03482) (June 2024) | Used in v1-v4, removed in v5 |

## Version History

- **v7+VTQ** (2026-04-16): KTQ/VTQ split. VTQ types (VTQ1_1..VTQ4_1) for V-cache -- no FWHT, no sign bits, `__forceinline__` dequant. TQ renamed to KTQ. Shared PQ_CODEBOOK constants.
- **v7** (2026-04-14): Hadamard-domain KQ dot product, branchless sign x norm fusion. PP +13% on CC 6.1, TG +65% on CC 7.5 vs v6.
- **v6** (2026-04-13): Warp-cooperative FWHT in FA, SET_ROWS kernel, FA dispatch registration, warp-cooperative V-dequant.
- **v5** (2026-04-10): Precomputed sign bits, struct compaction (3.5/4.5/5.5 bpw), norm correction, Philox 6r.
- **v4** (2026-04-09): TQ4_1 (4-bit, 16 centroids), Sparse V Dequant, asymmetric K/V support.
- **v3** (2026-04-08): Paper-compliant: stored r_norm, QJL on CUDA, Beta-exact codebooks.
- **v2** (2026-04-07): Warp-parallel FWHT, CUDA QJL attempt.
- **v1** (2026-04-06): Initial TurboQuant-inspired implementation + CPU reference.
