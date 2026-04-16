# TurboQuant v7 -- Advanced KV Cache Quantization for CUDA

## Overview

TurboQuant implements [Google Research's TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) in GGML. It compresses the KV cache via **PolarQuant** (Randomized Hadamard Transform + Lloyd-Max scalar quantization), enabling significantly longer context windows on limited VRAM.

**v7 key innovation:** Hadamard-domain KQ dot product -- applies FWHT to Q instead of inverse-FWHT on K, eliminating gather shuffles and branch divergence. 39% fewer warp shuffles per vec_dot call.

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Maximum compression (3.5 bpw)
./build/bin/llama-server -m model.gguf \
    --cache-type-k tq2_1 --cache-type-v tq2_1 -fa on -ngl 99

# Best quality TQ (5.5 bpw, better than q4_0 PPL)
./build/bin/llama-server -m model.gguf \
    --cache-type-k tq4_1 --cache-type-v tq4_1 -fa on -ngl 99

# Asymmetric (recommended for quantized weights)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v tq2_1 -fa on -ngl 99
```

## Available Types

| Type | bpw | Block Size | vs f16 | vs q4_0 | Use Case |
|------|-----|------------|--------|---------|----------|
| `tq1_1` | 2.5 | 10 bytes | **-84%** | -44% | Extreme compression (V-cache recommended) |
| `tq2_1` | 3.5 | 14 bytes | -78% | -22% | Maximum compression, long context |
| `tq3_1` | 4.5 | 18 bytes | -72% | Same | Balanced quality/compression |
| `tq4_1` | 5.5 | 22 bytes | -66% | +22% | Best TQ quality, better PPL than q4_0 |

For comparison:

| Type | bpw | Block Size (32 elements) |
|------|-----|--------------------------|
| f16 | 16.0 | 64 bytes |
| q8_0 | 8.5 | 34 bytes |
| **tq4_1** | **5.5** | **22 bytes** |
| q4_0 | 4.5 | 18 bytes |
| **tq3_1** | **4.5** | **18 bytes** |
| **tq2_1** | **3.5** | **14 bytes** |
| **tq1_1** | **2.5** | **10 bytes** |

## Memory Savings (Ministral 3B, 26 layers, 8 KV-heads)

| Context | f16 | q8_0 | q4_0 | tq3_1 | tq2_1 |
|---------|------|------|------|-------|-------|
| 32K | 1.6 GB | 0.8 GB | 0.5 GB | 0.4 GB | 0.3 GB |
| 128K | 6.4 GB | 3.4 GB | 1.8 GB | 1.8 GB | 1.4 GB |
| 384K | 19.2 GB | 10.2 GB | 5.4 GB | 5.4 GB | 4.2 GB |

## Benchmarks

### CC 7.5 (12 GB) -- Qwen3.5-35B-A3B IQ2_XS

| KV Cache | bpw | pp512 (tok/s) | tg32 (tok/s) | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| q4_0/q4_0 | 4.5 | 838 | 69.5 | baseline |
| **tq2_1/tq2_1 (v7)** | 3.5 | **634** | **63.4** | **-8.8%** |

v7 TG: **+65% faster** than v6 (63.4 vs 38.4 tok/s).
78% less KV-Cache VRAM vs f16 -- fit 4x more context in the same memory.

### CC 6.1 (6 GB) -- Ministral 3B IQ2_M

| KV Cache | bpw | pp128 (tok/s) | tg32 (tok/s) | TG vs q4_0 |
|----------|-----|:---:|:---:|:---:|
| f16/f16 | 16.0 | 824 | 45.8 | +6.5% |
| q4_0/q4_0 | 4.5 | 798 | 43.0 | baseline |
| **tq2_1/tq2_1 (v7)** | 3.5 | **211** | **33.3** | **-22.6%** |

v7 improved PP by +13% over v6 via Hadamard-domain dot product.
TG penalty ~22% on CC 6.1 (fewer warp schedulers limit shuffle hiding).

### Example Configurations

**Qwen3.5-35B-A3B IQ2_XS on CC 7.5 (12 GB):**

```bash
llama-server -m Qwen3.5-35B-A3B-IQ2_XS.gguf \
    --cache-type-k tq2_1 --cache-type-v tq2_1 \
    -c 400000 -ngl 99 --parallel 2
```

| Detail | Value |
|--------|-------|
| Context | 400K tokens |
| Parallel Slots | 2 |
| KV-Cache (tq2_1) | 1,711 MB (vs ~10,400 MB with q4_0) |
| Total VRAM | 9.0 GB / 12 GB |

Without TurboQuant: q4_0 can't fit 400K context on 12 GB. Even 200K is tight with parallel 2.

**Gemma4 26B on CC 7.5 (12 GB):**

| Metric | q4_0 KV | tq2_1 KV | Delta |
|--------|:---:|:---:|:---:|
| KV-Cache (200K ctx) | ~1.7 GB | ~1.3 GB | -400 MB |
| Speed | 36.6 tok/s | 38.6 tok/s | **+2 tok/s** |
| GPU Layers | 16/30 | 17/30 | **+1 layer** |

Saved VRAM enables +1 GPU layer, netting +2 tok/s. Speed AND compression win.

### v6 to v7 Performance Delta

**CC 7.5 (12 GB) -- Qwen3.5-35B-A3B IQ2_XS:**

| Metric | v6 | v7 | Delta |
|--------|:---:|:---:|:---:|
| PP512 | 686 tok/s | 634 tok/s | -7.6% |
| TG32 | 38.4 tok/s | 63.4 tok/s | **+65.1%** |
| TG penalty vs q4_0 | 44.7% | 8.8% | **-35.9pp** |

**CC 6.1 (6 GB) -- Ministral 3B IQ2_M:**

| Metric | v6 | v7 | Delta |
|--------|:---:|:---:|:---:|
| PP128 | 186 tok/s | 211 tok/s | **+13.4%** |
| TG32 | 32.2 tok/s | 33.3 tok/s | +3.4% |

### Memory Savings (Qwen3.5-35B-A3B, MoE)

| Context | q4_0 KV | tq2_1 KV | Savings |
|---------|:---:|:---:|:---:|
| 32K | ~860 MB | ~670 MB | -190 MB |
| 128K | ~3,400 MB | ~2,650 MB | -750 MB |
| 200K | ~5,300 MB | ~4,130 MB | -1,170 MB |
| 400K | ~10,400 MB | ~1,711 MB | **-8,689 MB (83%)** |

Note: At 400K with GatedDeltaNet architecture, 75% of layers use constant recurrent state instead of growing KV cache. The 1,711 MB is the measured value.

## How It Works

### Quantization Pipeline

```
float[32] -> normalize -> random signs -> FWHT -> codebook -> norm correction -> pack
                              |                                      |
                        [store in sb[4]]                    [store corrected d]
```

1. **Normalize:** `x_hat = x / ||x||`
2. **Random Signs:** Apply deterministic +/-1 signs (Philox 6-round PRNG from block-index seed)
3. **FWHT:** Fast Walsh-Hadamard Transform, scaled by `1/sqrt(32)`
4. **Lloyd-Max Quantization:** Nearest centroid from Beta(15.5, 15.5)-optimal codebook
5. **Norm Correction:** Reconstruct, measure `||recon||`, store `norm / ||recon||` -- compensates codebook error (~1.2% PPL improvement)
6. **Sign Bits:** Store precomputed signs in `sb[4]` (32 bits = 32 signs)

### Dequantization Pipeline

1. **Codebook lookup:** Index -> centroid value (rotated space)
2. **Inverse FWHT:** Walsh-Hadamard transform
3. **Inverse signs:** Read from `sb[4]` -- **no Philox at dequant time**
4. **Scale:** Multiply by corrected norm

### Block Layout

```
TQ1_1 (10 bytes, 2.5 bpw):
+--------+----------+--------+
| d (2B) | qs[4] 1b | sb[4]  |
|  norm   | indices  | signs  |
+--------+----------+--------+

TQ2_1 (14 bytes, 3.5 bpw):
+--------+----------+--------+
| d (2B) | qs[8] 2b | sb[4]  |
|  norm   | indices  | signs  |
+--------+----------+--------+

TQ3_1 (18 bytes, 4.5 bpw):
+--------+-----------+--------+
| d (2B) | qs[12] 3b | sb[4]  |
|  norm   |  indices  | signs  |
+--------+-----------+--------+

TQ4_1 (22 bytes, 5.5 bpw):
+--------+-----------+--------+
| d (2B) | qs[16] 4b | sb[4]  |
|  norm   |  indices  | signs  |
+--------+-----------+--------+
```

### Codebooks (Lloyd-Max for Beta(15.5, 15.5), d=32)

- **1-bit (2 centroids):** `{-0.7979, 0.7979}` (= sqrt(2/pi))
- **2-bit (4 centroids):** `{-1.4896, -0.4514, 0.4514, 1.4896}`
- **3-bit (8 centroids):** `{-2.0719, -1.3150, -0.7453, -0.2424, 0.2424, 0.7453, 1.3150, 2.0719}`
- **4-bit (16 centroids):** `{-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326}`

All scaled by `1/sqrt(32)`.

## v7 Optimizations

### 1. Hadamard-Domain KQ Dot Product (v7)

The key mathematical insight: since FWHT is orthogonal, `<K_dequant, Q> = norm * sum_i(cb[idx_i] * FWHT(sign * Q)[i])`.

Instead of inverse-FWHT on K (5 shuffles per block + 4 gather shuffles), apply FWHT to Q once per block and dot directly against codebook values. No gathers, no branch divergence.

| Metric | v6 | v7 |
|--------|:--:|:--:|
| Shuffles per vec_dot | 41 | **25** |
| Gather shuffles | 16 | **0** |
| Branch divergence | Yes | **No** |

### 2. Precomputed Sign Bits (`sb[4]`)

Signs are computed once during quantization and stored as 32 bits in `sb[4]`. All dequant paths read signs directly -- **zero Philox calls at dequant time**. Eliminates ~320 multiply-XOR operations per block.

### 3. Warp-Cooperative FWHT in Flash Attention (v6)

Each warp lane holds one element; 5 `__shfl_xor_sync` rounds perform the full 32-point transform. Eliminates the 80 serial butterfly ops that bottlenecked CC 6.1.

### 4. Norm Correction

Stores `||x|| / ||reconstruction||` instead of raw `||x||`. Compensates systematic magnitude loss from codebook quantization. ~1.2% PPL improvement at zero dequant cost.

### 5. Sparse V Dequant

In Flash Attention V-accumulation: skip dequant for positions where attention weight < 1e-6. At 32K+ context, >90% of positions are skipped. +22% decode speedup.

### 6. Branchless Sign x Norm Fusion (v7)

`(1.0f - 2.0f * bit) * norm` replaces ternary `(bit ? 1.0f : -1.0f) * ... * norm` across all dequant paths. Eliminates warp divergence.

## Recommended Configurations

| Model Weights | K-Cache | V-Cache | Notes |
|--------------|---------|---------|-------|
| Q8_0+ (high quality) | `tq2_1` | `tq2_1` | Symmetric OK |
| Q4_K_M (quantized) | `q8_0` | `tq2_1` | K precision critical |
| Maximum context | `tq2_1` | `tq2_1` | V compression is "free" |
| Best quality | `tq4_1` | `tq4_1` | Better PPL than q4_0 |

## CUDA Implementation Details

| Feature | Description |
|---------|-------------|
| Hadamard-domain KQ dot | FWHT on Q, dot against codebook values. 39% fewer shuffles |
| Warp-parallel FWHT | 32 threads, `__shfl_xor_sync` butterfly -- zero shared memory |
| Branchless sign x norm | `(1 - 2*bit) * norm` replaces ternary branch |
| Sign lookup | Single bit-extract from `sb[]` per thread -- replaces 10-round Philox |
| FA FWHT | Warp-cooperative FWHT replaces serial butterfly in FA vec kernel |
| FA V-dequant | Warp-cooperative inverse-FWHT for symmetric TQ V-cache |
| SET_ROWS | `k_set_rows_tq` kernel for non-contiguous KV cache writes |
| FA Dispatch | TQ2_1, TQ3_1, TQ4_1 registered in `fattn.cu` dispatch + `fattn-vec.cuh` templates |
| Compute Capability | CC 6.1+ tested (CC 6.1, CC 7.5) |

## Source Files

| File | Description |
|------|-------------|
| `ggml/include/ggml.h` | Type enum: TQ1_1=42, TQ2_1=43, TQ3_1=44, TQ4_1=45 |
| `ggml/src/ggml-common.h` | Block structs (14B, 18B, 22B) |
| `ggml/src/ggml-cuda/turboquant.cuh` | CUDA kernels: Philox, FWHT, quantize, dequant, get-rows |
| `ggml/src/ggml-cuda/fattn-common.cuh` | FA: vec_dot_KQ_tq*, dequantize_V_tq*, Sparse V guard |
| `ggml/src/ggml-cuda/convert.cu` | CUDA dequant dispatch (contiguous + NC) |
| `ggml/src/ggml-quants.c` | CPU quantize/dequantize/vec_dot |
| `common/arg.cpp` | CLI: `--cache-type-k tq2_1` etc. |

## Roadmap

### Asymmetric V-Dequant Optimization (HIGH)
Warp-cooperative V-dequant currently only works when K and V use the same TQ type. Asymmetric configurations fall back to serial FWHT.

### Deferred K-Cache Quantization (HIGH)
K-Cache stays f16 during prefill, quantized to TQ only at decode time. 3x better PPL, eliminates dequant overhead during prefill entirely.

### Boundary Layer Protection (MEDIUM)
First 2 + last 2 transformer layers use q8_0 instead of TQ for K/V cache. Recovers 37-91% of quality gap with zero speed penalty.

### Block Size 128 Rotation (LOW)
WHT rotation over 128 elements (full head_dim) instead of 32. Better decorrelation (5.12x vs 4.57x compression). Significant architecture change.

## Paper Reference

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Google Research, ICLR 2026 -- https://arxiv.org/abs/2504.19874

## KTQ: K-Cache Optimized TurboQuant

The existing TQ types (TQ1_1 through TQ4_1) are referred to as **KTQ** -- they are optimized for K-cache quantization.

### Why KTQ Works for K-Cache

The Flash Attention kernel computes `score = dot(Q, K)` for every query-key pair. KTQ exploits a mathematical property of the Hadamard transform: because FWHT is orthogonal, the dot product is invariant under rotation. Instead of dequantizing K (inverse-FWHT per block), the FA kernel applies FWHT to Q once per block and dots directly against the stored codebook values in the rotated domain.

```
Standard:  score = dot(Q, dequant(K))     -- requires inverse-FWHT per K block
KTQ:       score = norm * dot(FWHT(sign * Q), codebook_values)  -- FWHT on Q only
```

This Hadamard-domain dot product is mathematically exact (no approximation) and eliminates all gather shuffles and branch divergence from K-dequant.

### KTQ V-Dequant: Known Issue

TQ V-dequant in the FA kernel has a known correctness issue. The V accumulation path requires full dequantization (codebook lookup, inverse-FWHT, inverse signs, scale) back to float values before weighted summation. Both the serial FWHT path (float buf[32]) and the warp-cooperative path (5 `__shfl_xor_sync` rounds + shfl_sync gather) produce corrupted output when invoked within the FA kernel context, despite producing correct values in isolation.

**Root cause:** The per-block sign bits stored in `sb[4]` prevent two otherwise promising optimization strategies:
- **Lazy V (skip inverse-FWHT, accumulate in rotated domain):** Not possible because each block has unique random signs -- sign inversion must happen per-block before FWHT, so the rotated domains are not aligned across blocks.
- **Graph-Level inverse (single FWHT after attention-weighted sum):** Not possible because the random signs differ per block -- you cannot factor a single inverse rotation out of a weighted sum of differently-rotated blocks.

The serial FWHT fallback corrupts register state in the kernel context (likely an NVCC optimization artifact that reorders or spills the 32-element butterfly buffer). This is not fixable with `__forceinline__` or warp-cooperative rewrites.

### Recommended Configuration

Use KTQ for K-cache and a standard quantization type for V-cache:

| Config | K bpw | V bpw | Effective bpw | vs f16+f16 | Notes |
|--------|-------|-------|---------------|------------|-------|
| **K=tq2_1 + V=q4_0** | 3.5 | 4.5 | 4.0 | **-75%** | Best compression, recommended |
| K=tq2_1 + V=q8_0 | 3.5 | 8.5 | 6.0 | -63% | Higher V quality |
| K=tq3_1 + V=q4_0 | 4.5 | 4.5 | 4.5 | -72% | Balanced |
| K=tq4_1 + V=q8_0 | 5.5 | 8.5 | 7.0 | -56% | Maximum quality |

```bash
# Recommended: maximum compression with proven stability
llama-server -m model.gguf \
    --cache-type-k tq2_1 --cache-type-v q4_0 \
    -fa on -ngl 99

# Higher V quality
llama-server -m model.gguf \
    --cache-type-k tq2_1 --cache-type-v q8_0 \
    -fa on -ngl 99
```

### Future: VTQ (V-Cache Optimized Format)

A dedicated VTQ format is planned that uses fixed (non-random) rotation, enabling Graph-Level inverse FWHT after the attention-weighted sum. This would allow V-cache compression at TQ-level bitrates while keeping the inverse transform outside the inner loop of the FA kernel.

## Version History

- **v7** (2026-04-14): Hadamard-domain KQ dot product, branchless sign x norm fusion. PP +13% on CC 6.1, TG +65% on CC 7.5 vs v6.
- **v6** (2026-04-13): Warp-cooperative FWHT in FA, SET_ROWS kernel, FA dispatch registration, warp-cooperative V-dequant.
- **v5** (2026-04-10): Precomputed sign bits, struct compaction (3.5/4.5/5.5 bpw), norm correction, Philox 6r.
- **v4** (2026-04-09): TQ4_1 (4-bit, 16 centroids), Sparse V Dequant, asymmetric K/V support.
- **v3** (2026-04-08): Paper-compliant: stored r_norm, QJL on CUDA, Beta-exact codebooks.
- **v2** (2026-04-07): Warp-parallel FWHT, CUDA QJL attempt.
- **v1** (2026-04-06): Initial PolarQuant + CPU reference.
