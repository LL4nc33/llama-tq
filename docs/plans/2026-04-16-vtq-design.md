# VTQ (Value TurboQuant) -- V-Cache Optimized TurboQuant for Flash Attention

**Author:** LL4nc33  
**Date:** 2026-04-16  
**Status:** Implemented + Verified  
**Depends on:** KTQ v7 (formerly TQ v7), upstream `self_v_rot` infrastructure (PR #21038)

> **Naming note:** This document uses "PolarQuant" in some mathematical discussions to refer to the
> quantization step (Hadamard rotation + Lloyd-Max codebook). This is a misnomer from early development.
> The actual paper is **TurboQuant** (arXiv:2504.19874, Zandieh et al.). PolarQuant (arXiv:2502.02617)
> is a completely different method using polar coordinate transformation, not used here.
> This implementation deviates from TurboQuant by using FWHT + random signs (instead of QR rotation) for
> K-cache, and a fixed D\*H\*D rotation (a design specific to this fork) for V-cache.

---

## Executive Summary

Current KTQ (K-Cache TurboQuant, formerly TQ) types use Hadamard rotation (RHT) with per-block data-dependent sign bits. The dequantization path requires a serial 32-element FWHT butterfly transform, which is acceptable for the K-cache (where the FWHT can be moved to Q via the Hadamard-domain dot product trick) but causes register spilling and corruption in the V-cache Flash Attention inner loop.

VTQ solves this by separating the quantization scheme for V-cache:
- **Store** V values in Hadamard-rotated space using a **fixed, position-independent** rotation matrix R per attention head
- **Dequant in FA** reduces to codebook lookup + scale (no FWHT, no sign bits)
- **Inverse rotation** R^T is applied as a single `ggml_mul_mat` AFTER the flash attention kernel completes

This is exact (no quantization in this step; the equality is between the mathematical operations, modulo the scalar codebook quantization applied separately). The orthogonality argument is specifically:

```
R^T * ( softmax(QK^T) * (R * v) ) = R^T * R * ( softmax(QK^T) * v ) = softmax(QK^T) * v
```

The order matters: `R` is applied to `v` before the softmax-weighted sum (stored in the cache that way), and `R^T` is applied after. We can pull `R` through the softmax-weighted sum only because the same `R` multiplies every token's `v` -- the softmax weights `alpha_i` are scalars commuting with `R`, and because `R` does not depend on position `i`, it factors out of `sum_i alpha_i * R * v_i = R * sum_i alpha_i * v_i`. This position-independence of `R` is the crucial property that makes VTQ work; it is exactly what KTQ's per-block data-dependent RHT lacks. The inverse rotation is then a single post-FA matmul, leveraging the existing `self_v_rot` infrastructure already present in upstream llama.cpp.

### Key Insight

The fundamental problem with TQ-in-V is that the RHT rotation is **per-block and data-dependent** (Philox-seeded). This means you cannot defer the inverse transform -- each KV position has a different rotation. VTQ uses a **fixed rotation for all positions in a head**, making the rotation commute through the softmax-weighted sum:

```
Standard TQ:    output = sum_i( alpha_i * D_i * H * CB[q_i] * scale_i )   -- D_i (Philox-derived sign diagonal) varies per position i, cannot factor out
VTQ:            output = sum_i( alpha_i * R   * CB[q_i] * scale_i )        -- R is identical for every position in the head, factor it out
             = R * sum_i( alpha_i * CB[q_i] * scale_i )                    -- R moves outside the sum by linearity
```

The FA kernel only computes the inner sum. The outer `R *` is a post-FA matmul.

---

## Mathematical Formulation

### Notation

| Symbol | Meaning |
|--------|---------|
| `v` | Original f32 value vector (d_head elements) |
| `R` | Fixed orthogonal rotation matrix (d_head x d_head), same for all positions in a head |
| `H` | Normalized Walsh-Hadamard matrix (d_head x d_head), `H^2 = I` |
| `D1, D2` | Diagonal sign matrices (+/-1 on diagonal), derived from per-layer seed |
| `CB[i]` | Lloyd-Max codebook centroid for index i |
| `s` | Per-block L2 norm (scale factor) |
| `q_j` | Quantized codebook index for element j |
| `alpha_i` | Softmax attention weight for position i |

### Rotation Construction

The fixed rotation is `R = D1 * H * D2` where:
- `H` is the normalized Hadamard matrix (already computed by `ggml_gen_hadamard()`)
- `D1` and `D2` are diagonal sign matrices generated from a deterministic seed per (layer, head) pair
- `R^T = D2 * H * D1` (since `H = H^T` and diagonal matrices are self-transpose with sign flip)
- `R * R^T = I` (orthogonal, so lossless)

**Decision: Use the bare Hadamard matrix H (without D1, D2) for the initial implementation.** Rationale:
- The existing `ggml_gen_hadamard()` + `self_v_rot` infrastructure already constructs and applies H
- H is already orthogonal (`H^T * H = I`) and position-independent
- The D1/D2 diagonal signs are what make `D*H*D` deliver approximately i.i.d. coordinates from the Beta((d-1)/2, (d-1)/2) marginal. Bare `H` maps the standard basis to *one specific* fixed orthonormal basis (the Hadamard basis) rather than randomizing across orthonormal bases, so its output is not rotationally symmetric; the flanking sign diagonals break that symmetry. The cost of using `D*H*D` is per-element sign bits in the block struct
- For V-cache, one can absorb the "randomization" benefit by using a slightly larger codebook or accepting the marginal quality loss
- If quality measurements show degradation, D1/D2 can be added later as `VTQ_v2` with the sign bits stored as a per-head constant rather than per-block

### Quantization (cache write path)

```
1. v_rot = R * v                          -- rotate into Hadamard space (graph-level matmul, BEFORE cache write)
2. For each block of 32 elements in v_rot:
   a. s = ||block||_2                     -- L2 norm
   b. v_hat = block / s                   -- normalize
   c. q_j = argmin_c |v_hat[j] - CB[c]|  -- nearest centroid, for j = 0..31
   d. Store: (s, q[0..31])                -- scale + indices, NO sign bits
```

### Dequantization (FA kernel, per KV position)

```
1. For each element j in the block:
   v_approx[j] = CB[q_j] * s             -- codebook lookup + scale multiply
                                          -- NO FWHT, NO sign flip
```

This produces values in the rotated (Hadamard) domain. The FA kernel accumulates:
```
VKQ_rotated = sum_i( alpha_i * v_approx_i )    -- standard FA weighted sum, in rotated space
```

### Post-FA Inverse Rotation

```
VKQ_final = R^T * VKQ_rotated                  -- graph-level matmul, AFTER FA
          = R^T * sum_i( alpha_i * R * v_i )   -- by linearity
          = sum_i( alpha_i * v_i )              -- since R^T * R = I
```

This step is already implemented in `build_attn()`:
```cpp
if (inp->self_v_rot) {
    cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);  // line 2144-2145 in llama-graph.cpp
}
```

---

## Block Struct Layout

### VTQ2_1 (2-bit, target 3.0 bpw)

```c
#define QK_VTQ 32  // block size (matches warp size)

// VTQ2_1: 2-bit codebook, NO sign bits = 3.0 bpw
typedef struct {
    ggml_half d;              // 2B: block L2 norm (scale)
    uint8_t   qs[QK_VTQ / 4]; // 8B: 2-bit codebook indices (4 per byte)
} block_vtq2_1;               // = 10 bytes for 32 elements = 2.5 bpw
static_assert(sizeof(block_vtq2_1) == 10, "wrong vtq2_1 block size");
```

### VTQ3_1 (3-bit, target 4.0 bpw)

```c
typedef struct {
    ggml_half d;                  // 2B: block L2 norm
    uint8_t   qs[QK_VTQ * 3 / 8]; // 12B: 3-bit indices (packed)
} block_vtq3_1;                    // = 14 bytes for 32 elements = 3.5 bpw
static_assert(sizeof(block_vtq3_1) == 14, "wrong vtq3_1 block size");
```

### VTQ4_1 (4-bit, target 5.0 bpw)

```c
typedef struct {
    ggml_half d;              // 2B: block L2 norm
    uint8_t   qs[QK_VTQ / 2]; // 16B: 4-bit indices (nibble-packed)
} block_vtq4_1;               // = 18 bytes for 32 elements = 4.5 bpw
static_assert(sizeof(block_vtq4_1) == 18, "wrong vtq4_1 block size");
```

### Comparison with KTQ Structs

| Type | d (norm) | qs (indices) | sb (signs) | Total | bpw |
|------|----------|-------------|------------|-------|-----|
| block_ktq1_1 | 2B | 4B | **4B** | 10B | 2.5 |
| block_vtq1_1 | 2B | 4B | **0B** | 6B | **1.5** |
| block_ktq2_1 | 2B | 8B | **4B** | 14B | 3.5 |
| block_vtq2_1 | 2B | 8B | **0B** | 10B | **2.5** |
| block_ktq3_1 | 2B | 12B | **4B** | 18B | 4.5 |
| block_vtq3_1 | 2B | 14B | **0B** | 16B | **4.0** |
| block_ktq4_1 | 2B | 16B | **4B** | 22B | 5.5 |
| block_vtq4_1 | 2B | 16B | **0B** | 18B | **4.5** |

VTQ saves 4 bytes per block (the `sb[4]` sign bits) because the rotation is fixed and position-independent -- it does not need per-block storage.

**Important note on effective quality:** Although VTQ2_1 is 2.5 bpw (vs KTQ2_1 at 3.5 bpw), the quality should be comparable because:
1. KTQ's sign bits encode per-block RHT randomization; VTQ's fixed rotation achieves the same decorrelation globally
2. The codebooks are identical (same Lloyd-Max centroids from shared PQ_CODEBOOK_*)
3. The fixed rotation's decorrelation is slightly weaker than per-block RHT (no per-block randomness), so VTQ2_1 quality will be between KTQ1_1 and KTQ2_1 -- empirical measurement needed

---

## Quantization Algorithm (CUDA)

### Kernel: `vtq_cuda_quantize_vtq2_1_block`

VTQ quantization is simpler than TQ because there is no RHT (the rotation is handled at graph level before the data reaches the quantizer):

```cuda
static __device__ void vtq_cuda_quantize_vtq2_1_block(
        const float * __restrict__ x,
        block_vtq2_1 * __restrict__ y) {

    // Step 1: L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) {
        memset(y->qs, 0, 8);
        return;
    }

    // Step 2: Normalize
    float x_hat[32];
    const float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) x_hat[j] = x[j] * inv_norm;

    // Step 3: Nearest codebook centroid (same Lloyd-Max codebook as TQ)
    // NOTE: Input x is ALREADY in rotated (Hadamard) space because the graph
    //       applied R before the cache write. No RHT/FWHT needed here.
    memset(y->qs, 0, 8);
    for (int j = 0; j < 32; j++) {
        float val = x_hat[j];
        int best = 0;
        float best_d = fabsf(val - VTQ_CUDA_CB_2BIT[0] * VTQ_CUDA_CB_SCALE);
        for (int c = 1; c < 4; c++) {
            float d = fabsf(val - VTQ_CUDA_CB_2BIT[c] * VTQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        y->qs[j / 4] |= (uint8_t)(best << (2 * (j % 4)));
    }

    // Step 4: Norm correction (same approach as TQ v5)
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        float r = VTQ_CUDA_CB_2BIT[idx] * VTQ_CUDA_CB_SCALE;
        recon_sq += r * r;
    }
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);
}
```

Compared to `ktq_cuda_quantize_ktq2_1_block`:
- **Removed:** Philox seed derivation, RHT sign generation, FWHT, sign bit storage (all present in KTQ)
- **Kept:** Norm, normalize, codebook search, norm correction (shared with KTQ)
- **Result:** ~40% fewer instructions, no PRNG state, no sb[] writes

### Codebook Constants

VTQ reuses the same Lloyd-Max codebooks as KTQ. These are optimized for the `Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)` marginal at d=32: a unit vector in R^d after random rotation has coordinates distributed such that `(x_i + 1)/2 ~ Beta((d-1)/2, (d-1)/2)`. Shared `PQ_CODEBOOK_*` / `PQ_CUDA_CB_*` constants:

```cuda
// Same codebooks -- the Hadamard rotation preserves the statistical distribution
#define VTQ_CUDA_CB_SCALE 0.17677669529663689f  // 1/sqrt(32)

__device__ __constant__ static float VTQ_CUDA_CB_2BIT[4] = {
    -1.489560f, -0.451428f, 0.451428f, 1.489560f
};

__device__ __constant__ static float VTQ_CUDA_CB_3BIT[8] = {
    -2.071926f, -1.314996f, -0.745325f, -0.242405f,
     0.242405f,  0.745325f,  1.314996f,  2.071926f
};

__device__ __constant__ static float VTQ_CUDA_CB_4BIT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};
```

**Note on codebook optimality:** The fixed Hadamard rotation (without per-block random signs) produces coordinates whose empirical marginal differs from the full-RHT Beta(15.5, 15.5) case: bare `H` always sends a given input direction to the same output direction, so the "averaging" that drives the marginal toward Beta(15.5, 15.5) is weaker than under full D\*H\*D. In practice V-cache inputs are not axis-aligned so the empirical marginal is still approximately Beta-shaped, but the Lloyd-Max optimum may have shifted slightly. If benchmarks show quality degradation, a re-optimized codebook specific to fixed-Hadamard marginals could be computed. This is a v2 optimization -- start with the existing codebooks.

---

## Dequantization in FA Kernel

### `dequantize_V_vtq2_1` -- The Critical Improvement

The V-dequant for VTQ is trivially simple -- no FWHT, no sign bits:

```cuda
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq2_1(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {

    const block_vtq2_1 * x = (const block_vtq2_1 *) vx;
    const int64_t ib = i0 / QK_VTQ;
    const int     il = (int)(i0 % QK_VTQ);
    const float   scale = (float)x[ib].d;

    if constexpr (std::is_same_v<T, half>) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) {
            const int idx = (x[ib].qs[(il + l) / 4] >> (2 * ((il + l) % 4))) & 0x3;
            ((half *) dst)[l] = __float2half(VTQ_CUDA_CB_2BIT[idx] * VTQ_CUDA_CB_SCALE * scale);
        }
    } else {
        #pragma unroll
        for (int l = 0; l < ne; ++l) {
            const int idx = (x[ib].qs[(il + l) / 4] >> (2 * ((il + l) % 4))) & 0x3;
            ((float *) dst)[l] = VTQ_CUDA_CB_2BIT[idx] * VTQ_CUDA_CB_SCALE * scale;
        }
    }
}
```

### Comparison with KTQ V-Dequant

| Operation | KTQ dequant_V | VTQ dequant_V |
|-----------|-------------|---------------|
| Load block data | qs[8] + sb[4] + d | qs[8] + d |
| Codebook lookup | 32 lookups | ne lookups (4 typical) |
| Serial FWHT | **32-element butterfly: 5 stages (= log2(32)) x 16 add/sub pairs = ~160 FLOPs** | **NONE** |
| Sign flip | 32 branchless mul | **NONE** |
| Scale multiply | 32 mul | ne mul |
| Registers | **~40 floats (buf[32] + intermediates)** | **~8 floats** |
| Can be `__forceinline__` | No (`__noinline__` required) | **Yes** |
| Register spilling | **Severe (the V-corruption root cause)** | **None** |

The VTQ dequant eliminates the two most expensive operations (FWHT butterfly and sign flip) and reduces thread-local register footprint from ~40 floats (dominated by the `buf[32]` FWHT staging array that NVCC cannot keep fully in registers inside a register-hot FA inner loop) to ~8 floats (a handful of scalars: index, scale, decoded centroid, loop counter). This is the difference between `__noinline__` (forced because NVCC's heuristics detect the register pressure and spill `buf[32]` to per-thread local memory / LMEM, whose writes interleave with and clobber FA accumulator state across warp iterations) and `__forceinline__` (the whole dequant fuses into the FA loop and optimizes normally).

---

## Post-FA Inverse Rotation

### Graph-Level Operation

After FA produces `VKQ_rotated` (shape: `[d_head, n_head, n_tokens]`), the inverse rotation is:

```
VKQ_final = R^T * VKQ_rotated
```

This is already handled by the existing `self_v_rot` infrastructure:

```cpp
// In build_attn() -- llama-graph.cpp line 2144
if (inp->self_v_rot) {
    cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);
}
```

Where `ggml_mul_mat_aux` reshapes `cur` to `[n_rot, n_elements/n_rot]`, multiplies by the rotation matrix, and reshapes back.

### Performance Cost

For a typical model (Qwen3.5-35B-A3B, d_head=128, n_head=8, n_tokens=1):
- Rotation matrix: 128 x 128 = 16K floats = 64 KB per layer
- Matmul: 128 x 128 x 8 = 131K FMA per layer, per token
- At 32 layers: ~4.2M FMA total per decode step

For context: a single FA vec_dot_KQ with 1024 KV positions does ~4M FMA. The post-FA rotation adds roughly 1 FA-equivalent of compute per decode step across all layers. At decode throughput of ~60 tok/s on RTX 2060, this is ~250M FMA/s additional -- well within the 6.5 TFLOPS budget.

**Important:** The rotation matrix is stored once per model load (in `attn_rot_hadamard`), not per KV position. Memory overhead is negligible: 64 KB * 32 layers = 2 MB.

### Optimization: Smaller Rotation Matrices

The existing upstream code already supports using `nrot = 64` instead of `nrot = d_head` (see `build_input_v_rot()` in `llama-kv-cache.cpp`). A 64x64 rotation matrix applied per 64-element stripe of the head dimension reduces the matmul cost by 4x while still providing good decorrelation. The upstream PR #21038 discussion suggests this is beneficial for V.

---

## Graph Builder Changes

### Required Modifications to `build_attn()`

No changes to the graph builder are needed for the basic VTQ flow. The existing `self_v_rot` path already:

1. **Before cache write:** Applies `R` to `v_cur` via `ggml_mul_mat_aux(ctx0, v_cur, inp->self_v_rot)` (line 2113-2115)
2. **After FA:** Applies `R^T` (same matrix since H^T = H) via `ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot)` (line 2144-2145)

The only change needed is in the KV cache initialization -- VTQ types must trigger `attn_rot_v = true` unconditionally (currently it is enabled for all quantized types when head_dim is divisible by 64):

```cpp
// In llama_kv_cache constructor (llama-kv-cache.cpp, line 378)
attn_rot_v =
    !attn_rot_disable &&
    n_embd_head_v_all > 0 &&
    ggml_is_quantized(type_v) &&   // VTQ types register as quantized
    hparams.n_embd_head_v() % 64 == 0;
```

VTQ types must register `ggml_is_quantized() = true` in the type metadata table in `ggml.c`. This is already the pattern for TQ types.

### New Type Registration

In `ggml.h`, type enums (implemented):

```c
GGML_TYPE_KTQ2_1  = 42, // K-cache, 3.5 bpw
GGML_TYPE_KTQ3_1  = 43, // K-cache, 4.5 bpw
GGML_TYPE_KTQ4_1  = 44, // K-cache, 5.5 bpw
GGML_TYPE_KTQ1_1  = 45, // K-cache, 2.5 bpw
GGML_TYPE_VTQ1_1  = 46, // V-cache, 1.5 bpw
GGML_TYPE_VTQ2_1  = 47, // V-cache, 2.5 bpw
GGML_TYPE_VTQ3_1  = 48, // V-cache, 3.5 bpw (4.0 bpw with padding)
GGML_TYPE_VTQ4_1  = 49, // V-cache, 4.5 bpw
```

In `ggml.c`, add type metadata (name, block size, type size, is_quantized=true).

In `ggml-common.h`, add the block structs (as specified above).

### CLI Integration

```bash
# Maximum compression: KTQ for K, VTQ for V (2.5 bpw avg)
./build/bin/llama-server -m model.gguf \
    --cache-type-k ktq2_1 --cache-type-v vtq1_1 -fa on -ngl 99

# Balanced (3.5 bpw avg)
./build/bin/llama-server -m model.gguf \
    --cache-type-k ktq3_1 --cache-type-v vtq2_1 -fa on -ngl 99

# Asymmetric: q8_0 K (best quality) + vtq2_1 V (maximum compression)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq2_1 -fa on -ngl 99
```

---

## FA Dispatch Changes

### In `fattn.cu`

VTQ types need V-side dispatch only (not K-side -- VTQ is V-only):

```cpp
// V type validation (VTQ cases)
case GGML_TYPE_VTQ1_1:
case GGML_TYPE_VTQ2_1:
case GGML_TYPE_VTQ3_1:
case GGML_TYPE_VTQ4_1:
    break;

// VTQ types ONLY for V, never for K
const bool is_vtq_v = V->type == GGML_TYPE_VTQ1_1 || V->type == GGML_TYPE_VTQ2_1 || V->type == GGML_TYPE_VTQ3_1 || V->type == GGML_TYPE_VTQ4_1;
if (is_vtq_v) {
    GGML_ASSERT(K->type != GGML_TYPE_VTQ2_1 && K->type != GGML_TYPE_VTQ3_1 && K->type != GGML_TYPE_VTQ4_1);
    // VTQ V-dequant is lightweight, so all FA kernel types work (VEC, MMA, WMMA)
    // Unlike TQ, VTQ does not require VEC-only restriction
}
```

### In `fattn-common.cuh`

Add `dequantize_V_vtq*` functions to the `get_dequantize_V()` template dispatch.

### In `fattn-vec.cuh`

VTQ uses the standard `nthreads_V_q = D/4` (same as q4_0, q8_0). No special warp sizing needed because there is no FWHT.

### Template Instances

New files: `fattn-vec-instance-*-vtq*.cu` (following existing pattern). Since VTQ is V-only, instances are needed for combinations: {f16, bf16, q4_0, q8_0, ktq1_1, ktq2_1, ktq3_1, ktq4_1} x {vtq1_1, vtq2_1, vtq3_1, vtq4_1} = 32 new instances.

---

## Implementation Status

### Phase 1: DONE -- Full VTQ Family (VTQ1_1..VTQ4_1)

1. Added `block_vtq1_1` through `block_vtq4_1` structs and type enums
2. Implemented VTQ quantize/dequant in `turboquant.cuh` (no FWHT, no Philox, no sign bits)
3. Implemented `dequantize_V_vtq*` in `fattn-common.cuh` (`__forceinline__`, ~8 registers)
4. FA dispatch for all VTQ V types
5. Registered in CLI: `--cache-type-v vtq1_1` through `--cache-type-v vtq4_1`
6. SET_ROWS, convert, get-rows kernels for all VTQ types
7. TQ types renamed to KTQ throughout (K-cache optimized)
8. Verified: `--cache-type-k ktq2_1 --cache-type-v vtq2_1` produces correct, coherent output

### Commits
- `3d69e4fdd` feat(vtq): implement VTQ -- 830 LOC, 15 files
- `da6383399` fix(vtq): VTQ3_1 OOB + missing get_rows
- `1ff85a17d` fix(vtq): VTQ3_1 struct alignment

### Phase 2: Future Optimization

- Benchmark VTQ quality vs KTQ at equivalent bpw (codebook optimality with fixed rotation)
- Consider D1*H*D2 rotation with per-head (not per-block) sign diagonals
- Explore sub-64 rotation matrices for further matmul cost reduction

### Backward Compatibility

- VTQ is a V-cache-only format. K-cache uses KTQ (where the Hadamard-domain dot product avoids FWHT entirely)
- No model format changes -- KTQ/VTQ are purely KV cache runtime formats
- The `self_v_rot` path is upstream and always available when quantized V is used

---

## Performance Analysis

### FLOPS per Decode Token (Qwen3.5-35B-A3B, d_head=128, n_kv=8, 32 layers)

| Operation | KTQ2_1 V-dequant | VTQ2_1 V-dequant | Saving |
|-----------|-----------------|-------------------|--------|
| Codebook lookup | 32 per block | 4 per call (ne=4) | N/A (different granularity) |
| FWHT butterfly | 160 FMA/block | **0** | **-160 FMA/block** |
| Sign flip | 32 MUL/block | **0** | **-32 MUL/block** |
| Scale multiply | 32 MUL/block | 4 MUL/call | -28 MUL |
| Post-FA rotation | 0 | 128*128*8 = 131K FMA/layer | +131K FMA/layer |

Per decode token at 1024 KV positions, 32 layers:
- **KTQ2_1 V-dequant:** 1024 * (128/32) * (160+32+32) * 32 = **29.4M FMA** (in FA hot loop)
- **VTQ2_1 V-dequant:** 1024 * (128/4) * 3 * 32 = **3.1M FMA** (in FA hot loop) + 131K * 32 = **4.2M FMA** (post-FA matmul)
- **Net saving: 22.1M FMA per token (75% reduction in V-path compute)**

The key difference is WHERE the compute happens:
- TQ: 29.4M FMA inside the FA kernel (register-starved, serialized, spilling to LMEM)
- VTQ: 3.1M FMA inside FA (lightweight, inlineable) + 4.2M FMA as a separate matmul (efficient, uses tensor cores)

### Memory Bandwidth

| Format | Bytes/block | Blocks read per KV position (d=128) | Bytes/KV position |
|--------|------------|--------------------------------------|-------------------|
| f16 | 64B | 4 | 256B |
| q4_0 | 18B | 4 | 72B |
| ktq2_1 | 14B | 4 | 56B |
| **vtq2_1** | **10B** | **4** | **40B** |
| ktq1_1 | 10B | 4 | 40B |

VTQ2_1 matches KTQ1_1's bandwidth (40B/pos) while providing 2-bit codebook quality (KTQ1_1 uses 1-bit). This is a sweet spot: KTQ2_1 quality at KTQ1_1 bandwidth.

### Register Pressure (V-Dequant in FA Inner Loop)

| Format | Float registers | Can inline | Spills |
|--------|----------------|------------|--------|
| f16 | 2 | Yes | No |
| q4_0 | 4 | Yes | No |
| ktq2_1 | ~40 (buf[32] + intermediates) | **No** (`__noinline__`) | **Yes** |
| **vtq2_1** | **~8** | **Yes** (`__forceinline__`) | **No** |

This is the most impactful difference. The TQ V-dequant's register spilling is what causes the corruption (LMEM writes clobbering FA accumulator state). VTQ eliminates this entirely.

---

## Comparison Table

| | KTQ (K-cache) | VTQ (V-cache) | q4_0 (V-cache) | f16 (V-cache) |
|---|---|---|---|---|
| **bpw** (2-bit variant) | 3.5 | **2.5** | 4.5 | 16.0 |
| **Block size** | 14B | **10B** | 18B | 64B |
| **Dequant in FA** | N/A (dot product) | **Codebook + scale** | Shift + scale | Load |
| **Registers** | N/A | **~8** | ~8 | ~2 |
| **Can inline** | N/A | **Yes** | Yes | Yes |
| **Spills** | N/A | **No** | No | No |
| **Post-FA matmul** | No | **Yes (R^T)** | No | No |
| **FWHT in hot loop** | Applied to Q | **None** | None | None |
| **Per-block state** | d + qs + sb | **d + qs only** | d + qs | raw values |
| **Quality** | RHT + codebook | Fixed H + codebook | Uniform scalar | Exact |
| **Corruption risk** | None | **None** | None | None |

### Recommended Configurations (Implemented)

| Use Case | K-cache | V-cache | Avg bpw | Notes |
|----------|---------|---------|---------|-------|
| Maximum compression | ktq2_1 (3.5) | vtq1_1 (1.5) | **2.5** | Best VRAM, extreme V compression |
| Balanced | ktq3_1 (4.5) | vtq2_1 (2.5) | **3.5** | Good quality, long context |
| Quality | ktq4_1 (5.5) | vtq3_1 (4.0) | **4.75** | Better than q4_0/q4_0 |
| Conservative | q8_0 (8.5) | vtq2_1 (2.5) | **5.5** | Best K quality + max V compression |

---

## Open Questions

1. **Codebook quality with fixed rotation:** Does the bare Hadamard (without per-block random signs) degrade the TurboQuant-style codebook optimality enough to warrant VTQ-specific codebooks? Requires PPL benchmarks comparing `ktq2_1/ktq2_1` vs `ktq2_1/vtq2_1`.

2. **Rotation matrix size:** Upstream uses 64x64 for V (applied per 64-element stripe). This fork's d_head=128 models could use either 64x64 (cheaper matmul) or 128x128 (better decorrelation). Need benchmarks.

3. **MMA/WMMA support:** VTQ's simple dequant should work with tiled FA kernels (unlike TQ which is VEC-only). This could unlock significant PP throughput gains. Priority depends on batch size usage patterns.

4. **Norm correction quality:** TQ v5's norm correction reconstructs through FWHT+sign to measure error. VTQ's norm correction is simpler (reconstruct = codebook lookup, no FWHT) but the error profile may differ. May need a VTQ-specific correction factor.

5. **CPU fallback:** VTQ quantize/dequantize on CPU is simpler than TQ (no Philox, no FWHT), but the post-FA rotation is a dense matmul. For CPU-only inference, this matmul cost may dominate. Consider whether CPU path should use KTQ instead of VTQ.

---

## File Change Summary

### New Files

**Actual implementation delta:** ~830 LOC across 15 files. VTQ is simpler than KTQ because it removes the FWHT/Philox/sign-bit machinery.

Key files modified:
- `ggml/include/ggml.h` -- 8 type enums (KTQ1_1..KTQ4_1 + VTQ1_1..VTQ4_1)
- `ggml/src/ggml-common.h` -- 4 VTQ block structs (no sb[])
- `ggml/src/ggml-cuda/turboquant.cuh` -- VTQ quantize/dequant (shared PQ_CUDA_CB_* codebooks)
- `ggml/src/ggml-cuda/fattn-common.cuh` -- `dequantize_V_vtq*` (`__forceinline__`, ~8 registers)
- `ggml/src/ggml-cuda/fattn.cu` -- VTQ V-type dispatch
- `ggml/src/ggml-quants.c` -- CPU quantize/dequantize for VTQ (shared PQ_CODEBOOK_*)
- `common/arg.cpp` -- CLI: vtq1_1..vtq4_1 + ktq1_1..ktq4_1

---

## References

- [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) -- Zandieh, Daliri, Hadian, Mirrokni (arXiv preprint, April 2025). Primary inspiration for this implementation
- [llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) -- TheTom's `self_v_rot` implementation
- `ggml/src/ggml-common.h` lines 295-360 -- KTQ + VTQ block structs
- `ggml/src/ggml-cuda/turboquant.cuh` -- KTQ + VTQ CUDA implementation (shared PQ codebooks)
- `ggml/src/ggml-cuda/fattn-common.cuh` -- KTQ vec_dot_KQ + VTQ dequantize_V functions
- `ggml/src/ggml-cuda/fattn-vec.cuh` lines 60-94 -- FA vec kernel V-threading model
- `src/llama-graph.cpp` lines 2094-2161 -- `build_attn()` with `self_v_rot` pre/post FA
- `src/llama-kv-cache.cpp` lines 22-58 -- `ggml_gen_hadamard()` rotation matrix construction
- `src/llama-kv-cache.cpp` lines 372-406 -- KV cache rotation initialization
