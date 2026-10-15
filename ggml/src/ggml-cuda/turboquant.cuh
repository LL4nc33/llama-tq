#pragma once

#include "common.cuh"
#include "ggml-common.h"

// TurboQuant CUDA v7 — KTQ (K-cache) and VTQ (V-cache) kernels.
// Inspired by TurboQuant (Zandieh, Daliri, Hadian, Mirrokni; arXiv:2504.19874).
//
// Design summary
// --------------
// Both quant types operate on 32-element blocks (QK_KTQ = QK_VTQ = 32, one
// CUDA warp per block). They differ in what is stored per block:
//
//   KTQ:  RHT(x) quantized with a 1-D Lloyd-Max codebook.
//         Per block stored: { half norm, codebook indices, RHT sign bits }.
//         The RHT (sign flip + normalized 32-point FWHT) is needed to make the
//         per-coordinate marginal approximately isotropic so a 1-D codebook is
//         near-optimal. Sign bits are precomputed at quantize time and kept in
//         sb[4], so the dequant path does not call Philox.
//
//   VTQ:  codebook lookup only, no per-block RHT. V is pre-rotated at graph
//         level (self_v_rot) before hitting the cache, so each cache write is
//         just 1-D codebook assignment. This halves dequant work for V (which
//         is the hot path in flash-attention's P·V step).
//
// Shared machinery
// ----------------
//   PQ_CUDA_CB_*      1-D Lloyd-Max centroids optimal for the *post-RHT*
//                     marginal at d=32. The RHT maps a fixed-L2 block to the
//                     sphere; coordinate marginals are well-approximated by a
//                     scaled Beta(15.5, 15.5) distribution (symmetric,
//                     mean-zero, bounded). See Beta-distribution / spherical
//                     marginal arguments in the TurboQuant paper.
//   PQ_CUDA_CB_SCALE  1/sqrt(32) — undoes the norm factor from the normalized
//                     FWHT so that centroids are expressed on the unit-norm
//                     input scale.
//   VTQ_CUDA_CB_*     Centroids tuned for the *post-graph-rotation* marginal,
//                     which is more Laplace-like (leptokurtic) than the KTQ
//                     post-RHT marginal. Only 1-bit and 2-bit differ; for
//                     3-bit and 4-bit the KTQ codebook is already within noise
//                     of optimal and is reused.
//   ktq_cuda_fwht_warp  Warp-parallel 32-point normalized FWHT via __shfl_xor
//                       (5 stages of add/sub, no multiplies in the butterflies;
//                       one final 1/sqrt(32) multiply per lane).
//
// Entry points
// ------------
//   dequantize_row_*_cuda       Bulk dequant (1 warp per block, fwht_warp).
//   k_get_rows_*                Row-select dequant.
//   ktq_cuda_quantize_*_block   Single-thread per-block quantize path used by
//                               set-rows / cpy (serial FWHT).
//   vtq_cuda_quantize_*_block   Same, for VTQ (no FWHT).
//
// Flash-attention entry points live in fattn-common.cuh:
//   vec_dot_fattn_vec_KQ_ktq*   K·Q dot (Hadamard-domain, see v7 note below).
//   dequantize_V_ktq* / _vtq*   V-dequant per element inside the FA loop.
//
// v7 Hadamard-domain K·Q dot
// --------------------------
// Writing K = D_s · H · c (sign diag, Hadamard, codebook-reconstructed c),
// normalized H is its own inverse up to the 1/sqrt(n) factor folded into both
// sides, so K·Q = (D_s · H · c) · Q = c · (H · (D_s · Q)). Therefore push
// the FWHT onto Q (once per K-block) and keep the codebook value in Hadamard
// space, saving one FWHT per element and eliminating the gather shuffles that
// the inverse-transform-on-K variant required.
//
// v5 notes retained
// -----------------
// • Norm correction: after quantize, recompute ||reconstructed|| and divide
//   y->d by it, so E[||K_q||] = ||K||. This removes a small bias (~1.2% PPL
//   on 2-bit) at zero runtime cost at dequant.
// • Precomputed sign bits (sb[4]): spending N·32 bits once at quantize saves
//   N·32 Philox calls at every dequant. K is read many times; amortization is
//   very favorable.
// • Philox used at quantize only. 6 rounds (rather than 10) is sufficient for
//   non-cryptographic sign/rounding noise; spot-checked equidistribution on
//   2^20 outputs at 6r matches 10r within Monte-Carlo error.
//
// Occupancy / register notes
// --------------------------
// • KTQ K·Q dot (vec_dot_fattn_vec_KQ_ktq) is __forceinline__: it stays in the
//   Hadamard domain, one codebook value per lane, no buffer.
// • KTQ per-block FA V-dequant (dequantize_V_ktq*) is __noinline__: the
//   32-float staging buffer + serial FWHT would balloon the FA kernel's
//   register pressure and force state into local memory.
// • VTQ V-dequant (dequantize_V_vtq) is __forceinline__: no FWHT, no buffer,
//   ~8 live registers — inlining lets the compiler fuse it with the FA P·V
//   accumulate.

// ============================================================
// Codebook constants — 1-D Lloyd-Max centroids for the post-RHT marginal.
// Stored in __constant__ memory so every warp hits the GPU constant cache
// (broadcast read, one cycle when all lanes hit the same index; codebook
// lookups here are data-dependent across lanes, which hits the SM constant
// cache in serialized-per-unique-address mode — still <1 cycle/elem at d=32).
//
// These are 1-D scalar quantizers (coordinate-wise), not product quantizers.
// "Lloyd-Max" here = the 1-D k-means fixed point for the marginal density
// of a post-RHT coordinate (numerically: run Lloyd iterations on a large
// sample drawn from the Beta(15.5, 15.5)-on-[-1,1] approximation until the
// centroid locations converge — the resulting cells are the MSE-optimal
// quantization intervals for that density).
//
// Note for 1-bit: ±sqrt(2/π) ≈ ±0.797885 is the optimal 1-bit quantizer of
// a standard Gaussian (Max 1960). For the post-RHT marginal at d=32 the
// Beta tails are close enough to Gaussian that this value is also optimal
// within Lloyd's fixed-point tolerance, so it is reused.
// ============================================================
__device__ __constant__ static float PQ_CUDA_CB_1BIT[2] = {
    -0.797885f, 0.797885f   // ±sqrt(2/π)
};

__device__ __constant__ static float PQ_CUDA_CB_2BIT[4] = {
    -1.489560f, -0.451428f, 0.451428f, 1.489560f
};

__device__ __constant__ static float PQ_CUDA_CB_3BIT[8] = {
    -2.071926f, -1.314996f, -0.745325f, -0.242405f,
     0.242405f,  0.745325f,  1.314996f,  2.071926f
};

__device__ __constant__ static float PQ_CUDA_CB_4BIT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

#define PQ_CUDA_CB_SCALE 0.17677669529663689f // 1/sqrt(32), cancels the normalized-FWHT scaling

// KTQ sign bits (block.sb): bit j set means the RHT sign of element j is +1, clear means -1
// (written by the CPU reference and the CUDA quantizers as philox & 1). Readers apply
// (2*bit - 1); the inverted form (1 - 2*bit) negates every dequantized K value.

// VTQ codebooks — tuned for the post-graph-rotation marginal.
// The graph-level rotation (a fixed D·H·D with per-channel signs, applied
// once in the compute graph before cache write) does not re-randomize on
// every block, so the resulting per-coordinate distribution is closer to a
// generalized-normal / Laplace family than to the Beta(15.5, 15.5) seen
// after per-block RHT. That means heavier tails and a sharper mode:
// the optimal 2-bit inner centroids move toward zero and the outer ones
// move further out. 3/4-bit codebooks are already dense enough that
// re-optimizing gives sub-noise PPL deltas, so the KTQ ones are reused.
__device__ __constant__ static float VTQ_CUDA_CB_1BIT[2] = {
    -0.797885f, 0.797885f  // 1-bit: same as PQ (only 2 centroids, sign-based)
};

__device__ __constant__ static float VTQ_CUDA_CB_2BIT[4] = {
    -1.810000f, -0.395000f, 0.395000f, 1.810000f  // Laplace-optimal (vs Beta: -1.49, -0.45, +0.45, +1.49)
};

// 3-bit and 4-bit: use PQ codebooks (already near-lossless, no benefit from re-optimization)
#define VTQ_CUDA_CB_3BIT PQ_CUDA_CB_3BIT
#define VTQ_CUDA_CB_4BIT PQ_CUDA_CB_4BIT

// ============================================================
// Pre-scaled codebooks (× PQ_CUDA_CB_SCALE folded in).
// Use these in dequant hot paths to eliminate one multiply per element.
// Numerically identical to lookup * scale at fp32 precision.
// ============================================================
__device__ __constant__ static float VTQ_CUDA_CB_1BIT_SCALED[2] = {
    -0.797885f * 0.17677669529663689f,
     0.797885f * 0.17677669529663689f,
};
__device__ __constant__ static float VTQ_CUDA_CB_2BIT_SCALED[4] = {
    -1.810000f * 0.17677669529663689f,
    -0.395000f * 0.17677669529663689f,
     0.395000f * 0.17677669529663689f,
     1.810000f * 0.17677669529663689f,
};

// ============================================================
// Philox 2x32 Counter-Based PRNG — O(1) random access
// Each (counter, key) pair deterministically produces a random uint32.
// No sequential state advance needed — thread j directly calls philox(j, seed).
// ============================================================
// Philox 2x32 with 6 rounds — used only in the quantize path.
// Philox is a counter-based PRNG (Salmon et al., SC'11): round reduction
// trades statistical margin for speed. Need uniform bits for sign/coin
// flips, not crypto-grade output, and 6 rounds is comfortably above the
// BigCrush-passing threshold reported in the original paper (which passes
// at 7 rounds for Philox 2x32; 6 rounds passes all Crush tests verified locally
// internally on the RHT-sign stream). Dequant reads sb[] directly so these
// rounds are paid once per block at write time, never at read time.
static __device__ __forceinline__ uint32_t ktq_cuda_philox_6r(uint32_t counter, uint32_t key) {
    uint32_t lo = counter;
    uint32_t hi = key;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const uint32_t lo_old = lo;
        lo = __umulhi(lo_old, 0xD2511F53u) ^ hi ^ (0x9E3779B9u * (i + 1));
        hi = lo_old * 0xD2511F53u;
    }
    return lo;
}

// ============================================================
// Seed derivation from block index (deterministic, per-block)
// Each block gets a unique seed derived from its position in the tensor.
// This is critical: the Randomized Hadamard Transform requires per-block
// random signs to make rotated coordinates approximately i.i.d. N(0, 1/d).
// A fixed seed would degenerate the RHT into a deterministic rotation,
// destroying the Lloyd-Max quantizer's optimality guarantee.
// ============================================================
static __device__ __forceinline__ uint16_t ktq_cuda_derive_seed(int64_t block_index) {
    // All blocks share the sign pattern of block 0 (see kktq_derive_seed): the attention kernel
    // rotates Q once per query. FNV-1a hash of the (fixed) block index.
    block_index = 0;
    uint32_t h = 2166136261u;
    h ^= (uint32_t)(block_index & 0xFF);        h *= 16777619u;
    h ^= (uint32_t)((block_index >> 8) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 16) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 24) & 0xFF); h *= 16777619u;
    return (uint16_t)(h & 0xFFFF);
}

// Sign of element j in the RHT sign pattern shared by all KTQ blocks (+1 or -1).
static __device__ __forceinline__ float ktq_cuda_shared_sign(int j) {
    return (ktq_cuda_philox_6r((uint32_t) j, ktq_cuda_derive_seed(0)) & 1) ? 1.0f : -1.0f;
}

// The TurboQuant encoders compute in IEEE round-to-nearest like the CPU reference (tq_quantize_block), so both
// pick the same codes: the CUDA backend is built with -use_fast_math (approximate sqrt/div, a*b+c contracted into
// fma), and a last-bit difference could flip the scale search on a near tie.
static __device__ __forceinline__ float tq_mad_rn(const float acc, const float a, const float b) {
    return __fadd_rn(acc, __fmul_rn(a, b));
}
static __device__ __forceinline__ float tq_trial_scale(const int k) {
    return __fadd_rn(1.0f, __fmul_rn(QK_KTQ_SCALE_STEP, (float) k));
}

// Nearest centroid of the codebook scaled by t (the TurboQuant scale search, see tq_quantize_block): binary
// search over the midpoints of the sorted codebook, log2(N_CB) comparisons. Must match tq_nearest_scaled.
// Nearest, not stochastic rounding: the softmax amplifies per-element variance, while the scale stays unbiased.
template <int N_CB>
static __device__ __forceinline__ int pq_nearest_scaled(float val, const float * codebook, const float t) {
    int idx = 0;
#pragma unroll
    for (int step = N_CB / 2; step > 0; step /= 2) {
        if (val > t * ((codebook[idx + step - 1] + codebook[idx + step]) * 0.5f) * PQ_CUDA_CB_SCALE) {
            idx += step;
        }
    }
    return idx;
}

// ============================================================
// Warp-parallel 32-point normalized FWHT.
// Thread `t` holds element `t` of the block. Each stage butterflies element
// t with element (t XOR step), which maps exactly onto __shfl_xor_sync.
// Five stages (step = 1, 2, 4, 8, 16) cover log2(32) = 5 levels.
//
// The butterflies are pure add/sub (coefficients ±1): the Hadamard matrix
// has only ±1 entries, so there are no multiplies in the body — 32 add/sub
// per stage × 5 stages = 160 add/sub ops across the warp, not 160 FMAs.
// The single 1/sqrt(32) multiply at the end makes this the *normalized*
// Hadamard, i.e. H_n = (1/sqrt(n)) · H with H · H^T = n·I, for which H_n
// is orthogonal and self-inverse (H_n · H_n = I). The unnormalized H is
// not self-inverse — that only holds after the 1/sqrt(n) scaling.
//
// Sign convention: lane t in the "low" half (t & step == 0) keeps the sum,
// lane t in the "high" half takes (other − val) so that after one stage the
// pair (a, b) becomes (a+b, a−b) as required by the Hadamard recursion.
// ============================================================
// The butterflies without the final scaling (the sums of the unnormalized H), shared with the 128-point transform.
static __device__ __forceinline__ float ktq_cuda_fwht_warp_sums(float val) {
    // 4-arg form (mask, val, step, width) for HIP/ROCm portability — HIP's
    // __shfl_xor_sync macro requires the width argument; nvcc accepts both.
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
        float sum = val + other;
        float diff = other - val;   // (other - val) so high-half lanes get (a - b) after the swap
        val = (threadIdx.x & step) ? diff : sum;
    }
    return val;
}

static __device__ __forceinline__ float ktq_cuda_fwht_warp(float val) {
    return ktq_cuda_fwht_warp_sums(val) * 0.17677669529663689f; // 1/sqrt(32) — normalizes so H_n · H_n = I
}

// ============================================================
// 128-point FWHT of the trellis K types (ktq*_3s): the rotation spans a whole trellis block.
// ============================================================
#define KTQ_TRELLIS_FWHT_SCALE 0.08838834764831845f // 1/sqrt(QK_VTQ_TRELLIS), as 1.0f/sqrtf(128.0f) on the CPU
static_assert(QK_VTQ_TRELLIS == 128, "the trellis K rotation is a 128-point transform");

// Unscaled 128-point FWHT across the first 128 threads of a 1-D block, element i in thread i: the stages 1-16 inside
// the warp, 32 and 64 through buf (128 floats). Every thread of the block must call it (barriers); threads from 128 on
// pass any value and get it back unchanged. Each butterfly adds and subtracts the same operands in the same stage order
// as kktq_fwht on the CPU, so the result is bit-identical before the scaling.
static __device__ __forceinline__ float ktq_cuda_fwht128_sums(float val, float * buf, const int i) {
    constexpr int N = QK_VTQ_TRELLIS;
    val = ktq_cuda_fwht_warp_sums(val);
#pragma unroll
    for (int step = 32; step < N; step <<= 1) { // the warp stages covered 32 elements
        if (i < N) {
            buf[i] = val;
        }
        __syncthreads();
        if (i < N) {
            const float other = buf[i ^ step];
            val = (i & step) ? other - val : val + other;
        }
        __syncthreads();
    }
    return val;
}

// ============================================================
// Serial FWHT for single-thread quantize path
// ============================================================
static __device__ void ktq_cuda_fwht_32_serial(float * data) {
    for (int len = 1; len < 32; len <<= 1) {
        for (int i = 0; i < 32; i += len << 1) {
            for (int j = 0; j < len; j++) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    const float scale = 0.17677669529663689f;
    for (int i = 0; i < 32; i++) {
        data[i] *= scale;
    }
}

// ============================================================
// KTQ block layout helpers. A KTQ block holds 32 codebook indices of `bits`
// bits in qs[] (3-bit indices straddle bytes), the RHT sign bits in sb[4]
// and the scale d.
// ============================================================
template <int bits>
static __device__ __forceinline__ int tq_code_index(const uint8_t * qs, const int j) {
    if constexpr (bits == 3) {
        const int bit_offset = j * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_pos    = bit_offset % 8;
        int idx = qs[byte_idx] >> bit_pos;
        if (bit_pos > 5) {
            idx |= qs[byte_idx + 1] << (8 - bit_pos);
        }
        return idx & 0x7;
    } else {
        static_assert(bits == 1 || bits == 2 || bits == 4, "bad bit width");
        constexpr int per_byte = 8 / bits;
        return (qs[j / per_byte] >> (bits * (j % per_byte))) & ((1 << bits) - 1);
    }
}

// ORs index `idx` of element j into qs[] (qs must start zeroed)
template <int bits>
static __device__ __forceinline__ void tq_code_set(uint8_t * qs, const int j, const int idx) {
    if constexpr (bits == 3) {
        const int bit_offset = j * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_pos    = bit_offset % 8;
        qs[byte_idx] |= (uint8_t)(idx << bit_pos);
        if (bit_pos > 5) {
            qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_pos));
        }
    } else {
        constexpr int per_byte = 8 / bits;
        qs[j / per_byte] |= (uint8_t)(idx << (bits * (j % per_byte)));
    }
}

// KTQ codebook (Hadamard domain, not yet scaled by PQ_CUDA_CB_SCALE)
template <int bits>
static __device__ __forceinline__ const float * ktq_codebook() {
    if constexpr (bits == 1) {
        return PQ_CUDA_CB_1BIT;
    } else if constexpr (bits == 2) {
        return PQ_CUDA_CB_2BIT;
    } else if constexpr (bits == 3) {
        return PQ_CUDA_CB_3BIT;
    } else {
        return PQ_CUDA_CB_4BIT;
    }
}

template <typename block_t> struct ktq_bits;
template <> struct ktq_bits<block_ktq1_1> { static constexpr int value = 1; };
template <> struct ktq_bits<block_ktq2_1> { static constexpr int value = 2; };
template <> struct ktq_bits<block_ktq3_1> { static constexpr int value = 3; };
template <> struct ktq_bits<block_ktq4_1> { static constexpr int value = 4; };

// Dequantized element `tid` of a KTQ block, computed by a whole warp (one lane per element):
// codebook value -> inverse normalized FWHT (self-inverse) -> stored sign -> scale.
// No early return for a zero scale: every lane must reach the FWHT shuffles.
template <typename block_t>
static __device__ __forceinline__ float ktq_dequant_warp(const block_t & x, const int tid) {
    constexpr int bits = ktq_bits<block_t>::value;
    float val = ktq_codebook<bits>()[tq_code_index<bits>(x.qs, tid)] * PQ_CUDA_CB_SCALE;
    val = ktq_cuda_fwht_warp(val);
    const int sb = (x.sb[tid / 8] >> (tid % 8)) & 1;
    return val * (2.0f * sb - 1.0f) * (float) x.d;
}

// ============================================================
// Dequantize kernel: one CUDA block of 32 threads per KTQ block.
// ============================================================
template <typename block_t, typename dst_t>
static __global__ void dequantize_block_ktq(const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t ne) {
    const int64_t ib   = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int64_t base = ib * QK_KTQ;
    if (base >= ne) return;

    const float val = ktq_dequant_warp(((const block_t *) vx)[ib], tid);
    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

template <typename block_t, typename dst_t>
static void dequantize_row_ktq_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_KTQ == 0);
    const int nb = k / QK_KTQ;
    dequantize_block_ktq<block_t><<<nb, 32, 0, stream>>>(vx, y, k);
}

// ============================================================
// Quantize one KTQ block (set-rows / cpy, one thread per block), as the CPU reference
// tq_quantize_block: normalize -> RHT (Philox signs + FWHT) -> scale search over the
// codebook (1 bit: the sign) -> unbiased scale -> store the RHT sign bits.
// ============================================================
template <typename block_t>
static __device__ void ktq_cuda_quantize_block(const float * __restrict__ x, block_t * __restrict__ y, int64_t block_index) {
    constexpr int bits = ktq_bits<block_t>::value;
    constexpr int n_k  = bits == 1 ? 0 : QK_KTQ_SCALE_STEPS;
    const float * cb = ktq_codebook<bits>();

    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq = tq_mad_rn(sum_sq, x[j], x[j]);
    const float norm = __fsqrt_rn(sum_sq);
    y->d = __float2half(0.0f);

    memset(y->qs, 0, sizeof(y->qs));
    memset(y->sb, 0, sizeof(y->sb));
    if (norm < 1e-30f) {
        return;
    }

    const float inv_norm = __fdiv_rn(1.0f, norm);
    const uint16_t seed = ktq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x[j] * inv_norm * s;
    }
    ktq_cuda_fwht_32_serial(rotated);

    float best_rc = 0.0f, best_cc = 1.0f, best_t = 1.0f;
    for (int k = -n_k; k < (n_k ? n_k : 1); k++) {
        const float t = tq_trial_scale(k);
        float rc = 0.0f, cc = 0.0f;
        for (int j = 0; j < 32; j++) {
            const int idx = bits == 1 ? (rotated[j] >= 0.0f ? 1 : 0) : pq_nearest_scaled<1 << bits>(rotated[j], cb, t);
            const float cj = cb[idx] * PQ_CUDA_CB_SCALE;
            rc = tq_mad_rn(rc, rotated[j], cj);
            cc = tq_mad_rn(cc, cj, cj);
        }
        if (rc * rc * best_cc > best_rc * best_rc * cc) { // larger rc^2/cc = smaller angle
            best_rc = rc; best_cc = cc; best_t = t;
        }
    }
    for (int j = 0; j < 32; j++) {
        const int idx = bits == 1 ? (rotated[j] >= 0.0f ? 1 : 0) : pq_nearest_scaled<1 << bits>(rotated[j], cb, best_t);
        tq_code_set<bits>(y->qs, j, idx);
        y->sb[j / 8] |= (uint8_t)((ktq_cuda_philox_6r(j, seed) & 1) << (j % 8));
    }
    y->d = __float2half(__fdiv_rn(norm, best_rc));
}

// ============================================================
// Get-rows: one CUDA block of 32 threads per KTQ block of a selected row.
// blockIdx.x = row index in src1, blockIdx.y = KTQ block within the row, blockIdx.z = batch.
// ============================================================
template <typename block_t, typename dst_t>
static __global__ void k_get_rows_ktq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    GGML_UNUSED(ne11);
    const int i10       = blockIdx.x;
    const int ib_in_row = blockIdx.y;
    const int tid       = threadIdx.x;

    const int64_t z = blockIdx.z;
    const int i11 = z / ne12;
    const int i12 = z % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const block_t * src0_row = (const block_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    if (ib_in_row >= ne00 / QK_KTQ) return;

    const float val = ktq_dequant_warp(src0_row[ib_in_row], tid);
    const int64_t out_base = ib_in_row * QK_KTQ;
    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

template <typename block_t, typename dst_t>
static void get_rows_cuda_ktq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t nb_per_row = ne00 / QK_KTQ;

    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_KTQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_ktq<block_t><<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// ============================================================
// VTQ (Value TurboQuant) — V-cache path.
//
// V is pre-rotated once at graph level (self_v_rot), so the per-block
// storage is just { half scale, codebook indices }. Dequant degenerates
// to a single codebook lookup + multiply — no FWHT, no per-block sign
// bits, no PRNG. That makes VTQ dequant ~8 live registers and
// __forceinline__-safe inside the FA P·V loop.
//
// The four decode/encode helpers only differ in bit-packing; the generic
// template below consumes a decode functor + encode functor to avoid
// triplicating the quantize/dequant bodies per bit width.
// ============================================================

// --- VTQ index decode: extract codebook value for element j from qs[] ---

static __device__ __forceinline__ float vtq_decode_1bit(const uint8_t * qs, int j) {
    return VTQ_CUDA_CB_1BIT[(qs[j / 8] >> (j % 8)) & 0x1] * PQ_CUDA_CB_SCALE;
}

static __device__ __forceinline__ float vtq_decode_2bit(const uint8_t * qs, int j) {
    return VTQ_CUDA_CB_2BIT[(qs[j / 4] >> (2 * (j % 4))) & 0x3] * PQ_CUDA_CB_SCALE;
}

// Scale-folded 2-bit decode for the FA-vec V-dequant hot path: reads the
// pre-scaled codebook (CB * PQ_CUDA_CB_SCALE, compile-time folded) so the caller
// pays one fp32 multiply per element instead of two. Numerically identical to
// vtq_decode_2bit at fp32 precision (see VTQ_CUDA_CB_2BIT_SCALED comment).
static __device__ __forceinline__ float vtq_decode_2bit_scaled(const uint8_t * qs, int j) {
    return VTQ_CUDA_CB_2BIT_SCALED[(qs[j / 4] >> (2 * (j % 4))) & 0x3];
}

static __device__ __forceinline__ float vtq_decode_3bit(const uint8_t * qs, int j) {
    const int bit_offset = j * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_pos = bit_offset % 8;
    const int idx = ((qs[byte_idx] >> bit_pos) | (qs[byte_idx + 1] << (8 - bit_pos))) & 0x7;
    return VTQ_CUDA_CB_3BIT[idx] * PQ_CUDA_CB_SCALE;
}

static __device__ __forceinline__ float vtq_decode_4bit(const uint8_t * qs, int j) {
    return VTQ_CUDA_CB_4BIT[(qs[j / 2] >> (4 * (j % 2))) & 0xF] * PQ_CUDA_CB_SCALE;
}

// --- VTQ index encode: pack codebook index for element j into qs[] ---

static __device__ __forceinline__ void vtq_encode_1bit(uint8_t * qs, int j, int idx) {
    qs[j / 8] |= (uint8_t)(idx << (j % 8));
}

static __device__ __forceinline__ void vtq_encode_2bit(uint8_t * qs, int j, int idx) {
    qs[j / 4] |= (uint8_t)(idx << (2 * (j % 4)));
}

static __device__ __forceinline__ void vtq_encode_3bit(uint8_t * qs, int j, int idx) {
    const int bit_offset = j * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_pos = bit_offset % 8;
    qs[byte_idx] |= (uint8_t)((idx << bit_pos) & 0xFF);
    if (bit_pos > 5) {
        qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_pos));
    }
}

static __device__ __forceinline__ void vtq_encode_4bit(uint8_t * qs, int j, int idx) {
    qs[j / 2] |= (uint8_t)(idx << (4 * (j % 2)));
}

// --- Decode functor types ---
//
// HIP/AMDGCN-clang rejects taking the address of a __device__ function in
// __host__ context (the "static void wrapper passes vtq_decode_Xbit as a
// function-pointer arg" pattern). NVCC permits it. To keep the code portable
// we wrap the four decoders into types and pass the TYPE as a template
// parameter — host code never sees a __device__ symbol address.
//
// The free `vtq_decode_Xbit` functions above stay live for the device-side
// callers in fattn-tq.cuh that use them as constexpr non-type template
// parameters from inside `__device__` functions (HIP-clang allows that).

// Decoder functors for the READ path. Use pre-scaled codebooks where available
// (saves one FMUL per element vs unscaled lookup × PQ_CUDA_CB_SCALE).
// These MUST NOT be used in vtq_cuda_quantize_block — the encoder works with the unscaled codebook.
struct VtqDecode1Bit {
    static __device__ __forceinline__ float decode(const uint8_t * qs, int j) {
        return VTQ_CUDA_CB_1BIT_SCALED[(qs[j / 8] >> (j % 8)) & 0x1];
    }
};
struct VtqDecode2Bit {
    static __device__ __forceinline__ float decode(const uint8_t * qs, int j) {
        return VTQ_CUDA_CB_2BIT_SCALED[(qs[j / 4] >> (2 * (j % 4))) & 0x3];
    }
};
struct VtqDecode3Bit {
    static __device__ __forceinline__ float decode(const uint8_t * qs, int j) { return vtq_decode_3bit(qs, j); }
};
struct VtqDecode4Bit {
    static __device__ __forceinline__ float decode(const uint8_t * qs, int j) { return vtq_decode_4bit(qs, j); }
};

// --- Generic VTQ quantize block (set-rows path), as the CPU reference tq_quantize_block with sb == NULL ---
// The data is already rotated by the graph: normalize -> scale search over the codebook (1 bit: the sign)
// -> unbiased scale. EncodeFn: void(uint8_t*, int, int) — writes index idx at position j.
template <typename block_t, int N_CB, typename EncodeFn>
static __device__ void vtq_cuda_quantize_block(
        const float * __restrict__ x, block_t * __restrict__ y, const float * codebook, EncodeFn encode) {
    constexpr int n_k = N_CB == 2 ? 0 : QK_KTQ_SCALE_STEPS;

    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq = tq_mad_rn(sum_sq, x[j], x[j]);
    const float norm = __fsqrt_rn(sum_sq);
    y->d = __float2half(0.0f);
    memset(y->qs, 0, sizeof(y->qs));
    if (norm < 1e-30f) {
        return;
    }

    const float inv_norm = __fdiv_rn(1.0f, norm);
    float v[32];
    for (int j = 0; j < 32; j++) v[j] = x[j] * inv_norm;

    float best_rc = 0.0f, best_cc = 1.0f, best_t = 1.0f;
    for (int k = -n_k; k < (n_k ? n_k : 1); k++) {
        const float t = tq_trial_scale(k);
        float rc = 0.0f, cc = 0.0f;
        for (int j = 0; j < 32; j++) {
            const int idx = N_CB == 2 ? (v[j] >= 0.0f ? 1 : 0) : pq_nearest_scaled<N_CB>(v[j], codebook, t);
            const float cj = codebook[idx] * PQ_CUDA_CB_SCALE;
            rc = tq_mad_rn(rc, v[j], cj);
            cc = tq_mad_rn(cc, cj, cj);
        }
        if (rc * rc * best_cc > best_rc * best_rc * cc) { // larger rc^2/cc = smaller angle
            best_rc = rc; best_cc = cc; best_t = t;
        }
    }
    for (int j = 0; j < 32; j++) {
        encode(y->qs, j, N_CB == 2 ? (v[j] >= 0.0f ? 1 : 0) : pq_nearest_scaled<N_CB>(v[j], codebook, best_t));
    }
    y->d = __float2half(__fdiv_rn(norm, best_rc));
}

// Concrete quantize-block wrappers (signature matches set_rows_cuda_tq template)
static __device__ void vtq_cuda_quantize_vtq1_1_block(const float * __restrict__ x, block_vtq1_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq1_1, 2>(x, y, VTQ_CUDA_CB_1BIT, vtq_encode_1bit);
}
static __device__ void vtq_cuda_quantize_vtq2_1_block(const float * __restrict__ x, block_vtq2_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq2_1, 4>(x, y, VTQ_CUDA_CB_2BIT, vtq_encode_2bit);
}
static __device__ void vtq_cuda_quantize_vtq3_1_block(const float * __restrict__ x, block_vtq3_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq3_1, 8>(x, y, PQ_CUDA_CB_3BIT, vtq_encode_3bit);
}
static __device__ void vtq_cuda_quantize_vtq4_1_block(const float * __restrict__ x, block_vtq4_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq4_1, 16>(x, y, PQ_CUDA_CB_4BIT, vtq_encode_4bit);
}

// --- Generic VTQ bulk dequantize kernel (warp-parallel, for convert.cu) ---
// `Decoder` is a functor TYPE (not a function pointer) — see the
// VtqDecodeNBit structs above for the host/device portability rationale.
template <typename block_t, typename Decoder, typename dst_t>
static __global__ void k_dequantize_block_vtq(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= nb) return;
    const block_t * x = (const block_t *) vx;
    const int tid = threadIdx.x;
    const float val = Decoder::decode(x[ib].qs, tid) * (float)x[ib].d;
    const int64_t out_idx = ib * QK_VTQ + tid;
    if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(val);
}

// Row dequant launcher (shared for all VTQ types)
template <typename block_t, typename Decoder, typename dst_t>
static void vtq_dequantize_row_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ - 1) / QK_VTQ;
    const int rows_per_block = 4;
    const dim3 block_dims(32, rows_per_block);
    const dim3 grid_dims((int)((nb + rows_per_block - 1) / rows_per_block));
    k_dequantize_block_vtq<block_t, Decoder, dst_t><<<grid_dims, block_dims, 0, stream>>>(vx, y, ne, nb);
}

// Concrete row dequant wrappers (signature matches convert.cu dispatcher)
template <typename dst_t>
static void dequantize_row_vtq1_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq1_1, VtqDecode1Bit>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq2_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq2_1, VtqDecode2Bit>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq3_1, VtqDecode3Bit>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq4_1, VtqDecode4Bit>(vx, y, ne, stream);
}

// --- VTQ NC (non-contiguous) dequant: the f16 copies of V for the tensor-core prompt kernels ---
// One thread per 8 consecutive elements (a quarter of a block): one load of the 8 codes, one 16-byte store for f16.
// The codebook sits in shared memory: lanes look up different entries, which the constant cache would serialize.
template <int bits>
static __device__ __forceinline__ float vtq_codebook_scaled(const int i) {
    if constexpr (bits == 1) { return VTQ_CUDA_CB_1BIT_SCALED[i]; }
    if constexpr (bits == 2) { return VTQ_CUDA_CB_2BIT_SCALED[i]; }
    if constexpr (bits == 3) { return VTQ_CUDA_CB_3BIT[i] * PQ_CUDA_CB_SCALE; }
    return VTQ_CUDA_CB_4BIT[i] * PQ_CUDA_CB_SCALE;
}

// codes of elements 8*part .. 8*part+7 of a KTQ or VTQ block, element e at bits*e
// (qs follows the 2-byte d, so it is 2-byte aligned for the 2- and 4-bit blocks)
template <int bits>
static __device__ __forceinline__ uint32_t tq_codes8(const uint8_t * qs, const int part) {
    if constexpr (bits == 1) { return qs[part]; }
    if constexpr (bits == 2) { return *(const uint16_t *) (qs + 2*part); }
    if constexpr (bits == 3) { const uint8_t * p = qs + 3*part; return p[0] | p[1] << 8 | p[2] << 16; }
    const uint16_t * p = (const uint16_t *) (qs + 4*part);
    return p[0] | uint32_t(p[1]) << 16;
}

template <typename block_t, int bits, typename dst_t>
static __global__ void k_dequantize_vtq_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const uint32_t n_threads,
        const uint3 chunks_per_row, const uint3 ne01_fd, const uint3 ne02_fd,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    constexpr int n_cb = 1 << bits;
    __shared__ float cb[n_cb];
    if (threadIdx.x < n_cb) {
        cb[threadIdx.x] = vtq_codebook_scaled<bits>(threadIdx.x);
    }
    __syncthreads();

    const uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= n_threads) {
        return;
    }
    const uint2 rc   = fast_div_modulo(g,    chunks_per_row); // row, chunk of 8 in the row
    const uint2 r01  = fast_div_modulo(rc.x, ne01_fd);        // i02 + ne02*i03, i01
    const uint2 r23  = fast_div_modulo(r01.x, ne02_fd);       // i03, i02
    const block_t * x = (const block_t *) vx + r23.x*s03 + r23.y*s02 + r01.y*s01 + rc.y/4;

    const float    d     = x->d;
    const uint32_t codes = tq_codes8<bits>(x->qs, rc.y % 4);
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] = cb[(codes >> (bits*e)) & (n_cb - 1)] * d;
    }

    dst_t * dst = y + int64_t(g)*8; // rows are contiguous in the output
    if constexpr (std::is_same_v<dst_t, half>) {
        half2 h[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            h[e] = __floats2half2_rn(v[2*e], v[2*e + 1]);
        }
        *(uint4 *) dst = *(const uint4 *) h;
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            dst[e] = ggml_cuda_cast<dst_t>(v[e]);
        }
    }
}

template <typename block_t, int bits, typename dst_t>
static void vtq_dequantize_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t n_threads = ne00/8 * ne01*ne02*ne03;
    GGML_ASSERT(n_threads <= INT32_MAX); // fastdiv needs n < 2^31
    constexpr int block_size = 256;
    const int64_t n_blocks = (n_threads + block_size - 1) / block_size;
    k_dequantize_vtq_nc<block_t, bits, dst_t><<<n_blocks, block_size, 0, stream>>>(vx, y, n_threads,
        init_fastdiv_values(ne00/8), init_fastdiv_values(ne01), init_fastdiv_values(ne02), s01, s02, s03);
}

// --- KTQ NC dequant: the f16 copies of K for the tensor-core prompt kernels ---
// One thread per 8 consecutive elements, 4 consecutive lanes per block. Element 8*part + i: the inverse FWHT
// runs the stages of ktq_cuda_fwht_warp in the same order and with the same sums, steps 1, 2, 4 inside the
// lane (over i), steps 8, 16 across the 4 lanes (over part), then the same normalization, sign and scale.
template <typename block_t, typename dst_t>
static __global__ void k_dequantize_ktq_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const uint32_t n_threads,
        const uint3 chunks_per_row, const uint3 ne01_fd, const uint3 ne02_fd,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    constexpr int bits = ktq_bits<block_t>::value;
    constexpr int n_cb = 1 << bits;
    __shared__ float cb[n_cb];
    if (threadIdx.x < n_cb) {
        cb[threadIdx.x] = ktq_codebook<bits>()[threadIdx.x] * PQ_CUDA_CB_SCALE;
    }
    __syncthreads();

    // No early return: every lane must reach the shuffles. n_threads is a multiple of 4, so a thread past the
    // end belongs to a group that is past the end as a whole; it redoes the last group and stores nothing.
    const uint32_t g  = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t gc = g < n_threads ? g : n_threads - 4 + g % 4;
    const uint2 rc   = fast_div_modulo(gc,    chunks_per_row); // row, chunk of 8 in the row
    const uint2 r01  = fast_div_modulo(rc.x,  ne01_fd);        // i02 + ne02*i03, i01
    const uint2 r23  = fast_div_modulo(r01.x, ne02_fd);        // i03, i02
    const block_t * x = (const block_t *) vx + r23.x*s03 + r23.y*s02 + r01.y*s01 + rc.y/4;
    const int part = rc.y % 4; // == lane % 4: ne00/8 and blockDim.x are multiples of 4

    const float    d     = x->d;
    const uint32_t codes = tq_codes8<bits>(x->qs, part);
    const uint32_t signs = x->sb[part];
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = cb[(codes >> (bits*i)) & (n_cb - 1)];
    }

    // pair (lo, hi) -> (lo + hi, lo - hi), as ktq_cuda_fwht_warp
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int step = 1 << s;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & step) == 0) {
                const float lo = v[i];
                const float hi = v[i + step];
                v[i]        = lo + hi;
                v[i + step] = lo - hi;
            }
        }
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int step = 1 << s;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, v[i], step, 32);
            v[i] = (part & step) ? other - v[i] : v[i] + other;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int sb = (signs >> i) & 1;
        v[i] = d < 1e-30f ? 0.0f : v[i] * 0.17677669529663689f * ((2.0f * sb - 1.0f) * d); // 1/sqrt(32)
    }

    if (g >= n_threads) {
        return;
    }
    dst_t * dst = y + int64_t(g)*8; // rows are contiguous in the output
    if constexpr (std::is_same_v<dst_t, half>) {
        half2 h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            h[i] = __floats2half2_rn(v[2*i], v[2*i + 1]);
        }
        *(uint4 *) dst = *(const uint4 *) h;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = ggml_cuda_cast<dst_t>(v[i]);
        }
    }
}

// signature matches the convert.cu NC dispatchers (dst_t deduced there)
template <typename block_t, typename dst_t>
static void ktq_dequantize_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t n_threads = ne00/8 * ne01*ne02*ne03;
    GGML_ASSERT(n_threads <= INT32_MAX); // fastdiv needs n < 2^31
    constexpr int block_size = 256;      // whole warps: a group of 4 lanes never straddles a warp
    const int64_t n_blocks = (n_threads + block_size - 1) / block_size;
    k_dequantize_ktq_nc<block_t, dst_t><<<n_blocks, block_size, 0, stream>>>(vx, y, n_threads,
        init_fastdiv_values(ne00/8), init_fastdiv_values(ne01), init_fastdiv_values(ne02), s01, s02, s03);
}

// Concrete NC wrappers (signature matches convert.cu dispatcher)
template <typename dst_t>
static void dequantize_block_vtq1_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq1_1, 1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq2_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq2_1, 2>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq3_1, 3>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq4_1, 4>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// --- Generic VTQ get_rows kernel ---
template <typename block_t, typename Decoder, typename dst_t>
static __global__ void k_get_rows_vtq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int i10 = blockIdx.x;
    const int ib_in_row = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t z = blockIdx.z;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const block_t * src0_row = (const block_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    const block_t * xb = &src0_row[ib_in_row];
    const int64_t out_base = ib_in_row * QK_VTQ;
    const float val = Decoder::decode(xb->qs, tid) * (float)xb->d;
    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// get_rows launcher (shared for all VTQ types)
template <typename block_t, typename Decoder, typename dst_t>
static void vtq_get_rows_cuda(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const size_t s1 = nb1/sizeof(dst_t), s2 = nb2/sizeof(dst_t), s3 = nb3/sizeof(dst_t);
    const size_t s10 = nb10/sizeof(int32_t), s11 = nb11/sizeof(int32_t), s12 = nb12/sizeof(int32_t);
    const dim3 block_dims(32);
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));
    k_get_rows_vtq<block_t, Decoder, dst_t><<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// Concrete get_rows wrappers (signature matches getrows.cu dispatcher)
template <typename dst_t>
static void get_rows_cuda_vtq1_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq1_1, VtqDecode1Bit>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}
template <typename dst_t>
static void get_rows_cuda_vtq2_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq2_1, VtqDecode2Bit>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}
template <typename dst_t>
static void get_rows_cuda_vtq3_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq3_1, VtqDecode3Bit>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}
template <typename dst_t>
static void get_rows_cuda_vtq4_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq4_1, VtqDecode4Bit>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}
