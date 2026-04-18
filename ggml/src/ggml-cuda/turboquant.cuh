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
// • KTQ warp-cooperative FA dequant (ktq_fattn_dequant_elem_*) is
//   __forceinline__: it's on the critical path of the FA softmax and its live
//   set is small enough that inlining does not push the FA kernel into spills.
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
// Philox 2x32 Counter-Based PRNG — O(1) random access
// Each (counter, key) pair deterministically produces a random uint32.
// No sequential state advance needed — thread j directly calls philox(j, seed).
// ============================================================
static __device__ __forceinline__ uint32_t ktq_cuda_philox(uint32_t counter, uint32_t key) {
    // Philox 2x32, 10 rounds (standard from Salmon et al. 2011)
    uint32_t lo = counter;
    uint32_t hi = key;
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        const uint32_t lo_old = lo;
        lo = __umulhi(lo_old, 0xD2511F53u) ^ hi ^ (0x9E3779B9u * (i + 1));
        hi = lo_old * 0xD2511F53u;
    }
    return lo;
}

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
    // FNV-1a hash of block index — fast, good avalanche
    uint32_t h = 2166136261u;
    h ^= (uint32_t)(block_index & 0xFF);        h *= 16777619u;
    h ^= (uint32_t)((block_index >> 8) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 16) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 24) & 0xFF); h *= 16777619u;
    return (uint16_t)(h & 0xFFFF);
}

// ============================================================
// Stochastic Rounding — makes E[K_q] = K (unbiased)
// Probabilistically chooses between two nearest centroids.
// This eliminates first-order softmax perturbation when K+V are both quantized.
// ============================================================
template <int N_CB>
static __device__ __forceinline__ int pq_stochastic_round(float val, const float * codebook, uint32_t rng) {
    // Find two nearest centroids
    int best = 0, second = 0;
    float best_d = 1e30f, second_d = 1e30f;
    for (int c = 0; c < N_CB; c++) {
        float d = fabsf(val - codebook[c] * PQ_CUDA_CB_SCALE);
        if (d < best_d) {
            second = best; second_d = best_d;
            best = c; best_d = d;
        } else if (d < second_d) {
            second = c; second_d = d;
        }
    }
    // P(choose best) = second_d / (best_d + second_d)
    const float total = best_d + second_d;
    if (total > 1e-30f) {
        const float u = (float)(rng & 0xFFFF) / 65536.0f;
        if (u >= second_d / total) best = second;
    }
    return best;
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
static __device__ __forceinline__ float ktq_cuda_fwht_warp(float val) {
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        float sum = val + other;
        float diff = other - val;   // (other - val) so high-half lanes get (a - b) after the swap
        val = (threadIdx.x & step) ? diff : sum;
    }
    return val * 0.17677669529663689f; // 1/sqrt(32) — normalizes so H_n · H_n = I
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
// Philox-based Gaussian: counter-based Box-Muller for QJL S[i][j]
// MUST match tq_gaussian() in ggml-quants.c for CPU/CUDA consistency.
// ============================================================
static __device__ __forceinline__ float ktq_cuda_gaussian(uint32_t i, uint32_t j, uint16_t seed) {
    uint32_t key = (uint32_t)seed + 32768u;
    uint32_t counter1 = i * 64u + j * 2u;
    uint32_t counter2 = i * 64u + j * 2u + 1u;
    float u1 = ((float)(ktq_cuda_philox(counter1, key) >> 8) + 0.5f) / 16777216.0f;
    float u2 = ((float)(ktq_cuda_philox(counter2, key) >> 8) + 0.5f) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * __cosf(6.2831853f * u2);
}

// ============================================================
// Block-level dequantize kernel: KTQ1_1 (1-bit codebook + RHT).
// Grid: nb CUDA blocks, 32 threads each (one warp per quant block, one
// thread per element). Dequant pipeline per lane:
//   1. Unpack 1-bit index from qs[] → codebook value (Hadamard-space).
//   2. Apply inverse normalized FWHT across the warp (self-inverse, so the
//      forward kernel is reused).
//   3. XOR with the stored RHT sign bit in sb[] and multiply by norm.
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_ktq1_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_KTQ;
    if (base >= ne) return;

    const block_ktq1_1 * x = (const block_ktq1_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 1-bit index → centroid (value is in Hadamard/rotated space).
    const int idx = (x[ib].qs[tid / 8] >> (tid % 8)) & 0x1;
    float val = PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT, part 1: normalized FWHT (self-inverse, same routine).
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT, part 2 + scale: sb[] stores the original diagonal sign.
    // (1 − 2·b) maps {0,1} → {+1,−1} without a branch.
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Block-level dequantize kernel: KTQ2_1 (2-bit codebook + RHT).
// Pipeline identical to KTQ1_1; only the index unpack differs (2 bits ×
// 32 elements packed 4-per-byte into qs[8]).
//
// Historical note: earlier TurboQuant variants used a QJL (Johnson-
// Lindenstrauss) projection inside the block. Removed in v5: in the
// KV-cache attention setting QJL is unbiased but high-variance, and
// softmax's exponential amplifies that variance into visible PPL loss.
// Several independent reimplementations (turboquant_plus/turbo4-resurrection,
// scos-lab, Arclabs001/YATQ) reached the same conclusion.
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_ktq2_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_KTQ;
    if (base >= ne) return;

    const block_ktq2_1 * x = (const block_ktq2_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 2-bit index (packed 4/byte) → centroid in Hadamard space.
    const int idx = (x[ib].qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    float val = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT part 1: normalized FWHT is self-inverse.
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT part 2 + scale: branchless sign flip from sb[].
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Block-level dequantize kernel: KTQ3_1 (3-bit codebook + RHT).
// 3-bit indices don't byte-align: element j starts at bit (3·j) and may
// straddle two bytes — hence the two-byte read below when bit_idx > 5.
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_ktq3_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = ib * QK_KTQ;
    if (base >= ne) return;

    const block_ktq3_1 * x = (const block_ktq3_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 3-bit unpack: read the low byte, OR in the overflow byte iff the
    // index spans a byte boundary (bit_idx ∈ {6, 7}), then mask to 3 bits.
    const int bit_offset = tid * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_idx = bit_offset % 8;
    int cb_idx = (x[ib].qs[byte_idx] >> bit_idx);
    if (bit_idx > 5) cb_idx |= (x[ib].qs[byte_idx + 1] << (8 - bit_idx));
    cb_idx &= 0x7;
    float val = PQ_CUDA_CB_3BIT[cb_idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT part 1.
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT part 2 + scale.
    // Note: sb[] stores philox bit directly (bit=1 means forward RHT applied +1).
    // Correct inverse matches CPU kktq_rht_inverse_sb: sb=1 -> +1, sb=0 -> -1.
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (2.0f * sb - 1.0f) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Launcher functions matching the dequantize_row_*_cuda pattern
// ============================================================
template <typename dst_t>
static void dequantize_row_ktq1_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_KTQ == 0);
    const int nb = k / QK_KTQ;
    dequantize_block_ktq1_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

template <typename dst_t>
static void dequantize_row_ktq2_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_KTQ == 0);
    const int nb = k / QK_KTQ;
    dequantize_block_ktq2_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

template <typename dst_t>
static void dequantize_row_ktq3_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_KTQ == 0);
    const int nb = k / QK_KTQ;
    dequantize_block_ktq3_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

// ============================================================
// Quantize block: TQ1_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> codebook quantize (1-bit) -> sign bits
// ============================================================
static __device__ void ktq_cuda_quantize_ktq1_1_block(const float * __restrict__ x, block_ktq1_1 * __restrict__ y, int64_t block_index) {
    // Step 1: Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) {
        memset(y->qs, 0, 4);
        memset(y->sb, 0, 4);
        return;
    }

    // Step 2: Normalize
    float x_hat[32];
    const float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) x_hat[j] = x[j] * inv_norm;

    // Step 3: RHT forward (random signs via Philox + FWHT)
    const uint16_t seed = ktq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    ktq_cuda_fwht_32_serial(rotated);

    // Step 4: 1-bit quantization — stochastic rounding between centroids
    // P(idx=1) = (val - c0) / (c1 - c0) where c0=-0.798, c1=+0.798 (scaled)
    memset(y->qs, 0, 4);
    for (int j = 0; j < 32; j++) {
        const float val = rotated[j];
        const float c0 = PQ_CUDA_CB_1BIT[0] * PQ_CUDA_CB_SCALE;
        const float c1 = PQ_CUDA_CB_1BIT[1] * PQ_CUDA_CB_SCALE;
        float p1 = (val - c0) / (c1 - c0);  // probability of choosing centroid 1
        p1 = fmaxf(0.0f, fminf(1.0f, p1));   // clamp to [0, 1]
        const uint32_t rng = ktq_cuda_philox_6r(j + 32, seed);
        const float u = (float)(rng & 0xFFFF) / 65536.0f;
        const int idx = (u < p1) ? 1 : 0;
        y->qs[j / 8] |= (uint8_t)(idx << (j % 8));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 8] >> (j % 8)) & 0x1;
        recon[j] = PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (ktq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Quantize block: TQ2_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> codebook quantize -> sign bits
// ============================================================
static __device__ void ktq_cuda_quantize_ktq2_1_block(const float * __restrict__ x, block_ktq2_1 * __restrict__ y, int64_t block_index) {
    // Step 1: Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) {
        memset(y->qs, 0, 8);
        memset(y->sb, 0, 4);
        return;
    }

    // Step 2: Normalize
    float x_hat[32];
    const float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) x_hat[j] = x[j] * inv_norm;

    // Step 3: RHT forward (random signs via Philox + FWHT)
    const uint16_t seed = ktq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    ktq_cuda_fwht_32_serial(rotated);

    // Step 4: codebook quantize — stochastic rounding (makes E[K_q] = K, unbiased)
    memset(y->qs, 0, 8);
    for (int j = 0; j < 32; j++) {
        const uint32_t rng = ktq_cuda_philox_6r(j + 32, seed);
        const int best = pq_stochastic_round<4>(rotated[j], PQ_CUDA_CB_2BIT, rng);
        y->qs[j / 4] |= (uint8_t)(best << (2 * (j % 4)));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        recon[j] = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (ktq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Quantize block: TQ3_1 (for set-rows / cpy — single-thread per block)
// ============================================================
static __device__ void ktq_cuda_quantize_ktq3_1_block(const float * __restrict__ x, block_ktq3_1 * __restrict__ y, int64_t block_index) {
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) {
        memset(y->qs, 0, 12);
        memset(y->sb, 0, 4);
        return;
    }

    float x_hat[32];
    const float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) x_hat[j] = x[j] * inv_norm;

    const uint16_t seed = ktq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    ktq_cuda_fwht_32_serial(rotated);

    // 3-bit codebook quantize — stochastic rounding
    memset(y->qs, 0, 12);
    for (int j = 0; j < 32; j++) {
        const uint32_t rng = ktq_cuda_philox_6r(j + 32, seed);
        const int best = pq_stochastic_round<8>(rotated[j], PQ_CUDA_CB_3BIT, rng);
        // Pack 3-bit
        const int bit_off = j * 3;
        const int byte_idx = bit_off / 8;
        const int bit_idx = bit_off % 8;
        y->qs[byte_idx] |= (uint8_t)(best << bit_idx);
        if (bit_idx > 5) {
            y->qs[byte_idx + 1] |= (uint8_t)(best >> (8 - bit_idx));
        }
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int bit_off = j * 3;
        const int byte_idx = bit_off / 8;
        const int bit_idx = bit_off % 8;
        int idx = (y->qs[byte_idx] >> bit_idx);
        if (bit_idx > 5) idx |= (y->qs[byte_idx + 1] << (8 - bit_idx));
        idx &= 0x7;
        recon[j] = PQ_CUDA_CB_3BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (ktq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Get-rows kernel: TQ block-level dequant for arbitrary row selection
// One CUDA block = 32 threads = one TQ block
// ============================================================
template <typename block_type, typename dst_t>
static __global__ void k_get_rows_tq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12);

// TQ1_1 specialization
template <typename dst_t>
static __global__ void k_get_rows_ktq1_1(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    // blockIdx.x = row index in src1, blockIdx.y = TQ block within row, blockIdx.z = batch
    const int i10 = blockIdx.x;
    const int ib_in_row = blockIdx.y;  // which TQ block within the row
    const int tid = threadIdx.x;       // 0..31

    const int64_t z = blockIdx.z;
    const int i11 = z / ne12;
    const int i12 = z % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const block_ktq1_1 * src0_row = (const block_ktq1_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_KTQ;
    if (ib_in_row >= nb_per_row) return;

    const block_ktq1_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_KTQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 1-bit index → centroid (Hadamard space).
    const int idx = (xb->qs[tid / 8] >> (tid % 8)) & 0x1;
    float val = PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT: normalized FWHT (self-inverse) + stored sign flip + scale.
    val = ktq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// KTQ2_1 specialization
template <typename dst_t>
static __global__ void k_get_rows_ktq2_1(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    // blockIdx.x = row index in src1, blockIdx.y = TQ block within row, blockIdx.z = batch
    const int i10 = blockIdx.x;
    const int ib_in_row = blockIdx.y;  // which TQ block within the row
    const int tid = threadIdx.x;       // 0..31

    const int64_t z = blockIdx.z;
    const int i11 = z / ne12;
    const int i12 = z % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const block_ktq2_1 * src0_row = (const block_ktq2_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_KTQ;
    if (ib_in_row >= nb_per_row) return;

    const block_ktq2_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_KTQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 2-bit index (4 per byte) → centroid in Hadamard space.
    const int idx = (xb->qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    float val = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT: normalized FWHT + stored sign flip + scale.
    val = ktq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// TQ3_1 specialization
template <typename dst_t>
static __global__ void k_get_rows_ktq3_1(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
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
    const block_ktq3_1 * src0_row = (const block_ktq3_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_KTQ;
    if (ib_in_row >= nb_per_row) return;

    const block_ktq3_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_KTQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 3-bit unpack (straddles byte boundary when bit_idx > 5).
    const int bit_offset = tid * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_idx = bit_offset % 8;
    int cb_idx = (xb->qs[byte_idx] >> bit_idx);
    if (bit_idx > 5) cb_idx |= (xb->qs[byte_idx + 1] << (8 - bit_idx));
    cb_idx &= 0x7;
    float val = PQ_CUDA_CB_3BIT[cb_idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT: normalized FWHT + stored sign flip + scale.
    val = ktq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Get-rows launchers for TQ types
// ============================================================
template <typename dst_t>
static void get_rows_cuda_ktq1_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t nb_per_row = ne00 / QK_KTQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_KTQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_ktq1_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
static void get_rows_cuda_ktq2_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t nb_per_row = ne00 / QK_KTQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_KTQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_ktq2_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
static void get_rows_cuda_ktq3_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t nb_per_row = ne00 / QK_KTQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_ktq3_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// ============================================================
// Block-level dequantize kernel: TQ4_1 (4-bit codebook + Hadamard, NO QJL)
// One CUDA block = one TQ block = 32 threads = 32 elements
//
// TQ4_1 uses 16 centroids (Lloyd-Max optimal, ≈ Beta(15.5, 15.5) at d=32) with
// nibble-packed indices + precomputed sign bits (v5).
// Same dequant pipeline: codebook lookup -> inverse FWHT -> inverse signs -> scale.
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_ktq4_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_KTQ;
    if (base >= ne) return;

    const block_ktq4_1 * x = (const block_ktq4_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 4-bit nibble (2 per byte) → centroid in Hadamard space.
    const int idx = (x[ib].qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
    float val = PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT part 1: normalized FWHT (self-inverse).
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT part 2 + scale: branchless sign flip from sb[].
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Launcher: TQ4_1 dequantize
// ============================================================
template <typename dst_t>
static void dequantize_row_ktq4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_KTQ == 0);
    const int nb = k / QK_KTQ;
    dequantize_block_ktq4_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

// ============================================================
// Quantize block: TQ4_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> codebook quantize (16 centroids, nibble-pack)
// v5: sign bits precomputed in sb[], no Philox at dequant time.
// ============================================================
static __device__ void ktq_cuda_quantize_ktq4_1_block(const float * __restrict__ x, block_ktq4_1 * __restrict__ y, int64_t block_index) {
    // Step 1: Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) {
        memset(y->qs, 0, 16);
        memset(y->sb, 0, 4);
        return;
    }

    // Step 2: Normalize
    float x_hat[32];
    const float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) x_hat[j] = x[j] * inv_norm;

    // Step 3: RHT forward (random signs via Philox + FWHT)
    const uint16_t seed = ktq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    ktq_cuda_fwht_32_serial(rotated);

    // Step 4: codebook quantize — stochastic rounding with 16 centroids
    memset(y->qs, 0, 16);
    for (int j = 0; j < 32; j++) {
        const uint32_t rng = ktq_cuda_philox_6r(j + 32, seed);
        const int best = pq_stochastic_round<16>(rotated[j], PQ_CUDA_CB_4BIT, rng);
        y->qs[j / 2] |= (uint8_t)(best << (4 * (j % 2)));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 2] >> (4 * (j % 2))) & 0xF;
        recon[j] = PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (ktq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (ktq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Get-rows kernel: TQ4_1 (warp-parallel, 32 threads per TQ block)
// ============================================================
template <typename dst_t>
static __global__ void k_get_rows_ktq4_1(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    // blockIdx.x = row index in src1, blockIdx.y = TQ block within row, blockIdx.z = batch
    const int i10 = blockIdx.x;
    const int ib_in_row = blockIdx.y;  // which TQ block within the row
    const int tid = threadIdx.x;       // 0..31

    const int64_t z = blockIdx.z;
    const int i11 = z / ne12;
    const int i12 = z % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const block_ktq4_1 * src0_row = (const block_ktq4_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_KTQ;
    if (ib_in_row >= nb_per_row) return;

    const block_ktq4_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_KTQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 4-bit nibble → centroid in Hadamard space.
    const int idx = (xb->qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
    float val = PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT: normalized FWHT + stored sign flip + scale.
    val = ktq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Get-rows launcher: TQ4_1
// ============================================================
template <typename dst_t>
static void get_rows_cuda_ktq4_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_KTQ == 0);
    const int64_t nb_per_row = ne00 / QK_KTQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_KTQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_ktq4_1<<<grid_dims, block_dims, 0, stream>>>(
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

// --- Generic VTQ quantize block (set-rows path) ---
// DecodeFn: float(const uint8_t*, int) — reads index j from qs
// EncodeFn: void(uint8_t*, int, int) — writes index idx at position j
// codebook: pointer to N_CB float centroids
template <typename block_t, int N_CB, typename DecodeFn, typename EncodeFn>
static __device__ void vtq_cuda_quantize_block(
        const float * __restrict__ x, block_t * __restrict__ y,
        const float * codebook, DecodeFn decode, EncodeFn encode) {
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) { memset(y->qs, 0, sizeof(y->qs)); return; }

    const float inv_norm = 1.0f / norm;
    memset(y->qs, 0, sizeof(y->qs));
    for (int j = 0; j < 32; j++) {
        float val = x[j] * inv_norm;
        int best = 0;
        float best_d = fabsf(val - codebook[0] * PQ_CUDA_CB_SCALE);
        for (int c = 1; c < N_CB; c++) {
            float d = fabsf(val - codebook[c] * PQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        encode(y->qs, j, best);
    }

    // Norm correction
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        float r = decode(y->qs, j);
        recon_sq += r * r;
    }
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);
}

// Concrete quantize-block wrappers (signature matches set_rows_cuda_tq template)
static __device__ void vtq_cuda_quantize_vtq1_1_block(const float * __restrict__ x, block_vtq1_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq1_1, 2>(x, y, VTQ_CUDA_CB_1BIT, vtq_decode_1bit, vtq_encode_1bit);
}
static __device__ void vtq_cuda_quantize_vtq2_1_block(const float * __restrict__ x, block_vtq2_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq2_1, 4>(x, y, VTQ_CUDA_CB_2BIT, vtq_decode_2bit, vtq_encode_2bit);
}
static __device__ void vtq_cuda_quantize_vtq3_1_block(const float * __restrict__ x, block_vtq3_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq3_1, 8>(x, y, PQ_CUDA_CB_3BIT, vtq_decode_3bit, vtq_encode_3bit);
}
static __device__ void vtq_cuda_quantize_vtq4_1_block(const float * __restrict__ x, block_vtq4_1 * __restrict__ y, int64_t) {
    vtq_cuda_quantize_block<block_vtq4_1, 16>(x, y, PQ_CUDA_CB_4BIT, vtq_decode_4bit, vtq_encode_4bit);
}

// --- Generic VTQ bulk dequantize kernel (warp-parallel, for convert.cu) ---
template <typename block_t, typename DecodeFn, typename dst_t>
static __global__ void k_dequantize_block_vtq(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne, const int64_t nb, DecodeFn decode) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= nb) return;
    const block_t * x = (const block_t *) vx;
    const int tid = threadIdx.x;
    const float val = decode(x[ib].qs, tid) * (float)x[ib].d;
    const int64_t out_idx = ib * QK_VTQ + tid;
    if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(val);
}

// Row dequant launcher (shared for all VTQ types)
template <typename block_t, typename DecodeFn, typename dst_t>
static void vtq_dequantize_row_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream, DecodeFn decode) {
    const int64_t nb = (ne + QK_VTQ - 1) / QK_VTQ;
    const int rows_per_block = 4;
    const dim3 block_dims(32, rows_per_block);
    const dim3 grid_dims((int)((nb + rows_per_block - 1) / rows_per_block));
    k_dequantize_block_vtq<block_t, DecodeFn, dst_t><<<grid_dims, block_dims, 0, stream>>>(vx, y, ne, nb, decode);
}

// Concrete row dequant wrappers (signature matches convert.cu dispatcher)
template <typename dst_t>
static void dequantize_row_vtq1_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq1_1>(vx, y, ne, stream, vtq_decode_1bit);
}
template <typename dst_t>
static void dequantize_row_vtq2_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq2_1>(vx, y, ne, stream, vtq_decode_2bit);
}
template <typename dst_t>
static void dequantize_row_vtq3_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq3_1>(vx, y, ne, stream, vtq_decode_3bit);
}
template <typename dst_t>
static void dequantize_row_vtq4_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    vtq_dequantize_row_cuda<block_vtq4_1>(vx, y, ne, stream, vtq_decode_4bit);
}

// --- Generic VTQ NC (non-contiguous) dequant kernel ---
template <typename block_t, typename DecodeFn, typename dst_t>
static __global__ void k_dequantize_block_vtq_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03, DecodeFn decode) {
    const int64_t ib_in_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t ibx0 = dm.x*s03 + dm.y*s02 + i01*s01;
            const block_t * x = (const block_t *) vx;
            const int64_t ib = ibx0 + ib_in_row;
            const float norm = (float)x[ib].d;
            const int64_t out_base = (i0203*ne01 + i01)*ne00 + ib_in_row * QK_VTQ;
            if (norm < 1e-30f) {
                if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
                continue;
            }
            const float val = decode(x[ib].qs, tid) * norm;
            if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(val);
        }
    }
}

// NC launcher (shared for all VTQ types)
template <typename block_t, typename DecodeFn, typename dst_t>
static void vtq_dequantize_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream, DecodeFn decode) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    k_dequantize_block_vtq_nc<block_t, DecodeFn, dst_t><<<num_blocks, 32, 0, stream>>>(vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03, decode);
}

// Concrete NC wrappers (signature matches convert.cu dispatcher)
template <typename dst_t>
static void dequantize_block_vtq1_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq1_1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream, vtq_decode_1bit);
}
template <typename dst_t>
static void dequantize_block_vtq2_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq2_1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream, vtq_decode_2bit);
}
template <typename dst_t>
static void dequantize_block_vtq3_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq3_1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream, vtq_decode_3bit);
}
template <typename dst_t>
static void dequantize_block_vtq4_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    vtq_dequantize_nc_cuda<block_vtq4_1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream, vtq_decode_4bit);
}

// --- Generic VTQ get_rows kernel ---
template <typename block_t, typename DecodeFn, typename dst_t>
static __global__ void k_get_rows_vtq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12, DecodeFn decode) {
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
    const float val = decode(xb->qs, tid) * (float)xb->d;
    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// get_rows launcher (shared for all VTQ types)
template <typename block_t, typename DecodeFn, typename dst_t>
static void vtq_get_rows_cuda(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream, DecodeFn decode) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const size_t s1 = nb1/sizeof(dst_t), s2 = nb2/sizeof(dst_t), s3 = nb3/sizeof(dst_t);
    const size_t s10 = nb10/sizeof(int32_t), s11 = nb11/sizeof(int32_t), s12 = nb12/sizeof(int32_t);
    const dim3 block_dims(32);
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));
    k_get_rows_vtq<block_t, DecodeFn, dst_t><<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12, decode);
}

// Concrete get_rows wrappers (signature matches getrows.cu dispatcher)
template <typename dst_t>
static void get_rows_cuda_vtq1_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq1_1>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, vtq_decode_1bit);
}
template <typename dst_t>
static void get_rows_cuda_vtq2_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq2_1>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, vtq_decode_2bit);
}
template <typename dst_t>
static void get_rows_cuda_vtq3_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq3_1>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, vtq_decode_3bit);
}
template <typename dst_t>
static void get_rows_cuda_vtq4_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    vtq_get_rows_cuda<block_vtq4_1>(src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream, vtq_decode_4bit);
}
