#pragma once

#include "common.cuh"
#include "ggml-common.h"

// TurboQuant CUDA v7 — PolarQuant + Hadamard, precomputed sign bits
// Based on arXiv:2504.19874. v7: Hadamard-domain KQ dot, branchless sign×norm.
// v5: sb[] eliminates Philox from dequant path.

// ============================================================
// Codebook constants (Lloyd-Max for Beta(15.5, 15.5), d=32, v3) — __constant__ for GPU cache
// ============================================================
__device__ __constant__ static float TQ_CUDA_CB_1BIT[2] = {
    -0.797885f, 0.797885f
};

__device__ __constant__ static float TQ_CUDA_CB_2BIT[4] = {
    -1.489560f, -0.451428f, 0.451428f, 1.489560f
};

__device__ __constant__ static float TQ_CUDA_CB_3BIT[8] = {
    -2.071926f, -1.314996f, -0.745325f, -0.242405f,
     0.242405f,  0.745325f,  1.314996f,  2.071926f
};

__device__ __constant__ static float TQ_CUDA_CB_4BIT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

#define TQ_CUDA_CB_SCALE 0.17677669529663689f // 1/sqrt(32)

// ============================================================
// Philox 2x32 Counter-Based PRNG — O(1) random access
// Each (counter, key) pair deterministically produces a random uint32.
// No sequential state advance needed — thread j directly calls philox(j, seed).
// ============================================================
static __device__ __forceinline__ uint32_t tq_cuda_philox(uint32_t counter, uint32_t key) {
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

// Philox 2x32 with 6 rounds — sufficient for RHT sign generation (non-cryptographic)
// v5: Used only in quantize path (dequant reads precomputed sb[])
static __device__ __forceinline__ uint32_t tq_cuda_philox_6r(uint32_t counter, uint32_t key) {
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
static __device__ __forceinline__ uint16_t tq_cuda_derive_seed(int64_t block_index) {
    // FNV-1a hash of block index — fast, good avalanche
    uint32_t h = 2166136261u;
    h ^= (uint32_t)(block_index & 0xFF);        h *= 16777619u;
    h ^= (uint32_t)((block_index >> 8) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 16) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 24) & 0xFF); h *= 16777619u;
    return (uint16_t)(h & 0xFFFF);
}

// ============================================================
// Warp-Parallel FWHT via __shfl_xor_sync (32 elements, one per thread)
// ============================================================
static __device__ __forceinline__ float tq_cuda_fwht_warp(float val) {
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        float sum = val + other;
        float diff = other - val;  // note: other - val, not val - other
        val = (threadIdx.x & step) ? diff : sum;
    }
    return val * 0.17677669529663689f; // 1/sqrt(32)
}

// ============================================================
// Serial FWHT for single-thread quantize path
// ============================================================
static __device__ void tq_cuda_fwht_32_serial(float * data) {
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
static __device__ __forceinline__ float tq_cuda_gaussian(uint32_t i, uint32_t j, uint16_t seed) {
    uint32_t key = (uint32_t)seed + 32768u;
    uint32_t counter1 = i * 64u + j * 2u;
    uint32_t counter2 = i * 64u + j * 2u + 1u;
    float u1 = ((float)(tq_cuda_philox(counter1, key) >> 8) + 0.5f) / 16777216.0f;
    float u2 = ((float)(tq_cuda_philox(counter2, key) >> 8) + 0.5f) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * __cosf(6.2831853f * u2);
}

// ============================================================
// Block-level dequantize kernel: TQ1_1 (1-bit PolarQuant + Hadamard, NO QJL)
// One CUDA block = one TQ block = 32 threads = 32 elements
//
// TQ1_1 is the simplest TurboQuant variant: 1-bit codebook {-0.7979, +0.7979}.
// Index extraction: 1 bit per element, 8 elements packed per byte in qs[4].
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_tq1_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_TQ;
    if (base >= ne) return;

    const block_tq1_1 * x = (const block_tq1_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // Step 1: PolarQuant codebook lookup (in rotated space) — 1-bit index
    const int idx = (x[ib].qs[tid / 8] >> (tid % 8)) & 0x1;
    float val = TQ_CUDA_CB_1BIT[idx] * TQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles (inverse RHT part 1)
    val = tq_cuda_fwht_warp(val);

    // Step 3+4: Fused sign*norm — branchless: (1 - 2*bit) * norm
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Block-level dequantize kernel: TQ2_1 (PolarQuant + Hadamard, NO QJL)
// One CUDA block = one TQ block = 32 threads = 32 elements
//
// QJL is intentionally disabled everywhere for KV cache attention:
// Multiple independent groups confirmed QJL eliminates bias but explodes
// variance, which softmax amplifies. PolarQuant-only gives better quality.
// See: TheTom/turboquant_plus turbo4-resurrection, scos-lab, Arclabs001/YATQ
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_tq2_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_TQ;
    if (base >= ne) return;

    const block_tq2_1 * x = (const block_tq2_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // Step 1: PolarQuant codebook lookup (in rotated space)
    const int idx = (x[ib].qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    float val = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles (inverse RHT part 1)
    val = tq_cuda_fwht_warp(val);

    // Step 3+4: Fused sign×norm — branchless: (1 - 2*bit) * norm
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Block-level dequantize kernel: TQ3_1 (PolarQuant + Hadamard, NO QJL)
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_tq3_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = ib * QK_TQ;
    if (base >= ne) return;

    const block_tq3_1 * x = (const block_tq3_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // Step 1: 3-bit unpack for element tid (in rotated space)
    const int bit_offset = tid * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_idx = bit_offset % 8;
    int cb_idx = (x[ib].qs[byte_idx] >> bit_idx);
    if (bit_idx > 5) cb_idx |= (x[ib].qs[byte_idx + 1] << (8 - bit_idx));
    cb_idx &= 0x7;
    float val = TQ_CUDA_CB_3BIT[cb_idx] * TQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles
    val = tq_cuda_fwht_warp(val);

    // Step 3+4: Fused sign×norm — branchless: (1 - 2*bit) * norm
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Launcher functions matching the dequantize_row_*_cuda pattern
// ============================================================
template <typename dst_t>
static void dequantize_row_tq1_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ == 0);
    const int nb = k / QK_TQ;
    dequantize_block_tq1_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

template <typename dst_t>
static void dequantize_row_tq2_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ == 0);
    const int nb = k / QK_TQ;
    dequantize_block_tq2_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

template <typename dst_t>
static void dequantize_row_tq3_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ == 0);
    const int nb = k / QK_TQ;
    dequantize_block_tq3_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

// ============================================================
// Quantize block: TQ1_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> PolarQuant (1-bit) -> sign bits
// ============================================================
static __device__ void tq_cuda_quantize_tq1_1_block(const float * __restrict__ x, block_tq1_1 * __restrict__ y, int64_t block_index) {
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
    const uint16_t seed = tq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    tq_cuda_fwht_32_serial(rotated);

    // Step 4: PolarQuant — 1-bit: sign threshold
    memset(y->qs, 0, 4);
    for (int j = 0; j < 32; j++) {
        const int idx = (rotated[j] >= 0.0f) ? 1 : 0;
        y->qs[j / 8] |= (uint8_t)(idx << (j % 8));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 8] >> (j % 8)) & 0x1;
        recon[j] = TQ_CUDA_CB_1BIT[idx] * TQ_CUDA_CB_SCALE;
    }
    tq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (tq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Quantize block: TQ2_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> PolarQuant -> QJL sign bits
// ============================================================
static __device__ void tq_cuda_quantize_tq2_1_block(const float * __restrict__ x, block_tq2_1 * __restrict__ y, int64_t block_index) {
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
    const uint16_t seed = tq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    tq_cuda_fwht_32_serial(rotated);

    // Step 4: PolarQuant — nearest codebook centroid
    memset(y->qs, 0, 8);
    for (int j = 0; j < 32; j++) {
        float val = rotated[j];
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_2BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 4; c++) {
            float d = fabsf(val - TQ_CUDA_CB_2BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        y->qs[j / 4] |= (uint8_t)(best << (2 * (j % 4)));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        recon[j] = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE;
    }
    tq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (tq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Quantize block: TQ3_1 (for set-rows / cpy — single-thread per block)
// ============================================================
static __device__ void tq_cuda_quantize_tq3_1_block(const float * __restrict__ x, block_tq3_1 * __restrict__ y, int64_t block_index) {
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

    const uint16_t seed = tq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    tq_cuda_fwht_32_serial(rotated);

    // PolarQuant with 3-bit codebook
    memset(y->qs, 0, 12);
    for (int j = 0; j < 32; j++) {
        float val = rotated[j];
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_3BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 8; c++) {
            float d = fabsf(val - TQ_CUDA_CB_3BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
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
        recon[j] = TQ_CUDA_CB_3BIT[idx] * TQ_CUDA_CB_SCALE;
    }
    tq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (tq_cuda_philox_6r(j, seed) & 1);
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
static __global__ void k_get_rows_tq1_1(
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
    const block_tq1_1 * src0_row = (const block_tq1_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_TQ;
    if (ib_in_row >= nb_per_row) return;

    const block_tq1_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_TQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // PolarQuant codebook lookup — 1-bit index (rotated space)
    const int idx = (xb->qs[tid / 8] >> (tid % 8)) & 0x1;
    float val = TQ_CUDA_CB_1BIT[idx] * TQ_CUDA_CB_SCALE;

    // Inverse FWHT + precomputed signs from sb[] (v5)
    val = tq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// TQ2_1 specialization
template <typename dst_t>
static __global__ void k_get_rows_tq2_1(
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
    const block_tq2_1 * src0_row = (const block_tq2_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_TQ;
    if (ib_in_row >= nb_per_row) return;

    const block_tq2_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_TQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // PolarQuant codebook lookup (rotated space)
    const int idx = (xb->qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    float val = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE;

    // Inverse FWHT + precomputed signs from sb[] (v5)
    val = tq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// TQ3_1 specialization
template <typename dst_t>
static __global__ void k_get_rows_tq3_1(
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
    const block_tq3_1 * src0_row = (const block_tq3_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_TQ;
    if (ib_in_row >= nb_per_row) return;

    const block_tq3_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_TQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // 3-bit unpack (rotated space)
    const int bit_offset = tid * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_idx = bit_offset % 8;
    int cb_idx = (xb->qs[byte_idx] >> bit_idx);
    if (bit_idx > 5) cb_idx |= (xb->qs[byte_idx + 1] << (8 - bit_idx));
    cb_idx &= 0x7;
    float val = TQ_CUDA_CB_3BIT[cb_idx] * TQ_CUDA_CB_SCALE;

    // Inverse FWHT + precomputed signs from sb[] (v5 �� see dequantize_block comment)
    val = tq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Get-rows launchers for TQ types
// ============================================================
template <typename dst_t>
static void get_rows_cuda_tq1_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ == 0);
    const int64_t nb_per_row = ne00 / QK_TQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_TQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_tq1_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
static void get_rows_cuda_tq2_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ == 0);
    const int64_t nb_per_row = ne00 / QK_TQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_TQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_tq2_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
static void get_rows_cuda_tq3_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ == 0);
    const int64_t nb_per_row = ne00 / QK_TQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_tq3_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// ============================================================
// Block-level dequantize kernel: TQ4_1 (4-bit PolarQuant + Hadamard, NO QJL)
// One CUDA block = one TQ block = 32 threads = 32 elements
//
// TQ4_1 uses 16 centroids (Lloyd-Max for Beta(15.5,15.5), d=32) with
// nibble-packed indices + precomputed sign bits (v5).
// Same dequant pipeline: codebook lookup -> inverse FWHT -> inverse signs -> scale.
// ============================================================
template <typename dst_t>
static __global__ void dequantize_block_tq4_1_v2(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t ib = blockIdx.x;
    const int tid = threadIdx.x;     // 0..31, one per element
    const int64_t base = ib * QK_TQ;
    if (base >= ne) return;

    const block_tq4_1 * x = (const block_tq4_1 *) vx;
    const float norm = (float)x[ib].d;

    if (norm < 1e-30f) {
        if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // Step 1: PolarQuant codebook lookup — nibble unpack (in rotated space)
    const int idx = (x[ib].qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
    float val = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles (inverse RHT part 1)
    val = tq_cuda_fwht_warp(val);

    // Step 3+4: Fused sign×norm — branchless: (1 - 2*bit) * norm
    const int sb = (x[ib].sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (base + tid < ne) yy[base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Launcher: TQ4_1 dequantize
// ============================================================
template <typename dst_t>
static void dequantize_row_tq4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ == 0);
    const int nb = k / QK_TQ;
    dequantize_block_tq4_1_v2<<<nb, 32, 0, stream>>>(vx, y, k);
}

// ============================================================
// Quantize block: TQ4_1 (for set-rows / cpy — single-thread per block)
// normalize -> random signs -> FWHT -> PolarQuant (16 centroids, nibble-pack)
// v5: sign bits precomputed in sb[], no Philox at dequant time.
// ============================================================
static __device__ void tq_cuda_quantize_tq4_1_block(const float * __restrict__ x, block_tq4_1 * __restrict__ y, int64_t block_index) {
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
    const uint16_t seed = tq_cuda_derive_seed(block_index);
    float rotated[32];
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        rotated[j] = x_hat[j] * s;
    }
    tq_cuda_fwht_32_serial(rotated);

    // Step 4: PolarQuant — nearest of 16 codebook centroids, nibble-packed
    memset(y->qs, 0, 16);
    for (int j = 0; j < 32; j++) {
        float val = rotated[j];
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_4BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 16; c++) {
            float d = fabsf(val - TQ_CUDA_CB_4BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        y->qs[j / 2] |= (uint8_t)(best << (4 * (j % 2)));
    }

    // v5: Norm correction — compensate for codebook quantization error
    float recon[32];
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 2] >> (4 * (j % 2))) & 0xF;
        recon[j] = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE;
    }
    tq_cuda_fwht_32_serial(recon);
    for (int j = 0; j < 32; j++) {
        const float s = (tq_cuda_philox_6r(j, seed) & 1) ? 1.0f : -1.0f;
        recon[j] *= s;
    }
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) recon_sq += recon[j] * recon[j];
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);

    // v5: Precompute RHT sign bits and store in sb[4]
    memset(y->sb, 0, 4);
    for (int j = 0; j < 32; j++) {
        uint8_t sign_bit = (tq_cuda_philox_6r(j, seed) & 1);
        y->sb[j / 8] |= (sign_bit << (j % 8));
    }
}

// ============================================================
// Get-rows kernel: TQ4_1 (warp-parallel, 32 threads per TQ block)
// ============================================================
template <typename dst_t>
static __global__ void k_get_rows_tq4_1(
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
    const block_tq4_1 * src0_row = (const block_tq4_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

    const int64_t nb_per_row = ne00 / QK_TQ;
    if (ib_in_row >= nb_per_row) return;

    const block_tq4_1 * xb = &src0_row[ib_in_row];
    const float norm = (float)xb->d;
    const int64_t out_base = ib_in_row * QK_TQ;

    if (norm < 1e-30f) {
        if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
        return;
    }

    // PolarQuant codebook lookup — nibble unpack (rotated space)
    const int idx = (xb->qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
    float val = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE;

    // Inverse FWHT + precomputed signs from sb[] (v5)
    val = tq_cuda_fwht_warp(val);
    const int sb = (xb->sb[tid / 8] >> (tid % 8)) & 1;
    val *= (1.0f - 2.0f * sb) * norm;

    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

// ============================================================
// Get-rows launcher: TQ4_1
// ============================================================
template <typename dst_t>
static void get_rows_cuda_tq4_1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ == 0);
    const int64_t nb_per_row = ne00 / QK_TQ;

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32);  // warp size = QK_TQ
    const dim3 grid_dims(ne10, (unsigned int)MIN(nb_per_row, (int64_t)UINT16_MAX), (unsigned int)MIN(ne11*ne12, (int64_t)UINT16_MAX));

    k_get_rows_tq4_1<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// ============================================================
// VTQ (Value TurboQuant) — V-cache optimized, NO FWHT/sign bits
// Data arrives pre-rotated via self_v_rot. Dequant = CB * scale.
// Reuses TQ codebook constants (same Lloyd-Max centroids).
// ============================================================

// --- VTQ2_1 quantize block (set-rows path) ---
static __device__ void vtq_cuda_quantize_vtq2_1_block(const float * __restrict__ x, block_vtq2_1 * __restrict__ y, int64_t /*block_index*/) {
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) { memset(y->qs, 0, 8); return; }

    const float inv_norm = 1.0f / norm;
    memset(y->qs, 0, 8);
    for (int j = 0; j < 32; j++) {
        float val = x[j] * inv_norm;
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_2BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 4; c++) {
            float d = fabsf(val - TQ_CUDA_CB_2BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        y->qs[j / 4] |= (uint8_t)(best << (2 * (j % 4)));
    }

    // Norm correction
    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        float r = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE;
        recon_sq += r * r;
    }
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);
}

// --- VTQ3_1 quantize block ---
static __device__ void vtq_cuda_quantize_vtq3_1_block(const float * __restrict__ x, block_vtq3_1 * __restrict__ y, int64_t /*block_index*/) {
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) { memset(y->qs, 0, 12); return; }

    const float inv_norm = 1.0f / norm;
    memset(y->qs, 0, 12);
    for (int j = 0; j < 32; j++) {
        float val = x[j] * inv_norm;
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_3BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 8; c++) {
            float d = fabsf(val - TQ_CUDA_CB_3BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos = bit_offset % 8;
        y->qs[byte_idx] |= (uint8_t)((best << bit_pos) & 0xFF);
        if (bit_pos > 5) {
            y->qs[byte_idx + 1] |= (uint8_t)(best >> (8 - bit_pos));
        }
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos = bit_offset % 8;
        int idx = ((y->qs[byte_idx] >> bit_pos) | (y->qs[byte_idx + 1] << (8 - bit_pos))) & 0x7;
        float r = TQ_CUDA_CB_3BIT[idx] * TQ_CUDA_CB_SCALE;
        recon_sq += r * r;
    }
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);
}

// --- VTQ4_1 quantize block ---
static __device__ void vtq_cuda_quantize_vtq4_1_block(const float * __restrict__ x, block_vtq4_1 * __restrict__ y, int64_t /*block_index*/) {
    float sum_sq = 0.0f;
    for (int j = 0; j < 32; j++) sum_sq += x[j] * x[j];
    const float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-30f) { memset(y->qs, 0, 16); return; }

    const float inv_norm = 1.0f / norm;
    memset(y->qs, 0, 16);
    for (int j = 0; j < 32; j++) {
        float val = x[j] * inv_norm;
        int best = 0;
        float best_d = fabsf(val - TQ_CUDA_CB_4BIT[0] * TQ_CUDA_CB_SCALE);
        for (int c = 1; c < 16; c++) {
            float d = fabsf(val - TQ_CUDA_CB_4BIT[c] * TQ_CUDA_CB_SCALE);
            if (d < best_d) { best_d = d; best = c; }
        }
        y->qs[j / 2] |= (uint8_t)(best << (4 * (j % 2)));
    }

    float recon_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        const int idx = (y->qs[j / 2] >> (4 * (j % 2))) & 0xF;
        float r = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE;
        recon_sq += r * r;
    }
    const float recon_norm = sqrtf(recon_sq);
    y->d = __float2half((recon_norm > 1e-30f) ? norm / recon_norm : norm);
}

// --- VTQ bulk dequantize kernels (warp-parallel, for convert.cu) ---
template <typename dst_t>
static __global__ void dequantize_block_vtq2_1_v2(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= nb) return;
    const block_vtq2_1 * x = (const block_vtq2_1 *) vx;
    const int tid = threadIdx.x;  // 0..31
    const float norm = (float)x[ib].d;
    const int idx = (x[ib].qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    const float val = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE * norm;
    const int64_t out_idx = ib * QK_VTQ + tid;
    if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(val);
}

template <typename dst_t>
static __global__ void dequantize_block_vtq3_1_v2(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= nb) return;
    const block_vtq3_1 * x = (const block_vtq3_1 *) vx;
    const int tid = threadIdx.x;
    const float norm = (float)x[ib].d;
    const int bit_offset = tid * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_pos = bit_offset % 8;
    const int idx = ((x[ib].qs[byte_idx] >> bit_pos) | (x[ib].qs[byte_idx + 1] << (8 - bit_pos))) & 0x7;
    const float val = TQ_CUDA_CB_3BIT[idx] * TQ_CUDA_CB_SCALE * norm;
    const int64_t out_idx = ib * QK_VTQ + tid;
    if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(val);
}

template <typename dst_t>
static __global__ void dequantize_block_vtq4_1_v2(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (ib >= nb) return;
    const block_vtq4_1 * x = (const block_vtq4_1 *) vx;
    const int tid = threadIdx.x;
    const float norm = (float)x[ib].d;
    const int idx = (x[ib].qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
    const float val = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE * norm;
    const int64_t out_idx = ib * QK_VTQ + tid;
    if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(val);
}

// Row dequant launchers
template <typename dst_t>
static void dequantize_row_vtq2_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ - 1) / QK_VTQ;
    const int rows_per_block = 4;
    const dim3 block_dims(32, rows_per_block);
    const dim3 grid_dims((int)((nb + rows_per_block - 1) / rows_per_block));
    dequantize_block_vtq2_1_v2<<<grid_dims, block_dims, 0, stream>>>(vx, y, ne, nb);
}

template <typename dst_t>
static void dequantize_row_vtq3_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ - 1) / QK_VTQ;
    const int rows_per_block = 4;
    const dim3 block_dims(32, rows_per_block);
    const dim3 grid_dims((int)((nb + rows_per_block - 1) / rows_per_block));
    dequantize_block_vtq3_1_v2<<<grid_dims, block_dims, 0, stream>>>(vx, y, ne, nb);
}

template <typename dst_t>
static void dequantize_row_vtq4_1_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ - 1) / QK_VTQ;
    const int rows_per_block = 4;
    const dim3 block_dims(32, rows_per_block);
    const dim3 grid_dims((int)((nb + rows_per_block - 1) / rows_per_block));
    dequantize_block_vtq4_1_v2<<<grid_dims, block_dims, 0, stream>>>(vx, y, ne, nb);
}

// --- VTQ NC (non-contiguous) dequant kernels ---
template <typename dst_t>
static __global__ void dequantize_block_vtq2_1_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t ib_in_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;
            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;
            const int64_t ib = ibx0 + ib_in_row;
            const block_vtq2_1 * x = (const block_vtq2_1 *) vx;
            const float norm = (float)x[ib].d;
            const int64_t out_base = (i0203*ne01 + i01)*ne00 + ib_in_row * QK_VTQ;
            if (norm < 1e-30f) {
                if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
                continue;
            }
            const int idx = (x[ib].qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
            float val = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE * norm;
            if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(val);
        }
    }
}

template <typename dst_t>
static __global__ void dequantize_block_vtq3_1_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t ib_in_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;
            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;
            const int64_t ib = ibx0 + ib_in_row;
            const block_vtq3_1 * x = (const block_vtq3_1 *) vx;
            const float norm = (float)x[ib].d;
            const int64_t out_base = (i0203*ne01 + i01)*ne00 + ib_in_row * QK_VTQ;
            if (norm < 1e-30f) {
                if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
                continue;
            }
            const int bit_offset = tid * 3;
            const int byte_idx = bit_offset / 8;
            const int bit_pos = bit_offset % 8;
            const int idx = ((x[ib].qs[byte_idx] >> bit_pos) | (x[ib].qs[byte_idx + 1] << (8 - bit_pos))) & 0x7;
            float val = TQ_CUDA_CB_3BIT[idx] * TQ_CUDA_CB_SCALE * norm;
            if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(val);
        }
    }
}

template <typename dst_t>
static __global__ void dequantize_block_vtq4_1_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t ib_in_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;
            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;
            const int64_t ib = ibx0 + ib_in_row;
            const block_vtq4_1 * x = (const block_vtq4_1 *) vx;
            const float norm = (float)x[ib].d;
            const int64_t out_base = (i0203*ne01 + i01)*ne00 + ib_in_row * QK_VTQ;
            if (norm < 1e-30f) {
                if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(0.0f);
                continue;
            }
            const int idx = (x[ib].qs[tid / 2] >> (4 * (tid % 2))) & 0xF;
            float val = TQ_CUDA_CB_4BIT[idx] * TQ_CUDA_CB_SCALE * norm;
            if (ib_in_row * QK_VTQ + tid < ne00) y[out_base + tid] = ggml_cuda_cast<dst_t>(val);
        }
    }
}

// NC launcher functions
template <typename dst_t>
static void dequantize_block_vtq2_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block_vtq2_1_nc<<<num_blocks, 32, 0, stream>>>(vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename dst_t>
static void dequantize_block_vtq3_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block_vtq3_1_nc<<<num_blocks, 32, 0, stream>>>(vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename dst_t>
static void dequantize_block_vtq4_1_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block_vtq4_1_nc<<<num_blocks, 32, 0, stream>>>(vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

// --- VTQ get_rows kernels ---
template <typename dst_t>
static __global__ void k_get_rows_vtq2_1(
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
    const block_vtq2_1 * src0_row = (const block_vtq2_1 *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
    const int64_t nb_per_row = ne00 / QK_VTQ;
    if (ib_in_row >= nb_per_row) return;
    const block_vtq2_1 * xb = &src0_row[ib_in_row];
    const int64_t out_base = ib_in_row * QK_VTQ;
    const float norm = (float)xb->d;
    const int idx = (xb->qs[tid / 4] >> (2 * (tid % 4))) & 0x3;
    float val = TQ_CUDA_CB_2BIT[idx] * TQ_CUDA_CB_SCALE * norm;
    if (out_base + tid < ne00) dst_row[out_base + tid] = ggml_cuda_cast<dst_t>(val);
}

template <typename dst_t>
static void get_rows_cuda_vtq2_1(
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
    k_get_rows_vtq2_1<<<grid_dims, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}
