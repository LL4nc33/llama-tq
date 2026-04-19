#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "turboquant.cuh"
#include "trellis.cuh"   // Phase-2c: VTQ{2,3,4}_2 trellis decoder for FA-vec V-dequant

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

// ============================================================
// KTQ Flash-Attention dequant helpers — warp-cooperative, one lane per element.
//
// Serial FWHT:     5 stages × 32 butterflies = 160 add/sub ops performed by
//                  one thread over a 32-float local buffer.
// Warp FWHT:       same 160 butterflies but distributed across 32 lanes
//                  using __shfl_xor_sync — only 5 shuffles per lane, no
//                  local/shared buffer, no per-thread 32-float staging.
//
// The warp variants are used inside the FA kernels, where the 32-float
// buffer would push register usage past the point the FA kernel can sustain
// its chosen block size without spilling. See ktq_cuda_fwht_warp in
// turboquant.cuh for the butterfly/sign-convention notes.
//
// `lane` is the thread's index within the 32-thread group covering one
// QK_KTQ block. Returns the dequantized value for that lane's element.
// ============================================================

static __device__ __forceinline__ float ktq_fattn_dequant_elem_ktq1_1(
        const block_ktq1_1 * __restrict__ x, const int64_t ib, const int lane) {
    const float norm = (float)x[ib].d;
    // No early return for norm==0: the FWHT uses __shfl_xor_sync, which
    // requires every lane in the mask to be active. norm==0 is allowed to fall
    // through and zero the result via the final multiply.

    // 1-bit index → Hadamard-space codebook value.
    const int idx = (x[ib].qs[lane / 8] >> (lane % 8)) & 0x1;
    float val = PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT part 1: normalized FWHT (self-inverse).
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT part 2 + scale: branchless sign flip; norm==0 zeros result.
    const int sign_bit = (x[ib].sb[lane / 8] >> (lane % 8)) & 1;
    return val * (1.0f - 2.0f * sign_bit) * norm;
}

static __device__ __forceinline__ float ktq_fattn_dequant_elem_ktq2_1(
        const block_ktq2_1 * __restrict__ x, const int64_t ib, const int lane) {
    const float norm = (float)x[ib].d;
    // See KTQ1_1 note: no early-out — all lanes must reach the FWHT shuffle.

    // 2-bit index → Hadamard-space codebook value.
    const int idx = (x[ib].qs[lane / 4] >> (2 * (lane % 4))) & 0x3;
    float val = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;

    // Inverse RHT part 1.
    val = ktq_cuda_fwht_warp(val);

    // Inverse RHT part 2 + scale.
    const int sign_bit = (x[ib].sb[lane / 8] >> (lane % 8)) & 1;
    return val * (1.0f - 2.0f * sign_bit) * norm;
}

static __device__ __forceinline__ float ktq_fattn_dequant_elem_ktq3_1(
        const block_ktq3_1 * __restrict__ x, const int64_t ib, const int lane) {
    const float norm = (float)x[ib].d;

    // Step 1: 3-bit unpack
    const int bit_offset = lane * 3;
    const int byte_idx = bit_offset / 8;
    const int bit_idx  = bit_offset % 8;
    int cb_idx = (x[ib].qs[byte_idx] >> bit_idx);
    if (bit_idx > 5) cb_idx |= (x[ib].qs[byte_idx + 1] << (8 - bit_idx));
    cb_idx &= 0x7;
    float val = PQ_CUDA_CB_3BIT[cb_idx] * PQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles
    val = ktq_cuda_fwht_warp(val);

    // Step 3: Fused sign×norm — branchless
    const int sign_bit = (x[ib].sb[lane / 8] >> (lane % 8)) & 1;
    return val * (1.0f - 2.0f * sign_bit) * norm;
}

static __device__ __forceinline__ float ktq_fattn_dequant_elem_ktq4_1(
        const block_ktq4_1 * __restrict__ x, const int64_t ib, const int lane) {
    const float norm = (float)x[ib].d;

    // Step 1: 4-bit codebook lookup
    const int idx = (x[ib].qs[lane / 2] >> (4 * (lane % 2))) & 0xF;
    float val = PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE;

    // Step 2: Inverse FWHT via warp shuffles
    val = ktq_cuda_fwht_warp(val);

    // Step 3: Fused sign×norm — branchless
    const int sign_bit = (x[ib].sb[lane / 8] >> (lane % 8)) & 1;
    return val * (1.0f - 2.0f * sign_bit) * norm;
}

// Legacy serial dequant — kept for non-FA paths (e.g. standalone dequantize kernels)
static __device__ __forceinline__ void ktq_fattn_dequant_block_ktq1_1(const block_ktq1_1 * __restrict__ x, const int64_t ib, float * __restrict__ buf) {
    const float norm = (float)x[ib].d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int idx = (x[ib].qs[j / 8] >> (j % 8)) & 0x1;
        buf[j] = PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x[ib].sb[j / 8] >> (j % 8)) & 1;
        buf[j] *= (1.0f - 2.0f * sb) * norm;
    }
}

static __device__ __forceinline__ void ktq_fattn_dequant_block_ktq2_1(const block_ktq2_1 * __restrict__ x, const int64_t ib, float * __restrict__ buf) {
    const float norm = (float)x[ib].d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int idx = (x[ib].qs[j / 4] >> (2 * (j % 4))) & 0x3;
        buf[j] = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x[ib].sb[j / 8] >> (j % 8)) & 1;
        buf[j] *= (1.0f - 2.0f * sb) * norm;
    }
}

static __device__ __forceinline__ void ktq_fattn_dequant_block_ktq3_1(const block_ktq3_1 * __restrict__ x, const int64_t ib, float * __restrict__ buf) {
    const float norm = (float)x[ib].d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int bit_offset = j * 3;
        const int byte_idx = bit_offset / 8;
        const int bit_idx  = bit_offset % 8;
        int idx = (x[ib].qs[byte_idx] >> bit_idx);
        if (bit_idx > 5) idx |= (x[ib].qs[byte_idx + 1] << (8 - bit_idx));
        idx &= 0x7;
        buf[j] = PQ_CUDA_CB_3BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x[ib].sb[j / 8] >> (j % 8)) & 1;
        buf[j] *= (1.0f - 2.0f * sb) * norm;
    }
}

static __device__ __forceinline__ void ktq_fattn_dequant_block_ktq4_1(const block_ktq4_1 * __restrict__ x, const int64_t ib, float * __restrict__ buf) {
    const float norm = (float)x[ib].d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int idx = (x[ib].qs[j / 2] >> (4 * (j % 2))) & 0xF;
        buf[j] = PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x[ib].sb[j / 8] >> (j % 8)) & 1;
        buf[j] *= (1.0f - 2.0f * sb) * norm;
    }
}

// K·Q vec-dot for KTQ types — v7 Hadamard-domain formulation.
//
// For an RHT-quantized K-block, K = D_s · H_n · c (D_s diagonal signs
// from sb[], H_n normalized 32-point Hadamard, c codebook reconstruction).
// Then  K · Q = c · (H_n^T · D_s^T · Q) = c · (H_n · (D_s · Q))  because
// H_n is orthogonal (self-transpose, self-inverse) and D_s is its own inverse.
// Therefore transform *Q* into Hadamard space once per K-block (5 shuffles)
// and dot against the codebook value directly, skipping the per-element
// inverse FWHT and the gather shuffles the v6 path needed.
//
// Warp-parallel path (nthreads == WARP_SIZE, i.e. head dim D ≥ 128): every
// lane owns one element of each 32-element block; the FWHT and dot both
// fit inside a single warp shuffle pattern.
//
// Serial fallback (nthreads < WARP_SIZE, typically D == 64): the warp is
// already split across heads, so cooperating on a 32-element FWHT is
// unsafe — drop back to ktq_fattn_dequant_block_* (serial FWHT into a
// 32-float buffer) and do the dot in registers.
//
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq1_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_ktq1_1 * K_tq = (const block_ktq1_1 *) K_c;
    const int lane = threadIdx.x;

    if constexpr (nthreads == WARP_SIZE) {
        // v7 Hadamard-domain dot product for 1-bit TQ
        const float * Q_f32 = (const float *) Q_v;
        GGML_UNUSED(Q_q8);
        GGML_UNUSED(Q_ds_v);
        float accum = 0.0f;

        constexpr int nblocks = D / QK_KTQ;

        #pragma unroll
        for (int bi = 0; bi < nblocks; ++bi) {
            const float norm = (float)K_tq[bi].d;
            // NOTE: no early-exit — all lanes must participate in FWHT warp shuffle

            // 1. Sign-flip Q for this K-block (branchless)
            const int sb = (K_tq[bi].sb[lane / 8] >> (lane % 8)) & 1;
            float Q_signed = Q_f32[bi] * (1.0f - 2.0f * sb);

            // 2. FWHT(Q_signed) -> rotate Q into Hadamard space (5 shuffles)
            float Q_rot = ktq_cuda_fwht_warp(Q_signed);

            // 3. Codebook lookup — 1-bit index
            const int idx = (K_tq[bi].qs[lane / 8] >> (lane % 8)) & 0x1;

            // 4. Multiply + accumulate: norm==0 naturally zeros the contribution
            accum += PQ_CUDA_CB_1BIT[idx] * PQ_CUDA_CB_SCALE * Q_rot * norm;
        }

        return accum;  // NOT reduced — caller does warp_reduce_sum
    } else {
        // Fallback: serial FWHT for nthreads < WARP_SIZE (D == 64)
        GGML_UNUSED(Q_v);
        const int lane_q = threadIdx.x % nthreads;
        float sum = 0.0f;

        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
            const int k_KQ     = k_KQ_0 + lane_q;
            const int my_ib    = k_KQ / (QK_KTQ / 4);
            const int iqs      = k_KQ % (QK_KTQ / 4);
            const int elem_off = iqs * 4;

            const int q8_val = Q_q8[k_KQ_0 / nthreads];
            const int8_t * q8 = (const int8_t *) &q8_val;
            float block_sum = 0.0f;

            float buf[32];
            ktq_fattn_dequant_block_ktq1_1(K_tq, my_ib, buf);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                block_sum += buf[elem_off + l] * (float)q8[l];
            }

            const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0 / nthreads];
            sum += block_sum * Q_ds.x;
        }
        return sum;
    }
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq2_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_ktq2_1 * K_tq = (const block_ktq2_1 *) K_c;
    const int lane = threadIdx.x;

    if constexpr (nthreads == WARP_SIZE) {
        // Hadamard-domain dot (see v7 note above the template). For D=128
        // this is 4 blocks × 5 FWHT shuffles + 5 reduction shuffles = 25
        // warp shuffles total (was 41 in the v6 inverse-FWHT-on-K path).
        //
        // Q_v layout: each thread holds D / WARP_SIZE = D/32 scalars of Q,
        // striped so that lane t holds Q[bi·32 + t] for bi = 0..nblocks-1.
        // This matches the element-per-lane layout of the K-block so the
        // FWHT operates on Q directly with no reshuffle.
        const float * Q_f32 = (const float *) Q_v;
        GGML_UNUSED(Q_q8);
        GGML_UNUSED(Q_ds_v);
        float accum = 0.0f;

        constexpr int nblocks = D / QK_KTQ;  // 4 for D=128

        #pragma unroll
        for (int bi = 0; bi < nblocks; ++bi) {
            const float norm = (float)K_tq[bi].d;
            // All 32 lanes must reach the FWHT below: __shfl_xor_sync on a
            // partial mask would desync the warp. norm==0 is handled by the
            // final multiply instead of an early return.

            // 1. Apply D_s (diagonal signs from sb[]) to Q — pushing the
            //    inverse of the quantizer's RHT onto the query side.
            const int sb = (K_tq[bi].sb[lane / 8] >> (lane % 8)) & 1;
            float Q_signed = Q_f32[bi] * (1.0f - 2.0f * sb);

            // 2. Rotate Q into Hadamard space: H_n · (D_s · Q).
            float Q_rot = ktq_cuda_fwht_warp(Q_signed);

            // 3. K stays in Hadamard space as a codebook index — no inverse FWHT.
            const int idx = (K_tq[bi].qs[lane / 4] >> (2 * (lane % 4))) & 0x3;

            // 4. Dot in Hadamard space. norm==0 zeros this block's contribution.
            accum += PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE * Q_rot * norm;
        }

        return accum;  // NOT reduced — caller does warp_reduce_sum
    } else {
        // Fallback: serial FWHT for nthreads < WARP_SIZE (D == 64)
        GGML_UNUSED(Q_v);
        const int lane_q = threadIdx.x % nthreads;
        float sum = 0.0f;

        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
            const int k_KQ     = k_KQ_0 + lane_q;
            const int my_ib    = k_KQ / (QK_KTQ / 4);
            const int iqs      = k_KQ % (QK_KTQ / 4);
            const int elem_off = iqs * 4;

            const int q8_val = Q_q8[k_KQ_0 / nthreads];
            const int8_t * q8 = (const int8_t *) &q8_val;
            float block_sum = 0.0f;

            float buf[32];
            ktq_fattn_dequant_block_ktq2_1(K_tq, my_ib, buf);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                block_sum += buf[elem_off + l] * (float)q8[l];
            }

            const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0 / nthreads];
            sum += block_sum * Q_ds.x;
        }
        return sum;
    }
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq3_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_ktq3_1 * K_tq = (const block_ktq3_1 *) K_c;
    const int lane = threadIdx.x;

    if constexpr (nthreads == WARP_SIZE) {
        const float * Q_f32 = (const float *) Q_v;
        GGML_UNUSED(Q_q8);
        GGML_UNUSED(Q_ds_v);
        float accum = 0.0f;
        constexpr int nblocks = D / QK_KTQ;

        #pragma unroll
        for (int bi = 0; bi < nblocks; ++bi) {
            const float norm = (float)K_tq[bi].d;
            // NOTE: no early-exit — all lanes must participate in FWHT warp shuffle

            const int sb = (K_tq[bi].sb[lane / 8] >> (lane % 8)) & 1;
            float Q_signed = Q_f32[bi] * (1.0f - 2.0f * sb);
            float Q_rot = ktq_cuda_fwht_warp(Q_signed);

            // 3-bit unpack
            const int bit_offset = lane * 3;
            const int byte_idx = bit_offset / 8;
            const int bit_idx  = bit_offset % 8;
            int cb_idx = (K_tq[bi].qs[byte_idx] >> bit_idx);
            if (bit_idx > 5) cb_idx |= (K_tq[bi].qs[byte_idx + 1] << (8 - bit_idx));
            cb_idx &= 0x7;

            accum += PQ_CUDA_CB_3BIT[cb_idx] * PQ_CUDA_CB_SCALE * Q_rot * norm;
        }
        return accum;
    } else {
        GGML_UNUSED(Q_v);
        const int lane_q = threadIdx.x % nthreads;
        float sum = 0.0f;
        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
            const int k_KQ     = k_KQ_0 + lane_q;
            const int my_ib    = k_KQ / (QK_KTQ / 4);
            const int iqs      = k_KQ % (QK_KTQ / 4);
            const int elem_off = iqs * 4;
            const int q8_val = Q_q8[k_KQ_0 / nthreads];
            const int8_t * q8 = (const int8_t *) &q8_val;
            float block_sum = 0.0f;
            float buf[32];
            ktq_fattn_dequant_block_ktq3_1(K_tq, my_ib, buf);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                block_sum += buf[elem_off + l] * (float)q8[l];
            }
            const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0 / nthreads];
            sum += block_sum * Q_ds.x;
        }
        return sum;
    }
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_ktq4_1 * K_tq = (const block_ktq4_1 *) K_c;
    const int lane = threadIdx.x;

    if constexpr (nthreads == WARP_SIZE) {
        const float * Q_f32 = (const float *) Q_v;
        GGML_UNUSED(Q_q8);
        GGML_UNUSED(Q_ds_v);
        float accum = 0.0f;
        constexpr int nblocks = D / QK_KTQ;

        #pragma unroll
        for (int bi = 0; bi < nblocks; ++bi) {
            const float norm = (float)K_tq[bi].d;
            // NOTE: no early-exit — all lanes must participate in FWHT warp shuffle

            const int sb = (K_tq[bi].sb[lane / 8] >> (lane % 8)) & 1;
            float Q_signed = Q_f32[bi] * (1.0f - 2.0f * sb);
            float Q_rot = ktq_cuda_fwht_warp(Q_signed);

            // 4-bit nibble unpack
            const int idx = (K_tq[bi].qs[lane / 2] >> (4 * (lane % 2))) & 0xF;

            accum += PQ_CUDA_CB_4BIT[idx] * PQ_CUDA_CB_SCALE * Q_rot * norm;
        }
        return accum;
    } else {
        GGML_UNUSED(Q_v);
        const int lane_q = threadIdx.x % nthreads;
        float sum = 0.0f;
        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
            const int k_KQ     = k_KQ_0 + lane_q;
            const int my_ib    = k_KQ / (QK_KTQ / 4);
            const int iqs      = k_KQ % (QK_KTQ / 4);
            const int elem_off = iqs * 4;
            const int q8_val = Q_q8[k_KQ_0 / nthreads];
            const int8_t * q8 = (const int8_t *) &q8_val;
            float block_sum = 0.0f;
            float buf[32];
            ktq_fattn_dequant_block_ktq4_1(K_tq, my_ib, buf);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                block_sum += buf[elem_off + l] * (float)q8[l];
            }
            const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0 / nthreads];
            sum += block_sum * Q_ds.x;
        }
        return sum;
    }
}

// V-dequant for KTQ types, used inside the FA P·V loop.
//
// These are __noinline__ on purpose: each call materializes a 32-float
// buffer and runs a serial FWHT over it. Inlining into the FA kernel would
// add ~32 live floats + FWHT temporaries to an already register-tight loop
// and force spills to local memory (measured: ~15-20% FA decode slowdown
// on sm_75/sm_89 in local benchmarks). Keeping them as a separate call lets nvcc
// allocate the transient state in the callee frame.
template <typename T, int ne>
static __device__ __noinline__ void dequantize_V_ktq1_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_ktq1_1 * x = (const block_ktq1_1 *) vx;
    const int64_t ib = i0 / QK_KTQ;
    const int     il = (int)(i0 % QK_KTQ);

    float buf[32];
    ktq_fattn_dequant_block_ktq1_1(x, ib, buf);

    if constexpr (std::is_same_v<T, half>) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((half *) dst)[l] = __float2half(buf[il + l]);
    } else {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((float *) dst)[l] = buf[il + l];
    }
}

template <typename T, int ne>
static __device__ __noinline__ void dequantize_V_ktq2_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    // Reuses the proven ktq_fattn_dequant_block_ktq2_1 function (used in K-path, verified correct).
    // Dequants full 32-element block, then extracts ne consecutive values starting at il.
    const block_ktq2_1 * x = (const block_ktq2_1 *) vx;
    const int64_t ib = i0 / QK_KTQ;
    const int     il = (int)(i0 % QK_KTQ);

    float buf[32];
    ktq_fattn_dequant_block_ktq2_1(x, ib, buf);

    if constexpr (std::is_same_v<T, half>) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((half *) dst)[l] = __float2half(buf[il + l]);
    } else {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((float *) dst)[l] = buf[il + l];
    }
}

template <typename T, int ne>
static __device__ __noinline__ void dequantize_V_ktq3_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_ktq3_1 * x = (const block_ktq3_1 *) vx;
    const int64_t ib = i0 / QK_KTQ;
    const int     il = (int)(i0 % QK_KTQ);

    float buf[32];
    ktq_fattn_dequant_block_ktq3_1(x, ib, buf);

    if constexpr (std::is_same_v<T, half>) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((half *) dst)[l] = __float2half(buf[il + l]);
    } else {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((float *) dst)[l] = buf[il + l];
    }
}

template <typename T, int ne>
static __device__ __noinline__ void dequantize_V_ktq4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_ktq4_1 * x = (const block_ktq4_1 *) vx;
    const int64_t ib = i0 / QK_KTQ;
    const int     il = (int)(i0 % QK_KTQ);

    float buf[32];
    ktq_fattn_dequant_block_ktq4_1(x, ib, buf);

    if constexpr (std::is_same_v<T, half>) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((half *) dst)[l] = __float2half(buf[il + l]);
    } else {
        #pragma unroll
        for (int l = 0; l < ne; ++l) ((float *) dst)[l] = buf[il + l];
    }
}

// ============================================================
// VTQ V-dequant — codebook lookup · scale, nothing else.
//
// VTQ moves the rotation out of the cache path (self_v_rot runs once per
// graph, not per cache block), so there is no FWHT and no per-block sign
// bits at read time. The live set is ~8 registers (block pointer, ib, il,
// scale, loop index, decoded value, ne, output pointer) which is small
// enough to __forceinline__ into the FA kernel without degrading its
// occupancy. See vtq_decode_* helpers in turboquant.cuh.
// ============================================================

template <typename block_t, typename T, int ne, auto decode_fn>
static __device__ __forceinline__ void dequantize_V_vtq(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ;
    const int     il = (int)(i0 % QK_VTQ);
    const float   scale = (float)x[ib].d;

    #pragma unroll
    for (int l = 0; l < ne; ++l) {
        const float val = decode_fn(x[ib].qs, il + l) * scale;
        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(val);
        } else {
            ((float *) dst)[l] = val;
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq2_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq<block_vtq2_1, T, ne, vtq_decode_2bit>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq3_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq<block_vtq3_1, T, ne, vtq_decode_3bit>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq<block_vtq4_1, T, ne, vtq_decode_4bit>(vx, dst, i0);
}

// ============================================================
// Phase-2c (WIP): VTQ{2,3,4}_2 (Trellis v2) V-dequant in FA-vec.
//
// The decoder is a shift register; random access to element `i0`
// requires replaying from start_state. We use the per-element
// variant `trellis_decode_element<K>` from trellis.cuh. This is
// O(i0) per element — fine for D<=256 heads, inefficient for larger.
//
// See trellis.cuh for the optimal Strategy A (warp-shmem block cache).
// That requires invasive fattn-vec.cuh changes (deferred to Phase-2d).
// ============================================================

template <typename block_t, int K, typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ_TRELLIS;
    const int     il = (int)(i0 % QK_VTQ_TRELLIS);
    const float   d  = (float) x[ib].d;
    const uint16_t s0 = x[ib].start_state;
    const uint8_t * qs = x[ib].qs;

    // Walk the shift register once from 0 to il+ne-1, storing the last `ne`
    // values. Cost: O(il+ne) per call instead of O((il+ne)²) via per-element
    // replay. For D=256 ne=2 this is 2× faster; for D=256 ne=4 it's 2×.
    // (Real fix is warp-collaborative shmem cache — Phase-2e.)
    constexpr int N = QK_VTQ_TRELLIS;
    constexpr int L = VTQ_TRELLIS_L;
    constexpr uint32_t Lmask = 0xFFFFu;
    constexpr uint32_t Kmask = (1u << K) - 1u;
    const float cb_scale = rsqrtf((float)N);
    const float ds = cb_scale * d;

    if (d == 0.0f) {
        #pragma unroll
        for (int l = 0; l < ne; ++l) {
            if constexpr (std::is_same_v<T, half>) {
                ((half *) dst)[l] = __float2half(0.0f);
            } else {
                ((float *) dst)[l] = 0.0f;
            }
        }
        return;
    }

    uint32_t state = (uint32_t)s0 & Lmask;
    const int last = il + ne;

    for (int i = 0; i < last; ++i) {
        const int bit_off = i * K;
        const int byte    = bit_off >> 3;
        const int shift   = bit_off & 7;
        uint32_t b0 = qs[byte];
        uint32_t b1 = qs[byte + 1];
        uint32_t b2 = (shift + K > 16) ? qs[byte + 2] : 0u;
        uint32_t w  = b0 | (b1 << 8) | (b2 << 16);
        uint32_t bits = (w >> shift) & Kmask;
        state = ((state >> K) | (bits << (L - K))) & Lmask;

        if (i >= il) {
            const int l = i - il;
            const float val = vtq_trellis_table_storage[state] * ds;
            if constexpr (std::is_same_v<T, half>) {
                ((half *) dst)[l] = __float2half(val);
            } else {
                ((float *) dst)[l] = val;
            }
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq2_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_2<block_vtq2_2, 2, T, ne>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq3_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_2<block_vtq3_2, 3, T, ne>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq4_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_2<block_vtq4_2, 4, T, ne>(vx, dst, i0);
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ1_1) {
        return vec_dot_fattn_vec_KQ_ktq1_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ2_1) {
        return vec_dot_fattn_vec_KQ_ktq2_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ3_1) {
        return vec_dot_fattn_vec_KQ_ktq3_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ4_1) {
        return vec_dot_fattn_vec_KQ_ktq4_1<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ1_1) {
        return dequantize_V_ktq1_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ2_1) {
        return dequantize_V_ktq2_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ3_1) {
        return dequantize_V_ktq3_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ4_1) {
        return dequantize_V_ktq4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ1_1) {
        return dequantize_V_vtq<block_vtq1_1, T, ne, vtq_decode_1bit>;
    } else if constexpr (type_V == GGML_TYPE_VTQ2_1) {
        return dequantize_V_vtq2_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ3_1) {
        return dequantize_V_vtq3_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ4_1) {
        return dequantize_V_vtq4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ2_2) {
        return dequantize_V_vtq2_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ3_2) {
        return dequantize_V_vtq3_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ4_2) {
        return dequantize_V_vtq4_2<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16.ptr;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
