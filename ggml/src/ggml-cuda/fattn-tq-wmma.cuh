#pragma once

#include "fattn-vec-gqa.cuh"

// Tensor-core decode kernel for quantized KV caches with grouped-query attention (GGML_CUDA_TQ_WMMA=0 disables it).
//
// The GQA decode kernel (fattn-vec-gqa.cuh) reduces every K-row dot product across a warp with shuffles; that,
// not memory bandwidth, bounds it. Here a block takes one KV head and the Q heads of its group as NC columns,
// dequantizes tiles of T K/V rows to f16 in shared memory and computes S = K·Qᵀ and O += Vᵀ·P with WMMA.
// KTQ K stays in the rotated (Hadamard) domain: Q is rotated once (shared sign pattern, then the 32-point FWHT).
//
// The kernels are compiled in template-instances/fattn-tq-wmma-instance-*.cu, one TU per head size D and block width
// NC; fattn-tq-wmma.cu dispatches.

// nvcuda::wmma exists only in the host pass and for Volta and newer: a Pascal device pass must not name it
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA)
#include <mma.h>
namespace tq_wmma = nvcuda::wmma;
#define FATTN_TQ_WMMA_AVAILABLE
#endif

#ifndef GGML_CUDA_TQ_WMMA_NWARPS
#define GGML_CUDA_TQ_WMMA_NWARPS 4
#endif
// K/V rows per tile for head size D and NC columns: 64 measured faster at D 128 with 16 columns; the smaller tiles keep
// the static shared memory within 48 KB, and with 32 columns at two blocks per SM on Turing (64 KB)
#ifndef GGML_CUDA_TQ_WMMA_T
#define GGML_CUDA_TQ_WMMA_T(D, NC) ((D) == 512 ? 16 : (D) == 256 ? 32/((NC)/16) : (D) == 128 ? 64/((NC)/16) : 64)
#endif
#ifndef GGML_CUDA_TQ_WMMA_MIN_BLOCKS
#define GGML_CUDA_TQ_WMMA_MIN_BLOCKS 1
#endif

template <ggml_type type>
static constexpr bool fattn_tq_wmma_supported() {
    return type == GGML_TYPE_KTQ2_1 || type == GGML_TYPE_KTQ3_1 || type == GGML_TYPE_KTQ4_1 ||
           type == GGML_TYPE_VTQ2_1 || type == GGML_TYPE_VTQ3_1 || type == GGML_TYPE_VTQ4_1 ||
           type == GGML_TYPE_Q8_0   || type == GGML_TYPE_Q5_0   || type == GGML_TYPE_Q4_0   || type == GGML_TYPE_F16;
}

// dequantize rows [0, T) of a K/V tile (row stride nb bytes) to f16, row r at tile + r*ldt
template <ggml_type type, int D, int T, int ldt, int nthreads>
static __device__ __forceinline__ void fattn_tq_wmma_load_tile(
        const char * __restrict__ base, const int64_t nb, half * __restrict__ tile, const float * __restrict__ cb, const int tid) {
    constexpr int nblocks = D/32;
    if constexpr (type == GGML_TYPE_F16) {
        // 16-byte chunks: D/8 per row
        for (int u = tid; u < T*(D/8); u += nthreads) {
            const int r = u / (D/8);
            const int c = u % (D/8);
            *(int4 *) (tile + r*ldt + c*8) = *(const int4 *) (base + r*nb + c*16);
        }
    } else {
    for (int u = tid; u < T*nblocks; u += nthreads) {
        const int r  = u / nblocks;
        const int bi = u % nblocks;
        // decode the 32 elements of the block into registers, then 4 stores of 8 halves
        half2 v[16];
        if constexpr (type == GGML_TYPE_Q8_0) {
            const block_q8_0 * b = (const block_q8_0 *) (base + r*nb) + bi;
            const float d = __half2float(b->d);
            const uint16_t * q16 = (const uint16_t *) b->qs; // qs starts at byte 2
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint16_t w = q16[k];
                v[k] = __floats2half2_rn(d*(int8_t) (w & 0xFF), d*(int8_t) (w >> 8));
            }
        } else if constexpr (type == GGML_TYPE_Q4_0) {
            // element j < 16 in the low nibble of qs[j], element j + 16 in the high nibble, offset by 8
            const block_q4_0 * b = (const block_q4_0 *) (base + r*nb) + bi;
            const float d = __half2float(b->d);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int q0 = b->qs[2*k + 0];
                const int q1 = b->qs[2*k + 1];
                v[k]     = __floats2half2_rn(d*((q0 & 0xF) - 8), d*((q1 & 0xF) - 8));
                v[k + 8] = __floats2half2_rn(d*((q0 >>  4) - 8), d*((q1 >>  4) - 8));
            }
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            const block_q5_0 * b = (const block_q5_0 *) (base + r*nb) + bi;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                v[k] = __floats2half2_rn(fattn_gqa_q5_0(b, 2*k), fattn_gqa_q5_0(b, 2*k + 1));
            }
        } else {
            constexpr int bits = fattn_gqa_tq<type>::bits;
            const typename fattn_gqa_tq<type>::block * b = (const typename fattn_gqa_tq<type>::block *) (base + r*nb) + bi;
            const float d = __half2float(b->d);
            if constexpr (bits == 4) {
                const uint16_t * q16 = (const uint16_t *) b->qs; // 2 bytes = 4 indices
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t w = q16[k];
                    v[2*k + 0] = __floats2half2_rn(cb[w & 0xF] * d, cb[(w >> 4) & 0xF] * d);
                    v[2*k + 1] = __floats2half2_rn(cb[(w >> 8) & 0xF] * d, cb[(w >> 12) & 0xF] * d);
                }
            } else if constexpr (bits == 2) {
                const uint16_t * q16 = (const uint16_t *) b->qs; // 2 bytes = 8 indices
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t w = q16[k];
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        v[4*k + l] = __floats2half2_rn(cb[(w >> (4*l)) & 0x3] * d, cb[(w >> (4*l + 2)) & 0x3] * d);
                    }
                }
            } else {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    v[k] = __floats2half2_rn(cb[tq_code_index<bits>(b->qs, 2*k)] * d, cb[tq_code_index<bits>(b->qs, 2*k + 1)] * d);
                }
            }
        }
        int4 * dst = (int4 *) (tile + r*ldt + bi*32);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            dst[k] = *(const int4 *) &v[4*k];
        }
    }
    }
}

template<int D, int NC, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(GGML_CUDA_TQ_WMMA_NWARPS*32, GGML_CUDA_TQ_WMMA_MIN_BLOCKS)
static __global__ void flash_attn_ext_tq_wmma(
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
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FATTN_TQ_WMMA_AVAILABLE) && defined(FLASH_ATTN_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    GGML_UNUSED_VARS(KV_max, max_bias, m0, m1, n_head_log2, ne00, ne01, ne03, nb01,
                     ne10, ne13, ne31, ne32, nb31, nb32);

    constexpr int nwarps   = GGML_CUDA_TQ_WMMA_NWARPS;
    constexpr int nthreads = nwarps*WARP_SIZE;
    constexpr int T        = GGML_CUDA_TQ_WMMA_T(D, NC);
    constexpr int ldt      = D + 8;              // tile row stride in halves (avoids bank conflicts, keeps 32 B alignment)
    constexpr int ldp      = NC + 8;
    constexpr bool K_tq    = GGML_TYPE_IS_KTQ(type_K);
    constexpr int bits_K   = fattn_gqa_tq<type_K>::bits;
    constexpr int bits_V   = fattn_gqa_tq<type_V>::bits;
    constexpr float LOG2E  = 1.4426950408f;
    constexpr int nrt      = T/16;                            // 16-row tiles of S
    constexpr int nct      = NC/16;                           // 16-column tiles of S and P
    // warps per S tile, each over D/nsplit (partial sums): head 512, and head 256 with 32 columns (one row tile)
    constexpr int nsplit   = (D >= 512 || NC > 16) && nwarps/(nrt*nct) > 1 ? nwarps/(nrt*nct) : 1;
    // the partial O of a tile goes through the tile buffer as float, in nphase slices of the head dimension
    constexpr int nphase   = (D*NC*sizeof(float) + T*ldt*sizeof(half) - 1) / (T*ldt*sizeof(half));
    constexpr int tpc      = nthreads / NC;                   // softmax threads per column, consecutive lanes
    static_assert(NC % 16 == 0 && nthreads % NC == 0 && WARP_SIZE % tpc == 0, "bad column count");
    static_assert(T % 16 == 0 && D % (16*nwarps) == 0 && (D/16) % nsplit == 0, "bad tile shape");
    static_assert(D % nphase == 0 && (D/nphase) % (D/nwarps) == 0, "an O slice must hold whole warp ranges");
    static_assert((D/nphase)*NC*sizeof(float) <= T*ldt*sizeof(half), "an O slice must fit the tile buffer");

    __shared__ __align__(32) half  tile[T*ldt];  // K tile, then V tile, then the partial O as float
    __shared__ __align__(32) half  Qs[NC*ldt];
    __shared__ __align__(32) float S[nsplit*T*NC];
    __shared__ __align__(32) half  P[T*ldp];
    __shared__ float m_s[NC];
    __shared__ float l_s[NC];
    __shared__ float alpha_s[NC];
    __shared__ float cb_K[bits_K ? 1 << bits_K : 1];
    __shared__ float cb_V[bits_V ? 1 << bits_V : 1];

    const int tid  = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int warp = threadIdx.y;

    const int gqa_ratio   = ne02 / ne12;
    const int ntiles_gqa  = (gqa_ratio + NC - 1) / NC;
    const int sequence    = blockIdx.z / (ntiles_gqa*ne12);
    const int z           = blockIdx.z - sequence*ntiles_gqa*ne12;
    const int kv_head     = z / ntiles_gqa;
    const int head0       = kv_head*gqa_ratio + (z - kv_head*ntiles_gqa)*NC;
    const int ncols_valid = min(NC, (kv_head + 1)*gqa_ratio - head0);

    Q += nb03*sequence + nb02*head0;
    K += nb13*sequence + int64_t(nb12)*kv_head;
    V += nb23*sequence + int64_t(nb22)*kv_head;
    const half * maskh = (const half *) (mask + nb33*(sequence % ne33));

    if constexpr (bits_K) {
        if (tid < (1 << bits_K)) {
            cb_K[tid] = fattn_gqa_tq_centroid<type_K>(tid);
        }
    }
    if constexpr (bits_V) {
        if (tid < (1 << bits_V)) {
            cb_V[tid] = fattn_gqa_tq_centroid<type_V>(tid);
        }
    }
    if (tid < NC) {
        m_s[tid] = -FLT_MAX/2.0f;
        l_s[tid] = 0.0f;
    }

    // Q (scaled; KTQ: rotated into the Hadamard domain of the K blocks), warp w loads columns w, w + nwarps, ...
    for (int j = warp; j < NC; j += nwarps) {
        const float * Q_j = (const float *) (Q + j*nb02);
#pragma unroll
        for (int bi = 0; bi < D/WARP_SIZE; ++bi) {
            float q = j < ncols_valid ? Q_j[bi*WARP_SIZE + threadIdx.x] * scale : 0.0f;
            if constexpr (K_tq) {
                q = ktq_cuda_fwht_warp(q * ktq_cuda_shared_sign(threadIdx.x));
            }
            Qs[j*ldt + bi*WARP_SIZE + threadIdx.x] = __float2half(q);
        }
    }

    // this thread's share of O [D][NC]: elements tid + i*nthreads, all in column tid % NC
    constexpr int n_own = D*NC/nthreads;
    static_assert(n_own % nphase == 0, "O slices must split the thread's elements evenly");
    float O[n_own];
#pragma unroll
    for (int i = 0; i < n_own; ++i) {
        O[i] = 0.0f;
    }
    __syncthreads();

    for (int k0 = blockIdx.y*T; k0 < ne11; k0 += gridDim.y*T) {
        // S = K·Qᵀ for the T rows of this tile: nrt*nct 16x16 tiles, each split into nsplit partial sums over D
        fattn_tq_wmma_load_tile<type_K, D, T, ldt, nthreads>(K + int64_t(k0)*nb11, nb11, tile, cb_K, tid);
        __syncthreads();
        for (int it = warp; it < nrt*nct*nsplit; it += nwarps) {
            const int rt = it % nrt;
            const int ct = (it / nrt) % nct;
            const int ks = it / (nrt*nct);
            tq_wmma::fragment<tq_wmma::accumulator, 16, 16, 16, float> s_acc;
            tq_wmma::fill_fragment(s_acc, 0.0f);
#pragma unroll
            for (int kk = ks*(D/16/nsplit); kk < (ks + 1)*(D/16/nsplit); ++kk) {
                tq_wmma::fragment<tq_wmma::matrix_a, 16, 16, 16, half, tq_wmma::row_major> a;
                tq_wmma::fragment<tq_wmma::matrix_b, 16, 16, 16, half, tq_wmma::col_major> b;
                tq_wmma::load_matrix_sync(a, tile + rt*16*ldt + kk*16, ldt);
                tq_wmma::load_matrix_sync(b, Qs + ct*16*ldt + kk*16, ldt);
                tq_wmma::mma_sync(s_acc, a, b, s_acc);
            }
            tq_wmma::store_matrix_sync(S + ks*T*NC + rt*16*NC + ct*16, s_acc, NC, tq_wmma::mem_row_major);
        }
        __syncthreads();

        // online softmax per column: tpc threads per column (consecutive lanes), rows sub, sub + tpc, ...
        {
            const int c   = tid / tpc;
            const int sub = tid % tpc;
            float mx = -FLT_MAX/2.0f;
            for (int r = sub; r < T; r += tpc) {
                float s = S[r*NC + c];
#pragma unroll
                for (int j = 1; j < nsplit; ++j) {
                    s += S[j*T*NC + r*NC + c];
                }
                if (use_logit_softcap) {
                    s = logit_softcap*tanhf(s);
                }
                s += __half2float(maskh[k0 + r]);
                S[r*NC + c] = s;
                mx = fmaxf(mx, s);
            }
#pragma unroll
            for (int o = 1; o < tpc; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, o, WARP_SIZE));
            }
            const float m_old = m_s[c];
            const float m_new = fmaxf(m_old, mx + FATTN_KQ_MAX_OFFSET);
            float sum = 0.0f;
            for (int r = sub; r < T; r += tpc) {
                const float p = exp2f((S[r*NC + c] - m_new) * LOG2E);
                P[r*ldp + c] = __float2half(p);
                sum += p;
            }
#pragma unroll
            for (int o = 1; o < tpc; o <<= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, o, WARP_SIZE);
            }
            __syncwarp();
            if (sub == 0) {
                const float alpha = exp2f((m_old - m_new) * LOG2E);
                m_s[c]     = m_new;
                l_s[c]     = l_s[c]*alpha + sum;
                alpha_s[c] = alpha;
            }
        }
        __syncthreads();

        // O_tile = Vᵀ·P: warp w computes the output dims [w*D/nwarps, (w + 1)*D/nwarps) of all NC columns
        fattn_tq_wmma_load_tile<type_V, D, T, ldt, nthreads>(V + int64_t(k0)*nb21, nb21, tile, cb_V, tid);
        __syncthreads();
        constexpr int mt_per_warp = D/nwarps/16;
        tq_wmma::fragment<tq_wmma::accumulator, 16, 16, 16, float> o_acc[mt_per_warp][nct];
#pragma unroll
        for (int mt = 0; mt < mt_per_warp; ++mt) {
            const int d0 = warp*(D/nwarps) + mt*16;
#pragma unroll
            for (int ct = 0; ct < nct; ++ct) {
                tq_wmma::fill_fragment(o_acc[mt][ct], 0.0f);
            }
#pragma unroll
            for (int kt = 0; kt < T/16; ++kt) {
                tq_wmma::fragment<tq_wmma::matrix_a, 16, 16, 16, half, tq_wmma::col_major> a;
                tq_wmma::load_matrix_sync(a, tile + kt*16*ldt + d0, ldt);
#pragma unroll
                for (int ct = 0; ct < nct; ++ct) {
                    tq_wmma::fragment<tq_wmma::matrix_b, 16, 16, 16, half, tq_wmma::row_major> b;
                    tq_wmma::load_matrix_sync(b, P + kt*16*ldp + ct*16, ldp);
                    tq_wmma::mma_sync(o_acc[mt][ct], a, b, o_acc[mt][ct]);
                }
            }
        }
        __syncthreads(); // all warps are done with the V tile before it is reused for O_tile

        float * Ot = (float *) tile;
#pragma unroll
        for (int ph = 0; ph < nphase; ++ph) {
            constexpr int Dp = D/nphase;
            if (warp*(D/nwarps) / Dp == ph) {
#pragma unroll
                for (int mt = 0; mt < mt_per_warp; ++mt) {
                    const int d0 = warp*(D/nwarps) + mt*16;
#pragma unroll
                    for (int ct = 0; ct < nct; ++ct) {
                        tq_wmma::store_matrix_sync(Ot + (d0 - ph*Dp)*NC + ct*16, o_acc[mt][ct], NC, tq_wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = ph*(n_own/nphase); i < (ph + 1)*(n_own/nphase); ++i) {
                const int e = tid + i*nthreads;
                O[i] = O[i]*alpha_s[e % NC] + Ot[e - ph*Dp*NC];
            }
            __syncthreads(); // before the next slice or tile overwrites tile, S and P
        }
    }

    // attention sinks (gpt-oss): one extra logit per Q head in the softmax denominator, added once (first KV block)
    if (sinks && blockIdx.y == 0) {
        if (tid < NC) {
            float alpha = 1.0f;
            if (tid < ncols_valid) {
                const float sink  = ((const float *) sinks)[head0 + tid];
                const float m_new = fmaxf(sink, m_s[tid]);
                alpha    = exp2f((m_s[tid] - m_new) * LOG2E);
                l_s[tid] = l_s[tid]*alpha + exp2f((sink - m_new) * LOG2E);
                m_s[tid] = m_new;
            }
            alpha_s[tid] = alpha;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < n_own; ++i) {
            O[i] *= alpha_s[(tid + i*nthreads) % NC];
        }
    }

#pragma unroll
    for (int i = 0; i < n_own; ++i) {
        const int e = tid + i*nthreads;
        const int d = e / NC;
        const int c = e % NC;
        if (c < ncols_valid) {
            const int dst_row = (sequence*ne02 + head0 + c)*gridDim.y + blockIdx.y;
            dst[dst_row*D + d] = gridDim.y == 1 ? O[i] / l_s[c] : O[i];
        }
    }
    if (gridDim.y != 1 && tid < ncols_valid) {
        const int dst_row = (sequence*ne02 + head0 + tid)*gridDim.y + blockIdx.y;
        dst_meta[dst_row] = make_float2(m_s[tid], l_s[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

template <int D, int NC, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_tq_wmma_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    fattn_kernel_t fattn_kernel = logit_softcap == 0.0f ?
        flash_attn_ext_tq_wmma<D, NC, type_K, type_V, false> :
        flash_attn_ext_tq_wmma<D, NC, type_K, type_V, true>;
    constexpr int T = GGML_CUDA_TQ_WMMA_T(D, NC);
    launch_fattn<D, 1, NC>(ctx, dst, fattn_kernel, GGML_CUDA_TQ_WMMA_NWARPS, 0, T, false, false, false);
}

// The cases: X(D, type_K, type_V) for every head size and K/V pair the kernel is compiled for.
// ggml_cuda_flash_attn_ext tries the GQA decode kernel first; at heads 128 and 256 this kernel only gets what that one
// does not take (GQA ratio 2, a single KQ tile). Heads 64 (gpt-oss, with attention sinks) and 512 (Gemma 4 global
// layers, GQA 8-16) have no GQA decode kernel. f16 K next to a VTQ V: KTQ K staged as f16 (deferred K, the default).
#define FATTN_TQ_WMMA_CASES(X)                          \
    X( 64, GGML_TYPE_KTQ4_1, GGML_TYPE_VTQ4_1)          \
    X( 64, GGML_TYPE_KTQ3_1, GGML_TYPE_VTQ3_1)          \
    X( 64, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ2_1)          \
    X( 64, GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0)            \
    X( 64, GGML_TYPE_Q8_0,   GGML_TYPE_VTQ4_1)          \
    X( 64, GGML_TYPE_Q8_0,   GGML_TYPE_VTQ2_1)          \
    X( 64, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0)            \
    X( 64, GGML_TYPE_F16,    GGML_TYPE_VTQ4_1)          \
    X( 64, GGML_TYPE_F16,    GGML_TYPE_VTQ3_1)          \
    X( 64, GGML_TYPE_F16,    GGML_TYPE_VTQ2_1)          \
    X(128, GGML_TYPE_KTQ4_1, GGML_TYPE_VTQ4_1)          \
    X(128, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ2_1)          \
    X(128, GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0)            \
    X(128, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0)            \
    X(128, GGML_TYPE_KTQ3_1, GGML_TYPE_VTQ3_1)          \
    X(256, GGML_TYPE_KTQ4_1, GGML_TYPE_VTQ4_1)          \
    X(256, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ2_1)          \
    X(256, GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0)            \
    X(256, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0)            \
    X(256, GGML_TYPE_KTQ3_1, GGML_TYPE_VTQ3_1)          \
    X(256, GGML_TYPE_KTQ4_1, GGML_TYPE_F16)             \
    X(256, GGML_TYPE_KTQ2_1, GGML_TYPE_F16)             \
    X(256, GGML_TYPE_F16,    GGML_TYPE_VTQ4_1)          \
    X(256, GGML_TYPE_F16,    GGML_TYPE_VTQ2_1)          \
    X(512, GGML_TYPE_KTQ4_1, GGML_TYPE_VTQ4_1)          \
    X(512, GGML_TYPE_KTQ3_1, GGML_TYPE_VTQ3_1)          \
    X(512, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ2_1)          \
    X(512, GGML_TYPE_F16,    GGML_TYPE_VTQ4_1)          \
    X(512, GGML_TYPE_F16,    GGML_TYPE_VTQ2_1)          \
    X(512, GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0)            \
    X(512, GGML_TYPE_Q8_0,   GGML_TYPE_VTQ4_1)          \
    X(512, GGML_TYPE_Q8_0,   GGML_TYPE_VTQ2_1)          \
    X(512, GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0)            \
    X(512, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0)

// runs the case of the K/V pair of dst at head size D with NC columns; false if there is none
template <int D, int NC>
bool ggml_cuda_flash_attn_ext_tq_wmma_d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

#define FATTN_TQ_WMMA_CASE_D(D_, type_K_, type_V_)                          \
    if constexpr (D == (D_)) {                                              \
        if (K->type == (type_K_) && V->type == (type_V_)) {                 \
            ggml_cuda_flash_attn_ext_tq_wmma_case<D, NC, type_K_, type_V_>(ctx, dst); \
            return true;                                                    \
        }                                                                   \
    }
    FATTN_TQ_WMMA_CASES(FATTN_TQ_WMMA_CASE_D)
#undef FATTN_TQ_WMMA_CASE_D

    return false;
}

// GGML_FATTN_VEC_GQA_USED (fattn-vec-gqa.cuh): RDC must not prune the instances
#define DECL_FATTN_TQ_WMMA_CASE(D, NC) \
    template GGML_FATTN_VEC_GQA_USED bool ggml_cuda_flash_attn_ext_tq_wmma_d<D, NC>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

extern DECL_FATTN_TQ_WMMA_CASE( 64, 16);
extern DECL_FATTN_TQ_WMMA_CASE(128, 16);
extern DECL_FATTN_TQ_WMMA_CASE(256, 16);
extern DECL_FATTN_TQ_WMMA_CASE(512, 16);
