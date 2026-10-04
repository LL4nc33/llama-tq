#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tq.cuh"

// Flash attention decode kernel for TurboQuant KV caches with grouped-query attention.
//
// The generic VEC kernel runs one block per Q head, so every K/V row is read and dequantized once per Q head
// (6 times for Qwen3.8 with 24 Q heads over 4 KV heads). Here one block handles the ncols Q heads of a GQA group
// for one query: each K/V row is read and dequantized once per group.
//
// To keep ncols columns in registers, a whole warp works on one K/V row (each lane owns D/32 elements), and the
// ncols partial KQ dot products of a row are reduced across the warp together: log2(ncols) exchange steps that
// halve the number of values per lane, then plain shuffles for the rest (9 shuffles for 8 columns instead of 40).
//
// Supported: one query (ne01 == 1), no ALiBi, no sinks; K f16 (D == 256) or KTQ; V f16, KTQ or codebook VTQ.
//
// use_sparse: only the K/V rows of the index list built from the mask are visited (sparse attention, the query sees
// at most n_kv_max cells); KV_max then holds the list of each sequence, followed by the number of live entries.

// After fattn_gqa_reduce_cols<ncols>, lane l holds the full sum of this column:
template <int ncols>
static __device__ __forceinline__ int fattn_gqa_col(const int lane) {
    int col = 0;
#pragma unroll
    for (int count = ncols, offset = WARP_SIZE/2; count > 1; count >>= 1, offset >>= 1) {
        col += (lane & offset) ? count/2 : 0;
    }
    return col;
}

template <int ncols>
static __device__ __forceinline__ float fattn_gqa_reduce_cols(float * s) {
    static_assert(ncols >= 1 && ncols <= WARP_SIZE && (ncols & (ncols - 1)) == 0, "ncols must be a power of 2");
    const int lane = threadIdx.x;
    int offset = WARP_SIZE/2;
#pragma unroll
    for (int count = ncols; count > 1; count >>= 1, offset >>= 1) {
        const bool upper = lane & offset;
#pragma unroll
        for (int k = 0; k < count/2; ++k) {
            const float send = upper ? s[k]           : s[k + count/2];
            const float keep = upper ? s[k + count/2] : s[k];
            s[k] = keep + __shfl_xor_sync(0xFFFFFFFF, send, offset, WARP_SIZE);
        }
    }
    float sum = s[0];
#pragma unroll
    for (; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    return sum;
}

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool use_sparse>
__launch_bounds__(128, 2)
static __global__ void flash_attn_ext_vec_gqa(
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
#ifdef FLASH_ATTN_AVAILABLE
    GGML_UNUSED_VARS(sinks, max_bias, m0, m1, n_head_log2, ne00, ne01, ne03, nb01,
                     ne10, ne13, ne31, ne32, nb31, nb32);

    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads / WARP_SIZE;
    constexpr bool K_tq = type_K == GGML_TYPE_KTQ1_1 || type_K == GGML_TYPE_KTQ2_1 || type_K == GGML_TYPE_KTQ3_1 || type_K == GGML_TYPE_KTQ4_1;
    static_assert(K_tq || (type_K == GGML_TYPE_F16 && D == 256), "K must be KTQ, or f16 with D == 256");
    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 64");

    // one warp per K row and per V row
    constexpr int nthreads_KQ       = WARP_SIZE;
    constexpr int nthreads_V        = WARP_SIZE;
    constexpr int V_rows_per_thread = D / WARP_SIZE;
    constexpr vec_dot_KQ_t   vec_dot_KQ   = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();

    const int gqa_ratio  = ne02 / ne12;
    const int ntiles_gqa = (gqa_ratio + ncols - 1) / ncols;
    const int sequence   = blockIdx.z / (ntiles_gqa*ne12);
    const int z          = blockIdx.z - sequence*ntiles_gqa*ne12;
    const int kv_head    = z / ntiles_gqa;
    const int head0      = kv_head*gqa_ratio + (z - kv_head*ntiles_gqa)*ncols; // first Q head of this block
    const int ncols_valid = min(ncols, (kv_head + 1)*gqa_ratio - head0);

    Q += nb03*sequence + nb02*head0;
    K += nb13*sequence + nb12*kv_head;
    V += nb23*sequence + nb22*kv_head;
    const half * maskh = (const half *) (mask + nb33*(sequence % ne33));

    // sparse: the rows of position k are indices[k], positions from n_rows on are padding
    const int32_t * indices = use_sparse ? KV_max + int64_t(sequence % ne33)*ne11 : nullptr;
    const int       n_rows  = use_sparse ? KV_max[int64_t(ne33)*ne11 + sequence % ne33] : ne11;

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;

    // Q in registers, lane t holds elements [i*32 + t] (KTQ: rotated into the Hadamard domain) or the
    // float2 pairs [t*cpy_ne ...] of the f16 dot product
    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes() / 4;
    float  Q_f32[K_tq ? ncols : 1][D/WARP_SIZE];
    __align__(16) float2 Q_f2[K_tq ? 1 : ncols][(D/2)/nthreads_KQ];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb02);
        if constexpr (K_tq) {
            const float sign = ktq_cuda_shared_sign(threadIdx.x);
#pragma unroll
            for (int bi = 0; bi < D/WARP_SIZE; ++bi) {
                const float q = j < ncols_valid ? Q_j[bi*WARP_SIZE + threadIdx.x] * scale : 0.0f;
                Q_f32[j][bi] = ktq_cuda_fwht_warp(q * sign);
            }
        } else {
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + threadIdx.x*cpy_ne;
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    const float2 q = j < ncols_valid ? ((const float2 *) Q_j)[i + i1] : make_float2(0.0f, 0.0f);
                    Q_f2[j][i0/nthreads_KQ + i1] = make_float2(q.x*scale, q.y*scale);
                }
            }
        }
    }

    __shared__ float KQ[ncols*nthreads];   // scores, then softmax weights, of the nthreads rows of an iteration
    __shared__ float VKQ_combine[nwarps*D];

    float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    constexpr float LOG2E = 1.4426950408f;

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < n_rows; k_VKQ_0 += gridDim.y*nthreads) {
        // the row of each position of this iteration; lane t of warp w owns position k_VKQ_0 + w*32 + t
        const int  pos_own   = k_VKQ_0 + tid;
        const bool valid_own = !use_sparse || pos_own < n_rows;
        const int  row_own   = use_sparse ? (valid_own ? indices[pos_own] : 0) : pos_own;

        // KQ: warp w computes the scores of rows [w*32, w*32 + 32) of this iteration for all columns
#pragma unroll 4
        for (int i_KQ_0 = 0; i_KQ_0 < WARP_SIZE; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + i_KQ_0;
            const char * K_row = K + int64_t(__shfl_sync(0xFFFFFFFF, row_own, i_KQ_0, WARP_SIZE))*nb11;
            float s[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (K_tq) {
                    s[j] = vec_dot_KQ(K_row, Q_f32[j], nullptr, nullptr);
                } else {
                    s[j] = vec_dot_KQ(K_row, Q_f2[j], nullptr, nullptr);
                }
            }
            const float sum = fattn_gqa_reduce_cols<ncols>(s);
            if (threadIdx.x % (WARP_SIZE/ncols) == 0) {
                KQ[fattn_gqa_col<ncols>(threadIdx.x)*nthreads + i_KQ] = sum;
            }
        }
        __syncwarp();

        // online softmax: lane t of warp w owns row w*32 + t
        const float mask_val = valid_own ? __half2float(maskh[row_own]) : -INFINITY;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float s = KQ[j*nthreads + tid];
            if (use_logit_softcap) {
                s = logit_softcap*tanhf(s);
            }
            s += mask_val;

            const float KQ_max_new = fmaxf(KQ_max[j], warp_reduce_max(s + FATTN_KQ_MAX_OFFSET));
            const float KQ_max_scale = exp2f((KQ_max[j] - KQ_max_new) * LOG2E);
            KQ_max[j] = KQ_max_new;

            const float p = exp2f((s - KQ_max[j]) * LOG2E);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + p;
            KQ[j*nthreads + tid] = p;

#pragma unroll
            for (int i = 0; i < (D/2)/nthreads_V; ++i) {
                VKQ[j][i].x *= KQ_max_scale;
                VKQ[j][i].y *= KQ_max_scale;
            }
        }
        __syncwarp();

        // VKQ: warp w accumulates its 32 V rows, each dequantized once for all columns
#pragma unroll 2
        for (int k0 = 0; k0 < WARP_SIZE; ++k0) {
            const int k = threadIdx.y*WARP_SIZE + k0;
            const char * V_row = V + int64_t(__shfl_sync(0xFFFFFFFF, row_own, k0, WARP_SIZE))*nb21;
            float2 tmp[V_rows_per_thread/2];
            dequantize_V(V_row, tmp, threadIdx.x*V_rows_per_thread);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const float p = KQ[j*nthreads + k];
#pragma unroll
                for (int i = 0; i < V_rows_per_thread/2; ++i) {
                    VKQ[j][i].x += tmp[i].x*p;
                    VKQ[j][i].y += tmp[i].y*p;
                }
            }
        }
        __syncwarp();
    }

    // combine the nwarps partial results of each column
    __shared__ float KQ_max_shared[ncols][nwarps];
    __shared__ float KQ_sum_shared[ncols][nwarps];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_sum[j] = warp_reduce_sum(KQ_sum[j]);
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
            KQ_sum_shared[j][threadIdx.y] = KQ_sum[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (j >= ncols_valid) {
            break;
        }
        float kqmax = KQ_max_shared[j][0];
#pragma unroll
        for (int w = 1; w < nwarps; ++w) {
            kqmax = fmaxf(kqmax, KQ_max_shared[j][w]);
        }
        const float scale_w = exp2f((KQ_max[j] - kqmax) * LOG2E);

        // lane t holds the output elements [t*V_rows_per_thread, (t+1)*V_rows_per_thread)
#pragma unroll
        for (int i = 0; i < V_rows_per_thread/2; ++i) {
            VKQ_combine[threadIdx.y*D + threadIdx.x*V_rows_per_thread + 2*i + 0] = VKQ[j][i].x * scale_w;
            VKQ_combine[threadIdx.y*D + threadIdx.x*V_rows_per_thread + 2*i + 1] = VKQ[j][i].y * scale_w;
        }
        float kqsum = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            kqsum += KQ_sum_shared[j][w] * exp2f((KQ_max_shared[j][w] - kqmax) * LOG2E);
        }
        __syncthreads();

        const int dst_row = (sequence*ne02 + head0 + j)*gridDim.y + blockIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += nthreads) {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                val += VKQ_combine[w*D + i0 + tid];
            }
            if (gridDim.y == 1) {
                val /= kqsum;
            }
            dst[dst_row*D + i0 + tid] = val;
        }
        if (gridDim.y != 1 && tid == 0) {
            dst_meta[dst_row] = make_float2(kqmax, kqsum);
        }
        __syncthreads();
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
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, int ncols, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_vec_gqa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool use_sparse) {
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (use_sparse) {
        fattn_kernel = logit_softcap == 0.0f ?
            flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, false, true> :
            flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, true,  true>;
    } else {
        fattn_kernel = logit_softcap == 0.0f ?
            flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, false, false> :
            flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, true,  false>;
    }
    constexpr int nwarps = 128 / WARP_SIZE;
    launch_fattn<D, 1, ncols>(ctx, dst, fattn_kernel, nwarps, 0, 128, false, false, false, WARP_SIZE, use_sparse);
}
