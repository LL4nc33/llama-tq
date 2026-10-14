#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tq.cuh"

#ifndef GGML_CUDA_FA_GQA_MIN_BLOCKS
#define GGML_CUDA_FA_GQA_MIN_BLOCKS 2
#endif

// Flash attention decode kernel for quantized KV caches with grouped-query attention.
//
// The generic VEC kernel runs one block per Q head, so every K/V row is read and dequantized once per Q head
// (6 times for Qwen3.8 with 24 Q heads over 4 KV heads). Here one block handles the ncols Q heads of a GQA group
// for one query: each K/V row is read and dequantized once per group.
//
// To keep ncols columns in registers, a whole warp works on one K/V row (lane t owns D/32 elements of it, which ones
// depends on the type, see fattn_gqa_idx), and the ncols partial KQ dot products of a row are reduced across the warp
// together: log2(ncols) exchange steps that halve the number of values per lane, then plain shuffles for the rest.
// Each lane reads only a few bytes per row, so the kernel is bound by instruction latency, not by DRAM bandwidth.
//
// Supported: one query (ne01 == 1), no ALiBi, no sinks; K f16, q8_0, q5_0, q4_0 or ktq2/3/4_1; V f16, q8_0, q5_0, q4_0
// or vtq2/3/4_1 (fattn_gqa_load reads these types).
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

// The codebook lookups of the TurboQuant types index __constant__ memory with a different index per lane, which the
// constant cache serves one address at a time (up to 16 serialized reads for 4 bit). This kernel copies the codebooks
// to shared memory, where the at most 16 entries sit in distinct banks.
template <ggml_type type> struct fattn_gqa_tq {
    static constexpr int bits = 0; // not a codebook type
};
template <> struct fattn_gqa_tq<GGML_TYPE_KTQ2_1> { static constexpr int bits = 2; using block = block_ktq2_1; };
template <> struct fattn_gqa_tq<GGML_TYPE_KTQ3_1> { static constexpr int bits = 3; using block = block_ktq3_1; };
template <> struct fattn_gqa_tq<GGML_TYPE_KTQ4_1> { static constexpr int bits = 4; using block = block_ktq4_1; };
template <> struct fattn_gqa_tq<GGML_TYPE_VTQ2_1> { static constexpr int bits = 2; using block = block_vtq2_1; };
template <> struct fattn_gqa_tq<GGML_TYPE_VTQ3_1> { static constexpr int bits = 3; using block = block_vtq3_1; };
template <> struct fattn_gqa_tq<GGML_TYPE_VTQ4_1> { static constexpr int bits = 4; using block = block_vtq4_1; };

// codebook of a type, scaled by PQ_CUDA_CB_SCALE (as the decoders of fattn-tq.cuh apply it)
template <ggml_type type>
static __device__ __forceinline__ float fattn_gqa_tq_centroid(const int i) {
    if constexpr (type == GGML_TYPE_VTQ2_1) {
        return VTQ_CUDA_CB_2BIT[i] * PQ_CUDA_CB_SCALE;
    } else if constexpr (fattn_gqa_tq<type>::bits == 2) {
        return PQ_CUDA_CB_2BIT[i] * PQ_CUDA_CB_SCALE;
    } else if constexpr (fattn_gqa_tq<type>::bits == 3) {
        return PQ_CUDA_CB_3BIT[i] * PQ_CUDA_CB_SCALE;
    } else {
        return PQ_CUDA_CB_4BIT[i] * PQ_CUDA_CB_SCALE;
    }
}

// element j (0..31) of a q5_0 block: 4 low bits from qs, the fifth from qh, offset by 16 (used by fattn-tq-wmma.cuh)
static __device__ __forceinline__ float fattn_gqa_q5_0(const block_q5_0 * b, const int j) {
    const int lo = j < 16 ? b->qs[j] & 0xF : b->qs[j - 16] >> 4;
    const int hi = (b->qh[j / 8] >> (j % 8)) & 1;
    return __half2float(b->d) * float((lo | (hi << 4)) - 16);
}

// The 2-bit codebooks have 4 entries: kept in registers and selected, no memory access per element.
template <ggml_type type>
static __device__ __forceinline__ float4 fattn_gqa_cb2() {
    return make_float4(fattn_gqa_tq_centroid<type>(0), fattn_gqa_tq_centroid<type>(1),
                       fattn_gqa_tq_centroid<type>(2), fattn_gqa_tq_centroid<type>(3));
}

static __device__ __forceinline__ float fattn_gqa_sel2(const float4 cb, const int c) {
    const float lo = c & 1 ? cb.y : cb.x;
    const float hi = c & 1 ? cb.w : cb.z;
    return c & 2 ? hi : lo;
}

// Which elements of a K or V row lane t owns (EPL = D/32 of them), per type, chosen so that every load instruction of
// a warp is as wide as the block layout allows and reads one contiguous stretch of the row (a warp load spread over
// the whole row touches more 32-byte sectors, measured slower for q8_0):
//   q8_0 K: element t of every block (one byte per lane and block, the scale a broadcast; element pairs and 4
//           consecutive elements per lane were 3-7 % slower at 32k context on an RTX 2060)
//   q8_0 V: the EPL consecutive elements from t*EPL, one block, so one scale per row
//   f16: element pairs, lane t owns 2t and 2t+1 of every 64 elements
//   q4_0, q5_0: one 2-byte load holds elements 2s, 2s+1, 2s+16, 2s+17 of a block (s = t % 8, the block t/8 of each 128)
//   codebook types: groups of 4, lane t owns 4t .. 4t+3 of every 128 elements
template <ggml_type type, bool as_V, int EPL>
static constexpr __device__ int fattn_gqa_group() {
    return type == GGML_TYPE_Q8_0 ? (as_V ? EPL : 1) : type == GGML_TYPE_F16 ? 2 : 4;
}

template <ggml_type type, bool as_V, int EPL>
static __device__ __forceinline__ int fattn_gqa_idx(const int t, const int e) {
    if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
        return 32*(t/8 + 4*(e/4)) + 2*(t % 8) + (e % 2) + 16*((e/2) % 2);
    } else {
        constexpr int g = fattn_gqa_group<type, as_V, EPL>();
        return 32*g*(e/g) + g*t + e % g;
    }
}

// Reads the EPL elements lane t owns of a K or V row (see fattn_gqa_idx), scaled by their block scale: once per
// element, so that the dot products and the V accumulation take one FMA per element and column. The quantized blocks
// put a 2-byte scale first and are an even number of bytes long, so 2-byte loads are aligned. Codebook types: cb holds
// the 3- and 4-bit codebook (shared memory), cb2 the 2-bit one.
template <ggml_type type, bool as_V, int EPL>
static __device__ __forceinline__ void fattn_gqa_load(
        const char * __restrict__ row, const int t, const float * __restrict__ cb, const float4 cb2,
        float * __restrict__ v) {
    static_assert(EPL == 4 || EPL == 8, "4 or 8 elements per lane");
    constexpr int g = fattn_gqa_group<type, as_V, EPL>();
#pragma unroll
    for (int gi = 0; gi < EPL/g; ++gi) {
        float * vg = v + gi*g;
        if constexpr (type == GGML_TYPE_F16) {
            const float2 f = __half22float2(((const half2 *) row)[32*gi + t]);
            vg[0] = f.x;
            vg[1] = f.y;
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            if constexpr (g == 1) {
                const block_q8_0 * b = (const block_q8_0 *) row + gi;
                vg[0] = __half2float(b->d) * b->qs[t];
            } else {
                const block_q8_0 * b = (const block_q8_0 *) row + t*g/QK8_0;
                const float d = __half2float(b->d);
#pragma unroll
                for (int i = 0; i < g; ++i) {
                    vg[i] = d * b->qs[t*g % QK8_0 + i];
                }
            }
        } else if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
            const int s = t % 8;
            using block_t = std::conditional_t<type == GGML_TYPE_Q4_0, block_q4_0, block_q5_0>;
            const block_t * b = (const block_t *) row + t/8 + 4*gi;
            uchar2 q;
            ggml_cuda_memcpy_1<2>(&q, b->qs + 2*s);
            int c[4] = {q.x & 0xF, q.y & 0xF, q.x >> 4, q.y >> 4};
            if constexpr (type == GGML_TYPE_Q5_0) {
                uint32_t qh;
                ggml_cuda_memcpy_1<4, 2>(&qh, b->qh);
                c[0] |= ((qh >> (2*s +  0)) & 1) << 4;
                c[1] |= ((qh >> (2*s +  1)) & 1) << 4;
                c[2] |= ((qh >> (2*s + 16)) & 1) << 4;
                c[3] |= ((qh >> (2*s + 17)) & 1) << 4;
            }
            constexpr int offset = type == GGML_TYPE_Q4_0 ? 8 : 16;
            const float d = __half2float(b->d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vg[i] = d * float(c[i] - offset);
            }
        } else {
            constexpr int bits = fattn_gqa_tq<type>::bits;
            static_assert(bits >= 2 && bits <= 4, "no reader for this type");
            const typename fattn_gqa_tq<type>::block * b = (const typename fattn_gqa_tq<type>::block *) row + 4*gi + t/8;
            const int o = 4*(t % 8); // first of the 4 elements in the block
            uint32_t w;
            int shift = 0;
            if constexpr (bits == 4) {
                uint16_t w16;
                ggml_cuda_memcpy_1<2>(&w16, b->qs + o/2);
                w = w16;
            } else if constexpr (bits == 2) {
                w = b->qs[o/4];
            } else {
                // 3 bit: the codes start at bit 3*o; read the 4 bytes from the even byte at or below it (KTQ3: may reach
                // into the sign bytes, VTQ3: into the tail padding, both inside the block)
                const int eo = (3*o/8) & ~1;
                ggml_cuda_memcpy_1<4, 2>(&w, b->qs + eo);
                shift = 3*o - 8*eo;
            }
            const float d = __half2float(b->d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = (w >> (shift + bits*i)) & ((1 << bits) - 1);
                vg[i] = d * (bits == 2 ? fattn_gqa_sel2(cb2, c) : cb[c]);
            }
        }
    }
}

// component i of a float4 (i known at compile time after unrolling; a pointer cast would put the value on the stack)
static __device__ __forceinline__ float fattn_gqa_f4(const float4 v, const int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool use_sparse>
__launch_bounds__(128, GGML_CUDA_FA_GQA_MIN_BLOCKS)
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
    constexpr int EPL      = D / WARP_SIZE; // elements per lane of every K and V row, see fattn_gqa_idx
    constexpr bool K_tq    = GGML_TYPE_IS_KTQ(type_K);
    static_assert(D == 128 || D == 256, "head size 128 or 256");
    static_assert(ncols == 4 || ncols == 8, "4 or 8 columns");


    const int gqa_ratio  = ne02 / ne12;
    const int ntiles_gqa = (gqa_ratio + ncols - 1) / ncols;
    const int sequence   = blockIdx.z / (ntiles_gqa*ne12);
    const int z          = blockIdx.z - sequence*ntiles_gqa*ne12;
    const int kv_head    = z / ntiles_gqa;
    const int head0      = kv_head*gqa_ratio + (z - kv_head*ntiles_gqa)*ncols; // first Q head of this block
    const int ncols_valid = min(ncols, (kv_head + 1)*gqa_ratio - head0);

    Q += nb03*sequence + nb02*head0;
    K += nb13*sequence + int64_t(nb12)*kv_head; // int64: past about a million cells nb12*kv_head overflows int32
    V += nb23*sequence + int64_t(nb22)*kv_head;
    const half * maskh = (const half *) (mask + nb33*(sequence % ne33));

    // sparse: the rows of position k are indices[k], positions from n_rows on are padding
    const int32_t * indices = use_sparse ? KV_max + int64_t(sequence % ne33)*ne11 : nullptr;
    const int       n_rows  = use_sparse ? KV_max[int64_t(ne33)*ne11 + sequence % ne33] : ne11;

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;

    // codebooks: 3 and 4 bit in shared memory (at most 16 entries, distinct banks), 2 bit in registers
    constexpr int cb_K_n = fattn_gqa_tq<type_K>::bits >= 3 ? 1 << fattn_gqa_tq<type_K>::bits : 1;
    constexpr int cb_V_n = fattn_gqa_tq<type_V>::bits >= 3 ? 1 << fattn_gqa_tq<type_V>::bits : 1;
    __shared__ float cb_K[cb_K_n];
    __shared__ float cb_V[cb_V_n];
    if constexpr (cb_K_n > 1) {
        if (tid < cb_K_n) {
            cb_K[tid] = fattn_gqa_tq_centroid<type_K>(tid);
        }
    }
    if constexpr (cb_V_n > 1) {
        if (tid < cb_V_n) {
            cb_V[tid] = fattn_gqa_tq_centroid<type_V>(tid);
        }
    }
    float4 cb2_K = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 cb2_V = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (fattn_gqa_tq<type_K>::bits == 2) {
        cb2_K = fattn_gqa_cb2<type_K>();
    }
    if constexpr (fattn_gqa_tq<type_V>::bits == 2) {
        cb2_V = fattn_gqa_cb2<type_V>();
    }

    __shared__ __align__(16) float KQ[nthreads*ncols]; // [row][column]: scores, then softmax weights, of the rows of an iteration
    __shared__ float VKQ_combine[nwarps*D];

    // Q in registers, lane t holds the elements of K it owns (KTQ: rotated into the Hadamard domain, which works on
    // 32-element blocks with one element per lane, so the rotated values go through shared memory once)
    float Q_f32[ncols][EPL];
    float * Q_tmp = VKQ_combine + threadIdx.y*D; // this warp's part
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb02);
        if constexpr (K_tq) {
            const float sign = ktq_cuda_shared_sign(threadIdx.x);
#pragma unroll
            for (int bi = 0; bi < D/WARP_SIZE; ++bi) {
                const float q = j < ncols_valid ? Q_j[bi*WARP_SIZE + threadIdx.x] * scale : 0.0f;
                Q_tmp[bi*WARP_SIZE + threadIdx.x] = ktq_cuda_fwht_warp(q * sign);
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < EPL; ++e) {
                Q_f32[j][e] = Q_tmp[fattn_gqa_idx<type_K, false, EPL>(threadIdx.x, e)];
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int e = 0; e < EPL; ++e) {
                Q_f32[j][e] = j < ncols_valid ? Q_j[fattn_gqa_idx<type_K, false, EPL>(threadIdx.x, e)] * scale : 0.0f;
            }
        }
    }
    __syncthreads(); // codebooks written, Q_tmp no longer needed

    float VKQ[ncols][EPL] = {{0.0f}};
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

        // KQ: warp w computes the scores of rows [w*32, w*32 + 32) of this iteration, one row at a time; the ncols
        // partial sums of a row are reduced together (reducing several rows together was 5-25 % slower at 32k context:
        // more live registers)
#pragma unroll 4
        for (int i_KQ_0 = 0; i_KQ_0 < WARP_SIZE; ++i_KQ_0) {
            const int    row_KQ = use_sparse ? __shfl_sync(0xFFFFFFFF, row_own, i_KQ_0, WARP_SIZE) :
                                               k_VKQ_0 + threadIdx.y*WARP_SIZE + i_KQ_0;
            float k[EPL];
            fattn_gqa_load<type_K, false, EPL>(K + int64_t(row_KQ)*nb11, threadIdx.x, cb_K, cb2_K, k);
            float s[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                s[j] = 0.0f;
#pragma unroll
                for (int e = 0; e < EPL; ++e) {
                    s[j] += k[e] * Q_f32[j][e];
                }
            }
            const float sum = fattn_gqa_reduce_cols<ncols>(s);
            if (threadIdx.x % (WARP_SIZE/ncols) == 0) {
                KQ[(threadIdx.y*WARP_SIZE + i_KQ_0)*ncols + fattn_gqa_col<ncols>(threadIdx.x)] = sum;
            }
        }
        __syncwarp();

        // online softmax: lane t of warp w owns row w*32 + t
        const float mask_val = valid_own ? __half2float(maskh[row_own]) : -INFINITY;
        float4 * KQ_own = (float4 *) (KQ + tid*ncols);
#pragma unroll
        for (int j4 = 0; j4 < ncols/4; ++j4) {
            const float4 s4 = KQ_own[j4];
            float p4[4];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = 4*j4 + jj;
                float s = fattn_gqa_f4(s4, jj);
                if (use_logit_softcap) {
                    s = logit_softcap*tanhf(s);
                }
                s += mask_val;

                const float KQ_max_new = fmaxf(KQ_max[j], warp_reduce_max(s + FATTN_KQ_MAX_OFFSET));
                const float KQ_max_scale = exp2f((KQ_max[j] - KQ_max_new) * LOG2E);
                KQ_max[j] = KQ_max_new;

                const float p = exp2f((s - KQ_max[j]) * LOG2E);
                KQ_sum[j] = KQ_sum[j]*KQ_max_scale + p;
                p4[jj] = p;

#pragma unroll
                for (int e = 0; e < EPL; ++e) {
                    VKQ[j][e] *= KQ_max_scale;
                }
            }
            KQ_own[j4] = make_float4(p4[0], p4[1], p4[2], p4[3]);
        }
        __syncwarp();

        // VKQ: warp w accumulates its 32 V rows, each read once for all columns (unrolled 4 times: 2-4 % faster at 32k
        // context than 2 times, 8 times gains nothing more)
#pragma unroll 4
        for (int k0 = 0; k0 < WARP_SIZE; ++k0) {
            const int k = threadIdx.y*WARP_SIZE + k0;
            const int row_V = use_sparse ? __shfl_sync(0xFFFFFFFF, row_own, k0, WARP_SIZE) : k_VKQ_0 + k;
            float v[EPL];
            fattn_gqa_load<type_V, true, EPL>(V + int64_t(row_V)*nb21, threadIdx.x, cb_V, cb2_V, v);
            const float4 * w4 = (const float4 *) (KQ + k*ncols); // softmax weights of row k, all columns
#pragma unroll
            for (int j4 = 0; j4 < ncols/4; ++j4) {
                const float4 p = w4[j4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float pj = fattn_gqa_f4(p, jj);
#pragma unroll
                    for (int e = 0; e < EPL; ++e) {
                        VKQ[4*j4 + jj][e] += v[e]*pj;
                    }
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

        // lane t holds the output elements of the V positions it owns
#pragma unroll
        for (int e = 0; e < EPL; ++e) {
            VKQ_combine[threadIdx.y*D + fattn_gqa_idx<type_V, true, EPL>(threadIdx.x, e)] = VKQ[j][e] * scale_w;
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
static fattn_kernel_t fattn_gqa_kernel(const bool use_logit_softcap, const bool use_sparse) {
    if (use_sparse) {
        return use_logit_softcap ? flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, true,  true> :
                                   flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, false, true>;
    }
    return use_logit_softcap ? flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, true,  false> :
                               flash_attn_ext_vec_gqa<D, ncols, type_K, type_V, false, false>;
}

template <int D, int ncols, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_vec_gqa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool use_sparse) {
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel = fattn_gqa_kernel<D, ncols, type_K, type_V>(logit_softcap != 0.0f, use_sparse);
    constexpr int nwarps = 128 / WARP_SIZE;
    launch_fattn<D, 1, ncols>(ctx, dst, fattn_kernel, nwarps, 0, 128, false, false, false, WARP_SIZE, use_sparse);
}

// The supported (K type, V type) pairs for one head size D, block width ncols and K type: every V type of the list
// below. Each (D, ncols, K type) is compiled in its own TU (template-instances/fattn-gqa-instance-*.cu) so that they
// build in parallel; the dispatcher in fattn-vec-gqa.cu only references them. Returns false if V is not covered.
// f16 K with f16 V stays with the MMA kernel (decode at 32k reads the cache at ~90 % of the bandwidth there).
#define FATTN_VEC_GQA_CASE_V(type_V_)                                                               \
    if (V->type == (type_V_)) {                                                                     \
        if constexpr (!(type_K == GGML_TYPE_F16 && (type_V_) == GGML_TYPE_F16)) {                   \
            ggml_cuda_flash_attn_ext_vec_gqa_case<D, ncols, type_K, type_V_>(ctx, dst, use_sparse); \
            return true;                                                                            \
        }                                                                                           \
        return false;                                                                               \
    }

template <int D, int ncols, ggml_type type_K>
bool ggml_cuda_flash_attn_ext_vec_gqa_d(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool use_sparse) {
    const ggml_tensor * V = dst->src[2];

    FATTN_VEC_GQA_CASE_V(GGML_TYPE_F16)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_Q8_0)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_Q5_0)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_Q4_0)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_VTQ2_1)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_VTQ3_1)
    FATTN_VEC_GQA_CASE_V(GGML_TYPE_VTQ4_1)

    return false;
}

// __attribute__((used)): same reason as GGML_FATTN_VEC_USED in fattn-vec.cuh (RDC must not prune the instances)
#if defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
#  define GGML_FATTN_VEC_GQA_USED
#else
#  define GGML_FATTN_VEC_GQA_USED __attribute__((used))
#endif

#define DECL_FATTN_VEC_GQA_CASE(D, ncols, type_K)                                     \
    template GGML_FATTN_VEC_GQA_USED bool ggml_cuda_flash_attn_ext_vec_gqa_d             \
    <D, ncols, type_K>(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool use_sparse)

#define DECL_FATTN_VEC_GQA_CASES_K(D, ncols)                       \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_F16);       \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_Q8_0);      \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_Q5_0);      \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_Q4_0);      \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_KTQ2_1);    \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_KTQ3_1);    \
    extern DECL_FATTN_VEC_GQA_CASE(D, ncols, GGML_TYPE_KTQ4_1)

DECL_FATTN_VEC_GQA_CASES_K(256, 4);
DECL_FATTN_VEC_GQA_CASES_K(128, 4);
DECL_FATTN_VEC_GQA_CASES_K(256, 8);
DECL_FATTN_VEC_GQA_CASES_K(128, 8);
