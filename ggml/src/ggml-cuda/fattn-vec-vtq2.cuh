// fattn-vec-vtq2.cuh — E11-style cached V-dequant kernel for VTQ_2 family.
//
// Phase 3A1 scope: KTQ2_1 × VTQ3_2, D=128, ncols=1 (2 template instances).
// Feature-flag gated by `FATTN_VTQ2_CACHED` (default OFF). When OFF, this
// header is still compilable but the kernel is never dispatched.
//
// Design: docs/plans/2026-04-21-e11-cuda-port-spec.md
// Triton ref: gpu00:/home/claude/llama-tq/triton-autoresearch/variants_r3.py
//
// Key difference from `fattn_vec` (legacy):
//   Legacy: each thread independently calls `dequantize_V_vtq_2` per V-row
//           per inner-loop iteration. `vtq_trellis_table_storage[state]`
//           looked up from constant/global LUT every time.
//   Here:   once per outer K-loop iteration × warp × V-row, the warp
//           cooperatively decodes all 128 samples of the row's VTQ block
//           into shared memory (`smem_V_cache[nwarps][128]`). The inner
//           accumulate loop reads pre-decoded fp16 samples from shmem.
//           → 0 LUT accesses per query-step; LUT traffic collapses by ~128x.
//
// Occupancy target (sm_75, D=128):
//   __launch_bounds__(128, 4)   — 4 blocks/SM. Requires reg/thread ≤ 127.
//   Shmem: KQ + V_cache = ~1.3 KiB/block → far below 64 KiB cap.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tq.cuh"  // get_vec_dot_KQ + KTQ/VTQ vec-dot helpers

#include <type_traits>

// Guard: this translation unit's kernel is only compiled in when the
// feature flag is defined. We still compile the dispatch thunks below
// unconditionally so CMake sees the symbols, but the actual kernel body
// only produces template instantiations under the flag.
#ifndef FATTN_VTQ2_CACHED
// Not active: nothing from here is used by the dispatcher.
#endif

static constexpr __device__ int ggml_cuda_fattn_vec_vtq2_get_nthreads_device() {
    return 128;
}

// Warp-cooperative decode of ONE VTQ_2 block (128 samples) into shmem.
// Each lane (0..31) decodes 4 samples (128 / 32 = 4) using the O(1)
// `vtq_state_at<K>` formula. Output is written as fp16 into `smem_row`
// which points at the per-warp slot (size == QK_VTQ_TRELLIS halfs).
//
// Template params:
//   block_t : block_vtq2_2 / block_vtq3_2 / block_vtq4_2
//   K       : bits-per-sample (2, 3, 4)
template <typename block_t, int K>
static __device__ __forceinline__ void vtq2_block_warm(
        const block_t * __restrict__ x_block,
        half          * __restrict__ smem_row) {
    const int lane = threadIdx.x;  // 0..31 within warp
    const float   d  = (float) x_block->d;
    const uint16_t s0 = x_block->start_state;
    const uint8_t * qs = x_block->qs;

    constexpr int N = QK_VTQ_TRELLIS;  // 128
    const float cb_scale = rsqrtf((float) N);
    const float ds = cb_scale * d;

    if (d == 0.0f) {
        // Zero block — fill with zeros.
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            smem_row[lane * 4 + s] = __float2half(0.0f);
        }
        return;
    }

    // 32 lanes × 4 samples = 128 samples decoded per call.
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int il = lane * 4 + s;  // 0..127
        const uint32_t state = vtq_state_at<K>(s0, qs, il + 1);
        const float val = vtq_trellis_table_storage[state] * ds;
        smem_row[il] = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// Main kernel. Template parameters match `flash_attn_ext_vec` so dispatch
// is a drop-in replacement under the ifdef.
// ---------------------------------------------------------------------------
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

template <int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_vec_vtq2_get_nthreads_device(), 4)
static __global__ void flash_attn_ext_vec_vtq2_cached(
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

    // Phase 3A1: only KTQ2_1 × VTQ3_2, D=128, ncols=1 are real instances.
    // Every other combination is compiled to an early-out so cicc does not
    // explode on template variants we do not ship yet.
    static_assert(type_V == GGML_TYPE_VTQ2_2 || type_V == GGML_TYPE_VTQ3_2 || type_V == GGML_TYPE_VTQ4_2,
                  "flash_attn_ext_vec_vtq2_cached: V must be VTQ_2 family");

    if (use_logit_softcap && !(D == 128 || D == 256 || D == 512)) {
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
        return;
    }

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);

    constexpr int nthreads    = ggml_cuda_fattn_vec_vtq2_get_nthreads_device();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_tq   = type_K == GGML_TYPE_KTQ1_1 || type_K == GGML_TYPE_KTQ2_1 || type_K == GGML_TYPE_KTQ3_1 || type_K == GGML_TYPE_KTQ4_1;
    constexpr bool Q_q8_1 = !Q_tq && type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;

    const int ic0 = blockIdx.x * ncols;

    const int sequence  = blockIdx.z / ne02;
    const int head      = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;
    Q += nb03*sequence + nb02*head + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    // ---- Shmem layout ----
    // [KQ buffer]           (existing, used for Q-conv + VKQ combine)
    // [V_cache[nwarps][128]] (new, fp16) — 1 KiB for nwarps=4.
    // For D=256, we'd need 2x; Phase 3A1 is D=128 only so one block per row.
    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif

    // V-cache: one fp16 row of 128 samples per warp.
    // VTQ blocks encode exactly QK_VTQ_TRELLIS (=128) samples. For D=128 each
    // V-row is exactly one VTQ block (il 0..127 == head-dim 0..127).
    __shared__ half smem_V_cache[nwarps][QK_VTQ_TRELLIS];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // --- Q conversion (same as legacy fattn-vec) ---
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ];
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}};
#endif
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float   Q_f32[ncols][D/WARP_SIZE];

    if constexpr (Q_tq) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int bi = 0; bi < D/WARP_SIZE; ++bi) { Q_f32[j][bi] = 0.0f; }
            } else {
                const float * Q_j = (const float *) (Q + j*nb01);
#pragma unroll
                for (int bi = 0; bi < D/WARP_SIZE; ++bi) {
                    Q_f32[j][bi] = Q_j[bi * WARP_SIZE + threadIdx.x] * scale;
                }
            }
        }
    } else if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (j0 + nwarps > ncols && j >= ncols) break;

            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }
        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) { Q_reg[j][k] *= scale_h2; }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale; Q_reg[j][k].y *= scale;
            }
        }
#endif
    }

    // --- Pick block type for cooperative warm based on type_V ---
    // Phase 3A1 only instantiates VTQ3_2. Other VTQ_2 types compile in for
    // completeness but never get launched (dispatch guards via ifdef).
    using block_t =
        typename std::conditional<type_V == GGML_TYPE_VTQ2_2, block_vtq2_2,
        typename std::conditional<type_V == GGML_TYPE_VTQ3_2, block_vtq3_2,
                                                              block_vtq4_2>::type>::type;
    constexpr int K_bits = (type_V == GGML_TYPE_VTQ2_2) ? 2
                         : (type_V == GGML_TYPE_VTQ3_2) ? 3 : 4;
    static_assert(D == 128, "Phase 3A1 supports D=128 only");
    static_assert(QK_VTQ_TRELLIS == 128, "Phase 3A1 assumes QK_VTQ_TRELLIS == 128");

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // --- KQ computation (identical to legacy) ---
        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) { KQ_max_new[j] = KQ_max[j]; }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (Q_tq) {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_f32[j], nullptr, nullptr);
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) sum = logit_softcap*tanhf(sum);
                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }
                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);
                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

        // --- V accumulate: E11-style cached decode ---
        // Each warp iterates through (WARP_SIZE / V_cols_per_iter) V-rows.
        // For nthreads_V=32 (VTQ_2 path), V_cols_per_iter=1 → 32 rows/warp.
        // For each V-row, we cooperatively decode its 128 samples into the
        // warp's shmem slot ONCE, then the ncols-fold query consume does
        // shmem reads only (no LUT access).

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            // Boundary guard: skip if this V-row is past the KV max.
            // Legacy kernel does not guard here because the outer loop
            // caps k_VKQ_0+nthreads; but at tail iterations some `k` values
            // within the warp may still be out of range for non-aligned ne11.
            // Legacy relies on KQ values being 0 for out-of-range rows via
            // the earlier mask/fmax path; we preserve that same behavior.
            // (No explicit guard added — matches legacy semantics.)

            // --- Warm: warp-cooperative decode of 128 samples into smem_V_cache[warpid] ---
            const block_t * x_block = (const block_t *) (V + k*nb21);
            vtq2_block_warm<block_t, K_bits>(x_block, smem_V_cache[threadIdx.y]);
            __syncwarp();

            // --- Consume: read decoded fp16 from shmem ---
#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                const int base = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;
#pragma unroll
                for (int i1 = 0; i1 < V_rows_per_thread/2; ++i1) {
                    // Cast through uint32_t to do an aligned 4B load of 2 halves.
                    const uint32_t packed = *reinterpret_cast<const uint32_t*>(&smem_V_cache[threadIdx.y][base + 2*i1]);
                    tmp[i1] = *reinterpret_cast<const half2*>(&packed);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) { KQ_k[j] = KQ[j*nthreads + k]; }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                const int base = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;
#pragma unroll
                for (int i1 = 0; i1 < V_rows_per_thread/2; ++i1) {
                    const half h0 = smem_V_cache[threadIdx.y][base + 2*i1 + 0];
                    const half h1 = smem_V_cache[threadIdx.y][base + 2*i1 + 1];
                    tmp[i1].x = __half2float(h0);
                    tmp[i1].y = __half2float(h1);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif
        }
    }

    // --- Sinks + combine + writeback (identical to legacy) ---
    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (j0 + nwarps > ncols && j >= ncols) break;
            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);
#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) KQ_max_shared[j][threadIdx.y] = KQ_max[j];
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) break;

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);
        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);
            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) dst_val /= KQ_sum[j_VKQ];
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) __syncthreads();
    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }

#else  // !FLASH_ATTN_AVAILABLE
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

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// ---------------------------------------------------------------------------
// Launcher — analogous to `ggml_cuda_flash_attn_ext_vec_case_impl` in
// fattn-vec.cuh, but launches `flash_attn_ext_vec_vtq2_cached` instead.
// ---------------------------------------------------------------------------
template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_vtq2_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    (void) cc;

    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads / WARP_SIZE;

    fattn_kernel_t fattn_kernel = flash_attn_ext_vec_vtq2_cached<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_vtq2_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            ggml_cuda_flash_attn_ext_vec_vtq2_case_impl<D, cols_per_block, type_K, type_V, false>(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_vtq2_case_impl<D, cols_per_block, type_K, type_V, true>(ctx, dst);
        }
        return;
    }

    // ncols>=2 not yet supported in the cached kernel. The dispatch site
    // in fattn-vec-dispatch-vtq2.cu only routes Q->ne[1] == 1 requests
    // here, so this path is unreachable in Phase 3A1. Abort if hit as a
    // safety net — a silent fallback to the legacy kernel would hide
    // accidental dispatch bugs.
    GGML_ABORT("flash_attn_ext_vec_vtq2_cached: ncols>=2 not supported in Phase 3A1");
}
