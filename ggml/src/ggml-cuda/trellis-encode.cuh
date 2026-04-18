// Trellis v2 CUDA encoder (full Viterbi, no pruning).
//
// 1 CUDA block == 1 trellis block (N=256 samples, S=2^16=65536 states).
// VTQ_ENC_THREADS=256 threads per CUDA block. Each thread owns 256 "prev"
// states (stride-256 sharding). At every DP step, for every (prev, bits)
// transition the candidate (cost, prev) is committed into dp_next[next] via
// 64-bit atomicMin where the HIGH 32 bits hold __float_as_uint(cost) (costs
// are non-negative so uint ordering matches float ordering) and the LOW 32
// bits hold the prev index. This is the DP winner on ties (lowest prev)
// which is numerically equivalent to the CPU reference within fp32 precision.
//
// Workspace (dp_cur, dp_next, backtrack) lives in a global-memory pool owned
// by trellis.cu. Pool has `pool_slots` slots; host launches the kernel in
// waves of `pool_slots` CUDA blocks with cudaStreamSynchronize between waves
// so slots can be re-used.
//
// This kernel folds the `k_set_rows_pq` per-block indexing into thread 0
// (computing src_block + dst_block pointers once per CUDA block), so that
// the same call signature can replace the PQ path from set-rows.cu.

#pragma once

#include "common.cuh"
#include "ggml-common.h"
#include "../ggml-trellis.h"
#include "trellis.cuh"

#include <cstdint>
#include <cfloat>

#ifndef VTQ_ENC_THREADS
#define VTQ_ENC_THREADS 256
#endif

#define VTQ_ENC_N 256
#define VTQ_ENC_L 16
#define VTQ_ENC_S (1u << VTQ_ENC_L)

struct vtq_encode_workspace {
    float    * dp_cur;
    float    * dp_next;
    uint16_t * bt;
    int        pool_slots;
};

const vtq_encode_workspace * vtq_get_encode_workspace(cudaStream_t stream);
void vtq_free_encode_workspace(void);

// One CUDA block == one flat trellis-block index.
//   i runs [0 .. ne_total-1], ne_total = (ne00*ne01*ne02*ne03)/qk.
// Per-block, thread 0 decodes (i00, i01, i02, i03) and computes:
//   - src_block = src0 + i01*s01 + i02*s02 + i03*s03 + i00   (float *)
//   - dst_block = dst + ((dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_t))
//                       + (i00 / qk)
// Other threads read resolved pointers from shared memory.
template <typename block_t, typename idx_t, int K>
__global__ void k_vtq_encode_trellis_set_rows(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_t     * __restrict__ dst,
        const int64_t ne_total,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00,
        const uint3   ne01,
        const uint3   ne02,
        const uint3   ne11_fd,
        const uint3   ne12_fd,
        float       * __restrict__ ws_dp_cur,
        float       * __restrict__ ws_dp_next,
        uint16_t    * __restrict__ ws_bt,
        const int                     pool_slots,
        const int64_t                 wave_block_offset)
{
    constexpr int      N      = VTQ_ENC_N;
    constexpr int      L      = VTQ_ENC_L;
    constexpr int      QK_VTQ_T = N;  // 256
    constexpr uint32_t S      = VTQ_ENC_S;
    constexpr uint32_t Kmask  = (1u << K) - 1u;
    constexpr int      kshift = L - K;

    const int     tid = threadIdx.x;
    const int64_t i   = wave_block_offset + (int64_t)blockIdx.x;
    if (i >= ne_total) return;

    __shared__ const float * s_src_block;
    __shared__ block_t     * s_dst_block;

    if (tid == 0) {
        const int64_t i_base = i * QK_VTQ_T;
        uint32_t tmp = (uint32_t) i_base;
        uint2    dm;

        dm = fast_div_modulo(tmp, ne00);
        const int64_t i00 = dm.y;
        tmp = dm.x;

        dm = fast_div_modulo(tmp, ne01);
        const int64_t i01 = dm.y;
        tmp = dm.x;

        dm = fast_div_modulo(tmp, ne02);
        const int64_t i02 = dm.y;
        const int64_t i03 = dm.x;

        const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
        const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
        const int64_t i10 = i01;

        const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

        const float * src0_row   = src0 + i01*s01 + i02*s02 + i03*s03;
        block_t     * dst_row_p  = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_t);

        s_src_block = src0_row + i00;
        s_dst_block = dst_row_p + i00 / QK_VTQ_T;

        (void) ne10; (void) ne11; (void) ne12; (void) ne13;
    }
    __syncthreads();

    const float * x_row     = s_src_block;
    block_t     * dst_block = s_dst_block;

    const int slot = (int)(blockIdx.x % (uint32_t)pool_slots);
    float    * dp_cur  = ws_dp_cur  + (int64_t)slot * (int64_t)S;
    // dp_next is sized for uint64_t per state; advance by 2*S floats per slot.
    float    * dp_next = ws_dp_next + (int64_t)slot * (int64_t)(2u * S);
    uint16_t * bt_base = ws_bt      + (int64_t)slot * (int64_t)N * (int64_t)S;

    __shared__ float    s_xn[VTQ_ENC_N];
    __shared__ float    s_partial[VTQ_ENC_THREADS];
    __shared__ uint32_t s_partial_u[VTQ_ENC_THREADS];
    __shared__ float    s_norm;
    __shared__ float    s_inv_norm;
    __shared__ uint32_t s_best_state;

    // 1) Load x + compute norm
    float local_sq = 0.0f;
    for (int j = tid; j < N; j += VTQ_ENC_THREADS) {
        float v = x_row[j];
        s_xn[j] = v;
        local_sq += v * v;
    }
    s_partial[tid] = local_sq;
    __syncthreads();
    for (int off = VTQ_ENC_THREADS >> 1; off > 0; off >>= 1) {
        if (tid < off) s_partial[tid] += s_partial[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        float nv = sqrtf(s_partial[0]);
        s_norm     = nv;
        s_inv_norm = (nv > 1e-30f) ? (1.0f / nv) : 0.0f;
    }
    __syncthreads();

    const float norm = s_norm;

    if (norm <= 1e-30f) {
        if (tid == 0) {
            dst_block->start_state = 0;
            dst_block->d           = __float2half(0.0f);
        }
        const int qs_bytes = (N * K + 7) / 8;
        for (int b = tid; b < qs_bytes; b += VTQ_ENC_THREADS) dst_block->qs[b] = 0;
        return;
    }

    const float inv_norm = s_inv_norm;
    for (int j = tid; j < N; j += VTQ_ENC_THREADS) s_xn[j] *= inv_norm;
    __syncthreads();

    const float cb_scale = rsqrtf((float)N);

    // 2) dp_cur = 0
    for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) dp_cur[s] = 0.0f;
    __syncthreads();

    // 3) Viterbi DP
    for (int step = 0; step < N; step++) {
        uint16_t * bt_i = bt_base + (int64_t)step * (int64_t)S;

        {
            uint64_t * dpn = reinterpret_cast<uint64_t *>(dp_next);
            const uint64_t init_v = ((uint64_t)__float_as_uint(FLT_MAX) << 32) | 0xFFFFFFFFull;
            for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) dpn[s] = init_v;
        }
        __syncthreads();

        const float xi = s_xn[step];

        for (uint32_t prev = tid; prev < S; prev += VTQ_ENC_THREADS) {
            const float pc = dp_cur[prev];
            if (pc >= FLT_MAX) continue;  // skip unreachable states (cost would saturate/overwrite init)
            #pragma unroll
            for (uint32_t bits = 0; bits <= Kmask; bits++) {
                uint32_t next = ((prev >> K) | (bits << kshift)) & 0xFFFFu;
                float code   = vtq_trellis_table_storage[next] * cb_scale;
                float diff   = xi - code;
                float cost   = pc + diff * diff;
                uint64_t packed = ((uint64_t)__float_as_uint(cost) << 32) | (uint64_t)prev;
                unsigned long long * dst_u =
                    reinterpret_cast<unsigned long long *>(
                        reinterpret_cast<uint64_t *>(dp_next) + next);
                atomicMin(dst_u, (unsigned long long)packed);
            }
        }
        __syncthreads();

        {
            uint64_t * dpn = reinterpret_cast<uint64_t *>(dp_next);
            for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) {
                uint64_t v = dpn[s];
                uint32_t cost_u = (uint32_t)(v >> 32);
                uint32_t prev_u = (uint32_t)(v & 0xFFFFFFFFu);
                dp_cur[s] = __uint_as_float(cost_u);
                bt_i[s]   = (uint16_t)(prev_u & 0xFFFFu);
            }
        }
        __syncthreads();
    }

    // 4) argmin
    float    local_best = FLT_MAX;
    uint32_t local_arg  = 0;
    for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) {
        float c = dp_cur[s];
        if (c < local_best) { local_best = c; local_arg = s; }
    }
    s_partial[tid]   = local_best;
    s_partial_u[tid] = local_arg;
    __syncthreads();
    for (int off = VTQ_ENC_THREADS >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = s_partial[tid], b = s_partial[tid + off];
            if (b < a) {
                s_partial[tid]   = b;
                s_partial_u[tid] = s_partial_u[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) s_best_state = s_partial_u[0];
    __syncthreads();

    // 5) Backtrack + emit (thread 0)
    __shared__ uint16_t s_states[VTQ_ENC_N + 1];
    if (tid == 0) {
        s_states[N] = (uint16_t)(s_best_state & 0xFFFFu);
        for (int step = N - 1; step >= 0; step--) {
            uint16_t * bt_i = bt_base + (int64_t)step * (int64_t)S;
            s_states[step] = bt_i[s_states[step + 1]];
        }

        dst_block->start_state = s_states[0];

        uint8_t * qs = dst_block->qs;
        const int qs_bytes = (N * K + 7) / 8;
        for (int b = 0; b < qs_bytes; b++) qs[b] = 0;

        float recon_sq = 0.0f;
        for (int step = 0; step < N; step++) {
            uint32_t st   = (uint32_t)s_states[step + 1];
            uint32_t bits = (st >> kshift) & Kmask;
            int bo        = step * K;
            int byte      = bo >> 3;
            int shift     = bo & 7;
            qs[byte] |= (uint8_t)((bits << shift) & 0xFFu);
            if (shift + K > 8) {
                qs[byte + 1] |= (uint8_t)((bits >> (8 - shift)) & 0xFFu);
                if (shift + K > 16) {
                    qs[byte + 2] |= (uint8_t)((bits >> (16 - shift)) & 0xFFu);
                }
            }
            float code = vtq_trellis_table_storage[st] * cb_scale;
            recon_sq += code * code;
        }
        float recon_norm = sqrtf(recon_sq);
        float d_out = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
        dst_block->d = __float2half(d_out);

#ifdef VTQ_ENC_DEBUG
        if (blockIdx.x == 0) {
            printf("[GPU_ENC] blk=%lld start=%u d=%.6f qs[0..3]=%02x %02x %02x %02x norm=%.6f recon=%.6f\n",
                (long long)i, (unsigned)s_states[0], d_out,
                dst_block->qs[0], dst_block->qs[1], dst_block->qs[2], dst_block->qs[3],
                norm, recon_norm);
            printf("[GPU_ENC] x[0..5]=%.4f %.4f %.4f %.4f %.4f %.4f\n",
                s_xn[0] * norm, s_xn[1] * norm, s_xn[2] * norm,
                s_xn[3] * norm, s_xn[4] * norm, s_xn[5] * norm);
        }
#endif
    }
}

// Host-side launcher that mirrors set_rows_cuda_pq signature.
template <typename idx_t, typename block_t, int K>
static void vtq_cuda_encode_set_rows(
        const float * src0_d, const idx_t * src1_d, block_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream)
{
    GGML_ASSERT(ne00 % VTQ_ENC_N == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / VTQ_ENC_N;
    if (ne_total == 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) return;

    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const vtq_encode_workspace * ws = vtq_get_encode_workspace(stream);
    const int pool_slots = ws->pool_slots;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
    const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
    const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

    const int64_t num_waves = (ne_total + pool_slots - 1) / pool_slots;
    for (int64_t w = 0; w < num_waves; w++) {
        const int64_t wave_start = w * pool_slots;
        const int64_t remaining  = ne_total - wave_start;
        const int     wave_n     = (int)((remaining < pool_slots) ? remaining : pool_slots);

        dim3 grid(wave_n, 1, 1);
        dim3 block(VTQ_ENC_THREADS, 1, 1);

        k_vtq_encode_trellis_set_rows<block_t, idx_t, K><<<grid, block, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total,
            ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd,
            ws->dp_cur, ws->dp_next, ws->bt,
            pool_slots, wave_start);

        if (num_waves > 1) {
            cudaStreamSynchronize(stream);
        }
    }
}
