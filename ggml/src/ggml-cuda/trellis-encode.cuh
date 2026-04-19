// Trellis v2 CUDA encoder (full Viterbi, no pruning).
//
// 1 CUDA block == 1 trellis block (N=256 samples, S=2^16=65536 states).
// VTQ_ENC_THREADS=256 threads per CUDA block.
//
// RECEIVER-SIDE DP (Trick 6): each thread owns 256 "next" states
// (stride-256 sharding). At every DP step, per-next we gather the 2^K
// candidate prev states via the deterministic inverse of the bit-shift
// trellis: prev = ((next << K) | e) & 0xFFFF for e in [0, 2^K). The min
// is computed in private registers and committed to dp_next[next] with a
// single coalesced store — no atomics. The backtrack winner (prev index
// giving the min) is written to bt[step*S + next].
//
// Why this is faster than sender-side atomicMin:
//   * No serialization: 2^K edges into the same `next` collapse to a
//     register-level loop rather than 2^K contending atomicMin ops.
//   * Coalesced writes to dp_next (adjacent tids write adjacent next).
//   * Coalesced LUT reads: vtq_trellis_table_storage[next] reads 256
//     consecutive floats per warp.
//   * Tie-break matches sender-side packed atomicMin: we iterate e
//     ascending with strict `<`, and prev = (next<<K)|e → e=0 gives the
//     lowest prev, which is kept on ties (lowest prev wins).
//
// Workspace (dp_cur, dp_next, backtrack) lives in a global-memory pool owned
// by trellis.cu. Pool has `pool_slots` slots; host launches the kernel in
// waves of `pool_slots` CUDA blocks with cudaStreamSynchronize between waves
// so slots can be re-used. dp_next is allocated with uint64 stride (2*S
// floats per slot) — receiver-side only uses the first S floats of each slot;
// the tail remains reserved to preserve ABI with the host-side pool.
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

    // 2) dp_cur = 0 (all states reachable at step 0 with cost 0)
    for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) dp_cur[s] = 0.0f;
    __syncthreads();

    // 3) Viterbi DP — receiver-side (atomic-free).
    //
    // For each `next`, gather the 2^K incoming edges:
    //   prev = ((next << K) | e) & 0xFFFF  for e in [0, 2^K)
    // The code/cost-delta depends only on `next` (not prev), so we compute
    // it once per next and reuse across the 2^K candidates.
    //
    // Tie-break: sender-side packed atomicMin keeps the lowest prev on ties.
    // Here prev values for fixed next are consecutive: prev = base + e where
    // base = (next << K) & 0xFFFF, so iterating e=0..2^K-1 with strict `<`
    // keeps e=0 (lowest prev) on ties — matches sender-side.
    //
    // Unreachable prev states carry cost = FLT_MAX from the previous step;
    // FLT_MAX + d2 saturates to +inf in IEEE fp32 and naturally loses the
    // comparison against any finite cost — no explicit skip needed.
    for (int step = 0; step < N; step++) {
        uint16_t * bt_i = bt_base + (int64_t)step * (int64_t)S;

        const float xi = s_xn[step];

        for (uint32_t next = tid; next < S; next += VTQ_ENC_THREADS) {
            // LUT read is coalesced across a warp (adjacent tids → adjacent next).
            // Do NOT use __ldg: LUT is 256 KiB, 5x larger than Turing's 48 KiB RO
            // cache per SM — goes through L2 instead (~100% hit after warmup).
            const float code = vtq_trellis_table_storage[next] * cb_scale;
            const float diff = xi - code;
            const float d2   = diff * diff;

            const uint32_t base = (next << K) & 0xFFFFu;

            float    best_cost = FLT_MAX;
            uint32_t best_prev = 0;
            #pragma unroll
            for (uint32_t e = 0; e <= Kmask; e++) {
                const uint32_t prev = base | e;
                const float    pc   = dp_cur[prev];
                const float    cost = pc + d2;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_prev = prev;
                }
            }

            // Coalesced writes: adjacent tids → adjacent next.
            dp_next[next] = best_cost;
            bt_i[next]    = (uint16_t)(best_prev & 0xFFFFu);
        }
        __syncthreads();

        // Ping-pong swap: next step reads from what we just wrote.
        // Per-slot strides differ (dp_cur: S floats, dp_next: 2*S floats)
        // but both point into slots that are individually large enough for
        // S floats — swapping the pointers is safe for the DP payload.
        float * tmp = dp_cur;
        dp_cur  = dp_next;
        dp_next = tmp;
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
    }
}

// ============================================================
// BEAM-SEARCH encoder fast path for tg (ne11=1) — avoids Viterbi's
// 65k-state DP when we only encode 1 token per layer per step.
//
// Keeps the top-B best-cost states at each step (B=16 beams).
// Each step tries 2^K transitions per beam → B·2^K = 128 candidates,
// keeps top-B. Per block: N·B·2^K = ~32k ops vs Viterbi 134M (~4000×
// less GPU work), while quality stays within 1-3% PPL of Viterbi.
//
// Greedy (B=1) was measured to yield +60% PPL on wikitext-2 — too
// aggressive. B=16 is the smallest beam that empirically tracks
// Viterbi global-optimal for our 1-D codebook + shift-register
// structure (all within-beam ambiguity resolves in first ~4 steps).
//
// One CUDA block per (ne00/VTQ_ENC_N × ne01 × ne02 × ne03) item.
// One thread per block (sequential walk with B-wide SIMD in regs).
template <typename block_t, typename idx_t, int K>
__global__ void k_vtq_greedy_encode_set_rows(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_t     * __restrict__ dst,
        const int64_t ne_total,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3   ne00, const uint3   ne01, const uint3   ne02,
        const uint3   ne11_fd, const uint3   ne12_fd)
{
    constexpr int      N       = VTQ_ENC_N;
    constexpr int      L       = VTQ_ENC_L;
    constexpr uint32_t Lmask   = 0xFFFFu;
    constexpr uint32_t Kmask   = (1u << K) - 1u;
    constexpr int      kshift  = L - K;

    const int64_t i = blockIdx.x;
    if (i >= ne_total) return;
    if (threadIdx.x != 0) return;

    // Index decomposition — same as Viterbi kernel.
    const int64_t i_base = i * N;
    uint32_t tmp = (uint32_t) i_base;
    uint2    dm;
    dm = fast_div_modulo(tmp, ne00); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02); const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_t     * dst_row_p = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_t);
    const float * x_row    = src0_row + i00;
    block_t     * dst_block = dst_row_p + i00 / N;

    constexpr int B = 16;  // beam width

    // 1) Norm
    float sq = 0.0f;
    for (int j = 0; j < N; j++) { float v = x_row[j]; sq += v*v; }
    const float norm = sqrtf(sq);
    if (norm <= 1e-30f) {
        dst_block->start_state = 0;
        dst_block->d = __float2half(0.0f);
        const int qs_bytes = (N * K + 7) / 8;
        for (int b = 0; b < qs_bytes; b++) dst_block->qs[b] = 0;
        return;
    }
    const float inv_norm = 1.0f / norm;
    const float cb_scale = rsqrtf((float)N);

    // Beam state: cost, trailing state, backtrack. We keep a parent-
    // and-edge history so we can reconstruct the winning path at end.
    // Per-step storage: beam[b].state, beam[b].cost, parent[step][b]
    // (= parent-beam index), edge[step][b] (= K-bit emitted).
    float    beam_cost[B];
    uint32_t beam_state[B];
    uint8_t  parent[N][B];
    uint8_t  edge  [N][B];

    // Start: all beams at state 0, cost 0 (free start). This mirrors
    // the open-start Viterbi; we'll store start_state from winner.
    #pragma unroll
    for (int b = 0; b < B; b++) {
        beam_cost[b]  = (b == 0) ? 0.0f : FLT_MAX;
        beam_state[b] = 0;
    }

    // Candidate buffer: B × 2^K = 128 for K=3
    constexpr int Cmax = B * (1 << 3);  // 128 for K=3
    float    cand_cost [Cmax];
    uint32_t cand_state[Cmax];
    uint8_t  cand_parent[Cmax];
    uint8_t  cand_edge  [Cmax];

    for (int step = 0; step < N; step++) {
        const float xi = x_row[step] * inv_norm;

        int ncand = 0;
        #pragma unroll
        for (int b = 0; b < B; b++) {
            const float pc = beam_cost[b];
            if (pc >= FLT_MAX * 0.5f) continue;
            const uint32_t ps = beam_state[b];
            #pragma unroll
            for (uint32_t e = 0; e <= Kmask; e++) {
                const uint32_t ns   = ((ps >> K) | (e << kshift)) & Lmask;
                const float    code = vtq_trellis_table_storage[ns] * cb_scale;
                const float    diff = xi - code;
                const float    d2   = diff * diff;
                cand_cost  [ncand] = pc + d2;
                cand_state [ncand] = ns;
                cand_parent[ncand] = (uint8_t)b;
                cand_edge  [ncand] = (uint8_t)e;
                ncand++;
            }
        }

        // Top-B selection via simple partial-sort (B small).
        // For each slot 0..B-1, find min in cand[slot..ncand-1] and swap.
        #pragma unroll
        for (int i = 0; i < B && i < ncand; i++) {
            int best = i;
            for (int j = i + 1; j < ncand; j++) {
                if (cand_cost[j] < cand_cost[best]) best = j;
            }
            if (best != i) {
                float    tc = cand_cost [i]; cand_cost [i] = cand_cost [best]; cand_cost [best] = tc;
                uint32_t ts = cand_state[i]; cand_state[i] = cand_state[best]; cand_state[best] = ts;
                uint8_t  tp = cand_parent[i]; cand_parent[i] = cand_parent[best]; cand_parent[best] = tp;
                uint8_t  te = cand_edge  [i]; cand_edge  [i] = cand_edge  [best]; cand_edge  [best] = te;
            }
            beam_cost  [i] = cand_cost  [i];
            beam_state [i] = cand_state [i];
            parent[step][i] = cand_parent[i];
            edge  [step][i] = cand_edge  [i];
        }
        // Pad unused beams with infinity
        for (int i = ncand; i < B; i++) beam_cost[i] = FLT_MAX;
    }

    // Backtrack from best beam at step N-1
    int winner = 0;
    float best_c = beam_cost[0];
    #pragma unroll
    for (int b = 1; b < B; b++) {
        if (beam_cost[b] < best_c) { best_c = beam_cost[b]; winner = b; }
    }

    // Collect edges from N-1 down to 0
    uint8_t path_e[N];
    int cur = winner;
    for (int step = N - 1; step >= 0; step--) {
        path_e[step] = edge[step][cur];
        cur          = parent[step][cur];
    }

    dst_block->start_state = 0;  // beam start was state 0
    uint8_t * qs = dst_block->qs;
    const int qs_bytes = (N * K + 7) / 8;
    for (int b = 0; b < qs_bytes; b++) qs[b] = 0;

    // Re-walk to compute states for qs emit + recon_sq for d.
    uint32_t state = 0;
    float recon_sq = 0.0f;
    for (int step = 0; step < N; step++) {
        const uint32_t e = path_e[step];
        state = ((state >> K) | (e << kshift)) & Lmask;

        const int bo = step * K, byte = bo >> 3, shift = bo & 7;
        qs[byte] |= (uint8_t)((e << shift) & 0xFFu);
        if (shift + K > 8) {
            qs[byte + 1] |= (uint8_t)((e >> (8 - shift)) & 0xFFu);
            if (shift + K > 16) {
                qs[byte + 2] |= (uint8_t)((e >> (16 - shift)) & 0xFFu);
            }
        }
        const float code = vtq_trellis_table_storage[state] * cb_scale;
        recon_sq += code * code;
    }

    const float recon_norm = sqrtf(recon_sq);
    const float d_out      = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
    dst_block->d = __float2half(d_out);
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

    // Fast path: for single-token writes (tg decode), use greedy encoder.
    // Viterbi's O(N×2^L) state space is severe overkill for 1 token per
    // layer and dominates wall clock in autoregressive decode.
    // Override via GGML_VTQ_FORCE_VITERBI=1 for quality A/B testing.
    static int force_viterbi = -1;
    if (force_viterbi == -1) {
        const char * env = getenv("GGML_VTQ_FORCE_VITERBI");
        force_viterbi = (env && atoi(env) > 0) ? 1 : 0;
    }
    const bool use_greedy = (ne11 == 1) && !force_viterbi;

    if (use_greedy) {
        dim3 grid((int)ne_total, 1, 1);
        dim3 block(32, 1, 1);  // 1 warp; only thread 0 works, but warp launches fastest
        k_vtq_greedy_encode_set_rows<block_t, idx_t, K><<<grid, block, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        return;
    }

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
