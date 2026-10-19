// Trellis v2 CUDA encoder (full Viterbi, no pruning): k_vtq_encode_trellis_smem, one launch per SET_ROWS, or
// k_vtq_encode_trellis_pair for the K and the V write of a decoded token together.
//
// 1 trellis block = N = QK_VTQ_TRELLIS = 128 samples, S = 2^L states (L = 12 for vtq*_2s, L = 16 for vtq*_2, vtq*_3 and
// vtq3_v8). The DP runs over butterflies (see the kernel), NG = S / 2^K values per step. Where they live depends on L:
//   L <= 14: both DP rows in shared memory, the codes of a thread's predecessor states in registers.
//   L = 16:  a DP row is 16-64 KiB, two of them do not fit, and a thread would need 128 codes. A thread keeps the DP
//            values of its own butterflies in registers and passes them to the others through a shared exchange
//            buffer of up to 32 KiB; it keeps the codes of a quarter of its butterflies in registers and reads the
//            rest from the code table (L2) at every step.
// The backtrack (one byte per step and butterfly) goes to the CUDA block's slot of a global workspace (trellis.cu).
// A CUDA block walks over trellis blocks in a grid-stride loop and keeps its slot, so no host synchronization is needed.
// The trellis K types (ktq*_3s, ROT) rotate the block after the outlier pick (KTQ signs, 128-point FWHT) and encode that.

#pragma once

#include "common.cuh"
#include "ggml-common.h"
#include "../ggml-trellis.h"
#include "trellis.cuh"
#include "turboquant.cuh"

#include <cstdint>
#include <cfloat>

// Must match QK_VTQ_TRELLIS (128) — block qs[] is sized for that many samples.
// Was 256 prior to task #143; OOB-wrote 32-64 bytes into next-block headers,
// corrupting all VTQ_2 / VTQ_3 PPL measurements (see blog 2026-04-25).
#define VTQ_ENC_N QK_VTQ_TRELLIS

struct vtq_encode_workspace {
    uint8_t * bt;    // backtrack slots
    int64_t   bytes;
};

// the backtrack pool for the encoder of L state bits on the current device, allocated on first use
const vtq_encode_workspace * vtq_get_encode_workspace(int L);

// Source and destination block of flat trellis block i = ((i03*ne02 + i02)*ne01 + i01)*(ne00/N) + i00/N: the samples
// src0[i03, i02, i01, i00 .. i00+N) go to block i00/N of row src1[i12, i11, i01] of dst[i03, i02], with i11 = i02 % ne11
// and i12 = i03 % ne12.
template <typename block_t, typename idx_t>
static __device__ __forceinline__ void vtq_enc_block_ptrs(
        const int64_t i, const float * __restrict__ src0, const idx_t * __restrict__ src1, block_t * __restrict__ dst,
        const int64_t s01, const int64_t s02, const int64_t s03, const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00, const uint3 ne01, const uint3 ne02, const uint3 ne11_fd, const uint3 ne12_fd,
        const float ** src_block, block_t ** dst_block) {
    const int64_t i_base = i * VTQ_ENC_N;
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

    *src_block = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    *dst_block = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_t) + i00 / VTQ_ENC_N;
}

// One warp: the N_OUT largest |x| (earlier position first among equal |x|, as ggml_trellis_outliers_pick on the CPU) are
// stored in fp16 in ascending position order and zeroed in xn. N_OUT rounds of a warp argmax over (|x|, position).
template <typename block_t, int N_OUT>
static __device__ __forceinline__ void vtq_enc_pick_outliers(float * xn, block_t * dst_block, const int lane) {
    constexpr int N   = VTQ_ENC_N;
    constexpr int PER = N / WARP_SIZE; // values per lane, at positions lane + WARP_SIZE*i
    static_assert(N % WARP_SIZE == 0, "the block splits evenly over the lanes");
    float mag[PER];
    #pragma unroll
    for (int i = 0; i < PER; i++) {
        mag[i] = fabsf(xn[lane + WARP_SIZE*i]);
    }
    int top_pos[N_OUT];
    #pragma unroll
    for (int j = 0; j < N_OUT; j++) {
        // a NaN never wins (no compare with it is true), as on the CPU; -1 = no value left
        float bm = -1.0f;
        int   bp = -1;
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            if (mag[i] > bm) { bm = mag[i]; bp = lane + WARP_SIZE*i; }
        }
        #pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float om = __shfl_xor_sync(0xFFFFFFFF, bm, off, WARP_SIZE);
            const int   op = __shfl_xor_sync(0xFFFFFFFF, bp, off, WARP_SIZE);
            if (om > bm || (om == bm && (unsigned) op < (unsigned) bp)) { bm = om; bp = op; }
        }
        top_pos[j] = bp;
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            if (lane + WARP_SIZE*i == bp) { mag[i] = -1.0f; }
        }
    }
    if (lane != 0) {
        return;
    }
    for (int a = 1; a < N_OUT; a++) {
        const int key = top_pos[a];
        int b = a - 1;
        while (b >= 0 && top_pos[b] > key) { top_pos[b + 1] = top_pos[b]; b--; }
        top_pos[b + 1] = key;
    }
    #pragma unroll
    for (int j = 0; j < N_OUT; j++) {
        const int p = top_pos[j];
        if (p < 0) { // only with NaN input; the CPU leaves pos 0 / val 0 there as well
            dst_block->outlier_pos[j] = 0;
            dst_block->outlier_val[j] = __float2half(0.0f);
            continue;
        }
        dst_block->outlier_pos[j] = (uint8_t) p;
        dst_block->outlier_val[j] = __float2half(xn[p]);
        xn[p] = 0.0f;
    }
}

// The CUDA code is built with -use_fast_math, which makes sqrtf and float division approximate. The block norm, the
// code scale and the block scale therefore use IEEE operations (double sqrt, __fdiv_rn) and sums in double in a fixed
// order, the same as ggml_trellis_encode_group on the CPU: both encoders write identical blocks.

// sum over a warp by an xor butterfly, in double (trellis_warp_order_sum on the CPU)
static __device__ __forceinline__ double vtq_enc_warp_sum(double v) {
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE);
    }
    return v;
}

// One warp: packs the emitted bits of the state path states[0..N] and stores start state and block scale. Each lane
// assembles whole bytes of qs, so the block is written once instead of bit by bit.
template <typename block_t, int K, int L>
static __device__ __forceinline__ void vtq_enc_emit(
        const uint16_t * states, const float * xn, const float norm, const float cb_scale, block_t * dst_block,
        const int lane) {
    constexpr int      N        = VTQ_ENC_N;
    constexpr uint32_t Kmask    = (1u << K) - 1u;
    constexpr int      kshift   = L - K;
    constexpr int      qs_bytes = (N * K + 7) / 8;

    // in double and in the order of ggml_trellis_encode_group (trellis_warp_order_sum), as the scale below
    double recon_sq = 0.0;
    double dot      = 0.0;
    for (int step = lane; step < N; step += WARP_SIZE) {
        const float code = vtq_trellis_table_storage[states[step + 1]] * cb_scale;
        recon_sq += (double) code * (double) code;
        dot      += (double) code * (double) xn[step];
    }
    recon_sq = vtq_enc_warp_sum(recon_sq);
    dot      = vtq_enc_warp_sum(dot);

    // byte b holds the bits [8b, 8b+8) of the stream; step s starts at bit s*K
    for (int b = lane; b < qs_bytes; b += WARP_SIZE) {
        uint32_t byte = 0;
        for (int step = (8*b) / K; step <= (8*b + 7) / K && step < N; step++) {
            const uint32_t bits = ((uint32_t) states[step + 1] >> kshift) & Kmask;
            const int      pos  = step*K - 8*b;
            byte |= pos >= 0 ? bits << pos : bits >> -pos;
        }
        dst_block->qs[b] = (uint8_t) (byte & 0xFFu);
    }

    if (lane == 0) {
        dst_block->start_state = states[0];
        const float recon_norm = (float) sqrt(recon_sq);
#if GGML_TRELLIS_UNBIASED_SCALE
        const float d_out = dot > 1e-30 ? (float) ((double) norm / dot) : ((recon_norm > 1e-30f) ? __fdiv_rn(norm, recon_norm) : norm);
#else
        (void) dot;
        const float d_out = (recon_norm > 1e-30f) ? __fdiv_rn(norm, recon_norm) : norm;
#endif
        dst_block->d = __float2half(d_out);
    }
}

// Viterbi over butterflies.
//
// The shift-register trellis decomposes into butterflies: the 2^K states g<<K | e (e < 2^K) are exactly the
// predecessors of the 2^K states g | b<<(L-K) (b < 2^K), for every g < 2^(L-K). The cost of an edge depends only on
// its target state, so all targets of a butterfly share the same best predecessor. The DP therefore keeps one value per
// butterfly instead of one per state: m_i[g], the cost of the cheapest predecessor of butterfly g at time i. A state t
// costs m_i[t & (NG-1)] + (x_i - code_t)^2 at time i+1, and m_{i+1}[g] is the min of that over the states (g<<K) | e.
// Per step and butterfly: 2^K additions, one min, one stored index e (ties keep the lowest e), one stored value.
#define VTQ_ENC_SMEM_THREADS 512
// up to L = 14, a launch with more trellis blocks than SMs (the conversion of a prompt) uses CUDA blocks of 256 threads,
// two per SM: the barriers of one block overlap the work of the other (RTX 2060, 7168-token prompt, Qwen3-4B: 2 bits
// 6.00 -> 4.63 s, 3 bits 4.58 -> 3.92 s). One block of 256 threads takes longer than one of 512, so a launch that fits
// in one wave keeps 512 (decode).
#define VTQ_ENC_BULK_THREADS 256
#define VTQ_ENC_SMEM_MAX_S   16384 // up to here both DP rows fit in shared memory and a thread keeps all its S/512 codes
#define VTQ_ENC_CODE_REGS    32    // codes a thread keeps in registers, more spill at L = 16
#define VTQ_ENC_XCH_MAX      8192  // L = 16: DP values per exchange round (32 KiB)

// threads of a CUDA block of at most NTM threads (VTQ_ENC_SMEM_THREADS or VTQ_ENC_BULK_THREADS): one per butterfly
template <int K, int L, int NTM>
static constexpr __host__ __device__ int vtq_enc_smem_threads() {
    return (1 << (L - K)) < NTM ? 1 << (L - K) : NTM;
}

// L = 16 backtrack: d steps below a known state at time hi, the walk can only reach the 2^(d*K) butterflies from
// (g << d*K) & (NG-1) on, g the butterfly of the known state, so a chunk loads only these windows of the backtrack rows.
template <int K, int L>
static constexpr __host__ __device__ int vtq_enc_win_bytes(const int d) {
    return d*K < L - K ? 1 << (d*K) : 1 << (L - K);
}

// steps per chunk: as many windows as fit in buf_bytes
template <int K, int L>
static constexpr __host__ __device__ int vtq_enc_win_steps(const int buf_bytes) {
    int d   = 0;
    int sum = 0;
    while (d < VTQ_ENC_N && sum + vtq_enc_win_bytes<K, L>(d) <= buf_bytes) {
        sum += vtq_enc_win_bytes<K, L>(d);
        d++;
    }
    return d;
}

// offset of window d in a chunk of D steps: the deeper (larger) windows come first, so the windows of 16 bytes or more
// stay 16-byte aligned
template <int K, int L>
static constexpr __host__ __device__ int vtq_enc_win_offset(const int d, const int D) {
    int off = 0;
    for (int j = d + 1; j < D; j++) {
        off += vtq_enc_win_bytes<K, L>(j);
    }
    return off;
}

// what one encode needs: sources, destination and strides as for vtq_enc_block_ptrs, and its backtrack slots
template <typename block_t, typename idx_t>
struct vtq_enc_args {
    const float * src0;
    const idx_t * src1;
    block_t     * dst;
    int64_t       ne_total;
    int64_t       s01, s02, s03, s10, s11, s12, s1, s2, s3;
    uint3         ne00, ne01, ne02, ne11, ne12;
    uint8_t     * bt;
};

// the constants of the encoder for K bits per sample, L state bits and blocks of at most NTM threads, and its shared
// memory
template <typename block_t, int K, int L, int NTM>
struct vtq_enc_smem {
    static constexpr int      N   = VTQ_ENC_N;
    static constexpr uint32_t S   = 1u << L;
    static constexpr int      NT  = vtq_enc_smem_threads<K, L, NTM>();
    static constexpr int      NE  = 1 << K;              // predecessors per butterfly
    static constexpr int      NG  = S / NE;              // butterflies
    static constexpr int      GPT = (NG + NT - 1) / NT;  // butterflies per thread

    // L <= 14: two DP rows of NG values in m4, the backtrack chunks in bt4. L = 16: bt4 holds one exchange round of XCH
    // values and, after the DP, the backtrack chunks.
    static constexpr bool SMEM_DP  = S <= VTQ_ENC_SMEM_MAX_S;
    static constexpr int  XCH      = NG < VTQ_ENC_XCH_MAX ? NG : VTQ_ENC_XCH_MAX;
    static constexpr int  BT_CHUNK = !SMEM_DP ? XCH*sizeof(float) : 2*NG*sizeof(float) <= 16384 ? 16384 : 8192; // within 48 KiB
    // the codes of the predecessor states of a thread's first CK butterflies stay in registers
    static constexpr int  CK       = GPT*NE <= VTQ_ENC_CODE_REGS ? GPT : VTQ_ENC_CODE_REGS / NE;

    // L <= 14: butterfly g reads the NE values of segment (g << K) & (NG-1) of the DP row, and the NE butterflies
    // g % NS + j*NS (NS = NG/NE segments, j < NE) read the same one. Slot q = tid + k*NT therefore holds the butterfly
    // g = q/NE + (q%NE)*NS: the readers of a segment sit in one warp and a load is a broadcast. A row stores
    // butterfly x at x + PAD*(x/NS), which spreads the stores of a warp over the banks; the backtrack is stored by slot.
    static constexpr int NS  = NG / NE;
    static constexpr int PAD = !SMEM_DP ? 0 : ((32/NE > 4 ? 32/NE : 4) - NS % 32 + 32) % 32;

    float4        m4[2][SMEM_DP ? (NG + PAD*NE)/4 : 1]; // unused at L = 16
    uint4         bt4[BT_CHUNK / sizeof(uint4)];
    float         xn[N];
    float         red[NT/WARP_SIZE];
    uint32_t      red_u[NT/WARP_SIZE];
    uint16_t      states[N + 1];
    const float * src_block;
    block_t     * dst_block;
    float         norm;
};

// The encodes of trellis blocks cta, cta + n_cta, ... by one CUDA block of NT threads.
template <typename block_t, typename idx_t, int K, int L, int NTM, int N_OUT, bool ROT>
static __device__ __forceinline__ void vtq_encode_trellis_blocks(
        const vtq_enc_args<block_t, idx_t> & a, const int cta, const int n_cta, vtq_enc_smem<block_t, K, L, NTM> & sm) {
    using SM = vtq_enc_smem<block_t, K, L, NTM>;
    constexpr int      N        = SM::N;
    constexpr int      NT       = SM::NT;
    constexpr int      NE       = SM::NE;
    constexpr int      NG       = SM::NG;
    constexpr int      GPT      = SM::GPT;
    constexpr bool     SMEM_DP  = SM::SMEM_DP;
    constexpr int      XCH      = SM::XCH;
    constexpr int      BT_CHUNK = SM::BT_CHUNK;
    constexpr int      CK       = SM::CK;
    constexpr int      NS       = SM::NS;
    constexpr int      PAD      = SM::PAD;
    static_assert(L <= 16, "states holds states in 16 bits");
    static_assert(NE % 4 == 0, "predecessor costs and codes are read as float4");
    static_assert(!ROT || NT >= N, "the rotation takes one thread per sample");
    static_assert(!SMEM_DP || CK == GPT, "with the DP rows in shared memory a thread keeps all its codes");
    static_assert(PAD % 4 == 0, "segments stay float4-aligned");

    auto     & s_m4     = sm.m4;
    uint4    * s_bt4    = sm.bt4;
    float    * s_xn     = sm.xn;
    float    * s_red    = sm.red;
    uint32_t * s_red_u  = sm.red_u;
    uint16_t * s_states = sm.states;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    uint8_t * bt   = a.bt + (int64_t) cta * N * NG;
    const float cb_scale = __fdiv_rn(1.0f, __fsqrt_rn((float) N));

    // butterfly of slot q, row position of butterfly x, slot of butterfly g (identities at L = 16)
    const auto bfly_of = [](const int q) { return SMEM_DP ? q/NE + (q % NE)*NS : q; };
    const auto row_pos = [](const int x) { return x + PAD*(x / NS); };
    const auto slot_of = [](const int g) { return SMEM_DP ? (g % NS)*NE + g / NS : g; };

    // the codes of the predecessor states (g << K) | e of a thread's butterflies, the same at every step
    float code[CK][NE];
    #pragma unroll
    for (int k = 0; k < CK; k++) {
        const int g = bfly_of(tid + k*NT);
        #pragma unroll
        for (int e = 0; e < NE; e++) {
            code[k][e] = g < NG ? __fmul_rn(__ldg(vtq_trellis_table_storage + ((g << K) | e)), cb_scale) : 0.0f;
        }
    }

    // ROT: the KTQ sign of the sample this thread rotates (the same pattern in every block, see kktq_derive_seed)
    const bool rot_pos = !ROT || tid >= N || (ktq_cuda_philox_6r((uint32_t) tid, ktq_cuda_derive_seed(0)) & 1);

    for (int64_t i = cta; i < a.ne_total; i += n_cta) {
        if (tid == 0) {
            vtq_enc_block_ptrs(i, a.src0, a.src1, a.dst, a.s01, a.s02, a.s03, a.s10, a.s11, a.s12, a.s1, a.s2, a.s3,
                a.ne00, a.ne01, a.ne02, a.ne11, a.ne12, &sm.src_block, &sm.dst_block);
        }
        __syncthreads();
        block_t * dst_block = sm.dst_block;

        // 1) load x, outliers, norm
        for (int j = tid; j < N; j += NT) {
            s_xn[j] = sm.src_block[j];
        }
        if constexpr (N_OUT > 0) {
            __syncthreads();
            if (warp == 0) {
                vtq_enc_pick_outliers<block_t, N_OUT>(s_xn, dst_block, lane);
            }
        }
        __syncthreads();
        if constexpr (ROT) {
            // the remainder with the KTQ signs, then the 128-point FWHT and its scale, as kktq_rht_forward_sb
            float r = 0.0f;
            if (tid < N) {
                r = rot_pos ? s_xn[tid] : -s_xn[tid];
            }
            r = ktq_cuda_fwht128_sums(r, s_xn, tid);
            if (tid < N) {
                s_xn[tid] = r * KTQ_TRELLIS_FWHT_SCALE;
            }
            __syncthreads();
        }
        if (warp == 0) {
            double sq = 0.0;
            for (int j = lane; j < N; j += WARP_SIZE) {
                sq += (double) s_xn[j] * (double) s_xn[j];
            }
            sq = vtq_enc_warp_sum(sq);
            if (lane == 0) {
                sm.norm = (float) sqrt(sq);
            }
        }
        __syncthreads();
        const float norm = sm.norm;
        if (norm <= 1e-30f) {
            if (tid == 0) {
                dst_block->start_state = 0;
                dst_block->d           = __float2half(0.0f);
            }
            for (int b = tid; b < (N * K + 7) / 8; b += NT) {
                dst_block->qs[b] = 0;
            }
            __syncthreads();
            continue;
        }
        const float inv_norm = __fdiv_rn(1.0f, norm);
        for (int j = tid; j < N; j += NT) {
            s_xn[j] *= inv_norm;
        }

        // 2) open start: at time 0 every state costs 0, so every butterfly's best predecessor costs 0 (e = 0)
        // 3) forward DP: m_step from m_{step-1} and x_{step-1}; the last pass (step == N) only looks for the cheapest
        // state at time N (lowest state wins ties)
        float    best   = FLT_MAX;
        uint32_t best_s = 0;
        if constexpr (SMEM_DP) {
            float * m0 = (float *) s_m4[0];
            for (int x = tid; x < NG; x += NT) {
                m0[row_pos(x)] = 0.0f;
                bt[x]          = 0;
            }
            __syncthreads();

            for (int step = 1; step <= N; step++) {
                const float * m_cur  = (const float *) s_m4[(step - 1) & 1];
                float       * m_next = (float *) s_m4[step & 1];
                const float xi = s_xn[step - 1];
                #pragma unroll
                for (int k = 0; k < GPT; k++) {
                    const int q = tid + k*NT;
                    if (q >= NG) {
                        break;
                    }
                    const int      g  = bfly_of(q);
                    const float4 * mp = (const float4 *) (m_cur + row_pos((g << K) & (NG - 1)));
                    float   gbest   = FLT_MAX;
                    uint8_t gbest_e = 0;
                    #pragma unroll
                    for (int e4 = 0; e4 < NE/4; e4++) {
                        const float4 pm = mp[e4];
                        const float  pc[4] = { pm.x, pm.y, pm.z, pm.w };
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const float diff = xi - code[k][4*e4 + j];
                            const float c    = pc[j] + diff*diff;
                            if (c < gbest) { gbest = c; gbest_e = 4*e4 + j; }
                        }
                    }
                    if (step < N) {
                        m_next[row_pos(g)] = gbest;
                        bt[step * NG + q]  = gbest_e;
                    } else if (gbest < best || (gbest == best && ((g << K) | gbest_e) < best_s)) {
                        best   = gbest;
                        best_s = (g << K) | gbest_e;
                    }
                }
                if (step < N) {
                    __syncthreads();
                }
            }
        } else {
            // A thread owns the butterflies g = tid + k*NT and keeps m[g] in registers. Per step it writes them to the
            // exchange buffer, in NR rounds of XCH values, and in the round that holds the predecessors of a butterfly
            // computes its new value. With NT << K <= XCH that round depends on k only.
            constexpr int NR = NG / XCH;
            static_assert(NG % NT == 0 && NG % XCH == 0 && XCH % NT == 0, "butterflies split evenly over threads and rounds");
            static_assert(NR == 1 || (NT << K) <= XCH, "the round of a butterfly is the same for all threads");
            float * s_x = (float *) s_bt4;

            float m[GPT];
            #pragma unroll
            for (int k = 0; k < GPT; k++) {
                m[k]           = 0.0f;
                bt[tid + k*NT] = 0;
            }

            for (int step = 1; step <= N; step++) {
                float m_new[GPT];
                #pragma unroll
                for (int r = 0; r < NR; r++) {
                    #pragma unroll
                    for (int k = r*(XCH/NT); k < (r + 1)*(XCH/NT); k++) {
                        s_x[tid + k*NT - r*XCH] = m[k];
                    }
                    __syncthreads();
                    const float xi = s_xn[step - 1];
                    #pragma unroll
                    for (int k = 0; k < GPT; k++) {
                        if ((((k*NT) << K) & (NG - 1)) / XCH != r) {
                            continue;
                        }
                        const int      g  = tid + k*NT;
                        const float4 * mp = (const float4 *) (s_x + ((g << K) & (NG - 1)) - r*XCH);
                        const float4 * cp = (const float4 *) (vtq_trellis_table_storage + (g << K));
                        float   gbest   = FLT_MAX;
                        uint8_t gbest_e = 0;
                        #pragma unroll
                        for (int e4 = 0; e4 < NE/4; e4++) {
                            const float4 pm = mp[e4];
                            const float  pc[4] = { pm.x, pm.y, pm.z, pm.w };
                            float cd[4];
                            if (k < CK) {
                                #pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    cd[j] = code[k < CK ? k : 0][4*e4 + j];
                                }
                            } else {
                                // __fmul_rn: the product must not be fused into the subtraction below, as it is not for
                                // the codes in registers
                                const float4 t = __ldg(cp + e4);
                                cd[0] = __fmul_rn(t.x, cb_scale);
                                cd[1] = __fmul_rn(t.y, cb_scale);
                                cd[2] = __fmul_rn(t.z, cb_scale);
                                cd[3] = __fmul_rn(t.w, cb_scale);
                            }
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float diff = xi - cd[j];
                                const float c    = pc[j] + diff*diff;
                                if (c < gbest) { gbest = c; gbest_e = 4*e4 + j; }
                            }
                        }
                        if (step < N) {
                            m_new[k]          = gbest;
                            bt[step * NG + g] = gbest_e;
                        } else if (gbest < best) {
                            best   = gbest;
                            best_s = (g << K) | gbest_e;
                        }
                    }
                    if (step < N || r < NR - 1) { // after the last pass the barriers of steps 4 and 5 suffice
                        __syncthreads();
                    }
                }
                if (step < N) {
                    #pragma unroll
                    for (int k = 0; k < GPT; k++) {
                        m[k] = m_new[k];
                    }
                }
            }
        }

        // 4) argmin over the states at time N
        #pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float    ob = __shfl_xor_sync(0xFFFFFFFF, best,   off, WARP_SIZE);
            const uint32_t os = __shfl_xor_sync(0xFFFFFFFF, best_s, off, WARP_SIZE);
            if (ob < best || (ob == best && os < best_s)) { best = ob; best_s = os; }
        }
        if (lane == 0) {
            s_red[warp]   = best;
            s_red_u[warp] = best_s;
        }
        __syncthreads();

        // 5) backtrack (thread 0): the predecessor of st is (g << K) | e with g the low L-K bits of st. The whole block
        // loads the backtrack bytes into shared memory a chunk of steps at a time, from the last step down, so the
        // serial walk reads shared instead of global memory.
        uint32_t st = 0;
        if (tid == 0) {
            best   = s_red[0];
            best_s = s_red_u[0];
            for (int w = 1; w < NT/WARP_SIZE; w++) {
                if (s_red[w] < best || (s_red[w] == best && s_red_u[w] < best_s)) { best = s_red[w]; best_s = s_red_u[w]; }
            }
            st = best_s;
            s_states[N] = (uint16_t) st;
        }
        if constexpr (SMEM_DP) {
            // whole rows of the backtrack
            constexpr int ROWS = BT_CHUNK / NG; // steps per chunk
            static_assert(ROWS * NG % sizeof(uint4) == 0, "backtrack chunks are copied as uint4");
            const uint8_t * s_bt = (const uint8_t *) s_bt4;
            for (int hi = N; hi > 0; hi -= ROWS) {
                const int lo = hi > ROWS ? hi - ROWS : 0;
                const uint4 * src = (const uint4 *) (bt + lo * NG);
                __syncthreads(); // the previous chunk has been read
                for (int j = tid; j < (hi - lo) * NG / (int) sizeof(uint4); j += NT) {
                    s_bt4[j] = src[j];
                }
                __syncthreads();
                if (tid == 0) {
                    for (int step = hi - 1; step >= lo; step--) {
                        const uint32_t g = st & (NG - 1);
                        st = (g << K) | s_bt[(step - lo) * NG + slot_of(g)];
                        s_states[step] = (uint16_t) st;
                    }
                }
            }
        } else {
            // only the windows of the rows the walk can reach (vtq_enc_win_bytes): with whole rows a 2-bit block would
            // copy 2 MiB in 64 chunks, with windows 350 KiB in 16
            constexpr int D = vtq_enc_win_steps<K, L>(BT_CHUNK); // steps per chunk
            uint8_t * s_win = (uint8_t *) s_bt4;
            for (int hi = N; hi > 0; hi -= D) {
                const int lo = hi > D ? hi - D : 0;
                __syncthreads(); // s_states[hi] is known, the previous chunk has been read
                const uint32_t g0 = s_states[hi] & (NG - 1);
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    if (hi - 1 - d < lo) {
                        break;
                    }
                    const int       w   = vtq_enc_win_bytes<K, L>(d);
                    const int       off = vtq_enc_win_offset<K, L>(d, D);
                    const uint8_t * src = bt + (hi - 1 - d) * NG + ((g0 << (d*K)) & (NG - 1));
                    if (w >= (int) sizeof(uint4)) {
                        for (int j = tid; j < w / (int) sizeof(uint4); j += NT) {
                            ((uint4 *) (s_win + off))[j] = ((const uint4 *) src)[j];
                        }
                    } else if (tid < w) {
                        s_win[off + tid] = src[tid];
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    #pragma unroll
                    for (int d = 0; d < D; d++) {
                        if (hi - 1 - d < lo) {
                            break;
                        }
                        const uint32_t g = st & (NG - 1);
                        st = (g << K) | s_win[vtq_enc_win_offset<K, L>(d, D) + g - ((g0 << (d*K)) & (NG - 1))];
                        s_states[hi - 1 - d] = (uint16_t) st;
                    }
                }
            }
        }
        __syncthreads();

        // 6) emit (one warp)
        if (warp == 0) {
            vtq_enc_emit<block_t, K, L>(s_states, s_xn, norm, cb_scale, dst_block, lane);
        }
        __syncthreads();
    }
}

template <typename block_t, typename idx_t, int K, int L, int NTM, int N_OUT, bool ROT>
__launch_bounds__(NTM, VTQ_ENC_SMEM_THREADS / NTM)
__global__ void k_vtq_encode_trellis_smem(const vtq_enc_args<block_t, idx_t> a) {
    __shared__ vtq_enc_smem<block_t, K, L, NTM> sm;
    vtq_encode_trellis_blocks<block_t, idx_t, K, L, NTM, N_OUT, ROT>(a, blockIdx.x, gridDim.x, sm);
}

// Two KV writes of 12 state bits in one launch, one CUDA block per trellis block: the first n_cta_a CUDA blocks encode
// a, the others b (ggml_cuda_set_rows_trellis_pair). A decoded token writes a few trellis blocks per cache, and each
// encode takes the time of one block on one SM, so the K and the V write of a layer take that time once instead of twice.
template <typename block_a, int K_A, int N_OUT_A, bool ROT_A, typename block_b, int K_B, int N_OUT_B, bool ROT_B>
__launch_bounds__(VTQ_ENC_SMEM_THREADS, 1)
__global__ void k_vtq_encode_trellis_pair(
        const vtq_enc_args<block_a, int64_t> a, const vtq_enc_args<block_b, int64_t> b, const int n_cta_a) {
    constexpr int L   = GGML_TRELLIS_L_S;
    constexpr int NTM = VTQ_ENC_SMEM_THREADS;
    static_assert(vtq_enc_smem_threads<K_A, L, NTM>() == vtq_enc_smem_threads<K_B, L, NTM>(), "both encodes use the same block size");
    __shared__ union {
        vtq_enc_smem<block_a, K_A, L, NTM> a;
        vtq_enc_smem<block_b, K_B, L, NTM> b;
    } sm;
    if ((int) blockIdx.x < n_cta_a) {
        vtq_encode_trellis_blocks<block_a, int64_t, K_A, L, NTM, N_OUT_A, ROT_A>(a, blockIdx.x, n_cta_a, sm.a);
    } else {
        vtq_encode_trellis_blocks<block_b, int64_t, K_B, L, NTM, N_OUT_B, ROT_B>(b, blockIdx.x - n_cta_a, gridDim.x - n_cta_a, sm.b);
    }
}

// the arguments of one encode; bt: its first backtrack slot
template <typename idx_t, typename block_t>
static vtq_enc_args<block_t, idx_t> vtq_enc_make_args(
        const float * src0_d, const idx_t * src1_d, block_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, uint8_t * bt) {
    vtq_enc_args<block_t, idx_t> a;
    a.src0     = src0_d;
    a.src1     = src1_d;
    a.dst      = dst_d;
    a.ne_total = (ne00 * ne01 * ne02 * ne03) / VTQ_ENC_N;
    a.s01      = nb01 / sizeof(float);
    a.s02      = nb02 / sizeof(float);
    a.s03      = nb03 / sizeof(float);
    a.s10      = nb10 / sizeof(idx_t);
    a.s11      = nb11 / sizeof(idx_t);
    a.s12      = nb12 / sizeof(idx_t);
    a.s1       = nb1;
    a.s2       = nb2;
    a.s3       = nb3;
    a.ne00     = init_fastdiv_values((uint32_t) ne00);
    a.ne01     = init_fastdiv_values((uint32_t) ne01);
    a.ne02     = init_fastdiv_values((uint32_t) ne02);
    a.ne11     = init_fastdiv_values((uint32_t) ne11);
    a.ne12     = init_fastdiv_values((uint32_t) ne12);
    a.bt       = bt;
    return a;
}

// Host-side launcher that mirrors set_rows_cuda_pq signature. L: state bits of the type, ROT: rotate (ktq*_3s).
template <typename idx_t, typename block_t, int K, int L, int N_OUT = 0, bool ROT = false>
static void vtq_cuda_encode_set_rows(
        const float * src0_d, const idx_t * src1_d, block_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream)
{
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_ASSERT(ne00 % VTQ_ENC_N == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / VTQ_ENC_N;
    if (ne_total == 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) return;

    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const vtq_encode_workspace * ws = vtq_get_encode_workspace(L);

    // one launch, each CUDA block keeps one slot (N * S/2^K bytes of backtrack) and walks over trellis blocks
    const int64_t slot_bytes = (int64_t) VTQ_ENC_N * ((1 << L) >> K);
    const int     nblocks    = (int) std::min<int64_t>(ne_total, ws->bytes / slot_bytes);
    const auto    a          = vtq_enc_make_args(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, ne11, ne12,
                                                 nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, ws->bt);
    if constexpr ((1 << L) <= VTQ_ENC_SMEM_MAX_S) {
        if (ne_total > ggml_cuda_info().devices[ggml_cuda_get_device()].nsm) {
            constexpr int NTM = VTQ_ENC_BULK_THREADS;
            k_vtq_encode_trellis_smem<block_t, idx_t, K, L, NTM, N_OUT, ROT>
                <<<nblocks, vtq_enc_smem_threads<K, L, NTM>(), 0, stream>>>(a);
            return;
        }
    }
    constexpr int NTM = VTQ_ENC_SMEM_THREADS;
    k_vtq_encode_trellis_smem<block_t, idx_t, K, L, NTM, N_OUT, ROT><<<nblocks, vtq_enc_smem_threads<K, L, NTM>(), 0, stream>>>(a);
}

// the arguments of the encode of SET_ROWS node t (src0 f32, src1 i64)
template <typename block_t>
static vtq_enc_args<block_t, int64_t> vtq_enc_args_of(const ggml_tensor * t, uint8_t * bt) {
    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];
    return vtq_enc_make_args((const float *) src0->data, (const int64_t *) src1->data, (block_t *) t->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[1], src1->ne[2],
        src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[0], src1->nb[1], src1->nb[2], t->nb[1], t->nb[2], t->nb[3], bt);
}

// the KV writes a and b (SET_ROWS, 12 state bits, one wave of CUDA blocks together) in one launch
template <typename block_a, int K_A, int N_OUT_A, bool ROT_A, typename block_b, int K_B, int N_OUT_B, bool ROT_B>
static void vtq_cuda_encode_set_rows_pair(const ggml_tensor * ta, const ggml_tensor * tb, cudaStream_t stream) {
    constexpr int L = GGML_TRELLIS_L_S;
    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const vtq_encode_workspace * ws = vtq_get_encode_workspace(L);
    const auto a = vtq_enc_args_of<block_a>(ta, ws->bt);
    const auto b = vtq_enc_args_of<block_b>(tb, ws->bt + a.ne_total * VTQ_ENC_N * ((1 << L) >> K_A));
    GGML_ASSERT(a.ne_total * VTQ_ENC_N * ((1 << L) >> K_A) + b.ne_total * VTQ_ENC_N * ((1 << L) >> K_B) <= ws->bytes);
    k_vtq_encode_trellis_pair<block_a, K_A, N_OUT_A, ROT_A, block_b, K_B, N_OUT_B, ROT_B>
        <<<(int) (a.ne_total + b.ne_total), vtq_enc_smem_threads<K_A, L, VTQ_ENC_SMEM_THREADS>(), 0, stream>>>(a, b, (int) a.ne_total);
}
