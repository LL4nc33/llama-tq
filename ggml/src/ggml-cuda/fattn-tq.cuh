#pragma once

// ============================================================
// Flash-Attention TurboQuant helpers — extracted from fattn-common.cuh.
//
// Background (see docs/plans/2026-04-26-fa-tu-bloat-profile.md):
// Including turboquant.cuh (1376 LOC) + trellis.cuh (311 LOC) from
// fattn-common.cuh leaks ~1700 LOC of TQ machinery (codebooks, FWHT
// shuffles, Philox PRNG, trellis decoder LUTs) into every FA TU,
// causing ptxas to allocate +23 to +79 more registers per thread on
// the hot pure-f16 MMA prefill kernels. Result on Qwen3.6-35B-A3B
// (head_dim=128) was a 13.6% pp512 regression vs upstream.
//
// This header isolates the TQ-specific dequant + KQ-dot helpers and
// the constexpr type-dispatchers that pick a function pointer per
// `ggml_type`. fattn-common.cuh now contains zero TQ knowledge; only
// TUs that actually instantiate KTQ/VTQ paths include this header.
//
// Include rules:
//   • fattn-vec.cuh, fattn-vec-vtq2.cuh — they call the dispatchers
//     and have to be able to instantiate every branch, so they pull
//     this header.
//   • fattn-mma-ktq.cuh / fattn-mma-ktq-inline.cuh — TQ-aware MMA
//     paths use codebook constants and ktq_cuda_fwht_warp directly.
//   • Pure-f16 paths (fattn-mma-f16.cuh, fattn-tile.cu, fattn-wmma-f16.cu)
//     do NOT include this — that is the entire point of the split.
// ============================================================

#include "fattn-common.cuh"
#include "turboquant.cuh"
#include "trellis.cuh"   // Phase-2c: VTQ{2,3,4}_2 trellis decoder for FA-vec V-dequant

#include <cstdint>

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

// XQuant Phase 2 — paired dequant for XKTQ2_1 subordinate block.
// Reads quantized codes (qs) and RHT sign bits (sb) from sibling block_ktq2_1
// at the SAME block index ib (dominant layer at l-1), but applies the
// subordinate's own per-block scale (x_sub[ib].d). RHT is layer-independent
// (Philox seed = block_index), so sharing sb is mathematically sound.
//
// Compared to ktq_fattn_dequant_block_ktq2_1, the only difference is the
// `norm` source — codes/sb come from x_dom rather than x. Same warp register
// footprint, same FWHT cost. Used by FA-vec when iSWA pairing maps a
// subordinate K layer to a dominant sibling.
static __device__ __forceinline__ void ktq_fattn_dequant_block_xktq2_1_paired(
        const block_xktq2_1 * __restrict__ x_sub,
        const block_ktq2_1  * __restrict__ x_dom,
        const int64_t ib, float * __restrict__ buf) {
    const float norm = (float)x_sub[ib].d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int idx = (x_dom[ib].qs[j / 4] >> (2 * (j % 4))) & 0x3;
        buf[j] = PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x_dom[ib].sb[j / 8] >> (j % 8)) & 1;
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

// XQuant Phase 3c — paired vec_dot for XKTQ2_1 subordinate.
//
// The subordinate `K_c` block (block_xktq2_1) holds only its own per-block
// scale `d`. The 2-bit codes `qs[]` and RHT sign bits `sb[]` come from the
// sibling dominant block_ktq2_1 at the SAME block index ib (passed via
// `K_dom`). Mathematically identical to vec_dot_fattn_vec_KQ_ktq2_1 except
// the per-block norm reads from the subordinate.
//
// PHASE 3c gate: this template is instantiated and dispatchable, but the
// caller-side wiring of `K_dom` into the FA-vec kernel is Phase 3d. The
// dispatcher in fattn-vec-dispatch-ktq.cu currently aborts before ever
// reaching this code; the kv-cache `xquant_dispatch_ready=false` gate
// stops the abort from firing in any current build.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_xktq2_1_paired(
    const char * __restrict__ K_c,
    const void * __restrict__ Q_v,
    const int  * __restrict__ Q_q8,
    const void * __restrict__ Q_ds_v,
    const char * __restrict__ K_dom_c) {
    const block_xktq2_1 * K_sub = (const block_xktq2_1 *) K_c;
    const block_ktq2_1  * K_dom = (const block_ktq2_1  *) K_dom_c;
    const int lane = threadIdx.x;

    if constexpr (nthreads == WARP_SIZE) {
        // Hadamard-domain dot — same shape as ktq2_1 path; only the per-block
        // norm comes from the subordinate. RHT signs (sb) are layer-shared
        // because Philox seed = block_index, so sourcing sb from K_dom is sound.
        const float * Q_f32 = (const float *) Q_v;
        GGML_UNUSED(Q_q8);
        GGML_UNUSED(Q_ds_v);
        float accum = 0.0f;
        constexpr int nblocks = D / QK_KTQ;

        #pragma unroll
        for (int bi = 0; bi < nblocks; ++bi) {
            const float norm = (float)K_sub[bi].d;     // subordinate's own scale
            const int sb  = (K_dom[bi].sb[lane / 8] >> (lane % 8)) & 1;
            float Q_signed = Q_f32[bi] * (1.0f - 2.0f * sb);
            float Q_rot    = ktq_cuda_fwht_warp(Q_signed);
            const int idx  = (K_dom[bi].qs[lane / 4] >> (2 * (lane % 4))) & 0x3;
            accum += PQ_CUDA_CB_2BIT[idx] * PQ_CUDA_CB_SCALE * Q_rot * norm;
        }
        return accum;  // caller does warp_reduce_sum
    } else {
        // Serial fallback (D == 64). Reuses the verified paired block dequant.
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
            ktq_fattn_dequant_block_xktq2_1_paired(K_sub, K_dom, my_ib, buf);
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

// Compute state(i) directly from the bitstream — O(1) per sample.
//
// Insight (from QTIP decoder, arXiv:2406.11235): the shift register state
// after i updates is just an L-bit sliding window over the concatenated
// stream `[start_state low L bits || qs bits]`. Read 16 bits from
// position i*K — that IS state(i+1) after the post-update LUT lookup.
//
// Equivalence proof sketch:
//   state(1) = (s0 >> K) | (bits(0) << (L-K))
//            = bits of s0[K..L-1] in the low positions, bits(0) in the top K
//   Reading L bits from stream position 1*K = K:
//            = stream[K..K+L-1] = s0[K..L-1] || qs[0..K-1] = same thing ✓
//
// This makes each sample O(1) instead of O(i), eliminating the main
// bottleneck that made VTQ_2 FA-vec TG 26x slower than f16.
template <int K>
static __device__ __forceinline__ uint32_t vtq_state_at(uint16_t s0, const uint8_t * qs, int i) {
    // Bit position in the combined [s0 || qs] stream where the state window
    // for sample i starts. After i shift-updates, the window is bits [i*K..i*K+L-1].
    const int stream_bit = i * K;
    constexpr int L = VTQ_TRELLIS_L;

    if (stream_bit + L <= L) {
        // Trivial case, should not happen for i>=1
        return (uint32_t)s0 & 0xFFFFu;
    }

    if (stream_bit < L) {
        // Window straddles s0/qs boundary.
        // High (stream_bit) bits come from qs low bits;
        // Low (L - stream_bit) bits come from s0 shifted right by stream_bit.
        const int from_ss = L - stream_bit;
        uint32_t lo = ((uint32_t)s0 >> stream_bit) & ((1u << from_ss) - 1u);
        // Read stream_bit bits from the start of qs (low side).
        uint32_t qs_word = (uint32_t)qs[0] | ((uint32_t)qs[1] << 8) | ((uint32_t)qs[2] << 16);
        uint32_t hi = qs_word & ((1u << stream_bit) - 1u);
        return lo | (hi << from_ss);
    }

    // Window fully in qs. Read 16 consecutive bits from qs starting at
    // bit position (stream_bit - L).
    const int qs_bit = stream_bit - L;
    const int byte   = qs_bit >> 3;
    const int shift  = qs_bit & 7;
    uint32_t b0 = qs[byte];
    uint32_t b1 = qs[byte + 1];
    uint32_t b2 = qs[byte + 2];
    uint32_t w  = b0 | (b1 << 8) | (b2 << 16);
    return (w >> shift) & 0xFFFFu;
}

template <typename block_t, int K, typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ_TRELLIS;
    const int     il = (int)(i0 % QK_VTQ_TRELLIS);
    const float   d  = (float) x[ib].d;
    const uint16_t s0 = x[ib].start_state;
    const uint8_t * qs = x[ib].qs;

    constexpr int N = QK_VTQ_TRELLIS;
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

    // Direct O(1) per-sample decode — no shift-register replay.
    #pragma unroll
    for (int l = 0; l < ne; ++l) {
        const uint32_t state = vtq_state_at<K>(s0, qs, il + l + 1);
        const float val = vtq_trellis_table_storage[state] * ds;
        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(val);
        } else {
            ((float *) dst)[l] = val;
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

// VTQ_3 family — same trellis backbone as VTQ_2 plus VTQ_OUTLIER_K=4 fp16
// outlier samples per block. After trellis decode, positions listed in
// outlier_pos[] are overwritten with outlier_val[] (Phase 3 Step 4b).
template <typename block_t, int K, typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq_3(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ_TRELLIS;
    const int     il = (int)(i0 % QK_VTQ_TRELLIS);
    const float   d  = (float) x[ib].d;
    const uint16_t s0 = x[ib].start_state;
    const uint8_t * qs = x[ib].qs;

    constexpr int N = QK_VTQ_TRELLIS;
    const float cb_scale = rsqrtf((float)N);
    const float ds = cb_scale * d;

    // Load outlier sidecar once into registers for cheap per-sample compare.
    const uint8_t op0 = x[ib].outlier_pos[0];
    const uint8_t op1 = x[ib].outlier_pos[1];
    const uint8_t op2 = x[ib].outlier_pos[2];
    const uint8_t op3 = x[ib].outlier_pos[3];
    const half ov0 = ((const half *) x[ib].outlier_val)[0];
    const half ov1 = ((const half *) x[ib].outlier_val)[1];
    const half ov2 = ((const half *) x[ib].outlier_val)[2];
    const half ov3 = ((const half *) x[ib].outlier_val)[3];

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

    #pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int pos = il + l;
        const uint32_t state = vtq_state_at<K>(s0, qs, pos + 1);
        float val = vtq_trellis_table_storage[state] * ds;
        // Outlier patch — at most one of the four positions matches.
        if      (pos == (int)op0) val = __half2float(ov0);
        else if (pos == (int)op1) val = __half2float(ov1);
        else if (pos == (int)op2) val = __half2float(ov2);
        else if (pos == (int)op3) val = __half2float(ov3);
        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(val);
        } else {
            ((float *) dst)[l] = val;
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq2_3(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_3<block_vtq2_3, 2, T, ne>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq3_3(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_3<block_vtq3_3, 3, T, ne>(vx, dst, i0);
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq4_3(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_vtq_3<block_vtq4_3, 4, T, ne>(vx, dst, i0);
}

// ============================================================
// Constexpr type-dispatchers — pick a vec-dot / V-dequant function
// pointer based on the runtime ggml_type. Kept in this header (not
// in fattn-common.cuh) because the TQ branches reference symbols
// that only exist when this header is included.
// ============================================================

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
    } else if constexpr (type_V == GGML_TYPE_VTQ2_3) {
        return dequantize_V_vtq2_3<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ3_3) {
        return dequantize_V_vtq3_3<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ4_3) {
        return dequantize_V_vtq4_3<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

// XQuant Phase 3c — paired vec-dot dispatcher. Currently only XKTQ2_1 is
// defined as a paired subordinate type; future xquant levels (XKTQ3_1,
// XKTQ4_1) would extend this branch list.
template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_paired_t get_vec_dot_KQ_paired() {
    if constexpr (type_K == GGML_TYPE_XKTQ2_1) {
        return vec_dot_fattn_vec_KQ_xktq2_1_paired<D, nthreads>;
    } else {
        static_assert(type_K == -1, "get_vec_dot_KQ_paired: only XKTQ types supported");
        return nullptr;
    }
}
