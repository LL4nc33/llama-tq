#pragma once

// ============================================================
// Flash-Attention TurboQuant helpers — extracted from fattn-common.cuh.
//
// Background:
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
// KTQ helpers, one template per role; the bit width follows from the block type.
//
// A KTQ block holds codebook indices in the Hadamard domain, the shared RHT
// sign bits sb[] and the scale d. Reading K for the KQ dot product needs no
// inverse transform: Q is rotated once per query and dotted against the
// codebook values directly. Reading KTQ as V (or outside FA) needs the full
// inverse: codebook -> serial FWHT -> sign flip -> scale.
// ============================================================

// tq_code_index / ktq_codebook live in turboquant.cuh

// Full dequant of one KTQ block into buf (one thread, serial FWHT).
template <typename block_t>
static __device__ __forceinline__ void ktq_dequant_block(const block_t & x, float * __restrict__ buf) {
    constexpr int bits = ktq_bits<block_t>::value;
    const float norm = (float) x.d;
    if (norm < 1e-30f) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) buf[j] = 0.0f;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        buf[j] = ktq_codebook<bits>()[tq_code_index<bits>(x.qs, j)] * PQ_CUDA_CB_SCALE;
    }
    ktq_cuda_fwht_32_serial(buf);
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int sb = (x.sb[j / 8] >> (j % 8)) & 1;
        buf[j] *= (2.0f * sb - 1.0f) * norm;
    }
}

// K·Q in the Hadamard domain. For an RHT-quantized block K = D_s · H_n · c, so
// K · Q = c · (H_n · (D_s · Q)): Q is rotated once per query (all blocks share one
// sign pattern) and every lane dots its own codebook value. Each lane holds
// Q[bi·32 + lane] in Q_v[bi]. The result is per lane; the caller reduces it.
// The kernels run KTQ K with whole warps (see nthreads_KQ in fattn-vec.cuh).
template <typename block_t, int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    static_assert(nthreads == WARP_SIZE, "KTQ K needs one lane per block element");
    constexpr int bits = ktq_bits<block_t>::value;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    const block_t * K_tq  = (const block_t *) K_c;
    const float   * Q_f32 = (const float *) Q_v;
    const int lane = threadIdx.x;

    float accum = 0.0f;
    #pragma unroll
    for (int bi = 0; bi < D / QK_KTQ; ++bi) {
        const int idx = tq_code_index<bits>(K_tq[bi].qs, lane);
        accum += ktq_codebook<bits>()[idx] * PQ_CUDA_CB_SCALE * Q_f32[bi] * (float) K_tq[bi].d;
    }
    return accum;
}

// V-dequant for KTQ types, used inside the FA P·V loop.
//
// __noinline__ on purpose: each call materializes a 32-float buffer and runs a
// serial FWHT over it. Inlining into the FA kernel adds ~32 live floats to a
// register-tight loop and spills (measured ~15-20 % slower decode on sm_75/sm_89).
template <typename block_t, typename T, int ne>
static __device__ __noinline__ void dequantize_V_ktq(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x  = (const block_t *) vx;
    const int64_t   ib = i0 / QK_KTQ;
    const int       il = (int)(i0 % QK_KTQ);

    float buf[32];
    ktq_dequant_block(x[ib], buf);

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
    // Scale-folded decode: pre-scaled codebook removes the per-element
    // PQ_CUDA_CB_SCALE multiply (one fp32 FMUL/element saved).
    dequantize_V_vtq<block_vtq2_1, T, ne, vtq_decode_2bit_scaled>(vx, dst, i0);
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
// Trellis V types (vtq*_2, vtq*_2s, vtq*_3, vtq3_v8) in FA-vec.
// vtq_state_at (trellis.cuh) gives the state of each sample directly from the bitstream.

// The trellis V types: block, code bits K per element, state bits L, and the outliers per block (fp16 values that
// replace the decoded ones at the listed positions: vtq*_3, vtq3_v8). K = 0: not a trellis type.
template <ggml_type type> struct vtq_trellis { static constexpr int K = 0, L = 0, n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ2_2>  { using block = block_vtq2_2;  static constexpr int K = 2, L = GGML_TRELLIS_L,   n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ3_2>  { using block = block_vtq3_2;  static constexpr int K = 3, L = GGML_TRELLIS_L,   n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ4_2>  { using block = block_vtq4_2;  static constexpr int K = 4, L = GGML_TRELLIS_L,   n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ2_2S> { using block = block_vtq2_2s; static constexpr int K = 2, L = GGML_TRELLIS_L_S, n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ3_2S> { using block = block_vtq3_2s; static constexpr int K = 3, L = GGML_TRELLIS_L_S, n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ4_2S> { using block = block_vtq4_2s; static constexpr int K = 4, L = GGML_TRELLIS_L_S, n_out = 0; };
template <> struct vtq_trellis<GGML_TYPE_VTQ2_3>  { using block = block_vtq2_3;  static constexpr int K = 2, L = GGML_TRELLIS_L,   n_out = VTQ_OUTLIER_K; };
template <> struct vtq_trellis<GGML_TYPE_VTQ3_3>  { using block = block_vtq3_3;  static constexpr int K = 3, L = GGML_TRELLIS_L,   n_out = VTQ_OUTLIER_K; };
template <> struct vtq_trellis<GGML_TYPE_VTQ4_3>  { using block = block_vtq4_3;  static constexpr int K = 4, L = GGML_TRELLIS_L,   n_out = VTQ_OUTLIER_K; };
template <> struct vtq_trellis<GGML_TYPE_VTQ3_V8> { using block = block_vtq3_v8; static constexpr int K = 3, L = GGML_TRELLIS_L,   n_out = VTQ_OUTLIER_K_V8; };

// Factor of the table values of a trellis block with scale d: element i is table[state_i] * vtq_trellis_ds(d). Every
// flash attention reader of the trellis types computes it here, so that they decode the same values.
static __device__ __forceinline__ float vtq_trellis_ds(const float d) {
    return rsqrtf((float) QK_VTQ_TRELLIS) * d;
}

template <typename block_t, int K, int L, typename T, int ne>
static __device__ __forceinline__ void dequantize_V_vtq_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ_TRELLIS;
    const int     il = (int)(i0 % QK_VTQ_TRELLIS);
    const float   d  = (float) x[ib].d;
    const uint16_t s0 = x[ib].start_state;
    const uint8_t * qs = x[ib].qs;

    const float ds = vtq_trellis_ds(d);

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
        const uint32_t state = vtq_state_at<K, L>(s0, qs, il + l + 1);
        const float val = vtq_trellis_table_storage[state] * ds;
        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(val);
        } else {
            ((float *) dst)[l] = val;
        }
    }
}

// VTQ_3 family — same trellis backbone as VTQ_2 plus OUTLIER_K fp16
// outlier samples per block. After trellis decode, positions listed in
// outlier_pos[] are overwritten with outlier_val[] (Phase 3 Step 4b).
//
// OUTLIER_K is the outlier count: VTQ_OUTLIER_K=4 for vtq{2,3,4}_3, VTQ_OUTLIER_K_V8=2 for VTQ3_V8 (vtq_trellis).
template <typename block_t, int K, int L, typename T, int ne, int OUTLIER_K>
static __device__ __forceinline__ void dequantize_V_vtq_3(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_t * x = (const block_t *) vx;
    const int64_t ib = i0 / QK_VTQ_TRELLIS;
    const int     il = (int)(i0 % QK_VTQ_TRELLIS);
    const float   d  = (float) x[ib].d;
    const uint16_t s0 = x[ib].start_state;
    const uint8_t * qs = x[ib].qs;

    const float ds = vtq_trellis_ds(d);

    // Load outlier sidecar once into registers for cheap per-sample compare.
    // Static array of size OUTLIER_K — compiler hoists into registers.
    int   op[OUTLIER_K];
    float ov[OUTLIER_K];
    #pragma unroll
    for (int k = 0; k < OUTLIER_K; ++k) {
        op[k] = (int) x[ib].outlier_pos[k];
        ov[k] = __half2float(((const half *) x[ib].outlier_val)[k]);
    }

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
        const uint32_t state = vtq_state_at<K, L>(s0, qs, pos + 1);
        float val = vtq_trellis_table_storage[state] * ds;
        // Outlier patch — at most one of the OUTLIER_K positions matches.
        #pragma unroll
        for (int k = 0; k < OUTLIER_K; ++k) {
            if (pos == op[k]) val = ov[k];
        }
        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(val);
        } else {
            ((float *) dst)[l] = val;
        }
    }
}

// ============================================================
// Trellis K types (ktq*_3s): the vtq*_3 block with 12 state bits, the remainder after the outliers rotated with the KTQ
// signs and a 128-point FWHT before coding (see ggml-quants.c). Q·K needs no inverse transform: with Q rotated the same
// way, q·(R^T t) = q_rot·t over the decoded samples t. The CPU dequant then overwrites the inverse-rotated value x_p at
// each outlier position p with the stored value, so
//   Q·K = q_rot·t - Σ_p q_p x_p + Σ_p q_p val_p,   x_p = sign(p)/sqrt(128) Σ_m H(p, m) t_m   (H(p, m) = ±1)
// and the readers add the outlier terms per block from the decoded samples a lane holds.
// ============================================================

template <ggml_type type> struct ktq_trellis { static constexpr int K = 0, L = 0, n_out = 0; }; // K = 0: not a trellis K type
template <> struct ktq_trellis<GGML_TYPE_KTQ2_3S> { using block = block_ktq2_3s; static constexpr int K = 2, L = GGML_TRELLIS_L_S, n_out = VTQ_OUTLIER_K; };
template <> struct ktq_trellis<GGML_TYPE_KTQ3_3S> { using block = block_ktq3_3s; static constexpr int K = 3, L = GGML_TRELLIS_L_S, n_out = VTQ_OUTLIER_K; };
template <> struct ktq_trellis<GGML_TYPE_KTQ4_3S> { using block = block_ktq4_3s; static constexpr int K = 4, L = GGML_TRELLIS_L_S, n_out = VTQ_OUTLIER_K; };
static_assert(VTQ_OUTLIER_K == 4, "the readers load the outlier positions as one 32-bit word");

// Q of one 128-value block, lane t holding q[c] = element 32*c + t (c < 4), rotated in place like the K blocks: the
// KTQ signs, the warp FWHT over the low 5 bits of the element index, two butterfly stages across c for the high 2.
static __device__ __forceinline__ void ktq_trellis_rotate_q(float * q) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        q[c] = ktq_cuda_fwht_warp_sums(ktq_trellis_sign_pos(32*c + threadIdx.x) ? q[c] : -q[c]);
    }
    const float a0 = q[0] + q[1];
    const float a1 = q[0] - q[1];
    const float a2 = q[2] + q[3];
    const float a3 = q[2] - q[3];
    q[0] = (a0 + a2) * KTQ_TRELLIS_FWHT_SCALE;
    q[1] = (a1 + a3) * KTQ_TRELLIS_FWHT_SCALE;
    q[2] = (a0 - a2) * KTQ_TRELLIS_FWHT_SCALE;
    q[3] = (a1 - a3) * KTQ_TRELLIS_FWHT_SCALE;
}

// The lane's share of Σ_m H(p_k, m) t_m for the 4 outlier positions p_k (the bytes of pos4): t[e] is the decoded
// sample m_e = (lane << LSH) | (e << ESH) of the block (e < 4), so the sign of H(p, m_e) is the parity of p & lane << LSH
// flipped by the bits of p at ESH (e odd) and ESH + 1 (e >= 2).
template <int LSH, int ESH>
static __device__ __forceinline__ void ktq_trellis_outlier_sums(const uint32_t pos4, const float * t, float * sums) {
#pragma unroll
    for (int k = 0; k < VTQ_OUTLIER_K; ++k) {
        const uint32_t p  = (pos4 >> (8*k)) & 0xFF;
        const uint32_t h  = __popc(p & (threadIdx.x << LSH)) & 1;
        const uint32_t b0 = (p >> ESH) & 1;
        const uint32_t b1 = (p >> (ESH + 1)) & 1;
        sums[k] = (h ? -t[0] : t[0]) + ((h ^ b0) ? -t[1] : t[1]) + ((h ^ b1) ? -t[2] : t[2]) + ((h ^ b0 ^ b1) ? -t[3] : t[3]);
    }
}

// K·Q of a trellis K row in the VEC kernel (whole warps): lane t decodes the samples 32*c + t of every block (c < 4).
// Q_v: the rotated query, lane t holding element 32*bi + t in Q_v[bi]. Q_ds_v: per element of the row the query and
// the query times sign/sqrt(128) (float2, shared memory). The result is per lane; the caller reduces it.
template <ggml_type type_K, int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_ktq_trellis(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    static_assert(nthreads == WARP_SIZE, "trellis K needs whole warps");
    static_assert(D % QK_VTQ_TRELLIS == 0, "head size must be a multiple of the trellis block");
    GGML_UNUSED(Q_q8);
    using tr      = ktq_trellis<type_K>;
    using block_t = typename tr::block;
    const block_t * K_b   = (const block_t *) K_c;
    const float   * Q_rot = (const float *) Q_v;
    const float2  * Q_o   = (const float2 *) Q_ds_v;
    const int lane = threadIdx.x;

    float accum = 0.0f;
#pragma unroll
    for (int ib = 0; ib < D/QK_VTQ_TRELLIS; ++ib) {
        const block_t & b  = K_b[ib];
        const float     ds = vtq_trellis_ds(__half2float(b.d));
        float t[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            t[c] = vtq_trellis_table_storage[vtq_state_at<tr::K, tr::L>(b.start_state, b.qs, 32*c + lane + 1)] * ds;
            accum += t[c] * Q_rot[4*ib + c];
        }
        uint32_t pos4;
        half     val[VTQ_OUTLIER_K];
        ggml_cuda_memcpy_1<4>(&pos4, b.outlier_pos);
        ggml_cuda_memcpy_1<2*VTQ_OUTLIER_K, 4>(val, b.outlier_val);
        float sums[VTQ_OUTLIER_K];
        ktq_trellis_outlier_sums<0, 5>(pos4, t, sums);
#pragma unroll
        for (int k = 0; k < VTQ_OUTLIER_K; ++k) {
            const float2 q = Q_o[QK_VTQ_TRELLIS*ib + ((pos4 >> (8*k)) & 0xFF)];
            accum -= q.y * sums[k];
            if (lane == 0) {
                accum += q.x * __half2float(val[k]);
            }
        }
    }
    return accum;
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
        return vec_dot_fattn_vec_KQ_ktq<block_ktq1_1, D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ2_1) {
        return vec_dot_fattn_vec_KQ_ktq<block_ktq2_1, D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ3_1) {
        return vec_dot_fattn_vec_KQ_ktq<block_ktq3_1, D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_KTQ4_1) {
        return vec_dot_fattn_vec_KQ_ktq<block_ktq4_1, D, nthreads>;
    } else if constexpr (ktq_trellis<type_K>::K > 0) {
        return vec_dot_fattn_vec_KQ_ktq_trellis<type_K, D, nthreads>;
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
        return dequantize_V_ktq<block_ktq1_1, T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ2_1) {
        return dequantize_V_ktq<block_ktq2_1, T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ3_1) {
        return dequantize_V_ktq<block_ktq3_1, T, ne>;
    } else if constexpr (type_V == GGML_TYPE_KTQ4_1) {
        return dequantize_V_ktq<block_ktq4_1, T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ1_1) {
        return dequantize_V_vtq<block_vtq1_1, T, ne, vtq_decode_1bit>;
    } else if constexpr (type_V == GGML_TYPE_VTQ2_1) {
        return dequantize_V_vtq2_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ3_1) {
        return dequantize_V_vtq3_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_VTQ4_1) {
        return dequantize_V_vtq4_1<T, ne>;
    } else if constexpr (vtq_trellis<type_V>::n_out > 0) {
        using tr = vtq_trellis<type_V>;
        return dequantize_V_vtq_3<typename tr::block, tr::K, tr::L, T, ne, tr::n_out>;
    } else if constexpr (vtq_trellis<type_V>::K > 0) {
        using tr = vtq_trellis<type_V>;
        return dequantize_V_vtq_2<typename tr::block, tr::K, tr::L, T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

