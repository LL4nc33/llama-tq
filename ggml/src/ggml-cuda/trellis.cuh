#pragma once

#include "common.cuh"
#include "convert.cuh"   // ggml_cuda_cast
#include "ggml-common.h"
#include "../ggml-trellis.h"  // ggml_trellis_table() host accessor
#include "turboquant.cuh"     // the KTQ rotation of the trellis K types

// Trellis v2 CUDA decoders + dequantize kernels for the trellis types. The decoders take the state bits L as a
// template parameter: GGML_TRELLIS_L for vtq*_2, vtq*_3, vtq3_v8 and GGML_TRELLIS_L_S for vtq*_2s.
// See ggml-trellis.h for the algorithmic description (shift register + LUT).
//
// Every sample is a table lookup at its L-bit state, which is a window of the bitstream (vtq_state_at), so the
// samples of a block decode independently: one thread per sample. The trellis K types (ktq*_3s, ROT) then undo the
// rotation of the block across its threads.
//
// The table (2^GGML_TRELLIS_L floats, 256 KiB) exceeds constant memory (64 KiB) and lives in global memory, where the
// lookups hit L2.

// QK_VTQ_TRELLIS now defined in ggml/src/ggml-common.h (task #143: halved to 128).
// This file previously redefined it to 256, which was a silent stale value.

// Trellis LUT: one table per device in global memory (trellis.cu), reached through a pointer in constant memory
// that every TU holds for itself. A TU binds its pointer before its first launch on a device
// (GGML_CUDA_INIT_TRELLIS_TABLE_IMPL). A single extern __device__ table would need relocatable device code for the
// whole backend, which costs registers and load scheduling in every kernel (the q8_0 flash attention decode kernel
// ran at half speed on Turing). The table has 2^GGML_TRELLIS_L entries; the first 2^L of them are the table for L
// state bits, so all trellis types share it.
static __constant__ const float * vtq_trellis_table_storage;

// the device copy of ggml_trellis_table() for this device, created on first use
const float * ggml_cuda_trellis_table_device(int device);

// Casts the return values to (void) because HIP's hipError_t is [[nodiscard]].
#define GGML_CUDA_INIT_TRELLIS_TABLE_IMPL()                                                    \
    do {                                                                                       \
        int _cur_dev = 0;                                                                      \
        (void) cudaGetDevice(&_cur_dev);                                                       \
        static bool _init_done[GGML_CUDA_MAX_DEVICES] = {false};                               \
        if (_cur_dev >= 0 && _cur_dev < GGML_CUDA_MAX_DEVICES && !_init_done[_cur_dev]) {      \
            const float * _tbl = ggml_cuda_trellis_table_device(_cur_dev);                     \
            (void) cudaMemcpyToSymbol(vtq_trellis_table_storage, &_tbl, sizeof(_tbl));         \
            _init_done[_cur_dev] = true;                                                       \
        }                                                                                      \
    } while (0)

// Compute state(i) directly from the bitstream — O(1) per sample.
//
// Insight (from QTIP decoder, arXiv:2406.11235): the shift register state
// after i updates is just an L-bit sliding window over the concatenated
// stream `[start_state low L bits || qs bits]`. Read L bits from
// position i*K — that IS state(i+1) after the post-update LUT lookup.
//
// Equivalence proof sketch:
//   state(1) = (s0 >> K) | (bits(0) << (L-K))
//            = bits of s0[K..L-1] in the low positions, bits(0) in the top K
//   Reading L bits from stream position 1*K = K:
//            = stream[K..K+L-1] = s0[K..L-1] || qs[0..K-1] = same thing ✓
//
// This makes each sample O(1) instead of O(i), so every thread of a block
// can decode its own samples (flash attention readers, bulk dequant).
template <int K, int L>
static __device__ __forceinline__ uint32_t vtq_state_at(uint16_t s0, const uint8_t * qs, int i) {
    // Bit position in the combined [s0 || qs] stream where the state window
    // for sample i starts. After i shift-updates, the window is bits [i*K..i*K+L-1].
    const int stream_bit = i * K;

    if (stream_bit + L <= L) {
        // Trivial case, should not happen for i>=1
        return (uint32_t)s0 & ((1u << L) - 1u);
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

    // Window fully in qs. Read L consecutive bits from qs starting at
    // bit position (stream_bit - L).
    const int qs_bit = stream_bit - L;
    const int byte   = qs_bit >> 3;
    const int shift  = qs_bit & 7;
    uint32_t b0 = qs[byte];
    uint32_t b1 = qs[byte + 1];
    uint32_t b2 = qs[byte + 2];
    uint32_t w  = b0 | (b1 << 8) | (b2 << 16);
    return (w >> shift) & ((1u << L) - 1u);
}

// The outliers of the vtq*_3 / vtq3_v8 / ktq*_3s blocks: the value of sample i if it is one (a later k wins, as in the
// CPU dequant), else v.
template <typename block_t, int OUTLIER_K>
static __device__ __forceinline__ float trellis_outliers_apply(const block_t & b, const int i, float v) {
    #pragma unroll
    for (int k = 0; k < OUTLIER_K; k++) {
        if ((int) b.outlier_pos[k] == i) {
            v = __half2float(b.outlier_val[k]);
        }
    }
    return v;
}

// Sample i of trellis block b (vtq_state_at), with the outlier overlay of the vtq*_3 / vtq3_v8 types. The decode
// expression is the one of the flash attention readers (fattn-tq.cuh).
template <typename block_t, int K, int L, bool WITH_OUTLIERS, int OUTLIER_K>
static __device__ __forceinline__ float trellis_sample(const block_t & b, const int i) {
    const float d = (float) b.d;
    float v = d == 0.0f ? 0.0f : vtq_trellis_table_storage[vtq_state_at<K, L>(b.start_state, b.qs, i + 1)] * (rsqrtf((float) QK_VTQ_TRELLIS) * d);
    if constexpr (WITH_OUTLIERS) {
        v = trellis_outliers_apply<block_t, OUTLIER_K>(b, i, v);
    }
    return v;
}

// Sample i of a ktq*_3s block, in thread i of the QK_VTQ_TRELLIS threads that call it together: the trellis sample, the
// inverse KTQ rotation (128-point FWHT through buf, scale, sign_pos: whether the KTQ sign of sample i is +1), then the
// outliers, as ktq_trellis_dequantize_block on the CPU.
template <typename block_t, int K>
static __device__ __forceinline__ float ktq_trellis_sample(const block_t & b, const int i, const bool sign_pos, float * buf) {
    float v = trellis_sample<block_t, K, GGML_TRELLIS_L_S, false, VTQ_OUTLIER_K>(b, i);
    v = ktq_cuda_fwht128_sums(v, buf, i) * KTQ_TRELLIS_FWHT_SCALE;
    return trellis_outliers_apply<block_t, VTQ_OUTLIER_K>(b, i, sign_pos ? v : -v);
}

// whether the KTQ sign of sample i of a trellis K block is +1 (all blocks share the pattern, see kktq_derive_seed)
static __device__ __forceinline__ bool ktq_trellis_sign_pos(const int i) {
    return ktq_cuda_philox_6r((uint32_t) i, ktq_cuda_derive_seed(0)) & 1;
}

// Bulk dequant of contiguous blocks: one CUDA block of QK_VTQ_TRELLIS threads per trellis block, one sample per thread.
// ROT: a trellis K type (ktq_trellis_sample, the outliers then come from the block, WITH_OUTLIERS is not used).
template <typename block_t, int K, int L, typename dst_t, bool WITH_OUTLIERS = false, int OUTLIER_K = VTQ_OUTLIER_K,
          bool ROT = false>
static __global__ void k_dequantize_trellis(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne) {
    const int64_t i = (int64_t) blockIdx.x * QK_VTQ_TRELLIS + threadIdx.x;
    const block_t * x = (const block_t *) vx;
    if constexpr (ROT) {
        static_assert(L == GGML_TRELLIS_L_S, "the trellis K types have 12 state bits");
        __shared__ float buf[QK_VTQ_TRELLIS];
        y[i] = ggml_cuda_cast<dst_t>(ktq_trellis_sample<block_t, K>(x[blockIdx.x], threadIdx.x, ktq_trellis_sign_pos(threadIdx.x), buf));
    } else {
        if (i >= ne) return;
        y[i] = ggml_cuda_cast<dst_t>(trellis_sample<block_t, K, L, WITH_OUTLIERS, OUTLIER_K>(x[blockIdx.x], threadIdx.x));
    }
}

template <typename block_t, int K, int L, typename dst_t, bool WITH_OUTLIERS = false, int OUTLIER_K = VTQ_OUTLIER_K,
          bool ROT = false>
static void trellis_dequantize_row_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    GGML_ASSERT(!ROT || ne % QK_VTQ_TRELLIS == 0);
    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const int64_t nb = (ne + QK_VTQ_TRELLIS - 1) / QK_VTQ_TRELLIS;
    k_dequantize_trellis<block_t, K, L, dst_t, WITH_OUTLIERS, OUTLIER_K, ROT><<<(int) nb, QK_VTQ_TRELLIS, 0, stream>>>(vx, y, ne);
}

// Concrete wrappers matching convert.cu dispatcher signature
template <typename dst_t>
static void dequantize_row_vtq2_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq2_2, 2, GGML_TRELLIS_L, dst_t>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_2, 3, GGML_TRELLIS_L, dst_t>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq4_2, 4, GGML_TRELLIS_L, dst_t>(vx, y, ne, stream);
}

// VTQ_2S family: the VTQ_2 layout with GGML_TRELLIS_L_S state bits.
template <typename dst_t>
static void dequantize_row_vtq2_2s_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq2_2s, 2, GGML_TRELLIS_L_S, dst_t>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_2s_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_2s, 3, GGML_TRELLIS_L_S, dst_t>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_2s_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq4_2s, 4, GGML_TRELLIS_L_S, dst_t>(vx, y, ne, stream);
}

// VTQ_3 family: trellis decode + 4-position fp16 outlier overlay.
template <typename dst_t>
static void dequantize_row_vtq2_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq2_3, 2, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_3, 3, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq4_3, 4, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}

// VTQ3_V8 (TurboQuant v8): trellis-3bit + VTQ_OUTLIER_K_V8=2 outliers (3.625 bpw).
template <typename dst_t>
static void dequantize_row_vtq3_v8_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_v8, 3, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true, /*OUTLIER_K=*/VTQ_OUTLIER_K_V8>(vx, y, ne, stream);
}

// KTQ{K}_3S: the vtq{K}_3 layout with 12 state bits, the block rotated with the KTQ signs and a 128-point FWHT.
template <int K, typename block_t, typename dst_t>
static void dequantize_row_ktq_trellis_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_t, K, GGML_TRELLIS_L_S, dst_t, false, VTQ_OUTLIER_K, /*ROT=*/true>(vx, y, ne, stream);
}

// Non-contiguous bulk dequant (convert.cu ggml_get_to_{fp16,bf16,fp32}_nc_cuda; flash attention converts a quantized
// V to f16 with it for batches): one CUDA block per trellis block of a row, one sample per thread.
template <typename block_t, int K, int L, typename dst_t, bool WITH_OUTLIERS = false, int OUTLIER_K = VTQ_OUTLIER_K,
          bool ROT = false>
static __global__ void k_dequantize_trellis_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t ib_in_row = blockIdx.x;
    const int i = threadIdx.x;
    const block_t * x = (const block_t *) vx;
    __shared__ float buf[ROT ? QK_VTQ_TRELLIS : 1];
    const bool sign_pos = ROT && ktq_trellis_sign_pos(i);

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t ib = dm.x*s03 + dm.y*s02 + i01*s01 + ib_in_row;
            float v;
            if constexpr (ROT) {
                v = ktq_trellis_sample<block_t, K>(x[ib], i, sign_pos, buf);
            } else {
                v = trellis_sample<block_t, K, L, WITH_OUTLIERS, OUTLIER_K>(x[ib], i);
            }
            y[(i0203*ne01 + i01)*ne00 + ib_in_row*QK_VTQ_TRELLIS + i] = ggml_cuda_cast<dst_t>(v);
        }
    }
}

template <typename block_t, int K, int L, typename dst_t, bool WITH_OUTLIERS = false, int OUTLIER_K = VTQ_OUTLIER_K,
          bool ROT = false>
static void trellis_dequantize_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ_TRELLIS == 0);
    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const int64_t nb_per_row = ne00 / QK_VTQ_TRELLIS;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    k_dequantize_trellis_nc<block_t, K, L, dst_t, WITH_OUTLIERS, OUTLIER_K, ROT><<<num_blocks, QK_VTQ_TRELLIS, 0, stream>>>(
        vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename dst_t>
static void dequantize_block_vtq2_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq2_2, 2, GGML_TRELLIS_L, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_2, 3, GGML_TRELLIS_L, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq4_2, 4, GGML_TRELLIS_L, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// VTQ_2S family NC variants.
template <typename dst_t>
static void dequantize_block_vtq2_2s_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq2_2s, 2, GGML_TRELLIS_L_S, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_2s_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_2s, 3, GGML_TRELLIS_L_S, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_2s_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq4_2s, 4, GGML_TRELLIS_L_S, dst_t>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// VTQ_3 family NC variants: trellis decode + 4-position fp16 outlier overlay.
template <typename dst_t>
static void dequantize_block_vtq2_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq2_3, 2, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_3, 3, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq4_3, 4, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// VTQ3_V8 (TurboQuant v8): trellis-3bit + VTQ_OUTLIER_K_V8=2 outliers (3.625 bpw).
template <typename dst_t>
static void dequantize_block_vtq3_v8_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_v8, 3, GGML_TRELLIS_L, dst_t, /*WITH_OUTLIERS=*/true, /*OUTLIER_K=*/VTQ_OUTLIER_K_V8>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// KTQ{K}_3S NC variant (see dequantize_row_ktq_trellis_cuda).
template <int K, typename block_t, typename dst_t>
static void dequantize_block_ktq_trellis_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_t, K, GGML_TRELLIS_L_S, dst_t, false, VTQ_OUTLIER_K, /*ROT=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// Get-rows of a trellis K type: one CUDA block of QK_VTQ_TRELLIS threads per trellis block of a selected row.
// blockIdx.x = row index in src1, blockIdx.y = trellis block within the row, blockIdx.z = batch (as k_get_rows_ktq).
template <int K, typename block_t, typename dst_t>
static __global__ void k_get_rows_ktq_trellis(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    __shared__ float buf[QK_VTQ_TRELLIS];
    const int i10 = blockIdx.x;
    const int ib  = blockIdx.y;
    const int i11 = blockIdx.z / ne12;
    const int i12 = blockIdx.z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const block_t * src0_row = (const block_t *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
    const float v = ktq_trellis_sample<block_t, K>(src0_row[ib], threadIdx.x, ktq_trellis_sign_pos(threadIdx.x), buf);
    dst[i10*s1 + i11*s2 + i12*s3 + ib*QK_VTQ_TRELLIS + threadIdx.x] = ggml_cuda_cast<dst_t>(v);
}

template <int K, typename block_t, typename dst_t>
static void get_rows_cuda_ktq_trellis(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ_TRELLIS == 0);
    GGML_ASSERT(ne00 / QK_VTQ_TRELLIS <= UINT16_MAX && ne11*ne12 <= UINT16_MAX);
    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();
    const dim3 grid_dims(ne10, ne00 / QK_VTQ_TRELLIS, ne11*ne12);
    k_get_rows_ktq_trellis<K, block_t><<<grid_dims, QK_VTQ_TRELLIS, 0, stream>>>(
        src0_d, src1_d, dst_d, ne12, nb1/sizeof(dst_t), nb2/sizeof(dst_t), nb3/sizeof(dst_t), nb01, nb02, nb03,
        nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t));
}

// No forward decl — init is done inside convert.cu via the macro.
