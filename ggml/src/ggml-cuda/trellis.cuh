#pragma once

#include "common.cuh"
#include "convert.cuh"   // ggml_cuda_cast
#include "ggml-common.h"
#include "../ggml-trellis.h"  // ggml_trellis_table() host accessor

// Trellis v2 CUDA decoders + dequantize kernels for VTQ{2,3,4}_2.
// See ggml-trellis.h for the algorithmic description (shift register + LUT).
//
// Strategy:
//   Decoder is fundamentally sequential (shift register), so we use
//   1 CUDA thread per block — no intra-block parallelism. But many blocks
//   run in parallel across grids. A 27B model has thousands of V-cache
//   rows active per forward pass, so GPU is saturated.
//
// LUT (65536 × fp32 = 256 KiB) stored in __constant__ memory? No — 256 KiB
// exceeds __constant__ cap (64 KiB). We use __device__ const array in
// global memory, cached into L2. Lookup is 1 random access per sample,
// ~100 cycles uncached, ~30 cached. 256 samples/block × 30 cycles ≈
// 7700 cycles/block = 3µs on a 2 GHz clock — fast enough.

// QK_VTQ_TRELLIS now defined in ggml/src/ggml-common.h (task #143: halved to 128).
// This file previously redefined it to 256, which was a silent stale value.
#define VTQ_TRELLIS_L  16

// Trellis LUT: SINGLE device-global definition in trellis.cu, referenced
// via extern from every TU that includes this header. Requires RDC
// (CUDA_SEPARABLE_COMPILATION=ON) — enabled in ggml/src/ggml-cuda/CMakeLists.txt.
#ifdef VTQ_TRELLIS_TABLE_DEFINE
__device__ float vtq_trellis_table_storage[1 << VTQ_TRELLIS_L];
#else
extern __device__ float vtq_trellis_table_storage[1 << VTQ_TRELLIS_L];
#endif

// Host-side init: called once per CUDA context before any dequant.
// Uses cudaMemcpyToSymbol on the caller's TU symbol.
#define GGML_CUDA_INIT_TRELLIS_TABLE_IMPL()                              \
    do {                                                                 \
        int _cur_dev = 0;                                                \
        cudaGetDevice(&_cur_dev);                                        \
        static bool _init_done[16] = {false};                            \
        if (_cur_dev >= 0 && _cur_dev < 16 && !_init_done[_cur_dev]) {   \
            const float * host_tbl = ggml_trellis_table();               \
            cudaMemcpyToSymbol(vtq_trellis_table_storage, host_tbl,      \
                               sizeof(float) * (1 << VTQ_TRELLIS_L));    \
            _init_done[_cur_dev] = true;                                 \
        }                                                                \
    } while (0)

// Decode one block: sequential shift register, 256 samples.
template <int K>
__device__ __forceinline__
void trellis_decode_block(uint16_t start_state, float d, const uint8_t * qs, float * y) {
    constexpr int N = QK_VTQ_TRELLIS;
    constexpr int L = VTQ_TRELLIS_L;
    constexpr uint32_t Lmask = 0xFFFFu;
    constexpr uint32_t Kmask = (1u << K) - 1u;
    const float cb_scale = rsqrtf((float)N);

    if (d == 0.0f) {
        #pragma unroll 8
        for (int i = 0; i < N; i++) y[i] = 0.0f;
        return;
    }

    uint32_t state = (uint32_t)start_state & Lmask;
    // Pre-compute scale factor once
    const float ds = cb_scale * d;

    for (int i = 0; i < N; i++) {
        // Read K bits at offset i*K (little-endian)
        const int bit_off = i * K;
        const int byte = bit_off >> 3;
        const int shift = bit_off & 7;
        uint32_t b0 = qs[byte];
        uint32_t b1 = qs[byte + 1];
        uint32_t b2 = (shift + K > 16) ? qs[byte + 2] : 0u;
        uint32_t w = b0 | (b1 << 8) | (b2 << 16);
        uint32_t bits = (w >> shift) & Kmask;

        state = ((state >> K) | (bits << (L - K))) & Lmask;
        // Plain load: 256 KiB LUT fits in L2 (~100% hit) but exceeds
        // Turing's 48 KiB/SM RO cache — __ldg here causes a ~20% hit
        // rate + L2 refill stalls. L2 is the right cache for this.
        y[i] = vtq_trellis_table_storage[state] * ds;
    }
}

// Generic bulk dequant kernel. One thread per block.
// WITH_OUTLIERS=true: after Trellis decode, overwrite 4 positions with
// fp16 outlier values stored in block_t::outlier_pos / ::outlier_val
// (VTQ_3 family). Single-thread per block, so no syncwarp needed here.
template <typename block_t, int K, typename dst_t, bool WITH_OUTLIERS = false>
static __global__ void k_dequantize_trellis(const void * __restrict__ vx, dst_t * __restrict__ y,
                                             const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= nb) return;
    const block_t * x = (const block_t *) vx;

    float decoded[QK_VTQ_TRELLIS];
    trellis_decode_block<K>(x[ib].start_state, (float)x[ib].d, x[ib].qs, decoded);

    if constexpr (WITH_OUTLIERS) {
        // Overwrite 4 outlier positions with stored fp16 values.
        #pragma unroll
        for (int k = 0; k < VTQ_OUTLIER_K; k++) {
            const int p = (int)x[ib].outlier_pos[k];
            decoded[p] = __half2float(x[ib].outlier_val[k]);
        }
    }

    const int64_t out_base = ib * QK_VTQ_TRELLIS;
    #pragma unroll 8
    for (int i = 0; i < QK_VTQ_TRELLIS; i++) {
        const int64_t out_idx = out_base + i;
        if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(decoded[i]);
    }
}

template <typename block_t, int K, typename dst_t, bool WITH_OUTLIERS = false>
static void trellis_dequantize_row_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ_TRELLIS - 1) / QK_VTQ_TRELLIS;
    const int threads = 128;
    const int blocks = (int)((nb + threads - 1) / threads);
    k_dequantize_trellis<block_t, K, dst_t, WITH_OUTLIERS><<<blocks, threads, 0, stream>>>(vx, y, ne, nb);
}

// Concrete wrappers matching convert.cu dispatcher signature
template <typename dst_t>
static void dequantize_row_vtq2_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq2_2, 2>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_2, 3>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_2_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq4_2, 4>(vx, y, ne, stream);
}

// VTQ_3 family: trellis decode + 4-position fp16 outlier overlay.
template <typename dst_t>
static void dequantize_row_vtq2_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq2_3, 2, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq3_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq3_3, 3, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}
template <typename dst_t>
static void dequantize_row_vtq4_3_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    trellis_dequantize_row_cuda<block_vtq4_3, 4, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne, stream);
}

// ============================================================
// Non-contiguous (NC) dequant kernel for VTQ_2 — required so that
// convert.cu's ggml_get_to_{fp16,bf16,fp32}_nc_cuda dispatch tables
// return a valid kernel for VTQ_2 types. Without this, FA falls
// back to cuBLAS + CPU dequant path (observed 24x slowdown).
//
// One CUDA block per trellis block (512 samples). Thread 0 decodes
// the entire block into shared memory, then all 128 threads write
// strided output. Shift-register decode is inherently sequential so
// no thread parallelism within a block. That's OK for V-cache
// dequant: thousands of blocks per layer × multiple layers gives
// plenty of grid-level parallelism to saturate the GPU.
template <typename block_t, int K, typename dst_t, bool WITH_OUTLIERS = false>
static __global__ void k_dequantize_trellis_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t ib_in_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t nb_per_row = ne00 / QK_VTQ_TRELLIS;
    if (ib_in_row >= nb_per_row) return;

    __shared__ float decoded[QK_VTQ_TRELLIS];

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t ibx0 = dm.x*s03 + dm.y*s02 + i01*s01;
            const block_t * x = (const block_t *) vx;
            const int64_t ib = ibx0 + ib_in_row;

            if (tid == 0) {
                trellis_decode_block<K>(x[ib].start_state, (float)x[ib].d, x[ib].qs, decoded);
                if constexpr (WITH_OUTLIERS) {
                    // Overwrite 4 outlier positions with stored fp16 values.
                    #pragma unroll
                    for (int k = 0; k < VTQ_OUTLIER_K; k++) {
                        const int p = (int)x[ib].outlier_pos[k];
                        decoded[p] = __half2float(x[ib].outlier_val[k]);
                    }
                }
            }
            __syncthreads();

            const int64_t out_base = (i0203*ne01 + i01)*ne00 + ib_in_row * QK_VTQ_TRELLIS;
            for (int i = tid; i < QK_VTQ_TRELLIS; i += blockDim.x) {
                if (ib_in_row * QK_VTQ_TRELLIS + i < ne00) {
                    y[out_base + i] = ggml_cuda_cast<dst_t>(decoded[i]);
                }
            }
            __syncthreads();
        }
    }
}

template <typename block_t, int K, typename dst_t, bool WITH_OUTLIERS = false>
static void trellis_dequantize_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_VTQ_TRELLIS == 0);
    const int64_t nb_per_row = ne00 / QK_VTQ_TRELLIS;
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((int)nb_per_row, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    k_dequantize_trellis_nc<block_t, K, dst_t, WITH_OUTLIERS><<<num_blocks, 128, 0, stream>>>(
        vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename dst_t>
static void dequantize_block_vtq2_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq2_2, 2>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_2, 3>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_2_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq4_2, 4>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// VTQ_3 family NC variants: trellis decode + 4-position fp16 outlier overlay.
template <typename dst_t>
static void dequantize_block_vtq2_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq2_3, 2, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq3_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq3_3, 3, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}
template <typename dst_t>
static void dequantize_block_vtq4_3_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    trellis_dequantize_nc_cuda<block_vtq4_3, 4, dst_t, /*WITH_OUTLIERS=*/true>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

// ============================================================
// Per-element decoder variant for FA-vec V-dequant path (Phase-2c WIP).
//
// The FA-vec kernel calls `dequantize_V_t(vx, dst, i0)` per-row with
// `ne` consecutive elements. VTQ_2 is a shift-register; random access
// to element `i0` requires replaying the shift register from
// `start_state` up to index `i0+ne-1`. Cost per call: O(i0+ne).
//
// For typical D=128 attention heads, `ne` is 2-4 and `i0` is thread-
// indexed across [0, D). Average replay depth is D/2 ≈ 64 iterations
// per thread per call. Compared to the CPU-FA fallback this is still
// a massive win, but see trellis.cuh Strategy A note: the optimal
// path is a warp-collaborative shmem block cache, which requires
// invasive changes to fattn-vec.cuh (out of scope for Phase-2c).
//
// NOTE: this decoder replays from start_state on every call and is
// O(N^2) when called N/ne times for a full block. Acceptable for
// D <= 256 (Turing) but inefficient for larger heads. Ship after
// correctness validation; optimize to shmem cache in Phase-2d.
template <int K>
__device__ __forceinline__
float trellis_decode_element(uint16_t start_state, float d, const uint8_t * qs, int j) {
    constexpr int N = QK_VTQ_TRELLIS;
    constexpr int L = VTQ_TRELLIS_L;
    constexpr uint32_t Lmask = 0xFFFFu;
    constexpr uint32_t Kmask = (1u << K) - 1u;
    const float cb_scale = rsqrtf((float)N);

    if (d == 0.0f) return 0.0f;

    uint32_t state = (uint32_t)start_state & Lmask;
    // Replay shift register for indices 0..j (state after j-th bit extract).
    #pragma unroll 1
    for (int i = 0; i <= j; i++) {
        const int bit_off = i * K;
        const int byte   = bit_off >> 3;
        const int shift  = bit_off & 7;
        uint32_t b0 = qs[byte];
        uint32_t b1 = qs[byte + 1];
        uint32_t b2 = (shift + K > 16) ? qs[byte + 2] : 0u;
        uint32_t w  = b0 | (b1 << 8) | (b2 << 16);
        uint32_t bits = (w >> shift) & Kmask;
        state = ((state >> K) | (bits << (L - K))) & Lmask;
    }
    return vtq_trellis_table_storage[state] * (cb_scale * d);
}

// No forward decl — init is done inside convert.cu via the macro.
