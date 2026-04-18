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

#define QK_VTQ_TRELLIS 256
#define VTQ_TRELLIS_L  16

// Trellis LUT: per-TU `static __device__` definition.
// Without CUDA relocatable-device-code (RDC), cross-TU `extern __device__`
// arrays don't link — nvcc warning 20044-D shows the extern decl gets
// silently demoted to static anyway. The pragmatic pattern:
//   - every consuming TU has its own 256-KiB copy (memory cheap)
//   - GGML_CUDA_INIT_TRELLIS_TABLE_IMPL() macro initializes the copy
//   - each consumer calls the init before first decode on a device
// Memory cost: ~120 TUs × 256 KiB = 30 MiB/device — acceptable.
// Used as-is in convert.cu + trellis.cu. For FA-vec (Phase-2c), each
// template-instance TU must also call the init before first decode.
static __device__ float vtq_trellis_table_storage[1 << VTQ_TRELLIS_L];

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
template <typename block_t, int K, typename dst_t>
static __global__ void k_dequantize_trellis(const void * __restrict__ vx, dst_t * __restrict__ y,
                                             const int64_t ne, const int64_t nb) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= nb) return;
    const block_t * x = (const block_t *) vx;

    float decoded[QK_VTQ_TRELLIS];
    trellis_decode_block<K>(x[ib].start_state, (float)x[ib].d, x[ib].qs, decoded);

    const int64_t out_base = ib * QK_VTQ_TRELLIS;
    #pragma unroll 8
    for (int i = 0; i < QK_VTQ_TRELLIS; i++) {
        const int64_t out_idx = out_base + i;
        if (out_idx < ne) y[out_idx] = ggml_cuda_cast<dst_t>(decoded[i]);
    }
}

template <typename block_t, int K, typename dst_t>
static void trellis_dequantize_row_cuda(const void * vx, dst_t * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = (ne + QK_VTQ_TRELLIS - 1) / QK_VTQ_TRELLIS;
    const int threads = 128;
    const int blocks = (int)((nb + threads - 1) / threads);
    k_dequantize_trellis<block_t, K, dst_t><<<blocks, threads, 0, stream>>>(vx, y, ne, nb);
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
