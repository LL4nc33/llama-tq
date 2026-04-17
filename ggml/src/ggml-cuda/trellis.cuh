#pragma once

#include "common.cuh"
#include "ggml-common.h"

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

// Declared extern; filled once by a host-side init that copies the
// LUT from ggml-trellis.c.
extern __device__ float vtq_trellis_table[1 << VTQ_TRELLIS_L];

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
        y[i] = vtq_trellis_table[state] * ds;
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

// Host-side init: copy LUT from ggml-trellis.c into __device__ memory.
// Called once per CUDA context.
void ggml_cuda_init_trellis_table(void);
