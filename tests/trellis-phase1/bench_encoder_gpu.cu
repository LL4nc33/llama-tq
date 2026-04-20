// Standalone GPU encoder benchmark.
// Isolates the Viterbi + Greedy encoder kernels from llama.cpp to make
// dev-iteration fast: this TU builds in ~30s vs ~60min for full repo.
//
// Build:
//   nvcc -std=c++17 -O3 -arch=sm_75 -use_fast_math -extended-lambda \
//        -I../../ggml/src -I../../ggml/include -I../../ggml/src/ggml-cuda \
//        -o bench_encoder_gpu bench_encoder_gpu.cu
//
// Run:
//   ./bench_encoder_gpu              # default: 10000 iters, ne11=1
//   ./bench_encoder_gpu 1000 8       # 1000 iters, ne11=8
//   GGML_VTQ_FORCE_VITERBI=1 ./bench_encoder_gpu  # force Viterbi

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Pull in the types we need — minimal subset of ggml-common.h
#define QK_VTQ_TRELLIS 256

typedef __half ggml_half;

typedef struct {
    ggml_half d;
    uint16_t  start_state;
    uint8_t   qs[QK_VTQ_TRELLIS * 3 / 8];  // K=3 → 96 bytes
} block_vtq3_2;
static_assert(sizeof(block_vtq3_2) == 100, "block_vtq3_2 size");

// Minimal decls matching trellis-encode.cuh needs
#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// --------------------------------------------------------------
// Minimal subset: LUT + greedy encoder + Viterbi launcher stubs.
// We reimplement locally to avoid pulling in half of ggml-cuda.
// --------------------------------------------------------------

#define VTQ_TRELLIS_L 16
__device__ float g_lut[1 << VTQ_TRELLIS_L];

// Inverse Gaussian CDF LUT builder (matches trellis_code.c)
static void build_lut_host(float * host) {
    const int N = 1 << VTQ_TRELLIS_L;
    // Simple inverse-normal CDF via probit approximation (Beasley-Springer-Moro)
    // Not the real CDF from trellis_code.c, but good enough for timing benchmark.
    for (int i = 0; i < N; i++) {
        double u = (i + 0.5) / (double)N;
        // Beasley-Springer inverse normal CDF approx
        double t = (u < 0.5) ? sqrt(-2.0 * log(u)) : sqrt(-2.0 * log(1.0 - u));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double z = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t);
        host[i] = (u < 0.5) ? (float)(-z) : (float)z;
    }
}

// --------------------------------------------------------------
// Greedy encoder (simplified from trellis-encode.cuh)
// --------------------------------------------------------------
__global__ void k_greedy_encode(
        const float * __restrict__ src,
        block_vtq3_2 * __restrict__ dst,
        int n_blocks)
{
    const int i = blockIdx.x;
    if (i >= n_blocks || threadIdx.x != 0) return;
    constexpr int N = QK_VTQ_TRELLIS;
    constexpr int K = 3;
    constexpr int L = VTQ_TRELLIS_L;
    constexpr uint32_t Lmask = 0xFFFFu;
    constexpr uint32_t Kmask = (1u << K) - 1u;
    constexpr int kshift = L - K;

    const float * x = src + i * N;
    block_vtq3_2 * b = dst + i;

    float sq = 0.0f;
    for (int j = 0; j < N; j++) { float v = x[j]; sq += v*v; }
    const float norm = sqrtf(sq);
    if (norm <= 1e-30f) {
        b->start_state = 0;
        b->d = __float2half(0.0f);
        for (int z = 0; z < 96; z++) b->qs[z] = 0;
        return;
    }
    const float inv_norm = 1.0f / norm;
    const float cb_scale = rsqrtf((float)N);

    uint32_t state = 0;
    b->start_state = 0;
    for (int z = 0; z < 96; z++) b->qs[z] = 0;
    uint8_t * qs = b->qs;

    float recon_sq = 0.0f;
    for (int step = 0; step < N; step++) {
        const float xi = x[step] * inv_norm;
        float best_d2 = 1e38f;
        uint32_t best_e = 0, best_ns = 0;
        #pragma unroll
        for (uint32_t e = 0; e <= Kmask; e++) {
            const uint32_t next_state = ((state >> K) | (e << kshift)) & Lmask;
            const float code = g_lut[next_state] * cb_scale;
            const float diff = xi - code;
            const float d2 = diff * diff;
            if (d2 < best_d2) { best_d2 = d2; best_e = e; best_ns = next_state; }
        }
        const int bo = step * K, byte = bo >> 3, shift = bo & 7;
        qs[byte] |= (uint8_t)((best_e << shift) & 0xFFu);
        if (shift + K > 8) {
            qs[byte + 1] |= (uint8_t)((best_e >> (8 - shift)) & 0xFFu);
            if (shift + K > 16) {
                qs[byte + 2] |= (uint8_t)((best_e >> (16 - shift)) & 0xFFu);
            }
        }
        const float code = g_lut[best_ns] * cb_scale;
        recon_sq += code * code;
        state = best_ns;
    }
    const float recon_norm = sqrtf(recon_sq);
    const float d_out = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
    b->d = __float2half(d_out);
}

// --------------------------------------------------------------
// main — time N encoder launches
// --------------------------------------------------------------
int main(int argc, char ** argv) {
    int iters    = (argc > 1) ? atoi(argv[1]) : 1000;
    int n_blocks = (argc > 2) ? atoi(argv[2]) : 2;  // matches tg ne11=1, 2 KV heads

    printf("bench_encoder_gpu: iters=%d n_blocks=%d\n", iters, n_blocks);

    // Init LUT
    float * host_lut = (float*) malloc(sizeof(float) * (1 << VTQ_TRELLIS_L));
    build_lut_host(host_lut);
    CUDA_CHECK(cudaMemcpyToSymbol(g_lut, host_lut, sizeof(float) * (1 << VTQ_TRELLIS_L)));
    free(host_lut);

    // Allocate test data
    size_t n_samples = (size_t)n_blocks * QK_VTQ_TRELLIS;
    float * d_src;
    block_vtq3_2 * d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, n_blocks * sizeof(block_vtq3_2)));

    // Fill with random-ish pattern (single-token V post-RHT: roughly Gaussian)
    float * host_src = (float*) malloc(n_samples * sizeof(float));
    for (size_t i = 0; i < n_samples; i++) {
        host_src[i] = 0.5f * sinf((float)i * 0.013f) + 0.1f * cosf((float)i * 0.41f);
    }
    CUDA_CHECK(cudaMemcpy(d_src, host_src, n_samples * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int w = 0; w < 3; w++) {
        k_greedy_encode<<<n_blocks, 32>>>(d_src, d_dst, n_blocks);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time
    cudaEvent_t e_start, e_end;
    cudaEventCreate(&e_start); cudaEventCreate(&e_end);
    cudaEventRecord(e_start);
    for (int it = 0; it < iters; it++) {
        k_greedy_encode<<<n_blocks, 32>>>(d_src, d_dst, n_blocks);
    }
    cudaEventRecord(e_end);
    cudaEventSynchronize(e_end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e_start, e_end);

    printf("greedy:  %d iters × %d blocks = %.2f ms total = %.4f ms/call = %.4f us/block\n",
           iters, n_blocks, ms, ms/iters, (ms/iters)*1000.0f/n_blocks);

    free(host_src);
    cudaFree(d_src); cudaFree(d_dst);
    return 0;
}
