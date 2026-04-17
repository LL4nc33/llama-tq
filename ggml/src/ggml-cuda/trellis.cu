#include "trellis.cuh"
#include "ggml-trellis.h"   // ggml_trellis_table() host accessor

#include <cuda_runtime.h>
#include <atomic>

// Global device-memory LUT. 256 KiB — too large for __constant__.
__device__ float vtq_trellis_table[1 << VTQ_TRELLIS_L];

static std::atomic<int> g_init_done{0};

// Call once at first dequant. Thread-safe idempotent init.
void ggml_cuda_init_trellis_table(void) {
    int expected = 0;
    if (!std::atomic_compare_exchange_strong(&g_init_done, &expected, 1)) {
        // Another thread is initializing or done.
        while (g_init_done.load() != 2) {}  // spin
        return;
    }
    const float * host_table = ggml_trellis_table();
    cudaMemcpyToSymbol(vtq_trellis_table, host_table,
                       sizeof(float) * (1 << VTQ_TRELLIS_L), 0,
                       cudaMemcpyHostToDevice);
    g_init_done.store(2);
}

// Encoder: for set_rows path. We do quantize on CPU via the existing
// from_float type_traits_cpu, then cudaMemcpy the block to device. This
// is a workable "phase 2a" that unlocks measurement without a full CUDA
// Viterbi encoder. Phase 2b adds the GPU Viterbi for production speed.
