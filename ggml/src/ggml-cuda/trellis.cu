// Trellis v2 CUDA support TU:
//   - Decoder lives in trellis.cuh (header-only); this TU owns the per-device LUT copies.
//   - Encoder kernel templates live in trellis-encode.cuh and are instantiated
//     directly in set-rows.cu for each (idx_t, block_t, K, L) combination.
//   - This TU owns the encoder's backtrack pools, one per device and state bit count L, in slots of N * S/4 bytes:
//     L = 12 (vtq*_2s): 128 KiB per slot, 256 slots (32 MiB) by default;
//     L = 16 (vtq*_2, vtq*_3, vtq3_v8): 2 MiB per slot, one slot per SM by default (60 MiB on an RTX 2060).
//     Override the slot count via env: GGML_CUDA_VTQ_POOL_SLOTS=<1..4096>.

#include "trellis.cuh"
#include "trellis-encode.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

const float * ggml_cuda_trellis_table_device(int device) {
    static float *     tables[GGML_CUDA_MAX_DEVICES] = {};
    static std::mutex  mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (tables[device] == nullptr) {
        const size_t size = sizeof(float) * (1 << GGML_TRELLIS_L);
        CUDA_CHECK(cudaMalloc((void **) &tables[device], size));
        CUDA_CHECK(cudaMemcpy(tables[device], ggml_trellis_table(), size, cudaMemcpyHostToDevice));
    }
    return tables[device];
}

// One backtrack pool per CUDA device and state bit count (index L), allocated when that L is first encoded on the
// device (so the _2s types alone never reserve the L = 16 pool) and kept: CUDA graphs hold its address.
// bytes == 0: not allocated.
static vtq_encode_workspace g_vtq_ws[GGML_CUDA_MAX_DEVICES][GGML_TRELLIS_L + 1] = {};

// A slot holds the backtrack of one 2-bit trellis block (N * S/4 bytes; a 3/4-bit encode runs 2/4 CUDA blocks per slot).
// Default: 32 MiB, and at least one slot per SM, as one CUDA block of the L = 16 encoder fills an SM.
static int vtq_pool_slots(int L, int device) {
    const char * env = getenv("GGML_CUDA_VTQ_POOL_SLOTS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 4096) return v;
    }
    const int64_t slot_bytes = (int64_t) VTQ_ENC_N << (L - 2);
    return (int) std::max<int64_t>(ggml_cuda_info().devices[device].nsm, (32 << 20) / slot_bytes);
}

const vtq_encode_workspace * vtq_get_encode_workspace(int L) {
    GGML_ASSERT(L > 2 && L <= GGML_TRELLIS_L);
    int device = 0;
    (void) cudaGetDevice(&device);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) device = 0;

    vtq_encode_workspace * ws = &g_vtq_ws[device][L];

    if (ws->bytes == 0) {
        const int    slots = vtq_pool_slots(L, device);
        const size_t bytes = (size_t) slots * ((size_t) VTQ_ENC_N << (L - 2));
        const cudaError_t err = cudaMalloc((void **) &ws->bt, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vtq-enc] dev %d cudaMalloc(bt, %zu B) failed: %s\n",
                    device, bytes, cudaGetErrorString(err));
            abort();
        }
        ws->bytes = (int64_t) bytes;
        fprintf(stderr, "[vtq-enc] dev %d backtrack pool for L = %d allocated: %d slots, %.1f MiB\n",
                device, L, slots, bytes / (1024.0 * 1024.0));
    }
    return ws;
}
