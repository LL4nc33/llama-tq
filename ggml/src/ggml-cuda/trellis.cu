// Trellis v2 CUDA support TU:
//   - Decoder lives in trellis.cuh (header-only); this TU owns the per-device LUT copies.
//   - Encoder kernel templates live in trellis-encode.cuh and are instantiated
//     directly in set-rows.cu for each (idx_t, block_t, K, L) combination.
//   - This TU owns the global-memory WORKSPACE POOLS for the encoder, one per device and state bit count L:
//       * bt      : pool_slots * N * S  uint16_t (backtrack)
//       * dp_cur  : pool_slots * S   floats, and
//         dp_next : pool_slots * S   uint64 (packed cost<<32|prev), only for S > VTQ_ENC_SMEM_MAX_S, where the DP
//                   rows do not fit in shared memory
//     Per slot at L = 12 (vtq*_2s): 1 MiB (32 MiB with the default 32 slots);
//     at L = 16 (vtq*_2, vtq*_3, vtq3_v8): 16.75 MiB (536 MiB).
//     A pool is allocated when its L is first encoded on that device, so the _2s types alone never reserve the
//     L = 16 pool. Override via env: GGML_CUDA_VTQ_POOL_SLOTS=<1..128>.

#include "trellis.cuh"
#include "trellis-encode.cuh"

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

// One pool per CUDA device and state bit count (index L). Pools are lazily allocated when the first encode
// is requested from that device so that multi-GPU runs don't over-subscribe
// when only one device owns VTQ_2 cache. pool_slots == 0: not allocated.
static vtq_encode_workspace g_vtq_ws[GGML_CUDA_MAX_DEVICES][GGML_TRELLIS_L + 1] = {};

static int get_pool_slots_env(void) {
    const char * env = getenv("GGML_CUDA_VTQ_POOL_SLOTS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 128) return v;
    }
    // Default 32 (536 MiB per device at L = 16): ~4× faster than 8 on 0.8B (57s vs 177s/pass).
    // Bigger (64) gives further 1.3× speedup at twice the memory — override via env if VRAM allows.
    return 32;
}

const vtq_encode_workspace * vtq_get_encode_workspace(int L) {
    GGML_ASSERT(L > 0 && L <= GGML_TRELLIS_L);
    int device = 0;
    (void) cudaGetDevice(&device);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) device = 0;

    vtq_encode_workspace * ws = &g_vtq_ws[device][L];

    if (ws->pool_slots == 0) {
        const int    slots         = get_pool_slots_env();
        const size_t S             = (size_t) 1 << L;
        // the shared-memory encoder keeps its DP rows on chip and needs only the backtrack
        const bool   dp_global     = S > VTQ_ENC_SMEM_MAX_S;
        const size_t bytes_dp_cur  = dp_global ? (size_t)slots * S * sizeof(float)    : 0;
        // dp_next is reinterpreted as uint64_t during DP (packed cost + prev).
        const size_t bytes_dp_next = dp_global ? (size_t)slots * S * sizeof(uint64_t) : 0;
        const size_t bytes_bt      = (size_t)slots * (size_t)VTQ_ENC_N * S * sizeof(uint16_t);

        cudaError_t err;
        if (dp_global) {
            err = cudaMalloc((void **)&ws->dp_cur, bytes_dp_cur);
            if (err != cudaSuccess) {
                fprintf(stderr, "[vtq-enc] dev %d cudaMalloc(dp_cur, %zu B) failed: %s\n",
                        device, bytes_dp_cur, cudaGetErrorString(err));
                abort();
            }
            err = cudaMalloc((void **)&ws->dp_next, bytes_dp_next);
            if (err != cudaSuccess) {
                fprintf(stderr, "[vtq-enc] dev %d cudaMalloc(dp_next, %zu B) failed: %s\n",
                        device, bytes_dp_next, cudaGetErrorString(err));
                abort();
            }
        }
        err = cudaMalloc((void **)&ws->bt, bytes_bt);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vtq-enc] dev %d cudaMalloc(bt, %zu B) failed: %s\n",
                    device, bytes_bt, cudaGetErrorString(err));
            abort();
        }
        ws->pool_slots = slots;
        fprintf(stderr, "[vtq-enc] dev %d workspace pool for L = %d allocated: %d slots, %.1f MiB total\n",
                device, L, slots,
                (bytes_dp_cur + bytes_dp_next + bytes_bt) / (1024.0 * 1024.0));
    }
    return ws;
}

void vtq_free_encode_workspace(void) {
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        for (vtq_encode_workspace & ws : g_vtq_ws[d]) {
            if (ws.pool_slots == 0) continue;
            // Best-effort: device may already be unusable at teardown.
            if (ws.dp_cur)  (void) cudaFree(ws.dp_cur);
            if (ws.dp_next) (void) cudaFree(ws.dp_next);
            if (ws.bt)      (void) cudaFree(ws.bt);
            ws = {};
        }
    }
}
