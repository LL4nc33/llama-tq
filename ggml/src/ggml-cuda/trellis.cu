// Trellis v2 CUDA support TU:
//   - Decoder lives in trellis.cuh (header-only); this TU owns the per-device LUT copies.
//   - Encoder kernel template lives in trellis-encode.cuh and is instantiated
//     directly in set-rows.cu for each (idx_t, block_t, K) triple.
//   - This TU owns the global-memory WORKSPACE POOL for the encoder:
//       * bt      : pool_slots * N * S  uint16_t (backtrack)
//       * dp_cur  : pool_slots * S   floats, and
//         dp_next : pool_slots * S   uint64 (packed cost<<32|prev), only for S > VTQ_ENC_SMEM_MAX_S, where the DP
//                   rows do not fit in shared memory
//     Per slot at L = 12: 1 MiB (8 MiB with the default 8 slots); at L = 16: 32.75 MiB (262 MiB).
//     Override via env: GGML_CUDA_VTQ_POOL_SLOTS=<1..64>.

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
        const size_t size = sizeof(float) * (1 << VTQ_TRELLIS_L);
        CUDA_CHECK(cudaMalloc((void **) &tables[device], size));
        CUDA_CHECK(cudaMemcpy(tables[device], ggml_trellis_table(), size, cudaMemcpyHostToDevice));
    }
    return tables[device];
}

// One pool per CUDA device. Pools are lazily allocated when the first encode
// is requested from that device so that multi-GPU runs don't over-subscribe
// when only one device owns VTQ_2 cache.
static vtq_encode_workspace g_vtq_ws[GGML_CUDA_MAX_DEVICES]        = {};
static bool                 g_vtq_ws_inited[GGML_CUDA_MAX_DEVICES] = {};

static int get_pool_slots_env(void) {
    const char * env = getenv("GGML_CUDA_VTQ_POOL_SLOTS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 128) return v;
    }
    // Default 32: ~1 GB per device, ~4× faster than 8 on 0.8B (57s vs 177s/pass).
    // Bigger (64) gives further 1.3× speedup at 2 GB — override via env if VRAM allows.
    return 32;
}

const vtq_encode_workspace * vtq_get_encode_workspace(cudaStream_t /*stream*/) {
    int device = 0;
    (void) cudaGetDevice(&device);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) device = 0;

    vtq_encode_workspace * ws     = &g_vtq_ws[device];
    bool *                 inited = &g_vtq_ws_inited[device];

    if (!*inited) {
        const int    slots         = get_pool_slots_env();
        // the shared-memory encoder keeps its DP rows on chip and needs only the backtrack
        constexpr bool dp_global   = VTQ_ENC_S > VTQ_ENC_SMEM_MAX_S;
        const size_t bytes_dp_cur  = dp_global ? (size_t)slots * (size_t)VTQ_ENC_S * sizeof(float)    : 0;
        // dp_next is reinterpreted as uint64_t during DP (packed cost + prev).
        const size_t bytes_dp_next = dp_global ? (size_t)slots * (size_t)VTQ_ENC_S * sizeof(uint64_t) : 0;
        const size_t bytes_bt      = (size_t)slots * (size_t)VTQ_ENC_N * (size_t)VTQ_ENC_S * sizeof(uint16_t);

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
        *inited        = true;
        fprintf(stderr, "[vtq-enc] dev %d workspace pool allocated: %d slots, %.1f MiB total\n",
                device, slots,
                (bytes_dp_cur + bytes_dp_next + bytes_bt) / (1024.0 * 1024.0));
    }
    return ws;
}

void vtq_free_encode_workspace(void) {
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (!g_vtq_ws_inited[d]) continue;
        vtq_encode_workspace * ws = &g_vtq_ws[d];
        // Best-effort: device may already be unusable at teardown.
        if (ws->dp_cur)  (void) cudaFree(ws->dp_cur);
        if (ws->dp_next) (void) cudaFree(ws->dp_next);
        if (ws->bt)      (void) cudaFree(ws->bt);
        *ws = { nullptr, nullptr, nullptr, 0 };
        g_vtq_ws_inited[d] = false;
    }
}
