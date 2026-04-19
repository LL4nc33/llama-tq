// Trellis v2 CUDA support TU:
//   - Decoder lives in trellis.cuh (header-only, per-TU LUT copy).
//   - Encoder kernel template lives in trellis-encode.cuh and is instantiated
//     directly in set-rows.cu for each (idx_t, block_t, K) triple.
//   - This TU owns the global-memory WORKSPACE POOL for the encoder:
//       * dp_cur  : pool_slots * S   floats
//       * dp_next : pool_slots * S   uint64 (packed cost<<32|prev during DP)
//       * bt      : pool_slots * N * S  uint16_t
//     Per slot: 256 KiB + 512 KiB + 32 MiB  ≈  32.75 MiB.
//     Default pool_slots = 8 → 262 MiB reserved per-device on first use.
//     Override via env: GGML_CUDA_VTQ_POOL_SLOTS=<1..64>.

// Emit the single definition of the LUT in THIS TU.
#define VTQ_TRELLIS_TABLE_DEFINE
#include "trellis.cuh"
#include "trellis-encode.cuh"

#include <cstdio>
#include <cstdlib>

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
    cudaGetDevice(&device);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) device = 0;

    vtq_encode_workspace * ws     = &g_vtq_ws[device];
    bool *                 inited = &g_vtq_ws_inited[device];

    if (!*inited) {
        const int    slots         = get_pool_slots_env();
        const size_t bytes_dp_cur  = (size_t)slots * (size_t)VTQ_ENC_S * sizeof(float);
        // dp_next is reinterpreted as uint64_t during DP (packed cost + prev).
        const size_t bytes_dp_next = (size_t)slots * (size_t)VTQ_ENC_S * sizeof(uint64_t);
        const size_t bytes_bt      = (size_t)slots * (size_t)VTQ_ENC_N * (size_t)VTQ_ENC_S * sizeof(uint16_t);

        cudaError_t err;
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
        if (ws->dp_cur)  cudaFree(ws->dp_cur);
        if (ws->dp_next) cudaFree(ws->dp_next);
        if (ws->bt)      cudaFree(ws->bt);
        *ws = { nullptr, nullptr, nullptr, 0 };
        g_vtq_ws_inited[d] = false;
    }
}
