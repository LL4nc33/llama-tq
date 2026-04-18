// Trellis v2 CUDA support TU:
//   - Decoder lives in trellis.cuh (header-only, per-TU LUT copy).
//   - Encoder kernel template lives in trellis-encode.cuh and is instantiated
//     directly in set-rows.cu for each (idx_t, block_t, K) triple.
//   - This TU owns the global-memory WORKSPACE POOL for the encoder:
//       * dp_cur  : pool_slots * S   floats
//       * dp_next : pool_slots * S   floats (used as uint64 during DP)
//       * bt      : pool_slots * N * S  uint16_t
//     Per slot: 256 KiB + 256 KiB + 32 MiB  ≈  32.5 MiB.
//     Default pool_slots = 8 → 260 MiB reserved on first encode call.
//     Override via env: GGML_CUDA_VTQ_POOL_SLOTS=<1..64>.

#include "trellis.cuh"
#include "trellis-encode.cuh"

#include <cstdio>
#include <cstdlib>

static vtq_encode_workspace g_vtq_ws        = { nullptr, nullptr, nullptr, 0 };
static bool                 g_vtq_ws_inited = false;

static int get_pool_slots_env(void) {
    const char * env = getenv("GGML_CUDA_VTQ_POOL_SLOTS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 64) return v;
    }
    return 8;
}

const vtq_encode_workspace * vtq_get_encode_workspace(cudaStream_t /*stream*/) {
    if (!g_vtq_ws_inited) {
        const int    slots    = get_pool_slots_env();
        // dp_cur: float per state (pure float cost row, carried across steps).
        const size_t bytes_dp_cur  = (size_t)slots * (size_t)VTQ_ENC_S * sizeof(float);
        // dp_next: during DP we reinterpret this buffer as uint64 (packed (cost<<32)|prev)
        // for atomicMin. So allocate sizeof(uint64_t) per state.
        const size_t bytes_dp_next = (size_t)slots * (size_t)VTQ_ENC_S * sizeof(uint64_t);
        const size_t bytes_bt      = (size_t)slots * (size_t)VTQ_ENC_N * (size_t)VTQ_ENC_S * sizeof(uint16_t);
        const size_t bytes_dp      = bytes_dp_cur;  // legacy alias for log msg

        cudaError_t err;
        err = cudaMalloc((void **)&g_vtq_ws.dp_cur,  bytes_dp_cur);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vtq-enc] cudaMalloc(dp_cur, %zu B) failed: %s\n",
                    bytes_dp, cudaGetErrorString(err));
            abort();
        }
        err = cudaMalloc((void **)&g_vtq_ws.dp_next, bytes_dp_next);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vtq-enc] cudaMalloc(dp_next, %zu B) failed: %s\n",
                    bytes_dp_next, cudaGetErrorString(err));
            abort();
        }
        err = cudaMalloc((void **)&g_vtq_ws.bt, bytes_bt);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vtq-enc] cudaMalloc(bt, %zu B) failed: %s\n",
                    bytes_bt, cudaGetErrorString(err));
            abort();
        }
        g_vtq_ws.pool_slots = slots;
        g_vtq_ws_inited     = true;
        fprintf(stderr, "[vtq-enc] workspace pool allocated: %d slots, %.1f MiB total\n",
                slots,
                (bytes_dp_cur + bytes_dp_next + bytes_bt) / (1024.0 * 1024.0));
    }
    return &g_vtq_ws;
}

void vtq_free_encode_workspace(void) {
    if (g_vtq_ws_inited) {
        if (g_vtq_ws.dp_cur)  cudaFree(g_vtq_ws.dp_cur);
        if (g_vtq_ws.dp_next) cudaFree(g_vtq_ws.dp_next);
        if (g_vtq_ws.bt)      cudaFree(g_vtq_ws.bt);
        g_vtq_ws = { nullptr, nullptr, nullptr, 0 };
        g_vtq_ws_inited = false;
    }
}
