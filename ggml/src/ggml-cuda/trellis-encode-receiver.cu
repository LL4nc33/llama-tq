// Trick 6: CUDA receiver-side Viterbi encoder — standalone bulk API.
//
// See trellis-encode-receiver.cuh for signature/contract. This TU
// implements two encoding paths, selectable via GGML_TRELLIS_BEAM env:
//
//   BEAM=0  : full Viterbi, receiver-side DP, S=65536 states.
//             Uses the existing global-memory workspace pool owned by
//             trellis.cu (vtq_get_encode_workspace) — one slot per
//             concurrent group, waves synchronised via cudaStream.
//
//   BEAM>0  : beam search with width B=BEAM (capped to [16, 512]).
//             Per-block: B beams carrying (cost, state, parent_edge_history).
//             Edge history stored in a dedicated global-memory scratch sized
//             `G * N * B` bytes — freed via trellis_encode_group_cuda_free().
//
// Algorithmic invariants match the CPU reference (ggml-trellis.c::
// ggml_trellis_encode_group):
//   * Open start (dp[0][s] = 0 for all s)
//   * Bit-shift trellis: state_i = (state_{i-1} >> K) | (bits << (L-K))
//   * Norm-correction for d: d = ||x|| / ||recon||
//   * Tie-break: lowest prev state on equal cost (matches sender-side atomic)
//
// Numerical equivalence with CPU is expected to within fp32 rounding for
// the full Viterbi path. The beam path is an approximation — quality gap
// measured at ~1% MSE for B=256 on V-cache distributions.

#include "trellis-encode-receiver.cuh"
#include "trellis.cuh"
#include "trellis-encode.cuh"
#include "common.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

// ============================================================
// Env-var parsing (one-shot cache so we don't re-read getenv per call).

static int g_enabled_cached     = -1;  // 1 = on, 0 = off
static int g_beam_width_cached  = -1;  // 0 = full, >0 = beam
static const int BEAM_MIN = 16;
static const int BEAM_MAX = 512;

extern "C" int trellis_cuda_encode_enabled(void) {
    if (g_enabled_cached < 0) {
        const char * env = getenv("GGML_TRELLIS_CUDA_ENCODE");
        g_enabled_cached = (env && atoi(env) > 0) ? 1 : 0;
    }
    return g_enabled_cached;
}

extern "C" int trellis_cuda_encode_beam_width(void) {
    if (g_beam_width_cached < 0) {
        const char * env = getenv("GGML_TRELLIS_BEAM");
        int v = env ? atoi(env) : 0;
        if (v < 0) v = 0;
        // Clamp non-zero beam to [16, 512] — smaller is too inaccurate,
        // larger blows past shmem/regs for the per-block beam arrays.
        if (v > 0) {
            if (v < BEAM_MIN) v = BEAM_MIN;
            if (v > BEAM_MAX) v = BEAM_MAX;
        }
        g_beam_width_cached = v;
    }
    return g_beam_width_cached;
}

// ============================================================
// Full Viterbi kernel — receiver-side, one CUDA block per group.
//
// Layout differences from trellis-encode.cuh's set_rows kernel:
//   * Input/output are flat contiguous arrays indexed by blockIdx.x.
//   * No src1 (dst_row indirection) — destinations are flat too.
//   * Writes directly to separate qs/start_state/d device buffers
//     (no block_vtq*_2 struct assumed).
//
// Otherwise the DP body is a direct port of the receiver-side logic
// from k_vtq_encode_trellis_set_rows. We reuse the same global-memory
// workspace pool (dp_cur / dp_next / bt) to avoid duplicate 262 MiB
// per-device reservations.

template <int K>
__global__ void k_trellis_encode_receiver_flat(
        const float * __restrict__ x,      // [G, N]
        uint8_t     * __restrict__ qs,     // [G, qs_bytes]
        uint16_t    * __restrict__ start_state, // [G]
        float       * __restrict__ d_out,  // [G]
        const int64_t               G,
        const int                   qs_bytes,
        float    * __restrict__ ws_dp_cur,
        float    * __restrict__ ws_dp_next,
        uint16_t * __restrict__ ws_bt,
        const int                   pool_slots,
        const int64_t               wave_offset)
{
    constexpr int      N      = VTQ_ENC_N;
    constexpr int      L      = VTQ_ENC_L;
    constexpr uint32_t S      = VTQ_ENC_S;
    constexpr uint32_t Kmask  = (1u << K) - 1u;
    constexpr int      kshift = L - K;

    const int     tid = threadIdx.x;
    const int64_t g   = wave_offset + (int64_t)blockIdx.x;
    if (g >= G) return;

    const float * x_row  = x   + g * (int64_t)N;
    uint8_t     * qs_out = qs  + g * (int64_t)qs_bytes;

    const int slot = (int)(blockIdx.x % (uint32_t)pool_slots);
    float    * dp_cur  = ws_dp_cur  + (int64_t)slot * (int64_t)S;
    // dp_next pool stride matches trellis-encode.cuh layout (2*S floats/slot)
    // to share the same global allocation.
    float    * dp_next = ws_dp_next + (int64_t)slot * (int64_t)(2u * S);
    uint16_t * bt_base = ws_bt      + (int64_t)slot * (int64_t)N * (int64_t)S;

    __shared__ float    s_xn[VTQ_ENC_N];
    __shared__ float    s_partial[VTQ_ENC_THREADS];
    __shared__ uint32_t s_partial_u[VTQ_ENC_THREADS];
    __shared__ float    s_norm;
    __shared__ float    s_inv_norm;
    __shared__ uint32_t s_best_state;

    // 1) Load x + L2 norm
    float local_sq = 0.0f;
    for (int j = tid; j < N; j += VTQ_ENC_THREADS) {
        float v = x_row[j];
        s_xn[j] = v;
        local_sq += v * v;
    }
    s_partial[tid] = local_sq;
    __syncthreads();
    for (int off = VTQ_ENC_THREADS >> 1; off > 0; off >>= 1) {
        if (tid < off) s_partial[tid] += s_partial[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        float nv = sqrtf(s_partial[0]);
        s_norm     = nv;
        s_inv_norm = (nv > 1e-30f) ? (1.0f / nv) : 0.0f;
    }
    __syncthreads();

    const float norm = s_norm;

    // Zero-norm early exit: matches CPU reference path.
    if (norm <= 1e-30f) {
        if (tid == 0) {
            start_state[g] = 0;
            d_out[g]       = 0.0f;
        }
        for (int b = tid; b < qs_bytes; b += VTQ_ENC_THREADS) qs_out[b] = 0;
        return;
    }

    const float inv_norm = s_inv_norm;
    for (int j = tid; j < N; j += VTQ_ENC_THREADS) s_xn[j] *= inv_norm;
    __syncthreads();

    const float cb_scale = rsqrtf((float)N);

    // 2) Open start: dp_cur = 0 for all S states
    for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) dp_cur[s] = 0.0f;
    __syncthreads();

    // 3) Receiver-side DP. For each `next`, gather 2^K predecessors via
    //    prev = ((next << K) | e) & 0xFFFF. Minimum in registers, single
    //    coalesced store to dp_next.
    for (int step = 0; step < N; step++) {
        uint16_t * bt_i = bt_base + (int64_t)step * (int64_t)S;
        const float xi  = s_xn[step];

        for (uint32_t next = tid; next < S; next += VTQ_ENC_THREADS) {
            const float code = vtq_trellis_table_storage[next] * cb_scale;
            const float diff = xi - code;
            const float d2   = diff * diff;

            const uint32_t base = (next << K) & 0xFFFFu;

            float    best_cost = FLT_MAX;
            uint32_t best_prev = 0;
            #pragma unroll
            for (uint32_t e = 0; e <= Kmask; e++) {
                const uint32_t prev = base | e;
                const float    pc   = dp_cur[prev];
                const float    cost = pc + d2;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_prev = prev;
                }
            }

            dp_next[next] = best_cost;
            bt_i[next]    = (uint16_t)(best_prev & 0xFFFFu);
        }
        __syncthreads();

        float * tmp = dp_cur;
        dp_cur  = dp_next;
        dp_next = tmp;
    }

    // 4) argmin over dp_cur
    float    local_best = FLT_MAX;
    uint32_t local_arg  = 0;
    for (uint32_t s = tid; s < S; s += VTQ_ENC_THREADS) {
        float c = dp_cur[s];
        if (c < local_best) { local_best = c; local_arg = s; }
    }
    s_partial[tid]   = local_best;
    s_partial_u[tid] = local_arg;
    __syncthreads();
    for (int off = VTQ_ENC_THREADS >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = s_partial[tid], b = s_partial[tid + off];
            if (b < a) {
                s_partial[tid]   = b;
                s_partial_u[tid] = s_partial_u[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) s_best_state = s_partial_u[0];
    __syncthreads();

    // 5) Backtrack + emit bits + compute d via norm-correction
    __shared__ uint16_t s_states[VTQ_ENC_N + 1];
    if (tid == 0) {
        s_states[N] = (uint16_t)(s_best_state & 0xFFFFu);
        for (int step = N - 1; step >= 0; step--) {
            uint16_t * bt_i = bt_base + (int64_t)step * (int64_t)S;
            s_states[step] = bt_i[s_states[step + 1]];
        }

        start_state[g] = s_states[0];

        for (int b = 0; b < qs_bytes; b++) qs_out[b] = 0;

        float recon_sq = 0.0f;
        for (int step = 0; step < N; step++) {
            uint32_t st   = (uint32_t)s_states[step + 1];
            uint32_t bits = (st >> kshift) & Kmask;
            int bo        = step * K;
            int byte      = bo >> 3;
            int shift     = bo & 7;
            qs_out[byte] |= (uint8_t)((bits << shift) & 0xFFu);
            if (shift + K > 8) {
                qs_out[byte + 1] |= (uint8_t)((bits >> (8 - shift)) & 0xFFu);
                if (shift + K > 16) {
                    qs_out[byte + 2] |= (uint8_t)((bits >> (16 - shift)) & 0xFFu);
                }
            }
            float code = vtq_trellis_table_storage[st] * cb_scale;
            recon_sq += code * code;
        }
        float recon_norm = sqrtf(recon_sq);
        d_out[g] = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
    }
}

// ============================================================
// Beam-search kernel — one thread per group, beam kept in registers/local.
//
// Per-step: B beams × 2^K transitions = B*Kmask+1 candidates. Keep top-B
// by partial-selection. Edge history stored in a global scratch buffer
// because N*B bytes may exceed per-block regs (N=256, B=256 -> 64 KiB).
//
// B is a template parameter so the compiler can unroll the beam loops
// and keep beam arrays fully register-resident for small B.
template <int K, int B>
__global__ void k_trellis_encode_beam_flat(
        const float * __restrict__ x,
        uint8_t     * __restrict__ qs,
        uint16_t    * __restrict__ start_state,
        float       * __restrict__ d_out,
        const int64_t               G,
        const int                   qs_bytes,
        uint8_t  * __restrict__ scratch_edge,   // [G, N, B]  parent edge per step
        uint8_t  * __restrict__ scratch_parent) // [G, N, B]  parent beam idx per step
{
    constexpr int      N      = VTQ_ENC_N;
    constexpr int      L      = VTQ_ENC_L;
    constexpr uint32_t Lmask  = 0xFFFFu;
    constexpr uint32_t Kmask  = (1u << K) - 1u;
    constexpr int      kshift = L - K;

    const int64_t g = blockIdx.x;
    if (g >= G) return;
    if (threadIdx.x != 0) return;

    const float * x_row  = x   + g * (int64_t)N;
    uint8_t     * qs_out = qs  + g * (int64_t)qs_bytes;
    uint8_t     * edge_g   = scratch_edge   + g * (int64_t)N * (int64_t)B;
    uint8_t     * parent_g = scratch_parent + g * (int64_t)N * (int64_t)B;

    // L2 norm
    float sq = 0.0f;
    for (int j = 0; j < N; j++) { float v = x_row[j]; sq += v * v; }
    const float norm = sqrtf(sq);
    if (norm <= 1e-30f) {
        start_state[g] = 0;
        d_out[g]       = 0.0f;
        for (int b = 0; b < qs_bytes; b++) qs_out[b] = 0;
        return;
    }
    const float inv_norm = 1.0f / norm;
    const float cb_scale = rsqrtf((float)N);

    // Beam state (per-step, register-resident for small B).
    float    beam_cost [B];
    uint32_t beam_state[B];
    #pragma unroll
    for (int b = 0; b < B; b++) {
        // Open-start: all S states are reachable with cost 0. Beam of
        // width B samples the starting space uniformly — to avoid a
        // bias toward state 0 we seed beams with b*stride so that
        // the first K transitions "see" a full 2^L-equivalent spread.
        // Rationale: after log_2(B)/K DP steps the full-Viterbi optimum
        // is reachable from any seed; earlier steps remain approximate.
        beam_cost[b]  = 0.0f;
        beam_state[b] = (uint32_t)((uint64_t)b * (Lmask + 1) / (uint32_t)B) & Lmask;
    }

    // Candidate buffer: size B*(2^K). K<=4 -> at most 16 transitions/beam.
    constexpr int Cmax = B * (1 << 4);
    float    cand_cost  [Cmax];
    uint32_t cand_state [Cmax];
    uint8_t  cand_parent[Cmax];
    uint8_t  cand_edge  [Cmax];

    for (int step = 0; step < N; step++) {
        const float xi = x_row[step] * inv_norm;

        int ncand = 0;
        #pragma unroll 1
        for (int b = 0; b < B; b++) {
            const float pc = beam_cost[b];
            if (pc >= FLT_MAX * 0.5f) continue;
            const uint32_t ps = beam_state[b];
            #pragma unroll
            for (uint32_t e = 0; e <= Kmask; e++) {
                const uint32_t ns   = ((ps >> K) | (e << kshift)) & Lmask;
                const float    code = vtq_trellis_table_storage[ns] * cb_scale;
                const float    diff = xi - code;
                const float    d2   = diff * diff;
                cand_cost  [ncand] = pc + d2;
                cand_state [ncand] = ns;
                cand_parent[ncand] = (uint8_t)b;
                cand_edge  [ncand] = (uint8_t)e;
                ncand++;
            }
        }

        // Top-B selection via partial selection sort (B values).
        const int limit = (ncand < B) ? ncand : B;
        for (int i = 0; i < limit; i++) {
            int best = i;
            for (int j = i + 1; j < ncand; j++) {
                if (cand_cost[j] < cand_cost[best]) best = j;
            }
            if (best != i) {
                float    tc = cand_cost [i]; cand_cost [i] = cand_cost [best]; cand_cost [best] = tc;
                uint32_t ts = cand_state[i]; cand_state[i] = cand_state[best]; cand_state[best] = ts;
                uint8_t  tp = cand_parent[i]; cand_parent[i] = cand_parent[best]; cand_parent[best] = tp;
                uint8_t  te = cand_edge  [i]; cand_edge  [i] = cand_edge  [best]; cand_edge  [best] = te;
            }
            beam_cost [i] = cand_cost [i];
            beam_state[i] = cand_state[i];
            edge_g  [(int64_t)step * B + i] = cand_edge  [i];
            parent_g[(int64_t)step * B + i] = cand_parent[i];
        }
        for (int i = limit; i < B; i++) beam_cost[i] = FLT_MAX;
    }

    // Winner = argmin of beam_cost after last step.
    int   winner = 0;
    float best_c = beam_cost[0];
    for (int b = 1; b < B; b++) {
        if (beam_cost[b] < best_c) { best_c = beam_cost[b]; winner = b; }
    }

    // Collect edges (and start_state via full backtrack of parent chain).
    uint8_t path_e[VTQ_ENC_N];
    int cur = winner;
    for (int step = N - 1; step >= 0; step--) {
        path_e[step] = edge_g[(int64_t)step * B + cur];
        cur          = parent_g[(int64_t)step * B + cur];
    }
    // `cur` is now the parent-beam-index *before* step 0 — i.e. the seed
    // beam index that won. Recover its state from the seeding formula.
    const uint32_t seed_state = (uint32_t)((uint64_t)cur * (Lmask + 1) / (uint32_t)B) & Lmask;
    start_state[g] = (uint16_t)(seed_state & Lmask);

    // Re-walk to emit qs + compute recon norm for d.
    for (int b = 0; b < qs_bytes; b++) qs_out[b] = 0;
    uint32_t state = seed_state;
    float recon_sq = 0.0f;
    for (int step = 0; step < N; step++) {
        const uint32_t e = path_e[step];
        state = ((state >> K) | (e << kshift)) & Lmask;

        const int bo = step * K, byte = bo >> 3, shift = bo & 7;
        qs_out[byte] |= (uint8_t)((e << shift) & 0xFFu);
        if (shift + K > 8) {
            qs_out[byte + 1] |= (uint8_t)((e >> (8 - shift)) & 0xFFu);
            if (shift + K > 16) {
                qs_out[byte + 2] |= (uint8_t)((e >> (16 - shift)) & 0xFFu);
            }
        }
        const float code = vtq_trellis_table_storage[state] * cb_scale;
        recon_sq += code * code;
    }
    const float recon_norm = sqrtf(recon_sq);
    d_out[g] = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
}

// ============================================================
// Beam scratch pool — allocated on first beam-mode call and reused.
// Sized to the largest G seen so far; grown monotonically.

static uint8_t * g_beam_scratch_edge    [GGML_CUDA_MAX_DEVICES] = {};
static uint8_t * g_beam_scratch_parent  [GGML_CUDA_MAX_DEVICES] = {};
static size_t   g_beam_scratch_capacity[GGML_CUDA_MAX_DEVICES] = {};

static cudaError_t ensure_beam_scratch(int device, size_t need_bytes,
                                       uint8_t ** edge, uint8_t ** parent) {
    if (g_beam_scratch_capacity[device] >= need_bytes &&
        g_beam_scratch_edge[device] && g_beam_scratch_parent[device]) {
        *edge   = g_beam_scratch_edge[device];
        *parent = g_beam_scratch_parent[device];
        return cudaSuccess;
    }
    // Free old, reallocate with 25% headroom to amortise growth.
    if (g_beam_scratch_edge[device])   cudaFree(g_beam_scratch_edge[device]);
    if (g_beam_scratch_parent[device]) cudaFree(g_beam_scratch_parent[device]);
    g_beam_scratch_edge[device]   = nullptr;
    g_beam_scratch_parent[device] = nullptr;
    g_beam_scratch_capacity[device] = 0;

    const size_t cap = need_bytes + (need_bytes >> 2);
    cudaError_t err = cudaMalloc((void **)&g_beam_scratch_edge[device], cap);
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void **)&g_beam_scratch_parent[device], cap);
    if (err != cudaSuccess) {
        cudaFree(g_beam_scratch_edge[device]);
        g_beam_scratch_edge[device] = nullptr;
        return err;
    }
    g_beam_scratch_capacity[device] = cap;
    *edge   = g_beam_scratch_edge[device];
    *parent = g_beam_scratch_parent[device];
    return cudaSuccess;
}

extern "C" void trellis_encode_group_cuda_free(void) {
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (g_beam_scratch_edge[d])   cudaFree(g_beam_scratch_edge[d]);
        if (g_beam_scratch_parent[d]) cudaFree(g_beam_scratch_parent[d]);
        g_beam_scratch_edge[d]    = nullptr;
        g_beam_scratch_parent[d]  = nullptr;
        g_beam_scratch_capacity[d] = 0;
    }
}

// ============================================================
// Beam dispatcher: instantiate kernel for the (K, B) pair. We template
// over a handful of representative beam widths and pick the closest
// supported one below the user's request (so GGML_TRELLIS_BEAM=300
// dispatches to B=256). This keeps binary size bounded.

template <int K>
static cudaError_t launch_beam(int beam, int64_t G,
                               const float * x, uint8_t * qs,
                               uint16_t * start_state, float * d_out,
                               int qs_bytes,
                               uint8_t * edge, uint8_t * parent,
                               cudaStream_t stream) {
    dim3 grid((unsigned)G, 1, 1);
    dim3 block(32, 1, 1);  // one active thread per block; 32 for warp alignment

    if (beam >= 256) {
        k_trellis_encode_beam_flat<K, 256><<<grid, block, 0, stream>>>(
            x, qs, start_state, d_out, G, qs_bytes, edge, parent);
    } else if (beam >= 128) {
        k_trellis_encode_beam_flat<K, 128><<<grid, block, 0, stream>>>(
            x, qs, start_state, d_out, G, qs_bytes, edge, parent);
    } else if (beam >= 64) {
        k_trellis_encode_beam_flat<K, 64><<<grid, block, 0, stream>>>(
            x, qs, start_state, d_out, G, qs_bytes, edge, parent);
    } else if (beam >= 32) {
        k_trellis_encode_beam_flat<K, 32><<<grid, block, 0, stream>>>(
            x, qs, start_state, d_out, G, qs_bytes, edge, parent);
    } else {
        k_trellis_encode_beam_flat<K, 16><<<grid, block, 0, stream>>>(
            x, qs, start_state, d_out, G, qs_bytes, edge, parent);
    }
    return cudaGetLastError();
}

// ============================================================
// Public entry point.

extern "C" cudaError_t trellis_encode_group_cuda(
        const float * x,
        int           K,
        int           G,
        uint8_t     * qs,
        uint16_t    * start_state,
        float       * d,
        cudaStream_t  stream) {
    if (G <= 0) return cudaSuccess;
    if (K != 2 && K != 3 && K != 4) return cudaErrorInvalidValue;

    // Ensure LUT is initialised on this device (idempotent).
    GGML_CUDA_INIT_TRELLIS_TABLE_IMPL();

    const int qs_bytes = (VTQ_ENC_N * K + 7) / 8;
    const int beam     = trellis_cuda_encode_beam_width();

    if (beam == 0) {
        // -------------------- Full Viterbi --------------------
        const vtq_encode_workspace * ws = vtq_get_encode_workspace(stream);
        const int pool_slots = ws->pool_slots;

        const int64_t num_waves = (G + pool_slots - 1) / pool_slots;
        for (int64_t w = 0; w < num_waves; w++) {
            const int64_t wave_start = w * pool_slots;
            const int64_t remaining  = G - wave_start;
            const int     wave_n     = (int)((remaining < pool_slots) ? remaining : pool_slots);

            dim3 grid(wave_n, 1, 1);
            dim3 block(VTQ_ENC_THREADS, 1, 1);

            switch (K) {
                case 2:
                    k_trellis_encode_receiver_flat<2><<<grid, block, 0, stream>>>(
                        x, qs, start_state, d, (int64_t)G, qs_bytes,
                        ws->dp_cur, ws->dp_next, ws->bt, pool_slots, wave_start);
                    break;
                case 3:
                    k_trellis_encode_receiver_flat<3><<<grid, block, 0, stream>>>(
                        x, qs, start_state, d, (int64_t)G, qs_bytes,
                        ws->dp_cur, ws->dp_next, ws->bt, pool_slots, wave_start);
                    break;
                case 4:
                    k_trellis_encode_receiver_flat<4><<<grid, block, 0, stream>>>(
                        x, qs, start_state, d, (int64_t)G, qs_bytes,
                        ws->dp_cur, ws->dp_next, ws->bt, pool_slots, wave_start);
                    break;
            }
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return err;

            if (num_waves > 1) {
                cudaError_t serr = cudaStreamSynchronize(stream);
                if (serr != cudaSuccess) return serr;
            }
        }
        return cudaSuccess;
    }

    // -------------------- Beam search --------------------
    int device = 0;
    cudaGetDevice(&device);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) device = 0;

    // Round beam up/down to the nearest supported template instantiation.
    int B_inst;
    if      (beam >= 256) B_inst = 256;
    else if (beam >= 128) B_inst = 128;
    else if (beam >= 64)  B_inst = 64;
    else if (beam >= 32)  B_inst = 32;
    else                  B_inst = 16;

    const size_t bytes_per_buf = (size_t)G * (size_t)VTQ_ENC_N * (size_t)B_inst;
    uint8_t * edge = nullptr, * parent = nullptr;
    cudaError_t err = ensure_beam_scratch(device, bytes_per_buf, &edge, &parent);
    if (err != cudaSuccess) return err;

    switch (K) {
        case 2: return launch_beam<2>(beam, G, x, qs, start_state, d, qs_bytes, edge, parent, stream);
        case 3: return launch_beam<3>(beam, G, x, qs, start_state, d, qs_bytes, edge, parent, stream);
        case 4: return launch_beam<4>(beam, G, x, qs, start_state, d, qs_bytes, edge, parent, stream);
    }
    return cudaErrorInvalidValue;
}
