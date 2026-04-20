// Dispatch helper for the VTQ_2 family slice of ggml_cuda_flash_attn_ext_vec.
// Covers V=VTQ2_2, VTQ3_2, VTQ4_2 (Trellis v2).
//
// Split from fattn-vec-dispatch-vtq.cu so VTQ_1 and VTQ_2 family cases
// compile in parallel (each ~22min instead of combined ~45min).

#include "fattn-vec-dispatch.cuh"

// Phase 3A1 (E11 CUDA port, 2026-04-21):
// When FATTN_VTQ2_CACHED is defined at compile time, route the
// KTQ2_1 × VTQ3_2 combo at D=128, ncols=1 through the new warp-cooperative
// cached-decode kernel (flash_attn_ext_vec_vtq2_cached). All other combos
// still fall through to the legacy path via FATTN_VEC_CASES_ALL_D_WITH_512_RET.
//
// Spec: docs/plans/2026-04-21-e11-cuda-port-spec.md
#ifdef FATTN_VTQ2_CACHED
#include "fattn-vec-vtq2.cuh"

// Intentionally constrained: only a single (D, K, V, ncols) combo is routed
// to the cached kernel in Phase 3A1 so the cicc instance count stays low.
// ncols=1 guarded via Q->ne[1] == 1.
#define FATTN_VEC_VTQ2_CACHED_CASE_RET(D_val, type_K, type_V)                                                    \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D_val) && type_K_okay && type_V_okay && Q->ne[1] == 1) {                                \
            ggml_cuda_flash_attn_ext_vec_vtq2_case<D_val, type_K, type_V>(ctx, dst);                             \
            return true;                                                                                         \
        }                                                                                                        \
    }
#endif  // FATTN_VTQ2_CACHED

bool try_dispatch_vec_vtq2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef FATTN_VTQ2_CACHED
    // Phase 3A1 hot path. Only KTQ2_1 × VTQ3_2 at D=128, ncols=1 rerouted.
    // Everything else falls through to the legacy macro expansion below.
    FATTN_VEC_VTQ2_CACHED_CASE_RET(128, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ3_2)
#endif

    // VTQ_2 reduced matrix — same under GGML_CUDA_FA_ALL_QUANTS or default.
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,    GGML_TYPE_VTQ2_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,    GGML_TYPE_VTQ3_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,    GGML_TYPE_VTQ4_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,   GGML_TYPE_VTQ2_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,   GGML_TYPE_VTQ3_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,   GGML_TYPE_VTQ4_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ2_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ3_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ4_2)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_VTQ3_2)

    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V);
    return false;
}
