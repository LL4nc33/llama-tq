// Dispatch helper for the VTQ_2 family slice of ggml_cuda_flash_attn_ext_vec.
// Covers V=VTQ2_2, VTQ3_2, VTQ4_2 (Trellis v2).
//
// Split from fattn-vec-dispatch-vtq.cu so VTQ_1 / VTQ_2 / VTQ_3 family cases
// compile in parallel. Each TU instantiates a disjoint slice of the
// (D, type_K, type_V) template graph.

// The cases themselves live in one TU per V type (fattn-vec-dispatch-vtq<bits>_<family>.cu).
#include "fattn-vec-dispatch.cuh"

// Phase 3A1 (E11 CUDA port, 2026-04-21):
// DISPATCH-HOOK DISABLED — measured 60x slowdown vs legacy on D=128 benches
// (1.6 tok/s on Qwen3-0.6B vs expected ~100). Kernel source kept in
// fattn-vec-vtq2.cuh for future ncu diagnosis.
// Re-enable with -DFATTN_VTQ2_CACHED_ENABLE=1 only after kernel rewrite.
#if defined(FATTN_VTQ2_CACHED) && defined(FATTN_VTQ2_CACHED_ENABLE)
#include "fattn-vec-vtq2.cuh"

#define FATTN_VEC_VTQ2_CACHED_CASE_RET(D_val, type_K, type_V)                                                    \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D_val) && type_K_okay && type_V_okay && Q->ne[1] == 1) {                                \
            ggml_cuda_flash_attn_ext_vec_vtq2_case<D_val, type_K, type_V>(ctx, dst);                             \
            return true;                                                                                         \
        }                                                                                                        \
    }
#endif  // FATTN_VTQ2_CACHED && FATTN_VTQ2_CACHED_ENABLE

bool try_dispatch_vec_vtq2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#if defined(FATTN_VTQ2_CACHED) && defined(FATTN_VTQ2_CACHED_ENABLE)
    // Phase 3A1 hot path — disabled by default (see header above).
    FATTN_VEC_VTQ2_CACHED_CASE_RET(128, GGML_TYPE_KTQ2_1, GGML_TYPE_VTQ3_2)
#endif

    if (try_dispatch_vec_vtq2_2(ctx, dst)) return true;
    if (try_dispatch_vec_vtq3_2(ctx, dst)) return true;
    if (try_dispatch_vec_vtq4_2(ctx, dst)) return true;

    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V);
    return false;
}
