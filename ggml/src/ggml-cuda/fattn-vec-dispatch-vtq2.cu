// Dispatch helper for the VTQ_2 family slice of ggml_cuda_flash_attn_ext_vec.
// Covers V=VTQ2_2, VTQ3_2, VTQ4_2 (Trellis v2).
//
// Split from fattn-vec-dispatch-vtq.cu so VTQ_1 and VTQ_2 family cases
// compile in parallel (each ~22min instead of combined ~45min).

#include "fattn-vec-dispatch.cuh"

bool try_dispatch_vec_vtq2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

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
