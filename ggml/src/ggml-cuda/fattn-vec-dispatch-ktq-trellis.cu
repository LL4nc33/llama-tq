// Dispatch helper for the trellis K types (ktq2_3s, ktq3_3s, ktq4_3s) in ggml_cuda_flash_attn_ext_vec: heads 128, 256
// and 512, V f16, q8_0, q5_0, q4_0, vtq1/2/3/4_1 or vtq2/3/4_2s. Other pairs convert K (and V) to f16 for the MMA
// kernel. Not split by GGML_CUDA_FA_ALL_QUANTS: these are the pairs trellis K is used with.

#include "fattn-vec-dispatch.cuh"

bool try_dispatch_vec_ktq_trellis(ggml_backend_cuda_context * ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    FATTN_VEC_CASES_KTQ_TRELLIS_V_RET(GGML_TYPE_KTQ2_3S)
    FATTN_VEC_CASES_KTQ_TRELLIS_V_RET(GGML_TYPE_KTQ3_3S)
    FATTN_VEC_CASES_KTQ_TRELLIS_V_RET(GGML_TYPE_KTQ4_3S)

    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V);
    return false;
}
