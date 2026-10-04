#include "fattn-vec-gqa.cuh"

// Decode with a TurboQuant K or V cache and grouped-query attention: one block per GQA group instead of one per
// Q head (see fattn-vec-gqa.cuh). Returns false if the shape or the types are not covered.

#define FATTN_VEC_GQA_CASE(D_, type_K_, type_V_)                                                  \
    if (D == (D_) && K->type == (type_K_) && V->type == (type_V_)) {                              \
        ggml_cuda_flash_attn_ext_vec_gqa_case<D_, ncols, type_K_, type_V_>(ctx, dst, use_sparse); \
        return true;                                                                              \
    }

#define FATTN_VEC_GQA_CASES_V(D_, type_K_)                       \
    FATTN_VEC_GQA_CASE(D_, type_K_, GGML_TYPE_VTQ2_1)            \
    FATTN_VEC_GQA_CASE(D_, type_K_, GGML_TYPE_VTQ3_1)            \
    FATTN_VEC_GQA_CASE(D_, type_K_, GGML_TYPE_VTQ4_1)

bool ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    static const bool disabled = getenv("GGML_CUDA_TQ_VEC_NO_GQA") != nullptr;
    if (disabled) {
        return false;
    }

    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || gqa_ratio <= 2 || !mask || sinks || max_bias != 0.0f ||
            K->ne[1] % FATTN_KQ_STRIDE != 0 || Q->ne[0] != V->ne[0]) {
        return false;
    }

    // sparse attention (the mask selects at most n_kv_max cells per query): visit only the selected rows
    // once the context is clearly longer than the selection
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const bool use_sparse = n_kv_max > 0 && mask->ne[0] == K->ne[1] && mask->ne[2] == 1 && K->ne[1] >= 2*int64_t(n_kv_max);

    constexpr int ncols = 8;
    const int64_t D = Q->ne[0];

    FATTN_VEC_GQA_CASE   (256, GGML_TYPE_KTQ2_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASE   (256, GGML_TYPE_KTQ3_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASE   (256, GGML_TYPE_KTQ4_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASES_V(256, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASES_V(256, GGML_TYPE_KTQ2_1)
    FATTN_VEC_GQA_CASES_V(256, GGML_TYPE_KTQ3_1)
    FATTN_VEC_GQA_CASES_V(256, GGML_TYPE_KTQ4_1)

    FATTN_VEC_GQA_CASE   (128, GGML_TYPE_KTQ2_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASE   (128, GGML_TYPE_KTQ3_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASE   (128, GGML_TYPE_KTQ4_1, GGML_TYPE_F16)
    FATTN_VEC_GQA_CASES_V(128, GGML_TYPE_KTQ2_1)
    FATTN_VEC_GQA_CASES_V(128, GGML_TYPE_KTQ3_1)
    FATTN_VEC_GQA_CASES_V(128, GGML_TYPE_KTQ4_1)

    return false;
}
