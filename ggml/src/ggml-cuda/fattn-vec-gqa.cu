#include "fattn-vec-gqa.cuh"

// Decode with a TurboQuant K or V cache and grouped-query attention: one block per GQA group instead of one per
// Q head (see fattn-vec-gqa.cuh). Returns false if the shape or the types are not covered.

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

    // GQA groups of up to 4 Q heads fit one 4-column block, larger groups use blocks of 8.
    // The kernels are compiled in template-instances/fattn-gqa-instance-*.cu, one TU per (head size, ncols).
    const int64_t D = Q->ne[0];
    if (gqa_ratio <= 4) {
        if (D == 256) return ggml_cuda_flash_attn_ext_vec_gqa_d<256, 4>(ctx, dst, use_sparse);
        if (D == 128) return ggml_cuda_flash_attn_ext_vec_gqa_d<128, 4>(ctx, dst, use_sparse);
        return false;
    }
    if (D == 256) return ggml_cuda_flash_attn_ext_vec_gqa_d<256, 8>(ctx, dst, use_sparse);
    if (D == 128) return ggml_cuda_flash_attn_ext_vec_gqa_d<128, 8>(ctx, dst, use_sparse);
    return false;
}
