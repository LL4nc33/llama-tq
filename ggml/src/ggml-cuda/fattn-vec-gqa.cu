#include "fattn-vec-gqa.cuh"

// Decode with a quantized K or V cache and grouped-query attention: one block per GQA group instead of one per
// Q head (see fattn-vec-gqa.cuh). Returns false if the shape or the types are not covered.

// the kernels are compiled in template-instances/fattn-gqa-instance-*.cu, one TU per (head size, ncols, K type)
template <ggml_type type_K>
static bool fattn_gqa_dispatch_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int64_t D, const int ncols, const bool use_sparse) {
    if (D == 128) {
        return ncols == 4 ? ggml_cuda_flash_attn_ext_vec_gqa_d<128, 4, type_K>(ctx, dst, use_sparse) :
                            ggml_cuda_flash_attn_ext_vec_gqa_d<128, 8, type_K>(ctx, dst, use_sparse);
    }
    return ncols == 4 ? ggml_cuda_flash_attn_ext_vec_gqa_d<256, 4, type_K>(ctx, dst, use_sparse) :
                        ggml_cuda_flash_attn_ext_vec_gqa_d<256, 8, type_K>(ctx, dst, use_sparse);
}

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

    // with a single KQ tile there is nothing to split across the GQA blocks: the per-head VEC kernel is faster there
    // (Qwen3-4B q8_0 decode, RTX 2060: 87.3 vs 85.8 t/s at 256 cells, 83.7 vs 85.4 at 512)
    if (!use_sparse && K->ne[1] <= FATTN_KQ_STRIDE) {
        return false;
    }

    // GQA groups of up to 4 Q heads fit one 4-column block, larger groups use blocks of 8.
    const int64_t D = Q->ne[0];
    if (D != 128 && D != 256) {
        return false;
    }
    const int ncols = gqa_ratio <= 4 ? 4 : 8;
    switch (K->type) {
        case GGML_TYPE_F16:    return fattn_gqa_dispatch_k<GGML_TYPE_F16>   (ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_Q8_0:   return fattn_gqa_dispatch_k<GGML_TYPE_Q8_0>  (ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_Q5_0:   return fattn_gqa_dispatch_k<GGML_TYPE_Q5_0>  (ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_Q4_0:   return fattn_gqa_dispatch_k<GGML_TYPE_Q4_0>  (ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_KTQ2_1: return fattn_gqa_dispatch_k<GGML_TYPE_KTQ2_1>(ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_KTQ3_1: return fattn_gqa_dispatch_k<GGML_TYPE_KTQ3_1>(ctx, dst, D, ncols, use_sparse);
        case GGML_TYPE_KTQ4_1: return fattn_gqa_dispatch_k<GGML_TYPE_KTQ4_1>(ctx, dst, D, ncols, use_sparse);
        default:               return false;
    }
}
