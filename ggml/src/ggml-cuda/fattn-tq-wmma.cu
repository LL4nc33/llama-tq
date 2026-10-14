#include "fattn-tq-wmma.cuh"

// Tensor-core flash attention for quantized KV caches (fattn-tq-wmma.cuh). Returns false if the shape or the types
// are not covered.

// the kernels are compiled in template-instances/fattn-tq-wmma-instance-*.cu, one TU per (head size, columns)
static bool fattn_tq_wmma_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int64_t D) {
    switch (D) {
        case  64: return ggml_cuda_flash_attn_ext_tq_wmma_d< 64, 16>(ctx, dst);
        case 128: return ggml_cuda_flash_attn_ext_tq_wmma_d<128, 16>(ctx, dst);
        case 256: return ggml_cuda_flash_attn_ext_tq_wmma_d<256, 16>(ctx, dst);
        case 512: return ggml_cuda_flash_attn_ext_tq_wmma_d<512, 16>(ctx, dst);
        default:  return false;
    }
}

bool ggml_cuda_flash_attn_ext_tq_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_TQ_WMMA");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING) {
        return false;
    }

    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || gqa_ratio < 2 || gqa_ratio > 16 || !mask || max_bias != 0.0f ||
            n_kv_max > 0 || K->ne[1] % 64 != 0 || Q->ne[0] != V->ne[0]) {
        return false;
    }
    return fattn_tq_wmma_dispatch(ctx, dst, Q->ne[0]);
}
