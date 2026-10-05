// MMA-KTQ entry point: TurboQuant K/V are dequantized to f16 scratch buffers and the f16 MMA kernel runs
// unchanged (split-dequant).

#include "fattn-mma-ktq.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

// Dequantizes a TurboQuant K or V view into a contiguous f16 scratch buffer and points the tensor at
// it; restore() puts the original view back.
struct ggml_cuda_tq_f16_view {
    ggml_tensor * t = nullptr;
    ggml_type     type;
    void *        data;
    size_t        nb[GGML_MAX_DIMS];

    void dequantize(ggml_backend_cuda_context & ctx, ggml_tensor * tensor, ggml_cuda_pool_alloc<half> & scratch) {
        if (tensor->type == GGML_TYPE_F16) {
            return;
        }
        to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(tensor->type);
        GGML_ASSERT(to_fp16 != nullptr);

        // the dequantize_block_*_nc kernels take strides in block units, not bytes
        const size_t ts = ggml_type_size(tensor->type);
        GGML_ASSERT(tensor->nb[0] == ts);
        const int64_t ne0 = tensor->ne[0], ne1 = tensor->ne[1], ne2 = tensor->ne[2], ne3 = tensor->ne[3];
        scratch.alloc(ne0*ne1*ne2*ne3);
        to_fp16(tensor->data, scratch.get(), ne0, ne1, ne2, ne3, tensor->nb[1]/ts, tensor->nb[2]/ts, tensor->nb[3]/ts, ctx.stream());

        t    = tensor;
        type = tensor->type;
        data = tensor->data;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nb[i] = tensor->nb[i];
        }
        tensor->type  = GGML_TYPE_F16;
        tensor->data  = scratch.get();
        tensor->nb[0] = sizeof(half);
        tensor->nb[1] = tensor->nb[0]*ne0;
        tensor->nb[2] = tensor->nb[1]*ne1;
        tensor->nb[3] = tensor->nb[2]*ne2;
    }

    void restore() {
        if (t == nullptr) {
            return;
        }
        t->type = type;
        t->data = data;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = nb[i];
        }
    }
};

// Split-dequant path for batches: TurboQuant K (KTQ) and/or V (VTQ) are expanded to f16 and the
// tensor-core MMA kernel runs on the copies.
static void ggml_cuda_flash_attn_ext_mma_ktq_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    ggml_cuda_pool_alloc<half> k_scratch(ctx.pool());
    ggml_cuda_pool_alloc<half> v_scratch(ctx.pool());
    ggml_cuda_tq_f16_view k_view, v_view;
    k_view.dequantize(ctx, K, k_scratch);
    v_view.dequantize(ctx, V, v_scratch);

    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    v_view.restore();
    k_view.restore();
}

void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_mma_ktq_split(ctx, dst);
}
