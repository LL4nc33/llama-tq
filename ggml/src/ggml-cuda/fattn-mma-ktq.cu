// MMA-KTQ entry point. Dispatches on (DKQ, DV, ncols1, ncols2) matching what
// the f16 MMA dispatcher would have chosen, but to KTQ-aware template instances
// where available. Falls back to split-dequant for non-instantiated shapes.

#include "fattn-mma-ktq.cuh"
#include "fattn-mma-ktq-inline.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_ktq_inline_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

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
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Inline path: DKQ=DV=128, GQA ratio 4, KTQ2_1 K + f16 V (Ministral-3 family).
    // Its output does not match flash attention over the dequantized K (relative error ~0.8 at
    // 32 queries), while the split-dequant path below matches exactly, so it is opt-in for
    // debugging only (GGML_CUDA_KTQ_INLINE=1).
    static const bool inline_enabled = [] {
        const char * env = getenv("GGML_CUDA_KTQ_INLINE");
        return env != nullptr && atoi(env) == 1;
    }();
    if (inline_enabled && K->type == GGML_TYPE_KTQ2_1 && V->type == GGML_TYPE_F16 &&
        Q->ne[0] == 128 && V->ne[0] == 128) {
        const int gqa_ratio = Q->ne[2] / K->ne[2];
        if (gqa_ratio == 4) {
            constexpr int ncols2 = 4;
            // Only ncols1 ∈ {4, 8} instantiated — smaller configs have degenerate
            // MMA tile dimensions. For Q->ne[1] < 4 fall back to split-dequant.
            if (Q->ne[1] >= 8) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 8, ncols2>(ctx, dst);
                return;
            } else if (Q->ne[1] >= 4) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 4, ncols2>(ctx, dst);
                return;
            }
        }
    }

    // Fallback: split-dequant.
    ggml_cuda_flash_attn_ext_mma_ktq_split(ctx, dst);
}
