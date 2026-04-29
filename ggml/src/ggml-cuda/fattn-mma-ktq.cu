// MMA-KTQ entry point. Dispatches on (DKQ, DV, ncols1, ncols2) matching what
// the f16 MMA dispatcher would have chosen, but to KTQ-aware template instances
// where available. Falls back to split-dequant for non-instantiated shapes.

#include "fattn-mma-ktq.cuh"
#include "fattn-mma-ktq-inline.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_ktq_inline_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Inline path: DKQ=DV=128, GQA ratio 4, KTQ2_1 K + f16 V (Ministral-3 family).
    if (K->type == GGML_TYPE_KTQ2_1 && V->type == GGML_TYPE_F16 &&
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
