// MMA-KTQ entry point. Dispatches on (DKQ, DV, ncols1, ncols2) matching what
// the f16 MMA dispatcher would have chosen, but to KTQ-aware template instances
// where available. Falls back to split-dequant for non-instantiated shapes.

#include "fattn-mma-ktq.cuh"
#include "fattn-mma-ktq-inline.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

// Default value for V_is_vtq2_1 is defined in fattn-mma-ktq-inline.cuh; forward-decl must not redefine it.
template <int DKQ, int DV, int ncols1, int ncols2, bool V_is_vtq2_1>
void ggml_cuda_flash_attn_ext_mma_ktq_inline_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Helper: dequant a quantized cache tensor into an f16 scratch buffer in-place
// (mutating tensor metadata so the downstream MMA-f16 kernel sees it as f16).
// Returns saved metadata so the caller can restore after the kernel call.
struct fattn_cache_save {
    ggml_type type;
    void *    data;
    size_t    nb[4];
};

static void fattn_ktq_dequant_to_scratch(ggml_backend_cuda_context & ctx,
                                         ggml_tensor * t,
                                         ggml_cuda_pool_alloc<half> & scratch,
                                         fattn_cache_save & saved) {
    to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
    GGML_ASSERT(to_fp16 != nullptr);

    const int64_t ne00 = t->ne[0];
    const int64_t ne01 = t->ne[1];
    const int64_t ne02 = t->ne[2];
    const int64_t ne03 = t->ne[3];
    const int64_t scratch_elems = ne00 * ne01 * ne02 * ne03;

    // dequantize_block_*_nc kernels expect strides in block units, not bytes.
    const size_t ts = ggml_type_size(t->type);
    GGML_ASSERT(t->nb[0] == ts);
    const int64_t s01 = t->nb[1] / ts;
    const int64_t s02 = t->nb[2] / ts;
    const int64_t s03 = t->nb[3] / ts;

    scratch.alloc(ctx.pool(), scratch_elems);
    to_fp16(t->data, scratch.get(), ne00, ne01, ne02, ne03, s01, s02, s03, ctx.stream());

    saved.type   = t->type;
    saved.data   = t->data;
    saved.nb[0]  = t->nb[0];
    saved.nb[1]  = t->nb[1];
    saved.nb[2]  = t->nb[2];
    saved.nb[3]  = t->nb[3];

    t->type  = GGML_TYPE_F16;
    t->data  = scratch.get();
    t->nb[0] = sizeof(half);
    t->nb[1] = t->nb[0] * ne00;
    t->nb[2] = t->nb[1] * ne01;
    t->nb[3] = t->nb[2] * ne02;
}

static void fattn_cache_restore(ggml_tensor * t, const fattn_cache_save & saved) {
    t->type  = saved.type;
    t->data  = saved.data;
    t->nb[0] = saved.nb[0];
    t->nb[1] = saved.nb[1];
    t->nb[2] = saved.nb[2];
    t->nb[3] = saved.nb[3];
}

static void ggml_cuda_flash_attn_ext_mma_ktq_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    // K is always quantized in this fallback.
    ggml_cuda_pool_alloc<half> k_scratch;
    fattn_cache_save k_saved;
    fattn_ktq_dequant_to_scratch(ctx, K, k_scratch, k_saved);

    // V may be quantized (VTQ-family) or already f16; only dequant if needed.
    // This lets KTQ+VTQ shapes that don't fit the inline kernel (e.g. D=256 GQA=8
    // for Qwen3.6-A35-A3B) still benefit from the MMA-f16 prefill kernel instead
    // of falling back to the slower vec path.
    ggml_cuda_pool_alloc<half> v_scratch;
    fattn_cache_save           v_saved;
    const bool                 v_needs_dequant = V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_BF16;
    if (v_needs_dequant) {
        fattn_ktq_dequant_to_scratch(ctx, V, v_scratch, v_saved);
    }

    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    fattn_cache_restore(K, k_saved);
    if (v_needs_dequant) {
        fattn_cache_restore(V, v_saved);
    }
}

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

    // Phase 3: KTQ2_1 K + VTQ2_1 V inline path (Ministral-3 prefill at long ctx).
    // Avoids the full-K f16 dequant scratch buffer that the split fallback allocates.
    if (K->type == GGML_TYPE_KTQ2_1 && V->type == GGML_TYPE_VTQ2_1 &&
        Q->ne[0] == 128 && V->ne[0] == 128) {
        const int gqa_ratio = Q->ne[2] / K->ne[2];
        if (gqa_ratio == 4) {
            constexpr int ncols2 = 4;
            if (Q->ne[1] >= 8) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 8, ncols2, /*V_is_vtq2_1=*/true>(ctx, dst);
                return;
            } else if (Q->ne[1] >= 4) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 4, ncols2, /*V_is_vtq2_1=*/true>(ctx, dst);
                return;
            }
        }
    }

    // Fallback: split-dequant.
    ggml_cuda_flash_attn_ext_mma_ktq_split(ctx, dst);
}
