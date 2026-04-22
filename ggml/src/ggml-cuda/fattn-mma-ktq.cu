// MMA-KTQ entry point. Dispatches on (DKQ, DV, ncols1, ncols2) matching what
// the f16 MMA dispatcher would have chosen, but to KTQ-aware template instances
// where available. Falls back to split-dequant for non-instantiated shapes.

#include "fattn-mma-ktq.cuh"
#include "fattn-mma-ktq-inline.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_ktq_inline_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static void ggml_cuda_flash_attn_ext_mma_ktq_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * K = dst->src[1];

    to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
    GGML_ASSERT(to_fp16 != nullptr);

    const int64_t ne00 = K->ne[0];
    const int64_t ne01 = K->ne[1];
    const int64_t ne02 = K->ne[2];
    const int64_t ne03 = K->ne[3];
    const int64_t scratch_elems = ne00 * ne01 * ne02 * ne03;

    const int64_t s01 = K->nb[1];
    const int64_t s02 = K->nb[2];
    const int64_t s03 = K->nb[3];

    ggml_cuda_pool_alloc<half> k_scratch(ctx.pool(), scratch_elems);
    to_fp16(K->data, k_scratch.get(), ne00, ne01, ne02, ne03, s01, s02, s03, ctx.stream());

    const ggml_type saved_type = K->type;
    void * const    saved_data = K->data;
    const size_t    saved_nb0  = K->nb[0];
    const size_t    saved_nb1  = K->nb[1];
    const size_t    saved_nb2  = K->nb[2];
    const size_t    saved_nb3  = K->nb[3];

    K->type = GGML_TYPE_F16;
    K->data = k_scratch.get();
    K->nb[0] = sizeof(half);
    K->nb[1] = K->nb[0] * ne00;
    K->nb[2] = K->nb[1] * ne01;
    K->nb[3] = K->nb[2] * ne02;

    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    K->type = saved_type;
    K->data = saved_data;
    K->nb[0] = saved_nb0;
    K->nb[1] = saved_nb1;
    K->nb[2] = saved_nb2;
    K->nb[3] = saved_nb3;
}

void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Inline path: DKQ=DV=128, GQA ratio 4, KTQ2_1 K + f16 V (Ministral-3 family).
    {
        static int e = 0;
        if (e++ < 5) {
            fprintf(stderr, "[KTQ-ENTRY#%d] K=%d V=%d D=%d gqa=%d Qne1=%lld\n",
                    e, (int)K->type, (int)V->type, (int)Q->ne[0],
                    (int)(Q->ne[2]/K->ne[2]), (long long)Q->ne[1]);
            fflush(stderr);
        }
    }
    if (K->type == GGML_TYPE_KTQ2_1 && V->type == GGML_TYPE_F16 &&
        Q->ne[0] == 128 && V->ne[0] == 128) {
        const int gqa_ratio = Q->ne[2] / K->ne[2];
        if (gqa_ratio == 4) {
            static int i = 0;
            if (i++ < 3) {
                fprintf(stderr, "[KTQ-INLINE] Qne1=%lld\n", (long long)Q->ne[1]);
                fflush(stderr);
            }
            constexpr int ncols2 = 4;
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 1, ncols2>(ctx, dst);
                return;
            } else if (Q->ne[1] <= 2) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 2, ncols2>(ctx, dst);
                return;
            } else if (Q->ne[1] <= 4) {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 4, ncols2>(ctx, dst);
                return;
            } else {
                ggml_cuda_flash_attn_ext_mma_ktq_inline_case<128, 128, 8, ncols2>(ctx, dst);
                return;
            }
        }
    }

    // Fallback: split-dequant.
    ggml_cuda_flash_attn_ext_mma_ktq_split(ctx, dst);
}
