// MMA-KTQ split-dequant path.
//
// Strategy: re-use the existing KTQ→fp16 bulk dequant (convert.cu) and the
// existing MMA-F16 tensor-core FA kernel. Eliminates the PP regression that
// stems from fattn.cu routing KTQ unconditionally to the VEC kernel (which
// supports only cols_per_block ∈ {1, 2}, i.e. no tensor-core parallelism on
// PP batches).
//
// Trade-off: one extra K memory pass per FA call to stage an fp16 scratch
// buffer. That cost is amortized over the entire PP prefill batch, so the
// tensor-core win dominates. For TG (ne[1] ≤ 2) the dispatcher in fattn.cu
// keeps routing to VEC, so this path is prefill-only.

#include "fattn-mma-ktq.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"

void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * K = dst->src[1];

    to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
    GGML_ASSERT(to_fp16 != nullptr && "MMA-KTQ: no KTQ→fp16 bulk dequant registered for K->type");

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
