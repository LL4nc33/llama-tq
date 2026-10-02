#include "common.cuh"
#include "dsv4-hc.cuh"

// the strides are passed in elements, so the kernels also accept non-contiguous views

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);

    cudaStream_t stream = ctx.stream();

    if (gated) {
        dsv4_hc_pre_f32<true><<<grid_dims, block_dims, 0, stream>>>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float),
                scale);
    } else {
        dsv4_hc_pre_f32<false><<<grid_dims, block_dims, 0, stream>>>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float),
                scale);
    }
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);

    cudaStream_t stream = ctx.stream();

    if (comb) {
        dsv4_hc_post_f32<true><<<grid_dims, block_dims, 0, stream>>>(
                (const float *) x->data, (const float *) residual->data,
                (const float *) post->data, (const float *) comb->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float),
                nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbp0 / sizeof(float), nbp1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
    } else {
        dsv4_hc_post_f32<false><<<grid_dims, block_dims, 0, stream>>>(
                (const float *) x->data, (const float *) residual->data,
                (const float *) post->data, nullptr, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float),
                nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbp0 / sizeof(float), nbp1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
    }
}
