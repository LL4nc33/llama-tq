#include "mul-mat-id-grad-as.cuh"

// dst:    grad_as  [D_in, D_out, n_expert]  (mirrors original 'as' shape)
// src[0]: grad_c   [D_out, n_used,    n_tokens]
// src[1]: b        [D_in,  n_used_b,  n_tokens]
// src[2]: ids      [n_used, n_tokens]  i32
//
// grad_as[c, r, k] = sum over (e, t) where ids[e, t] == k of
//                        b[c, e mod n_used_b, t] * grad_c[r, e, t]
//
// One block per (c-block, r-block, k). Each thread accumulates one (c, r) entry
// over all (e, t) pairs in serial. No atomics needed because each (c, r, k) cell
// has exactly one writer thread.
__global__ void k_mul_mat_id_grad_as(
        const float * __restrict__ grad_c,
        const float * __restrict__ b,
        const int32_t * __restrict__ ids,
        float       * __restrict__ dst,
        const int64_t D_in,
        const int64_t D_out,
        const int64_t n_expert,
        const int64_t n_used,
        const int64_t n_used_b,
        const int64_t n_tokens,
        const int64_t grad_c_stride_e,
        const int64_t grad_c_stride_t,
        const int64_t b_stride_e,
        const int64_t b_stride_t,
        const int64_t dst_stride_r,
        const int64_t dst_stride_k) {

    const int64_t k = blockIdx.z;                              // expert
    const int64_t r = blockIdx.y * blockDim.y + threadIdx.y;   // D_out
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;   // D_in

    if (c >= D_in || r >= D_out || k >= n_expert) {
        return;
    }

    float acc = 0.0f;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_used; ++e) {
            const int32_t k_route = ids[e + t*n_used];
            if (k_route != (int32_t) k) {
                continue;
            }
            const int64_t e_b = e % n_used_b;
            const float grad_v = grad_c[r + e*grad_c_stride_e + t*grad_c_stride_t];
            const float b_v    = b[c + e_b*b_stride_e + t*b_stride_t];
            acc += grad_v * b_v;
        }
    }

    dst[c + r*dst_stride_r + k*dst_stride_k] = acc;
}

void ggml_cuda_op_mul_mat_id_grad_as(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad_c = dst->src[0];
    const ggml_tensor * b      = dst->src[1];
    const ggml_tensor * ids    = dst->src[2];

    GGML_ASSERT(grad_c->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type      == GGML_TYPE_F32);
    GGML_ASSERT(ids->type    == GGML_TYPE_I32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);

    const int64_t D_in     = dst->ne[0];
    const int64_t D_out    = dst->ne[1];
    const int64_t n_expert = dst->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_used_b = b->ne[1];

    GGML_ASSERT(grad_c->ne[0] == D_out);
    GGML_ASSERT(grad_c->ne[1] == n_used);
    GGML_ASSERT(grad_c->ne[2] == n_tokens);
    GGML_ASSERT(b->ne[0]      == D_in);
    GGML_ASSERT(b->ne[2]      == n_tokens);
    GGML_ASSERT(n_used % n_used_b == 0);

    const float   * grad_c_d = (const float   *) grad_c->data;
    const float   * b_d      = (const float   *) b->data;
    const int32_t * ids_d    = (const int32_t *) ids->data;
    float         * dst_d    = (float         *) dst->data;

    const int64_t grad_c_stride_e = grad_c->nb[1] / sizeof(float);
    const int64_t grad_c_stride_t = grad_c->nb[2] / sizeof(float);
    const int64_t b_stride_e      = b->nb[1]      / sizeof(float);
    const int64_t b_stride_t      = b->nb[2]      / sizeof(float);
    const int64_t dst_stride_r    = dst->nb[1]    / sizeof(float);
    const int64_t dst_stride_k    = dst->nb[2]    / sizeof(float);

    GGML_ASSERT(dst->nb[0]    == sizeof(float));
    GGML_ASSERT(grad_c->nb[0] == sizeof(float));
    GGML_ASSERT(b->nb[0]      == sizeof(float));

    cudaStream_t stream = ctx.stream();

    const dim3 block(16, 16, 1);
    const dim3 grid(
        (D_in     + block.x - 1) / block.x,
        (D_out    + block.y - 1) / block.y,
        (unsigned int) n_expert);

    k_mul_mat_id_grad_as<<<grid, block, 0, stream>>>(
        grad_c_d, b_d, ids_d, dst_d,
        D_in, D_out, n_expert, n_used, n_used_b, n_tokens,
        grad_c_stride_e, grad_c_stride_t,
        b_stride_e, b_stride_t,
        dst_stride_r, dst_stride_k);

    CUDA_CHECK(cudaGetLastError());
}
