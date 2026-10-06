#include "mul-mat-id-grad-b.cuh"
#include "convert.cuh"

#include <vector>

// G[:, j] = grad_c[:, e_j, t_j] for the routing pairs j of one expert
static __global__ void k_grad_b_gather(
        const float * __restrict__ grad_c, const int32_t * __restrict__ pairs, float * __restrict__ G,
        const int64_t D_out, const int64_t s_e, const int64_t s_t, const int64_t n_used) {
    const int64_t j = blockIdx.y;
    const int32_t p = pairs[j];
    const int64_t e = p % n_used;
    const int64_t t = p / n_used;
    for (int64_t r = blockIdx.x*blockDim.x + threadIdx.x; r < D_out; r += gridDim.x*blockDim.x) {
        G[j*D_out + r] = grad_c[e*s_e + t*s_t + r];
    }
}

// dst[:, e_j mod n_used_b, t_j] += Y[:, j]; the top-k experts of a token are distinct, so within one expert no two
// pairs hit the same dst row, and experts are processed one after another on the stream
static __global__ void k_grad_b_scatter(
        const float * __restrict__ Y, const int32_t * __restrict__ pairs, float * __restrict__ dst,
        const int64_t D_in, const int64_t d_e, const int64_t d_t, const int64_t n_used, const int64_t n_used_b) {
    const int64_t j = blockIdx.y;
    const int32_t p = pairs[j];
    const int64_t e = p % n_used;
    const int64_t t = p / n_used;
    float * out = dst + (e % n_used_b)*d_e + t*d_t;
    for (int64_t c = blockIdx.x*blockDim.x + threadIdx.x; c < D_in; c += gridDim.x*blockDim.x) {
        out[c] += Y[j*D_in + c];
    }
}

bool ggml_cuda_mul_mat_id_grad_b_supported(const ggml_tensor * dst) {
    const ggml_tensor * as = dst->src[0];
    return dst->src[1]->type == GGML_TYPE_F32 && dst->src[2]->type == GGML_TYPE_I32 && ggml_is_contiguous(as) &&
           (as->type == GGML_TYPE_F32 || ggml_get_to_fp32_cuda(as->type) != nullptr);
}

void ggml_cuda_op_mul_mat_id_grad_b(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as     = dst->src[0];
    const ggml_tensor * grad_c = dst->src[1];
    const ggml_tensor * ids    = dst->src[2];

    GGML_ASSERT(ggml_cuda_mul_mat_id_grad_b_supported(dst));
    GGML_ASSERT(grad_c->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int64_t D_in     = as->ne[0];
    const int64_t D_out    = as->ne[1];
    const int64_t n_expert = as->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_used_b = dst->ne[1];

    cudaStream_t stream = ctx.stream();

    // routing to the host, grouped by expert
    std::vector<int32_t> ids_host(n_used*n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data() + t*n_used, (const char *) ids->data + t*ids->nb[1],
            n_used*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<int32_t>> by_expert(n_expert);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_used; ++e) {
            const int32_t k = ids_host[t*n_used + e];
            GGML_ASSERT(k >= 0 && k < n_expert);
            by_expert[k].push_back((int32_t) (t*n_used + e));
        }
    }
    std::vector<int32_t> pairs_host;
    std::vector<int64_t> offset(n_expert + 1, 0);
    int64_t n_max = 0;
    for (int64_t k = 0; k < n_expert; ++k) {
        offset[k] = pairs_host.size();
        pairs_host.insert(pairs_host.end(), by_expert[k].begin(), by_expert[k].end());
        n_max = std::max<int64_t>(n_max, by_expert[k].size());
    }
    offset[n_expert] = pairs_host.size();

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    if (pairs_host.empty()) {
        return;
    }

    ggml_cuda_pool_alloc<int32_t> pairs(ctx.pool(), pairs_host.size());
    CUDA_CHECK(cudaMemcpyAsync(pairs.get(), pairs_host.data(), pairs_host.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const bool as_f32 = as->type == GGML_TYPE_F32;
    ggml_cuda_pool_alloc<float> w_f32(ctx.pool());
    if (!as_f32) {
        w_f32.alloc(D_in*D_out);
    }
    ggml_cuda_pool_alloc<float> G(ctx.pool(), n_max*D_out);
    ggml_cuda_pool_alloc<float> Y(ctx.pool(), n_max*D_in);

    const to_fp32_cuda_t to_fp32 = as_f32 ? nullptr : ggml_get_to_fp32_cuda(as->type);
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int64_t k = 0; k < n_expert; ++k) {
        const int64_t n_k = offset[k + 1] - offset[k];
        if (n_k == 0) {
            continue;
        }
        const char  * w_k = (const char *) as->data + k*as->nb[2];
        const float * W   = (const float *) w_k;
        if (!as_f32) {
            to_fp32(w_k, w_f32.get(), D_in*D_out, stream);
            W = w_f32.get();
        }
        const int32_t * pairs_k = pairs.get() + offset[k];

        k_grad_b_gather<<<dim3((D_out + 255)/256, n_k), 256, 0, stream>>>(
            (const float *) grad_c->data, pairs_k, G.get(), D_out,
            grad_c->nb[1]/sizeof(float), grad_c->nb[2]/sizeof(float), n_used);

        // Y [D_in, n_k] = W [D_in, D_out] * G [D_out, n_k] (column-major: W columns are the rows r of the expert)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D_in, n_k, D_out,
            &alpha, W, D_in, G.get(), D_out, &beta, Y.get(), D_in));

        k_grad_b_scatter<<<dim3((D_in + 255)/256, n_k), 256, 0, stream>>>(
            Y.get(), pairs_k, (float *) dst->data, D_in,
            dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), n_used, n_used_b);
    }
    CUDA_CHECK(cudaGetLastError());
}
