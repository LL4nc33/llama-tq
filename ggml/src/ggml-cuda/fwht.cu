#include "common.cuh"
#include "fwht.cuh"

// signs (optional): a sign per input column, applied before the transform; a row of the transform
// covers N of the width columns, so its signs start at (r*N) % width
template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale,
        const float * signs, const int64_t width, const float * post_signs) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    if (signs) {
        const float * s = signs + (r * N) % width;
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] = src[i * warp_size + lane] * s[i * warp_size + lane] * scale;
        }
    } else {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            reg[i] = src[i * warp_size + lane] * scale;
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    if (post_signs) {
        const float * s = post_signs + (r * N) % width;
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            dst[i * warp_size + lane] = reg[i] * s[i * warp_size + lane];
        }
    } else {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            dst[i * warp_size + lane] = reg[i];
        }
    }
}

// Wide rows: one block of FWHT_BLOCK_THREADS per row instead of one warp, which serialises every stage
// on a single warp. Element i*NT + tid lives in reg[i]: the stages inside a warp use shuffles, the ones
// across warps go through shared memory, the remaining ones are between registers of a thread.
#define FWHT_BLOCK_THREADS 256

template <int N, int NT>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const float * src, float * dst, const int64_t n_rows, const float scale,
        const float * signs, const int64_t width, const float * post_signs) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    float reg[NE];
    if (signs) {
        const float * sr = signs + (r * N) % width;
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = src[i * NT + tid] * sr[i * NT + tid] * scale;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = src[i * NT + tid] * scale;
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    if (post_signs) {
        const float * sr = post_signs + (r * N) % width;
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            dst[i * NT + tid] = reg[i] * sr[i * NT + tid];
        }
    } else {
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            dst[i * NT + tid] = reg[i];
        }
    }
}

// post_signs: optional signs (same layout as signs) applied to the result, for D*H*D in one pass
static bool ggml_cuda_fwht_launch(ggml_backend_cuda_context & ctx, const float * src_d, float * dst_d,
        const int n, const int64_t rows, const float * signs_d, const int64_t width, const float * post_signs_d = nullptr) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t stream = ctx.stream();
    dim3         grid_dims(num_blocks, 1, 1);
    dim3         block_dims(warp_size, rows_per_block, 1);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            fwht_cuda<64><<<grid_dims, block_dims, 0, stream>>>(src_d, dst_d, rows, scale, signs_d, width, post_signs_d);
            return true;
        case 128:
            fwht_cuda<128><<<grid_dims, block_dims, 0, stream>>>(src_d, dst_d, rows, scale, signs_d, width, post_signs_d);
            return true;
        case 256:
            fwht_cuda<256><<<grid_dims, block_dims, 0, stream>>>(src_d, dst_d, rows, scale, signs_d, width, post_signs_d);
            return true;
        case 512:
            fwht_cuda_block<512, FWHT_BLOCK_THREADS><<<rows, FWHT_BLOCK_THREADS, 0, stream>>>(src_d, dst_d, rows, scale, signs_d, width, post_signs_d);
            return true;
        case 1024:
            fwht_cuda_block<1024, FWHT_BLOCK_THREADS><<<rows, FWHT_BLOCK_THREADS, 0, stream>>>(src_d, dst_d, rows, scale, signs_d, width, post_signs_d);
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    return ggml_cuda_fwht_launch(ctx, (const float *) src->data, (float *) dst->data,
            src->ne[0], ggml_nrows(src), nullptr, 1);
}

bool ggml_cuda_op_fwht_signs(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * signs,
        const ggml_tensor * src1, ggml_tensor * dst) {
    // x [width, ...] times signs [width], then the transform over rows of src1->ne[0]
    const int64_t n     = src1->ne[0];
    const int64_t width = signs->ne[0];
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst) ||
            x->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ggml_nrows(signs) != 1 || x->ne[0] != width || width % n != 0 ||
            ggml_nelements(x) != ggml_nelements(dst) || !ggml_are_same_shape(src1, dst)) {
        return false;
    }
    return ggml_cuda_fwht_launch(ctx, (const float *) x->data, (float *) dst->data,
            n, ggml_nrows(dst), (const float *) signs->data, width);
}

bool ggml_cuda_op_fwht_signs2(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * signs,
        const ggml_tensor * src1, const ggml_tensor * post_signs, ggml_tensor * dst) {
    // signs * x, the transform over rows of src1->ne[0], times the same signs again (dst has the shape of x)
    const int64_t n     = src1->ne[0];
    const int64_t width = signs->ne[0];
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(signs) || !ggml_is_contiguous(post_signs) || !ggml_is_contiguous(dst) ||
            x->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 || post_signs->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || ggml_nrows(signs) != 1 || ggml_nrows(post_signs) != 1 ||
            post_signs->ne[0] != width || x->ne[0] != width || dst->ne[0] != width || width % n != 0 ||
            ggml_nelements(x) != ggml_nelements(dst) || ggml_nelements(src1) != ggml_nelements(dst)) {
        return false;
    }
    return ggml_cuda_fwht_launch(ctx, (const float *) x->data, (float *) dst->data,
            n, ggml_nelements(dst) / n, (const float *) signs->data, width, (const float *) post_signs->data);
}
