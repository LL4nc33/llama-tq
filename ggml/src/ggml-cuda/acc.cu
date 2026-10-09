#include "acc.cuh"

// dst = src0, then dst viewed with strides s1, s2, s3 (elements) at offset += src1. src1 may be strided (nb10..nb13
// in elements) and the view strides may be in any order (e.g. a permuted view in the backward pass), so the
// kernel walks the elements of src1, as the CPU implementation does.
static __global__ void acc_f32(const float * y, float * dst,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne10*ne11*ne12*ne13) {
        return;
    }

    const int64_t i10 = i % ne10;
    const int64_t i11 = (i / ne10) % ne11;
    const int64_t i12 = (i / (ne10*ne11)) % ne12;
    const int64_t i13 = i / (ne10*ne11*ne12);

    dst[offset + i10 + i11*s1 + i12*s2 + i13*s3] += y[i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13];
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / sizeof(float);
    const int64_t s2     = dst->op_params[1] / sizeof(float);
    const int64_t s3     = dst->op_params[2] / sizeof(float);
    const int64_t offset = dst->op_params[3] / sizeof(float);

    if (dst_d != src0_d) {
        CUDA_CHECK(cudaMemcpyAsync(dst_d, src0_d, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
    }

    const size_t fs = sizeof(float);
    GGML_ASSERT(src1->nb[0] % fs == 0 && src1->nb[1] % fs == 0 && src1->nb[2] % fs == 0 && src1->nb[3] % fs == 0);
    const int64_t n1 = ggml_nelements(src1);
    const int num_blocks = (n1 + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    acc_f32<<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(src1_d, dst_d,
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0]/fs, src1->nb[1]/fs, src1->nb[2]/fs, src1->nb[3]/fs, s1, s2, s3, offset);
}
