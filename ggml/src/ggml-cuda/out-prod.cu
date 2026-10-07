#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || ggml_is_contiguous(src0));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t lda = nb01 / sizeof(float);
    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    int64_t                 ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));
    // When the leading dim collapses (e.g. src1 inner dim==1 for a gate/scalar
    // gradient), nb11 == nb10 == sizeof(float) yields ldb=1. cuBLAS requires
    // ldb >= N (= ne1) when opB=CUBLAS_OP_T (B stored as N rows x K cols) and
    // ldb >= K (= ne01) when opB=CUBLAS_OP_N (B stored as K rows x N cols).
    // Raising ldb beyond that would read B with a wrong stride.
    const int64_t ldb_min = src1_T ? ne01 : ne1;
    if (ldb < ldb_min) {
        ldb = ldb_min;
    }

    // data strides in dimensions 2/3
    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (src0->type != GGML_TYPE_F32) {
        // quantized src0 (the frozen weight in the input gradient of a matmul during LoRA training): dequantize
        // its rows to f32 in chunks and accumulate dst += a[:, chunk] * b[:, chunk]^T, so that a large weight
        // (e.g. the output projection) never needs a full f32 copy
        const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32 != nullptr);
        const int64_t rows_per_chunk = std::max<int64_t>(1, std::min<int64_t>(ne01, (int64_t(64) << 20) / ne00)); // 256 MiB of f32
        ggml_cuda_pool_alloc<float> src0_f32(ctx.pool(), rows_per_chunk*ne00);

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const char * src0_slice = (const char *) src0->data + (i3/dps3)*nb03 + (i2/dps2)*nb02;
                for (int64_t r0 = 0; r0 < ne01; r0 += rows_per_chunk) {
                    const int64_t nr = std::min(rows_per_chunk, ne01 - r0);
                    to_fp32(src0_slice + r0*nb01, src0_f32.get(), nr*ne00, stream);
                    const float beta_chunk = r0 == 0 ? 0.0f : 1.0f;
                    CUBLAS_CHECK(
                        cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                                ne0, ne1, nr,
                                &alpha, src0_f32.get(), ne00,
                                        src1_d + i3*s13 + i2*s12 + (src1_T ? r0 : r0*ldb), ldb,
                                &beta_chunk, dst_d + i3*s3 + i2*s2, ldc));
                }
            }
        }
        return;
    }

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }
}
