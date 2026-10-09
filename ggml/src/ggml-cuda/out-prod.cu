#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>

// fp16 tensor-core path for a quantized src0 (the frozen weight in the input gradient of a matmul):
// the gradient src1 is scaled by a power of two so that its largest value lands near 2^14 before the f16
// conversion (as loss scaling in mixed precision training), the GEMM accumulates in f32 and alpha = 1/scale
// undoes the scaling. scale, 1/scale and the beta values live in device memory, so nothing synchronizes.

// src1 is read as rows of contiguous elements (cols along the stride-1 dimension), one block per row at a time
static __global__ void out_prod_absmax_f32(const float * x, const int nrows, const int ncols,
        const int64_t sr, const int64_t sc, unsigned int * absmax_bits) {
    float m = 0.0f;
    for (int r = blockIdx.x; r < nrows; r += gridDim.x) {
        const float * row = x + r*sr;
        for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
            m = fmaxf(m, fabsf(row[c*sc]));
        }
    }
    m = warp_reduce_max(m);
    __shared__ float smax[32];
    if ((threadIdx.x % WARP_SIZE) == 0) {
        smax[threadIdx.x / WARP_SIZE] = m;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        m = threadIdx.x < blockDim.x / WARP_SIZE ? smax[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            atomicMax(absmax_bits, __float_as_uint(m)); // non-negative floats order like their bit patterns
        }
    }
}

static __global__ void out_prod_scales(const unsigned int * absmax_bits, float * scales) {
    const float m = __uint_as_float(*absmax_bits);
    const float s = m > 0.0f && isfinite(m) ? exp2f(floorf(log2f(16384.0f / m))) : 1.0f;
    scales[0] = s;
    scales[1] = 1.0f / s;
    scales[2] = 0.0f;
    scales[3] = 1.0f;
}

// src1 -> f16 * scale in the layout the GEMM reads: row r, column c at y[c + r*ncols] (rows = ne11 and
// cols = ne10, or for a transposed src1 rows = ne10 and cols = ne11)
static __global__ void out_prod_src1_to_f16(const float * x, half * y, const int nrows, const int ncols,
        const int64_t sr, const int64_t sc, const float * scales) {
    const float scale = scales[0];
    for (int r = blockIdx.x; r < nrows; r += gridDim.x) {
        const float * row = x + r*sr;
        half * out = y + (int64_t) r*ncols;
        for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
            out[c] = __float2half(row[c*sc] * scale);
        }
    }
}

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
        // its rows in chunks and accumulate dst += a[:, chunk] * b[:, chunk]^T, so that a large weight
        // (e.g. the output projection) never needs a full copy. GGML_CUDA_OUT_PROD_F32=1 keeps the f32 SGEMM.
        static const bool force_f32 = getenv("GGML_CUDA_OUT_PROD_F32") != nullptr;
        const to_fp16_cuda_t to_fp16 = force_f32 ? nullptr : ggml_get_to_fp16_cuda(src0->type);
        if (to_fp16 != nullptr) {
            const int64_t rows_per_chunk = std::max<int64_t>(1, std::min<int64_t>(ne01, (int64_t(128) << 20) / ne00)); // 256 MiB of f16
            ggml_cuda_pool_alloc<half>         src0_f16(ctx.pool(), rows_per_chunk*ne00);
            ggml_cuda_pool_alloc<half>         src1_f16(ctx.pool(), ne10*ne11);
            ggml_cuda_pool_alloc<float>        scales(ctx.pool(), 4);
            ggml_cuda_pool_alloc<unsigned int> absmax(ctx.pool(), 1);
            const int64_t s10 = nb10 / sizeof(float);
            const int64_t s11 = nb11 / sizeof(float);
            const int64_t ldb16 = src1_T ? ne11 : ne10;
            const cublasOperation_t op16 = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    const float * src1_slice = src1_d + i3*s13 + i2*s12;
                    CUDA_CHECK(cudaMemsetAsync(absmax.get(), 0, sizeof(unsigned int), stream));
                    // rows along the strided dimension, columns along the contiguous one
                    const int     nrows = src1_T ? ne10 : ne11;
                    const int     ncols = src1_T ? ne11 : ne10;
                    const int64_t sr    = src1_T ? s10  : s11;
                    const int64_t sc    = src1_T ? s11  : s10;
                    const int nblocks = std::min(nrows, 4096);
                    out_prod_absmax_f32<<<nblocks, 256, 0, stream>>>(src1_slice, nrows, ncols, sr, sc, absmax.get());
                    out_prod_scales<<<1, 1, 0, stream>>>(absmax.get(), scales.get());
                    out_prod_src1_to_f16<<<nblocks, 256, 0, stream>>>(src1_slice, src1_f16.get(), nrows, ncols, sr, sc, scales.get());
                    const char * src0_slice = (const char *) src0->data + (i3/dps3)*nb03 + (i2/dps2)*nb02;
                    for (int64_t r0 = 0; r0 < ne01; r0 += rows_per_chunk) {
                        const int64_t nr = std::min(rows_per_chunk, ne01 - r0);
                        to_fp16(src0_slice + r0*nb01, src0_f16.get(), nr*ne00, stream);
                        CUBLAS_CHECK(
                            cublasGemmEx(handle, CUBLAS_OP_N, op16,
                                    ne0, ne1, nr,
                                    scales.get() + 1, src0_f16.get(), CUDA_R_16F, ne00,
                                                      src1_f16.get() + (src1_T ? r0 : r0*ldb16), CUDA_R_16F, ldb16,
                                    scales.get() + (r0 == 0 ? 2 : 3), dst_d + i3*s3 + i2*s2, CUDA_R_32F, ldc,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    }
                }
            }
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            return;
        }

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

    // one strided batched GEMM per (i3, position in the GQA group): src0 head j serves dst heads j*dps2 + k
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t k = 0; k < dps2; ++k) {
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03,          lda, s02,
                                src1_d +  i3      *s13 + k*s12,  ldb, dps2*s12,
                        &beta,  dst_d  +  i3      *s3  + k*s2,   ldc, dps2*s2,
                        ne02));
        }
    }
}
