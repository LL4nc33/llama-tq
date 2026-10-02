// PTQ1_0 mat-vec on a planar-transposed q8_1 activation layout (plain 2D MUL_MAT, 1..8 columns).
//
// Why: the generic mmvq kernel gives every thread one 128-weight PTQ1_0 block and reads the
// activations as 32 scattered 4-byte loads per column out of 36-byte block_q8_1 structs, so every
// extra column costs a full pass over the activations. It also maps the 128 threads of a block onto
// 128 K blocks of the same row, so a K = 5120 projection (40 blocks per row) keeps only 31% of the
// lanes busy.
//
// Here the activations of one column are stored so that the 128 quants a thread needs for one K
// block are 8 aligned 16-byte pieces, one per plane, and the 4 per-32 (d, isum) scales are one more
// 16-byte piece. Adjacent threads read adjacent pieces of the same plane. The work items are
// (row group, K block) pairs flattened into one index space, each thread handles ROWS rows per item
// so every activation piece serves ROWS rows, one fp32 partial per (row, column, K block) goes to
// shared memory and one thread per (row, column) sums them in an order that depends only on the
// weight shape. A column's result is therefore the same bits for every column count 1..8.
//
// PT layout, per activation column (sizes for the row padded to MATRIX_ROW_PADDING, nblk blocks):
//   plane t (t = 0..7): nblk * 16 bytes, byte b of block kb is the quant of element kb*128 + t*16 + b
//   plane 8:            nblk * 16 bytes, block kb holds 4 words, one per 32-element sub-block:
//                       low half = d (fp16), high half = exact int16 sum of the 32 quants
// The column stride is 9 * nblk * 16 = padded_row * 9/8 bytes, exactly the block_q8_1 stride, so
// the pool allocation and the column strides in block_q8_1 units stay valid.
//
// Integer part: sum((q - 1) * a) is computed as sum(q * a) - sum(a) from the raw digits {0, 1, 2}
// and the exact quant sum, the same integer as the generic vec_dot_ptq1_0_q8_1. Only the fp32
// summation order over sub-blocks and K blocks differs from the generic kernel.
//
// CUDA only. HIP and MUSA keep the generic block_q8_1 path (ptq1_0_pt_can_use() returns false).
#pragma once

#include "common.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_PTQ1_0_PT_AVAILABLE
#endif

#if defined(GGML_CUDA_PTQ1_0_PT_AVAILABLE)

#define PTQ1_0_PT_THREADS     128
#define PTQ1_0_PT_MAX_ROWS    16
#define PTQ1_0_PT_MAX_COLS    8    // equals MMVQ_MAX_BATCH_SIZE
#define PTQ1_0_PT_SMEM_FLOATS 4096 // 16 KiB target when choosing rows per CTA; one item may request more

// rows one thread handles per item: 4 up to 4 columns, 2 beyond (register pressure)
static constexpr __host__ __device__ int ptq1_0_pt_rows_per_item(const int ncols_dst) {
    return ncols_dst <= 4 ? 4 : 2;
}

// rows per CTA: fill whole 128-thread iterations where possible, within the shared memory target
static int ptq1_0_pt_rows_per_cta(const int blocks_per_row, const int ncols_dst) {
    const int rows_per_item = ptq1_0_pt_rows_per_item(ncols_dst);
    int rmax = PTQ1_0_PT_SMEM_FLOATS / (ncols_dst * (blocks_per_row + 1));
    rmax = rmax < rows_per_item ? rows_per_item : (rmax > PTQ1_0_PT_MAX_ROWS ? PTQ1_0_PT_MAX_ROWS : rmax);
    rmax -= rmax % rows_per_item;
    int    best      = rows_per_item;
    double best_util = 0.0;
    for (int r = rows_per_item; r <= rmax; r += rows_per_item) {
        const int    items = (r / rows_per_item) * blocks_per_row;
        const int    iters = (items + PTQ1_0_PT_THREADS - 1) / PTQ1_0_PT_THREADS;
        const double util  = (double) items / (double) (iters * PTQ1_0_PT_THREADS);
        if (util > best_util + 1e-9) {
            best_util = util;
            best      = r;
        }
        if (util > 0.999) {
            break;
        }
    }
    return best;
}

// dynamic shared memory of one CTA: one fp32 partial per (column, row, K block + 1 pad), doubled
// with gate fusion. The odd row stride keeps the per-pair epilogue reads bank-conflict-free.
static size_t ptq1_0_pt_smem_bytes(const int blocks_per_row, const int ncols_dst, const bool has_gate) {
    const int rows_per_cta = ptq1_0_pt_rows_per_cta(blocks_per_row, ncols_dst);
    return (size_t) ncols_dst * rows_per_cta * (blocks_per_row + 1) * sizeof(float) * (has_gate ? 2 : 1);
}

// true when ggml_cuda_mul_mat_vec_q routes this call to the planar kernel. The quantizer and the
// kernel launch are both chosen from this one decision, so the layout always matches the consumer.
static bool ptq1_0_pt_can_use(
        const ggml_type type_x, const bool has_ids, const int64_t ncols_x, const int64_t ncols_dst,
        const int64_t nchannels_x, const int64_t nsamples_x, const int64_t nchannels_dst, const int64_t nsamples_dst,
        const bool has_gate) {
    if (type_x != GGML_TYPE_PTQ1_0 || has_ids || ncols_x % QK_PTQ1_0 != 0 ||
            ncols_dst < 1 || ncols_dst > PTQ1_0_PT_MAX_COLS ||
            nchannels_x != 1 || nsamples_x != 1 || nchannels_dst != 1 || nsamples_dst != 1) {
        return false;
    }
    const size_t smem = ptq1_0_pt_smem_bytes((int) (ncols_x / QK_PTQ1_0), (int) ncols_dst, has_gate);
    return smem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
}

// ---------------------------------------------------------------------------------------------
// Quantizer: same per-32 quantization as quantize_q8_1 (same d, same q), written in the PT layout
// with the exact int16 quant sum instead of the float input sum. ne0 must be a multiple of 128.
// ---------------------------------------------------------------------------------------------
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1_ptq1_0_pt(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const float xi = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0.0f;
    float amax = fabsf(xi);

    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int64_t row_cont = (i3*ne2.z + i2) * ne1 + i1;
    char * ycol = (char *) vy + row_cont * (ne0 * 9 / 8); // same column stride as block_q8_1
    const int64_t nblk = ne0 / QK_PTQ1_0;
    const int64_t kb   = i0 / QK_PTQ1_0;
    const int     e    = (int) (i0 % QK_PTQ1_0);
    ycol[((e / 16)*nblk + kb) * 16 + (e % 16)] = q;

    int isum = q;
    isum = warp_reduce_sum<QK8_1>(isum);

    if (i0 % QK8_1 != 0) {
        return;
    }

    // |isum| <= 32*127 fits int16: low half d as fp16, high half the raw int16 sum
    const half     dh   = __float2half(d);
    const uint32_t bits = (uint32_t) *((const uint16_t *) &dh) | ((uint32_t) (uint16_t) (int16_t) isum << 16);
    uint32_t * ds = (uint32_t *) (ycol + 8*nblk*16) + kb*4 + e / QK8_1;
    *ds = bits;
}

static void quantize_row_q8_1_ptq1_0_pt_cuda(
        const float * x, void * vy, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne0 % QK_PTQ1_0 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_ptq1_0_pt<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
}

// ---------------------------------------------------------------------------------------------
// Block dot product
// ---------------------------------------------------------------------------------------------

static __device__ __forceinline__ int ptq1_0_pt_int4_at(const int4 & v, const int k) {
    switch (k & 3) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

// one base-3 digit step on four bytes held in the low bytes of two 16-bit-lane words
// (3*255 < 2^16, no carry between lanes): returns the raw digits {0, 1, 2} of the four bytes and
// advances the remainders
static __device__ __forceinline__ uint32_t ptq1_0_pt_trit_step(uint32_t & vlo, uint32_t & vhi) {
    const uint32_t wlo = vlo * 3;
    const uint32_t whi = vhi * 3;
    vlo = wlo & 0x00FF00FF;
    vhi = whi & 0x00FF00FF;
    return __byte_perm(wlo, whi, 0x7531);
}

// fold the finished integer sums of sub-block k (elements 32k..32k+31) into the fp32 accumulators:
// acc += d8_k * (sum(q*a) - sum(a)), then reset the integer sums
template <int ncols, int nrows>
static __device__ __forceinline__ void ptq1_0_pt_fold(
        const int4 (&dsraw)[ncols], const int k, int (&sumi)[ncols][nrows], float (&acc)[ncols][nrows]) {
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const int   w    = ptq1_0_pt_int4_at(dsraw[j], k);
        const float d8   = __low2float(*((const half2 *) &w));
        const int   isum = w >> 16; // arithmetic shift: the sign-extended int16 quant sum
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            acc[j][i]  = __fmaf_rn(d8, (float) (sumi[j][i] - isum), acc[j][i]);
            sumi[j][i] = 0;
        }
    }
}

// Dot products of nrows PTQ1_0 blocks with the same K block of ncols activation columns.
// bq[i] points at the weight block of row i, ycol[j] at the PT base of column j.
// Element order of block_ptq1_0: qs[0..15] trit t of byte m -> element 16*t + m,
// qs[16..23] trit t of byte m -> 80 + 8*t + m, qh[h] trit t -> 120 + 2*t + h.
template <int ncols, int nrows>
static __device__ __forceinline__ void ptq1_0_pt_block_dot(
        const block_ptq1_0 * const (&bq)[nrows], const char * const (&ycol)[ncols],
        const int kbx, const int nblk, float (&result)[ncols][nrows]) {
    int4 dsraw[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        dsraw[j] = *((const int4 *) ycol[j] + 8*nblk + kbx);
    }

    int   sumi[ncols][nrows];
    float acc[ncols][nrows];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            sumi[j][i] = 0;
            acc[j][i]  = 0.0f;
        }
    }

    // qs[0..15]: word g holds bytes 4g..4g+3, trit t of them is element 16*t + 4*g + b,
    // i.e. word g of plane t
    uint32_t vlo[nrows][4];
    uint32_t vhi[nrows][4];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t packed = get_int_b4(bq[i]->qs, g);
            vlo[i][g] = __byte_perm(packed, 0, 0x4140); // [b0, 0, b1, 0]
            vhi[i][g] = __byte_perm(packed, 0, 0x4342); // [b2, 0, b3, 0]
        }
    }
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        int4 u[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            u[j] = *((const int4 *) ycol[j] + t*nblk + kbx);
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int q = (int) ptq1_0_pt_trit_step(vlo[i][g], vhi[i][g]);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    // digits are 0..2, so the signed dp4a gives the same result as u8 x s8
                    sumi[j][i] = ggml_cuda_dp4a(q, ptq1_0_pt_int4_at(u[j], g), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            ptq1_0_pt_fold<ncols, nrows>(dsraw, 0, sumi, acc); // elements 0..31 done
        }
        if (t == 3) {
            ptq1_0_pt_fold<ncols, nrows>(dsraw, 1, sumi, acc); // elements 32..63 done
        }
    }

    // qs[16..23]: element 80 + 8*t + 4*g + b -> word 20 + 2*t + g, planes 5, 6 and the lower half of 7
    uint32_t vlo2[nrows][2];
    uint32_t vhi2[nrows][2];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t packed = get_int_b4(bq[i]->qs + 16, g);
            vlo2[i][g] = __byte_perm(packed, 0, 0x4140);
            vhi2[i][g] = __byte_perm(packed, 0, 0x4342);
        }
    }
    int4 u2[ncols];
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        if (t % 2 == 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                u2[j] = *((const int4 *) ycol[j] + (5 + t/2)*nblk + kbx);
            }
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int q = (int) ptq1_0_pt_trit_step(vlo2[i][g], vhi2[i][g]);
                const int w = 20 + 2*t + g; // word index within the 128-element block
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a(q, ptq1_0_pt_int4_at(u2[j], w & 3), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            ptq1_0_pt_fold<ncols, nrows>(dsraw, 2, sumi, acc); // elements 64..95 done (words 16..23)
        }
    }

    // qh: element 120 + 2*t + h -> words 30 and 31, the upper half of plane 7 that u2 still holds.
    // qh[0] and qh[1] sit in the low bytes of the two 16-bit lanes; two steps give the digits of
    // elements 120+2t, 121+2t, 122+2t, 123+2t in byte order.
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
        uint32_t v = (uint32_t) bq[i]->qh[0] | ((uint32_t) bq[i]->qh[1] << 16);
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
            const uint32_t w0 = v * 3;
            v                 = w0 & 0x00FF00FF;
            const uint32_t w1 = v * 3;
            v                 = w1 & 0x00FF00FF;
            const int q = (int) __byte_perm(w0, w1, 0x7531);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sumi[j][i] = ggml_cuda_dp4a(q, ptq1_0_pt_int4_at(u2[j], 2 + t/2), sumi[j][i]);
            }
        }
    }
    ptq1_0_pt_fold<ncols, nrows>(dsraw, 3, sumi, acc); // elements 96..127 done

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            result[j][i] = __fmul_rn((float) bq[i]->d, acc[j][i]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------

template <int ncols, int ROWS, bool has_fusion, bool has_gate>
__launch_bounds__(PTQ1_0_PT_THREADS, (ncols <= 2 ? 4 : (ncols <= 4 ? 3 : 2)))
static __global__ void mul_mat_vec_ptq1_0_pt(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int rows_per_cta, const uint3 bpr_fd, const uint3 rpc_fd) {
    extern __shared__ float ptq1_0_pt_partials[]; // [ncols][rows_per_cta][bprp], then the gate partials

    const int bpr  = ncols_x / QK_PTQ1_0; // K blocks per row
    const int bprp = bpr + 1;             // odd partials row stride
    float * partials      = ptq1_0_pt_partials;
    float * partials_gate = ptq1_0_pt_partials + ncols*rows_per_cta*bprp;
    (void) partials_gate;

    // plane stride of the PT layout: blocks in the row padded to MATRIX_ROW_PADDING
    const int nblk = ((ncols_x + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * (MATRIX_ROW_PADDING / QK_PTQ1_0);
    const int row0 = rows_per_cta * blockIdx.x;
    const int tid  = threadIdx.x;

    // rows this CTA really owns (the last CTA may be short); clamped item rows reread the last real row
    const int n_rows_cta = min(rows_per_cta, nrows_x - row0);

    const char * ycol[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        ycol[j] = (const char *) ((const block_q8_1 *) vy + j*stride_col_y);
    }

    const int n_items = (rows_per_cta / ROWS) * bpr;
    for (int idx = tid; idx < n_items; idx += PTQ1_0_PT_THREADS) {
        const int rg  = (int) fastdiv((uint32_t) idx, bpr_fd); // row group within the CTA
        const int kbx = idx - rg*bpr;

        const block_ptq1_0 * bq[ROWS];
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            int r = rg*ROWS + i;
            r = r < n_rows_cta ? r : n_rows_cta - 1; // clamp the tail, that result is not written
            bq[i] = (const block_ptq1_0 *) vx + (int64_t) (row0 + r)*stride_row_x + kbx;
        }
        float dots[ncols][ROWS];
        ptq1_0_pt_block_dot<ncols, ROWS>(bq, ycol, kbx, nblk, dots);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                partials[(j*rows_per_cta + rg*ROWS + i)*bprp + kbx] = dots[j][i];
            }
        }
        if constexpr (has_gate) {
            const block_ptq1_0 * bg[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                int r = rg*ROWS + i;
                r = r < n_rows_cta ? r : n_rows_cta - 1;
                bg[i] = (const block_ptq1_0 *) fusion.gate + (int64_t) (row0 + r)*stride_row_x + kbx;
            }
            ptq1_0_pt_block_dot<ncols, ROWS>(bg, ycol, kbx, nblk, dots);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    partials_gate[(j*rows_per_cta + rg*ROWS + i)*bprp + kbx] = dots[j][i];
                }
            }
        }
    }

    __syncthreads();

    // one thread per (row, column): four interleaved accumulators (K block mod 4), then (s0+s1)+(s2+s3).
    // The order depends only on the weight shape, never on the column count.
    for (int p = tid; p < rows_per_cta*ncols; p += PTQ1_0_PT_THREADS) {
        const int j   = (int) fastdiv((uint32_t) p, rpc_fd); // column
        const int r   = p - j*rows_per_cta;                   // row within the CTA
        const int row = row0 + r;
        const float * src  = partials      + (j*rows_per_cta + r)*bprp;
        const float * srcg = partials_gate + (j*rows_per_cta + r)*bprp;
        (void) srcg;

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        int kbx = 0;
        for (; kbx + 4 <= bpr; kbx += 4) {
            s0 += src[kbx + 0]; s1 += src[kbx + 1]; s2 += src[kbx + 2]; s3 += src[kbx + 3];
            if constexpr (has_gate) {
                g0 += srcg[kbx + 0]; g1 += srcg[kbx + 1]; g2 += srcg[kbx + 2]; g3 += srcg[kbx + 3];
            }
        }
        for (; kbx < bpr; ++kbx) {
            s0 += src[kbx];
            if constexpr (has_gate) {
                g0 += srcg[kbx];
            }
        }
        const float sum      = (s0 + s1) + (s2 + s3);
        const float sum_gate = (g0 + g1) + (g2 + g3);
        (void) sum_gate;

        if (row < nrows_x) {
            float result = sum;
            if constexpr (has_fusion) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[j*stride_col_dst + row];
                }
                if constexpr (has_gate) {
                    float gate_value = sum_gate;
                    if (fusion.gate_bias) {
                        gate_value += ((const float *) fusion.gate_bias)[j*stride_col_dst + row];
                    }
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + row] = result;
        }
    }
}

template <int ncols>
static void mul_mat_vec_ptq1_0_pt_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    constexpr int ROWS = ptq1_0_pt_rows_per_item(ncols);
    const int  bpr        = ncols_x / QK_PTQ1_0;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_gate   = fusion.gate != nullptr;

    const int  rows_per_cta = ptq1_0_pt_rows_per_cta(bpr, ncols);
    const uint3 bpr_fd = init_fastdiv_values((uint64_t) bpr);
    const uint3 rpc_fd = init_fastdiv_values((uint64_t) rows_per_cta);
    const dim3 block_nums((nrows_x + rows_per_cta - 1) / rows_per_cta, 1, 1);
    const dim3 block_dims(PTQ1_0_PT_THREADS, 1, 1);
    const size_t smem = ptq1_0_pt_smem_bytes(bpr, ncols, has_gate);

    if constexpr (ncols == 1) {
        if (has_fusion) {
            if (has_gate) {
                mul_mat_vec_ptq1_0_pt<ncols, ROWS, true, true><<<block_nums, block_dims, smem, stream>>>
                    (vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_fd, rpc_fd);
            } else {
                mul_mat_vec_ptq1_0_pt<ncols, ROWS, true, false><<<block_nums, block_dims, smem, stream>>>
                    (vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_fd, rpc_fd);
            }
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_ptq1_0_pt<ncols, ROWS, false, false><<<block_nums, block_dims, smem, stream>>>
        (vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_fd, rpc_fd);
}

// vy must have been written by quantize_row_q8_1_ptq1_0_pt_cuda; call only when ptq1_0_pt_can_use() is true
static void mul_mat_vec_ptq1_0_pt_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_ptq1_0_pt_launch<1>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 2: mul_mat_vec_ptq1_0_pt_launch<2>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_ptq1_0_pt_launch<3>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_ptq1_0_pt_launch<4>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_ptq1_0_pt_launch<5>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_ptq1_0_pt_launch<6>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_ptq1_0_pt_launch<7>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_ptq1_0_pt_launch<8>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#else // defined(GGML_CUDA_PTQ1_0_PT_AVAILABLE)

static bool ptq1_0_pt_can_use(
        const ggml_type type_x, const bool has_ids, const int64_t ncols_x, const int64_t ncols_dst,
        const int64_t nchannels_x, const int64_t nsamples_x, const int64_t nchannels_dst, const int64_t nsamples_dst,
        const bool has_gate) {
    GGML_UNUSED_VARS(type_x, has_ids, ncols_x, ncols_dst, nchannels_x, nsamples_x, nchannels_dst, nsamples_dst, has_gate);
    return false;
}

// never reached: ptq1_0_pt_can_use() is false on HIP/MUSA
static void quantize_row_q8_1_ptq1_0_pt_cuda(
        const float * x, void * vy, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_UNUSED_VARS(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
    GGML_ABORT("fatal error");
}

static void mul_mat_vec_ptq1_0_pt_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, stream);
    GGML_ABORT("fatal error");
}

#endif // defined(GGML_CUDA_PTQ1_0_PT_AVAILABLE)
