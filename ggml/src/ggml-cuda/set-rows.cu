#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turboquant.cuh"
#include "trellis-encode.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// TurboQuant-specific set_rows kernel template
// TQ quantize functions require a block_index parameter for RHT seed derivation,
// so they cannot use the standard 2-arg quantize_func template above.
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *, int64_t)>
static __global__ void k_set_rows_pq(const float * __restrict__ src0,
                                     const idx_t * __restrict__ src1,
                                     block_type * __restrict__ dst,
                                     const int64_t ne_total,
                                     const int64_t ne10,
                                     const int64_t ne11,
                                     const int64_t ne12,
                                     const int64_t ne13,
                                     const int64_t s01,
                                     const int64_t s02,
                                     const int64_t s03,
                                     const int64_t s10,
                                     const int64_t s11,
                                     const int64_t s12,
                                     const int64_t s1,
                                     const int64_t s2,
                                     const int64_t s3,
                                     const uint3   ne00,
                                     const uint3   ne01,
                                     const uint3   ne02,
                                     const uint3   ne11_fd,
                                     const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    // Compute flat destination block index for TQ RHT seed derivation
    const int64_t block_index = (int64_t)(dst_block - dst);

    quantize_func(src_block, dst_block, block_index);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Warp-cooperative TurboQuant quantization: one warp per 32-element block, lane j owns element j.
// The single-thread quantizers above run the whole block (signs, two FWHTs, the codebook search) in one
// thread, so a decode step with a handful of blocks is bound by the latency of one thread. The results
// are bit-identical to them: the warp FWHT does the same additions in the same order, and the norms are
// accumulated serially over the broadcast elements like the single-thread loop.
template <typename block_type> struct tq_warp_traits { static constexpr int bits = 0; };
template <> struct tq_warp_traits<block_ktq2_1> { static constexpr int bits = 2; static constexpr bool rht = true;  };
template <> struct tq_warp_traits<block_ktq3_1> { static constexpr int bits = 3; static constexpr bool rht = true;  };
template <> struct tq_warp_traits<block_ktq4_1> { static constexpr int bits = 4; static constexpr bool rht = true;  };
template <> struct tq_warp_traits<block_vtq2_1> { static constexpr int bits = 2; static constexpr bool rht = false; };
template <> struct tq_warp_traits<block_vtq3_1> { static constexpr int bits = 3; static constexpr bool rht = false; };
template <> struct tq_warp_traits<block_vtq4_1> { static constexpr int bits = 4; static constexpr bool rht = false; };

template <typename block_type>
static __device__ __forceinline__ const float * tq_warp_codebook() {
    constexpr int bits = tq_warp_traits<block_type>::bits;
    if constexpr (std::is_same_v<block_type, block_vtq2_1>) {
        return VTQ_CUDA_CB_2BIT;
    } else if constexpr (bits == 2) {
        return PQ_CUDA_CB_2BIT;
    } else if constexpr (bits == 3) {
        return PQ_CUDA_CB_3BIT;
    } else {
        return PQ_CUDA_CB_4BIT;
    }
}

// sum over the warp in the order of the single-thread loop (lane 0 first)
static __device__ __forceinline__ float tq_warp_serial_sum_sq(const float v) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < WARP_SIZE; ++j) {
        const float vj = __shfl_sync(0xFFFFFFFF, v, j, WARP_SIZE);
        acc = tq_mad_rn(acc, vj, vj);
    }
    return acc;
}

// pq_nearest_scaled with precomputed midpoints (the same float values, so the same result)
template <int n_cb>
static __device__ __forceinline__ int tq_nearest_mid(const float val, const float * __restrict__ mid, const float t) {
    int idx = 0;
#pragma unroll
    for (int step = n_cb / 2; step > 0; step /= 2) {
        if (val > t * mid[idx + step - 1] * PQ_CUDA_CB_SCALE) {
            idx += step;
        }
    }
    return idx;
}

template <typename block_type>
static __device__ __forceinline__ void tq_quantize_block_warp(const float x, block_type * __restrict__ y, const int lane,
                                                              const float * __restrict__ cb, const float * __restrict__ mid) {
    constexpr int  bits = tq_warp_traits<block_type>::bits;
    constexpr bool rht  = tq_warp_traits<block_type>::rht;
    constexpr int  n_cb = 1 << bits;

    const float norm = __fsqrt_rn(tq_warp_serial_sum_sq(x));
    if (norm < 1e-30f) {
        if (lane == 0) {
            y->d = __float2half(0.0f);
        }
        if (lane < int(sizeof(y->qs))) {
            y->qs[lane] = 0;
        }
        if constexpr (rht) {
            if (lane < 4) {
                y->sb[lane] = 0;
            }
        }
        return;
    }

    float v = __fmul_rn(x, __fdiv_rn(1.0f, norm));
    uint32_t sign_bit = 0;
    if constexpr (rht) {
        sign_bit = ktq_cuda_philox_6r((uint32_t) lane, ktq_cuda_derive_seed(0)) & 1;
        v = ktq_cuda_fwht_warp(v * (sign_bit ? 1.0f : -1.0f));
    }
    // scale search with an unbiased scale, as tq_quantize_block: lane k evaluates trial scale
    // 1 + (k - QK_KTQ_SCALE_STEPS)*QK_KTQ_SCALE_STEP on the whole block, then a warp argmax picks the
    // smallest angle (ties: the smaller scale, as the CPU loop)
    static_assert(2*QK_KTQ_SCALE_STEPS == WARP_SIZE, "one trial scale per lane");
    float vals[WARP_SIZE];
#pragma unroll
    for (int j = 0; j < WARP_SIZE; ++j) {
        vals[j] = __shfl_sync(0xFFFFFFFF, v, j, WARP_SIZE);
    }
    float best_rc = 0.0f, best_cc = 0.0f;
    {
        const float t = tq_trial_scale(lane - QK_KTQ_SCALE_STEPS);
#pragma unroll
        for (int j = 0; j < WARP_SIZE; ++j) {
            const float cj = cb[tq_nearest_mid<n_cb>(vals[j], mid, t)] * PQ_CUDA_CB_SCALE;
            best_rc = tq_mad_rn(best_rc, vals[j], cj);
            best_cc = tq_mad_rn(best_cc, cj, cj);
        }
    }
    int best_lane = lane;
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float rc = __shfl_xor_sync(0xFFFFFFFF, best_rc,   offset, WARP_SIZE);
        const float cc = __shfl_xor_sync(0xFFFFFFFF, best_cc,   offset, WARP_SIZE);
        const int   ol = __shfl_xor_sync(0xFFFFFFFF, best_lane, offset, WARP_SIZE);
        const float other = rc * rc * best_cc;   // rc^2/cc (smaller angle) compared without division
        const float mine  = best_rc * best_rc * cc;
        if (other > mine || (other == mine && ol < best_lane)) {
            best_rc = rc; best_cc = cc; best_lane = ol;
        }
    }
    const int   best  = tq_nearest_mid<n_cb>(v, mid, tq_trial_scale(best_lane - QK_KTQ_SCALE_STEPS));
    const float scale = __fdiv_rn(norm, best_rc);

    // pack the indices (and the sign bits) from ballots
    uint32_t m[bits];
#pragma unroll
    for (int b = 0; b < bits; ++b) {
        m[b] = __ballot_sync(0xFFFFFFFF, (best >> b) & 1);
    }
    const uint32_t m_sign = __ballot_sync(0xFFFFFFFF, sign_bit);
    if (lane < int(sizeof(y->qs))) {
        // byte `lane` of qs collects the index bits [8*lane, 8*lane + 8)
        uint32_t byte = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int bit = 8*lane + k;          // bit position in the packed stream
            const int j   = bit / bits;          // element
            const int b   = bit % bits;          // bit of its index
            if (j < 32) {
                byte |= ((m[b] >> j) & 1) << k;
            }
        }
        y->qs[lane] = (uint8_t) byte;
    }
    if constexpr (rht) {
        if (lane < 4) {
            y->sb[lane] = (uint8_t) (m_sign >> (8*lane));
        }
    }
    if (lane == 0) {
        y->d = __float2half(scale);
    }
}

template<typename idx_t, typename block_type>
static __global__ void k_set_rows_pq_warp(const float * __restrict__ src0,
                                          const idx_t * __restrict__ src1,
                                          block_type * __restrict__ dst,
                                          const int64_t ne_total,
                                          const int64_t s01, const int64_t s02, const int64_t s03,
                                          const int64_t s10, const int64_t s11, const int64_t s12,
                                          const int64_t s1, const int64_t s2, const int64_t s3,
                                          const uint3 ne00, const uint3 ne01, const uint3 ne02,
                                          const uint3 ne11_fd, const uint3 ne12_fd) {
    // the codebook and the midpoints of its neighbours in shared memory: the lanes look up different entries,
    // which constant memory would serialize
    constexpr int n_cb = 1 << tq_warp_traits<block_type>::bits;
    __shared__ float s_cb[n_cb];
    __shared__ float s_mid[n_cb];
    if (threadIdx.x < n_cb) {
        const float * cb = tq_warp_codebook<block_type>();
        s_cb[threadIdx.x]  = cb[threadIdx.x];
        s_mid[threadIdx.x] = threadIdx.x + 1 < n_cb ? (cb[threadIdx.x] + cb[threadIdx.x + 1]) * 0.5f : 0.0f;
    }
    __syncthreads();

    const int64_t i    = (int64_t(blockDim.x) * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;
    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) (i * 32);
    uint2    div_mod;
    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;
    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;
    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);

    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_type  * dst_block = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type) + i00 / 32;

    tq_quantize_block_warp(src_block[lane], dst_block, lane, s_cb, s_mid);
}

// Template dispatch function for TurboQuant set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*, int64_t)>
static void set_rows_cuda_pq(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        static const bool serial = getenv("GGML_CUDA_TQ_SET_ROWS_SERIAL") != nullptr; // A/B against the single-thread path
        if constexpr (tq_warp_traits<block_type>::bits > 0) if (!serial) {
            static_assert(qk == 32, "warp quantizer expects 32-element blocks");
            const int64_t n_threads = ne_total * WARP_SIZE;
            const dim3 grid_size_warp((n_threads + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);
            k_set_rows_pq_warp<idx_t, block_type><<<grid_size_warp, block_size, 0, stream>>>(
                src0_d, src1_d, dst_d, ne_total, s01, s02, s03, s10, s11, s12, s1, s2, s3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
            return;
        }
        k_set_rows_pq<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ1_1) {
        set_rows_cuda_pq<idx_t, block_ktq1_1, QK_KTQ, ktq_cuda_quantize_block<block_ktq1_1>>(
            src0_d, src1_d, (block_ktq1_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ2_1) {
        set_rows_cuda_pq<idx_t, block_ktq2_1, QK_KTQ, ktq_cuda_quantize_block<block_ktq2_1>>(
            src0_d, src1_d, (block_ktq2_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ3_1) {
        set_rows_cuda_pq<idx_t, block_ktq3_1, QK_KTQ, ktq_cuda_quantize_block<block_ktq3_1>>(
            src0_d, src1_d, (block_ktq3_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ4_1) {
        set_rows_cuda_pq<idx_t, block_ktq4_1, QK_KTQ, ktq_cuda_quantize_block<block_ktq4_1>>(
            src0_d, src1_d, (block_ktq4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ1_1) {
        set_rows_cuda_pq<idx_t, block_vtq1_1, QK_VTQ, vtq_cuda_quantize_vtq1_1_block>(
            src0_d, src1_d, (block_vtq1_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ2_1) {
        set_rows_cuda_pq<idx_t, block_vtq2_1, QK_VTQ, vtq_cuda_quantize_vtq2_1_block>(
            src0_d, src1_d, (block_vtq2_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ3_1) {
        set_rows_cuda_pq<idx_t, block_vtq3_1, QK_VTQ, vtq_cuda_quantize_vtq3_1_block>(
            src0_d, src1_d, (block_vtq3_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ4_1) {
        set_rows_cuda_pq<idx_t, block_vtq4_1, QK_VTQ, vtq_cuda_quantize_vtq4_1_block>(
            src0_d, src1_d, (block_vtq4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ2_2) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq2_2, 2, GGML_TRELLIS_L>(
            src0_d, src1_d, (block_vtq2_2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ3_2) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq3_2, 3, GGML_TRELLIS_L>(
            src0_d, src1_d, (block_vtq3_2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ4_2) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq4_2, 4, GGML_TRELLIS_L>(
            src0_d, src1_d, (block_vtq4_2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ2_2S) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq2_2s, 2, GGML_TRELLIS_L_S>(
            src0_d, src1_d, (block_vtq2_2s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ3_2S) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq3_2s, 3, GGML_TRELLIS_L_S>(
            src0_d, src1_d, (block_vtq3_2s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ4_2S) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq4_2s, 4, GGML_TRELLIS_L_S>(
            src0_d, src1_d, (block_vtq4_2s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ2_3) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq2_3, 2, GGML_TRELLIS_L, VTQ_OUTLIER_K>(
            src0_d, src1_d, (block_vtq2_3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ3_3) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq3_3, 3, GGML_TRELLIS_L, VTQ_OUTLIER_K>(
            src0_d, src1_d, (block_vtq3_3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ4_3) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq4_3, 4, GGML_TRELLIS_L, VTQ_OUTLIER_K>(
            src0_d, src1_d, (block_vtq4_3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_VTQ3_V8) {
        vtq_cuda_encode_set_rows<idx_t, block_vtq3_v8, 3, GGML_TRELLIS_L, VTQ_OUTLIER_K_V8>(
            src0_d, src1_d, (block_vtq3_v8*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ2_3S) {
        vtq_cuda_encode_set_rows<idx_t, block_ktq2_3s, 2, GGML_TRELLIS_L_S, VTQ_OUTLIER_K, /*ROT=*/true>(
            src0_d, src1_d, (block_ktq2_3s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ3_3S) {
        vtq_cuda_encode_set_rows<idx_t, block_ktq3_3s, 3, GGML_TRELLIS_L_S, VTQ_OUTLIER_K, /*ROT=*/true>(
            src0_d, src1_d, (block_ktq3_3s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_KTQ4_3S) {
        vtq_cuda_encode_set_rows<idx_t, block_ktq4_3s, 4, GGML_TRELLIS_L_S, VTQ_OUTLIER_K, /*ROT=*/true>(
            src0_d, src1_d, (block_ktq4_3s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}

// the tensor whose memory t lives in
static const ggml_tensor * set_rows_base(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

bool ggml_cuda_set_rows_trellis_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_SET_ROWS || b->op != GGML_OP_SET_ROWS) {
        return false;
    }
    for (const ggml_tensor * t : { a, b }) {
        if (t->src[0]->type != GGML_TYPE_F32 || t->src[1]->type != GGML_TYPE_I64) {
            return false;
        }
    }
    // independent writes: different caches, and neither reads the cache the other one writes
    const ggml_tensor * cache_a = set_rows_base(a);
    const ggml_tensor * cache_b = set_rows_base(b);
    if (cache_a == cache_b) {
        return false;
    }
    for (int k = 0; k < 2; ++k) {
        if (set_rows_base(a->src[k]) == cache_b || set_rows_base(b->src[k]) == cache_a) {
            return false;
        }
    }
    // one CUDA block per trellis block, all in one wave
    const int64_t n_blocks = (ggml_nelements(a->src[0]) + ggml_nelements(b->src[0])) / QK_VTQ_TRELLIS;
    if (n_blocks > ggml_cuda_info().devices[ctx.device].nsm) {
        return false;
    }

    // a trellis K type next to a trellis V type with the same CUDA block size (2 and 3 bits: 512 threads, 4 bits:
    // 256), in either graph order
    const bool a_is_k = a->type == GGML_TYPE_KTQ2_3S || a->type == GGML_TYPE_KTQ3_3S || a->type == GGML_TYPE_KTQ4_3S;
    const ggml_tensor * k = a_is_k ? a : b;
    const ggml_tensor * v = k == a ? b : a;
    cudaStream_t stream = ctx.stream();
#define TRELLIS_PAIR(TK, block_k, BK, TV, block_v, BV)                                                              \
    if (k->type == TK && v->type == TV) {                                                                           \
        vtq_cuda_encode_set_rows_pair<block_k, BK, VTQ_OUTLIER_K, true, block_v, BV, 0, false>(k, v, stream);         \
        return true;                                                                                                \
    }
    TRELLIS_PAIR(GGML_TYPE_KTQ2_3S, block_ktq2_3s, 2, GGML_TYPE_VTQ2_2S, block_vtq2_2s, 2)
    TRELLIS_PAIR(GGML_TYPE_KTQ2_3S, block_ktq2_3s, 2, GGML_TYPE_VTQ3_2S, block_vtq3_2s, 3)
    TRELLIS_PAIR(GGML_TYPE_KTQ3_3S, block_ktq3_3s, 3, GGML_TYPE_VTQ2_2S, block_vtq2_2s, 2)
    TRELLIS_PAIR(GGML_TYPE_KTQ3_3S, block_ktq3_3s, 3, GGML_TYPE_VTQ3_2S, block_vtq3_2s, 3)
    TRELLIS_PAIR(GGML_TYPE_KTQ4_3S, block_ktq4_3s, 4, GGML_TYPE_VTQ4_2S, block_vtq4_2s, 4)
#undef TRELLIS_PAIR
    return false;
}
