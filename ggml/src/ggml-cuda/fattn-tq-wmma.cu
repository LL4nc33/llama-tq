#include "fattn-tq-wmma.cuh"

// Tensor-core flash attention for quantized KV caches (fattn-tq-wmma.cuh). Returns false if the shape or the types
// are not covered.

// shortest cache (cells) from which one block of 17-32 columns runs here instead of converting the cache to f16. RTX
// 2060, against the conversion: q8_0 +4-9 % at 8k, +10-15 % at 16k; ktq4_1/vtq4_1 -1 to +2 % at 8k, -1 to +5 % at
// 16k; -3 % at 2k. Not with a KTQ K at head size 256: the 32-column tiles hold 16 rows, and the inverse Hadamard
// transform of each tile costs more than the conversion saves (Qwen3.6-35B-A3B, 3-4 queries: -0.5 % at 4k to -4 % at
// 32k; q8_0 +2-5 % at 32k)
#define FATTN_TQ_WMMA_NC32_MIN_KV 8192

static bool fattn_tq_wmma_enabled(const int device) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_TQ_WMMA");
        return e == nullptr || atoi(e) != 0;
    }();
    const int cc = ggml_cuda_info().devices[device].cc;
    return enabled && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
}

// (head size, K type, V type) is one of FATTN_TQ_WMMA_CASES; decode: one that a single query takes
static bool fattn_tq_wmma_has_case(const int64_t D, const ggml_type type_K, const ggml_type type_V, const bool decode) {
#define FATTN_TQ_WMMA_HAS_CASE(D_, type_K_, type_V_, decode_)                \
    if (D == (D_) && type_K == (type_K_) && type_V == (type_V_)) {           \
        return !decode || (decode_);                                         \
    }
    FATTN_TQ_WMMA_CASES(FATTN_TQ_WMMA_HAS_CASE)
#undef FATTN_TQ_WMMA_HAS_CASE
    return false;
}

// the kernels are compiled in template-instances/fattn-tq-wmma-instance-*.cu, one TU per (head size, columns)
static bool fattn_tq_wmma_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int64_t D, const int ncols) {
    switch (D) {
        case  64: return ncols == 16 ? ggml_cuda_flash_attn_ext_tq_wmma_d< 64, 16>(ctx, dst) :
                                       ggml_cuda_flash_attn_ext_tq_wmma_d< 64, 32>(ctx, dst);
        case 128: return ncols == 16 ? ggml_cuda_flash_attn_ext_tq_wmma_d<128, 16>(ctx, dst) :
                                       ggml_cuda_flash_attn_ext_tq_wmma_d<128, 32>(ctx, dst);
        case 256: return ncols == 16 ? ggml_cuda_flash_attn_ext_tq_wmma_d<256, 16>(ctx, dst) :
                                       ggml_cuda_flash_attn_ext_tq_wmma_d<256, 32>(ctx, dst);
        case 512: return ggml_cuda_flash_attn_ext_tq_wmma_d<512, 16>(ctx, dst);
        default:  return false;
    }
}

// Small batches of queries (speculative decoding) with a GQA ratio that divides the block width: the kernel reads the
// quantized cache once per block instead of converting it to f16 for the MMA kernel every step. Only while all
// (query, head) columns fit one block: a second block reads the cache again and loses to the conversion (RTX 2060,
// 32k: 36 columns -25 %); 32 columns only on long caches (FATTN_TQ_WMMA_NC32_MIN_KV). Also used by
// ggml_cuda_flash_attn_ext_get_alloc_size: no f16 conversion buffer is reserved when this path runs.
bool ggml_cuda_flash_attn_ext_tq_wmma_batch(const int device, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    if (!fattn_tq_wmma_enabled(device)) {
        return false;
    }
    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (Q->ne[1] < 2 || Q->ne[1] > 16 || Q->ne[3] != 1 || !mask || mask->ne[2] != 1 || max_bias != 0.0f ||
            n_kv_max > 0 || K->ne[1] % 64 != 0 || Q->ne[0] != V->ne[0] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio != 2 && gqa_ratio != 4 && gqa_ratio != 8 && gqa_ratio != 16) {
        return false;
    }
    const int64_t ncols     = Q->ne[1]*gqa_ratio;
    const int64_t ncols_max = Q->ne[0] == 512 ? 16 : 32;
    if (ncols > ncols_max || (ncols > 16 && (K->ne[1] < FATTN_TQ_WMMA_NC32_MIN_KV ||
                                             (Q->ne[0] == 256 && GGML_TYPE_IS_KTQ(K->type))))) {
        return false;
    }
    return fattn_tq_wmma_has_case(Q->ne[0], K->type, V->type, false);
}

bool ggml_cuda_flash_attn_ext_tq_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int device = ggml_cuda_get_device();
    const int64_t D = Q->ne[0];
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // 16 columns while all queries and their heads fit, else 32 (head 512: always 16)
    if (ggml_cuda_flash_attn_ext_tq_wmma_batch(device, dst)) {
        return fattn_tq_wmma_dispatch(ctx, dst, D, Q->ne[1]*gqa_ratio <= 16 ? 16 : 32);
    }

    if (!fattn_tq_wmma_enabled(device)) {
        return false;
    }
    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || gqa_ratio < 2 || gqa_ratio > 16 || !mask || max_bias != 0.0f ||
            n_kv_max > 0 || K->ne[1] % 64 != 0 || Q->ne[0] != V->ne[0]) {
        return false;
    }
    if (!fattn_tq_wmma_has_case(D, K->type, V->type, true)) {
        return false;
    }
    return fattn_tq_wmma_dispatch(ctx, dst, D, 16);
}
