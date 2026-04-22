// Template instances for inline MMA-KTQ kernel.
// DKQ=DV=128 matches Qwen3.5 MoE heads. ncols2=8 matches GQA ratio 8 (typical
// for modern MoE models with 64 Q heads / 8 KV heads).

#include "fattn-mma-ktq-inline.cuh"

DECL_FATTN_MMA_KTQ_INLINE_CASE(256, 256, 8, 8);
DECL_FATTN_MMA_KTQ_INLINE_CASE(256, 256, 4, 8);
DECL_FATTN_MMA_KTQ_INLINE_CASE(256, 256, 2, 8);
DECL_FATTN_MMA_KTQ_INLINE_CASE(256, 256, 1, 8);
