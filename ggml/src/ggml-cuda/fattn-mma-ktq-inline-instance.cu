// Template instances for inline MMA-KTQ kernel.
// Initial target: Ministral-3 family (DKQ=DV=128, 32 Q heads / 8 KV = GQA 4).

#include "fattn-mma-ktq-inline.cuh"

DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 8, 4);
DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 4, 4);
DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 2, 4);
DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 1, 4);
