// Template instances for inline MMA-KTQ kernel.
// Ministral-3 family: DKQ=DV=128, 32 Q / 8 KV = GQA 4.
// ncols2=4 fixed. ncols1 ∈ {4, 8} — smaller ncols1 values have degenerate
// MMA tile dimensions that fail compile.

#include "fattn-mma-ktq-inline.cuh"

DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 4, 4);
DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 8, 4);

// Phase 3: VTQ V variant — used when V cache is vtq2_1 (Ministral-3 prefill path).
DECL_FATTN_MMA_KTQ_VTQ_INLINE_CASE(128, 128, 4, 4);
DECL_FATTN_MMA_KTQ_VTQ_INLINE_CASE(128, 128, 8, 4);
