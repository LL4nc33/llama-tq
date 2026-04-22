// Minimal template instance for the inline MMA-KTQ kernel.
// Single config: DKQ=128, DV=128, ncols1=8, ncols2=1 — matches Qwen3.5 MoE
// attention heads. Expand matrix once correctness passes.

#include "fattn-mma-ktq-inline.cuh"

DECL_FATTN_MMA_KTQ_INLINE_CASE(128, 128, 8, 1);
