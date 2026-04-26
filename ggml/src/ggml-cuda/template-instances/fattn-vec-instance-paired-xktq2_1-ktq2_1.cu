// XQuant Phase 3d — paired-vec template instance for XKTQ2_1 K + KTQ2_1 V.
// Generated for the XQuant cross-layer KV reuse path. The non-paired
// (D, type_K, type_V) instances live in fattn-vec-instance-*.cu and use
// the original DECL_FATTN_VEC_CASE macro; paired uses DECL_FATTN_VEC_CASE_PAIRED.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_PAIRED( 64, GGML_TYPE_XKTQ2_1, GGML_TYPE_KTQ2_1);
DECL_FATTN_VEC_CASE_PAIRED(128, GGML_TYPE_XKTQ2_1, GGML_TYPE_KTQ2_1);
DECL_FATTN_VEC_CASE_PAIRED(256, GGML_TYPE_XKTQ2_1, GGML_TYPE_KTQ2_1);
DECL_FATTN_VEC_CASE_PAIRED(512, GGML_TYPE_XKTQ2_1, GGML_TYPE_KTQ2_1);
