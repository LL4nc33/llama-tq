#include "fattn-mma-ktq.cuh"
#include "fattn-vec-dispatch.cuh"

// Phase 1 stub: route back to the VEC KTQ path. Keeps dispatcher wiring
// testable before the real MMA kernel lands in Phase 2.
void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (try_dispatch_vec_ktq(ctx, dst)) {
        return;
    }
    GGML_ABORT("fattn-mma-ktq stub: VEC KTQ dispatch failed");
}
