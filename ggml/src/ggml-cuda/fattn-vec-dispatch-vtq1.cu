// Dispatch helper for the VTQ_1 family slice of ggml_cuda_flash_attn_ext_vec.
// Covers V=VTQ1_1..VTQ4_1. K may be any supported K type.
//
// Split from fattn-vec-dispatch-vtq.cu so VTQ_1 and VTQ_2 family cases
// compile in parallel (each ~22min instead of combined ~45min).

// The cases themselves live in one TU per V type (fattn-vec-dispatch-vtq<bits>_<family>.cu).
#include "fattn-vec-dispatch.cuh"

bool try_dispatch_vec_vtq1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (try_dispatch_vec_vtq1_1(ctx, dst)) return true;
    if (try_dispatch_vec_vtq2_1(ctx, dst)) return true;
    if (try_dispatch_vec_vtq3_1(ctx, dst)) return true;
    if (try_dispatch_vec_vtq4_1(ctx, dst)) return true;

    return false;
}
