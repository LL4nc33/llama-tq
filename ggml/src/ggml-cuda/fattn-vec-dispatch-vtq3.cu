// Dispatch helper for the VTQ_3 family slice of ggml_cuda_flash_attn_ext_vec.
// Covers V=VTQ2_3, VTQ3_3, VTQ4_3 (Trellis backbone + 4 fp16 outliers/block)
// + VTQ3_V8 (TurboQuant v8 redesign of vtq3_3 with 2 outliers, 3.625 bpw).
//
// Split from fattn-vec-dispatch-vtq2.cu so VTQ_2 and VTQ_3 family cases
// compile in parallel. Each TU instantiates a disjoint slice of the
// (D, type_K, type_V) template graph.

// The cases themselves live in one TU per V type (fattn-vec-dispatch-vtq<bits>_<family>.cu).
#include "fattn-vec-dispatch.cuh"

bool try_dispatch_vec_vtq3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (try_dispatch_vec_vtq2_3(ctx, dst)) return true;
    if (try_dispatch_vec_vtq3_3(ctx, dst)) return true;
    if (try_dispatch_vec_vtq4_3(ctx, dst)) return true;
    if (try_dispatch_vec_vtq3_v8(ctx, dst)) return true;

    return false;
}
