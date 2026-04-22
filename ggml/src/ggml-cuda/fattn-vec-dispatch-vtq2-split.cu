// E14 split-decode Dispatcher für VTQ_2 family (VTQ2_2, VTQ3_2, VTQ4_2).
//
// When enabled via FATTN_VTQ2_SPLIT_ENABLE, this helper intercepts
// ncols=1 (decode-path) FA calls on VTQ_2 V-cache, dequantizes V once
// into an fp16 scratch buffer, and dispatches to the existing F16 FA
// kernel. This eliminates the per-sample V-dequant that caused the
// 15× TG regression on Qwen3.5-35B-A3B (d=256).
//
// Phase 3B2 — primary Pfad aus
// docs/plans/2026-04-22-e14-split-decode-spec.md.
//
// Reuses `ggml_get_to_fp16_nc_cuda` (convert.cu:1035) for bulk dequant.
// No new dequantize kernel needed.

#include "fattn-vec-dispatch.cuh"
#include "convert.cuh"

// Forward declaration — implemented in fattn-vec-dispatch-f16.cu.
extern bool try_dispatch_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef FATTN_VTQ2_SPLIT_ENABLE

static bool is_vtq2_family(ggml_type t) {
    return t == GGML_TYPE_VTQ2_2 || t == GGML_TYPE_VTQ3_2 || t == GGML_TYPE_VTQ4_2;
}

// Attempt the E14 split-decode path.
// Returns true if this function handled the dispatch (and the FA kernel ran).
// Returns false if the fast-path preconditions are not met (caller should
// fall through to the legacy try_dispatch_vec_vtq2).
bool try_dispatch_vec_vtq2_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * V = dst->src[2];

    // Preconditions: VTQ_2 V-cache + ncols=1 (decode path).
    if (!is_vtq2_family(V->type)) {
        return false;
    }
    if (Q->ne[1] != 1) {
        // ncols>=2 (prefill) stays on legacy path for now. Phase 3B2b may
        // expand to larger ncols after 3B2a is validated.
        return false;
    }

    to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
    if (to_fp16 == nullptr) {
        return false;
    }

    // V tensor shape (ne00, ne01, ne02, ne03):
    //   ne00 = head_dim (D), ne01 = ctx (seq len), ne02 = n_kv_heads, ne03 = batch
    const int64_t ne00 = V->ne[0];
    const int64_t ne01 = V->ne[1];
    const int64_t ne02 = V->ne[2];
    const int64_t ne03 = V->ne[3];
    const int64_t scratch_elems = ne00 * ne01 * ne02 * ne03;

    // Strides are passed as BLOCK units, not bytes — the k_dequantize_trellis_nc
    // kernel uses them as block-index offsets (ibx0 = i03*s03 + i02*s02 + i01*s01,
    // then x[ib] indexes by sizeof(block_vtq{2,3,4}_2) via typed pointer). Passing
    // raw bytes over-indexes by a factor of sizeof(block_vtq*_2) → CUDA OOB.
    // Matches the convention in fattn-common.cuh:1705, fattn-mma-ktq.cu (post-fix),
    // and fattn-mma-ktq-inline.cuh:1662.
    const size_t ts = ggml_type_size(V->type);
    GGML_ASSERT(V->nb[0] == ts);
    const int64_t s01 = V->nb[1] / ts;
    const int64_t s02 = V->nb[2] / ts;
    const int64_t s03 = V->nb[3] / ts;

    ggml_cuda_pool_alloc<half> v_scratch(ctx.pool(), scratch_elems);
    to_fp16(V->data, v_scratch.get(), ne00, ne01, ne02, ne03, s01, s02, s03, ctx.stream());

    // Save + swap V tensor fields. dst->src[2] is a non-const ggml_tensor*,
    // mutation is observable to downstream dispatch. We restore before returning.
    const ggml_type  saved_type = V->type;
    void * const     saved_data = V->data;
    const size_t     saved_nb0  = V->nb[0];
    const size_t     saved_nb1  = V->nb[1];
    const size_t     saved_nb2  = V->nb[2];
    const size_t     saved_nb3  = V->nb[3];

    V->type = GGML_TYPE_F16;
    V->data = v_scratch.get();
    V->nb[0] = sizeof(half);
    V->nb[1] = V->nb[0] * ne00;
    V->nb[2] = V->nb[1] * ne01;
    V->nb[3] = V->nb[2] * ne02;

    const bool dispatched = try_dispatch_vec_f16(ctx, dst);

    // Restore V fields — must happen on every exit path.
    V->type = saved_type;
    V->data = saved_data;
    V->nb[0] = saved_nb0;
    V->nb[1] = saved_nb1;
    V->nb[2] = saved_nb2;
    V->nb[3] = saved_nb3;

    return dispatched;
}

#else // !FATTN_VTQ2_SPLIT_ENABLE

bool try_dispatch_vec_vtq2_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    return false;
}

#endif  // FATTN_VTQ2_SPLIT_ENABLE
