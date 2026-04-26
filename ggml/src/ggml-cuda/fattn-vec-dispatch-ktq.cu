// Dispatch helper for the KTQ slice of ggml_cuda_flash_attn_ext_vec.
// Covers every case where K or V is a KTQ type (KTQ1_1..KTQ4_1),
// excluding cases where V is a VTQ type — those live in
// fattn-vec-dispatch-vtq.cu.
//
// Split out of fattn.cu to allow parallel cicc instantiation.

#include "fattn-vec-dispatch.cuh"

bool try_dispatch_vec_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    // V = KTQ1_1
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,   GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_1,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_0,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_1,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_BF16,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_KTQ1_1)

    // V = KTQ2_1
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,   GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_1,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_0,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_1,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_BF16,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_KTQ2_1)

    // V = KTQ3_1
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,   GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_1,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_0,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_1,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_BF16,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_KTQ3_1)

    // V = KTQ4_1
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,   GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_1,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_0,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q5_1,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_BF16,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_KTQ4_1)

    // K = KTQ, V = standard
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_BF16)
#else
    // Symmetric K=KTQ=V
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_KTQ4_1)

    // TQ asymmetric: K=TQ, V=f16
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ1_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ2_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ3_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_KTQ4_1, GGML_TYPE_F16)

    // TQ asymmetric: K=standard, V=KTQ (recommended asymmetric configs)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,  GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,  GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,  GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_F16,  GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q4_0, GGML_TYPE_KTQ4_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0, GGML_TYPE_KTQ1_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0, GGML_TYPE_KTQ3_1)
    FATTN_VEC_CASES_ALL_D_WITH_512_RET(GGML_TYPE_Q8_0, GGML_TYPE_KTQ4_1)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V);
    return false;
}

// XQuant Phase 3d — XKTQ2_1 (paired subordinate) dispatch slice.
//
// Returns true iff K->type is a paired XKTQ subordinate type. The dominant
// sibling K-block is read from dst->src[5] (set by build_attn_mha via
// ggml_flash_attn_ext_set_sibling_k); the kv-cache resolves the dominant
// layer via xq_dominant_of_layer + get_dominant_k().
//
// PHASE 3d wiring (current state):
//   • Infrastructure (Phase 3c): paired typedef (vec_dot_KQ_paired_t),
//     paired vec_dot template (vec_dot_fattn_vec_KQ_xktq2_1_paired),
//     paired vec-dot dispatcher (get_vec_dot_KQ_paired<>).
//   • Kernel-side (Phase 3d): separate paired kernel flash_attn_ext_vec_paired
//     with its own kernel-pointer typedef (fattn_kernel_paired_t) and its own
//     launcher (launch_fattn_paired). The original flash_attn_ext_vec kernel,
//     fattn_kernel_t typedef, and launch_fattn() are byte-identical to before
//     this commit — Iron-Law: existing types provably untouched.
//   • Sibling K is read from dst->src[5]->data inside launch_fattn_paired.
//
// Runtime: the kv-cache `xquant_dispatch_ready=false` constexpr still gates
// any XKTQ2_1 K-tensor from reaching FA. Lance flips that gate after the
// bench gate passes. If the gate is flipped without bench-gate, this code
// path is now functional (no abort) — that's the correct behavior.
bool try_dispatch_vec_xktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    if (K->type != GGML_TYPE_XKTQ2_1) {
        return false;  // Not an xquant subordinate → leave to other dispatchers.
    }

    // From here on we KNOW the caller wants a paired XKTQ path; missing
    // sibling tensor is a hard error.
    GGML_ASSERT(dst->src[5] != nullptr &&
                "XKTQ2_1 K without sibling K (src[5]); did the kv-cache "
                "forget to call ggml_flash_attn_ext_set_sibling_k?");
    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_KTQ2_1 &&
                "XKTQ2_1 sibling K must be GGML_TYPE_KTQ2_1");

    // Paired CASE expansion. Currently only XKTQ2_1 is defined; future
    // xquant levels (XKTQ3_1, XKTQ4_1) extend this list.
    //
    // V-side: paired path supports the same V-types that ktq2_1 supports —
    // typically KTQ2_1 (symmetric paired), F16, Q8_0. We expand the common
    // configurations explicitly to keep instantiation count small.
    FATTN_VEC_CASES_PAIRED_ALL_D_WITH_512_RET(GGML_TYPE_XKTQ2_1, GGML_TYPE_KTQ2_1)
    FATTN_VEC_CASES_PAIRED_ALL_D_WITH_512_RET(GGML_TYPE_XKTQ2_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_PAIRED_ALL_D_WITH_512_RET(GGML_TYPE_XKTQ2_1, GGML_TYPE_Q8_0)

    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V);
    // Should not reach here for valid XKTQ2_1 setups — if we do, the (D, V)
    // combination is not instantiated. Surface a clear error.
    GGML_ABORT("XQuant Phase 3d: XKTQ2_1 paired dispatch reached but no "
               "matching (D=%lld, V=%s) paired template instantiation exists. "
               "Add the (D, type_V) pair to the FATTN_VEC_CASES_PAIRED_* list.",
               (long long) Q->ne[0], ggml_type_name(V->type));
    return false;
}
