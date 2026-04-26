// Shared dispatch helpers for ggml_cuda_flash_attn_ext_vec.
//
// fattn.cu was split into multiple TUs so nvcc's cicc can compile the
// (very large) template instantiation graph in parallel. The actual
// matching logic is identical to the original monolithic switch — we
// just partition the FATTN_VEC_CASE expansions across .cu files along
// K/V type-family boundaries.
//
// Each try_dispatch_vec_* helper walks the same FATTN_VEC_CASE list,
// but returns `true` as soon as a case matches (instead of `return;`)
// and `false` if no case in its slice matched. The top-level
// ggml_cuda_flash_attn_ext_vec in fattn.cu chains the helpers in the
// original order and calls GGML_ABORT if none matched.
//
// Order of evaluation across the split files MUST match the original
// monolithic list.

#pragma once

#include "common.cuh"
#include "fattn-vec.cuh"

// FATTN_VEC_CASE variant that returns `true` on a hit instead of `return;`.
// Used inside try_dispatch_vec_* helpers.
#define FATTN_VEC_CASE_RET(D, type_K, type_V)                                                                    \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return true;                                                                                         \
        }                                                                                                        \
    }

#define FATTN_VEC_CASES_ALL_D_RET(type_K, type_V) \
    FATTN_VEC_CASE_RET( 64, type_K, type_V)       \
    FATTN_VEC_CASE_RET(128, type_K, type_V)       \
    FATTN_VEC_CASE_RET(256, type_K, type_V)

// D=512 only for TQ types (Gemma4 global attention) — avoids massive
// compile time for non-TQ.
#define FATTN_VEC_CASES_ALL_D_WITH_512_RET(type_K, type_V) \
    FATTN_VEC_CASES_ALL_D_RET(type_K, type_V)              \
    FATTN_VEC_CASE_RET(512, type_K, type_V)

// XQuant Phase 3d — paired CASE macro. Mirrors FATTN_VEC_CASE_RET but routes
// to the paired kernel wrapper, which receives the sibling dominant-K block
// pointer (`K_dom`) read from dst->src[5]. Used only inside try_dispatch_vec_xktq.
//
// Note: K_dom validity is asserted by the caller before this macro fires.
#define FATTN_VEC_CASE_PAIRED_RET(D, type_K, type_V)                                              \
    {                                                                                             \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && K->type == (type_K) && type_V_okay) {                              \
            ggml_cuda_flash_attn_ext_vec_case_paired<D, type_K, type_V>(ctx, dst);                \
            return true;                                                                          \
        }                                                                                         \
    }

#define FATTN_VEC_CASES_PAIRED_ALL_D_RET(type_K, type_V) \
    FATTN_VEC_CASE_PAIRED_RET( 64, type_K, type_V)       \
    FATTN_VEC_CASE_PAIRED_RET(128, type_K, type_V)       \
    FATTN_VEC_CASE_PAIRED_RET(256, type_K, type_V)

// D=512 only enabled via WITH_512 macro (Gemma4 path) — kept symmetric with non-paired.
#define FATTN_VEC_CASES_PAIRED_ALL_D_WITH_512_RET(type_K, type_V) \
    FATTN_VEC_CASES_PAIRED_ALL_D_RET(type_K, type_V)              \
    FATTN_VEC_CASE_PAIRED_RET(512, type_K, type_V)

// Helpers — implemented in fattn-vec-dispatch-*.cu TUs.
// Each returns true iff a matching (D, type_K, type_V) was dispatched.
bool try_dispatch_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool try_dispatch_vec_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// XQuant Phase 3c — XKTQ2_1 (cross-layer paired K) dispatch. Reads sibling
// dominant-K block from dst->src[5] (set via ggml_flash_attn_ext_set_sibling_k).
// Currently dormant (kv-cache `xquant_dispatch_ready=false` gate stops this
// path from being reached at runtime); when the gate flips, this dispatcher
// triggers a clear ABORT pointing at Phase 3d (kernel-side sibling plumbing).
bool try_dispatch_vec_xktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool try_dispatch_vec_vtq1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool try_dispatch_vec_vtq2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool try_dispatch_vec_vtq3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// E14 split-decode path — see fattn-vec-dispatch-vtq2-split.cu.
bool try_dispatch_vec_vtq2_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// XQuant Phase 3c — paired vec_dot typedef is now defined in fattn-tq.cuh
// (transitively included via fattn-common.cuh → fattn-tq.cuh) so that
// template-instance TUs which include only fattn-vec.cuh (not this header)
// can still see it. Iron-Law: existing `vec_dot_KQ_t` typedef in
// fattn-common.cuh is unchanged.

// XQuant Phase 3d — paired flash-attention kernel function pointer.
//
// The actual typedef (`fattn_kernel_paired_fwd_t`) lives in fattn-common.cuh
// alongside `launch_fattn_paired` so the launcher's signature self-contains
// — there is no need to forward-include this dispatch header from common.
//
// IRON-LAW NOTE: that typedef is SEPARATE from `fattn_kernel_t` in
// fattn-common.cuh. The shared `fattn_kernel_t` typedef is used by mma-f16,
// mma-ktq, tile, wmma-f16, vec, and vec-vtq2; extending it with a `K_dom`
// parameter would force every existing kernel signature to either accept and
// ignore the extra arg (silent ABI churn / reg-pressure) or be re-declared.
// Parallel typedef = zero touch on stable backends.
