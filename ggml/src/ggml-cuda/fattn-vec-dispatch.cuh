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

// XQuant Phase 3c — paired vec_dot typedef.
//
// Standard vec_dot_KQ_t has 4 args (K_c, Q_v, Q_q8, Q_ds). The paired form
// adds a 5th — `K_dom` — which points to the sibling dominant-layer K-block
// holding the shared 2-bit codes + RHT sign bits. The subordinate XKTQ2_1
// block referenced by `K_c` carries only its own per-block scale `d`.
//
// Why a parallel typedef instead of extending vec_dot_KQ_t in place:
// extending the existing typedef would force every non-paired vec_dot
// (f16, q4_0, q8_0, ktq2_1, ...) to either ignore the new arg (silent
// reg-pressure ABI churn) or re-declare with default-args (function
// pointers don't take defaults). Parallel typedef = zero touch on existing
// types, zero risk to working KTQ/VTQ paths. PAIRED dispatch is selected
// at compile time from the K-type tag, never at runtime.
typedef float (*vec_dot_KQ_paired_t)(
    const char * __restrict__ K_c,
    const void * __restrict__ Q_v,
    const int  * __restrict__ Q_q8,
    const void * __restrict__ Q_ds,
    const char * __restrict__ K_dom);
