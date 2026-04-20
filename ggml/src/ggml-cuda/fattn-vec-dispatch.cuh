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
bool try_dispatch_vec_vtq1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool try_dispatch_vec_vtq2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
