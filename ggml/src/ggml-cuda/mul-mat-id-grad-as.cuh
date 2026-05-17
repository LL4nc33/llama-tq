#pragma once

#include "common.cuh"

// CUDA implementation of ggml_compute_forward_mul_mat_id_grad_as.
// Computes the gradient of mul_mat_id with respect to the stacked expert
// weights "as":
//
//   grad_as[i, j, k] = sum over (e, t) where ids[e, t] == k of
//                          grad_c[i, e, t] * b[j, e mod n_used_b, t]
//
// Inputs / outputs are F32 throughout. The kernel uses per-expert atomic
// accumulation; for moderate routing counts on Turing this is the simplest
// path and avoids the host-side reordering pass that the forward kernel
// uses.
void ggml_cuda_op_mul_mat_id_grad_as(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
