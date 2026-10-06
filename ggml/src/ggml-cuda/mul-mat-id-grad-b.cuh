#pragma once

#include "common.cuh"

// CUDA implementation of ggml_mul_mat_id_grad_b: the gradient of mul_mat_id w.r.t. b,
//   grad_b[c, e_b, t] = sum over e with e mod n_used_b == e_b of sum_r as[c, r, ids[e, t]] * grad_c[r, e, t].
// The routing is read back to the host and grouped by expert; each used expert is dequantized to f32 once and
// applied to its gathered gradient columns with cuBLAS.
void ggml_cuda_op_mul_mat_id_grad_b(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_mul_mat_id_grad_b_supported(const ggml_tensor * dst);
