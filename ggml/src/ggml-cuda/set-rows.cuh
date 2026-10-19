#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the trellis KV writes a and b of a decoded token in one launch (see k_vtq_encode_trellis_pair); false if this pair
// is not one of those, then nothing is launched
bool ggml_cuda_set_rows_trellis_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b);
