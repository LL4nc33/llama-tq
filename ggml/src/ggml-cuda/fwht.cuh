#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// The transform of x * signs (signs broadcast over the rows of x), viewed as rows of src1->ne[0]; dst has
// the shape of src1. Returns whether it could be used.
bool ggml_cuda_op_fwht_signs(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * signs,
        const ggml_tensor * src1, ggml_tensor * dst);
