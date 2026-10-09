#include "common.cuh"

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * dst);
void ggml_cuda_op_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
