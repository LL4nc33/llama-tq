#pragma once

#include "common.cuh"

// MMA-KTQ Phase 1: dispatcher stub.
// Phase 2 will replace the VEC fallback with a tensor-core kernel that
// performs warp-cooperative KTQ dequant into shmem before MMA tile-load.
void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
