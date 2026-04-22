#pragma once

#include "common.cuh"

// MMA-KTQ split-dequant path.
//
// Phase 2 approach: bulk-dequantize KTQ K-cache into an fp16 scratch buffer
// once per FA call, swap K tensor metadata, then dispatch to the existing
// tensor-core MMA-F16 kernel. Restores K metadata on exit.
//
// This trades one extra K-memory pass for tensor-core parallelism on PP.
// No new MMA template instances are needed — the existing (f16,f16) MMA
// kernel handles the real compute.
void ggml_cuda_flash_attn_ext_mma_ktq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Exposed from fattn.cu so the split-dequant wrapper can re-dispatch after
// swapping K.
void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
