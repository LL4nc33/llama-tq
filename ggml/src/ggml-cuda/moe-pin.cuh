#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Register a host pointer range as pinned for CUDA DMA (cudaHostRegister with
// READ_ONLY|PORTABLE flags, falling back to PORTABLE only on failure).
// Returns 0 on success, -1 on failure (failure logged to stderr).
// Idempotent for exact-match ranges.
int  ggml_cuda_pin_host_range(void * addr, size_t bytes);

// Unregister all ranges registered via ggml_cuda_pin_host_range.
void ggml_cuda_unpin_all(void);

#ifdef __cplusplus
}
#endif
