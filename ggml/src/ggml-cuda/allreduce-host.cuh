#include "common.cuh"

// Sum of an F32 tensor across devices through mapped pinned host memory, for GPUs without peer access.
// Every device writes its part to a host slot, raises a flag and adds the other slots once their flags are
// raised; no host thread takes part. The sum is taken in the same order on every device, so all devices get
// identical results. Returns false when the path does not apply (then the caller falls back).
// Disabled with GGML_CUDA_HOST_ALLREDUCE=0.
bool ggml_cuda_allreduce_host(ggml_backend_t * backends, ggml_tensor ** tensors, size_t n_backends);
