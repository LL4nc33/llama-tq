#include "common.cuh"
#include "ggml.h"

// where the op's new recurrent state (the K newest with rollback slots) is written when the graph evaluator folds the
// copy into the recurrent cache into the kernel (see ggml_cuda_try_gdn_scatter_skip); strides in floats
struct ggml_cuda_gated_delta_net_cache {
    float * data        = nullptr; // slot 0 (newest state) of the first sequence; sequences D = S_v*S_v*H apart
    int64_t slot_stride = 0;       // between rollback slots
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_gated_delta_net_to_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_cache & cache);
void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
