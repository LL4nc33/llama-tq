#pragma once

// Compute buffers shared between allocators / schedulers that never run at the same time (fork addition).
//
// A context that keeps several graphs ready, each with its own scheduler (llama_context: one per batch size of up to
// 16 tokens, so that MTP verify batches of changing size are not rebuilt), would otherwise hold a compute buffer per
// graph. A scheduler that shares places its graphs in the buffers of the other one whenever they fit and allocates its
// own only when they do not. The two must not have graphs in flight at the same time: synchronize before switching.

#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// place the graphs of galloc in the buffers of from when they fit (from == NULL: stop sharing)
GGML_API void ggml_gallocr_share_buffers(ggml_gallocr_t galloc, ggml_gallocr_t from);
// false when a borrowed buffer was replaced since galloc's graph was allocated: the graph has to be allocated again
GGML_API bool ggml_gallocr_buffers_valid(ggml_gallocr_t galloc);

// a meta buffer (-sm tensor) over the memory of buffer, with its own per-device tensor objects (used by the allocator
// for shared meta buffers: resetting one must not drop the tensor objects of the other's graph)
GGML_API ggml_backend_buffer_t ggml_backend_meta_buffer_alias(ggml_backend_buffer_t buffer);

// the same for schedulers over the same backends
GGML_API void ggml_backend_sched_share_compute_buffers(ggml_backend_sched_t sched, ggml_backend_sched_t from);
GGML_API bool ggml_backend_sched_compute_buffers_valid(ggml_backend_sched_t sched);

#ifdef  __cplusplus
}
#endif
