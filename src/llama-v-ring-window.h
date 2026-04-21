// V-cache Streaming-Window Ring Buffer — C1 Prototype Skeleton
//
// Idea: keep the LAST N tokens' V-values in fp16 (L2-friendly, direct read);
//       quantize only when tokens evict from the window.
//
// This file is a **forward-declaration skeleton**. The actual impl ties into
// llama-kv-cache.cpp's deferred-V infrastructure (see cpy_v() and the
// tq_deferred_state enum). See docs/plans/2026-04-23-c1-streaming-window-design.md.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Ring-buffer state per layer (managed by llama_kv_cache).
// NOTE: this is a **stub**. Real impl uses ggml_tensor + ggml_backend buffer.
typedef struct llama_v_ring_window {
    int64_t  capacity;                // tokens held in fp16 (e.g., 256)
    int64_t  write_pos;               // circular write head modulo capacity
    int64_t  n_tokens_in_window;      // 0..capacity
    int64_t  oldest_abs_pos;          // absolute token pos of window tail

    // fp16 ring storage per layer: [capacity, n_kv_heads, head_dim] half
    // Quantized archive buffer is the existing VTQ cache (no separate storage).
    void *   fp16_buf;                // actually ggml_tensor* in real impl
} llama_v_ring_window;

// Lifecycle operations (stubs for now)
void llama_v_ring_init  (llama_v_ring_window * w, int64_t capacity,
                         int64_t n_kv_heads, int64_t head_dim);
void llama_v_ring_write (llama_v_ring_window * w, const void * v_values,
                         int64_t abs_pos);
// On eviction: caller must quantize v_values into the VTQ archive at (abs_pos - capacity).
void llama_v_ring_evict (llama_v_ring_window * w, void * evicted_out,
                         int64_t * evicted_abs_pos_out);

// Attention-read helper: returns whether abs_pos is in window (read from fp16)
// or archive (read from VTQ).
static inline bool llama_v_ring_in_window(const llama_v_ring_window * w,
                                           int64_t abs_pos) {
    return abs_pos >= w->oldest_abs_pos &&
           abs_pos < (w->oldest_abs_pos + w->n_tokens_in_window);
}

#ifdef __cplusplus
}
#endif

// --------------------------------------------------------------------------
// FA-split-dispatch pseudo-code (informational):
//
// void layer_forward(Q, K_quant, V_cache /* ring + vtq_archive */) {
//     // Split V-reads by position:
//     auto out_archive = fa_kernel(Q, K_quant, V_cache.vtq_archive,
//                                   mask=abs_pos < window.oldest_abs_pos);
//     auto out_window  = fa_kernel(Q, K_quant, V_cache.fp16_ring,
//                                   mask=abs_pos >= window.oldest_abs_pos);
//     // Merge via log-sum-exp (standard FA split-attention trick):
//     return softmax_merge(out_archive, out_window, lse_a, lse_w);
// }
//
// Expected gain: 5-15% TG on typical chat workloads, because recent tokens
// dominate softmax weights (near-causal position bias) and fp16 reads have
// no dequant cost.
//
// Status: DESIGN ONLY. No impl yet. Blocked on:
//   1. Decision whether TG gain justifies effort (see TG bench results v6)
//   2. Review of `v_staging` infra in llama-kv-cache.cpp (lines 338, 1511)
