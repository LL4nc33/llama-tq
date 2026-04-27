#pragma once

#include "common.h"

#include <cstdio>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

// Phase 6a: Router confidence profiler.
// Captures post-softmax MoE router probabilities per (token, layer) and dumps
// them as a fixed-width binary stream for offline analysis (tools/profile-router.py).
//
// Hooked into the model graph via ggml_backend_sched_eval_callback. Filters on
// tensor names matching `^ffn_moe_logits-(\d+)$` (pre-gating logits, set at
// src/llama-graph.cpp:1300 via cb() → ggml_format_name in src/llama-context.cpp:2417).
//
// Why logits and not probs: Qwen3-Next uses LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT
// which sets probs=logits and applies softmax only AFTER top-k selection on the
// k=8 weights (graph-builder line 1391-1396). So the only tensor that always
// holds the full pre-gating distribution is ffn_moe_logits. The Python analyzer
// applies softmax host-side, making it gating-op-agnostic.
//
// Binary record layout (little-endian, fixed-width):
//
//   header (32 bytes, written once on first record):
//       char     magic[4];     // "TQRP"
//       uint32_t version;      // 1
//       uint32_t n_expert;     // experts per layer (e.g. 256)
//       uint32_t reserved;
//       float    tau;          // configured threshold
//       char     pad[12];
//
//   per-record:
//       uint32_t token_idx;    // monotonic, per-callback firing
//       uint16_t layer_idx;    // parsed from "ffn_moe_probs-<N>"
//       uint16_t n_expert;     // sanity, should match header
//       float    probs[n_expert];
//
// Multi-token batches: each ubatch slot of the probs tensor produces one record
// with a distinct token_idx. token_idx is the absolute counter across all
// callbacks for the run (not the position in the prompt).

struct router_profile_data {
    std::FILE *           fp           = nullptr;
    std::regex            filter;
    int                   n_expert     = 0;
    int                   max_tokens   = 4096;     // per-layer cap (not global)
    float                 tau          = 0.85f;
    std::vector<uint8_t>  scratch;     // host-side staging buffer
    int64_t               token_idx    = 0;        // global monotonic counter (records written)
    std::vector<int64_t>  per_layer_count;         // per-layer cap tracker (lazy-grown)
    bool                  header_written = false;

    router_profile_data();
    router_profile_data(const std::string & out_path, float tau, int max_tokens);
    ~router_profile_data();

    // Move-only (FILE * ownership).
    router_profile_data(router_profile_data &&) noexcept;
    router_profile_data & operator=(router_profile_data &&) noexcept;
    router_profile_data(const router_profile_data &) = delete;
    router_profile_data & operator=(const router_profile_data &) = delete;
};

// ggml_backend_sched_eval_callback signature.
// user_data must point to a router_profile_data instance.
bool router_profile_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
