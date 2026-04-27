#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// Phase 6f: Expert-hotness profile loader.
//
// Reads JSON output from `tools/profile-router.py --mode hotness` and exposes
// the top-N hot expert IDs per layer. Consumed at runtime by the mul_mat_id
// dispatch path to issue __builtin_prefetch on hot expert weight blocks
// before each MoE layer.
//
// File format (produced by tools/profile-router.py):
//   {
//     "schema_version": 1,
//     "model_name": "...",
//     "n_expert": 256,
//     "n_expert_used": 8,
//     "top_k": 20,
//     "stats": {...},
//     "layers": {
//       "0":  [42, 17, 301, ...],
//       "1":  [88, 412, ...]
//     },
//     ...
//   }

struct expert_hotness {
    int n_expert      = 0;
    int n_expert_used = 0;
    int top_k         = 0;
    std::string model_name;

    // per_layer[il] = vector of top-N expert IDs for layer il, in dispatch-rank order.
    // Empty vector for layers not in the JSON (no prefetch issued for those).
    std::vector<std::vector<int32_t>> per_layer;

    bool valid() const { return n_expert > 0 && !per_layer.empty(); }
    int  n_layers() const { return (int) per_layer.size(); }
};

// Load JSON file into `out`. Returns true on success. Logs warnings + returns
// false on parse error or file-not-found.
bool expert_hotness_load(const std::string & path, expert_hotness & out);

// Validate that the loaded profile is compatible with the given hparams.
// Logs and returns false on mismatch (n_expert, layer count).
bool expert_hotness_compatible(const expert_hotness & h, int n_expert, int n_layer);
