#pragma once

// Trick 2 PR2: per-layer mixed-precision V-cache profile + strategy resolver
//
// Given an optional JSON profile file (produced by PR1's --tq-profile-heads),
// a strategy name, base/high/low types, and layer count, returns a per-layer
// ggml_type vector indicating what V-cache quantisation to use on each layer.

#include "ggml.h"

#include <string>
#include <vector>

namespace llama_tq {

    struct head_stat {
        int   head_idx  = 0;
        float variance  = 0.0f;
        float kurtosis  = 0.0f;
    };

    struct layer_stat {
        int                    layer_idx = 0;
        std::vector<head_stat> heads;
    };

    struct profile {
        int                     n_samples = 0;
        std::vector<layer_stat> layers;
    };

    // Load profile JSON produced by PR1. Returns an empty profile on error.
    profile load_profile(const std::string & path, std::string * err = nullptr);

    // Resolve per-layer V-cache types.
    //
    //   profile_path   : empty OR path to PR1 JSON
    //   strategy       : one of "top-n:N", "ratio:X", "kurt:Y", "mixed", "auto", "manual"
    //   override_spec  : e.g. "0-1:vtq4_2,13:vtq4_2,*:vtq3_2" (empty disables)
    //   budget_bpw     : 0.0 disables; positive = downgrade worst upgrades until avg bpw <= budget
    //
    // Returns a vector of size n_layer. On failure (bad profile, bad strategy)
    // an error is written to err_out and an empty vector is returned; callers
    // should then fall back to uniform type_v.
    std::vector<ggml_type> resolve_v_types(
            const std::string & profile_path,
            const std::string & strategy,
            ggml_type           base,
            ggml_type           low,
            ggml_type           high,
            const std::string & override_spec,
            int                 n_layer,
            float               budget_bpw  = 0.0f,
            std::string *       err_out     = nullptr);

    // Parse a single override rule spec into a per-layer assignment overlay.
    // Rules are comma-separated, "RANGE:TYPE" where RANGE = "N" | "A-B" | "*".
    // Rules are evaluated left-to-right, later rules win per-layer.
    // Unset entries are GGML_TYPE_COUNT (= "no opinion").
    std::vector<ggml_type> parse_override(
            const std::string & spec,
            int                 n_layer,
            std::string *       err_out = nullptr);

    // Parse a type name like "vtq3_2" → GGML_TYPE_VTQ3_2. Returns GGML_TYPE_COUNT on error.
    ggml_type type_from_name(const std::string & name);

}   // namespace llama_tq
