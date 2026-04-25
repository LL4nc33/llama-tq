#include "tq-profile.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;

namespace llama_tq {

// ---------- helpers ----------

static std::string to_lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static void trim(std::string & s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) ++a;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char) s[b-1])) --b;
    s = s.substr(a, b - a);
}

ggml_type type_from_name(const std::string & name) {
    const std::string n = to_lower(name);
    // Only VTQ_2 Trellis types are valid targets for PR2 mixed-precision
    if (n == "vtq2_2") return GGML_TYPE_VTQ2_2;
    if (n == "vtq3_2") return GGML_TYPE_VTQ3_2;
    if (n == "vtq4_2") return GGML_TYPE_VTQ4_2;
    if (n == "vtq2_3") return GGML_TYPE_VTQ2_3;
    if (n == "vtq3_3") return GGML_TYPE_VTQ3_3;
    if (n == "vtq4_3") return GGML_TYPE_VTQ4_3;
    // Allow falling back to f16 / q8_0 for protected layers if explicitly requested
    if (n == "f16")    return GGML_TYPE_F16;
    if (n == "q8_0")   return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

// ---------- profile loader ----------

profile load_profile(const std::string & path, std::string * err) {
    profile p;
    std::ifstream f(path);
    if (!f.is_open()) {
        if (err) *err = "cannot open profile file: " + path;
        return p;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        if (err) *err = std::string("JSON parse error: ") + e.what();
        return p;
    }

    try {
        if (j.contains("n_samples")) p.n_samples = j.at("n_samples").get<int>();

        // Schema: { "layers": [ { "layer_idx": 0, "heads": [ {"head_idx":0,"variance":...,"kurtosis":...}, ... ] }, ... ] }
        if (!j.contains("layers")) {
            if (err) *err = "profile missing 'layers' array";
            return p;
        }

        for (const auto & lj : j.at("layers")) {
            layer_stat ls;
            ls.layer_idx = lj.value("layer_idx", (int) p.layers.size());
            if (lj.contains("heads")) {
                for (const auto & hj : lj.at("heads")) {
                    head_stat hs;
                    hs.head_idx  = hj.value("head_idx", 0);
                    hs.variance  = hj.value("variance",  0.0f);
                    hs.kurtosis  = hj.value("kurtosis",  0.0f);
                    ls.heads.push_back(hs);
                }
            }
            p.layers.push_back(std::move(ls));
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("profile schema error: ") + e.what();
        p.layers.clear();
        return p;
    }

    return p;
}

// Extract (max, min) variance and max kurtosis per layer
struct layer_summary {
    int   layer_idx = -1;
    float var_max   = 0.0f;
    float var_min   = 0.0f;
    float kurt_max  = 0.0f;
    float ratio     = 1.0f; // var_max / var_min (guarded)
};

static std::vector<layer_summary> summarize(const profile & p) {
    std::vector<layer_summary> out;
    out.reserve(p.layers.size());
    for (const auto & l : p.layers) {
        layer_summary s;
        s.layer_idx = l.layer_idx;
        if (l.heads.empty()) {
            out.push_back(s);
            continue;
        }
        s.var_max  = l.heads.front().variance;
        s.var_min  = l.heads.front().variance;
        s.kurt_max = l.heads.front().kurtosis;
        for (const auto & h : l.heads) {
            s.var_max  = std::max(s.var_max,  h.variance);
            s.var_min  = std::min(s.var_min,  h.variance);
            s.kurt_max = std::max(s.kurt_max, h.kurtosis);
        }
        s.ratio = (s.var_min > 1e-12f) ? (s.var_max / s.var_min) : 1e30f;
        out.push_back(s);
    }
    return out;
}

// ---------- override parser ----------

std::vector<ggml_type> parse_override(const std::string & spec, int n_layer, std::string * err_out) {
    std::vector<ggml_type> out(n_layer, GGML_TYPE_COUNT);
    if (spec.empty()) return out;

    // split by comma
    std::vector<std::string> rules;
    {
        std::stringstream ss(spec);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            trim(tok);
            if (!tok.empty()) rules.push_back(tok);
        }
    }

    for (const auto & rule : rules) {
        auto colon = rule.find(':');
        if (colon == std::string::npos) {
            if (err_out) *err_out = "override rule missing ':': " + rule;
            return std::vector<ggml_type>(n_layer, GGML_TYPE_COUNT);
        }
        std::string range_s = rule.substr(0, colon);
        std::string type_s  = rule.substr(colon + 1);
        trim(range_s); trim(type_s);

        ggml_type t = type_from_name(type_s);
        if (t == GGML_TYPE_COUNT) {
            if (err_out) *err_out = "override rule unknown type: " + type_s;
            return std::vector<ggml_type>(n_layer, GGML_TYPE_COUNT);
        }

        int lo = 0, hi = n_layer - 1;
        if (range_s == "*") {
            // all layers
        } else {
            auto dash = range_s.find('-');
            if (dash == std::string::npos) {
                try {
                    lo = hi = std::stoi(range_s);
                } catch (...) {
                    if (err_out) *err_out = "override rule bad layer index: " + range_s;
                    return std::vector<ggml_type>(n_layer, GGML_TYPE_COUNT);
                }
            } else {
                try {
                    lo = std::stoi(range_s.substr(0, dash));
                    hi = std::stoi(range_s.substr(dash + 1));
                } catch (...) {
                    if (err_out) *err_out = "override rule bad range: " + range_s;
                    return std::vector<ggml_type>(n_layer, GGML_TYPE_COUNT);
                }
            }
        }

        lo = std::max(0, lo);
        hi = std::min(n_layer - 1, hi);
        for (int i = lo; i <= hi; ++i) out[i] = t;
    }

    return out;
}

// ---------- strategy application ----------

// bpw approximations for budget solver (block size 32)
static float bpw_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_VTQ2_2: return 34.0f * 8.0f / 32.0f; // 8.5
        case GGML_TYPE_VTQ3_2: return 50.0f * 8.0f / 32.0f; // 12.5  -> actually 3.0625 bpw per value but matching block-level is fine for ranking
        case GGML_TYPE_VTQ4_2: return 66.0f * 8.0f / 32.0f; // 16.5
        case GGML_TYPE_VTQ2_3: return 48.0f * 8.0f / 128.0f; // 3.0 bpw (block=128 samples)
        case GGML_TYPE_VTQ3_3: return 64.0f * 8.0f / 128.0f; // 4.0 bpw
        case GGML_TYPE_VTQ4_3: return 80.0f * 8.0f / 128.0f; // 5.0 bpw
        case GGML_TYPE_F16:    return 16.0f;
        case GGML_TYPE_Q8_0:   return 8.5f;
        default:               return 32.0f;
    }
}

// True bpw per element (block=32 samples)
static float bpw_per_elem(ggml_type t) {
    switch (t) {
        case GGML_TYPE_VTQ2_2: return 34.0f * 8.0f / 32.0f / 8.0f * 1.0f; // 1.0625 bytes / val is wrong
        default: break;
    }
    // simpler: 8 * nbytes_per_block / block_size
    switch (t) {
        case GGML_TYPE_VTQ2_2: return (34.0f * 8.0f) / 32.0f; // 8.5
        case GGML_TYPE_VTQ3_2: return (50.0f * 8.0f) / 32.0f; // 12.5
        case GGML_TYPE_VTQ4_2: return (66.0f * 8.0f) / 32.0f; // 16.5
        case GGML_TYPE_VTQ2_3: return (48.0f * 8.0f) / 128.0f; // 3.00
        case GGML_TYPE_VTQ3_3: return (64.0f * 8.0f) / 128.0f; // 4.00
        case GGML_TYPE_VTQ4_3: return (80.0f * 8.0f) / 128.0f; // 5.00
        case GGML_TYPE_F16:    return 16.0f;
        case GGML_TYPE_Q8_0:   return 8.5f;
        default:               return 32.0f;
    }
}

// Apply overall strategy — returns which layers to upgrade/downgrade.
// Layer vector initialised to `base` beforehand; this modifies it.
static void apply_strategy(
        std::vector<ggml_type> & out,
        const std::vector<layer_summary> & s,
        const std::string & strategy,
        ggml_type base, ggml_type low, ggml_type high)
{
    const int n = (int) out.size();
    const std::string strat = to_lower(strategy);

    // ranking by var_max descending
    std::vector<int> rank(s.size());
    for (size_t i = 0; i < s.size(); ++i) rank[i] = (int) i;
    std::sort(rank.begin(), rank.end(), [&](int a, int b) { return s[a].var_max > s[b].var_max; });

    auto set_layer = [&](int layer_idx, ggml_type t) {
        if (layer_idx >= 0 && layer_idx < n) out[layer_idx] = t;
    };

    if (strat.rfind("top-n:", 0) == 0) {
        int N = std::stoi(strat.substr(6));
        N = std::max(0, std::min(N, (int) s.size()));
        for (int i = 0; i < N; ++i) set_layer(s[rank[i]].layer_idx, high);
        // bottom N/2 → low
        for (int i = 0; i < N/2 && i < (int) s.size(); ++i) {
            set_layer(s[rank[s.size() - 1 - i]].layer_idx, low);
        }
    } else if (strat.rfind("ratio:", 0) == 0) {
        float X = std::stof(strat.substr(6));
        for (const auto & ls : s) {
            if (ls.ratio >= X) set_layer(ls.layer_idx, high);
        }
    } else if (strat.rfind("kurt:", 0) == 0) {
        float Y = std::stof(strat.substr(5));
        for (const auto & ls : s) {
            if (ls.kurt_max >= Y) set_layer(ls.layer_idx, high);
        }
    } else if (strat == "mixed") {
        for (const auto & ls : s) {
            if (ls.ratio >= 3.0f || ls.kurt_max >= 500.0f) set_layer(ls.layer_idx, high);
        }
    } else if (strat == "auto") {
        // No profile needed: upgrade first 2 + last 2 layers (attention-sink heuristic)
        for (int i = 0; i < 2 && i < n; ++i)      out[i] = high;
        for (int i = n-2; i < n && i >= 0; --i)   out[i] = high;
    } else if (strat == "manual") {
        // leave to override
    } else {
        LOG_WRN("tq-v-strategy '%s' not recognised — using base type uniformly\n", strategy.c_str());
    }
}

// Enforce budget by downgrading hottest-layer HIGHs to BASE, coldest-layer BASEs to LOW
static void apply_budget(
        std::vector<ggml_type> & out,
        const std::vector<layer_summary> & s,
        ggml_type base, ggml_type low, ggml_type high,
        float budget_bpw)
{
    if (budget_bpw <= 0.0f) return;

    auto avg_bpw = [&]() {
        float sum = 0.0f;
        for (ggml_type t : out) sum += bpw_per_elem(t);
        return sum / (float) out.size();
    };

    // Rank layers by var_max ascending (least sensitive first → first to downgrade)
    std::vector<int> rank(s.size());
    for (size_t i = 0; i < s.size(); ++i) rank[i] = (int) i;
    std::sort(rank.begin(), rank.end(), [&](int a, int b) { return s[a].var_max < s[b].var_max; });

    // Step 1: downgrade HIGH → BASE on least-sensitive HIGH layers
    for (int idx : rank) {
        if (avg_bpw() <= budget_bpw) return;
        const int li = s[idx].layer_idx;
        if (li >= 0 && li < (int) out.size() && out[li] == high) {
            out[li] = base;
        }
    }
    // Step 2: downgrade BASE → LOW on least-sensitive BASE layers
    for (int idx : rank) {
        if (avg_bpw() <= budget_bpw) return;
        const int li = s[idx].layer_idx;
        if (li >= 0 && li < (int) out.size() && out[li] == base) {
            out[li] = low;
        }
    }
}

// ---------- public resolver ----------

std::vector<ggml_type> resolve_v_types(
        const std::string & profile_path,
        const std::string & strategy,
        ggml_type base, ggml_type low, ggml_type high,
        const std::string & override_spec,
        int n_layer,
        float budget_bpw,
        std::string * err_out)
{
    // Nothing requested at all → empty (caller falls back to uniform)
    if (strategy.empty() && override_spec.empty() && profile_path.empty()) {
        return {};
    }

    std::vector<ggml_type> out(n_layer, base);

    // Apply profile-driven strategy if profile present
    if (!profile_path.empty() && !strategy.empty() && to_lower(strategy) != "manual" && to_lower(strategy) != "auto") {
        std::string err;
        profile p = load_profile(profile_path, &err);
        if (p.layers.empty()) {
            if (err_out) *err_out = "profile load failed: " + err;
            LOG_WRN("tq-v-profile: %s — falling back to base type\n", err.c_str());
        } else {
            auto s = summarize(p);
            apply_strategy(out, s, strategy, base, low, high);
            apply_budget(out, s, base, low, high, budget_bpw);
        }
    } else if (!strategy.empty() && (to_lower(strategy) == "auto")) {
        // profile-free heuristic
        std::vector<layer_summary> empty;
        apply_strategy(out, empty, strategy, base, low, high);
    }

    // Apply override (highest priority)
    if (!override_spec.empty()) {
        std::string err;
        auto ov = parse_override(override_spec, n_layer, &err);
        if (!err.empty()) {
            if (err_out) *err_out = "override parse failed: " + err;
            LOG_WRN("tq-v-override: %s — ignoring overrides\n", err.c_str());
        } else {
            for (int i = 0; i < n_layer; ++i) {
                if (ov[i] != GGML_TYPE_COUNT) out[i] = ov[i];
            }
        }
    }

    // Log the resulting plan once
    {
        int cnt_low = 0, cnt_base = 0, cnt_high = 0, cnt_other = 0;
        for (ggml_type t : out) {
            if      (t == low)  cnt_low++;
            else if (t == base) cnt_base++;
            else if (t == high) cnt_high++;
            else                cnt_other++;
        }
        float avg = 0.0f;
        for (ggml_type t : out) avg += bpw_per_elem(t);
        avg /= (float) out.size();

        LOG_INF("tq-v-mixed: n_layer=%d  %s=%d  %s=%d  %s=%d  other=%d  avg_bpw=%.2f\n",
                n_layer,
                ggml_type_name(low),  cnt_low,
                ggml_type_name(base), cnt_base,
                ggml_type_name(high), cnt_high,
                cnt_other, avg);
    }

    return out;
}

}  // namespace llama_tq
