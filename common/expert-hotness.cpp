#include "expert-hotness.h"

#include "log.h"

#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

using json = nlohmann::ordered_json;

bool expert_hotness_load(const std::string & path, expert_hotness & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_WRN("%s: cannot open '%s'\n", __func__, path.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        LOG_ERR("%s: JSON parse error in '%s': %s\n", __func__, path.c_str(), e.what());
        return false;
    }

    try {
        out.n_expert      = j.value("n_expert",      0);
        out.n_expert_used = j.value("n_expert_used", 0);
        out.top_k         = j.value("top_k",         0);
        out.model_name    = j.value("model_name",    std::string{});

        if (out.n_expert <= 0) {
            LOG_ERR("%s: n_expert missing or non-positive in '%s'\n", __func__, path.c_str());
            return false;
        }
        if (!j.contains("layers") || !j["layers"].is_object()) {
            LOG_ERR("%s: 'layers' object missing in '%s'\n", __func__, path.c_str());
            return false;
        }

        // Find max layer index, then size the vector and fill.
        int max_lid = -1;
        for (auto it = j["layers"].begin(); it != j["layers"].end(); ++it) {
            const int lid = std::stoi(it.key());
            if (lid > max_lid) max_lid = lid;
        }
        if (max_lid < 0) {
            LOG_ERR("%s: 'layers' empty in '%s'\n", __func__, path.c_str());
            return false;
        }
        out.per_layer.assign(max_lid + 1, {});

        for (auto it = j["layers"].begin(); it != j["layers"].end(); ++it) {
            const int lid = std::stoi(it.key());
            const auto & arr = it.value();
            if (!arr.is_array()) continue;
            std::vector<int32_t> ids;
            ids.reserve(arr.size());
            for (const auto & v : arr) {
                const int eid = v.get<int>();
                if (eid < 0 || eid >= out.n_expert) {
                    LOG_WRN("%s: layer %d expert id %d out of range [0,%d), dropping\n",
                            __func__, lid, eid, out.n_expert);
                    continue;
                }
                ids.push_back(eid);
            }
            out.per_layer[lid] = std::move(ids);
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: schema error in '%s': %s\n", __func__, path.c_str(), e.what());
        return false;
    }

    LOG_INF("%s: loaded expert hotness profile '%s' (n_expert=%d, n_layers=%d, top_k=%d)\n",
            __func__, path.c_str(), out.n_expert, out.n_layers(), out.top_k);
    return true;
}

// Persistent ptr arrays: ggml_cpu_set_expert_hotness keeps the pointers; we
// must own backing storage that outlives the call. Stored statically here.
static std::vector<const int32_t *> g_install_hot_per_layer;
static std::vector<int>             g_install_n_per_layer;

void expert_hotness_install_cpu(const expert_hotness & h) {
    if (!h.valid()) {
        ggml_cpu_set_expert_hotness(nullptr, nullptr, 0);
        g_install_hot_per_layer.clear();
        g_install_n_per_layer.clear();
        return;
    }
    const int n_layers = h.n_layers();
    g_install_hot_per_layer.assign(n_layers, nullptr);
    g_install_n_per_layer.assign(n_layers, 0);
    for (int il = 0; il < n_layers; ++il) {
        const auto & v = h.per_layer[il];
        if (!v.empty()) {
            g_install_hot_per_layer[il] = v.data();
            g_install_n_per_layer[il]   = (int) v.size();
        }
    }
    ggml_cpu_set_expert_hotness(
        g_install_hot_per_layer.data(),
        g_install_n_per_layer.data(),
        n_layers);
    LOG_INF("%s: installed expert hotness into CPU backend (%d layers, top_k=%d)\n",
            __func__, n_layers, h.top_k);
}

bool expert_hotness_compatible(const expert_hotness & h, int n_expert, int n_layer) {
    if (h.n_expert != n_expert) {
        LOG_WRN("%s: n_expert mismatch (profile=%d model=%d), disabling hotness\n",
                __func__, h.n_expert, n_expert);
        return false;
    }
    if (h.n_layers() < n_layer) {
        LOG_WRN("%s: profile covers %d layers, model has %d — partial coverage, "
                "uncovered layers will not be prefetched\n",
                __func__, h.n_layers(), n_layer);
        // Not fatal: layers without hotness data simply skip prefetch.
    }
    return true;
}
