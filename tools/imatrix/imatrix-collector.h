// imatrix-collector.h
//
// Reusable importance-matrix (imatrix) activation collector, extracted from imatrix.cpp so that
// other tools can register it as a backend eval callback and gather an imatrix from their own
// forward passes (e.g. the diffusion-gemma CLI's denoise loop, instead of the causal prefill).
//
// This is the COLLECTOR part only (Stats / filter_tensor_name / IMatrixCollector::collect_imatrix /
// save_imatrix{,_legacy} / g_collector / ik_collect_imatrix). The imatrix-specific compute/PPL/
// statistics loop (compute_imatrix / show_statistics / load_imatrix*) stays in imatrix.cpp.
//
// Header-only: all methods are inline and the global g_collector / ik_collect_imatrix wrapper use
// C++17 `inline` linkage, so the header may be included in multiple translation units without ODR
// violations and without an extra .cpp / CMake target.
//
// Enable gate: the collector starts DISABLED (m_enabled == false). collect_imatrix() returns
// immediately (collecting nothing, and signalling no interest to the scheduler) while disabled.
// Call enable()/disable() to bracket exactly the forward passes that should contribute. imatrix.cpp
// enables it for its whole run; the diffusion CLI enables it only around the denoise decode.

#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct Stats {
    std::vector<float>   values;
    std::vector<int64_t> counts;
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
inline std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix_legacy(int32_t ncall = -1) const;
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix_legacy(const char * fname);
    bool load_imatrix(const char * file_name);
    const std::unordered_map<std::string, Stats> & get_mstats() const { return m_stats; }
    // collection gate: while disabled, collect_imatrix() gathers nothing and reports no interest
    // to the scheduler. Bracket the forward passes that should contribute with enable()/disable().
    void enable()  { m_enabled = true;  }
    void disable() { m_enabled = false; }
    bool is_enabled() const { return m_enabled; }
private:
    std::unordered_map<std::string, Stats> m_stats;
    common_params                          m_params;
    std::mutex                             m_mutex;
    std::vector<std::string>               m_datasets;
    int32_t                                m_last_chunk = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
    bool                                   m_enabled = false; // collection gate (off by default)
};

inline bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    // collection gate: when disabled, report no interest (ask) and collect nothing (!ask). This
    // guarantees nothing is gathered outside an enable()/disable() bracket, and the scheduler is
    // not asked to materialize any tensor on our behalf.
    if (!m_enabled) {
        return false;
    }

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight"))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *) src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int64_t n_as = src0->ne[2];
        const int64_t n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
        if (ggml_nrows(src1) != src1->ne[1] * src1->ne[2]) {
            LOG_ERR("%s: tensor has more than 3 dimensions: %s", __func__, wname.c_str());
            GGML_ASSERT(false);
        }

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0]*n_as));
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != (size_t)n_as) {
            LOG_ERR("%s: inconsistent expert count for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.counts.size(), (int)n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int64_t ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex*src1->ne[0];

            for (int64_t idx = 0; idx < n_ids; ++idx) {
                for (int64_t row = 0; row < src1->ne[2]; ++row) {
                    const int excur = *(const int32_t *) (m_ids.data() + row*ids->nb[1] + idx*ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check

                    if (excur != ex) continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float * x = (const float *)(data + i11*src1->nb[1] + i12*src1->nb[2]);

                    e.counts[ex]++;

                    for (int64_t j = 0; j < src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j] * x[j];
                        if (!std::isfinite((float)e.values[e_start + j])) {
                            LOG_ERR("%f detected in %s\n", (float)e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
            const int32_t n_chunk = e.counts[ex] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    } else {
        auto & e = m_stats[wname];
        const int64_t n_mat = src0->ne[2] * src0->ne[3];

        // use a single count per dense tensor
        // (necessary when merging older GGUF-imatrix files with 3d tensors)
        if (e.counts.size() > 1) {
            bool all_equal = true;
            for (size_t i = 1; i < e.counts.size(); ++i) {
                if (e.counts[0] != e.counts[i]) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                e.counts.resize(1);
            }
        }
        if (e.values.empty()) {
            e.values.resize(src1->ne[0] * n_mat, 0);
            e.counts.resize(1, 0);
        }
        else if (e.values.size() != (size_t)(src1->ne[0] * n_mat)) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0] * n_mat));
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2], (int)src1->type);

        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                // handle 3D+ tensors, but flatten 3D+ activations when model tensor is 2D
                const int64_t mat_id = (i3 % src0->ne[3]) * src0->ne[2] + (i2 % src0->ne[2]);
                const int64_t mat_start = mat_id * src1->ne[0];

                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *) (data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                    for (int64_t j = 0; j < src1->ne[0]; ++j) {
                        e.values[mat_start + j] += x[j] * x[j];
                        if (!std::isfinite((float)e.values[j])) {
                            LOG_ERR("%f detected in %s\n", (float)e.values[j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
        }
        // only 1 count in practice, except when a tensor is used for both MUL_MAT_ID and MUL_MAT
        for (size_t i = 0; i < e.counts.size(); ++i) {
            e.counts[i] += ggml_nrows(src1) / n_mat;
            const int32_t n_chunk = e.counts[i] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    }

    return true;
}

inline void IMatrixCollector::save_imatrix_legacy(int32_t ncall) const {
    auto fname = m_params.out_file;

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // warn when writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        // ceiling division to avoid accidental zeros
        const int32_t ncall = (*std::max_element(stat.counts.begin(), stat.counts.end()) + (chunk_size - 1)) / chunk_size;
        out.write((const char *) &ncall, sizeof(ncall));
        const int32_t nval = stat.values.size();
        const int32_t nmat = stat.counts.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0 && nmat > 0) {
            std::vector<float> tmp(nval);
            for (int32_t i = 0; i < nval; i++) {
                float count = static_cast<float>(stat.counts[i / (nval / nmat)]);
                float value = stat.values[i];
                if (count == 0.0f) {
                    // store 1 for partial data
                    value = 1.0f;
                    count = 1.0f;
                }
                tmp[i] = (value / count) * static_cast<float>(ncall);
            }
            out.write((const char *) tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_chunk, sizeof(m_last_chunk));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        const char * dataset_file = m_params.prompt_file.c_str();
        int32_t len = m_params.prompt_file.size();
        // When there is no prompt but there were other imatrix files loaded, use the last dataset
        if (m_params.prompt_file.empty() && !m_datasets.empty()) {
            const std::string & dataset_str = m_datasets[m_datasets.size() - 1];
            dataset_file = dataset_str.c_str();
            len = dataset_str.size();
        }
        out.write((const char *) &len, sizeof(len));
        out.write(dataset_file, len);
    }

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());
}

inline void IMatrixCollector::save_imatrix(int32_t n_chunk) const {
    auto fname = m_params.out_file;
    int8_t use_legacy_format = m_params.imat_dat;

    if (use_legacy_format > 0) {
        this->save_imatrix_legacy(n_chunk);
        return;
    }
    // only warn when `--output-format gguf` is not specified
    if (use_legacy_format == 0 && !string_ends_with(fname, ".gguf")) {
        LOG_WRN("\n%s: saving imatrix using GGUF format with a different suffix than .gguf\n", __func__);
        LOG_WRN("%s: if you want the previous imatrix format, use --output-format dat\n", __func__);
    }

    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }

    // write imatrix entries even if they don't have full data. (can be corrected when reading)
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    std::vector<std::string> to_store;
    size_t data_size = 0;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        int n_zeros = 0;
        for (const auto c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back(kv.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    {
        std::vector<const char *> datasets;
        datasets.reserve(m_datasets.size() + 1);
        for (size_t i = 0; i < m_datasets.size(); ++i) {
            datasets.push_back(m_datasets[i].c_str());
        }
        if (!m_params.prompt_file.empty()) {
            datasets.push_back(m_params.prompt_file.c_str());
        }

        gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
        // Write the dataset paths
        gguf_set_arr_str(ctx_gguf, "imatrix.datasets", datasets.data(), datasets.size());
        // Write the number of chunks the matrix was computed with
        gguf_set_val_u32(ctx_gguf, "imatrix.chunk_count", m_last_chunk);
        gguf_set_val_u32(ctx_gguf, "imatrix.chunk_size", m_params.n_ctx / m_params.n_parallel);
    }

    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t nval = (int32_t) stat.values.size();
        const int32_t nmat = (int32_t) stat.counts.size();
        if (nval > 0 && nmat > 0) {
            struct ggml_tensor * in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
            struct ggml_tensor * counts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
            ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
            ggml_format_name(counts, "%s.counts", name.c_str());

            for (int32_t j = 0; j < nval; ++j) {
                ((float *) in_sum2->data)[j] = (float) stat.values[j];
            }
            for (int32_t j = 0; j < nmat; ++j) {
                ((float *) counts->data)[j] = (float) stat.counts[j];
            }

            gguf_add_tensor(ctx_gguf, in_sum2);
            gguf_add_tensor(ctx_gguf, counts);
        }
    }

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

inline bool IMatrixCollector::load_imatrix_legacy(const char * fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n", __func__, fname);
        return false;
    }
    int n_entries;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, fname);
        return false;
    }
    // Guess the chunk size because it's not stored in the file
    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    for (int i = 0; i < n_entries; ++i) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        std::vector<char> name_as_vec(len + 1);
        in.read((char *) name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERR("%s: failed reading name for entry %d from %s\n", __func__, i + 1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{ name_as_vec.data() };
        auto & e = m_stats[std::move(name)];
        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERR("%s: failed reading number of values for entry %d\n", __func__, i);
            m_stats = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0.0f);
            e.counts.resize(1, 0);
        }

        std::vector<float> tmp(nval);
        in.read((char *) tmp.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERR("%s: failed reading data for entry %d\n", __func__, i);
            m_stats = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and correct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i] * chunk_size;
        }
        // The legacy format doesn't distinguish the counts for different experts
        for (size_t j = 0; j < e.counts.size(); ++j) {
            e.counts[j] += ncall * chunk_size;
        }
    }

    {
        // TODO: extract into its own method; this is also used by the GGUF-based format
        // Calculate the last chunk count
        int64_t max_count = 0;
        for (const auto & stats : m_stats) {
            for (int64_t count : stats.second.counts) {
                if (count > max_count) {
                    max_count = count;
                }
            }
        }
        m_last_chunk = max_count / (chunk_size);
    }

    {
        // Read the number of calls the matrix was computed with
        int32_t n_calls;
        in.read((char *) &n_calls, sizeof(n_calls));
        // ignore it because it's not important
    }

    // Read the dataset path to include it when writing to GGUF
    if (!in.fail()){
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        if (!in.fail()) {
            std::vector<char> dataset;
            dataset.resize(len + 1, 0);
            in.read(dataset.data(), len);
            if (!in.fail()) {
                m_datasets.push_back(dataset.data());
            }
        }
    }

    return true;
}

// Using GGUF as the file format, for greater extensibility
inline bool IMatrixCollector::load_imatrix(const char * file_name) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(file_name, meta_gguf_params);
    if (!ctx_gguf) {
        return this->load_imatrix_legacy(file_name);
    }
    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, file_name);
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return false;
    }

    const int64_t datasets_key = gguf_find_key(ctx_gguf, "imatrix.datasets");
    if (datasets_key != -1 && gguf_get_arr_type(ctx_gguf, datasets_key) == GGUF_TYPE_STRING) {
        const int64_t n = gguf_get_arr_n(ctx_gguf, datasets_key);
        m_datasets.reserve(m_datasets.size() + n);
        for (int64_t i = 0; i < n; ++i) {
            m_datasets.push_back(gguf_get_arr_str(ctx_gguf, datasets_key, i));
        }
    }

    const std::string in_sum2_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    // Could re-use m_stats instead, but this allows
    // checking for completeness of *each* loaded imatrix file
    // and also makes it easier to re-use a similar implementation in quantize.cpp
    // Using an ordered map to get a deterministic iteration order.
    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) { continue; }

        if (string_remove_suffix(name, in_sum2_suffix)) {
            // in_sum2
            sums_counts_for[std::move(name)].first = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            // counts
            sums_counts_for[std::move(name)].second = cur;
        } else {
            // ignore other tensors
        }
    }

    for (const auto & sc : sums_counts_for) {
        const std::string &        name    = sc.first;
        const struct ggml_tensor * in_sum2 = sc.second.first;
        const struct ggml_tensor * counts  = sc.second.second;

        if (!in_sum2 || !counts) {
            LOG_ERR("%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        auto & e = m_stats[name];

        int64_t nval = ggml_nelements(in_sum2);
        if (e.values.empty()) {
            e.values.resize(nval, 0.0f);
        } else if ((size_t) nval != e.values.size()) {
            LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        int64_t ncounts = ggml_nelements(counts);
        if (e.counts.empty()) {
            e.counts.resize(ncounts, 0);
        } else if (e.counts.size() == 1 && ncounts > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(ncounts, e.counts[0]);
        } else if ((size_t) ncounts != e.counts.size()) {
            LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) ncounts, e.counts.size());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        // Recreate the state as expected by save_imatrix()
        for (int64_t j = 0; j < nval; j++) {
            e.values[j] += ((const float *) in_sum2->data)[j];
        }
        for (int64_t j = 0; j < ncounts; j++) {
            e.counts[j] += std::lround(((const float *) counts->data)[j]);
        }
    }

    // TODO: extract into its own method; this is also used by the legacy format
    // Calculate the last chunk count
    int64_t max_count = 0;
    for (const auto & stats : m_stats) {
        for (int64_t count : stats.second.counts) {
            if (count > max_count) {
                max_count = count;
            }
        }
    }
    m_last_chunk = max_count / (m_params.n_ctx / m_params.n_parallel);

    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return true;
}

// Global collector + backend eval callback wrapper. `inline` (C++17) gives them external linkage
// with a single shared definition across all translation units that include this header.
inline IMatrixCollector g_collector;

inline bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}
