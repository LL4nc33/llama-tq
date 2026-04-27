#include "router-profile.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <utility>

router_profile_data::router_profile_data() = default;

router_profile_data::router_profile_data(const std::string & out_path, float tau_, int max_tokens_)
    : filter_logits("^ffn_moe_logits-([0-9]+)$", std::regex::optimize),
      filter_topk  ("^ffn_moe_topk-([0-9]+)$",   std::regex::optimize),
      max_tokens(max_tokens_ > 0 ? max_tokens_ : 256),
      tau(tau_) {
    fp = std::fopen(out_path.c_str(), "wb");
    if (!fp) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, out_path.c_str());
    }
}

router_profile_data::~router_profile_data() {
    if (fp) {
        std::fflush(fp);
        std::fclose(fp);
        fp = nullptr;
    }
}

router_profile_data::router_profile_data(router_profile_data && o) noexcept
    : fp(o.fp),
      filter_logits(std::move(o.filter_logits)),
      filter_topk(std::move(o.filter_topk)),
      n_expert(o.n_expert), n_expert_used(o.n_expert_used),
      max_tokens(o.max_tokens), tau(o.tau),
      scratch(std::move(o.scratch)),
      token_idx(o.token_idx),
      per_layer_count(std::move(o.per_layer_count)),
      per_layer_count_topk(std::move(o.per_layer_count_topk)),
      header_written(o.header_written) {
    o.fp = nullptr;
}

router_profile_data & router_profile_data::operator=(router_profile_data && o) noexcept {
    if (this != &o) {
        if (fp) std::fclose(fp);
        fp                   = o.fp;
        filter_logits        = std::move(o.filter_logits);
        filter_topk          = std::move(o.filter_topk);
        n_expert             = o.n_expert;
        n_expert_used        = o.n_expert_used;
        max_tokens           = o.max_tokens;
        tau                  = o.tau;
        scratch              = std::move(o.scratch);
        token_idx            = o.token_idx;
        per_layer_count      = std::move(o.per_layer_count);
        per_layer_count_topk = std::move(o.per_layer_count_topk);
        header_written       = o.header_written;
        o.fp = nullptr;
    }
    return *this;
}

static void write_header(router_profile_data & d, int n_expert, int n_expert_used) {
    uint8_t hdr[32] = { 0 };
    // bumped magic to TQR2 to flag the new (tagged) record format.
    std::memcpy(hdr + 0, "TQR2", 4);
    const uint32_t version  = 2;
    const uint32_t ne       = (uint32_t) n_expert;
    const uint32_t neu      = (uint32_t) n_expert_used;
    std::memcpy(hdr +  4, &version, 4);
    std::memcpy(hdr +  8, &ne,      4);
    std::memcpy(hdr + 12, &neu,     4);
    std::memcpy(hdr + 16, &d.tau,   4);
    // bytes 20..31 reserved/padding (zeroed)
    std::fwrite(hdr, 1, sizeof(hdr), d.fp);
    d.header_written = true;
    d.n_expert       = n_expert;
    d.n_expert_used  = n_expert_used;
}

static void stage_to_host(router_profile_data & d, struct ggml_tensor * t) {
    const size_t nbytes = ggml_nbytes(t);
    if (d.scratch.size() < nbytes) {
        d.scratch.resize(nbytes);
    }
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        ggml_backend_tensor_get(t, d.scratch.data(), 0, nbytes);
    }
}

static const uint8_t * host_ptr(const router_profile_data & d, const struct ggml_tensor * t) {
    return ggml_backend_buffer_is_host(t->buffer)
        ? (const uint8_t *) t->data
        : d.scratch.data();
}

static void write_record_logits(router_profile_data & d, int layer_idx,
                                int64_t n_expert, const float * row) {
    const uint32_t tok_u32 = (uint32_t) d.token_idx;
    const uint16_t lay_u16 = (uint16_t) layer_idx;
    const uint8_t  tag     = 'L';
    const uint8_t  cnt     = (uint8_t) std::min<int64_t>(n_expert, 255);
    (void) cnt; // count is in header.n_expert; keep field for record alignment.
    std::fwrite(&tok_u32, 4, 1, d.fp);
    std::fwrite(&lay_u16, 2, 1, d.fp);
    std::fwrite(&tag,     1, 1, d.fp);
    const uint8_t pad = 0; std::fwrite(&pad, 1, 1, d.fp);
    std::fwrite(row, sizeof(float), n_expert, d.fp);
}

static void write_record_topk(router_profile_data & d, int layer_idx,
                              int64_t n_expert_used, const int32_t * row) {
    const uint32_t tok_u32 = (uint32_t) d.token_idx;
    const uint16_t lay_u16 = (uint16_t) layer_idx;
    const uint8_t  tag     = 'K';
    std::fwrite(&tok_u32, 4, 1, d.fp);
    std::fwrite(&lay_u16, 2, 1, d.fp);
    std::fwrite(&tag,     1, 1, d.fp);
    const uint8_t pad = 0; std::fwrite(&pad, 1, 1, d.fp);
    std::fwrite(row, sizeof(int32_t), n_expert_used, d.fp);
}

bool router_profile_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (router_profile_data *) user_data;
    if (!d || !d->fp) {
        return false;
    }

    if (ask) {
        return std::regex_search(t->name, d->filter_logits)
            || std::regex_search(t->name, d->filter_topk);
    }

    // Match against either filter and dispatch.
    std::cmatch m;
    const bool is_logits = std::regex_search(t->name, m, d->filter_logits);
    const bool is_topk   = !is_logits && std::regex_search(t->name, m, d->filter_topk);
    if (!is_logits && !is_topk) {
        return true;
    }
    const int layer_idx = std::atoi(m[1].first);

    auto & counter = is_logits ? d->per_layer_count : d->per_layer_count_topk;
    if ((int) counter.size() <= layer_idx) {
        counter.resize(layer_idx + 1, 0);
    }
    if (counter[layer_idx] >= (int64_t) d->max_tokens) {
        return true;
    }

    const int64_t n_rows  = is_logits ? t->ne[0] : t->ne[0];   // n_expert / n_expert_used
    const int64_t n_tokens = t->ne[1];
    if (n_rows <= 0 || n_tokens <= 0) {
        return true;
    }

    if (is_logits) {
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
            return true;
        }
    } else {
        // topk tensor is I32.
        if (t->type != GGML_TYPE_I32) {
            return true;
        }
    }

    if (!d->header_written) {
        // First emission seeds n_expert (from logits) and n_expert_used (from topk).
        // We may see either first; record what we know and patch on the second emission
        // — but to keep the format simple, only initialize when both are known.
        // Easier: write header with what we have, leave the missing field as 0 and
        // fix it later with a second header rewrite. Simpler still: postpone record
        // emission until logits is seen first (logits > topk in nb of records anyway).
        if (is_logits) {
            write_header(*d, (int) n_rows, /*n_expert_used*/ 0);
        } else {
            // Skip topk records until we've seen at least one logits to write the header.
            return true;
        }
    } else if (is_logits && (int64_t) d->n_expert != n_rows) {
        return true;
    } else if (is_topk && d->n_expert_used == 0) {
        // Patch the header in place once we see the first topk.
        d->n_expert_used = (int) n_rows;
        const long pos = std::ftell(d->fp);
        std::fseek(d->fp, 12, SEEK_SET);
        const uint32_t neu = (uint32_t) n_rows;
        std::fwrite(&neu, 4, 1, d->fp);
        std::fseek(d->fp, pos, SEEK_SET);
    }

    stage_to_host(*d, t);
    const uint8_t * src = host_ptr(*d, t);

    std::vector<float> row_f32;
    if (is_logits && t->type == GGML_TYPE_F16) {
        row_f32.resize(n_rows);
    }

    const size_t row_stride = t->nb[1];
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        if (counter[layer_idx] >= (int64_t) d->max_tokens) {
            break;
        }
        const uint8_t * row_ptr = src + tok * row_stride;

        if (is_logits) {
            const float * row_data;
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * h = (const ggml_fp16_t *) row_ptr;
                for (int64_t e = 0; e < n_rows; ++e) {
                    row_f32[e] = ggml_fp16_to_fp32(h[e]);
                }
                row_data = row_f32.data();
            } else {
                row_data = (const float *) row_ptr;
            }
            write_record_logits(*d, layer_idx, n_rows, row_data);
        } else {
            const int32_t * row_data = (const int32_t *) row_ptr;
            write_record_topk(*d, layer_idx, n_rows, row_data);
        }

        ++d->token_idx;
        ++counter[layer_idx];
    }

    return true;
}
