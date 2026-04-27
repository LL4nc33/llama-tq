#include "router-profile.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <utility>

router_profile_data::router_profile_data() = default;

router_profile_data::router_profile_data(const std::string & out_path, float tau_, int max_tokens_)
    : filter("^ffn_moe_probs-([0-9]+)$", std::regex::optimize),
      max_tokens(max_tokens_ > 0 ? max_tokens_ : 4096),
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
    : fp(o.fp), filter(std::move(o.filter)), n_expert(o.n_expert),
      max_tokens(o.max_tokens), tau(o.tau), scratch(std::move(o.scratch)),
      token_idx(o.token_idx), header_written(o.header_written) {
    o.fp = nullptr;
}

router_profile_data & router_profile_data::operator=(router_profile_data && o) noexcept {
    if (this != &o) {
        if (fp) std::fclose(fp);
        fp             = o.fp;
        filter         = std::move(o.filter);
        n_expert       = o.n_expert;
        max_tokens     = o.max_tokens;
        tau            = o.tau;
        scratch        = std::move(o.scratch);
        token_idx      = o.token_idx;
        header_written = o.header_written;
        o.fp = nullptr;
    }
    return *this;
}

static void write_header(router_profile_data & d, int n_expert) {
    uint8_t hdr[32] = { 0 };
    std::memcpy(hdr + 0, "TQRP", 4);
    const uint32_t version = 1;
    const uint32_t ne      = (uint32_t) n_expert;
    const uint32_t reserved = 0;
    std::memcpy(hdr +  4, &version,  4);
    std::memcpy(hdr +  8, &ne,       4);
    std::memcpy(hdr + 12, &reserved, 4);
    std::memcpy(hdr + 16, &d.tau,    4);
    // bytes 20..31 reserved/padding (zeroed)
    std::fwrite(hdr, 1, sizeof(hdr), d.fp);
    d.header_written = true;
    d.n_expert       = n_expert;
}

bool router_profile_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (router_profile_data *) user_data;
    if (!d || !d->fp) {
        return false;
    }

    if (ask) {
        // Only request data for tensors named ffn_moe_probs-<N>.
        return std::regex_search(t->name, d->filter);
    }

    // Past max_tokens? Stop accepting more records — but keep returning true
    // so the graph keeps executing.
    if (d->token_idx >= (int64_t) d->max_tokens) {
        return true;
    }

    // Re-check filter (defensive — ggml may invoke ask=false without prior ask=true).
    std::cmatch m;
    if (!std::regex_search(t->name, m, d->filter)) {
        return true;
    }
    const int layer_idx = std::atoi(m[1].first);

    // Expected shape: [n_expert, n_tokens] (F32). Some arches use F16 — handle both.
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return true;
    }

    const int64_t n_expert = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (n_expert <= 0 || n_tokens <= 0) {
        return true;
    }

    if (!d->header_written) {
        write_header(*d, (int) n_expert);
    } else if ((int64_t) d->n_expert != n_expert) {
        // n_expert mismatch across layers — should not happen.
        return true;
    }

    // Stage tensor to host.
    const size_t nbytes = ggml_nbytes(t);
    if (d->scratch.size() < nbytes) {
        d->scratch.resize(nbytes);
    }
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * src = is_host ? (const uint8_t *) t->data : d->scratch.data();
    if (!is_host) {
        ggml_backend_tensor_get(t, d->scratch.data(), 0, nbytes);
    }

    // Per-token slot: [n_expert] floats.
    // For F16, convert to F32 row-by-row to write canonical F32 stream.
    std::vector<float> row_f32;
    if (t->type == GGML_TYPE_F16) {
        row_f32.resize(n_expert);
    }

    const size_t row_stride = t->nb[1];
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        if (d->token_idx >= (int64_t) d->max_tokens) {
            break;
        }
        const uint8_t * row_ptr = src + tok * row_stride;

        const float * row_data;
        if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * h = (const ggml_fp16_t *) row_ptr;
            for (int64_t e = 0; e < n_expert; ++e) {
                row_f32[e] = ggml_fp16_to_fp32(h[e]);
            }
            row_data = row_f32.data();
        } else {
            row_data = (const float *) row_ptr;
        }

        // Record header: token_idx u32, layer_idx u16, n_expert u16.
        const uint32_t tok_u32 = (uint32_t) d->token_idx;
        const uint16_t lay_u16 = (uint16_t) layer_idx;
        const uint16_t ne_u16  = (uint16_t) n_expert;
        std::fwrite(&tok_u32, 4, 1, d->fp);
        std::fwrite(&lay_u16, 2, 1, d->fp);
        std::fwrite(&ne_u16,  2, 1, d->fp);
        std::fwrite(row_data, sizeof(float), n_expert, d->fp);

        ++d->token_idx;
    }

    return true;
}
