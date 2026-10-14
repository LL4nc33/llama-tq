// Checks the state snapshots of ggml_gated_delta_net on the CPU backend: a run over n_tokens with K snapshot
// slots must give the same attention output as n_tokens single-token runs that carry the state, and slot s must
// hold the state s tokens before the last one (slot 0 = final state) for s < min(n_tokens, K).
// Covers the scalar gate, the vector gate (KDA) and the folded gate activations (raw gates).

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct gdn_case {
    int64_t S;       // head size (S_k == S_v)
    int64_t H_k;     // q/k heads
    int64_t H;       // value heads
    int64_t n_seqs;
    int64_t n_tokens;
    int64_t K;
    bool    kda;
    bool    raw;
};

struct gdn_inputs {
    std::vector<float> q, k, v, g, beta, s0, dt_bias, a;
};

struct gdn_result {
    std::vector<float> attn;   // [S, H, n_tokens, n_seqs]
    std::vector<float> slots;  // K x [S, S, H, n_seqs]
};

static void fill(std::vector<float> & dst, size_t n, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    dst.resize(n);
    for (auto & x : dst) {
        x = dist(rng);
    }
}

static void l2_normalize_rows(std::vector<float> & x, int64_t S) {
    for (size_t r = 0; r < x.size() / S; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < S; ++i) {
            sum += (double) x[r*S + i] * x[r*S + i];
        }
        const float scale = (float) (1.0 / std::sqrt(sum + 1e-6));
        for (int64_t i = 0; i < S; ++i) {
            x[r*S + i] *= scale;
        }
    }
}

// one gated_delta_net op over T tokens; the inputs are given for these T tokens only
static gdn_result run_op(const gdn_case & c, int64_t T, int64_t K, const gdn_inputs & in) {
    const int64_t G = c.kda ? c.S : 1;

    ggml_init_params params = { 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.H_k, T, c.n_seqs);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.H_k, T, c.n_seqs);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.H,   T, c.n_seqs);
    ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, G,   c.H,   T, c.n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   c.H,   T, c.n_seqs);
    ggml_tensor * s0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.S,   c.H, c.n_seqs);

    memcpy(q->data,    in.q.data(),    ggml_nbytes(q));
    memcpy(k->data,    in.k.data(),    ggml_nbytes(k));
    memcpy(v->data,    in.v.data(),    ggml_nbytes(v));
    memcpy(g->data,    in.g.data(),    ggml_nbytes(g));
    memcpy(beta->data, in.beta.data(), ggml_nbytes(beta));
    memcpy(s0->data,   in.s0.data(),   ggml_nbytes(s0));

    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, s0, K);
    if (c.raw) {
        ggml_tensor * dt_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.H);
        ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.H);
        memcpy(dt_bias->data, in.dt_bias.data(), ggml_nbytes(dt_bias));
        memcpy(a->data,       in.a.data(),       ggml_nbytes(a));
        ggml_gated_delta_net_set_raw_gates(out, dt_bias, a);
    }

    // slots that the op does not write keep this marker
    std::fill((float *) out->data, (float *) out->data + ggml_nelements(out), NAN);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const int64_t n_attn = c.S*c.H*T*c.n_seqs;
    const int64_t n_snap = c.S*c.S*c.H*c.n_seqs;
    const float * data = (const float *) out->data;

    gdn_result res;
    res.attn.assign(data, data + n_attn);
    res.slots.assign(data + n_attn, data + n_attn + K*n_snap);

    ggml_free(ctx);
    return res;
}

// the inputs of token t of every sequence
static gdn_inputs token_inputs(const gdn_case & c, const gdn_inputs & all, int64_t t, const std::vector<float> & state) {
    const int64_t G = c.kda ? c.S : 1;
    const int64_t T = c.n_tokens;

    auto slice = [&](const std::vector<float> & src, int64_t row) {
        // src is [row, T, n_seqs] with row floats per token
        std::vector<float> dst(row*c.n_seqs);
        for (int64_t s = 0; s < c.n_seqs; ++s) {
            memcpy(dst.data() + s*row, src.data() + (s*T + t)*row, row*sizeof(float));
        }
        return dst;
    };

    gdn_inputs in;
    in.q       = slice(all.q,    c.S*c.H_k);
    in.k       = slice(all.k,    c.S*c.H_k);
    in.v       = slice(all.v,    c.S*c.H);
    in.g       = slice(all.g,    G*c.H);
    in.beta    = slice(all.beta, c.H);
    in.s0      = state;
    in.dt_bias = all.dt_bias;
    in.a       = all.a;
    return in;
}

static bool close(float a, float b) {
    return std::fabs(a - b) <= 1e-5f + 1e-4f*std::fabs(b);
}

static bool check(const gdn_case & c, std::mt19937 & rng) {
    const int64_t G = c.kda ? c.S : 1;
    const int64_t T = c.n_tokens;

    gdn_inputs in;
    fill(in.q,    c.S*c.H_k*T*c.n_seqs, rng, -1.0f, 1.0f);
    fill(in.k,    c.S*c.H_k*T*c.n_seqs, rng, -1.0f, 1.0f);
    fill(in.v,    c.S*c.H*T*c.n_seqs,   rng, -1.0f, 1.0f);
    fill(in.s0,   c.S*c.S*c.H*c.n_seqs, rng, -0.5f, 0.5f);
    if (c.raw) {
        // pre-activation gates; the op applies sigmoid and a*softplus(g + dt_bias) itself
        fill(in.g,       G*c.H*T*c.n_seqs, rng, -3.0f, 1.0f);
        fill(in.beta,    c.H*T*c.n_seqs,   rng, -2.0f, 2.0f);
        fill(in.dt_bias, c.H,              rng, -1.0f, 1.0f);
        fill(in.a,       c.H,              rng, -1.0f, -0.1f);
    } else {
        fill(in.g,    G*c.H*T*c.n_seqs, rng, -0.5f, -0.01f); // log decay
        fill(in.beta, c.H*T*c.n_seqs,   rng, 0.1f, 0.9f);
    }
    l2_normalize_rows(in.q, c.S);
    l2_normalize_rows(in.k, c.S);

    const gdn_result full = run_op(c, T, c.K, in);

    // reference: single-token runs, keeping the state after every token
    const int64_t n_snap = c.S*c.S*c.H*c.n_seqs;
    std::vector<std::vector<float>> states; // states[t] = state after token t
    std::vector<float> ref_attn(c.S*c.H*T*c.n_seqs);
    std::vector<float> state = in.s0;
    for (int64_t t = 0; t < T; ++t) {
        const gdn_result step = run_op(c, 1, 1, token_inputs(c, in, t, state));
        for (int64_t s = 0; s < c.n_seqs; ++s) {
            memcpy(ref_attn.data() + (s*T + t)*c.S*c.H, step.attn.data() + s*c.S*c.H, c.S*c.H*sizeof(float));
        }
        state.assign(step.slots.begin(), step.slots.begin() + n_snap);
        states.push_back(state);
    }

    int n_bad = 0;
    for (size_t i = 0; i < ref_attn.size(); ++i) {
        if (!close(full.attn[i], ref_attn[i])) {
            if (n_bad++ < 3) {
                fprintf(stderr, "  attn[%zu] = %g, expected %g\n", i, full.attn[i], ref_attn[i]);
            }
        }
    }

    const int64_t n_written = std::min(T, c.K);
    for (int64_t slot = 0; slot < c.K; ++slot) {
        const float * got = full.slots.data() + slot*n_snap;
        if (slot >= n_written) {
            // unwritten slots are left untouched
            for (int64_t i = 0; i < n_snap; ++i) {
                if (!std::isnan(got[i])) {
                    if (n_bad++ < 3) {
                        fprintf(stderr, "  slot %lld was written although n_tokens = %lld\n", (long long) slot, (long long) T);
                    }
                    break;
                }
            }
            continue;
        }
        const std::vector<float> & want = states[T - 1 - slot];
        for (int64_t i = 0; i < n_snap; ++i) {
            if (!close(got[i], want[i])) {
                if (n_bad++ < 3) {
                    fprintf(stderr, "  slot %lld [%lld] = %g, expected %g\n", (long long) slot, (long long) i, got[i], want[i]);
                }
            }
        }
    }

    printf("%s: S=%lld H_k=%lld H=%lld n_seqs=%lld n_tokens=%3lld K=%lld kda=%d raw=%d\n", n_bad == 0 ? "OK  " : "FAIL",
        (long long) c.S, (long long) c.H_k, (long long) c.H, (long long) c.n_seqs, (long long) c.n_tokens, (long long) c.K,
        c.kda, c.raw);
    return n_bad == 0;
}

int main() {
    std::mt19937 rng(42);

    std::vector<gdn_case> cases;
    for (int64_t K : {1, 2, 9}) {
        for (int64_t n_tokens : {1, 2, 3, 8, 9, 10, 64}) {
            cases.push_back({16, 2, 4, 2, n_tokens, K, false, false});
            cases.push_back({16, 2, 4, 1, n_tokens, K, true,  false});
            cases.push_back({32, 4, 4, 2, n_tokens, K, false, true});
        }
    }

    int n_fail = 0;
    for (const auto & c : cases) {
        n_fail += check(c, rng) ? 0 : 1;
    }

    printf("%d/%zu cases passed\n", (int) (cases.size() - n_fail), cases.size());
    return n_fail == 0 ? 0 : 1;
}
