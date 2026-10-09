// Gradients of ggml_flash_attn_ext (GGML_OP_FLASH_ATTN_BACK through ggml_build_backward_expand) against
// float64 central differences of a float64 reimplementation of the forward pass. Norm-relative errors, so
// gradients close to zero do not dominate the result as in the element-wise check of test-backend-ops.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct fa_config {
    int D, DV, H, R, SEQ, KV, N; // head sizes, KV heads, query heads per KV head, sequences, keys, queries
    bool mask, sinks;
    float max_bias, softcap;
};

struct fa_data {
    fa_config c;
    std::vector<double> q, k, v, w, sinks;
    std::vector<double> mask; // [SEQ][N][KV], -inf for masked keys

    double loss() const {
        const int HQ = c.H*c.R;
        const double scale = 1.0/sqrt((double) c.D);
        const unsigned n_head_log2 = 1u << (unsigned) floor(log2(HQ));
        const double m0 = pow(2.0, -c.max_bias/n_head_log2);
        const double m1 = pow(2.0, -(c.max_bias/2.0)/n_head_log2);
        std::vector<double> s(c.KV);
        double L = 0.0;
        for (int sq = 0; sq < c.SEQ; ++sq) {
            for (int h = 0; h < HQ; ++h) {
                const int hk = h/c.R;
                const double slope = c.max_bias > 0.0f ? (h < (int) n_head_log2 ? pow(m0, h + 1) : pow(m1, 2*(h - n_head_log2) + 1)) : 1.0;
                for (int i = 0; i < c.N; ++i) {
                    double M = c.sinks ? sinks[h] : -INFINITY;
                    for (int j = 0; j < c.KV; ++j) {
                        const double mv = c.mask ? slope*mask[(sq*c.N + i)*c.KV + j] : 0.0;
                        if (std::isinf(mv)) {
                            s[j] = -INFINITY;
                            continue;
                        }
                        double d = 0.0;
                        for (int e = 0; e < c.D; ++e) {
                            d += q[((sq*HQ + h)*c.N + i)*c.D + e]*k[((sq*c.H + hk)*c.KV + j)*c.D + e];
                        }
                        d *= scale;
                        if (c.softcap != 0.0f) {
                            d = c.softcap*tanh(d/c.softcap);
                        }
                        s[j] = d + mv;
                        M = std::max(M, s[j]);
                    }
                    double sum = c.sinks ? exp(sinks[h] - M) : 0.0;
                    for (int j = 0; j < c.KV; ++j) {
                        s[j] = std::isinf(s[j]) ? 0.0 : exp(s[j] - M);
                        sum += s[j];
                    }
                    for (int e = 0; e < c.DV; ++e) {
                        double o = 0.0;
                        for (int j = 0; j < c.KV; ++j) {
                            o += s[j]/sum*v[((sq*c.H + hk)*c.KV + j)*c.DV + e];
                        }
                        L += w[((sq*c.N + i)*HQ + h)*c.DV + e]*o; // output layout [DV, HQ, N, SEQ]
                    }
                }
            }
        }
        return L;
    }
};

static double check(const fa_config & c, int seed) {
    const int HQ = c.H*c.R;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D,  c.N,  HQ,  c.SEQ);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D,  c.KV, c.H, c.SEQ);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.DV, c.KV, c.H, c.SEQ);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.DV, HQ,   c.N, c.SEQ);
    ggml_tensor * m = c.mask  ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.KV, c.N, 1, c.SEQ) : nullptr;
    ggml_tensor * s = c.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HQ) : nullptr;
    ggml_set_param(q);
    ggml_set_param(k);
    ggml_set_param(v);
    if (s) {
        ggml_set_param(s);
    }

    fa_data data;
    data.c = c;
    auto fill = [&](ggml_tensor * t, std::vector<double> & dst) {
        dst.resize(ggml_nelements(t));
        for (size_t i = 0; i < dst.size(); ++i) {
            ((float *) t->data)[i] = u(rng);
            dst[i] = ((float *) t->data)[i];
        }
    };
    fill(q, data.q);
    fill(k, data.k);
    fill(v, data.v);
    fill(w, data.w);
    if (s) {
        fill(s, data.sinks);
    }
    if (m) {
        // keys after i + KV - N are masked, plus random holes; key 0 stays visible in every row
        data.mask.resize(ggml_nelements(m));
        for (int sq = 0; sq < c.SEQ; ++sq) {
            for (int i = 0; i < c.N; ++i) {
                for (int j = 0; j < c.KV; ++j) {
                    float mv = j <= i + c.KV - c.N ? (u(rng) > 0.8f ? -INFINITY : u(rng)) : -INFINITY;
                    if (j == 0) {
                        mv = 0.0f;
                    }
                    const ggml_fp16_t h = ggml_fp32_to_fp16(mv);
                    ((ggml_fp16_t *) m->data)[(sq*c.N + i)*c.KV + j] = h;
                    data.mask[(sq*c.N + i)*c.KV + j] = ggml_fp16_to_fp32(h);
                }
            }
        }
    }

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), c.max_bias, c.softcap);
    ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, out, w));
    ggml_set_loss(loss);

    ggml_cgraph * gb = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gb, loss);
    ggml_build_backward_expand(ctx, gb, nullptr);
    ggml_graph_reset(gb);
    ggml_cplan plan = ggml_graph_plan(gb, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size + 1);
    plan.work_data = work.data();
    ggml_graph_compute(gb, &plan);

    struct param { ggml_tensor * t; std::vector<double> * x; };
    std::vector<param> params = { {q, &data.q}, {k, &data.k}, {v, &data.v} };
    if (s) {
        params.push_back({s, &data.sinks});
    }
    double worst = 0.0;
    for (auto & p : params) {
        const float * g = (const float *) ggml_graph_get_grad(gb, p.t)->data;
        double ref2 = 0.0, err2 = 0.0;
        for (size_t i = 0; i < p.x->size(); ++i) {
            const double x0 = (*p.x)[i];
            const double eps = 1e-6;
            (*p.x)[i] = x0 + eps; const double lp = data.loss();
            (*p.x)[i] = x0 - eps; const double lm = data.loss();
            (*p.x)[i] = x0;
            const double ref = (lp - lm)/(2*eps);
            ref2 += ref*ref;
            err2 += (ref - g[i])*(ref - g[i]);
        }
        worst = std::max(worst, sqrt(err2/std::max(ref2, 1e-30)));
    }
    ggml_free(ctx);
    return worst;
}

int main() {
    const fa_config configs[] = {
        // D  DV  H  R SEQ KV  N   mask   sinks  max_bias softcap
        { 32, 32, 2, 1, 1, 13, 5,  true,  false, 0.0f,    0.0f  },
        { 32, 32, 2, 4, 1, 13, 5,  true,  false, 0.0f,    0.0f  }, // GQA
        { 32, 32, 2, 2, 2,  7, 3,  true,  false, 0.0f,    0.0f  }, // two sequences
        { 32, 32, 2, 1, 1, 13, 5,  true,  false, 0.0f,    2.0f  }, // softcap
        { 32, 32, 4, 1, 1, 13, 5,  true,  false, 8.0f,    0.0f  }, // ALiBi
        { 32, 32, 2, 2, 1, 13, 5,  true,  true,  0.0f,    0.0f  }, // sinks
        { 32, 32, 2, 1, 1,  9, 4,  false, false, 0.0f,    0.0f  }, // no mask
        { 64, 32, 1, 1, 1, 11, 3,  true,  false, 0.0f,    0.0f  }, // DV != DK
        {128, 128, 1, 2, 1, 20, 6,  true,  true,  0.0f,    30.0f }, // D 128, everything on
    };
    int n_fail = 0;
    for (const auto & c : configs) {
        for (int seed = 1; seed <= 2; ++seed) {
            const double err = check(c, seed);
            const bool ok = err < 1e-5;
            n_fail += !ok;
            printf("%s D=%d DV=%d H=%d R=%d SEQ=%d KV=%d N=%d mask=%d sinks=%d max_bias=%g softcap=%g seed=%d: rel err %.2e\n",
                   ok ? "OK  " : "FAIL", c.D, c.DV, c.H, c.R, c.SEQ, c.KV, c.N, c.mask, c.sinks, c.max_bias, c.softcap, seed, err);
        }
    }
    printf("%s\n", n_fail == 0 ? "all flash attention gradients match" : "flash attention gradient mismatch");
    return n_fail == 0 ? 0 : 1;
}
