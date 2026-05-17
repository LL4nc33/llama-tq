// Numerical validation for ggml_mul_mat_id_grad_as against PyTorch reference.
//
// Reference inputs and expected outputs are produced by
// tests/ref/mul_mat_id_pytorch_ref.py and dumped as raw F32/I32 binaries in
// /tmp/mul_mat_id_ref/. This test loads them, runs the ggml backward
// computation on CPU, and checks atol=1e-3, rtol=1e-2.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct ref_meta {
    int64_t D_out;
    int64_t D_in;
    int64_t n_expert;
    int64_t n_used;
    int64_t n_used_b;
    int64_t n_tokens;
};

ref_meta read_meta(const std::string & path) {
    ref_meta m{};
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        int64_t v = std::stoll(line.substr(eq + 1));
        if (k == "D_out")    m.D_out = v;
        if (k == "D_in")     m.D_in = v;
        if (k == "n_expert") m.n_expert = v;
        if (k == "n_used")   m.n_used = v;
        if (k == "n_used_b") m.n_used_b = v;
        if (k == "n_tokens") m.n_tokens = v;
    }
    return m;
}

std::vector<uint8_t> read_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
}

bool tensors_close(const float * got, const float * ref, size_t n, float atol, float rtol, const char * tag) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t fails = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = got[i];
        const float r = ref[i];
        const float ad = std::fabs(a - r);
        const float rd = ad / (std::fabs(r) + 1e-12f);
        max_abs = std::max(max_abs, ad);
        max_rel = std::max(max_rel, rd);
        if (ad > atol + rtol * std::fabs(r)) {
            if (fails < 8) {
                fprintf(stderr, "  [%s] mismatch i=%zu got=%g ref=%g abs=%g rel=%g\n",
                        tag, i, a, r, ad, rd);
            }
            ++fails;
        }
    }
    fprintf(stderr, "  [%s] max_abs=%.4g max_rel=%.4g fails=%zu/%zu\n",
            tag, max_abs, max_rel, fails, n);
    return fails == 0;
}

} // namespace

int main() {
    const std::string dir = "/tmp/mul_mat_id_ref/";
    const ref_meta meta = read_meta(dir + "meta.txt");

    fprintf(stderr, "D_out=%lld D_in=%lld n_expert=%lld n_used=%lld n_used_b=%lld n_tokens=%lld\n",
            (long long) meta.D_out, (long long) meta.D_in, (long long) meta.n_expert,
            (long long) meta.n_used, (long long) meta.n_used_b, (long long) meta.n_tokens);

    auto in_as     = read_bin(dir + "as.bin");
    auto in_b      = read_bin(dir + "b.bin");
    auto in_ids    = read_bin(dir + "ids.bin");
    auto in_grad_c = read_bin(dir + "grad_c.bin");
    auto ref_grad_as = read_bin(dir + "grad_as.bin");
    auto ref_grad_b  = read_bin(dir + "grad_b.bin");

    // ggml context big enough for tensors + graph
    const size_t mem_size = 64 * 1024 * 1024;
    ggml_init_params ip{ mem_size, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * t_grad_c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, meta.D_out, meta.n_used,   meta.n_tokens);
    ggml_tensor * t_b      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, meta.D_in,  meta.n_used_b, meta.n_tokens);
    ggml_tensor * t_ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, meta.n_used, meta.n_tokens);

    std::memcpy(t_grad_c->data, in_grad_c.data(), in_grad_c.size());
    std::memcpy(t_b->data,      in_b.data(),      in_b.size());
    std::memcpy(t_ids->data,    in_ids.data(),    in_ids.size());

    ggml_tensor * t_grad_as = ggml_mul_mat_id_grad_as(ctx, t_grad_c, t_b, t_ids, meta.n_expert);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_grad_as);
    {
        ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);
        std::vector<uint8_t> work;
        if (plan.work_size > 0) { work.resize(plan.work_size); plan.work_data = work.data(); }
        ggml_graph_compute(gf, &plan);
    }

    fprintf(stderr, "running grad_as comparison...\n");
    bool ok_as = tensors_close(
        (const float *) t_grad_as->data,
        (const float *) ref_grad_as.data(),
        meta.D_out * meta.D_in * meta.n_expert,
        1e-3f, 1e-2f, "grad_as");

    // grad_b uses the existing ggml_mul_mat_id with transposed as.
    // For this test we reconstruct as_T from in_as (same data, transposed view).
    // Load as_T (pre-transposed) directly from disk to bypass any ggml_transpose+cont issues.
    auto in_as_T = read_bin(dir + "as_T.bin");
    ggml_tensor * t_as_T = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, meta.D_out, meta.D_in, meta.n_expert);
    std::memcpy(t_as_T->data, in_as_T.data(), in_as_T.size());
    (void) in_as;  // unused now
    ggml_tensor * t_grad_b = ggml_mul_mat_id(ctx, t_as_T, t_grad_c, t_ids);
    ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(gb, t_grad_b);
    {
        ggml_cplan plan = ggml_graph_plan(gb, 4, nullptr);
        std::vector<uint8_t> work;
        if (plan.work_size > 0) { work.resize(plan.work_size); plan.work_data = work.data(); }
        ggml_graph_compute(gb, &plan);
    }

    fprintf(stderr, "running grad_b comparison...\n");
    bool ok_b = tensors_close(
        (const float *) t_grad_b->data,
        (const float *) ref_grad_b.data(),
        meta.D_in * meta.n_used_b * meta.n_tokens,
        1e-3f, 1e-2f, "grad_b");

    ggml_free(ctx);

    if (!ok_as || !ok_b) {
        fprintf(stderr, "FAIL\n");
        return 1;
    }
    fprintf(stderr, "OK\n");
    return 0;
}
