// Round-trip test for the experimental trellis K-cache types ktq{2,3,4}_2s.
//
// A ktq*_2s block is the vtq*_2s trellis applied to the block after the KTQ rotation (a 128-point randomized Hadamard
// transform with the fixed KTQ sign pattern); dequantization decodes the trellis and undoes the rotation.
//
// What this test verifies on K-like rows (Gaussian channels, a per-channel offset and a few outlier channels):
//   * dequantize(quantize(x)) reconstructs x (a wrong inverse rotation would give a relative error near 2)
//   * the rotation pays off on outlier channels: lower error than vtq*_2s, which codes the same block unrotated
//   * the CPU Q.K path (vec_dot) equals the dot product with the dequantized row
//   * an all-zero row decodes to zeros

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int N_ROWS   = 32;
static constexpr int ROW_SIZE = 256;  // two blocks: head_dim 256, or two heads of 128

// rows of a K cache: channel j has offset mu_j and scale sigma_j; channels 7 and 77 are outliers
static std::vector<float> make_k_rows(std::mt19937 & rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> mu(ROW_SIZE), sigma(ROW_SIZE);
    for (int j = 0; j < ROW_SIZE; j++) {
        mu[j]    = 0.5f * gauss(rng);
        sigma[j] = 0.5f + std::fabs(gauss(rng));
    }
    for (int j : {7, 77, 7 + 128}) {
        mu[j]    = 12.0f;
        sigma[j] = 4.0f;
    }
    std::vector<float> x((size_t) N_ROWS * ROW_SIZE);
    for (int r = 0; r < N_ROWS; r++) {
        for (int j = 0; j < ROW_SIZE; j++) {
            x[(size_t) r * ROW_SIZE + j] = mu[j] + sigma[j] * gauss(rng);
        }
    }
    return x;
}

static std::vector<float> round_trip(ggml_type type, const std::vector<float> & x, std::vector<uint8_t> * q_out = nullptr) {
    const size_t row_bytes = ggml_row_size(type, ROW_SIZE);
    std::vector<uint8_t> q(row_bytes * N_ROWS);
    std::vector<float>   y(x.size());
    for (int r = 0; r < N_ROWS; r++) {
        ggml_get_type_traits_cpu(type)->from_float(x.data() + (size_t) r * ROW_SIZE, q.data() + r * row_bytes, ROW_SIZE);
        ggml_get_type_traits(type)->to_float(q.data() + r * row_bytes, y.data() + (size_t) r * ROW_SIZE, ROW_SIZE);
    }
    if (q_out) {
        *q_out = std::move(q);
    }
    return y;
}

static double rel_err(const std::vector<float> & x, const std::vector<float> & y) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        num += (double) (x[i] - y[i]) * (x[i] - y[i]);
        den += (double) x[i] * x[i];
    }
    return num / den;
}

int main() {
    ggml_cpu_init();

    struct type_case {
        ggml_type ktq;      // trellis K type
        ggml_type vtq;      // same trellis without the rotation
        ggml_type ktq_cb;   // codebook K type with the same index bits (printed for reference)
        double    max_err;  // relative squared error bound on the K-like rows
    };
    // bounds: about 2x the measured errors (0.067 / 0.016 / 0.0043)
    const type_case cases[] = {
        { GGML_TYPE_KTQ2_2S, GGML_TYPE_VTQ2_2S, GGML_TYPE_KTQ2_1, 0.14  },
        { GGML_TYPE_KTQ3_2S, GGML_TYPE_VTQ3_2S, GGML_TYPE_KTQ3_1, 0.036 },
        { GGML_TYPE_KTQ4_2S, GGML_TYPE_VTQ4_2S, GGML_TYPE_KTQ4_1, 0.010 },
    };

    std::mt19937 rng(1234);
    const std::vector<float> x = make_k_rows(rng);
    std::vector<float> q_vec(ROW_SIZE);
    for (float & v : q_vec) {
        v = std::normal_distribution<float>(0.0f, 1.0f)(rng);
    }

    int n_failed = 0;
    for (const type_case & c : cases) {
        std::vector<uint8_t> q;
        const std::vector<float> y_ktq = round_trip(c.ktq, x, &q);
        const double err_ktq = rel_err(x, y_ktq);
        const double err_vtq = rel_err(x, round_trip(c.vtq, x));
        const double err_cb  = rel_err(x, round_trip(c.ktq_cb, x));

        // vec_dot of every row against one query vector vs the dot product with the dequantized row
        const size_t row_bytes = ggml_row_size(c.ktq, ROW_SIZE);
        double max_dot_diff = 0.0;
        for (int r = 0; r < N_ROWS; r++) {
            float dot = 0.0f;
            ggml_get_type_traits_cpu(c.ktq)->vec_dot(ROW_SIZE, &dot, 0, q.data() + r * row_bytes, 0, q_vec.data(), 0, 1);
            double ref = 0.0, ref_norm = 0.0;
            for (int j = 0; j < ROW_SIZE; j++) {
                ref      += (double) y_ktq[(size_t) r * ROW_SIZE + j] * q_vec[j];
                ref_norm += std::fabs((double) y_ktq[(size_t) r * ROW_SIZE + j] * q_vec[j]);
            }
            max_dot_diff = std::max(max_dot_diff, std::fabs(dot - ref) / ref_norm);
        }

        // all-zero row
        const std::vector<float> zeros(ROW_SIZE, 0.0f);
        std::vector<uint8_t> qz(row_bytes);
        std::vector<float>   yz(ROW_SIZE, 1.0f);
        ggml_get_type_traits_cpu(c.ktq)->from_float(zeros.data(), qz.data(), ROW_SIZE);
        ggml_get_type_traits(c.ktq)->to_float(qz.data(), yz.data(), ROW_SIZE);
        bool zero_ok = true;
        for (float v : yz) {
            zero_ok = zero_ok && v == 0.0f;
        }

        const bool ok = err_ktq < c.max_err && err_ktq < err_vtq && max_dot_diff < 1e-5 && zero_ok;
        n_failed += !ok;
        printf("%-8s rel. sq. error %.5f (bound %.3f, unrotated %s %.5f, codebook %s %.5f), vec_dot rel. diff %.1e, "
               "zero row %s: %s\n", ggml_type_name(c.ktq), err_ktq, c.max_err, ggml_type_name(c.vtq), err_vtq,
               ggml_type_name(c.ktq_cb), err_cb, max_dot_diff, zero_ok ? "ok" : "FAILED", ok ? "ok" : "FAILED");
    }

    printf("%d tests failed\n", n_failed);
    return n_failed > 0;
}
