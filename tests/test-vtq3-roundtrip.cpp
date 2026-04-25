// Round-trip unit test for VTQ_3 (Outlier-Channel-Split, Phase 3 Step 3d).
//
// VTQ_3 layers a 4-outlier sidecar onto VTQ_2's trellis backbone:
//   1. Pick the K_OUT (=4) largest-|x| samples per 128-block, store
//      (pos, fp16 value) explicitly.
//   2. Mask those positions to 0 in the input vector before trellis encoding,
//      so Viterbi does not waste codebook entries on the long tail.
//   3. On dequant: trellis-decode all 128 samples then overwrite the K_OUT
//      outlier positions with their fp16 ground-truth.
//
// Block layouts: see ggml/src/ggml-common.h `block_vtq{2,3,4}_3`.
//
// What this test verifies:
//   * Round-trip path pick -> mask -> trellis encode -> trellis decode ->
//     outlier apply reproduces the injected outliers exactly (modulo fp16
//     quantization noise — fp16 mantissa = 10 bits, so |err|/|x| < 1e-3).
//   * The 4 outlier positions reconstruct to within 2^-9 relative error
//     (fp16 round-trip headroom).
//   * Total reconstruction MSE is strictly LOWER than the pure-VTQ_2 path
//     (no pick / apply) on the same input, for all K in {2,3,4}. This is
//     the whole point of VTQ_3 — without an MSE win the extra 12 B/block
//     would be wasted.
//
// Tolerances:
//   * fp16 outlier reconstruction: rel_err <= 2e-3
//     (fp16 has ~3.3 decimal digits; ±5σ injected values up to ~5.0 round
//      to within 5e-4 absolute, so rel_err < 1e-3 is safe but we use 2e-3.)
//   * MSE improvement: VTQ_3 MSE must be <= 0.95 * VTQ_2 MSE. The expected
//     improvement on Gaussian input with ±5σ outliers is much larger
//     (factor 5–20×) but a 5% margin keeps the test robust to RNG / encoder
//     refactors.
//
// API note: the encoder helpers `ggml_trellis_outliers_pick` and
// `ggml_trellis_outliers_apply` are being added in parallel by another
// agent. Until they land, this test ships its own reference implementations
// of the same semantics (clearly marked `LOCAL FALLBACK`). When the real
// helpers are committed, flip `VTQ3_HELPERS_AVAILABLE` to 1 to switch
// over without changing the test logic.

#include "ggml.h"
#include "../ggml/src/ggml-trellis.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef VTQ3_HELPERS_AVAILABLE
#define VTQ3_HELPERS_AVAILABLE 1  // helpers landed in ggml-trellis.{h,c}
#endif

#if VTQ3_HELPERS_AVAILABLE
extern "C" {
void ggml_trellis_outliers_pick(
        const float * x, int n_out,
        uint8_t * out_pos, float * out_val,
        float   * x_masked);

void ggml_trellis_outliers_apply(
        const uint8_t   * pos,
        const ggml_fp16_t * val,
        int               n_out,
        float           * y);
}
#else
// LOCAL FALLBACK reference implementations matching the planned API exactly.
// These are also useful as the spec the production code must satisfy.
//
// pick: select the n_out indices with largest |x|, write their positions
//       (uint8_t) and original fp32 values; produce x_masked which equals x
//       except those positions are set to 0.0f.
//
// apply: for i in [0..n_out), y[pos[i]] = fp16->fp32(val[i]).
static void ggml_trellis_outliers_pick(
        const float * x, int n_out,
        uint8_t * out_pos, float * out_val,
        float   * x_masked) {
    const int N = GGML_TRELLIS_QK_GROUP;
    std::memcpy(x_masked, x, sizeof(float) * N);

    // Index by descending |x|.
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + n_out, idx.end(),
        [&](int a, int b) { return std::fabs(x[a]) > std::fabs(x[b]); });

    for (int k = 0; k < n_out; ++k) {
        const int p = idx[k];
        out_pos[k] = (uint8_t) p;
        out_val[k] = x[p];
        x_masked[p] = 0.0f;
    }
}

static void ggml_trellis_outliers_apply(
        const uint8_t   * pos,
        const ggml_fp16_t * val,
        int               n_out,
        float           * y) {
    for (int k = 0; k < n_out; ++k) {
        y[pos[k]] = ggml_fp16_to_fp32(val[k]);
    }
}
#endif

static constexpr int N      = GGML_TRELLIS_QK_GROUP;  // 128
static constexpr int K_OUT  = 4;                       // VTQ_OUTLIER_K

// Generate Gaussian N(0,1) of length N with `n_inj` artificial outliers at ±5σ.
static void make_input(std::vector<float> & x, std::vector<int> & injected_pos,
                       uint32_t seed, int n_inj) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    x.assign(N, 0.0f);
    for (int i = 0; i < N; ++i) x[i] = gauss(rng);

    // Inject n_inj outliers at distinct random positions, alternating ±5.
    injected_pos.clear();
    std::vector<int> all(N);
    for (int i = 0; i < N; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    for (int k = 0; k < n_inj; ++k) {
        const int p = all[k];
        x[p] = (k % 2 == 0) ? 5.0f : -5.0f;
        injected_pos.push_back(p);
    }
}

static double mse(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (int i = 0; i < (int) a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        s += d * d;
    }
    return s / (double) a.size();
}

// Run the VTQ_3 round-trip path for a given K. Returns reconstruction MSE.
// Also asserts injected outlier positions are reconstructed within fp16 tol.
static double run_vtq3(int K, const std::vector<float> & x,
                       const std::vector<int> & injected_pos,
                       int & assertion_failures) {
    // Step 1: pick outliers.
    std::vector<uint8_t>     out_pos(K_OUT, 0);
    std::vector<float>       out_val(K_OUT, 0.0f);
    std::vector<float>       x_masked(N, 0.0f);
    ggml_trellis_outliers_pick(x.data(), K_OUT,
                               out_pos.data(), out_val.data(),
                               x_masked.data());

    // Sanity: the picked positions must be exactly the injected set
    // (since injected ±5 dominate Gaussian N(0,1) tails).
    for (int p : injected_pos) {
        bool found = false;
        for (int k = 0; k < K_OUT; ++k) {
            if (out_pos[k] == (uint8_t) p) { found = true; break; }
        }
        if (!found) {
            std::fprintf(stderr, "  K=%d: injected outlier pos=%d not picked\n", K, p);
            ++assertion_failures;
        }
    }

    // Step 2: convert outlier values to fp16 (production stores fp16).
    std::vector<ggml_fp16_t> outlier_val_h(K_OUT);
    for (int k = 0; k < K_OUT; ++k) {
        outlier_val_h[k] = ggml_fp32_to_fp16(out_val[k]);
    }

    // Step 3: trellis encode x_masked.
    uint16_t start_state = 0;
    float    d           = 0.0f;
    std::vector<uint8_t> qs((size_t) N * K / 8, 0);
    ggml_trellis_encode_group(x_masked.data(), K, &start_state, &d, qs.data());

    // Step 4: trellis decode.
    std::vector<float> y(N, 0.0f);
    ggml_trellis_decode_group(start_state, K, d, qs.data(), y.data());

    // Step 5: apply outliers.
    ggml_trellis_outliers_apply(out_pos.data(), outlier_val_h.data(), K_OUT, y.data());

    // Verify outlier-position reconstruction is within fp16 tolerance.
    const double tol = 2e-3;  // ~2× fp16 mantissa eps for ±5 magnitude
    for (int k = 0; k < K_OUT; ++k) {
        const int   p     = (int) out_pos[k];
        const float x_ref = x[p];
        const float y_got = y[p];
        const double rel  = std::fabs((double) y_got - (double) x_ref) /
                            std::max(std::fabs((double) x_ref), 1e-6);
        if (rel > tol) {
            std::fprintf(stderr,
                "  K=%d: outlier pos=%d reconstruct rel_err=%.3e exceeds %.1e (x=%.4f y=%.4f)\n",
                K, p, rel, tol, x_ref, y_got);
            ++assertion_failures;
        }
    }

    return mse(x, y);
}

// Run pure-VTQ_2 baseline (no pick / apply). Returns reconstruction MSE.
static double run_vtq2_baseline(int K, const std::vector<float> & x) {
    uint16_t start_state = 0;
    float    d           = 0.0f;
    std::vector<uint8_t> qs((size_t) N * K / 8, 0);
    ggml_trellis_encode_group(x.data(), K, &start_state, &d, qs.data());

    std::vector<float> y(N, 0.0f);
    ggml_trellis_decode_group(start_state, K, d, qs.data(), y.data());

    return mse(x, y);
}

int main() {
    std::printf("test-vtq3-roundtrip: VTQ_3 outlier-channel-split round-trip\n");
    std::printf("  N=%d K_OUT=%d helpers_available=%d\n", N, K_OUT, VTQ3_HELPERS_AVAILABLE);

    int assertion_failures = 0;
    int mse_failures       = 0;

    const int Ks[] = { 2, 3, 4 };
    for (int K : Ks) {
        std::vector<float> x;
        std::vector<int>   injected;
        make_input(x, injected, /*seed=*/(uint32_t) (1234 + K), /*n_inj=*/K_OUT);

        const double mse_baseline = run_vtq2_baseline(K, x);
        const double mse_vtq3     = run_vtq3(K, x, injected, assertion_failures);

        const double improvement_ratio = mse_vtq3 / std::max(mse_baseline, 1e-12);
        const bool   ok_mse = improvement_ratio < 0.95;

        std::printf("  K=%d  MSE_VTQ2=%.6e  MSE_VTQ3=%.6e  ratio=%.3f  %s\n",
                    K, mse_baseline, mse_vtq3, improvement_ratio,
                    ok_mse ? "PASS" : "FAIL");

        if (!ok_mse) ++mse_failures;
    }

    const int total = assertion_failures + mse_failures;
    if (total == 0) {
        std::printf("All VTQ_3 round-trip cases passed.\n");
        return 0;
    }
    std::fprintf(stderr,
        "FAILED: %d assertion(s), %d MSE regression(s)\n",
        assertion_failures, mse_failures);
    return 1;
}
