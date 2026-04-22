// Round-trip tests for Trick 4 "Correction Overlay Buffer" CPU helpers.
// See docs/plans/2026-04-20-trick4-correction-overlay-design.md

#include "ggml.h"
#include "../ggml/src/ggml-trellis.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define QK 256

static double mse(const float * a, const float * b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / n;
}

// Generates a "heavy-tailed" 256-sample block: mostly ~N(0,1) with K large
// outliers injected at random positions. Exercise case for overlay.
static void gen_heavy_tailed(float * out, int n_outliers, std::mt19937 & rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int i = 0; i < QK; i++) out[i] = gauss(rng);

    std::uniform_int_distribution<int> pos(0, QK - 1);
    std::uniform_real_distribution<float> mag(6.0f, 10.0f);
    std::bernoulli_distribution sign(0.5);

    for (int k = 0; k < n_outliers; k++) {
        const int p = pos(rng);
        out[p] = (sign(rng) ? 1.0f : -1.0f) * mag(rng);
    }
}

// Test 1: top-1 extraction picks the position with largest |src - decoded|.
static int test_top1_argmax() {
    float src[QK];
    float dec[QK];

    std::mt19937 rng(123);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int i = 0; i < QK; i++) src[i] = gauss(rng);
    for (int i = 0; i < QK; i++) dec[i] = src[i] + 0.1f * gauss(rng);

    // Inject a known large error at a known position.
    const int P = 137;
    const float injected = 5.0f;
    dec[P] = src[P] - injected;

    uint8_t entries[4];
    const int n_valid = ggml_trellis_overlay_extract(src, dec, 1, 0.0f, entries);
    if (n_valid != 1) {
        fprintf(stderr, "test_top1_argmax: expected 1 valid, got %d\n", n_valid);
        return 1;
    }
    const uint8_t pos   = entries[0];
    const uint8_t flags = entries[1];
    if (!(flags & 0x1)) {
        fprintf(stderr, "test_top1_argmax: valid bit not set\n");
        return 1;
    }
    if (pos != (uint8_t)P) {
        fprintf(stderr, "test_top1_argmax: expected pos %d, got %d\n", P, (int)pos);
        return 1;
    }
    printf("  test_top1_argmax: OK (pos=%d)\n", (int)pos);
    return 0;
}

// Test 2: overlay_apply restores the exact fp16 value at the stored position.
static int test_apply_restores_value() {
    float src[QK];
    float dec[QK];
    std::mt19937 rng(456);
    for (int i = 0; i < QK; i++) src[i] = 0.0f;
    for (int i = 0; i < QK; i++) dec[i] = 0.0f;

    const int P = 42;
    const float v = 3.14159f;
    src[P] = v;
    // decoded has 0 at that position => large error => top-1 picks it

    uint8_t entries[4];
    ggml_trellis_overlay_extract(src, dec, 1, 0.0f, entries);

    ggml_trellis_overlay_apply(entries, 1, dec);

    // fp16 round-trip: the applied value equals ggml_fp32_to_fp16 then back.
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    const float v_rt = ggml_fp16_to_fp32(h);
    if (std::fabs(dec[P] - v_rt) > 1e-6f) {
        fprintf(stderr, "test_apply_restores_value: expected %g, got %g\n", v_rt, dec[P]);
        return 1;
    }
    // Unrelated positions untouched
    for (int i = 0; i < QK; i++) {
        if (i == P) continue;
        if (dec[i] != 0.0f) {
            fprintf(stderr, "test_apply_restores_value: pos %d corrupted: %g\n", i, dec[i]);
            return 1;
        }
    }
    printf("  test_apply_restores_value: OK (%g -> %g)\n", v, dec[P]);
    return 0;
}

// Test 3: invalid flag => apply is a no-op.
static int test_invalid_is_noop() {
    float y[QK];
    for (int i = 0; i < QK; i++) y[i] = 1.0f;

    uint8_t entry[4] = { 50, 0x0 /* valid=0 */, 0, 0 };
    ggml_trellis_overlay_apply(entry, 1, y);
    if (y[50] != 1.0f) {
        fprintf(stderr, "test_invalid_is_noop: position was modified\n");
        return 1;
    }
    printf("  test_invalid_is_noop: OK\n");
    return 0;
}

// Test 4: threshold gating — tiny relative errors marked invalid.
static int test_threshold_gating() {
    float src[QK], dec[QK];
    for (int i = 0; i < QK; i++) { src[i] = 10.0f; dec[i] = 10.0f; }
    // Inject a tiny error: rel = 0.001, well below default threshold 0.25.
    dec[7] = 9.99f;

    uint8_t entry[4];
    const int n_valid = ggml_trellis_overlay_extract(src, dec, 1, 0.25f, entry);
    if (n_valid != 0) {
        fprintf(stderr, "test_threshold_gating: expected 0 valid, got %d\n", n_valid);
        return 1;
    }
    if (entry[1] & 0x1) {
        fprintf(stderr, "test_threshold_gating: valid bit set despite threshold\n");
        return 1;
    }
    printf("  test_threshold_gating: OK (dropped below threshold)\n");
    return 0;
}

// Test 5: End-to-end round trip with real trellis encode/decode (VTQ2_2).
// Expect overlay-corrected MSE << uncorrected MSE on heavy-tailed blocks.
static int test_roundtrip_improves_mse() {
    std::mt19937 rng(789);

    const int NB = 32;                  // number of blocks to average
    double sum_mse_plain = 0.0;
    double sum_mse_fix   = 0.0;

    for (int b = 0; b < NB; b++) {
        float src[QK];
        gen_heavy_tailed(src, 3, rng);  // 3 outliers per block

        // Encode with VTQ2_2 (K=2): this is the highest-error config.
        uint16_t ss = 0;
        float    d  = 0.0f;
        uint8_t  qs[64] = {0};
        ggml_trellis_encode_group(src, 2, &ss, &d, qs);

        float dec[QK];
        ggml_trellis_decode_group(ss, 2, d, qs, dec);

        const double m_plain = mse(src, dec, QK);

        // Extract top-1 overlay + apply
        uint8_t entry[4];
        ggml_trellis_overlay_extract(src, dec, 1, 0.0f, entry);
        ggml_trellis_overlay_apply(entry, 1, dec);

        const double m_fix = mse(src, dec, QK);

        sum_mse_plain += m_plain;
        sum_mse_fix   += m_fix;
    }

    const double avg_plain = sum_mse_plain / NB;
    const double avg_fix   = sum_mse_fix   / NB;

    printf("  test_roundtrip_improves_mse: plain=%.6g  fix=%.6g  ratio=%.3f\n",
           avg_plain, avg_fix, avg_fix / avg_plain);

    if (!(avg_fix < avg_plain)) {
        fprintf(stderr, "test_roundtrip_improves_mse: overlay did not reduce MSE!\n");
        return 1;
    }
    // Expect at least 5% reduction for heavy-tailed noise (outliers dominate).
    if (!(avg_fix < 0.95 * avg_plain)) {
        fprintf(stderr, "test_roundtrip_improves_mse: reduction <5%% (%.3f)\n",
                avg_fix / avg_plain);
        return 1;
    }
    return 0;
}

int main(void) {
    printf("test-vtq-overlay: Trick 4 correction overlay CPU round-trip\n");
    int rc = 0;
    rc |= test_top1_argmax();
    rc |= test_apply_restores_value();
    rc |= test_invalid_is_noop();
    rc |= test_threshold_gating();
    rc |= test_roundtrip_improves_mse();
    if (rc) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
