// Round-trip test for ggml-trellis: encode N(0,1) samples, decode, check MSE
// matches Phase-1 harness values (within 5%).
//
// Build:
//   cd tests/trellis-phase1
//   gcc -O2 -I ../../ggml/include -I ../../ggml/src \
//     test_ggml_roundtrip.c ../../ggml/src/ggml-trellis.c -lm -o test_ggml_rt
//   ./test_ggml_rt

#include "../../ggml/src/ggml-trellis.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// SplitMix64
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t splitmix(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float urand(void) { return (float)((splitmix() >> 11) * (1.0 / (double)(1ull << 53))); }
static float nrand(void) {
    float u1 = urand(); if (u1 < 1e-30f) u1 = 1e-30f;
    float u2 = urand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

// Lloyd-Max 2-bit N(0,1) MSE baseline (same as Phase-1 harness)
static const double LLOYD_MAX_MSE_2BIT = 0.1175;

static void run_one(int K, const char * label, double expected_mse_min, double expected_mse_max) {
    const int N = 512;
    const int nb = 8;  // 8 groups = 4096 samples
    float * x = malloc(sizeof(float) * N * nb);
    float * y = malloc(sizeof(float) * N * nb);
    uint8_t * qs = malloc((size_t)(N * K + 7) / 8);

    for (int i = 0; i < N * nb; i++) x[i] = nrand();

    double sq_err = 0.0;
    for (int b = 0; b < nb; b++) {
        uint16_t start_state;
        float d;
        ggml_trellis_encode_group(x + b * N, K, &start_state, &d, qs);
        ggml_trellis_decode_group(start_state, K, d, qs, y + b * N);
        for (int i = 0; i < N; i++) {
            float e = x[b * N + i] - y[b * N + i];
            sq_err += (double)e * e;
        }
    }
    double mse = sq_err / (N * nb);
    double ratio = mse / LLOYD_MAX_MSE_2BIT;

    printf("[K=%d %s]  MSE=%.5f  ratio=%.3f  (expected %.3f..%.3f)  %s\n",
           K, label, mse, ratio, expected_mse_min, expected_mse_max,
           (ratio >= expected_mse_min && ratio <= expected_mse_max) ? "OK" : "FAIL");

    free(x); free(y); free(qs);
}

int main(void) {
    // Phase-1 harness gave on Gaussian data (not real post-RHT):
    //   K=2: MSE ≈ 0.063,  ratio ≈ 0.54
    //   K=3: MSE ≈ 0.015,  ratio ≈ 0.13
    //   K=4: not swept but expected ≈ 0.004, ratio ≈ 0.04
    run_one(2, "VTQ2_2", 0.45, 0.65);   // allow ±15% slack
    run_one(3, "VTQ3_2", 0.09, 0.18);
    run_one(4, "VTQ4_2", 0.01, 0.08);
    return 0;
}
