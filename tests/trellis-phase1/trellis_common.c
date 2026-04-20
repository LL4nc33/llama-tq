// Shared utilities: RNG, timing, fp16 conversion.

#include "trellis_phase1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float uniform01(void);

// --- RNG: SplitMix64 + Box-Muller ---

static uint64_t g_rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t splitmix64(void) {
    uint64_t z = (g_rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float uniform01(void) {
    uint64_t bits = splitmix64();
    return (float)((bits >> 11) * (1.0 / (double)(1ULL << 53)));
}

void trellis_seed_rng(uint64_t seed) {
    g_rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

void trellis_gen_gaussian(float * buf, size_t n) {
    for (size_t i = 0; i + 1 < n; i += 2) {
        float u1 = uniform01();
        float u2 = uniform01();
        if (u1 < 1e-30f) u1 = 1e-30f;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 6.28318530717958647692f * u2;
        buf[i]     = r * cosf(theta);
        buf[i + 1] = r * sinf(theta);
    }
    if (n & 1) {
        float u1 = uniform01();
        float u2 = uniform01();
        if (u1 < 1e-30f) u1 = 1e-30f;
        buf[n - 1] = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
    }
}

// Laplace(0, 1): inverse CDF, unit variance needs scale 1/sqrt(2).
void trellis_gen_laplace(float * buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float u = uniform01() - 0.5f;          // (-0.5, 0.5)
        float s = (u >= 0.0f) ? 1.0f : -1.0f;
        float a = fabsf(u);
        if (a < 1e-30f) a = 1e-30f;
        buf[i] = -s * logf(1.0f - 2.0f * a) / 1.4142135624f;
    }
}

// Student-t with nu dof: chi-squared / Z. Heavy-tailed for nu small.
void trellis_gen_student_t(float * buf, size_t n, float nu) {
    for (size_t i = 0; i < n; i++) {
        // generate Gaussian Z
        float u1 = uniform01(); if (u1 < 1e-30f) u1 = 1e-30f;
        float u2 = uniform01();
        float z = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
        // chi-squared(nu) approx as sum of nu squared Gaussians (slow for large nu)
        // use rejection-free: chi2(nu) ≈ nu * (1 - 2/(9nu) + Z'/sqrt(9nu/2))^3
        float u3 = uniform01(); if (u3 < 1e-30f) u3 = 1e-30f;
        float u4 = uniform01();
        float zp = sqrtf(-2.0f * logf(u3)) * cosf(6.28318530717958647692f * u4);
        float term = 1.0f - 2.0f/(9.0f*nu) + zp / sqrtf(9.0f*nu/2.0f);
        float chi = nu * term * term * term;
        if (chi < 1e-10f) chi = 1e-10f;
        // variance of t is nu/(nu-2); rescale to unit variance
        float scale = sqrtf((nu - 2.0f) / nu);
        buf[i] = z / sqrtf(chi / nu) * scale;
    }
}

// Bimodal: 50/50 mix of N(-1.5, 0.5) and N(+1.5, 0.5). Unit variance.
void trellis_gen_bimodal(float * buf, size_t n) {
    for (size_t i = 0; i + 1 < n; i += 2) {
        float u1 = uniform01(); if (u1 < 1e-30f) u1 = 1e-30f;
        float u2 = uniform01();
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 6.28318530717958647692f * u2;
        float z0 = r * cosf(theta);
        float z1 = r * sinf(theta);
        // assign each to one mode
        float m0 = (uniform01() < 0.5f) ? -1.5f : 1.5f;
        float m1 = (uniform01() < 0.5f) ? -1.5f : 1.5f;
        // scale: total var = E[X²] - E[X]² = (0.25 + 2.25) - 0 = 2.5, std ≈ 1.58
        buf[i]     = (0.5f * z0 + m0) / 1.5811388f;
        buf[i + 1] = (0.5f * z1 + m1) / 1.5811388f;
    }
    if (n & 1) buf[n - 1] = 0.0f;
}

// V-cache like: base Gaussian + 5% outlier samples scaled to ~4-6 sigma.
// Simulates a post-rotation tensor that is near-Gaussian but not perfectly so.
void trellis_gen_vcache_like(float * buf, size_t n) {
    trellis_gen_gaussian(buf, n);
    for (size_t i = 0; i < n; i++) {
        if (uniform01() < 0.05f) {
            float sign = (buf[i] >= 0.0f) ? 1.0f : -1.0f;
            buf[i] = sign * (4.0f + 2.0f * uniform01());  // 4-6 sigma
        }
    }
}

// LLM V-cache realistic model (post-rotation):
// - Base Gaussian N(0, 1) from RHT effect
// - 1% of samples are outliers at 5-10 sigma (channel-specific even after
//   rotation; RHT distributes but does not eliminate them)
// - Optional: block-level scaling variance (some blocks are "quiet",
//   others carry most of the energy)
// Variance of the mixture is normalized to ~1.0 for fair comparison.
void trellis_gen_vcache_realistic(float * buf, size_t n) {
    trellis_gen_gaussian(buf, n);
    for (size_t i = 0; i < n; i++) {
        if (uniform01() < 0.01f) {
            float sign = (uniform01() < 0.5f) ? -1.0f : 1.0f;
            buf[i] = sign * (5.0f + 5.0f * uniform01());  // 5-10 sigma
        }
    }
    // Renormalize to unit variance
    double s1 = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) { s1 += buf[i]; s2 += (double)buf[i]*buf[i]; }
    double m = s1/n, v = s2/n - m*m;
    float scale = (v > 1e-12) ? (float)(1.0 / sqrt(v)) : 1.0f;
    for (size_t i = 0; i < n; i++) buf[i] = ((float)(buf[i] - m)) * scale;
}

int trellis_load_binary(const char * path, float * buf, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    return (got == n) ? 0 : -2;
}

// --- Timing ---

double trellis_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}
