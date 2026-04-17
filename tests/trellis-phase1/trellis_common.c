// Shared utilities: RNG, timing, fp16 conversion.

#include "trellis_phase1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
