// Code functions: state → approximate N(0,1) sample.
// Three variants from QTIP-style literature:
//   3GAUSS:  Weyl hash expansion, 3-byte CLT sum  (cheap, GPU-friendly)
//   1MAD:    single modular multiply-add           (cheapest)
//   TABLE:   precomputed 2^L Gaussian lookup       (highest quality, memory)

#include "trellis_phase1.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// --- 3GAUSS: Weyl-sequence hash + 3-byte CLT ---
// Sum of 3 uniform bytes ≈ Gaussian (CLT approximation).
// mean = 3 * 127.5 = 382.5; var = 3 * (256² - 1) / 12 = 16383.75; sd = 128.0
static float code_3gauss(uint32_t state) {
    uint32_t h = state * 0x9E3779B1u;          // Weyl multiplier, 1 IMAD
    uint32_t b0 = (h >>  0) & 0xFFu;
    uint32_t b1 = (h >>  8) & 0xFFu;
    uint32_t b2 = (h >> 16) & 0xFFu;
    float sum = (float)(b0 + b1 + b2);
    return (sum - 382.5f) * (1.0f / 128.0f);   // normalize to ~N(0,1)
}

// --- 1MAD: single modular mul, treat high bits as uniform [-1,1] via arcsin warp ---
// Crude but cheapest; expected to underperform 3GAUSS on MSE.
static float code_1mad(uint32_t state) {
    uint32_t h = state * 0x9E3779B1u + 0x7F4A7C15u;
    // Map high 24 bits to [0,1), then inverse-CDF approximation via tanh-ish shape.
    float u = (float)(h >> 8) * (1.0f / (float)(1u << 24));
    // Simple Box-Muller-free Gaussian: tanh(5*(u-0.5)) rescaled
    float t = 5.0f * (u - 0.5f);
    float e = expf(-2.0f * t);
    float tanh_t = (1.0f - e) / (1.0f + e);
    return 2.5f * tanh_t;                      // rough sd ≈ 1
}

// --- TABLE: lazy-init precomputed inverse-CDF table for 2^L states ---
// Memory: 2^L * 4B. L=16 → 256 KB. L=20 → 4 MB. Out of scope for GPU decoder
// but useful as a high-quality MSE upper bound in Phase-1 sweeps.
static float * g_table = NULL;
static int     g_table_bits = 0;

// Inverse normal CDF (Acklam 2003 approximation), p in (0,1).
static double inv_norm_cdf(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = { 7.784695709041462e-03, 3.224671290700398e-01,
                                 2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    double q, r;
    if (p < plow) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if (p <= phigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    q = sqrt(-2.0 * log(1.0 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
}

static void ensure_table(int state_bits) {
    if (g_table && g_table_bits == state_bits) return;
    free(g_table);
    size_t n = (size_t)1 << state_bits;
    g_table = (float *)malloc(n * sizeof(float));
    g_table_bits = state_bits;
    // Hash-scrambled inverse-CDF so sequential states don't give monotone
    // codes (which would kill the trellis entropy).
    for (size_t s = 0; s < n; s++) {
        uint32_t h = (uint32_t)s * 0x9E3779B1u + 0x7F4A7C15u;
        // stratified: map to (i+0.5)/n then permute rank via hash
        double p = ((double)(h >> 1) + 0.5) / (double)(1u << 31);
        if (p <= 0.0) p = 1e-12;
        if (p >= 1.0) p = 1.0 - 1e-12;
        g_table[s] = (float)inv_norm_cdf(p);
    }
}

static float code_table(uint32_t state, int state_bits) {
    ensure_table(state_bits);
    uint32_t mask = (state_bits >= 32) ? 0xFFFFFFFFu : ((1u << state_bits) - 1u);
    return g_table[state & mask];
}

// --- Dispatch ---
float trellis_code(trellis_code_fn fn, uint32_t state, int state_bits) {
    uint32_t mask = (state_bits >= 32) ? 0xFFFFFFFFu : ((1u << state_bits) - 1u);
    state &= mask;
    switch (fn) {
        case TRELLIS_CODE_3GAUSS: return code_3gauss(state);
        case TRELLIS_CODE_1MAD:   return code_1mad(state);
        case TRELLIS_CODE_TABLE:  return code_table(state, state_bits);
    }
    return 0.0f;
}
