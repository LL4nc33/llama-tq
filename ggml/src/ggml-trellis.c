// Trellis v2 shared code. See ggml-trellis.h for API.
// Ported from tests/trellis-phase1/trellis_{code,encdec}.c.

#include "ggml-trellis.h"

#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// --- Inverse Normal CDF (Acklam 2003) ---
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

// --- LUT: 2^16 = 65536 entries × 4 bytes = 256 KiB ---
// Thread-safe lazy init: the first thread sees g_table_ready=0, fills the
// table, then publishes the ready flag. Subsequent threads spin until ready.
static float g_table[1u << GGML_TRELLIS_L];
static atomic_int g_table_state = 0;  // 0=uninit, 1=filling, 2=ready

static void fill_table(void) {
    const size_t n = (size_t)1 << GGML_TRELLIS_L;
    for (size_t s = 0; s < n; s++) {
        uint32_t h = (uint32_t)s * 0x9E3779B1u + 0x7F4A7C15u;
        double p = ((double)(h >> 1) + 0.5) / (double)(1u << 31);
        if (p <= 0.0) p = 1e-12;
        if (p >= 1.0) p = 1.0 - 1e-12;
        g_table[s] = (float)inv_norm_cdf(p);
    }
}

const float * ggml_trellis_table(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&g_table_state, &expected, 1)) {
        fill_table();
        atomic_store(&g_table_state, 2);
    } else {
        while (atomic_load(&g_table_state) != 2) {
            // Spin — init is ~200µs, contention only at first call.
        }
    }
    return g_table;
}

// --- bit-packing helpers (little-endian K-bit emit) ---
static inline void write_bits_le(uint8_t * qs, int bit_offset, uint32_t value, int K) {
    value &= (1u << K) - 1u;
    int byte = bit_offset >> 3;
    int shift = bit_offset & 7;
    qs[byte] |= (uint8_t)((value << shift) & 0xFFu);
    if (shift + K > 8) {
        qs[byte + 1] |= (uint8_t)((value >> (8 - shift)) & 0xFFu);
        if (shift + K > 16) {
            qs[byte + 2] |= (uint8_t)((value >> (16 - shift)) & 0xFFu);
        }
    }
}

static inline uint32_t read_bits_le(const uint8_t * qs, int qs_bytes, int bit_offset, int K) {
    int byte = bit_offset >> 3;
    int shift = bit_offset & 7;
    uint32_t b0 = (byte     < qs_bytes) ? qs[byte]     : 0u;
    uint32_t b1 = (byte + 1 < qs_bytes) ? qs[byte + 1] : 0u;
    uint32_t b2 = (byte + 2 < qs_bytes) ? qs[byte + 2] : 0u;
    uint32_t w = b0 | (b1 << 8) | (b2 << 16);
    return (w >> shift) & ((1u << K) - 1u);
}

// --- Decoder: iterative shift register ---
void ggml_trellis_decode_group(
    uint16_t start_state, int K, float d, const uint8_t * qs, float * y) {
    const int L = GGML_TRELLIS_L;
    const int N = GGML_TRELLIS_QK_GROUP;
    const int qs_bytes = (N * K + 7) / 8;
    const float cb_scale = 1.0f / sqrtf((float)N);
    const float * table = ggml_trellis_table();
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = 0xFFFFu;  // L=16

    if (d == 0.0f) {
        memset(y, 0, (size_t)N * sizeof(float));
        return;
    }

    uint32_t state = (uint32_t)start_state & Lmask;
    for (int i = 0; i < N; i++) {
        uint32_t bits = read_bits_le(qs, qs_bytes, i * K, K) & Kmask;
        state = ((state >> K) | (bits << (L - K))) & Lmask;
        y[i] = table[state] * cb_scale * d;
    }
}

// --- Encoder: full Viterbi DP over 2^L states ---
// Memory: O(N · 2^L) for backtrack + O(2^L) for 2 DP rows.
//   L=16, N=512: 512·65536·4 = 128 MiB backtrack. Heavy but only at quant-time.
void ggml_trellis_encode_group(
    const float * x, int K,
    uint16_t * out_start_state, float * out_d, uint8_t * qs) {
    const int L = GGML_TRELLIS_L;
    const int N = GGML_TRELLIS_QK_GROUP;
    const uint32_t S = 1u << L;
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = S - 1u;
    const int qs_bytes = (N * K + 7) / 8;
    const float cb_scale = 1.0f / sqrtf((float)N);

    memset(qs, 0, (size_t)qs_bytes);

    // Group L2 norm
    double n2 = 0.0;
    for (int j = 0; j < N; j++) n2 += (double)x[j] * x[j];
    float norm = (float)sqrt(n2);
    if (norm < 1e-30f) {
        *out_start_state = 0;
        *out_d = 0.0f;
        return;
    }
    const float inv = 1.0f / norm;

    // normalize samples
    float * xn = (float *)malloc((size_t)N * sizeof(float));
    float * dp_cur  = (float *)malloc((size_t)S * sizeof(float));
    float * dp_next = (float *)malloc((size_t)S * sizeof(float));
    uint16_t * bt = (uint16_t *)malloc((size_t)N * S * sizeof(uint16_t));
    if (!xn || !dp_cur || !dp_next || !bt) {
        free(xn); free(dp_cur); free(dp_next); free(bt);
        *out_start_state = 0; *out_d = 0.0f; return;
    }
    for (int j = 0; j < N; j++) xn[j] = x[j] * inv;

    const float * table = ggml_trellis_table();

    // Open start: dp[0][s] = 0 for all s
    for (uint32_t s = 0; s < S; s++) dp_cur[s] = 0.0f;

    const int kshift = L - K;
    for (int i = 0; i < N; i++) {
        uint16_t * bt_i = bt + (size_t)i * S;
        for (uint32_t s = 0; s < S; s++) dp_next[s] = FLT_MAX;
        const float xi = xn[i];
        for (uint32_t prev = 0; prev < S; prev++) {
            float pc = dp_cur[prev];
            if (pc == FLT_MAX) continue;
            for (uint32_t kk = 0; kk <= Kmask; kk++) {
                uint32_t next = ((prev >> K) | (kk << kshift)) & Lmask;
                float code = table[next] * cb_scale;
                float diff = xi - code;
                float cost = pc + diff * diff;
                if (cost < dp_next[next]) {
                    dp_next[next] = cost;
                    bt_i[next] = (uint16_t)prev;
                }
            }
        }
        float * tmp = dp_cur; dp_cur = dp_next; dp_next = tmp;
    }

    // Best end state
    uint32_t best_s = 0;
    float best_c = FLT_MAX;
    for (uint32_t s = 0; s < S; s++) {
        if (dp_cur[s] < best_c) { best_c = dp_cur[s]; best_s = s; }
    }

    // Backtrack: states[0..N]
    uint32_t * states = (uint32_t *)malloc((size_t)(N + 1) * sizeof(uint32_t));
    if (!states) {
        free(xn); free(dp_cur); free(dp_next); free(bt);
        *out_start_state = 0; *out_d = 0.0f; return;
    }
    states[N] = best_s;
    for (int i = N - 1; i >= 0; i--) {
        states[i] = (uint32_t)bt[(size_t)i * S + states[i + 1]];
    }

    *out_start_state = (uint16_t)(states[0] & Lmask);

    const int kshift_emit = L - K;
    for (int i = 0; i < N; i++) {
        uint32_t bits = (states[i + 1] >> kshift_emit) & Kmask;
        write_bits_le(qs, i * K, bits, K);
    }

    // Norm-correction: scale so decoded group matches x's L2 norm.
    double recon_sq = 0.0;
    for (int i = 0; i < N; i++) {
        float code = table[states[i + 1]] * cb_scale;
        recon_sq += (double)code * code;
    }
    float recon_norm = (float)sqrt(recon_sq);
    *out_d = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;

    free(states); free(xn); free(dp_cur); free(dp_next); free(bt);
}
