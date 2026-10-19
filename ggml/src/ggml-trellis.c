// Trellis v2 shared code. See ggml-trellis.h for API.
// Ported from tests/trellis-phase1/trellis_{code,encdec}.c.

#include "ggml-trellis.h"
#include "ggml.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MSVC's <stdatomic.h> requires /experimental:c11atomics which our CI
// doesn't enable. Fall back to Interlocked* for the small set of atomic
// ops we use here. Same pattern as ggml-cpu.c.
#if defined(_MSC_VER) && !defined(__clang__)
#  include <windows.h>
typedef volatile LONG ggml_trellis_atomic_int;
#  define GGML_TRELLIS_ATOMIC_LOAD(p)               InterlockedCompareExchange((p), 0, 0)
#  define GGML_TRELLIS_ATOMIC_STORE(p, v)           InterlockedExchange((p), (v))
#  define GGML_TRELLIS_ATOMIC_CAS(p, expp, newval) \
        (InterlockedCompareExchange((p), (newval), *(expp)) == *(expp))
#else
#  include <stdatomic.h>
typedef atomic_int ggml_trellis_atomic_int;
#  define GGML_TRELLIS_ATOMIC_LOAD(p)               atomic_load((p))
#  define GGML_TRELLIS_ATOMIC_STORE(p, v)           atomic_store((p), (v))
#  define GGML_TRELLIS_ATOMIC_CAS(p, expp, newval)  atomic_compare_exchange_strong((p), (expp), (newval))
#endif

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

// --- LUT: 2^16 = 65536 entries × 4 bytes = 256 KiB, the first 2^L entries serve L < 16 ---
// Thread-safe lazy init: the first thread sees g_table_ready=0, fills the
// table, then publishes the ready flag. Subsequent threads spin until ready.
static float g_table[1u << GGML_TRELLIS_L];
static ggml_trellis_atomic_int g_table_state = 0;  // 0=uninit, 1=filling, 2=ready

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
    if (GGML_TRELLIS_ATOMIC_CAS(&g_table_state, &expected, 1)) {
        fill_table();
        GGML_TRELLIS_ATOMIC_STORE(&g_table_state, 2);
    } else {
        while (GGML_TRELLIS_ATOMIC_LOAD(&g_table_state) != 2) {
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
    uint16_t start_state, int K, int L, float d, const uint8_t * qs, float * y) {
    const int N = GGML_TRELLIS_QK_GROUP;
    const int qs_bytes = (N * K + 7) / 8;
    const float cb_scale = 1.0f / sqrtf((float)N);
    const float * table = ggml_trellis_table();
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = (1u << L) - 1u;

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

// --- Encoder: Viterbi over butterflies ---
// The same DP and the same float operations in the same order as the CUDA encoder (trellis-encode.cuh), so the CPU and
// the GPU write identical blocks. The 2^K states (g << K) | e (e < 2^K) are the predecessors of the 2^K states
// g | b << (L-K); the cost of an edge depends only on its target state, so all targets of a butterfly share its best
// predecessor and the DP keeps one value per butterfly: m[g] at time i is the cost of the cheapest predecessor of
// butterfly g. A state t costs m[t & (NG-1)] + (x_i - code_t)^2 at time i+1 (one fused multiply-add, as on the GPU),
// ties keep the lowest e, and at time N the cheapest state wins (the lowest state among equal costs).
// Memory: N * 2^(L-K) bytes of backtrack (L = 16, K = 2: 2 MiB) and 2^L codes.

// sum of v[0 .. GGML_TRELLIS_QK_GROUP) in the order of a CUDA warp: lane l adds v[l], v[l + 32], ..., then the lanes
// are combined by an xor butterfly. In double, as the GPU computes norms and scales.
static double trellis_warp_order_sum(const double * v) {
    double s[32];
    for (int l = 0; l < 32; l++) {
        s[l] = 0.0;
        for (int j = l; j < GGML_TRELLIS_QK_GROUP; j += 32) {
            s[l] += v[j];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        double t[32];
        for (int l = 0; l < 32; l++) {
            t[l] = s[l] + s[l ^ off];
        }
        memcpy(s, t, sizeof(s));
    }
    return s[0];
}

// fmaf is exact everywhere, but without FMA instructions in the build it is a library call: on x86-64 Linux the DP
// loop is also compiled for FMA and the faster version is picked at load time (same results, ~10x faster)
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
#    define GGML_TRELLIS_FMA_CLONES __attribute__((target_clones("fma", "default")))
#else
#    define GGML_TRELLIS_FMA_CLONES
#endif

// the forward DP: m_cur holds the values at time 0, bt[NG .. N*NG) receives the best e per step and butterfly;
// returns the cheapest state at time N
GGML_TRELLIS_FMA_CLONES
static uint32_t trellis_viterbi_forward(const float * xn, const float * code, float * m_cur, float * m_next,
                                        uint8_t * bt, int K, uint32_t NG) {
    const uint32_t NE = 1u << K;
    float    best   = FLT_MAX;
    uint32_t best_s = 0;
    for (int step = 1; step <= GGML_TRELLIS_QK_GROUP; step++) {
        const float xi = xn[step - 1];
        for (uint32_t g = 0; g < NG; g++) {
            const float * mp = m_cur + ((g << K) & (NG - 1));
            const float * cp = code + (g << K);
            float   gbest   = FLT_MAX;
            uint8_t gbest_e = 0;
            for (uint32_t e = 0; e < NE; e++) {
                const float diff = xi - cp[e];
                const float c    = fmaf(diff, diff, mp[e]);
                if (c < gbest) { gbest = c; gbest_e = (uint8_t)e; }
            }
            if (step < GGML_TRELLIS_QK_GROUP) {
                m_next[g]         = gbest;
                bt[step * NG + g] = gbest_e;
            } else if (gbest < best || (gbest == best && ((g << K) | gbest_e) < best_s)) {
                best   = gbest;
                best_s = (g << K) | gbest_e;
            }
        }
        float * tmp = m_cur; m_cur = m_next; m_next = tmp;
    }
    return best_s;
}

void ggml_trellis_encode_group(
    const float * x, int K, int L,
    uint16_t * out_start_state, float * out_d, uint8_t * qs) {
    GGML_ASSERT(L > K && L <= GGML_TRELLIS_L);
    enum { N = GGML_TRELLIS_QK_GROUP };
    const uint32_t S        = 1u << L;
    const uint32_t NE       = 1u << K;  // predecessors per butterfly
    const uint32_t NG       = S >> K;   // butterflies
    const uint32_t Kmask    = NE - 1u;
    const int      qs_bytes = (N * K + 7) / 8;
    const float    cb_scale = 1.0f / sqrtf((float)N);

    memset(qs, 0, (size_t)qs_bytes);

    double sq[N];
    for (int j = 0; j < N; j++) sq[j] = (double)x[j] * (double)x[j];
    const float norm = (float)sqrt(trellis_warp_order_sum(sq));
    if (norm <= 1e-30f) {
        *out_start_state = 0;
        *out_d = 0.0f;
        return;
    }
    const float inv = 1.0f / norm;
    float xn[N];
    for (int j = 0; j < N; j++) xn[j] = x[j] * inv;

    float   * code   = (float   *)malloc((size_t)S * sizeof(float));
    float   * m_cur  = (float   *)malloc((size_t)NG * sizeof(float));
    float   * m_next = (float   *)malloc((size_t)NG * sizeof(float));
    uint8_t * bt     = (uint8_t *)malloc((size_t)N * NG);
    if (!code || !m_cur || !m_next || !bt) {
        free(code); free(m_cur); free(m_next); free(bt);
        *out_start_state = 0; *out_d = 0.0f; return;
    }
    const float * table = ggml_trellis_table();
    for (uint32_t t = 0; t < S; t++) code[t] = table[t] * cb_scale;

    // open start: at time 0 every state costs 0, so every butterfly's best predecessor costs 0 (e = 0)
    for (uint32_t g = 0; g < NG; g++) m_cur[g] = 0.0f;
    memset(bt, 0, NG);

    const uint32_t best_s = trellis_viterbi_forward(xn, code, m_cur, m_next, bt, K, NG);

    // backtrack: the predecessor of state st is (g << K) | e with g the low L-K bits of st
    uint32_t states[N + 1];
    uint32_t st = best_s;
    states[N] = st;
    for (int step = N - 1; step >= 0; step--) {
        const uint32_t g = st & (NG - 1);
        st = (g << K) | bt[step * NG + g];
        states[step] = st;
    }

    *out_start_state = (uint16_t)states[0];

    const int kshift_emit = L - K;
    for (int i = 0; i < N; i++) {
        uint32_t bits = (states[i + 1] >> kshift_emit) & Kmask;
        write_bits_le(qs, i * K, bits, K);
    }

    // Norm-correction: scale so decoded group matches x's L2 norm.
    double rq[N];
    for (int i = 0; i < N; i++) {
        const float c = code[states[i + 1]];
        rq[i] = (double)c * (double)c;
    }
    const float recon_norm = (float)sqrt(trellis_warp_order_sum(rq));
#if GGML_TRELLIS_UNBIASED_SCALE
    double dt[N];
    for (int i = 0; i < N; i++) dt[i] = (double)code[states[i + 1]] * (double)xn[i];
    const double dot = trellis_warp_order_sum(dt);
    *out_d = (dot > 1e-30) ? (float)((double)norm / dot) : ((recon_norm > 1e-30f) ? (norm / recon_norm) : norm);
#else
    *out_d = (recon_norm > 1e-30f) ? (norm / recon_norm) : norm;
#endif

    free(code); free(m_cur); free(m_next); free(bt);
}

// --- Outlier-Channel-Split (VTQ_3) ---
// Ranked by |x[i]| (raw magnitude, before encode).
// We pick top-N then ascending-sort by position so dequant reads them in order.
void ggml_trellis_outliers_pick(
    const float * x,
    int           n_out,
    uint8_t     * out_pos,
    float       * out_val_fp32,
    float       * x_masked) {

    const int N = GGML_TRELLIS_QK_GROUP;
    if (n_out < 0)   n_out = 0;
    if (n_out > 255) n_out = 255;

    // Default-init outputs to safe values (pos=0, val=0). If n_out > N we cap.
    if (n_out > N) n_out = N;
    for (int j = 0; j < n_out; j++) { out_pos[j] = 0; out_val_fp32[j] = 0.0f; }

    // Start with x_masked = x; we'll zero out the picked outlier slots later.
    if (x_masked != x) memcpy(x_masked, x, (size_t)N * sizeof(float));

    if (n_out == 0) return;

    // Maintain top-N descending by |x[i]| in two parallel arrays. Linear insert
    // is fine for N<=4 (worst case 4 compares per sample).
    int   top_pos[256];
    float top_mag[256];
    for (int j = 0; j < n_out; j++) {
        top_pos[j] = -1;
        top_mag[j] = -1.0f;
    }

    for (int i = 0; i < N; i++) {
        const float m = fabsf(x[i]);
        if (m > top_mag[n_out - 1]) {
            int ins = n_out - 1;
            while (ins > 0 && m > top_mag[ins - 1]) {
                top_mag[ins] = top_mag[ins - 1];
                top_pos[ins] = top_pos[ins - 1];
                ins--;
            }
            top_mag[ins] = m;
            top_pos[ins] = i;
        }
    }

    // Sort the up-to-N picked positions ascending (insertion sort, n_out tiny).
    int sorted_pos[256];
    int n_picked = 0;
    for (int j = 0; j < n_out; j++) {
        if (top_pos[j] >= 0) sorted_pos[n_picked++] = top_pos[j];
    }
    for (int a = 1; a < n_picked; a++) {
        int   key = sorted_pos[a];
        int   b   = a - 1;
        while (b >= 0 && sorted_pos[b] > key) {
            sorted_pos[b + 1] = sorted_pos[b];
            b--;
        }
        sorted_pos[b + 1] = key;
    }

    // Emit (pos, val) and zero the masked input.
    for (int j = 0; j < n_picked; j++) {
        const int p = sorted_pos[j];
        out_pos[j]      = (uint8_t)(p & 0xFF);
        out_val_fp32[j] = x[p];
        x_masked[p]     = 0.0f;
    }
    // If we picked fewer than n_out (only possible when N==0), pad: leave
    // initial zeros from the default-init above.
}

void ggml_trellis_outliers_apply(
    const uint8_t     * pos,
    const ggml_fp16_t * val_fp16,
    int                 n_out,
    float             * y) {

    if (!pos || !val_fp16 || n_out <= 0) return;
    if (n_out > 255) n_out = 255;

    for (int j = 0; j < n_out; j++) {
        // Caller guarantees pos[j] < QK_GROUP.
        y[pos[j]] = ggml_fp16_to_fp32(val_fp16[j]);
    }
}
