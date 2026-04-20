// Full Viterbi encoder + bitshift decoder for the parametric sweep.
//
// Bitshift trellis (QTIP-style, little-endian emit):
//   state_i = (state_{i-1} >> K) | (bits_i << (L-K))    (mod 2^L)
//   emit:  bits_i  (K bits) at bit offset K·i in qs[]   (LSB first in byte)
//   code:  g(state_i) · scale
//
// This pairs directly with the decoder, which reads the L-bit window at
// qs offsets [K·i, K·i + L): the newly emitted K bits land at the high
// position in the window and oldest bits drop off the bottom.
//
// Encoder: standard Viterbi DP, O(N · 2^L) per block.
// Decoder: parallel read; sample i reads L-bit window at K·i, zero-pad.

#include "trellis_phase1.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// --- fp16 helpers (IEEE 754 half, round-to-nearest-even) ---
static uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // inf/nan-like
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t rounded = mant >> shift;
        // round half to even
        uint32_t halfbit = 1u << (shift - 1);
        if ((mant & (halfbit - 1)) || ((mant & halfbit) && (rounded & 1))) rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t m10 = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1))) {
        m10++;
        if (m10 == 0x400u) { m10 = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m10);
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else {
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu; exp++;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { u };
    return v.f;
}

// --- bit-packing: write K bits at offset K·i (little-endian) ---
static void write_bits(uint8_t * qs, int bit_offset, uint32_t value, int K) {
    value &= (K >= 32) ? 0xFFFFFFFFu : ((1u << K) - 1u);
    int byte = bit_offset >> 3;
    int shift = bit_offset & 7;
    // up to 5 bytes touched for K=3 worst case (shift=7, 3 bits), safe with 8
    qs[byte]     |= (uint8_t)((value << shift) & 0xFFu);
    if (shift + K > 8) {
        qs[byte + 1] |= (uint8_t)((value >> (8 - shift)) & 0xFFu);
        if (shift + K > 16) {
            qs[byte + 2] |= (uint8_t)((value >> (16 - shift)) & 0xFFu);
        }
    }
}

// Read up to L bits (L ≤ 16) at bit_offset. Zero-pads reads past qs_bytes.
static uint32_t read_bits(const uint8_t * qs, int qs_bytes, int bit_offset, int L) {
    uint32_t out = 0;
    int byte = bit_offset >> 3;
    int shift = bit_offset & 7;
    // read up to 3 bytes (covers L ≤ 16 with any shift)
    uint32_t b0 = (byte     < qs_bytes) ? qs[byte]     : 0u;
    uint32_t b1 = (byte + 1 < qs_bytes) ? qs[byte + 1] : 0u;
    uint32_t b2 = (byte + 2 < qs_bytes) ? qs[byte + 2] : 0u;
    uint32_t w = b0 | (b1 << 8) | (b2 << 16);
    out = (w >> shift) & ((L >= 32) ? 0xFFFFFFFFu : ((1u << L) - 1u));
    return out;
}

// Number of bytes needed to store QK · K bits (no pad).
static int qs_bytes_exact(int QK, int K) { return (QK * K + 7) / 8; }

// Quickselect: return the k-th smallest (0-indexed) element in arr[0..n-1].
// Destructive (partially partitions arr). O(n) expected.
static float nth_element_f(float * arr, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        // median-of-three pivot
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < arr[lo]) { float t = arr[mid]; arr[mid] = arr[lo]; arr[lo] = t; }
        if (arr[hi]  < arr[lo]) { float t = arr[hi];  arr[hi]  = arr[lo]; arr[lo] = t; }
        if (arr[mid] < arr[hi]) { float t = arr[mid]; arr[mid] = arr[hi]; arr[hi] = t; }
        float pivot = arr[hi];
        int i = lo - 1;
        for (int j = lo; j < hi; j++) {
            if (arr[j] <= pivot) {
                i++;
                float t = arr[i]; arr[i] = arr[j]; arr[j] = t;
            }
        }
        i++;
        float t = arr[i]; arr[i] = arr[hi]; arr[hi] = t;
        if (i == k) return arr[i];
        if (i < k)  lo = i + 1;
        else        hi = i - 1;
    }
    return arr[lo];
}

// --- Full Viterbi encoder ---
// dp[i][s] = min cost to reach state s after emitting i samples.
// prev[i][s] = predecessor state (top L-K bits of s, plus the K emit bits
//   are implicit in s itself).
uint32_t trellis_encode_block(const trellis_config * cfg, const float * x,
                              uint32_t start_state_in, float norm_override,
                              trellis_block * out) {
    const int L = cfg->state_bits;
    const int K = cfg->code_bits;
    const int N = cfg->block_size;
    const uint32_t S = 1u << L;
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = S - 1u;

    memset(out, 0, sizeof(*out));

    // block L2 norm for scaling; override allows shared-d across groups
    float norm;
    if (norm_override > 0.0f) {
        norm = norm_override;
    } else {
        double n2 = 0.0;
        for (int j = 0; j < N; j++) n2 += (double)x[j] * x[j];
        norm = (float)sqrt(n2);
    }
    if (norm < 1e-30f) { out->d = 0; return 0; }
    float inv = 1.0f / norm;

    // normalize to unit-norm samples (per-block; codebook is N(0,1))
    float * xn = (float *)malloc((size_t)N * sizeof(float));
    for (int j = 0; j < N; j++) xn[j] = x[j] * inv;

    // scale the reference codebook so reconstruction magnitude matches:
    // decoded val = g(state) * cb_scale; we want sum(val²) ≈ 1 after decode
    // since g(s) ~ N(0,1), sum over N samples ~ N, so cb_scale = 1/sqrt(N)
    const float cb_scale = 1.0f / sqrtf((float)N);

    // Rolling DP: only need dp_cur and dp_next (2·S), not full N·S history.
    // Backtrack still needs [N][S] to recover the path.
    float * dp_cur  = (float *)malloc((size_t)S * sizeof(float));
    float * dp_next = (float *)malloc((size_t)S * sizeof(float));
    uint32_t * bt = (uint32_t *)malloc((size_t)N * S * sizeof(uint32_t));
    if (!dp_cur || !dp_next || !bt) {
        free(dp_cur); free(dp_next); free(bt); free(xn); return 0;
    }

    const int open_start = (start_state_in == 0xFFFFFFFFu);
    if (open_start) {
        for (uint32_t s = 0; s < S; s++) dp_cur[s] = 0.0f;
    } else {
        uint32_t ss = start_state_in & Lmask;
        for (uint32_t s = 0; s < S; s++) dp_cur[s] = FLT_MAX;
        dp_cur[ss] = 0.0f;
    }

    // Beam pruning: if beam_width > 0 and < S, only keep top-B states per step.
    const int beam = cfg->beam_width;
    const int use_beam = (beam > 0 && (uint32_t)beam < S);
    float * beam_costs = use_beam ? (float *)malloc((size_t)S * sizeof(float)) : NULL;

    // Precompute per-step: cached codes array would save trellis_code() calls,
    // but trellis_code for TABLE is a single LUT load — negligible.
    for (int i = 0; i < N; i++) {
        uint32_t * bt_i = bt + (size_t)i * S;
        for (uint32_t s = 0; s < S; s++) dp_next[s] = FLT_MAX;

        const int kshift = L - K;
        const float xi = xn[i];
        for (uint32_t prev = 0; prev < S; prev++) {
            float pc = dp_cur[prev];
            if (pc == FLT_MAX) continue;
            for (uint32_t k = 0; k <= Kmask; k++) {
                uint32_t next = ((prev >> K) | (k << kshift)) & Lmask;
                float code = trellis_code(cfg->code, next, L) * cb_scale;
                float d = xi - code;
                float cost = pc + d * d;
                if (cost < dp_next[next]) {
                    dp_next[next] = cost;
                    bt_i[next] = (uint32_t)prev;
                }
            }
        }

        if (use_beam) {
            memcpy(beam_costs, dp_next, (size_t)S * sizeof(float));
            float thresh = nth_element_f(beam_costs, (int)S, beam - 1);
            for (uint32_t s = 0; s < S; s++) {
                if (dp_next[s] > thresh) dp_next[s] = FLT_MAX;
            }
        }

        // swap dp_cur / dp_next for next iteration
        float * tmp = dp_cur; dp_cur = dp_next; dp_next = tmp;
    }
    free(beam_costs);
    // dp_cur now holds dp[N]. Alias for clarity below.
    float * dp_end = dp_cur;

    // find best end state
    uint32_t best_s = 0;
    float best_c = FLT_MAX;
    for (uint32_t s = 0; s < S; s++) {
        if (dp_end[s] < best_c) { best_c = dp_end[s]; best_s = s; }
    }

    // backtrack: recover emitted bits per step
    // emitted bits at step i = low K bits of state_{i+1}
    uint32_t * states = (uint32_t *)malloc((size_t)(N + 1) * sizeof(uint32_t));
    states[N] = best_s;
    for (int i = N - 1; i >= 0; i--) {
        states[i] = bt[(size_t)i * S + states[i + 1]];
    }
    out->start_state = states[0];

    int qs_len = qs_bytes_exact(N, K);
    if (qs_len > (int)sizeof(out->qs)) { free(states); free(dp_cur); free(dp_next); free(bt); free(xn); return 0; }
    // bits_i is the high-K chunk of state_{i+1} (matches encoder update)
    const int kshift_emit = L - K;
    for (int i = 0; i < N; i++) {
        uint32_t bits = (states[i + 1] >> kshift_emit) & Kmask;
        write_bits(out->qs, i * K, bits, K);
    }

    // norm correction: compute reconstruction L2 with the chosen path,
    // pick d such that decoded block has the original norm.
    double recon_sq = 0.0;
    for (int i = 0; i < N; i++) {
        float code = trellis_code(cfg->code, states[i + 1], L) * cb_scale;
        recon_sq += (double)code * code;
    }
    float recon_norm = (float)sqrt(recon_sq);
    float d_scale = (cfg->norm_correction && recon_norm > 1e-30f)
                    ? (norm / recon_norm) : norm;
    // When norm_override is active, caller owns the scale — don't write d.
    out->d = (norm_override > 0.0f) ? 0 : fp32_to_fp16(d_scale);

    uint32_t end_state = states[N];
    free(states); free(dp_cur); free(dp_next); free(bt); free(xn);
    return end_state;
}

// --- Group-level Viterbi: one path over G·QK samples ---
// Runs a single encode with effective block_size = G·QK, then splits the
// output qs bits and per-sample metadata across G output blocks.
int trellis_encode_group(const trellis_config * cfg, const float * x,
                         trellis_block * blks) {
    const int G = cfg->group_size > 0 ? cfg->group_size : 1;
    const int QK = cfg->block_size;
    const int K  = cfg->code_bits;
    const int total_samples = G * QK;

    if (G == 1) {
        // Degenerate: just forward to block encoder.
        (void)trellis_encode_block(cfg, x, 0xFFFFFFFFu, -1.0f, &blks[0]);
        return 0;
    }

    // Temporary full-size block for the combined Viterbi. We need a qs
    // buffer big enough for G·QK samples · K bits / 8.
    int total_qs_bytes = (total_samples * K + 7) / 8;
    if (total_qs_bytes > (int)sizeof(((trellis_block *)0)->qs) * G) {
        return -1;  // would overflow G-block combined qs space
    }

    // Compute group L2 norm (shared_d path) or total norm (otherwise).
    float norm;
    {
        double n2 = 0.0;
        for (int j = 0; j < total_samples; j++) n2 += (double)x[j] * x[j];
        norm = (float)sqrt(n2);
    }
    if (norm < 1e-30f) {
        memset(blks, 0, sizeof(trellis_block) * G);
        return 0;
    }
    float inv = 1.0f / norm;

    // Normalize samples
    float * xn = (float *)malloc((size_t)total_samples * sizeof(float));
    if (!xn) return -3;
    for (int j = 0; j < total_samples; j++) xn[j] = x[j] * inv;

    const int L = cfg->state_bits;
    const uint32_t S = 1u << L;
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = S - 1u;
    const float cb_scale = 1.0f / sqrtf((float)total_samples);

    // Rolling DP + full backtrack over total_samples × S
    float * dp_cur  = (float *)malloc((size_t)S * sizeof(float));
    float * dp_next = (float *)malloc((size_t)S * sizeof(float));
    uint32_t * bt = (uint32_t *)malloc((size_t)total_samples * S * sizeof(uint32_t));
    if (!dp_cur || !dp_next || !bt) {
        free(dp_cur); free(dp_next); free(bt); free(xn);
        return -4;
    }

    for (uint32_t s = 0; s < S; s++) dp_cur[s] = 0.0f;  // open start
    const int kshift = L - K;

    for (int i = 0; i < total_samples; i++) {
        uint32_t * bt_i = bt + (size_t)i * S;
        for (uint32_t s = 0; s < S; s++) dp_next[s] = FLT_MAX;
        const float xi = xn[i];
        for (uint32_t prev = 0; prev < S; prev++) {
            float pc = dp_cur[prev];
            if (pc == FLT_MAX) continue;
            for (uint32_t kk = 0; kk <= Kmask; kk++) {
                uint32_t next = ((prev >> K) | (kk << kshift)) & Lmask;
                float code = trellis_code(cfg->code, next, L) * cb_scale;
                float d = xi - code;
                float cost = pc + d * d;
                if (cost < dp_next[next]) {
                    dp_next[next] = cost;
                    bt_i[next] = (uint32_t)prev;
                }
            }
        }
        float * tmp = dp_cur; dp_cur = dp_next; dp_next = tmp;
    }
    float * dp_end = dp_cur;

    uint32_t best_s = 0;
    float best_c = FLT_MAX;
    for (uint32_t s = 0; s < S; s++) {
        if (dp_end[s] < best_c) { best_c = dp_end[s]; best_s = s; }
    }

    uint32_t * states = (uint32_t *)malloc((size_t)(total_samples + 1) * sizeof(uint32_t));
    states[total_samples] = best_s;
    for (int i = total_samples - 1; i >= 0; i--) {
        states[i] = bt[(size_t)i * S + states[i + 1]];
    }

    const int kshift_emit = L - K;
    for (int i = 0; i < total_samples; i++) {
        uint32_t bits = (states[i + 1] >> kshift_emit) & Kmask;
        // Write into the BLOCK corresponding to this sample.
        int bi = i / QK;
        int off = (i % QK) * K;
        write_bits(blks[bi].qs, off, bits, K);
    }

    // Norm correction computed over whole group.
    // Encoder uses cb_scale = 1/sqrt(G·QK). Decoder uses cb_scale = 1/sqrt(QK).
    // To match reconstructed magnitude on decoder side, the stored d must
    // compensate: d_stored · (1/sqrt(QK)) = d_true · (1/sqrt(G·QK))
    // → d_stored = d_true / sqrt(G).
    double recon_sq = 0.0;
    for (int i = 0; i < total_samples; i++) {
        float code = trellis_code(cfg->code, states[i + 1], L) * cb_scale;
        recon_sq += (double)code * code;
    }
    float recon_norm = (float)sqrt(recon_sq);
    float d_true = (cfg->norm_correction && recon_norm > 1e-30f)
                   ? (norm / recon_norm) : norm;
    float d_scale = d_true / sqrtf((float)G);

    // Metadata: start_state goes into blks[0], d depends on shared_d
    memset(&blks[0], 0, sizeof(trellis_block));  // clear; qs was already written via pointer
    // qs was written via pointer so memset on blks[0] clobbered it — redo bits:
    for (int bi = 0; bi < G; bi++) {
        memset(blks[bi].qs, 0, sizeof(blks[bi].qs));
    }
    for (int i = 0; i < total_samples; i++) {
        uint32_t bits = (states[i + 1] >> kshift_emit) & Kmask;
        int bi = i / QK;
        int off = (i % QK) * K;
        write_bits(blks[bi].qs, off, bits, K);
    }
    blks[0].start_state = states[0];
    if (cfg->shared_d) {
        blks[0].d = fp32_to_fp16(d_scale);
        for (int bi = 1; bi < G; bi++) blks[bi].d = 0;
    } else {
        for (int bi = 0; bi < G; bi++) blks[bi].d = fp32_to_fp16(d_scale);
    }

    free(states); free(dp_cur); free(dp_next); free(bt); free(xn);
    return 0;
}

// --- Group-level decoder ---
void trellis_decode_group(const trellis_config * cfg, const trellis_block * blks,
                          float * y) {
    const int L = cfg->state_bits;
    const int K = cfg->code_bits;
    const int QK = cfg->block_size;
    const int G = cfg->group_size > 0 ? cfg->group_size : 1;
    const uint32_t Kmask = (1u << K) - 1u;
    const uint32_t Lmask = (L >= 32) ? 0xFFFFFFFFu : ((1u << L) - 1u);
    // Encoder used cb_scale = 1/sqrt(total); d absorbs the sqrt(G) factor
    // so per-block d must be applied with 1/sqrt(QK) to match original magnitude.
    const float cb_scale = 1.0f / sqrtf((float)QK);

    uint32_t state = blks[0].start_state & Lmask;
    float d_group = cfg->shared_d ? fp16_to_fp32(blks[0].d) : 0.0f;
    int qs_len_block = qs_bytes_exact(QK, K);

    for (int bi = 0; bi < G; bi++) {
        float d_use = cfg->shared_d ? d_group : fp16_to_fp32(blks[bi].d);
        if (d_use == 0.0f) {
            memset(y + bi * QK, 0, (size_t)QK * sizeof(float));
            continue;
        }
        for (int i = 0; i < QK; i++) {
            uint32_t bits = read_bits(blks[bi].qs, qs_len_block, i * K, K) & Kmask;
            state = ((state >> K) | (bits << (L - K))) & Lmask;
            float code = trellis_code(cfg->code, state, L) * cb_scale;
            y[bi * QK + i] = code * d_use;
        }
    }
}

// --- Bitshift decoder (parallel read) ---
void trellis_decode_block(const trellis_config * cfg, const trellis_block * in,
                          uint32_t start_state_in, float d_override, float * y) {
    const int L = cfg->state_bits;
    const int K = cfg->code_bits;
    const int N = cfg->block_size;
    const int qs_len = qs_bytes_exact(N, K);
    // In group-Viterbi mode, encoder used cb_scale = 1/sqrt(G·QK) and
    // absorbed the sqrt(G) factor into d. Decoder here still uses
    // 1/sqrt(N) per-block; the d field carries the compensation.
    const float cb_scale = 1.0f / sqrtf((float)N);
    const float d = (d_override > 0.0f) ? d_override : fp16_to_fp32(in->d);
    if (d == 0.0f) { memset(y, 0, (size_t)N * sizeof(float)); return; }

    // Sample i reads state from the L-bit window ending at bit (K·(i+1)).
    // That window contains: high bits = prior emitted bits (implicit state),
    // low K bits = bits emitted to produce sample i.
    // Bitshift property: start of window is at offset K·(i+1) - L.
    // Reconstruct each state by running the shift register from start_state.
    // state_{i+1} = (state_i >> K) | (bits_i << (L-K))
    const uint32_t Lmask = (L >= 32) ? 0xFFFFFFFFu : ((1u << L) - 1u);
    const uint32_t Kmask = (1u << K) - 1u;
    uint32_t state = ((start_state_in == 0xFFFFFFFFu) ? in->start_state : start_state_in) & Lmask;
    for (int i = 0; i < N; i++) {
        uint32_t bits = read_bits(in->qs, qs_len, i * K, K) & Kmask;
        state = ((state >> K) | (bits << (L - K))) & Lmask;
        float code = trellis_code(cfg->code, state, L) * cb_scale;
        y[i] = code * d;
    }
}
