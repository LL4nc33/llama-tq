// Hypothesis A test: do K=2/3/4 encoders produce K-bit-depth-specific qs bytes,
// or are they identical / merely zero-padded?
//
// Bug context: VTQ2_2/VTQ3_2/VTQ4_2 produce bit-identical PPL across K values.
// Encoder dispatch (CUDA path) was confirmed to call distinct templates, but
// it could still emit identical bytes if the trellis encoder uses only the
// low-K bits, with the upper bits zeroed out for K>2.
//
// This test calls ggml_trellis_encode_group() (the CPU reference encoder)
// with the SAME 128-sample input for K=2, K=3, K=4 and:
//   1. dumps the qs[] hex bytes
//   2. compares K=2 vs K=3 vs K=4 byte content
//   3. extracts the low-2 / low-3 bits from K=4 to see if they match K=2/K=3
//   4. verifies start_state and d differ across K
//
// If qs[] bytes for K=4 == qs[] bytes for K=2 (zero-padded) → encoder bug.
// If completely different → hypothesis A is REFUTED, look elsewhere.

#include "ggml.h"
#include "../ggml/src/ggml-trellis.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

static void hex_dump(const char * label, const uint8_t * data, int n) {
    printf("%s (%d bytes):\n  ", label, n);
    for (int i = 0; i < n; ++i) {
        printf("%02x ", data[i]);
        if ((i + 1) % 16 == 0) printf("\n  ");
    }
    printf("\n");
}

static int popcount32(uint32_t x) {
    int c = 0; while (x) { c += x & 1; x >>= 1; } return c;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;

    const int N = GGML_TRELLIS_QK_GROUP;  // 128
    if (N != 128) {
        printf("ERROR: expected QK_GROUP=128, got %d\n", N);
        return 1;
    }

    // Deterministic Gaussian input.
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = dist(rng);

    // Force LUT init (lazy).
    (void) ggml_trellis_table();

    auto run_encode = [&](int K) {
        const int qs_bytes = (N * K + 7) / 8;
        std::vector<uint8_t> qs(qs_bytes, 0);
        uint16_t start_state = 0;
        float    d           = 0.0f;
        ggml_trellis_encode_group(x.data(), K, &start_state, &d, qs.data());
        printf("\n=== K=%d ===  start_state=0x%04x  d=%.6f  qs_bytes=%d\n",
               K, start_state, d, qs_bytes);
        hex_dump("  qs", qs.data(), qs_bytes);
        return std::make_tuple(start_state, d, qs);
    };

    auto [s2, d2, qs2] = run_encode(2);
    auto [s3, d3, qs3] = run_encode(3);
    auto [s4, d4, qs4] = run_encode(4);

    // ---------- Comparisons ----------
    printf("\n--- Cross-K diagnostics ---\n");
    printf("start_state  K=2:0x%04x  K=3:0x%04x  K=4:0x%04x  %s\n",
           s2, s3, s4,
           (s2 == s3 && s3 == s4) ? "[ALL EQUAL — SUSPICIOUS]" : "[differ ✓]");
    printf("d            K=2:%.6f  K=3:%.6f  K=4:%.6f  %s\n",
           d2, d3, d4,
           (d2 == d3 && d3 == d4) ? "[ALL EQUAL — SUSPICIOUS]" : "[differ ✓]");

    // Compare K=4 to K=2 padded with zeros: if equal → bug.
    // K=2 has 32 bytes, K=4 has 64 bytes. If K=4[0..31] == K=2[0..31] AND
    // K=4[32..63] == 0 → encoder is only emitting low-K bits.
    bool k4_first_half_eq_k2 = std::memcmp(qs4.data(), qs2.data(), qs2.size()) == 0;
    bool k4_second_half_zero = true;
    for (size_t i = qs2.size(); i < qs4.size(); ++i) {
        if (qs4[i] != 0) { k4_second_half_zero = false; break; }
    }
    printf("K=4[0..%zu] == K=2: %s\n", qs2.size() - 1,
           k4_first_half_eq_k2 ? "YES [BUG]" : "no ✓");
    printf("K=4[%zu..%zu] all zero: %s\n", qs2.size(), qs4.size() - 1,
           k4_second_half_zero ? "YES [BUG]" : "no ✓");

    // Extract low-K' bits from K=4 stream and compare to K=K' stream.
    // The trellis emits little-endian bit packed: bit i of sample s lives at
    //   byte = (s*K + i) / 8, bitpos = (s*K + i) % 8.
    auto extract_low_bits = [&](const std::vector<uint8_t> & qs, int K, int low) {
        // For each of 128 samples, take low bits of its K-bit code, repack.
        std::vector<uint8_t> out((N * low + 7) / 8, 0);
        for (int s = 0; s < N; ++s) {
            uint32_t code = 0;
            for (int b = 0; b < K; ++b) {
                int byte = (s*K + b) / 8;
                int bit  = (s*K + b) % 8;
                if (qs[byte] & (1u << bit)) code |= (1u << b);
            }
            uint32_t lowmask = (1u << low) - 1;
            uint32_t lowcode = code & lowmask;
            for (int b = 0; b < low; ++b) {
                if (lowcode & (1u << b)) {
                    int byte = (s*low + b) / 8;
                    int bit  = (s*low + b) % 8;
                    out[byte] |= (uint8_t)(1u << bit);
                }
            }
        }
        return out;
    };

    auto k4_low2 = extract_low_bits(qs4, 4, 2);
    auto k4_low3 = extract_low_bits(qs4, 4, 3);
    auto k3_low2 = extract_low_bits(qs3, 3, 2);

    bool k4low2_eq_k2 = std::memcmp(k4_low2.data(), qs2.data(), qs2.size()) == 0;
    bool k4low3_eq_k3 = std::memcmp(k4_low3.data(), qs3.data(), qs3.size()) == 0;
    bool k3low2_eq_k2 = std::memcmp(k3_low2.data(), qs2.data(), qs2.size()) == 0;

    printf("low-2 bits of K=4 == K=2 bytes: %s\n",
           k4low2_eq_k2 ? "YES [encoder ignores high bits!]" : "no ✓");
    printf("low-3 bits of K=4 == K=3 bytes: %s\n",
           k4low3_eq_k3 ? "YES [encoder ignores high bits!]" : "no ✓");
    printf("low-2 bits of K=3 == K=2 bytes: %s\n",
           k3low2_eq_k2 ? "YES [encoder ignores high bits!]" : "no ✓");

    // Bit-population per stream — should grow with K (more bits packed).
    int pop2 = 0, pop3 = 0, pop4 = 0;
    for (auto b : qs2) pop2 += popcount32(b);
    for (auto b : qs3) pop3 += popcount32(b);
    for (auto b : qs4) pop4 += popcount32(b);
    printf("popcount  K=2:%d/%zu  K=3:%d/%zu  K=4:%d/%zu  (≈50%% expected for random codes)\n",
           pop2, qs2.size()*8, pop3, qs3.size()*8, pop4, qs4.size()*8);

    // ---------- Decode roundtrip MSE: should improve with K ----------
    auto decode_and_mse = [&](int K, uint16_t s, float d, const std::vector<uint8_t>& qs) {
        std::vector<float> y(N);
        ggml_trellis_decode_group(s, K, d, qs.data(), y.data());
        double mse = 0.0;
        for (int i = 0; i < N; ++i) {
            double e = (double)x[i] - (double)y[i];
            mse += e*e;
        }
        return mse / N;
    };
    double mse2 = decode_and_mse(2, s2, d2, qs2);
    double mse3 = decode_and_mse(3, s3, d3, qs3);
    double mse4 = decode_and_mse(4, s4, d4, qs4);
    printf("decode MSE  K=2:%.6f  K=3:%.6f  K=4:%.6f  %s\n",
           mse2, mse3, mse4,
           (mse4 < mse3 && mse3 < mse2) ? "[monotone ✓]" :
           (mse2 == mse3 && mse3 == mse4) ? "[ALL EQUAL — BUG]" : "[non-monotone]");

    // ---------- Verdict ----------
    printf("\n=== VERDICT ===\n");
    bool bug = k4_first_half_eq_k2 || k4low2_eq_k2 || k4low3_eq_k3 ||
               (mse2 == mse3 && mse3 == mse4);
    if (bug) {
        printf("HYPOTHESIS A CONFIRMED: encoder writes K-independent bytes.\n");
        return 2;
    }
    printf("HYPOTHESIS A REFUTED: encoder produces distinct K-bit-depth-specific bytes.\n");
    printf("  → Bug is downstream (decoder dispatch / FA path / cache layout).\n");
    return 0;
}
