// Round-trip sanity test for the VTQ_2 cached-decode path used by the
// E11 CUDA port (Phase 3A1). This test is intentionally CPU-only: the
// fattn-vec-vtq2.cuh kernel itself relies on CUDA launch + the device
// LUT, which is covered end-to-end by test-backend-ops FLASH_ATTN_EXT.
//
// What this test *does* verify:
//   - The `vtq_state_at<K>` bit-window formula used by both the legacy
//     `dequantize_V_vtq_2` and the new `vtq2_block_warm` cooperative
//     decoder produces the same 128-sample sequence as a naive
//     shift-register walk from `start_state`.
//   - The "direct O(1) per-sample read" is equivalent to the sequential
//     shift-register decoder on a known block layout.
//
// If this test fails, the cooperative warm kernel will produce garbage
// values in shmem and the FA-vec output will be off. Catching it here
// avoids chasing a correctness bug through a 45min CUDA rebuild.
//
// Spec: docs/plans/2026-04-21-e11-cuda-port-spec.md §7

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr int QK_VTQ_TRELLIS = 128;
static constexpr int VTQ_TRELLIS_L  = 16;

// --- Reference: naive sequential shift-register decode.
// After i updates, state = [s0 || qs-bits[0..i*K-1]] window of width L.
// This is the definition the encoder produces; every fast decoder must
// match it byte-for-byte.
static uint32_t ref_state_at(uint16_t s0, const uint8_t * qs, int K, int i) {
    // Build combined bit-stream [s0_lo .. s0_hi || qs[0].b0 .. qs[0].b7 || ...]
    // Read L bits starting at position i*K.
    const int stream_bit = i * K;
    // Emulate simple bit extraction (LSB-first within each byte, as qs is
    // stored). We treat s0 as the first 16 bits (little-endian) of the stream.
    // Lambda: read bit `pos` from the combined stream.
    auto read_bit = [&](int pos) -> uint32_t {
        if (pos < VTQ_TRELLIS_L) {
            return (uint32_t)((s0 >> pos) & 1u);
        }
        const int qs_bit = pos - VTQ_TRELLIS_L;
        return (uint32_t)((qs[qs_bit >> 3] >> (qs_bit & 7)) & 1u);
    };

    uint32_t w = 0;
    for (int b = 0; b < VTQ_TRELLIS_L; ++b) {
        w |= read_bit(stream_bit + b) << b;
    }
    return w;
}

// --- Fast (O(1)) variant: exactly the algorithm used in
// fattn-common.cuh::vtq_state_at<K>. Copied here so the test does not
// need to link against CUDA code.
template <int K>
static uint32_t fast_state_at(uint16_t s0, const uint8_t * qs, int i) {
    const int stream_bit = i * K;
    constexpr int L = VTQ_TRELLIS_L;

    if (stream_bit + L <= L) {
        return (uint32_t)s0 & 0xFFFFu;
    }

    if (stream_bit < L) {
        const int from_ss = L - stream_bit;
        uint32_t lo = ((uint32_t)s0 >> stream_bit) & ((1u << from_ss) - 1u);
        uint32_t qs_word = (uint32_t)qs[0] | ((uint32_t)qs[1] << 8) | ((uint32_t)qs[2] << 16);
        uint32_t hi = qs_word & ((1u << stream_bit) - 1u);
        return lo | (hi << from_ss);
    }

    const int qs_bit = stream_bit - L;
    const int byte   = qs_bit >> 3;
    const int shift  = qs_bit & 7;
    uint32_t b0 = qs[byte];
    uint32_t b1 = qs[byte + 1];
    uint32_t b2 = qs[byte + 2];
    uint32_t w  = b0 | (b1 << 8) | (b2 << 16);
    return (w >> shift) & 0xFFFFu;
}

static int test_state_equiv(int K_bits, int qs_bytes) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_int_distribution<int> u16(0, 65535);

    int failures = 0;
    for (int trial = 0; trial < 32; ++trial) {
        const uint16_t s0 = (uint16_t) u16(rng);
        std::vector<uint8_t> qs(qs_bytes + 4, 0);   // +4 slack for safe read ahead
        for (int b = 0; b < qs_bytes; ++b) qs[b] = (uint8_t) u8(rng);

        for (int i = 1; i <= QK_VTQ_TRELLIS; ++i) {
            const uint32_t ref = ref_state_at(s0, qs.data(), K_bits, i);
            uint32_t got = 0;
            switch (K_bits) {
                case 2: got = fast_state_at<2>(s0, qs.data(), i); break;
                case 3: got = fast_state_at<3>(s0, qs.data(), i); break;
                case 4: got = fast_state_at<4>(s0, qs.data(), i); break;
                default: std::fprintf(stderr, "unsupported K=%d\n", K_bits); return 1;
            }
            if (ref != got) {
                if (failures < 5) {
                    std::fprintf(stderr, "  K=%d trial=%d i=%d: ref=0x%04x got=0x%04x\n",
                                 K_bits, trial, i, ref, got);
                }
                ++failures;
            }
        }
    }
    return failures;
}

int main() {
    int total = 0;
    std::printf("VTQ_2 cached-decode round-trip test (vtq_state_at equivalence)\n");

    struct Case { int K; int qs_bytes; const char * name; };
    const Case cases[] = {
        { 2, QK_VTQ_TRELLIS * 2 / 8, "VTQ2_2 (32B qs)" },
        { 3, QK_VTQ_TRELLIS * 3 / 8, "VTQ3_2 (48B qs)" },
        { 4, QK_VTQ_TRELLIS * 4 / 8, "VTQ4_2 (64B qs)" },
    };
    for (const auto & c : cases) {
        const int f = test_state_equiv(c.K, c.qs_bytes);
        std::printf("  %-24s %s\n", c.name, f == 0 ? "PASS" : "FAIL");
        total += f;
    }

    if (total == 0) {
        std::printf("All %zu case(s) passed.\n", sizeof(cases)/sizeof(cases[0]));
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d mismatches\n", total);
    return 1;
}
