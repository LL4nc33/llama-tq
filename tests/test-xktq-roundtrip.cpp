// XQuant Phase 1 — CPU round-trip sanity test for paired KTQ2_1 + XKTQ2_1.
//
// Spec: docs/plans/2026-04-26-xquant-port-spec.md (Phase 1 deliverables, §10).
//
// Goal: verify that a "subordinate" XKTQ2_1 layer reusing a sibling KTQ2_1
// layer's quantized codes (qs) and RHT sign bits (sb), but applying its own
// per-block L2 scale, reconstructs an input vector with MSE within 1.5x of a
// plain per-layer KTQ2_1 round-trip. This is the math-only PoC; CUDA kernels,
// pairing logic, and FA dispatch land in later phases.
//
// Sharing rationale: the RHT (Hadamard with random Bernoulli signs) used by
// KTQ is seeded by block_index alone (kktq_derive_seed). Two layers at the
// same block_index thus share an identical rotation. A subordinate layer can
// therefore reuse the dominant layer's rotated-and-quantized codes and apply
// its own scale at dequant time. See ggml-common.h for the block layout.
//
// Test scenarios:
//   1. Identical input on both layers — XKTQ subordinate must match KTQ2_1
//      reconstruction exactly (same scale, same codes, same sb).
//   2. Sibling-driven input — subordinate's input is a perturbed copy of the
//      dominant input (correlated layer pair). Reuse codes from dominant +
//      own scale. Assert MSE <= 1.5 * plain_ktq_mse on sub's true vector.
//
// Both scenarios exercise the dequantize_row_xktq2_1_paired() entry point.

#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ggml-quants.h pulls in ggml-common.h (with GGML_COMMON_DECL_C) and exposes
// the block_* structs and GGML_API quantize/dequantize helpers under extern "C".
#include "../ggml/src/ggml-quants.h"

static double mse(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s / a.size();
}

int main() {
    // 32 elements per block (QK_KTQ = 32) — use 64 blocks for statistical signal.
    constexpr int N_BLOCKS = 64;
    constexpr int N        = QK_KTQ * N_BLOCKS;  // 2048 floats

    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // ---- Scenario 1: subordinate uses the SAME input as dominant.
    // Round-trip via XKTQ_paired must equal round-trip via KTQ2_1 directly
    // (within ~1e-6 fp16 noise — we share codes, sb, and scale to fp16).
    std::vector<float>       x_dom(N);
    for (int i = 0; i < N; i++) x_dom[i] = gauss(rng);

    std::vector<block_ktq2_1>  q_dom(N_BLOCKS);
    std::vector<block_xktq2_1> q_sub_same(N_BLOCKS);

    quantize_row_ktq2_1_ref       (x_dom.data(), q_dom.data(),       N);
    // Use the paired ref-quant variant (norm-corrected via dom's codes),
    // matching the KTQ2_1 norm-correction pattern. Yields ~fp16-floor MSE
    // when sub_input == dom_input.
    quantize_row_xktq2_1_ref_paired(x_dom.data(), q_dom.data(), q_sub_same.data(), N);

    std::vector<float> y_ktq (N), y_xktq_same(N);
    dequantize_row_ktq2_1          (q_dom.data(),                       y_ktq.data(),       N);
    dequantize_row_xktq2_1_paired  (q_sub_same.data(), q_dom.data(),    y_xktq_same.data(), N);

    double mse_same = mse(y_ktq, y_xktq_same);
    // With norm-corrected paired quant, MSE drops to fp16-floor (~1e-5 to 1e-4)
    // on Gauss(0,1) input. The previous unpaired path with raw L2-norm gave
    // ~7e-3, fixed in commit "fix(xquant): norm-correct paired quant".
    std::printf("[scenario 1] identical inputs — KTQ vs XKTQ-paired MSE: %.3e (expect <1e-3, fp16 floor)\n", mse_same);
    if (mse_same > 1e-3) {
        std::fprintf(stderr, "FAIL scenario 1: MSE %.3e exceeds fp16-floor 1e-3\n", mse_same);
        return 1;
    }

    // ---- Scenario 2: subordinate has a CORRELATED-but-different input.
    // Model adjacent transformer layers: x_sub = x_dom + small noise. The
    // subordinate stores its own L2 scale; codes + sb come from the dominant.
    // This is the realistic XLC use case.
    std::vector<float> x_sub(N);
    std::normal_distribution<float> noise(0.0f, 0.15f);  // ~15% per-element noise
    for (int i = 0; i < N; i++) x_sub[i] = x_dom[i] + noise(rng);

    // Baseline: plain KTQ2_1 round-trip on the subordinate's true input.
    std::vector<block_ktq2_1> q_sub_plain(N_BLOCKS);
    std::vector<float>        y_sub_plain(N);
    quantize_row_ktq2_1_ref(x_sub.data(), q_sub_plain.data(), N);
    dequantize_row_ktq2_1  (q_sub_plain.data(), y_sub_plain.data(), N);
    double mse_plain = mse(x_sub, y_sub_plain);

    // XQuant path: quantize sub as XKTQ paired (norm-corrected via dom's codes),
    // dequant via paired entry using dominant's codes + sb.
    std::vector<block_xktq2_1> q_sub_xq(N_BLOCKS);
    std::vector<float>         y_sub_xq(N);
    quantize_row_xktq2_1_ref_paired(x_sub.data(), q_dom.data(), q_sub_xq.data(), N);
    dequantize_row_xktq2_1_paired  (q_sub_xq.data(), q_dom.data(), y_sub_xq.data(), N);
    double mse_xq = mse(x_sub, y_sub_xq);

    std::printf("[scenario 2] correlated input (noise sigma=0.15)\n");
    std::printf("    plain KTQ2_1 MSE  : %.6f\n", mse_plain);
    std::printf("    XLC  XKTQ2_1 MSE  : %.6f\n", mse_xq);
    std::printf("    ratio  xq/plain   : %.3f (target < 1.5)\n", mse_xq / mse_plain);

    // Tolerance from spec §9 bench gate (round-trip MSE < 1.5x of plain KTQ2_1).
    if (mse_xq > 1.5 * mse_plain) {
        std::fprintf(stderr, "FAIL scenario 2: XQuant MSE %.6f exceeds 1.5x KTQ baseline %.6f\n",
                     mse_xq, mse_plain);
        return 1;
    }

    // ---- Scenario 3: block size + struct compactness sanity.
    if (sizeof(block_xktq2_1) != 8) {
        std::fprintf(stderr, "FAIL scenario 3: sizeof(block_xktq2_1)=%zu, expected 8\n",
                     sizeof(block_xktq2_1));
        return 1;
    }
    std::printf("[scenario 3] sizeof(block_xktq2_1) = %zu B (expected 8)\n", sizeof(block_xktq2_1));

    std::printf("OK — Phase 1 XQuant CPU round-trip passes\n");
    return 0;
}
