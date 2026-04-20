// Trellis v2 shared code for VTQ{2,3,4}_2 KV-cache quant types.
//
// Bitshift trellis (QTIP arXiv:2406.11235, little-endian emit):
//   state_i = (state_{i-1} >> K) | (bits_i << (L-K))    (mod 2^L)
// Decoder: iterative shift register + TABLE (inverse-Gaussian CDF) LUT.
// Encoder: full Viterbi DP over 2^L states, group-level (joint path).
//
// Fixed Phase-1 lockings:
//   L = 16  (state bits)  — 64 KiB LUT
//   G = 1 block per ggml-block (ggml block == one Trellis group of 512 samples)
//
// Only K varies: 2, 3, 4 → VTQ2_2, VTQ3_2, VTQ4_2 ggml types.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TRELLIS_L        16
#define GGML_TRELLIS_QK_GROUP 256  // samples per ggml-block
// 256 divides typical V-cache row sizes (head_dim * n_head_kv / tp):
//   Qwen3.5-27B:  128*2 = 256     (exactly)
//   Qwen3.5-0.8B: 256*1 = 256     (exactly)
//   Gemma4-27B:   128*16 = 2048   (divisible)
//   Mistral-7B:   128*8 = 1024    (divisible)
// A value of 512 would fail for head_dim=256 models (common).

// Returns pointer to lazy-initialized 2^L inverse-Gaussian-CDF LUT.
// Thread-safe first init via atomic flag; subsequent calls are free.
const float * ggml_trellis_table(void);

// Encode `QK_GROUP` float samples into `start_state` (L bits) +
// `qs_bytes` packed emitted bits. Uses full Viterbi over 2^L states.
// Returns the reconstructed group L2 norm (encoder will scale so that
// decoded magnitude matches x's magnitude).
//
// Output:
//   *out_start_state : 16-bit open-start state chosen by Viterbi
//   *out_d           : scale such that decode * (1/sqrt(QK_GROUP)) * d
//                      matches x. Stored as fp32; caller converts to fp16.
//   qs[0 .. (QK_GROUP*K + 7)/8 - 1] : packed emitted bits (little-endian)
void ggml_trellis_encode_group(
    const float * x,       // QK_GROUP input samples
    int           K,       // code bits (2, 3, or 4)
    uint16_t    * out_start_state,
    float       * out_d,
    uint8_t     * qs);     // caller ensures capacity QK_GROUP*K/8 bytes

// Decode QK_GROUP float samples from stored start_state + qs + d.
// Reconstruction: y[i] = table[state_i] / sqrt(QK_GROUP) * d
// where d is the encoder-computed scale (NOT the L2 norm of x).
void ggml_trellis_decode_group(
    uint16_t        start_state,
    int             K,
    float           d,
    const uint8_t * qs,
    float         * y);    // QK_GROUP output samples

// --- Correction Overlay (Trick 4) — CPU helpers ---
// See docs/plans/2026-04-20-trick4-correction-overlay-design.md and
// ggml-common.h `vtq_overlay_entry`. Operates on raw packed bytes (4 B/entry)
// to avoid a C header dependency on ggml-common.h's struct definition.
//
// Extract top-N (N<=4) correction entries for a single decoded trellis block.
// Writes `n_per_block * 4` bytes of packed entries to `out_entries`. Entries
// contain (pos, flags, fp16_value). If `err_threshold > 0`, entries whose
// relative error |src-decoded|/max(|src|,1e-6) falls below threshold are
// marked invalid (flag bit0=0) so decode skips them — avoids spending a slot
// on already-accurate blocks. Returns the number of VALID entries emitted.
int ggml_trellis_overlay_extract(
    const float * src,        // QK_GROUP ground-truth fp32 samples
    const float * decoded,    // QK_GROUP decoded fp32 samples
    int           n_per_block,
    float         err_threshold,  // 0 = accept all
    uint8_t     * out_entries);   // n_per_block*4 bytes

// Apply overlay corrections in place. Invalid entries are skipped.
void ggml_trellis_overlay_apply(
    const uint8_t * entries,  // n_per_block*4 bytes
    int             n_per_block,
    float         * y);       // QK_GROUP decoded samples, patched in place

#ifdef __cplusplus
}
#endif
