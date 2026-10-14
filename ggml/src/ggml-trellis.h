// Trellis v2 shared code for the trellis KV-cache quant types (vtq{2,3,4}_2, vtq{2,3,4}_3, vtq3_v8).
//
// Bitshift trellis (QTIP arXiv:2406.11235, little-endian emit):
//   state_i = (state_{i-1} >> K) | (bits_i << (L-K))    (mod 2^L)
// Decoder: iterative shift register + TABLE (inverse-Gaussian CDF) LUT.
// Encoder: full Viterbi DP over 2^L states, group-level (joint path).
//
// One ggml block == one trellis group of QK_GROUP samples. The state bits L are a parameter of the encoder and
// decoder; all types use L = 16 (65536 states).
// K (code bits per sample) varies with the type: 2, 3, 4.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include "ggml.h"  // ggml_fp16_t for outlier helpers

#ifdef __cplusplus
extern "C" {
#endif

// scale experiment: 0 = decoded block keeps the norm of x, 1 = unbiased (projection of the decoded block onto x equals x)
#ifndef GGML_TRELLIS_UNBIASED_SCALE
#define GGML_TRELLIS_UNBIASED_SCALE 0
#endif
// state bits (start_state holds up to 16 bits)
#define GGML_TRELLIS_L        16  // vtq*_2, vtq*_3, vtq3_v8
#define GGML_TRELLIS_QK_GROUP 128  // samples per ggml-block (task #143: was 256)
// 256 divides typical V-cache row sizes (head_dim * n_head_kv / tp):
//   Qwen3.5-27B:  128*2 = 256     (exactly)
//   Qwen3.5-0.8B: 256*1 = 256     (exactly)
//   Gemma4-27B:   128*16 = 2048   (divisible)
//   Mistral-7B:   128*8 = 1024    (divisible)
// A value of 512 would fail for head_dim=256 models (common).

// Returns pointer to lazy-initialized 2^GGML_TRELLIS_L inverse-Gaussian-CDF LUT.
// The code of a state depends only on the state, so the first 2^L entries are the table for L state bits.
// Thread-safe first init via atomic flag; subsequent calls are free.
GGML_API const float * ggml_trellis_table(void);

// Encode `QK_GROUP` float samples into `start_state` (L bits, L <= GGML_TRELLIS_L) +
// `qs_bytes` packed emitted bits. Uses full Viterbi over 2^L states.
// Returns the reconstructed group L2 norm (encoder will scale so that
// decoded magnitude matches x's magnitude).
//
// Output:
//   *out_start_state : L-bit open-start state chosen by Viterbi
//   *out_d           : scale such that decode * (1/sqrt(QK_GROUP)) * d
//                      matches x. Stored as fp32; caller converts to fp16.
//   qs[0 .. (QK_GROUP*K + 7)/8 - 1] : packed emitted bits (little-endian)
GGML_API void ggml_trellis_encode_group(
    const float * x,       // QK_GROUP input samples
    int           K,       // code bits (2, 3, or 4)
    int           L,       // state bits (at most GGML_TRELLIS_L)
    uint16_t    * out_start_state,
    float       * out_d,
    uint8_t     * qs);     // caller ensures capacity QK_GROUP*K/8 bytes

// Decode QK_GROUP float samples from stored start_state + qs + d.
// Reconstruction: y[i] = table[state_i] / sqrt(QK_GROUP) * d
// where d is the encoder-computed scale (NOT the L2 norm of x).
GGML_API void ggml_trellis_decode_group(
    uint16_t        start_state,
    int             K,
    int             L,
    float           d,
    const uint8_t * qs,
    float         * y);    // QK_GROUP output samples

// --- Outlier-Channel-Split (VTQ_3) — CPU helpers ---
// Pick the n_out positions with the largest |x[i]| in a QK_GROUP-sample block.
// Outputs:
//   out_pos[0..n_out-1]      : sorted ascending positions (uint8)
//   out_val_fp32[0..n_out-1] : the picked fp32 values, ordered to match out_pos
//   x_masked[0..QK_GROUP-1]  : copy of x with out_pos slots zeroed (Trellis-encode input)
// If n_out >= QK_GROUP, all positions are picked. n_out is clamped to [0, 255].
GGML_API void ggml_trellis_outliers_pick(
    const float * x,
    int           n_out,
    uint8_t     * out_pos,
    float       * out_val_fp32,
    float       * x_masked);

// Patch decoded y[] in place by overwriting y[pos[k]] with fp16->fp32(val_fp16[k]).
// Caller guarantees pos[k] < QK_GROUP.
GGML_API void ggml_trellis_outliers_apply(
    const uint8_t     * pos,
    const ggml_fp16_t * val_fp16,
    int                 n_out,
    float             * y);

#ifdef __cplusplus
}
#endif
