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
#define GGML_TRELLIS_QK_GROUP 512  // samples per ggml-block

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

#ifdef __cplusplus
}
#endif
