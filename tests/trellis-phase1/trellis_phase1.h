// Trellis v2 Phase-1 experimental harness.
// Standalone: no ggml deps. Produces MSE + timing data for config sweep.
// See docs/plans/2026-04-17-trellis-v2-design.md for context.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRELLIS_CODE_3GAUSS = 0,   // Weyl hash + 3-byte CLT (bounded ±3σ)
    TRELLIS_CODE_1MAD   = 1,   // Single modular multiply
    TRELLIS_CODE_TABLE  = 2,   // Precomputed 2^L Gaussians via inv-CDF
    TRELLIS_CODE_T5     = 3,   // Precomputed 2^L Student-t(5) via inv-CDF: heavy tails
} trellis_code_fn;

typedef struct {
    int  state_bits;           // L: 8, 12, 16
    int  code_bits;            // K: 2, 3
    int  block_size;           // QK: 32, 64, 128
    int  beam_width;           // 0 = full Viterbi, >0 = pruned
    int  norm_correction;      // 0 = off, 1 = on
    int  group_size;           // G ≥ 1. Blocks per shared-start-state group.
                               // First block in group stores start_state;
                               // subsequent blocks chain from previous end_state.
    trellis_code_fn code;
    const char * label;        // for CSV output
} trellis_config;

// Encoded block: fp16 scale `d` + start state + packed bit stream `qs`.
// qs length depends on QK and K: ceil(QK*K/8) + window pad.
// start_state costs L bits per block (stored separately here for clarity).
typedef struct {
    uint16_t d;                // fp16 scale (norm after quant)
    uint16_t start_state;      // L bits of the open start state (encoder's choice)
    uint8_t  qs[64];           // max: QK=128, K=3 → 48B + 2B pad = 50B. 64B reserve.
} trellis_block;

// Code function interface.
float trellis_code(trellis_code_fn fn, uint32_t state, int state_bits);

// Encoder: full Viterbi over 2^L states. Writes one block.
// start_state_in == 0xFFFFFFFFu means "open start" (encoder chooses freely
// and writes states[0] to out->start_state). Otherwise the encoder forces
// state_0 = start_state_in and out->start_state is left zero (caller knows).
// Returns end_state (states[N]), for chaining in groups.
uint32_t trellis_encode_block(const trellis_config * cfg,
                              const float * x,         // QK input samples
                              uint32_t start_state_in,  // 0xFFFFFFFFu = open
                              trellis_block * out);

// Decoder: rebuild from given start_state (or from out->start_state if
// start_state_in == 0xFFFFFFFFu).
void  trellis_decode_block(const trellis_config * cfg,
                           const trellis_block * in,
                           uint32_t start_state_in,
                           float * y);              // QK output samples

// RNG + test data.
void  trellis_seed_rng(uint64_t seed);
void  trellis_gen_gaussian(float * buf, size_t n);  // N(0,1)
void  trellis_gen_laplace(float * buf, size_t n);   // Laplace(0,1), heavier tails
void  trellis_gen_student_t(float * buf, size_t n, float nu);  // Student-t, very heavy
void  trellis_gen_bimodal(float * buf, size_t n);   // 0.5·N(-1.5,0.5) + 0.5·N(1.5,0.5)
void  trellis_gen_vcache_like(float * buf, size_t n);  // Gaussian + 5% outliers (3σ+)
int   trellis_load_binary(const char * path, float * buf, size_t n);

// Timing.
double trellis_now_ms(void);

#ifdef __cplusplus
}
#endif
