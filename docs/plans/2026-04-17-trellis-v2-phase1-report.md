# Trellis v2 Phase-1 Report

**Author:** LL4nc33
**Date:** 2026-04-17
**Branch:** `trellis-v2-phase1`
**Status:** Phase 1 complete, Phase 2 candidates locked

---

## Executive Summary

Phase 1 built a standalone parametric harness to evaluate trellis-coded
quantization (TCQ) for VTQ replacement. All Phase-1 exit gates cleared
on real post-RHT V-weight data from Qwen3.5-27B. Two GPU-port
candidates identified.

**Key result:** a 2-bit trellis config reaches **2.031 bpw** at MSE
ratio 0.59 vs Lloyd-Max baseline 1.0 on real V-weight data. A 3-bit
config reaches **3.063 bpw** at MSE ratio 0.17 — below buun's bpw
target (3.25) while likely competitive on PPL.

---

## What Was Built

A self-contained test binary at `tests/trellis-phase1/` with:

- Parametric Viterbi encoder (L ∈ {8, 12, 16, 20}, K ∈ {2, 3},
  QK ∈ {32, 64, 128, 256})
- Parallel bitshift decoder
- Three code functions:
  - **3GAUSS** — Weyl hash + 3-byte CLT sum (GPU-cheapest, bounded ±3σ)
  - **TABLE** — precomputed inverse-Gaussian CDF (quality winner)
  - **T5** — precomputed inverse-Student-t(5) CDF (heavy-tail safety)
- Data generators: Gaussian, Laplace, Student-t, bimodal, vcache-like,
  real-weight via GGUF extractor
- Beam-pruned Viterbi (quickselect threshold)
- Rolling 2-row DP buffer (drops memory from `(N+1)·S` to `2·S` floats)
- Group-chained start_state with configurable G
- Shared-d across group (one fp16 scale per G blocks)

No ggml dependency. Builds with plain gcc or cmake.

---

## Experimental Runs

Each CSV is committed alongside a markdown report with interpretation.

| Run | CSV                                   | Notes                                           |
|-----|---------------------------------------|-------------------------------------------------|
| 1a  | run1_gauss_4k_zero_start.csv          | zero-init start state; 2-bit fails gate         |
| 1b  | run1b_gauss_4k_open_start.csv         | open start → all 2-bit pass gate                |
| 2   | run2_gauss_4k_table_vs_3g.csv         | TABLE beats 3GAUSS by 3-10% MSE                 |
| 3a  | run3_beam_sweep.csv                   | beam pruning infrastructure (qsort, slow)       |
| 3b  | run3b_beam_quickselect.csv            | quickselect version (faster but marginal)       |
| 4   | run4_group_sweep.csv                  | group_size sweep — bpw wins                     |
| 5   | run5_dist_*.csv                       | heavy-tail stress (student5, vcachelike)        |
| 6a  | run6_t5_vcachelike.csv                | T5 code on heavy-tail                           |
| 6b  | run6b_t5_gauss.csv                    | T5 vs TABLE tradeoff on Gaussian                |
| 7a  | run7_real_vweights.csv                | Qwen3.5-0.8B V-slice + RHT                      |
| 7b  | run7b_real_27b.csv                    | Qwen3.5-27B V-slice + RHT                       |
| 8   | run8_bpw_floor.csv                    | QK=256, shared_d, G=4/8 — bpw floor             |

---

## Bug Trail

Systematic debugging found and fixed several nontrivial issues during
encoder/decoder bring-up:

1. **3GAUSS normalization constant was wrong** (73.9 → 128). Theory for
   sum of 3 uniform bytes is `sd = sqrt(3·(2^16-1)/12)` ≈ 128.0.
   Previous value gave std ~1.73 instead of 1.0.

2. **Bit-order mismatch between encoder update and decoder read.**
   Encoder packed `bits_i` at LSB position but decoder expected
   MSB-first in the state window. Fixed by making the emit convention
   little-endian: `state_i = (state_{i-1} >> K) | (bits_i << (L-K))`.

3. **Early-path decoder was wrong.** For samples before the state
   window is fully populated, the decoder needed to shift available
   bits into the TOP of the partial state to match encoder's
   initialization. Originally it placed them at the bottom.

4. **Zero-init start state wasted first L/K samples.** Encoder forced
   state_0 = 0, which meant state_1 had only 2^K distinct values
   instead of 2^L. Fixed by "open start": DP initializes dp[0][s] = 0
   for all s and writes the chosen start_state to the block header.
   Result: ~60% MSE reduction across all 2-bit configs.

---

## Key Findings

### 1. Post-RHT V-weights are near-perfectly Gaussian

Extracted from both Qwen3.5-0.8B and Qwen3.5-27B, the post-rotation
V-weight distribution has 1%-99% spread within ±2.3σ with tails at
±4σ. This matches the mathematical role of RHT: it gaussianizes.

Real-data MSE tracks Gaussian-synthetic MSE within 5%. The heavy-tail
failure modes observed on synthetic student5 and vcache-like data
do not appear on real post-RHT V-weights. **TABLE (inverse-Gaussian)
is the right code choice, not T5.**

### 2. Open start state is a 60% MSE win, almost free

Adds L bits per group (0.125-0.5 bpw depending on G). Paid once per
group, not per block. For Phase-2 configs with G=4, this is
<0.125 bpw overhead at Q=128 and 0.0625 bpw at Q=256.

### 3. Group-chained start_state halves bpw overhead

First block in a group of G uses open start; subsequent blocks chain
from the previous block's end_state at zero extra storage cost.
Combined with shared_d: overhead per group is 16 + L = 32 bits
regardless of G.

### 4. shared_d is free on post-RHT data

One fp16 scale per group instead of per block. MSE delta < 1%
relative on real V-weight data. Saves 16·(G-1)/(G·QK) bpw.
Would hurt more on heavy-tailed data; safe here.

### 5. QK=256 beats QK=128 at equal overhead ratios

Amortizing a fixed per-group header over more samples is
Pareto-dominant: Q256 configs always match or beat Q128 configs at
the same G/sharedD settings.

---

## Leaderboard

All numbers vs wikitext-2 PPL for Qwen3.5-27B with vtq V-cache.
Projections use linear MSE→PPL scaling calibrated on vtq2_1.

| Project              | Config           | bpw   | PPL delta         |
|----------------------|------------------|-------|-------------------|
| TheTom turbo4        | Q4_K_M + sym     | 4.25  | +0.23% (measured) |
| TheTom turbo3        | Q4_K_M + sym     | 3.50  | +1.06% (measured) |
| buun turbo3_tcq      | 3-bit TCQ        | 3.25  | -0.05% (measured) |
| **Uns K3_Q128_G4_sharedD** | trellis 3-bit | **3.06** | **~+0.9% (proj)** |
| TheTom turbo2        | Q4_K_M + sym     | 2.50  | +6.48% (measured) |
| vtq2_1 (ours, scalar)| Lloyd-Max 2-bit  | 2.50  | +5.10% (measured) |
| **Uns Q256_G1**      | trellis 2-bit    | **2.13** | **~+2.8% (proj)** |
| **Uns Q256_G4_sharedD**| trellis 2-bit  | **2.03** | **~+3.0% (proj)** |

---

## Phase-2 Candidates

**2-bit path: `Q256_G4_sharedD`**
- L=16, K=2, QK=256, G=4, shared_d=1, code=TABLE
- 2.031 bpw: 16-bit d shared across group + 16-bit start_state shared
  across group + 256·2 bit payload per block, averaged
- Projected +3.0% PPL at 19% lower bpw than vtq2_1

**3-bit path: `K3_Q128_G4_sharedD`**
- L=16, K=3, QK=128, G=4, shared_d=1, code=TABLE
- 3.063 bpw: first fork config that undercuts buun's 3.25 bpw
- Projected +0.9% PPL

Both use the same encoder/decoder with K as a compile-time constant.
A single GPU kernel can handle both.

---

## Phase-2 Scope

**Required for GPU port:**

- Block struct in `ggml-common.h`: `block_vtq2_2` and `block_vtq3_2`
  with shared-group metadata (d and start_state stored at group
  boundary, not per block). Storage layout:
  ```c
  struct block_vtq2_2 {
      uint8_t qs[256 * 2 / 8];     // 64 bytes payload per block
  };
  struct group_header_vtq2_2 {
      ggml_half d;                  // shared across G blocks
      uint16_t  start_state;        // shared across G blocks
  };
  ```
  KV-cache layout: one header per G blocks, inlined as first
  16 bytes of the group or stored in a parallel array.

- CUDA decoder: parallel bitshift read + TABLE LUT lookup + scale.
  Inner loop:
  ```cuda
  uint32_t state = read_bits(qs, K·i, L) ^ start_state;
  float val = LUT[state] * cb_scale * d;
  ```
  Register budget target: ≤16 live registers to inline in FA P·V loop.

- CUDA encoder: pruned Viterbi at beam 256-512. Runs on cache-write
  path (slow path), so CUDA port is less critical than decoder.

- FA dispatch table entries in `fattn.cu`.

**Deferred / optional:**

- Viterbi for cache-write on CPU initially; GPU encoder later if the
  CPU path is the bottleneck.
- Beam-width tuning — current beam infrastructure has known inefficiencies
  at S=65536; may need sparse DP for real speedup.

---

## Open Questions for Phase 2

1. **Does TABLE's 256 KB LUT fit in `__constant__` memory without
   evicting other things?** Test with ptxas verbose output.

2. **Can the decoder inline in the FA P·V inner loop?** Register count
   target is 16. If it blows budget, fall back to `__noinline__` with
   measured decode cost penalty.

3. **Real PPL measurement before GPU port?** CPU-path dequant through
   llama.cpp backend would validate the projections. Costs 1-2 days
   of integration work. Recommended before committing to Phase 2.

4. **K shared_d storage in kv-cache layout?** Per-group header adds
   a non-trivial indexing wrinkle vs the current per-block layout.
   Need a kv-cache allocation change.

---

## Source Layout

```
tests/trellis-phase1/
├── README.md                           # how to build and run
├── CMakeLists.txt                      # standalone cmake
├── trellis_phase1.h                    # public API
├── trellis_common.c                    # RNG, timing, data gens
├── trellis_code.c                      # 3GAUSS, TABLE, T5 code fns
├── trellis_encdec.c                    # Viterbi + bitshift decoder
├── trellis_main.c                      # config matrix + sweep driver
├── extract_v_samples.py                # GGUF V-weight extractor
└── results/
    ├── RUN1_NOTES.md, RUN1B_NOTES.md,
    ├── RUN2_NOTES.md, RUN3_NOTES.md,
    ├── RUN4_NOTES.md, RUN5_NOTES.md
    ├── run1_gauss_4k_zero_start.csv
    ├── run1b_gauss_4k_open_start.csv
    ├── run2_gauss_4k_table_vs_3g.csv
    ├── run3_beam_sweep.csv
    ├── run3b_beam_quickselect.csv
    ├── run4_group_sweep.csv
    ├── run5_dist_{gauss,laplace,student5,bimodal,vcachelike}.csv
    ├── run6_t5_vcachelike.csv
    ├── run6b_t5_gauss.csv
    ├── run7_real_vweights.csv
    ├── run7b_real_27b.csv
    └── run8_bpw_floor.csv
```

All commits on `trellis-v2-phase1` branch.

---

## Not Done

- **Real PPL validation.** Projections use linear MSE→PPL scaling
  from vtq2_1 baseline. Real PPL could diverge due to attention
  sensitivity effects not captured by MSE. Recommended before GPU port.
- **Parallel GPU decoder.** Current CPU decoder is iterative (shift
  register). For GPU we need the window-read variant, which requires
  an algebra check: does `read_bits(qs, Ki, L)` produce the same
  state as iterating the shift register from start_state? For
  open-start with chained groups the answer is non-trivial and may
  require storing the L-bit "phase" alongside start_state.
- **Metal/Vulkan port.** Out of scope; CUDA only per project charter.
