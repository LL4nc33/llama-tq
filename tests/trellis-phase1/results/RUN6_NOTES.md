# Run 6 — final configs, group-Viterbi, QK=512, real 27B data

**Date:** 2026-04-17
**Data:** 16384 post-RHT V-weight samples from Qwen3.5-27B
**Encoder:** group-Viterbi (joint DP over G·QK samples), TABLE code, shared_d
**Baseline:** vtq2_1 at 2.500 bpw, MSE ratio 1.00, measured PPL +5.10%

## Results (sorted by bpw)

### 2-bit path

| Config           | bpw    | MSE    | ratio | proj. PPL |
|------------------|--------|--------|-------|-----------|
| **Q512_G8_group**| **2.008** | 0.068 | 0.576 | **~+2.9%** |
| Q512_G4_group    | 2.016  | 0.067  | 0.574 | ~+2.9%    |
| Q256_G4_group    | 2.031  | 0.067  | 0.570 | ~+2.9%    |
| Q128_G4_group    | 2.063  | 0.066  | 0.562 | ~+2.9%    |
| vtq2_1 (base)    | 2.500  | 0.117  | 1.000 | +5.10%    |

### 3-bit path

| Config             | bpw    | MSE    | ratio | proj. PPL |
|--------------------|--------|--------|-------|-----------|
| **K3_Q512_G8_group** | **3.008** | 0.017 | 0.147 | **~+0.75%** |
| K3_Q512_G4_group   | 3.016  | 0.017  | 0.147 | ~+0.75%   |
| K3_Q256_G4_group   | 3.031  | 0.017  | 0.146 | ~+0.74%   |
| K3_Q128_G4_group   | 3.063  | 0.017  | 0.144 | ~+0.73%   |
| buun turbo3_tcq    | 3.250  | ?      | ?     | -0.05%    |

## Leaderboard

| Project              | Config            | bpw    | PPL       |
|----------------------|-------------------|--------|-----------|
| TheTom turbo4        | Q4_K_M symmetric  | 4.250  | +0.23% (m)|
| TheTom turbo3        | Q4_K_M symmetric  | 3.500  | +1.06% (m)|
| **Uns K3_Q512_G8**   | 3-bit trellis+grp | **3.008** | **~+0.75% (p)** |
| buun turbo3_tcq      | 3-bit TCQ         | 3.250  | -0.05% (m)|
| TheTom turbo2        | Q4_K_M symmetric  | 2.500  | +6.48% (m)|
| vtq2_1               | Lloyd-Max 2-bit   | 2.500  | +5.10% (m)|
| **Uns Q512_G8**      | 2-bit trellis+grp | **2.008** | **~+2.9% (p)**  |

(m) = measured, (p) = projected via linear MSE-to-PPL scaling.

## Interpretation

**2-bit: decisive improvement over vtq2_1 and TheTom.**
19.7% lower bpw than vtq2_1 with 43% less MSE. Projected PPL delta
+2.9% vs measured +5.1% / +6.48%.

**3-bit: closest fork yet to buun's PPL at lower bpw.**
7.4% lower bpw than buun. Projected PPL gap of ~0.8% at same bpw
spread. Real PPL measurement will settle whether we actually beat
buun or just match at lower bpw.

## Optimizations stacked in Phase 1

- Open start state (Run 1b): -60% MSE
- Group-chained start_state: -0.125 bpw
- TABLE code (inv-Gaussian CDF): -3-10% MSE
- shared_d across group: -0.125 bpw, ~0 MSE cost
- QK=256: -0.062 bpw
- Rolling DP buffer: enables QK=512
- QK=512 + G=8: -0.008 bpw
- Group-level Viterbi: -17% MSE at same bpw
- L=18/20 (marginal, deferred): -2-3% MSE at 4-17× encode time

## Phase-2 recommended configs

**Primary 2-bit: `Q256_G4_group`** at 2.031 bpw
Rationale: QK=256 matches common head_dim·n_kv_group, manageable
CUDA indexing. Q512 variants save 0.02 bpw at comparable MSE but
the kernel tiling is more awkward at 512 elements.

**Primary 3-bit: `K3_Q128_G4_group`** at 3.063 bpw
Rationale: head_dim=128 matches one block exactly; zero index math
in the decoder inner loop. Gives up 0.05 bpw vs Q512 but wins on
GPU-friendliness.

Both use L=16, TABLE code, shared_d, group-Viterbi encoder.
Single GPU kernel handles both with K as compile-time template.
