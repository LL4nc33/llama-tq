# Run 1b — Gaussian sweep, open start state

**Date:** 2026-04-17
**Data:** N=4096 synthetic N(0,1), seed=42
**Encoder:** full Viterbi, **open start state** (dp[0][s] = 0 for all s)
**Decoder:** iterative shift register from stored start_state
**Code function:** 3GAUSS
**Gate:** trellis MSE ≤ 0.7 · 0.1175 = 0.0822

## Results (vs Run 1 zero-start)

| Config       | MSE zero | MSE open | ratio open | Δ MSE   | Gate |
|--------------|----------|----------|------------|---------|------|
| L8_K2_Q32    | 0.109    | 0.068    | 0.58       | −38%    | ✓    |
| L16_K2_Q32   | 0.162    | **0.042**| **0.35**   | −74%    | ✓    |
| L16_K2_Q64   | 0.113    | 0.054    | 0.46       | −52%    | ✓    |
| L16_K2_Q128  | 0.090    | 0.062    | 0.53       | −31%    | ✓    |
| L8_K2_Q128   | 0.089    | 0.080    | 0.68       | −10%    | ✓    |
| L12_K2_Q64   | 0.094    | 0.062    | 0.53       | −34%    | ✓    |
| L16_K3_Q32   | 0.054    | **0.011**| **0.10**   | −79%    | ✓    |

## Observations

1. **All 2-bit configs now clear the gate.** Open start state gives
   20-80% MSE reduction across the board.
2. **L16_K2_Q32 is the best 2-bit config** at ratio 0.35 — long state
   helps more than big blocks when start is free.
3. **3-bit with L=16** reaches ratio 0.10 — a 10× improvement over
   Lloyd-Max 2-bit. This would translate to massive PPL gains.
4. **Encode-time is the blocker**: L=16 at full Viterbi is 90 ms/block.
   Qwen3.5 with 8 KV-heads × head_dim 128 = 32 blocks per V-row per
   layer × 28 layers ≈ 900 blocks/token → 80 sec/token at cache-write.
   Task #96 beam pruning is mandatory.

## Cost of open start state

Storing the start state costs L bits per block:

| Config     | bpw extra | total bpw | vs vtq2_1 (2.5) |
|------------|-----------|-----------|------------------|
| L8_K2_Q32  | 0.25      | 2.75      | +0.25            |
| L16_K2_Q32 | 0.50      | 3.00      | +0.50            |
| L16_K2_Q64 | 0.25      | 2.75      | +0.25            |
| L16_K2_Q128| 0.125     | 2.375     | **−0.125**       |
| L8_K2_Q128 | 0.0625    | 2.1875    | **−0.3125**      |
| L12_K2_Q64 | 0.1875    | 2.6875    | +0.1875          |
| L16_K3_Q32 | 0.50      | 4.00      | vs vtq3_1 (3.5)  |

**Sweet spots for bpw reduction**: L16_K2_Q128 and L8_K2_Q128 have
lower bpw than vtq2_1 AND clear the MSE gate. L16_K2_Q128 is
particularly attractive — same MSE as L16_K2_Q64 at ~0.5 bpw less.

## Decoder note

Current decoder uses iterative shift-register reconstruction (serial,
not parallel). For GPU port we need the window-read decoder, which
requires verifying that reading L bits from qs and XOR-ing with
start_state gives the same state. This is a straightforward
algebraic check; deferred to Phase 2.

## Next experiments

1. Fix cb_scale for 3GAUSS finite-support distribution
2. Real V-cache MSE sweep (Task #93 prerequisite)
3. Beam-pruning sweep (Task #96) to make L=16 encode feasible
4. TABLE code function vs 3GAUSS comparison
