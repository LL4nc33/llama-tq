# Run 1 — Gaussian sweep, zero-init start state

**Date:** 2026-04-17
**Data:** N=4096 synthetic N(0,1), seed=42
**Encoder:** full Viterbi (no pruning), norm correction on
**Code function:** 3GAUSS
**Lloyd-Max 2-bit baseline MSE:** 0.1175 (Beta(15.5,15.5))
**Gate:** trellis MSE ≤ 0.7 · baseline = 0.0822

## Results

| Config       | MSE    | ratio | Gate | Encode/block |
|--------------|--------|-------|------|--------------|
| L8_K2_Q32    | 0.109  | 0.92  | fail | 0.3 ms       |
| L16_K2_Q32   | 0.162  | 1.38  | fail | 58 ms        |
| L16_K2_Q64   | 0.113  | 0.96  | fail | 136 ms       |
| L16_K2_Q128  | 0.090  | 0.77  | fail (borderline) | 304 ms |
| L8_K2_Q128   | 0.089  | 0.76  | fail (borderline) | 1.1 ms |
| L12_K2_Q64   | 0.094  | 0.80  | fail | 8.5 ms       |
| L16_K3_Q32   | 0.054  | 0.46  | **pass** | 110 ms   |

## Observations

1. **2-bit trellis does not clear the 0.7 gate** with zero-init start state.
   Best 2-bit config (L8_K2_Q128) reaches ratio 0.76, still above 0.7.
2. **L=16 is worse than L=8 at QK=32.** State needs L/K = 8 emit steps to
   fill; on Q=32 that's 25% of the block with partial-state codes. On
   Q=128 only 6% — L16 beats L8 there.
3. **3-bit trellis clears the gate comfortably** (0.46). 3-bit may be
   the cheapest path to a real PPL improvement.
4. **Encode time with full Viterbi** is impractical at L=16 (100s of ms
   per block). Pruning is mandatory for cache-write path.

## Next experiments (before any GPU work)

### Fix 1: non-zero start state

Current encoder forces state_0 = 0, wasting the first L/K emit steps on
states with (L - K·i) zero bits feeding into the code function. Two
options:

- **Open start**: DP initializes dp[0][s] = 0 for all s. The decoder needs
  to know the start state — store it as a header bit pattern, or recover
  it by reading the first L bits of qs[] (which already encodes state_L).
- **Tail biting**: constrain end_state == start_state, DP over all
  possible start states. Cost: 2^L × current compute. At L=16, this is
  prohibitive with full Viterbi — only viable with pruning.

Start with option 1 (open start + stored start state).

### Fix 2: code function scale revisit

cb_scale = 1/sqrt(N) assumes sum of squared codes ≈ N. For 3GAUSS with
CLT approximation, codes have variance ≈ 1 but finite support [-3, 3]σ.
On blocks of 32, empirical sum-of-squares might differ from N. Measure
and rescale.

### Fix 3: test with real V-cache data

Pre-rotation V-cache values are not pure Gaussian — they have heavier
tails and occasional outliers. Real-data MSE could be lower or higher
than Gaussian. Required before rejecting the gate.
