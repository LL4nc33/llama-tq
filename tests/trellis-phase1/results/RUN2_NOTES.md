# Run 2 — TABLE code, beam, group-size

**Date:** 2026-04-17
**Data:** N=2048-4096 synthetic N(0,1), seed=42
**Encoder:** full Viterbi (beam=0), chained groups
**Decoder:** iterative shift register, chained across group
**Gate:** trellis MSE ≤ 0.7 · 0.1175 = 0.0822 (ratio ≤ 0.7)

## Code function: TABLE beats 3GAUSS by 3-7% MSE

Using precomputed inverse-CDF (TABLE) instead of 3GAUSS CLT hash:

| Config      | 3GAUSS MSE | TABLE MSE | Δ     |
|-------------|------------|-----------|-------|
| L8_K2_Q32   | 0.0683     | 0.0618    | −10%  |
| L16_K2_Q32  | 0.0416     | 0.0390    | −6%   |
| L16_K2_Q64  | 0.0544     | 0.0520    | −4%   |
| L16_K2_Q128 | 0.0621     | 0.0593    | −5%   |
| L16_K3_Q32  | 0.0113     | 0.0105    | −6%   |

TABLE at L=16 costs 256 KB of constant memory (fits CUDA __constant__).
No runtime cost since it's a single load + scale.

## Beam pruning: quickselect helps, but marginal win for this problem size

With qsort, beam was slower than full Viterbi. With quickselect-based
threshold finding it's 2x faster for b ≥ 2048 but **slower** for b ≤ 512
(the "prune INF states" path in the transition loop dominates). Real
speedup requires restructuring the transition loop to only iterate over
surviving predecessors (sparse DP). Deferred to Phase 2 (CUDA port will
have different performance characteristics anyway).

## Group-size: the bpw win

Sharing one start_state across G blocks gives the biggest bpw reduction
at small MSE cost. Best configs:

| Config              | bpw    | MSE    | ratio | Gate |
|---------------------|--------|--------|-------|------|
| L16_K2_Q128_G2      | 2.188  | 0.0667 | 0.57  | ✓    |
| L16_K2_Q128_G1      | 2.250  | 0.0591 | 0.50  | ✓    |
| L16_K2_Q64_G4       | 2.312  | 0.0758 | 0.65  | ✓    |
| L16_K2_Q64_G2       | 2.375  | 0.0674 | 0.57  | ✓    |
| L16_K2_Q64_G1       | 2.500  | 0.0513 | 0.44  | ✓    |
| L16_K2_Q32_G1       | 3.000  | 0.0387 | 0.33  | ✓    |
| L16_K3_Q32_G4       | 3.625  | 0.0248 | 0.21  | ✓ (3-bit) |

## Projected PPL vs other projects

MSE-to-PPL scaling (conservative linear):

| Project             | bpw   | MSE ratio | PPL delta (measured/est.) |
|---------------------|-------|-----------|----------------------------|
| vtq2_1 (baseline)   | 2.500 | 1.000     | +5.1% (measured)           |
| TheTom turbo2       | 2.500 | ~1.05     | +6.48% (measured)          |
| **L16_K2_Q128_G2**  | **2.188** | **0.57** | **~+2.9% (projected)** |
| L16_K2_Q64_G1       | 2.500 | 0.44      | ~+2.3% (projected)         |
| buun turbo3_tcq     | 3.250 | ?         | −0.05% (measured)          |
| **L16_K3_Q32_G4**   | **3.625** | **0.21** | **~+1.1% (projected)** |

**L16_K2_Q128_G2** reaches lower bpw than vtq2_1 AND lower MSE:
that's a strict improvement, not a trade-off. Primary Phase-2 candidate.

## Known issues

1. Decoder is currently iterative (not parallel). GPU port needs the
   window-read variant back, with verification that XOR-ing start_state
   into read-bits yields the same state as the shift-register walk.
2. Encode time at L=16 is 80-90 ms/block full Viterbi. Beam pruning
   doesn't yet speed up L=16 + small beam due to the transition-loop
   structure. Either sparse DP or GPU-parallel encoder is the real fix.
3. 3GAUSS has bounded tails (±3σ) — outliers may be quantized poorly.
   TABLE avoids this.
4. cb_scale = 1/sqrt(N) is theoretical; empirical E[g²] for 3GAUSS is
   0.978, so a 1.0113x rescale would gain ~1% more. Low priority.

## Next steps

- [x] Run 1 baseline: DONE
- [x] Run 1b open start: DONE
- [x] Run 2 TABLE + beam + group: DONE
- [ ] Real V-cache data (Task #93) — validate synthetic MSE projections
- [ ] PPL validation (Task #95) — top-2 configs via CPU-path llama.cpp
- [ ] Commit block struct (Task #97) — after PPL validates
