# Run 3 — distribution robustness + T5 heavy-tail code

**Date:** 2026-04-17
**Data:** N=2048, 5 distributions (gauss, laplace, bimodal, student5, vcachelike)
**Codes:** TABLE (Gaussian inv-CDF) vs T5 (Student-t(5) inv-CDF)

## Key finding: TABLE fails on heavy-tailed data

MSE ratio on each distribution (L16_K2 configs with open start, G=1):

| Config        | gauss | laplace | bimodal | student5 | vcachelike |
|---------------|-------|---------|---------|----------|------------|
| L16_K2_Q32    | 0.33  | 0.32    | 0.33    | **0.67** | **0.79**   |
| L16_K2_Q64    | 0.44  | 0.44    | 0.44    | **1.14** | **1.07**   |
| L16_K2_Q128   | 0.50  | 0.51    | 0.49    | **1.73** | **1.24**   |
| L16_K3_Q32_G4 | 0.21  | 0.23    | 0.21    | **0.33** | 0.56       |

Heavy-tailed data (student5, vcachelike) pushes 2-bit trellis **above**
Lloyd-Max baseline (ratio > 1.0 for some configs). Root cause: TABLE's
inverse-Gaussian codes saturate at ~±3σ; samples at ±5σ get clamped to
the nearest code and contribute large squared errors.

## T5 code (Student-t(5) inverse CDF) rescues heavy tails

| Config        | TBL (vcachelike) | T5 (vcachelike) | Δ      |
|---------------|------------------|------------------|---------|
| L16_K2_Q32    | 0.79             | **0.68**         | −14%   |
| L16_K2_Q64    | 1.07             | **0.91**         | −15%   |
| L16_K2_Q128   | 1.24             | **1.06**         | −15%   |
| L16_K3_Q32_G4 | 0.56             | **0.42**         | −24%   |

But on clean Gaussian, T5 loses ~5-10% to TABLE (spends code budget on
tails the data doesn't have). Classic bias-variance trade-off.

## Strategic implication

**For 2-bit V-cache**: trellis is competitive on Gaussian-like V-cache
but **risky** on real V-cache with outlier channels. Before any GPU
port, real V-cache data from a live forward pass is mandatory. The
alternative is a 2-3 week CUDA port that fails on PPL.

**For 3-bit V-cache**: `L16_K3_Q32_G4` with T5 holds up even on
vcachelike data (ratio 0.42). Projected PPL delta ~+2% (vs buun's
−0.05% at 3.25 bpw). Our 3.625 bpw is 0.375 higher than buun, but
with asymmetric K/V we may compensate elsewhere. This config is a
solid GPU port candidate.

## Updated best-config shortlist (with T5 for heavy-tail safety)

| Config              | bpw    | gauss MSE | vcachelike MSE | GPU port? |
|---------------------|--------|-----------|-----------------|-----------|
| L16_K2_Q32_T5       | 3.000  | 0.349     | 0.678           | borderline|
| L16_K2_Q64_T5       | 2.500  | 0.456     | 0.911           | no        |
| **L16_K3_Q32_T5_G4**| 3.625  | 0.268     | 0.422           | **yes**   |

## Next step (critical): real V-cache data

The heavy-tail sensitivity makes synthetic benchmarks insufficient.
To validate any 2-bit config we need **actual V-cache tensor data**
dumped from a forward pass of Qwen3.5-27B (post `self_v_rot`). This
is a CUDA-hook addition, roughly 30 LOC. Worth doing before GPU
encoder work begins.

## Status summary

- Phase 1 compute infrastructure: **COMPLETE** (harness, Viterbi,
  decoder, code functions, data generators, beam, group-sharing)
- Phase 1 data sweeps: **COMPLETE** (all 6 runs)
- Phase 1 decision gate: **BLOCKED on real V-cache data**
- Phase 2 GPU port: **held** pending above gate
