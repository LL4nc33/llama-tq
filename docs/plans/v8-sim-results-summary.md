# v8 Unified Sim — Results Summary (2026-05-02)

## Gate Decision: HYBRID-GO ✅

3 von 4 v8-Kandidaten **dominieren** die Pareto-Frontier deutlich. Der einzige "loser" ist `ktq1_v8` — und dieser Verlust ist eine **Bucket-Vergleichs-Anomalie**, kein echter Backend-Mangel:

- `ktq1_v8` hat 5.5 bpw aber nur 1-bit indices + 4 outliers
- Im 5.5-bpw bucket konkurriert es gegen `ktq4_1` (5.5 bpw, 4-bit codebook, 0 outliers)
- 4-bit codebook bei ähnlicher bpw schlägt 1-bit + outliers — erwartet

**Recommendation:** v8 deployment focused on V-cache (alle 3 winners). KTQ bleibt unverändert wie v7 (Aliase reichen).

## Detailed Results

### V-cache (vtq) — Pareto-dominiert

| bpw | Best existing | Best MSE | v8 winner | v8 MSE | Improvement |
|---|---|---|---|---|---|
| 2.5 | `vtq2_1` | 1.60e-1 | **`vtq2_v8`** (2.62 bpw) | **2.86e-2** | **5.6× besser** |
| 3.5 | `ktq2_1` | 1.12e-1 | **`vtq3_v8`** (3.62 bpw) | **7.77e-3** | **14.5× besser** |
| 4.5 | `vtq4_1` | 2.59e-2 | **`vtq4_v8`** (4.62 bpw) | **5.22e-3** | **5.0× besser** |

### K-cache (ktq) — bleibt v7

| bpw | Best existing | Best MSE | v8 candidate | Outcome |
|---|---|---|---|---|
| 5.5 | `ktq4_1` | 8.46e-3 | `ktq1_v8` | LOSER (Bucket-Anomalie, nicht echte Regression) |

KTQ keeps v7 layout (RHT + Lloyd-Max ohne Outlier). KTQ aliases `ktq1..ktq4` → existing `ktq*_1` enums.

## PPL-Drift Estimates (calibrated rule: rel_MSE × 0.6 × 100%)

Verglichen gegen f16/f16 baseline:

| Type | rel_MSE | PPL drift est | Real Triple-Goal config Ziel |
|---|---|---|---|
| `vtq2_v8` | 0.041 | **+2.4%** | < 35B-A3B current +3.85% ✅ |
| `vtq3_v8` | 0.011 | **+0.66%** | < 4B-dense current +0.20% ❓ (close call) |
| `vtq4_v8` | 0.007 | **+0.44%** | Lossless tier |

Falls real PPL-Drift den Sim-MSE folgt, wäre `vtq3_v8` der **klare Pareto-Gewinner**: 14× bessere MSE als ktq2_1 bei nur 0.12 bpw mehr (3.62 vs 3.5). Echter PPL drift muss in 35B + 4B Sweep gemessen werden.

## Implementation Status (commit progression)

- `a9b471026` — CLI aliases ktq1..4 + vtq1..4 (mapped to v7)
- `8cb4db877` — vtq3_v8 enum 58 + CPU quant/dequant
- `1910c4180` — vtq3_v8 CUDA dispatch (templated OUTLIER_K)
- `834b7e55d` — Validation harness + Python sim + plans

## Next Steps

1. **Build verify** auf gpu00 (running)
2. **Smoke test** vtq3_v8 mit qwen3.5-0.8b-q8_0 oder Qwen3.5-4B  
3. **PPL Sweep Gate B**: 35B-A3B mit ktq2/vtq3 v8 vs current prod (ktq2_1+vtq2_1)
4. **PPL Sweep Gate C**: 4B-Q4_K_M mit ktq2/vtq3 v8 vs current best (ktq2_1+vtq4_1)
5. **Deploy** falls Gates passed

## Caveats

- Trellis path uses beam-pruned Viterbi (beam=1024 statt 2^16). MSE leicht pessimistisch vs CUDA full-Viterbi.
- Synthetic V mit 1% 5σ outliers — echte Transformer-V-distribution kann schwerere Tails haben (höherer Outlier-Sidecar-Win).
- Calibration-Faustregel `PPL_drift ≈ rel_MSE × 0.6` ist single-anchor (ktq2_1+vtq2_2 +0.16%) — Größenordnung, nicht Punktvorhersage.
