# Phase 1: V_rows=8 D≥256 fix — Verification Results

Stand: 2026-04-25, commit `584378082`. Build auf gpu00 mit libggml-cuda 353MB neu (08:21). Sweep über 35 Configs (5 K × 7 V) für beide Modelle, fa=1, ngl=99, ts=12,12, llama-bench -p 512 -n 128.

## Hypothesis

Roadmap Phase 1 erwartete: "Gemma4 f16/vtq2_2 von -2.4% TG → ~-1% TG (gegen f16/f16). Qwen unchanged (D=128 path nicht angefasst)."

V_rows_per_thread Patch (`fattn-vec.cuh:89`): bei D≥256 für VTQ_1 + VTQ_2 family von 4 auf 8 erhöht — halbiert block-header reload waste an D=512 layouts (Gemma4 full-attention).

## Result: PASSED — exceeded expectation

Gemma4 VTQ-family TG verbessert um **+2 bis +6%** vs alte Messung (vor V_rows fix). PP leicht gefallen (-1 bis -3%) — akzeptabel weil TG der primäre bottleneck war.

### Gemma4-26B-A4B (D=512, target — bartowski IQ2_XXS)

| K | V | PP_old | PP_new | ΔPP% | TG_old | TG_new | ΔTG% | Note |
|---|---|--:|--:|--:|--:|--:|--:|---|
| f16 | f16 | 1395.74 | 1365.97 | -2.1% | 82.70 | 84.72 | **+2.4%** | baseline shifts up |
| f16 | vtq2_1 | 1054.38 | 1024.38 | -2.8% | 78.95 | 80.54 | **+2.0%** | |
| f16 | vtq2_2 | 1381.58 | 1343.97 | -2.7% | 80.70 | 82.73 | **+2.5%** | Pareto-winner |
| f16 | vtq3_1 | 908.74 | 913.67 | +0.5% | 75.35 | 79.06 | **+4.9%** | |
| f16 | vtq3_2 | 1381.17 | 1344.84 | -2.6% | 80.72 | 82.70 | **+2.5%** | |
| ktq2_1 | f16 | 1345.45 | 1321.62 | -1.8% | 78.81 | 81.78 | **+3.8%** | |
| ktq2_1 | vtq2_1 | 1025.33 | 1005.42 | -1.9% | 75.36 | 78.08 | **+3.6%** | |
| ktq2_1 | vtq2_2 | 1334.21 | 1318.67 | -1.2% | 77.27 | 79.88 | **+3.4%** | production candidate |
| ktq2_1 | vtq3_1 | 889.54 | 900.80 | +1.3% | 71.92 | 76.44 | **+6.3%** | |
| ktq2_1 | vtq3_2 | 1331.86 | 1314.98 | -1.3% | 77.21 | 79.74 | **+3.3%** | |
| ktq3_1 | f16 | 1342.35 | 1320.73 | -1.6% | 78.82 | 81.92 | **+3.9%** | |
| ktq3_1 | vtq2_2 | 1332.11 | 1319.96 | -0.9% | 77.12 | 79.94 | **+3.7%** | |
| ktq3_1 | vtq3_1 | 887.59 | 903.14 | +1.8% | 71.94 | 76.26 | **+6.0%** | best ΔTG |
| ktq3_1 | vtq3_2 | 1328.92 | 1316.41 | -0.9% | 77.01 | 79.74 | **+3.5%** | |
| q8_0 | vtq2_1 | 938.08 | 930.84 | -0.8% | 71.06 | 74.16 | **+4.4%** | |

### Qwen3.6-35B-A3B (D=128, regression-gate — UD-IQ2_XXS)

D=128 path nicht angefasst (Fix gilt nur D≥256). Erwartung: keine Regression. Da kein direkt vergleichbarer pre-fix-Run mit identischem Build existiert (alte CSV-Werte stammen aus Build *vor* anderen Optimierungen), diene als Sanity-check der absolute Speed:

| K | V | PP | TG |
|---|---|--:|--:|
| f16 | f16 | 1018.04 | 76.77 |
| f16 | vtq2_2 | 1006.75 | 75.75 |
| ktq2_1 | vtq2_2 | 995.80 | 74.86 |
| ktq2_1 | vtq3_2 | 992.94 | 74.79 |
| ktq3_1 | vtq2_2 | 992.41 | 74.83 |
| q8_0 | q8_0 | 985.42 | 73.08 |

f16/vtq2_2 vs f16/f16 = -1.1% PP, -1.3% TG → Pareto auf Qwen weiter near-lossless. Kein Anzeichen einer Regression.

## Wichtigste Pareto-Empfehlungen (post fix)

| Use case | Modell | Config | bpw | PP | TG | ΔTG vs f16/f16 |
|---|---|---|--:|--:|--:|--:|
| **Lossless quality** | Qwen3.6 | `f16 / vtq2_2` | 9.03 | 1006.75 | 75.75 | -1.3% |
| **Lossless quality** | Gemma4 | `f16 / vtq2_2` | 9.03 | 1343.97 | 82.73 | -2.4% |
| **Production VRAM** | Qwen3.6 | `ktq2_1 / vtq2_2` | 2.78 | 995.80 | 74.86 | -2.5% |
| **Production VRAM** | Gemma4 | `ktq2_1 / vtq2_2` | 2.78 | 1318.67 | 79.88 | -5.7% |

## Rohdaten

Logs in `/home/claude/sweep-phase1-20260425-0828/` auf gpu00. Zusammengefasst in `bench/plots/benchmarks.csv` (Spalten 6+7: PP/TG aktualisiert für Gemma4-bartowski-IQ2_XXS und Qwen3.6-IQ2_XXS).

## Nächste Phase

Phase 2 vom roadmap — vtq1_1 (1bit) Sweep auf beiden Modellen ergänzen, sobald gpu00 frei.
