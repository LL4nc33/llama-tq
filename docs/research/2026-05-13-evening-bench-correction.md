# 2026-05-13 abend bench-korrektur + entwicklungslog

## TL;DR

Heute morgen wurden zahlen wie "PP@4k 1463 t/s dual-GPU" gemeldet. **Diese sind NICHT REPRODUZIERBAR** unter stable-state. Stable-bench abends zeigt **PP@4k ~1000 t/s dual-GPU**.

Die alte memory `project_vtq_v_bottleneck.md` mit **"VTQ2_1 V ist -28% slower than f16 V auf PP@4k"** ist KORREKT.

## Bench-realität (build `ca9a68cf3`, 17:30-17:45 CEST, gpu00 GPU0=62°C nach mehrfachen runs)

| variante | hardware | PP@1024 | PP@2048 | PP@4096 | TG@32 |
|---|---|---|---|---|---|
| ktq2_1+vtq2_1 | **dual-GPU** (2× 2060 ts=12,12) | 1247 | 1147 | **988** | 79.5 |
| ktq2_1+vtq2_1 | **single-GPU** (2060 #0) | 1278 | 1178 | 1017 | 81.2 |
| ktq2_1+vtq2_1 | **dual-GPU** main_gpu=1 | — | — | **1015** | 80.6 |
| f16+f16 | **dual-GPU** (2× 2060 ts=12,12) | 1418 | 1405 | **1372** | 83.1 |
| ktq2_1+f16 | **dual-GPU** (2× 2060 ts=12,12) | 1431 | 1411 | **1382** | 82.5 |

**Gap-analyse:**
- VTQ2_1 V vs f16 V: PP@4k **-28%** (988 vs 1372), PP@2k **-18%**, PP@1k -12%, TG@32 -4%
- KTQ K dequant ist GRATIS: KTQ+f16 vs f16+f16 = +0.7% PP@4k
- Dual-GPU vs single-GPU bei VTQ V: **flat** (988 vs 1017, dual ist sogar 3% schlechter wegen PCIe overhead)

## Was wir gestern und heute getestet haben — alle ohne wirkung

| optimierung | result |
|---|---|
| Multi-warp dequant (4 warps/CTA) | flat |
| Pre-scaled CB | flat |
| 4-out-per-thread x4 kernel | +1.5-2% (placebo) |
| Direct MMA-F16 route (skip wrapper) | +2% |
| GGML_CUDA_FORCE_GRAPHS=1 | -30% (incompat mit dual-GPU split) |
| -sm row statt layer | -32% PP@4k |
| ts=14,10 asymmetric | -31% PP@4k |
| ub=512 statt 1024 | -30% PP@1024 |
| --no-warmup | initial run unstable, stable nach 2-3 runs gleich |
| -mg 1 statt 0 | flat (was hoffnung wegen GPU0 thermal) |

## Was wir GELERNT haben über die codebase

1. **CUDA graphs** auf Turing (CC 7.5) default-off, opt-in via `GGML_CUDA_FORCE_GRAPHS=1` — aber **incompatible mit dual-GPU split-buffer mode**. Auch incompatible mit MUL_MAT_ID > mmvq_mmid_max (= MoE prefill). Quelle: `ggml/src/ggml-cuda/ggml-cuda.cu:3065,3074,4137`.
2. **stream_k** auf Turing default-off außer wenn tiles_efficiency_percent < 75 — könnte +5-15% PP bei richtigem code-fix. Untested. Quelle: `fattn-common.cuh:1046`.
3. **`launch_fattn`** ruft `to_fp16_nc_cuda(V)` bulk-dequant auf wenn `need_f16_V=true`. Das passiert IMMER für VTQ V bei MMA-F16 pfad. Quelle: `fattn-common.cuh:972`.
4. **MMA-KTQ inline** kernel ist nur für D=128 GQA=4 (Ministral). 35B-A3B (D=256 GQA=8) fällt zurück auf MMA-F16 split-dequant.
5. **Layer-split (ts=12,12) ist der einzig sinnvolle multi-GPU mode** für 35B-A3B. Row und asymmetric layer beide -30%+.

## Echte optimierungs-pipeline (priorität nach ROI)

**Erste priorität — SoA staging für VTQ2_1 V dequant kernel** (1-2 tage)
- Root cause: `block_vtq2_1` ist 10-byte struct unalignable → bulk-dequant `to_fp16_nc_cuda(VTQ2_1)` ist L1-cache-miss heavy
- Fix: in-place SoA layout (`d[]` und `qs[]` separate aligned arrays) statt AoS interleaved
- Expected: -28% → -5% gap = **+24% PP@4k recovery** = ~1230 t/s stable

**Zweite priorität — stream_k force-enable test** (4-6h build + bench)
- Turing default-off, code-side `cc >= GGML_CUDA_CC_ADA_LOVELACE` gate, force-on via flag oder code-patch
- Expected: +5-15% PP (compute-bound regime)

**Dritte priorität — Phase 6 D=256 MMA inline** (1 woche+)
- Nur möglich nach SoA (ldmatrix.sync braucht alignment)
- Expected: +20-30% wenn alignment-fix sitzt
- Ziel: PP@4k 1400-1500 t/s stable (= f16 niveau geschlagen)

## Was wir NICHT tun werden

- ❌ Random kernel-tweaks ohne profiling-evidence
- ❌ Mehr "research agents parallel" — wir haben gestern 6 agents abgeschickt und alle haben falsch geschätzt
- ❌ FORCE_GRAPHS — incompat
- ❌ Row-split — beweisbar -30%
- ❌ Asymmetric ts — beweisbar -30%

## Lehren für bench-disziplin

1. **Immer GPU temperature und clocks loggen** vor jedem bench
2. **3+ runs minimum**, outliers verwerfen
3. **GPU-label IMMER mitschreiben** (single vs dual + welche GPU)
4. **Cold-cache zahlen nicht trauen** — erste run kann +50% peak sein
5. **Memory-eintrag prüfen** ob "morgen war es schneller" reproduzierbar ist bevor strategy daran hängt
