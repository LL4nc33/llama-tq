# TurboQuant v8 Unified — Master-Plan

**Started:** 2026-05-02 03:00 CEST (autonom über Nacht)
**Goal:** 15 KTQ/VTQ-Types → 8 unified types (`ktq{1,2,3,4}` + `vtq{1,2,3,4}`)
**Combine:** Trellis (vtq*_2) + Outlier-Sidecar (vtq*_3) + Lloyd-Max (vtq*_1) + RHT (ktq*_1)

## Phasen-Übersicht

| Phase | Status | Wallclock | Output |
|---|---|---|---|
| 1. Algorithm-Spec | DONE | ~5min | `v8-algorithm-spec.md` (research output) |
| 2. CUDA-Architecture-Plan | DONE | ~5min | `v8-cuda-architecture.md` |
| 3. Validation-Harness-Plan | DONE | ~5min | `v8-validation-harness.md` |
| 4. Python-PPL-Sim | IN PROGRESS | ~15min | `bench/sim/v8_unified_results.md` GO/NO-GO |
| 5. Block-Struct + CPU Quant/Dequant | PENDING | ~2-3h | ggml-common.h, ggml-quants.c |
| 6. CUDA FA Kernel + Dispatch | PENDING | ~3-4h | turboquant-v8.cuh, fattn-tq-v8.cuh |
| 7. PPL-Sweep auf 8 Modellen | PENDING (gates 5-6) | ~6-7h | bench/results/v8_ppl_*.tsv |
| 8. Speed-Bench auf 4 Modellen | PENDING | ~30min | bench/results/v8_speed_*.tsv |
| 9. Pareto-Chart-Update | PENDING | ~10min | docs/img/*.png |
| 10. Legacy-Marker + README | PENDING | ~30min | docs/turboquant.md |

## Decision Gates

- **Gate A (sim):** Python-Sim muss zeigen dass v8-default ≤ MSE von ktq2_1+vtq2_2 auf 35B-A3B-distribution.
  Falls FAIL: Pivot zu Option A (Aliasing, kein neuer Backend).
- **Gate B (35B):** v8-default PPL drift ≤ +3.85% auf 35B-A3B-IQ2_XXS, TG ≥ 70 t/s. 
  Falls FAIL: v8 nur als opt-in, prod bleibt auf legacy.
- **Gate C (4B):** v8-default PPL drift ≤ +0.20% auf Qwen3.5-4B-Q4_K_M. 
  Falls FAIL: Pro-Modell-Empfehlung dokumentieren.
- **Gate D (universal):** Bei keinem der 8 Modelle darf v8 schlechter sein als best-of-existing in derselben bpw-Klasse.

## Backwards-Compat-Strategie

- Alte Enums (42-57) bleiben unverändert. Neue Enums (60-67) für v8.
- CLI: `--cache-type-k ktq2` (kurz, neu) und `--cache-type-k ktq2_1` (legacy) parallel.
- Server-Log: deprecation warning für legacy types.
- ABI break erst in Phase 2 (in 3 Wochen, separates PR).

## Hardware-Umgebung

- **Dev-Sandbox:** GPU0 (RTX 2060 12GB, x16). Nach 03:02 CEST geräumt (8/11825 MB).
- **Untouched:** GPU1 (RTX 2060 12GB, x4) — TTS + FunctionGemma + 4B Triple-Goal Deploy live.
- **Build-Tool:** llama-tq turboquant branch, gpu00.

## Redeploy nach Dev-Phase

Snapshot der GPU0 Settings: `/home/claude/gpu0_redeploy_settings.md` auf gpu00.

```bash
ssh claude@gpu00.node "bash ~/llama-tq/scripts/deploy-35b-singlegpu-100k.sh"
```
