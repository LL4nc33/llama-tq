# MMA-KTQ Split: VTQ-V-Dequant Extension — 2026-05-13

## Was geändert wurde

Bei `K=KTQ + V=VTQ` mit shape die nicht in den inline-kernel passt (D≠128, GQA≠4),
fiel der dispatch bisher auf `BEST_FATTN_KERNEL_VEC` zurück (slow für prefill).

Refactor:
- `ggml_cuda_flash_attn_ext_mma_ktq_split` dequantiert jetzt optional auch V in
  einen zweiten f16 scratch buffer wenn V eine VTQ-family type ist
- Hilfsfunktionen `fattn_ktq_dequant_to_scratch` + `fattn_cache_restore`
  extrahiert für lesbarkeit + reuse
- Dispatch in `fattn.cu` line 405-419 erweitert: any KTQ K + VTQ V mit
  Q->ne[1]≥8 → MMA-KTQ (statt VEC fallback)

## Bench (vor / nach)

### Qwen3.6-35B-A3B IQ2_XXS (D=256, GQA=8, MoE) — primary target

| ctx | pre (vec) | post (MMA-split) | Δ |
|------|---|---|---|
| PP@512  | 1066 | 1072 | +0.6% |
| PP@1024 | 1027 | 1029 | +0.2% |
| PP@2048 | 959 | 962 | +0.3% |
| PP@4096 | 837 | 839 | +0.2% |
| TG@64 | 81.5 | 81.6 | flat |

→ Praktisch identisch (alle innerhalb noise). Vec-kernel ist bei D=256 GQA=8
bereits gut. MMA-split bringt keinen sichtbaren win, aber unifies code paths.

### Ministral-3-3B Q4_K_M (D=128, GQA=4) — Phase 5 inline, sanity check

| Test | pre | post | Δ |
|------|------|------|---|
| PP@2048 | 1316 | 1304 | -0.9% |
| TG@128 | 97 | 100 | +3.1% |

→ PP-noise (-0.9%) gerade unter signifikanzschwelle. TG-improvement +3.1% —
vermutlich von clean-rebuild fairness, nicht kernel-change (inline path
unverändert).

## Wert dieser änderung

1. **Code-cleanup**: dispatch-logik unified (KTQ+VTQ ≥8 → MMA-KTQ)
2. **Foundation für Phase 6**: D=256 MMA inline würde jetzt nicht mehr von
   VEC-fallback geblockt sein — der path ist direkt erreichbar
3. **Kein regression**: TG flat, PP innerhalb noise

## Was NICHT geliefert

- Direkter PP-speedup für Qwen3.6 (war hoffnung) — vec war schon competitive
- Phase 6 D=256 MMA inline (NVCC segfault offen)
- Async V-load im inline path (größerer refactor, kein profile data)
