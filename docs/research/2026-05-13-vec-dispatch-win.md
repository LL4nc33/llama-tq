# 2026-05-13 abend — VEC dispatch revert win (35B-A3B)

## TL;DR

Revert von commit f46e9626f (MMA-F16 routing für KTQ K + VTQ V) brachte +3-4% PP, +1.8% TG. Stable, reproduzierbar.

## Bench

Build `a6e74a21f`, dual-GPU 2× RTX 2060 ts=12,12, GPU0=47°C cool, 35B-A3B IQ2_XXS:

| metric | f46e9626f (MMA-F16) | a6e74a21f (VEC) | Δ |
|---|---|---|---|
| PP@1024 | 1247 | **1286** | +3.1% |
| PP@2048 | 1147 | **1184** | +3.2% |
| PP@4096 | 988 | **1024** | +3.6% |
| TG@32 | 79.5 | **80.9** | +1.8% |

Gap vs f16/f16 baseline (1372 PP@4k): vorher -28%, jetzt -25%.

## Cross-check: KTQ K Synergie wiederhergestellt

| K | V | dispatch | PP@4k |
|---|---|----------|-------|
| f16 | f16 | MMA-F16 | 1372 |
| ktq2_1 | f16 | MMA-F16 | 1382 |
| f16 | vtq2_1 | VEC (vorher + jetzt) | 1031 / 1017 |
| ktq2_1 | vtq2_1 | MMA-F16 (vorher) | 988 |
| **ktq2_1** | **vtq2_1** | **VEC (jetzt)** | **1024** |

Vor revert: KTQ K bei VTQ V kostete -4% (988 vs 1031). Nach revert: KTQ K bringt +0.7% (1024 vs 1017) — wieder konsistent mit "KTQ K dequant ist gratis".

## Root cause

MMA-F16 pfad ruft `to_fp16_nc_cuda(VTQ2_1)` auf für V → fp16 scratch (1 GB transient pro forward bei 4k context). Das verdrängt aus L2-cache und bremst den FA-kernel selber.

VEC pfad macht in-kernel dequant via `dequantize_V_vtq2_1` (fattn-tq.cuh:735) — kein scratch, keine cache-pressure.

Tensor-core advantage von MMA-F16 (theoretisch +20%) wurde durch cache-pressure overkompensiert.

## Hypothese-test ohne komplexen code-change

War nur 1-zeilen-revert. Genau das was wir wollten: **kleine code-änderung mit großem effekt**.

## Wo wir jetzt stehen

- PP@4k: 988 → **1024** (verbleibend -25% vs f16, war -28%)
- PP@2k: 1147 → 1184
- PP@1k: 1247 → 1286  
- TG@32: 79.5 → 80.9

## Nächste schritte

1. **stream_k force-enable** — Turing default-off, könnte +5-15% PP. Build needed.
2. **VEC pfad selber optimieren** — der `dequantize_V_vtq2_1` in fattn-tq.cuh wird pro element gecallt; könnte gewinnt aus row-cooperative pattern haben.
3. **SoA staging** — bleibt valide für recovery der verbleibenden -25%.
