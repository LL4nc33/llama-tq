# PP-Baseline nach V_rows=8 OOB-Fix — 2026-05-13

## Kontext

- Phase 5 kernel (KTQ2_1 K + VTQ2_1 V MMA inline) gemerged
- V_rows=8 OOB-bug auf D=128 gefixt (commit `135829149`)
- Build: `135829149 (17692)` auf gpu00, RTX 2060, sm_75
- Ziel: prefill-throughput im short-/mid-ctx-bereich (sweet-spot 2k-4k) maximieren

## Bench-results

Single-GPU0, `-ctk ktq2_1 -ctv vtq2_1 -fa 1 -ngl 99`, llama-bench -r 2.

### Ministral-3-3B

| ctx | Q4_K_M (1.99 GiB) | UD-IQ2_XXS (1.03 GiB) |
|------|---|---|
| PP@512  | **1977.5 t/s** | 1983.4 t/s |
| PP@1024 | 1698.8 t/s | 1685.2 t/s |
| **PP@2048** | **1316.0 t/s** | 1277.2 t/s |
| PP@4096 | 901.0 t/s | 858.7 t/s |
| PP@8192 | 546.0 t/s | 517.7 t/s |
| PP@10240 | 444.2 t/s | 431.7 t/s |
| TG@128 | 97.3 t/s | 97.8 t/s |

### Ministral-3-14B IQ2_XXS (3.77 GiB)

| ctx | t/s |
|------|------|
| PP@512  | 736.1 |
| PP@1024 | 668.1 |
| PP@2048 | 555.5 |
| PP@4096 | 412.1 |
| TG@128 | 31.4 |

## Analyse

### Scaling-decay
PP-decay folgt klar einer **O(N²) attention-quadratisierung**:
- 3B Q4_K_M: PP@512 / PP@4096 = 1977/901 = **2.19×** (theoretisch 8× erwartet bei linear)
- Decay-rate ist sublinear weil prefill kompute-dominiert ist, KV-cache-lookups erst bei großer ctx-länge knappen werden

### Bottleneck-hypothesen für PP@2k
1. **MMA tile-scheduling**: ncols1=8 fix instantiiert, scheduling-block muss durch ⌈2048/8⌉ = 256 tiles
2. **K-tile-load FWHT**: 32-element warp-FWHT pro K-block, kein cp.async (V-tile schon ohne cp.async, K mit FWHT muss synchron sein)
3. **shared-mem-bank conflicts**: tile_K und tile_V auf demselben bank-range bei D=128

### Optimization-targets
1. **K-tile-load + V-tile-load parallelisieren** (asym warp split: ncols1 warps für K, rest für V)
2. **cp.async für V-tile** (V hat keine warp-coop FWHT-abhängigkeit)
3. **`ldmatrix.x4` für tile_K → MMA input** (statt sequential half2 loads)

## TG-baseline
TG@128 = 97.3 t/s — quasi-identisch zu pre-Phase-5 (97-98 t/s war altes baseline), kein decode-regression durch V_rows-fix.

## Nächste schritte
1. PP@2k bottleneck identifizieren (profiling oder code-walk)
2. Micro-optimization → re-bench → commit-cycle
3. Ziel: PP@2k von 1316 → ≥1600 t/s (+22%)
