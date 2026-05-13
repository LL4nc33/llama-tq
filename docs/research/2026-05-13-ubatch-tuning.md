# Ubatch-Size Tuning: Model-Dependent PP Optimization — 2026-05-13

## TL;DR

Auf top von dual-GPU layer-split bringt **ubatch=128** weiteren +10-18% PP-gain
für Ministral-3 (3B, 14B), aber **schadet 35B-A3B** (-22% vs default).

**Best PP@2k konfiguration: 3B Q4_K_M dual-GPU + ub=128 = 2168 t/s.**

## Sweep results (build `6cc4920d0`, dual-GPU layer-split)

### Ministral-3-3B Q4_K_M

| -ub | PP@1024 | PP@2048 | PP@4096 |
|-----|---------|---------|---------|
| 128 | **2595** | **2168** | **1557** |
| 256 | 2099 | 2099 | — |
| 512 (default) | 2064 | 1851 | 1405 |
| 1024 | — | 1648 | — |

→ ub=128 wins clear für 3B short/mid ctx.

### Ministral-3-14B IQ2_XXS

| -ub | PP@1024 | PP@2048 | PP@4096 |
|-----|---------|---------|---------|
| 128 | 968 | **924** | 733 |
| 256 | 969 | 897 | 717 |
| 512 (default) | 857 | 837 | 698 |

→ ub=128 ≈ ub=256, both beat default for 14B.

### Qwen3.6-35B-A3B IQ2_XXS

| -ub | PP@1024 | PP@2048 | PP@4096 |
|-----|---------|---------|---------|
| 128 | — | — | 599 (regression!) |
| 256 | 1039 | 1047 | 990 |
| 512 (default) | 1246 | **1324** | **1273** |

→ 35B-A3B: **default 512 ist optimal**. ub=128 = -55% regression at PP@4k.

## Why model-dependent?

- **3B/14B (D=128, dense)**: kleinere head-dim, weniger SM-occupancy headroom.
  Kleinere ubatches = weniger lange tile-waves → bessere SM-utilization
  bei pipeline-parallel multi-GPU
- **35B-A3B (D=256, MoE)**: größere expert-matrix-multiplies → bigger
  ubatches = weniger expert-routing overhead pro token, besser amortized

## Cumulative gains (Ministral-3-3B PP@2k)

- Pre-Phase-5 single-GPU baseline: ~160 t/s
- Phase 5 single-GPU: 1316 t/s (8.2×)
- Phase 5 dual-GPU default ub: 1851 t/s (11.6×)
- Phase 5 dual-GPU + ub=128: **2168 t/s (13.6×)**

## Deploy recommendations

```bash
# Ministral-3-3B / -14B (dense D=128)
./llama-server -m model.gguf -ctk ktq2_1 -ctv vtq2_1 -fa -ngl 99 -ub 128 -b 2048

# Qwen3.6-35B-A3B (MoE D=256)
./llama-server -m model.gguf -ctk ktq2_1 -ctv vtq2_1 -fa -ngl 99 -ub 512 -b 2048
```

Vergleich gegen vorherige deploy-skripte: check ob `-ub` explizit gesetzt war
oder llama-server default genutzt wurde.
