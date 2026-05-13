# Master Bench Summary — 2026-05-13

**UPDATE (later 2026-05-13):** dual-GPU layer-split + ub=128 tuning delivers
additional +40-78% PP wins. See [MULTIGPU-WIN doc](./2026-05-13-MULTIGPU-WIN.md)
and [ubatch tuning doc](./2026-05-13-ubatch-tuning.md). Numbers below are
single-GPU-only baseline for reference.

Vollständiger snapshot aller relevanten model-shapes auf gpu00 RTX 2060 mit
`-ctk ktq2_1 -ctv vtq2_1 -fa 1 -ngl 99` (single-GPU0).

Build: `135829149 (17692)` (Phase 5 KTQ+VTQ MMA inline + V_rows OOB fix)

## PP-Sweep + TG

### Ministral-3-3B (D=128, GQA=4, dense — Phase 5 ACTIVE)

| ctx | Q4_K_M | UD-IQ2_XXS |
|------|---|---|
| PP@512  | **1977 t/s** | 1983 t/s |
| PP@1024 | 1699 t/s | 1685 t/s |
| **PP@2048** | **1316 t/s** | 1277 t/s |
| PP@4096 | 901 t/s | 859 t/s |
| PP@8192 | 546 t/s | 518 t/s |
| PP@10240 | 444 t/s | 432 t/s |
| TG@128 | 97 t/s | 98 t/s |

### Ministral-3-14B IQ2_XXS (D=128, GQA=4, dense — Phase 5 ACTIVE)

| ctx | t/s |
|------|------|
| PP@512  | 736 |
| PP@1024 | 668 |
| PP@2048 | 555 |
| PP@4096 | 412 |
| TG@128 | 31 |

### Qwen3.6-35B-A3B IQ2_XXS (D=256, GQA=8, MoE — Phase 5 N/A, split-dequant fallback)

| ctx | t/s |
|------|------|
| PP@512  | **1065** |
| PP@1024 | 1027 |
| PP@2048 | 959 |
| PP@4096 | 837 |
| TG@64 | 81.5 |

## PPL Quality (wikitext-2)

### Ministral-3-3B Q4_K_M (100 chunks)

| KV config | PPL | 1σ |
|-----------|-----|----|
| f16/f16 | 9.087 | ±0.141 |
| ktq2_1/vtq2_1 | **9.628** | ±0.149 (Δ +5.95%) |

### Ministral-3-14B IQ2_XXS (50 chunks)

| KV config | PPL | 1σ |
|-----------|-----|----|
| f16/f16 | 8.054 | ±0.175 |
| ktq2_1/vtq2_1 | **8.374** | ±0.180 (Δ +4.0%) |

→ Confirms 14B has more KV-quant capacity headroom than 3B (4.0% vs 5.95%).

## Observations

### Phase 5 wins (D=128 models)
- 3B sweet-spot: PP@2k = 1316 t/s, → 65% of theoretical FP16 tensor peak (13 TFLOPS / 6.86 GFLOPS-per-token)
- 14B PP-scaling: same kernel, 4.4× more params → 2.4× slower (good, not linear bc batched MMA)
- TG@128 = 97 t/s (3B) / 31 t/s (14B) — vec-decode pfad, kein direkter Phase 5 impact

### Qwen3.6-35B-A3B (split-dequant fallback)
- PP@1k = 1027 t/s, deutlich besser als noch vor Phase 5 baseline (~600 t/s vor MMA inline)
- 35B-A3B mit nur 3B aktiv pro forward = MoE pattern, scaling besser als dense 14B
- Phase 6 D=256 wäre nächster schritt für weitere wins (vorher gescheitert an NVCC segfault)

### Quality bei 3B
- 5.95% PPL-tax durch ktq2_1+vtq2_1 — höher als 35B (3.85%) wegen kleinerem D=128
- Akzeptabel für long-ctx-deployment, suboptimal für quality-critical short-ctx tasks

## Roofline Analysis (Ministral-3-3B @ PP=2k)

- Achieved: 1316 t/s × 6.86 GFLOPS/token = **9.03 TFLOPS**
- RTX 2060 FP16 tensor peak: **13 TFLOPS**
- **Utilization: 69.5%** — gut, aber ~30% headroom

### Headroom-quellen (vermutet, ohne profiling)
1. V-tile load synchron in Phase 5 (line 991-993 fattn-mma-ktq-inline.cuh) — kein cp.async
2. Inter-block scheduling bei ncols1=8 → 32 wave-tiles für 2k tokens, occupancy=2 → 60 concurrent blocks
3. K-tile FWHT warp-coop blocks tensor-core pipeline

Realistic ceiling: ~1700-1800 t/s @ 2k (90% peak). Würde dedizierter D=128
VTQ cp.async pipeline brauchen.

## Was nicht angegangen wurde

- D=256 MMA inline (Phase 6) — letzter versuch crashed NVCC, blieb reverted
- Async V-load — größerer kernel-refactor, risk-reward unklar ohne profiling

## FINAL OPTIMAL CONFIGURATION (after iteration)

Ministral-3-3B Q4_K_M, dual-GPU + ub=128 + Phase 5 (build `f8c68433b`):

| ctx | t/s | Δ vs single-GPU Phase 5 | Δ vs pre-Phase-5 |
|------|------|---|---|
| PP@1024 | **2637** | +55% | n/a |
| PP@2048 | **2194** | +67% | ~14× |
| PP@4096 | **1583** | +76% | n/a |
| PP@8192 | **972** | +78% | n/a |
| PP@10240 | **787** | +77% | ~3.7× |
| TG@128 | 97 | flat | flat |

Stack: KTQ2_1 K + VTQ2_1 V (MMA inline) × dual-GPU layer-split × ub=128.

See `scripts/deploy-ministral-3b-dualgpu-optimal.sh` for deployment template.
