# Qwen3.6-35B-A3B (the model-34B) — Optimal Configuration

## TL;DR

**ub=1024 statt default ub=512 bringt +13% PP@2k, +13% PP@4k.**
Kombiniert mit dual-GPU layer-split: **+50-71% PP** über single-GPU baseline.

## Sweep results (build `6cc4920d0`, IQ2_XXS, dual-GPU layer-split)

### ubatch sweep

| ub | PP@2k | PP@4k | PP@8k | TG |
|----|-------|-------|-------|-----|
| 128 | — | 599 | — | 76 (-55% regression!) |
| 256 | 1047 | 990 | — | 75 |
| 512 (default) | 1342 | 1285 | 1097 | 77 |
| 768 | 1416 | 1353 | 1123 | 77 |
| **1024 (optimum)** | **1443** | **1437** | **1227** | 76 |
| 1536 | 1414 | 1305 | 1040 | 77 |
| 2048 | 1270 | 1310 | 1145 | 77 |

→ **ub=1024 ist 35B-A3B sweet-spot.** ub=512→1024 = +13% PP@2k.

### split-mode test

| sm | PP@4k | TG@32 |
|-----|-------|-------|
| layer (default) | **1437** | 77 |
| row | 358 (-75%) | 32 (-58%) |

→ **sm=row ist CATASTROPHIC für 35B.** Default layer-split bleibt.

## Full sweep with optimal config (ub=1024, layer-split)

| ctx | t/s |
|------|------|
| PP@512  | 1059 |
| PP@1024 | 1265 |
| **PP@2048** | **1443** |
| **PP@4096** | **1437** |
| PP@8192 | 1227 |
| PP@16384 | 834 |
| TG@64 | 76 |

## Total wins vs single-GPU baseline

| ctx | single-GPU ub=512 | dual-GPU ub=1024 | Δ |
|------|---|---|---|
| PP@1024 | 1027 | 1265 | +23% |
| PP@2048 | 962 | **1443** | **+50%** |
| PP@4096 | 839 | **1437** | **+71%** |
| PP@8192 | — | 1227 | (single likely OOM with full ctx) |
| TG@64 | 81 | 76 | -6% (inter-GPU comm cost) |

## Why does 35B-A3B prefer larger ubatch (vs Ministral)?

MoE architecture: expert-routing has per-ubatch overhead (decide which 4-of-N experts
per token). Larger ubatch = more tokens share the routing decision = better
amortization. Dense 3B/14B has no such routing cost → smaller ubatches win
through pipeline parallelism efficiency.

**Rule of thumb (for our hardware/quant config):**
- Dense small (3B-14B, D=128): ub=128 optimal
- MoE (35B-A3B, D=256): ub=1024 optimal

## Production deploy notes

The existing `scripts/deploy-the-model-dualgpu-200k.sh` uses `-ub 512` and
`ktq2`/`vtq4` (different from our tested `ktq2_1`/`vtq2_1` Phase 5 path).
Considerations for an updated deploy:

- Switch to `ktq2_1`/`vtq2_1` to enable Phase 5 MMA inline (would need testing
  for D=256 fallback path — Phase 5 inline only for D=128 GQA=4)
- Set `-ub 1024` instead of 512 for +13% PP
- Keep `--moe-pin-experts` and `--backend-sampling` (helpful)
- For 200k ctx the VRAM budget is tight — verify ub=1024 fits

Recommended next step: test current production deploy with `-ub 1024` swap
(no other changes) on staging before full Phase 5 + ub upgrade.
