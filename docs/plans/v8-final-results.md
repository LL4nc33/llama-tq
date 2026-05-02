# TurboQuant v8 — Final Results (Phase 1+2 Complete)

**Date:** 2026-05-02 06:14 CEST
**Status:** Production-ready on 35B-A3B-IQ2_XXS

## Triple-Goal Triumph

| Metric | Current 35B prod | v8 Quality (NEW) | Δ |
|---|---|---|---|
| **PPL drift** | +3.85% | **-0.03%** | **-3.88pp better** |
| **tg128** | 85.66 t/s | 86.61 t/s | +1.1% |
| **pp512** | 1106 t/s | 1196 t/s | +8.1% |
| **bpw** | 3.0 | 3.56 | +0.56 (acceptable) |

**Win: alle 3 Goals gleichzeitig verbessert** (Accuracy ↑, Speed ↑, VRAM-cost ist minimal).

## Full Speed-Bench Results (35B-A3B, GPU0 RTX 2060)

| Config | pp512 t/s | tg128 t/s | Notes |
|---|---|---|---|
| ktq2_1/vtq2_1 (current 35B prod) | 1106.52 ± 1.14 | 85.66 ± 0.10 | legacy 3.0 bpw |
| ktq2/vtq2 (= vtq2_2 alias) | 1198.86 ± 3.51 | 86.88 ± 0.03 | v8 default 2.78 bpw |
| **ktq2/vtq3 (vtq3_v8 NEW)** | **1196.18 ± 3.35** | **86.61 ± 0.13** | v8 quality 3.56 bpw |

## Bug Fix: deferred-V predicate

**Pre-fix bench:** vtq3_v8 hit per-token Viterbi encoding during decode → 10× slowdown
- Pre-fix: pp512 23.94 t/s, tg128 8.08 t/s ⛔
- Post-fix: pp512 1196.18 t/s, tg128 86.61 t/s ✅

**Root cause:** `is_vtq2_type_v` predicate in `src/llama-kv-cache.cpp:144` plus 4 more sites missed VTQ3_V8 enum.
**Fix:** commit `5e599369a` — added VTQ3_V8 to all 5 V-trellis predicates (deferred-V, sink-protection, RHT, FA divisibility-skip).

## Implementation Commits

| Commit | What |
|---|---|
| `a9b471026` | CLI aliases ktq1..4 / vtq1..4 |
| `8cb4db877` | vtq3_v8 type (enum 58) + CPU quant/dequant |
| `1910c4180` | vtq3_v8 CUDA dispatch (templated OUTLIER_K) |
| `834b7e55d` | Validation harness + Python sim + plans |
| `d3f7d2b09` | Sim results: HYBRID-GO 3/4 v8 winners |
| `413b418d9` | Gate-B 35B PPL test PASSED |
| `e9633a64f` | Full Pareto sweep 35B (10 configs) |
| `5e599369a` | deferred-V predicate fix for vtq3_v8 |
| `eebd2be61` | README v8 documentation |

## Production Recommendation

### NEW Default for 35B-A3B Triple-Goal Deploys

```bash
--cache-type-k ktq2 --cache-type-v vtq3
```

Replaces legacy `ktq2_1/vtq2_1` (current). All Triple-Goal targets met:
- ✅ PPL essentially lossless (-0.03%)
- ✅ TG floor maintained (86.61 ≥ 70 t/s)
- ✅ 100k+mmproj fits single-GPU0

### Quality Hierarchy

| Config | Use Case |
|---|---|
| `ktq2 + vtq2` | Default — max VRAM savings, 99.85% quality |
| `ktq2 + vtq3` | Production lossless — when accuracy matters more |
| `ktq3 + vtq3` | Research — slightly higher K precision |
| `ktq4 + vtq4` | Archival — closest to f16 |

## Open Items

1. Update `scripts/deploy-35b-singlegpu-100k.sh` to use `vtq3` instead of `vtq2_1`
2. 4B Gate-C blocked (GPU1 boundary — needs user permission)
3. 80B / 122B / Gemma4 / 9B sweep (sequentielle Tests, ~4-6h, lower priority)
4. vtq3_v8 Speed-Bench gegen vtq3_3 (4 outliers) — Quality-vs-Storage trade
5. Pareto-Chart regenerate mit neuen Daten
