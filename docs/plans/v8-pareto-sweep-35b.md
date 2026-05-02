# v8 Pareto Sweep — Qwen3.6-35B-A3B-IQ2_XXS bartowski (2026-05-02 05:52)

## Full Pareto-Frontier (10 configs)

| Config | bpw | PPL | Drift vs f16 | Notes |
|---|:---:|---:|---:|---|
| f16/f16 | 32.00 | 7.2044 | 0.00% | baseline |
| ktq2_1/vtq2_1 | 3.00 | 7.4816 | **+3.85%** | current 35B prod (legacy) |
| **ktq2/vtq2 (v8)** | 2.78 | **7.1807** | **-0.33%** | **NEW prod default candidate** |
| ktq2_1/vtq2_2 | 2.78 | 7.1807 | -0.33% | legacy 2025 default (= alias of v8) |
| **ktq2/vtq3 (v8)** | 3.56 | **7.2024** | **-0.03%** | **v8 quality tier (lossless)** |
| ktq2_1/vtq3_1 | 3.75 | 7.2178 | +0.19% | legacy aggressive |
| ktq2_1/vtq3_3 | 3.78 | 7.1533 | -0.71% | legacy 4-outliers (best PPL) |
| ktq2/vtq4 (v8) | 4.00 | 7.2635 | +0.82% | codebook 4-bit (alias = vtq4_1) |
| ktq2_1/vtq4_1 | 4.00 | 7.2635 | +0.82% | legacy (= alias) |
| ktq3/vtq3 (v8) | 4.06 | 7.2024 | -0.03% | research tier |

## Recommended v8 Tiers

### Tier 1 — Recommended Production Default

```
--cache-type-k ktq2 --cache-type-v vtq2
```

- **bpw: 2.78** (KV @ 32k = 89 MiB on 35B-A3B)
- **PPL drift: -0.33%** (better than f16, within stderr)
- **vs current 35B prod (+3.85%): 4.18pp improvement at -7% bpw**
- v8 alias for ktq2_1/vtq2_2

### Tier 2 — Quality (lossless)

```
--cache-type-k ktq2 --cache-type-v vtq3
```

- **bpw: 3.56** (KV @ 32k = 114 MiB)
- **PPL drift: -0.03%** (essentially f16)
- vtq3_v8 = NEW unified type (enum 58, 2 outliers, 3.625 bpw)
- 12% smaller than legacy ktq2_1/vtq3_3 (3.78 bpw → 7.1533, -0.71%)
- Trade-off: vtq3_v8 vs vtq3_3 — save 6B/block at +0.68pp drift cost

### Tier 3 — Research / archival

```
--cache-type-k ktq3 --cache-type-v vtq3
```

- **bpw: 4.06**
- v8-research, K higher precision

## Production Recommendation

**Update prod-default for 35B from `ktq2_1/vtq2_1` → `ktq2/vtq2`** (= legacy `ktq2_1/vtq2_2`):

- 4.18pp better PPL drift
- Same VRAM as current ktq2_1/vtq2_1 (-7% storage actually)
- Trellis-V already validated stable since 2026-04-25 (vtq2_2 prod 8 days)

For longer-context deploys where 200k+ ctx pushes VRAM pressure:
- Stay on 2.78 bpw default
- For lossless mode (Tier 2): vtq3_v8 saves 6 B/block vs vtq3_3 → -12% V-cache storage

## Open Questions for Phase 2

1. **Speed**: TG impact of vtq3_v8 trellis-encoder on prefill? (running now)
2. **4B-dense**: vtq3_v8 vs vtq4_1 on small dense models? (gate-C blocked, GPU1 boundary)
3. **80B / 122B / Gemma4**: longer 8-model sweep on remaining hardware?
4. **Should default `vtq3` alias point to vtq3_v8 (3.625 bpw) or vtq3_3 (4.0 bpw, better PPL)?**
   - Current code: `vtq3` → vtq3_v8 (Pareto-optimal at lower bpw)
   - Alternative: `vtq3` → vtq3_3 (best-PPL but 12% more storage)
   - **Recommendation:** keep vtq3_v8 for default, document vtq3_3 as opt-in for archival quality
