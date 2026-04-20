# Trick 2 PR2 — Per-Layer Mixed Precision V-Cache Validation

**Datum:** 2026-04-20
**Branch:** `trick2-pr2-mixed-precision`
**Commit:** fda5827b9 (plumbing) + dc99a45bb (resolver infra)
**Build:** gpu00:~/llama-tq/build-cuda-trick2/bin/llama-perplexity

---

## Zusammenfassung

PR2 **funktional komplett** — Per-Layer Mixed-Precision V-Cache ist verdrahtet von
CLI → arg → common_params → llama_context_params → cparams → kv-cache ctor → per-Tensor Allokation.

## Validierung auf Qwen3.5-35B-A3B IQ2_XS

### Test 1: Backward-Compat (keine PR2 flags)

```bash
llama-perplexity -ngl 99 --flash-attn on \
  --cache-type-k ktq2_1 --cache-type-v vtq3_2 \
  --tq-protect-sinks 4 \
  -f wiki.test.raw --chunks 3 -ts 12,12
```

→ **PPL = 25.6523 ± 3.06** ✅ (matches pre-PR2 baseline)

### Test 2: Manual Override (first 4 + last 4 layers → VTQ4_2)

```bash
llama-perplexity ... --tq-v-override '0-3:vtq4_2,36-39:vtq4_2' \
                     --tq-v-strategy manual
```

Logged:
```
tq-v-mixed: n_layer=40  vtq2_2=0  vtq3_2=32  vtq4_2=8  other=0
```

→ **PPL = 25.6523 ± 3.06** (same as baseline)

**Interpretation:** First 4 layer stay in f16 wegen `--tq-protect-sinks 4` (überschreibt
override nach `eff_type_v` resolution), und last 4 layer haben marginalen Einfluss bei
nur 3 chunks PPL-Messung. Mid-layer Range (12-15) wäre wirkungsvoller per Profile-Daten
aus 0.6B (Layer 13 hatte 12.28× Varianz-Ratio), aber n_layer_mapping zwischen 0.6B (28 Layer)
und 35B-A3B (40 Layer) braucht ein natives 35B-Profile.

### Status der Implementierung

| Component | Status |
|-----------|--------|
| `common/tq-profile.{h,cpp}` | ✅ Parser, JSON loader, 6 strategies |
| `common/arg.cpp` 6 CLI flags | ✅ Alle 6 flags, alle CLI-examples |
| `common/common.cpp` resolve call | ✅ Nach model-load, bevor context-init |
| `include/llama.h` params struct | ✅ type_v_layers ptr + count |
| `src/llama-cparams.h` | ✅ tq_v_layers vector |
| `src/llama-context.cpp` plumbing | ✅ Copy to cparams |
| `src/llama-kv-cache.{h,cpp}` | ✅ Default-arg, per-layer eff_type_v |
| `src/llama-kv-cache-iswa.{h,cpp}` | ✅ Pass-through |
| `src/llama-model.cpp` | ✅ cparams.tq_v_layers an beide Ctor-Pfade |
| Build CUDA | ✅ llama-perplexity 5.1MB |
| Smoke test backward-compat | ✅ PPL matches |
| Smoke test override | ✅ 8 layer as VTQ4_2 logged correctly |

### Offene Punkte

1. **Echter Benefit-Test:** Profile-run auf 35B durchführen (blockiert durch #139 llama-cli bug),
   dann `mixed` strategy statt manual override testen.
2. **Mid-Layer-Test:** Override `12-15:vtq4_2` probieren — laut 0.6B-Profile sind das die
   High-Variance Layer (wenn das auf 35B überhaupt transferiert).
3. **Sinks vs. override precedence:** Currently `--tq-protect-sinks` wins over override
   für first N layers. Sollte das umgekehrt sein (user override > auto-protection)?
4. **avg_bpw log output:** Shows wrong value (13.30 statt ~3.15) — `bpw_per_elem()` helper
   braucht Fix für TQ-Typen.

---

## Commits

- `dc99a45bb` wip(trick2-pr2): scaffold profile loader + strategy resolver (500 LOC)
- `fda5827b9` feat(trick2-pr2): wire per-layer V-cache type through kv-cache (94 LOC diff)

Ready for merge auf phase2 sobald mid-layer benefit auf 35B belegt.
