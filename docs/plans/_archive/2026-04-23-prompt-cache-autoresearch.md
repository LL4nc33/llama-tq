# Prompt-Cache Autoresearch Sweep (MoE Reuse Problem)

**Date:** 2026-04-23
**Branch:** turboquant
**Status:** Design

## Problem Statement

Anthropic-kompatibles Prompt-Caching ist deployed (Phase 2, commits c7eaaced9 + 63c0880b1 + 2fd8f96a0 + bb5e744c2). Response-fields korrekt (cache_creation_input_tokens, cache_read_input_tokens, ephemeral_5m/1h). Auf **Qwen3-MoE** (Qwen3.6-35B-A3B prod) refused der slot das restored KV und macht trotzdem full re-prefill:

```
forcing full prompt re-processing due to lack of cache data (hybrid/recurrent memory)
```

**Consequence:** Wire-level caching works (network bytes saved, response shape correct), aber kein real-world prefill-speedup auf MoE. Auf dense models: full speedup.

## Goals

1. **Primärziel:** 50% prefill-time-reduction bei R2-hit (same system prefix ≥4k tokens) auf Qwen3.6-35B-A3B
2. **Sekundär:** Keine PPL-regression, kein correctness-bug, kein memory-leak
3. **Nebenbedingung:** keep asymmetric KTQ2_1+VTQ2_1, keep expert-offload, keep 200k ctx

## Baseline (pre-research)

| Config | R1 prefill (8k sys prompt) | R2 prefill (same prompt, hit) | Speedup |
|--------|----------------------------|-------------------------------|---------|
| Qwen3.6-35B-A3B + KTQ2_1+VTQ2_1 | TBD | TBD (currently ≈ R1) | ≈1× |

Messen mit `scripts/measure-prompt-cache-speedup.sh` (TODO — 8k system, 20-token user turn, R1+R2 back-to-back, timings.prompt_ms).

## Hypotheses

**H1 — save-after-generation taints KV:** Aktuelle save passiert nach full generation. Die KV-states in der slot enthalten dann generation-tokens, nicht nur prefix. Beim restore sind diese states unpassend. → Fix: save-after-prefill (direkt nach prompt processing, vor Generation).

**H2 — MoE expert-state nicht gecacht:** Expert routing entscheidet pro-token welche experts aktiv sind. Gecachter KV hat "historical" expert-activations; restored state passt nicht zu aktueller expert-routing-decision. → Fix: möglicherweise unfixbar ohne expert-state-cache.

**H3 — hybrid/recurrent-flag falsch gesetzt:** Qwen3-MoE wird als hybrid klassifiziert obwohl es reines MoE ist (kein Mamba/SSM). Der refusal-check ist zu konservativ. → Fix: flag-bedingung lockern für pure-MoE.

**H4 — slot-position-marker mismatch:** restore läuft durch, aber slot's internal n_past ist nicht synchron mit dem restored KV. → Fix: explicit slot.n_past = restored_n_tokens nach restore.

**H5 — cache-reuse 256 conflict:** --cache-reuse 256 hat eigene prefix-match-logic. Conflict mit anthropic-cache restore. → Fix: disable cache-reuse wenn anthropic-cache hit.

## Experiments

### E1 — Root-Cause Identification (C-Phase, blockiert alles andere)
Agent identifiziert refusal-condition in src/llama-kv-cache.cpp oder llama-memory*.
**Output:** file:line + 3 fix-options. **Status:** LAUFEND (agent `a70e3181843984fa8`).

### E2 — Save-after-prefill Implementation
Hook von `post_anthropic_messages` verschieben: save NACH prefill, VOR slot.decode-loop für generation. Re-test R2 hit rate.
**Gate:** R2 prefill < 0.5× R1 prefill.

### E3 — MoE flag-check bypass
Falls H3 stimmt: patch hybrid-check in llama-memory, compile, test.
**Gate:** No PPL regression, R2 hit speedup.

### E4 — Explicit n_past sync after restore
Im `handle_slots_restore` oder `post_anthropic_messages`: set slot.n_past = cache_manager.restored_tokens. Re-test.
**Gate:** Log shows "reusing cached N tokens" statt "forcing full".

### E5 — --cache-reuse interaction
Ablation: prod ohne --cache-reuse 256, nur anthropic-cache. Messe R2 speedup.
**Gate:** Identifies interaction.

### E6 — Dense model control
Same test auf Qwen3.5-0.6B (dense, non-MoE) für sanity-check ob Fix nicht kaputtmacht was schon geht.
**Gate:** Dense R2 speedup unchanged.

### E7 — Expert-state snapshot (H2 only, if unavoidable)
Falls E2-E5 fehlschlagen: recherchiere ob llama.cpp MoE-expert-routing-state serialisierbar ist. Wahrscheinlich HIGH effort, möglicherweise blocked by upstream.
**Gate:** Nur wenn H1/H3/H4/H5 alle fehlschlagen.

## Metrics

- `timings.prompt_ms` pre/post (primary)
- `timings.predicted_per_token_ms` (must-not-regress)
- `cache_read_input_tokens` in response (must be > 0 on R2)
- Server log pattern `"forcing full prompt re-processing"` occurrences (target: 0 on R2)
- wikitext-2 PPL 64-chunk delta (must be within noise, ±1%)
- RAM/VRAM delta (must not explode — cache index mutex-protected, LRU quota)

## Sequence

1. **E1** (Agent running) → Root-Cause report
2. **Based on E1:** pick E2 OR E3 OR E4 OR E5 first
3. Implement, deploy on gpu00, run measure-prompt-cache-speedup.sh
4. If ≥50% speedup: done
5. If <50%: next experiment
6. E6 (dense control) after first success
7. E7 only if all else fails

## Out of Scope

- Per-breakpoint individual key tracking (Phase 3 follow-up)
- LRU quota enforcement (Phase 3)
- SHA256 upgrade (Phase 3)
- Streaming response-shape patch for cache_creation fields (known gap)
- Cross-slot cache sharing (requires architecture change)

## Dependencies

- Agent E1 report (running)
- gpu00:8791 prod access (stable, 38 tok/s baseline)
- distillery-claude NOT training (GPU0 idle confirmed)

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fix breaks dense-model caching | HIGH | E6 control test before merge |
| PPL regression silent | MEDIUM | Mandatory 64-chunk wikitext PPL run per experiment |
| Expert-state truly uncacheable (H2) | MEDIUM | E7 scope explicit; accept response-shape-only caching if unavoidable |
| Upstream llama.cpp API change | LOW | Pin to turboquant branch |

## Success Criteria

- Qwen3.6-35B-A3B: R2 prefill ≤ 50% of R1 prefill on 8k system prompt
- Qwen3.5-0.6B dense: R2 prefill ≤ 30% of R1 prefill (should be stronger)
- wikitext-2 PPL delta within noise
- No server crash across 100-request sweep
- Documented in RUN7_NOTES.md + README Phase4 section update
