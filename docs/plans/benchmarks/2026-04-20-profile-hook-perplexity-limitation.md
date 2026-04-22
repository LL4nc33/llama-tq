# Profile Hook (Trick 2 PR1) — Perplexity-Pfad Limitation

**Datum:** 2026-04-20
**Status:** Known Issue — workaround verfügbar

---

## Problem

`--tq-profile-heads N` auf `llama-perplexity` triggert den Hook nicht, obwohl:
- Memory-Type-Unwrap funktioniert (iswa, hybrid, hybrid_iswa alle supported seit Commit b23a11417)
- Kein "unsupported" warning mehr
- Dennoch wird kein JSON geschrieben, auch nicht mit `--tq-profile-heads 3`

Getestet auf:
- Qwen3.5-35B-A3B IQ2_XS (hybrid_iswa) — kein dump
- Qwen3.5-27B Q4_K_M (iswa) — kein dump

## Hypothese

Perplexity nutzt einen batch-eval-Pfad der `llama_decode()` mit einem anderen
`memory` state ruft, bei dem `tq_profile_collect_v()` zwar reinkommt aber
`get_v_tensors_for_profile()` n_kv_layers=0 zurückgibt (z.B. weil V-cache noch
nicht materialisiert). Das setzt `tq_profile_done=true` sofort.

## Workaround

Profile funktioniert auf `llama-cli` mit kurzem Prompt (Qwen3-0.6B validation
vom 2026-04-20-trick2-profile-analysis.md zeigt es funktioniert dort). llama-cli
hat eigene Interactive-Mode bugs (log explosion mit `-no-cnv`), die separate Investigation brauchen.

**Pragmatische Strategie:** Profile mit 0.6B-Daten als Basis nehmen (sind repräsentativ
für allgemeine Attention-Patterns), PR2 damit designen, dann auf größeren Modellen
per manual `--tq-v-override` validieren.

## Offene Investigation

1. `tq_profile_collect_v` debug-log einbauen (LOG_INF vor `get_v_tensors_for_profile`)
   um zu sehen ob der Hook überhaupt reinkommt und was n_kv_layers ist.
2. Check `memory->init_batch()` vs state during perplexity — vielleicht wird memory
   zwischen chunks reseted?
3. Alternative: separater `llama-profile` tool-Pfad der den Hook deterministisch
   per Batch-Eval triggert.

## Related Fixes Applied

- Commit b2a3044b2 — iswa unwrap (fix #1)
- Commit b23a11417 — hybrid + hybrid_iswa unwrap (fix #2)

Beide Fixes bleiben richtig, aber das grundlegende Triggering-Problem im
perplexity-Pfad ist ungelöst.
