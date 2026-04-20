# Phase 2 Kickoff — Polish + Research-Parallelspur

**Datum:** 2026-04-20
**Branch:** `phase2`
**Vorgänger:** `trellis-v2-phase1` (merged → master, Commit ca126829b)
**Production:** gpu00:8791 (Qwen3.5-35B-A3B IQ2_XS, ktq2_1 + vtq3_2 + deferred + sink, parallel=1, 400K ctx, ~67 tok/s)

---

## Ziel

Die bestehende VTQ_2-Implementierung schrittweise robuster und schneller machen
(Phase 2 laut ROADMAP), parallel dazu die Trick-17-Research-Serie vorantreiben.
Kleine Wins, viel messen. Kein Scope-Creep.

---

## Offene Tasks

### Sofort (Polish + Bench)

**#137 — Trick 2 PR1: Profiling-Hook builden + validieren**
- Branch `trick2-pr1-profile-heads` (Commit 71b56f7fb) existiert
- Build auf gpu00 + 100-token Test-Run mit CSV-Output
- Ergebnis → Datenbasis für #121 Design

**#136 — 400K ctx OOM Fix**
- Analyse: `docs/plans/2026-04-20-400k-oom-analysis.md`
- Root-Cause: 4× Input-Replication durch pipeline parallelism (GGML_SCHED_MAX_COPIES=4)
- Fix: Rebuild mit `-DGGML_SCHED_MAX_COPIES=2` → ~400-500 MiB weniger auf CUDA0
- Deploy-Plan: parallel=2, ubatch=256, -ts 10,14, 400K ctx

### Research (Trick-Serie)

**#121 — Trick 2: Per-head precision mixing**
- Design: `docs/plans/2026-04-20-trick2-per-head-precision-design.md` (200 LOC)
- Status: wartet auf #137 Varianz-Daten
- Idee: Hohe-Varianz Heads bekommen mehr bpw (VTQ4_2), niedrige weniger (VTQ2_2)

**#123 — Trick 4: Correction overlay buffer**
- Idee: Top-N Fehler pro Token in f16 als Lossless-Patch
- PPL-Impact vermutlich deutlich, Memory-Overhead linear in N

**#124 — Trick 5: Per-head learned lambda sharpening**
- Braucht Training-Setup (Fine-Tune Recovery)
- Aufwändigster Trick — zurückgestellt bis #121 + #123 durch

### Low-Prio Offen

**#135 — q8_0 K + VTQ_2 V CPU-Fallback Bug**
- Hat keinen Prod-Impact (q8_0 + VTQ_2 wird nicht produktiv genutzt)
- Investigate wenn Zeit

**#127 — TQW2 CUDA Sprint** (Phase 4)
- Weights auf 2-3 bpw via Lloyd-Max + RHT
- Größeres Projekt, wartet auf Phase-2-Abschluss

---

## Working-Mode

1. **PR pro Trick.** Jeder Trick bekommt eigenen Branch `trick{N}-pr{M}-*` und eigene
   Messung (PPL + tg/pp bench) als Gate. Kein Merge ohne grüne Zahlen.
2. **Kontinuierlicher Bench.** Bei jedem Merge auf phase2: production-run
   (gpu00:8791) vs master-Baseline messen, in `docs/plans/benchmarks/` dokumentieren.
3. **Rollback ready.** Falls ein Trick eine Regression bringt → revert, nicht
   patchen. Tricks sind unabhängig.

---

## Branch-Strategie

- `master` — produktionsreif, Basis für Release-Tags
- `phase2` — aktive Entwicklung, Merge-Ziel für Tricks
- `trick{N}-pr{M}-*` — Feature-Branches, Rebase auf phase2 vor Merge
- `trellis-v2-phase1` — behalten als historisches Artefakt

Merge `phase2` → `master` wenn:
- ≥3 Tricks integriert
- PPL stabil oder besser vs Phase-1 Baseline
- Production-tg regression ≤ 2%

---

## Nächste Schritte (laufend)

- [ ] Agent-Run: #137 Build + CSV-Generierung (in progress)
- [ ] Agent-Run: #136 Rebuild mit MAX_COPIES=2 (in progress)
- [ ] Nach #137 Daten: #121 Design refinen mit echten Varianzzahlen
- [ ] Nach #136 Build: Deploy-Test parallel=2 + 400K ctx, tg-Messung
- [ ] README + ROADMAP nach jedem Merge updaten

---

## Referenzen

- [ROADMAP](../ROADMAP.md)
- [Phase-1 Report](2026-04-17-trellis-v2-design.md)
- [OOM Analysis](2026-04-20-400k-oom-analysis.md)
- [Trick 2 Design](2026-04-20-trick2-per-head-precision-design.md)
- [CUDA Stability](2026-04-20-cuda-stability-validation.md)
