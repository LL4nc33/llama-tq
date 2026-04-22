# llama-tq Roadmap

Letzte Aktualisierung: 2026-04-20
Maintainer: LL4nc33 (Lance / Maverick)

## Status: Phase 1 abgeschlossen ✅

Die `trellis-v2-phase1` Arbeit ist auf `master` gemerged und
production-validiert. Das VTQ_2 Trellis-coded V-cache System ist
fertig und deployable.

### Production Numbers (gpu00:8791, Qwen3.5-35B-A3B IQ2_XS)

| config | tg tok/s | speedup vs alt |
|--------|----------|----------------|
| alt (f16 V-cache) | ~8-10 | baseline |
| neu (vtq3_2 + ktq2_1 + deferred + sink, parallel=2) | **~66.7** | **7×** |

Messung: 250-token generation, deutsche Prompts, 2026-04-20.

### Was Phase 1 geliefert hat

**Typen:** VTQ2_2 / VTQ3_2 / VTQ4_2 (2.06 / 3.06 / 4.06 bpw V-cache,
bit-exact Trellis-Coded-Quantization mit Viterbi-Encoder und
Shift-Register-Decoder)

**CUDA-Pfad:** Dequant-Kernel, Viterbi-Encoder (~57μs/call),
Flash-Attention-Dispatch, `convert.cu` + `set-rows.cu` Integration

**Runtime-Features:**
- `--tq-deferred-v` — f16-Staging-Buffer, bulk Viterbi am prefill→decode-Übergang
- `--tq-protect-sinks N` — StreamingLLM-inspirierter Schutz des ersten KV-Layers
- `--tq-protect-layers N` — Boundary-Protection (first/last N layers q8_0)
- CLI-Flags in `server`, `cli`, `perplexity`, `bench`, `mtmd`

**Validierung:**
- Qwen3.5-0.8B tg512: 7 → 196 t/s (28×)
- Qwen3.5-27B dual-GPU tg1024: 14.62 vs f16 14.89 (-1.8%)
- Qwen3.5-35B-A3B live: byte-identischer deterministic output, ~6× schneller
- PPL: vtq3_2 +1.9%, vtq2_2 +8.0%, vtq4_2 +0.6% (wikitext-2)
- Zero crashes in 6h+ akkumulierten Runs

**Production Recipe:**
```
--cache-type-k ktq2_1 --cache-type-v vtq3_2 \
--tq-deferred-k --tq-deferred-v --tq-protect-sinks 4
```

---

## Benchmark-Spur (kontinuierlich, parallel zu allen Phasen)

Solide Baselines sind Voraussetzung für jede Optimierung. Dauerhafte
Messreihe in `docs/plans/benchmarks/` (pro Messung YYYY-MM-DD-topic.md).

**Regelmäßig:**
- Production-Vergleich: master-build vs ältere Commits (Regression-Check)
- Context-Scaling: tg @ 4k / 16k / 64k / 200k Ctx-Länge
- Model-Scaling: 0.8B / 2B / 27B / 35B mit fixem V-Recipe
- Competitor-Comparison: Q4_K_M / IQ4_XS / Q8_0 V vs VTQ_2

**Ad-hoc bei jedem Trick:**
- PPL wikitext-2 10/40 chunks
- tg/pp bench mit gleicher Hardware-Config
- Speicherverbrauch

---

## Phase 2 — Aktuelle Version verbessern (sofort)

Ziel: die bestehende VTQ_2-Implementierung schrittweise robuster und
schneller machen, ohne den Scope zu erweitern. Kleine Wins, viel
Messen.

**Offene Arbeitspunkte aus Phase 1:**
- ~~35B Production-Deploy auf `gpu00:8791`~~ ✅ DONE 2026-04-20
  (ctx=200K statt geplanten 400K wegen compute-buffer OOM bei parallel=2)
- PPL-Prefill im `--tq-deferred-v` Modus echt quantisiert messbar
  machen (aktuell bleibt State in STAGING bei pure prefill)
- 27B pp1024 um -3% unter f16 — Bulk-Viterbi am Übergang optimieren
- 400K ctx auf 2x 12GB GPU: entweder parallel=1 oder kleinere ubatch
- CUDA-Kernel-Review: noch überall `__syncthreads()` optimal?
- Fehlerzustände: was passiert bei OOM, invalidem `-ngl`, kaputten
  GGUFs — graceful failure messages statt crash

**Quality-of-life:**
- `--help` Output für VTQ-Flags aufräumen
- Defaults überprüfen (sollten die TQ-Flags ein default-on-Profil haben?)
- Fehler-Logging bei failed dequant (aktuell silent fallback auf f16)

**Zeithorizont:** 1-3 Wochen. Kein Research, nur polish.

---

## Phase 3 — Trick-17-Serie (Research-Parallelspur)

Ziel: Qualität/bpw-Ratio verbessern durch klügere Quantisierungs-
Algorithmen. Jeder Trick ist unabhängig, hat eigenes Mess-Gate.

Siehe `tests/trellis-phase1/BACKLOG.md` für Details. Hard-limit:
Es werden nie mehr als 17. Neue Ideen ersetzen alte.

**Done:**
- Trick 1 — Attention-sink protection (Layer-level)
- Trick 3 — Per-model RHT seed calibration

**Nächste:**
- Trick 2 — Per-head precision mixing (hohe Varianz → höhere bpw)
- Trick 4 — Correction overlay buffer (lossless top-N patch)
- Trick 5 — Per-head learned lambda sharpening (braucht Training)

**Später (6-16):** siehe BACKLOG — FWHT per token, deferred K hybrid
precision, learned RHT matrix, block-variable bpw, adaptive Lloyd-Max.

**Trick 17** — "The Big One". Reserviert. Wenn er kommt, ist das
Paper geschrieben.

**Zeithorizont:** Parallel zu Phase 2, pro Trick 1-2 Wochen.

---

## Phase 4 — TQW2 Weight Quantization (großer Hebel)

VTQ war nur KV-cache. Modell-Weights sind weiterhin IQ2_XXS / Q8_0.
**Weights stellen den Großteil des VRAM** — TQW2 würde Weights selbst
auf 2-3 bpw bringen mit Lloyd-Max-Qualität.

**Status:**
- Python-Validierung: RHT + Lloyd-Max vs IQ2_XXS MSE — DONE (Task #126)
- CUDA-Sprint: offen (Task #127, in_progress)

**Offene Fragen:**
- Separate Type-enums `TQW{1,2,3,4}_1` oder Reuse von KTQ*?
- Integration in `llama.cpp` Convert-Pipeline (gguf-py)
- Interaktion mit existierenden Quant-Typen

**Zeithorizont:** 1-3 Monate nach Phase-2-Abschluss. Größeres Projekt
als die Trick-Serie.

---

## Phase 5 — Community / Paper / Hardware

**Upstream-PR an `ggml-org/llama.cpp`:**
Sauber aufgeteilt in digestible PRs (type-enums → CPU-path → CUDA-path),
mit Paper-ähnlicher Dokumentation. Optional — nur falls relevante
Community-Nachfrage.

**Paper:**
Sobald Trick 17 benannt und validiert ist: Draft für ICLR 2027
oder ähnlich. Konkurrenz-Benchmarks: KVQuant, Aquila, QuaRot.

**Hardware-Support:**
- RTX 40-series tuning (Ada architecture)
- AMD ROCm-Path (falls Community-Interesse)
- Apple Silicon MPS (Metal-Shader-Äquivalente zum Viterbi-Encoder)

**Zeithorizont:** 3-12 Monate, abhängig von Paper-Timing.

---

## Infrastructure / Ops

### Repositories
- `LL4nc33/llama-tq` — aktiv gepflegter Fork mit VTQ_2
- `ggml-org/llama.cpp` — upstream (periodisch rebase/merge)
- `LL4nc33/oidanice-llama` — whitelabel AI platform (nutzt llama-tq als backend)

### Production-Server
- `gpu00:8791` — Qwen3.5-35B-A3B, VTQ_2 (ab 2026-04-20)
- `gpu00:8792` — FunctionGemma 270M (tool router)

### Testing
- Lokaler CPU-Roundtrip in `tests/trellis-phase1/`
- PPL sweep auf gpu00 (wikitext-2)
- Stability-Runs: bench tg1024, long generation

### LEGION
Shared message board mit `oidanice-distillery` für Training/Deployment
coordination. Lokal only, nie remote.

---

## Decision Log (lose Sammlung)

**Warum 17 Tricks?** Deutsches Idiom für "die geniale, scheinbar banale
Lösung die das Problem wegzaubert". Hard-limit gegen Featurism.

**Warum Trellis statt Codebook?** Paper-Validierung: Trellis schlägt
Lloyd-Max-Codebücher bei gleichem bpw um ~0.3-0.5% PPL. Kosten:
komplexerer Encoder (Viterbi DP) vs LUT.

**Warum deferred_v?** Per-token Viterbi auf kurzen ubatches = 21.7ms
GPU-call-overhead dominiert. Bulk-quantize am prefill→decode-Übergang
= eine einzige Viterbi-Instanz, dann lese-optimierter decoder.
26× tg Speedup, keine Quality-Änderung.

**Warum nicht k_cache-Protect-Sinks?** K-cache wird symmetrisch per
Token geschrieben und verhält sich anders unter Quantisierung.
Erste Messungen zeigen keine sink-Dominanz → nicht gemacht.

---

## Referenzen
- [docs/plans/2026-04-20-cuda-stability-validation.md](plans/2026-04-20-cuda-stability-validation.md)
- [docs/plans/2026-04-19-deferred-v-results.md](plans/2026-04-19-deferred-v-results.md)
- [docs/plans/2026-04-19-sink-protection-results.md](plans/2026-04-19-sink-protection-results.md)
- [docs/plans/2026-04-17-trellis-v2-design.md](plans/2026-04-17-trellis-v2-design.md)
- [tests/trellis-phase1/BACKLOG.md](../tests/trellis-phase1/BACKLOG.md)
- [docs/turboquant.md](turboquant.md) — TQ v5 technische Details
