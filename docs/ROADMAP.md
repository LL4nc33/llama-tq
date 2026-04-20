# llama-tq Roadmap

Letzte Aktualisierung: 2026-04-20
Maintainer: LL4nc33 (Lance / Maverick)

## Status: Phase 1 abgeschlossen ✅

Die `trellis-v2-phase1` Arbeit ist auf `master` gemerged und
production-validiert. Das VTQ_2 Trellis-coded V-cache System ist
fertig und deployable.

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

## Phase 2 — Trick-17-Serie (laufend)

Siehe `tests/trellis-phase1/BACKLOG.md` für Details zu allen 17 Tricks.
Hard-limit: Es werden nie mehr als 17. Neue Ideen ersetzen alte.

### Short-term (2-4 Wochen)

**Trick 1 — Attention-sink token protection** ✅ DONE
Layer-level Variante shipped, Token-level bleibt optional falls
Hybrid-Tensor-Layout später kommt.

**Trick 2 — Per-head precision mixing** → **in progress**
Ein Profiling-Pass klassifiziert Heads nach V-Varianz. High-variance
Heads kriegen vtq4_2, low-variance vtq2_2, Durchschnitt bleibt
bei ~3 bpw. Erwartung: substantielle Qualitätsverbesserung da
Varianz heavy-tailed ist.

**Trick 3 — Per-model RHT seed calibration** ✅ DONE

**Trick 4 — Correction overlay buffer**
fp16 `correction_buf[N]` pro Layer, speichert top-N schlimmste
Quantisierungsfehler. Lossless overlay. ~0.5-2% PPL-recovery
bei ~1% bpw overhead.

**Trick 5 — Per-head learned lambda sharpening**
Fine-tune-basierter Quality-Recovery. Braucht Training-Loop,
ist damit wesentlich größeres Projekt.

### Medium-term (1-3 Monate)

**Trick 6-16** — siehe BACKLOG. Highlights:
- FWHT-Rotation per Token statt per Group
- Deferred K-cache hybrid precision
- Learned RHT matrix (nicht nur random)
- Block-variable bpw
- Adaptive Lloyd-Max (pro Seq/pro Gen-Step)

**TQW2 — Weight Quantization** (Task #127, in progress)
VTQ war nur KV-cache. Modell-Weights bleiben IQ2_XXS / Q8_0.
TQW2 würde die Weights selbst auf 2-3 bpw bringen mit
Lloyd-Max-Qualität. **Größerer Hebel** als KV-cache, weil
Weights den Großteil des VRAM stellen. Python-Validierung schon
abgeschlossen (Task #126), CUDA-Sprint offen.

**Upstream-PR an ggml.org** (optional)
Wenn VTQ_2 für die Community relevant scheint: sauber aufteilen
in digestible PRs (type-enums → CPU-path → CUDA-path), mit
Papier-ähnlicher Dokumentation.

### Long-term (3+ Monate)

**Trick 17 — "The Big One"**
Reserviert für den finalen Trick — reine Notiz, noch nicht
definiert. Wenn er kommt, ist das Paper geschrieben.

**Paper**
Sobald Trick 17 benannt und validiert ist: Draft für ICLR 2027
oder ähnlich. Konkurrenz-Benchmarks: KVQuant, Aquila, QuaRot.

**Hardware-Support**
- RTX 40-series tuning (Ada architecture)
- AMD ROCm-Path (falls Community-Interesse)
- Apple Silicon MPS (Metal shader-Äquivalente zur Viterbi-Encoder)

---

## Infrastructure / Ops

### Repositories
- `LL4nc33/llama-tq` — aktiv gepflegter Fork mit VTQ_2
- `ggml-org/llama.cpp` — upstream (periodisch rebase/merge)
- `LL4nc33/oidanice-llama` — whitelabel AI platform (nutzt llama-tq als backend)

### Production-Server
- `localhost:8791` — Qwen3.5-35B-A3B, VTQ_2 (ab 2026-04-20)
- `internal-host:8792` — FunctionGemma 270M (tool router)

### Testing
- Lokaler CPU-Roundtrip in `tests/trellis-phase1/`
- PPL sweep auf test-rig (wikitext-2)
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
