# Phase 2 Sprint — Batched Tricks Workflow

**Datum:** 2026-04-20
**Branch:** `phase2`
**Motivation:** fattn.cu recompile dauert 40-50min. Pro Trick rebuilding ist teuer.
**Lösung:** Tricks nach Build-Footprint gruppieren, parallel entwickeln, gemeinsam testen.

---

## Status quo beim Planen

**Completed (phase2 + sub-branches):**
- ✅ #120 Trick 1 Attention-sink protection (merged master)
- ✅ #122 Trick 3 Per-model RHT seed (merged master)
- ✅ #137 Trick 2 PR1 Profile-Hook (branch `trick2-pr1-profile-heads`)
- ✅ #138 Trick 2 PR2 Mixed-Precision impl (branch `trick2-pr2-mixed-precision`) — **FA-rebuild läuft aktuell**

**Pending Production-Wins:**
- 🔄 PR2 mid-layer PPL validation (blockiert durch FA-rebuild)
- 📋 #140 Trick 4 Correction Overlay (design done, 307MB memory cost flagged)
- 📋 #121 Trick 2 PR3 Per-Head (stretch, nach PR2)
- 📋 #123 Trick 4 Impl
- 📋 #124 Trick 5 Learned λ

---

## Workflow-Regel

**Vor Commit-Push:** Trick in eine der zwei Kategorien einordnen.

| Kategorie | Kriterium | Build-Dauer |
|-----------|-----------|-------------|
| **A: Fast** | Keine `ggml/src/ggml-cuda/*` Änderung | ~5min |
| **B: CUDA** | Änderung in `fattn*.cu*` oder `trellis-*.cuh` oder neuer KV-Typ | ~50min |

**Batch-Prinzipien:**
- Batch A: serial in einer Session, ein Final-Rebuild
- Batch B: parallel durch Agents (jeder eigener Branch), dann gemeinsamer Merge + Rebuild
- Interface-Kontrakte pro Trick **vor** Code-Start designen (architect-agent)
- Alle Tricks opt-in via CLI-Flag, Default OFF → Backward-Compat garantiert
- Round-trip Unit-Tests (CPU) **vor** CUDA-Merge

---

## Batch A — Fast Tricks (single session, ~5min rebuild)

Keine CUDA-Kernel-Änderungen. Nur `common/`, `src/`, `include/`.

### A1: Trick 16 — Auto-bpw budget solver
**Scope:** CLI + common-path only
**Files:** `common/arg.cpp`, `common/tq-profile.{h,cpp}` (budget_bpw bereits scaffolded)
**Task:** Implement constraint solver that downgrades worst upgrades until
average bpw ≤ target. Already stubbed in resolve_v_types.
**Build-touch:** common only → 5min
**Prereq:** PR2 merged

### A2: Trick 10 — Block-variable bpw
**Scope:** Extension zu PR2 — statt per-layer, per-block (innerhalb eines Layers)
**Files:** `src/llama-kv-cache.cpp` (layer struct), `common/tq-profile.cpp`
**Task:** Wenn ein Layer heterogene Head-Varianz zeigt, split seine 10 V-blocks
in HIGH/LOW Groups. Opt-in via `--tq-v-block-mixing`.
**Build-touch:** common + llama-src only → 5min
**Prereq:** PR2 merged, per-layer funktional

### A3: Trick 7 — Shared LUT optimization
**Scope:** Mehrere TQ-Typen teilen sich eine CDF-LUT statt 3 Kopien
**Files:** `ggml/src/ggml-trellis.c` (existing file, no fattn touch)
**Task:** Static-init einmal, Pointer-share in code_table()
**Build-touch:** ggml-trellis only (wird auch in libggml-base compiliert) → 2min
**Impact:** -512KB Binary size, minimal perf gain aber "Cleanup Win"

### A4: LLama-CLI Interactive Bug Fix (#139)
**Scope:** llama-cli `-no-cnv` funktioniert nicht → Profile-Hook auf 35B blocked
**Files:** `tools/cli/cli.cpp`
**Task:** Respect `-no-cnv` when `-p` + `-n` are set; exit cleanly after `-n` tokens
**Build-touch:** tools/cli only → 3min
**Impact:** Unblockt 35B Profile-Run → ermöglicht natives Trick 2 Auto-Select

### A5: Profile-Hook Perplexity-Path Fix
**Scope:** Hook triggert nicht in `llama-perplexity`
**Files:** `src/llama-context.cpp` (tq_profile_collect_v)
**Task:** Debug warum `n_kv_layers=0` bei perplexity; fix so dass hook
auch in non-interactive decode-pfad feuert.
**Build-touch:** src only → 5min
**Impact:** Echter 35B Profile-Run auf wikitext möglich

**Gesamt Batch A:** ~5-8h Arbeit, **1 Build (~5min)**, validated gemeinsam.

---

## Batch B — CUDA Tricks (parallel agents, ~50min rebuild gemeinsam)

Jeder Agent arbeitet isoliert auf eigenem Branch, scopes überlappen nicht.

### B1: Trick 4 — Correction Overlay Buffer
**Design:** `docs/plans/2026-04-20-trick4-correction-overlay-design.md` (fertig)
**Scope:** Encode-hook in bulk Viterbi + decode-hook in FA kernel
**Files:** `ggml/src/ggml-cuda/trellis-decode.cuh`, `ggml/src/ggml-trellis.c`, `src/llama-kv-cache.cpp`
**Agent-Scope:** B1-only Branch
**CUDA-touch:** trellis-decode.cuh (FA wird nicht angefasst, nur der shift-register) → 20min
**Memory-Cost:** 307MB bei 200K ctx (flagged, opt-in via flag)
**Risk:** Encode-path Integration bei deferred-v (hook at kv-cache.cpp:2181)

### B2: Trick 6 — CUDA receiver-side Viterbi
**Scope:** Port CPU-Viterbi encoder nach CUDA kernel für bulk-convert at prefill→decode
**Files:** Neu: `ggml/src/ggml-cuda/trellis-encode-receiver.cu`
**Agent-Scope:** B2-only Branch
**CUDA-touch:** Neue Datei, keine fattn.cu-Änderung → 15min
**Impact:** 3-5× encoder speedup für deferred-V

### B3: Trick 9 — Precomputed V signs
**Scope:** Sign-bits für VTQ*_2 im GGUF vorab berechnen, decode nutzt direkt statt RHT
**Files:** `ggml/src/ggml-cuda/trellis-decode.cuh`, `src/llama-quant.cpp` (GGUF convert)
**Agent-Scope:** B3-only Branch
**CUDA-touch:** trellis-decode.cuh (decode path), no fattn.cu → 20min
**ABI:** Breaks old VTQ GGUFs (requires version bump or migration)

### B4: fattn.cu Split (Infrastructure)
**Scope:** fattn.cu in 4-6 kleinere Files per K-type-Gruppe splitten
**Files:** `ggml/src/ggml-cuda/fattn.cu` → fattn-base.cu + fattn-tq.cu + fattn-vtq.cu + fattn-mixed.cu
**Agent-Scope:** B4-only Branch
**CUDA-touch:** Massive refactor, muss parallel-buildbar sein
**Impact:** **Zukünftige Builds: 50min → ~8min mit -j4**

**Build-Strategie Batch B:**
```
1. Agent B1 spawns → commits auf Branch trick4-overlay
2. Agent B2 spawns → commits auf Branch trick6-cuda-viterbi
3. Agent B3 spawns → commits auf Branch trick9-precomputed-signs
4. Agent B4 spawns → commits auf Branch phase2-fattn-split
   (B4 SEPARAT testen, größtes Refactor-Risiko)
5. Parent wartet auf alle → merged B1+B2+B3 in batch-b-cuda
6. EIN rebuild (~50min) → Test-Suite
7. Wenn grün: merge batch-b-cuda → phase2
8. B4 nur mergen wenn A+B+Tests alle grün
```

**Risiko-Management:**
- Vor Agents-Dispatch: Design-Docs für jeden Trick reviewen (Interface-Kontrakte)
- Agents erhalten isolated file-scopes (kein overlap)
- Backward-compat: alle CUDA-features via flag + CPU-fallback
- Nach Merge: Round-trip MSE-Tests (CPU) vor PPL-Tests (CUDA)
- Bei Fehler: git bisect über die 3 Merges

---

## Execution-Plan

### Jetzt (in aktueller Session):
- Warte auf PR2 FA-Rebuild (~30min noch)
- PR2 Mid-Layer PPL validation → wenn positiv, merge phase2
- Commit + compact

### Nach Compact — Batch A Session:
Start in diesem File (Batch-Sprint-Plan) und arbeite serial durch A1-A5.
Ein gemeinsamer Build am Ende (~5min).

### Nach Batch A Test — Batch B Parallel Session:
Spawn 4 Agents parallel, jeder auf eigenem Branch. Parent koordiniert + merged.

### Final Validation:
- 35B PPL auf wikitext 50 chunks (distillery-empfehlung)
- TG benchmark vs baseline
- Blog-post + README update

---

## Compact Prompt (für nach /compact)

Siehe separate Datei: `docs/plans/2026-04-20-next-session-prompt.md`
