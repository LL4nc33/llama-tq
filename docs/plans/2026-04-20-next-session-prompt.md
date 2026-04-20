# Compact-Prompt für nächste Session (2026-04-20)

Kopier diesen Prompt nach `/compact`:

---

## Status Phase 2 — Sprint-Plan aktiv

Arbeitsverzeichnis: `/mnt/d/repos/oidanice-llama` (deployed OidaNice-Frontend,
Backend = llama-tq in `/mnt/d/repos/llama-tq/`)

Wir arbeiten an **llama-tq** TurboQuant-Optimierung in **Phase 2**. Sprint-Plan
siehe `/mnt/d/repos/llama-tq/docs/plans/2026-04-20-phase2-sprint-batched.md`.

### Letzter Stand (2026-04-20, vor Compact)

**Aktive Branches:**
- `phase2` — Merge-Target für alle Tricks (aktuell 7 Commits)
- `trick2-pr1-profile-heads` — Profile-Hook + iswa/hybrid unwrap fixes
- `trick2-pr2-mixed-precision` — **Commits: dc99a45bb, fda5827b9, 9d5f6e0fd, a0e428a60, ca7b22ac6**

**PR2-Status:**
- ✅ Resolver-Infra (tq-profile.cpp, 500 LOC, 6 Strategies)
- ✅ CLI-Flags (--tq-v-profile/strategy/base/high/low/override/budget-bpw)
- ✅ Plumbing durch kv_cache + kv_cache_iswa + hybrid + hybrid_iswa
- ✅ FA dispatch: KTQ2_1 × VTQ{2,3,4}_2 cases hinzugefügt (commit ca7b22ac6)
- 🔄 **Rebuild läuft auf gpu00 (fattn.cu ~50min single-thread)**, nach Compact checken:

```bash
ssh claude@gpu00.node "ls -la /home/claude/llama-tq/build-cuda-trick2/bin/llama-perplexity; tail -5 /tmp/pr2-fa-build2.log"
```

**Wenn Build fertig:** PPL-Validierung Mid-Layer-Upgrade mit **korrekten Attention-Indices**
(dank Distillery-Claude via LEGION):

Qwen3.5-35B-A3B Attention-Layer sind **3, 7, 11, 15, 19, 23, 27, 31, 35, 39**
(letzter Layer jedes 4er-Blocks, `full_attention_interval=4`).

Test-Commands nach Build-Success:
```bash
# Baseline uniform vtq3_2
ssh claude@gpu00.node "/home/claude/llama-tq/build-cuda-trick2/bin/llama-perplexity \
  -m /home/lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf -ngl 99 --flash-attn on \
  --cache-type-k ktq2_1 --cache-type-v vtq3_2 --tq-protect-sinks 4 \
  -f /home/lance/models/wiki.test.raw --chunks 5 -ts 12,12 --no-mmap 2>&1 | grep Final"

# Alle 10 Attention-Layer auf vtq4_2
ssh claude@gpu00.node "/home/claude/llama-tq/build-cuda-trick2/bin/llama-perplexity \
  -m /home/lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf -ngl 99 --flash-attn on \
  --cache-type-k ktq2_1 --cache-type-v vtq3_2 --tq-protect-sinks 4 \
  --tq-v-override '3:vtq4_2,7:vtq4_2,11:vtq4_2,15:vtq4_2,19:vtq4_2,23:vtq4_2,27:vtq4_2,31:vtq4_2,35:vtq4_2,39:vtq4_2' \
  --tq-v-strategy manual \
  -f /home/lance/models/wiki.test.raw --chunks 5 -ts 12,12 --no-mmap 2>&1 | grep -E 'Final|tq-v-mixed'"
```

Wenn PPL signifikant anders → PR2 ready for merge phase2→master.

### Infrastructure-Verbesserungen (gemacht)

- ✅ ccache **user-local** installiert (`~/.local/bin/ccache`, 20GB cache size)
  → Zukünftige Builds nutzen cache, erwartet 5min statt 40min bei unveränderten fattn.cu
- ⚠️ fattn.cu split als Batch B4 geplant (Infrastructure-Win)

### Nächste Arbeits-Session: Batched Workflow

Nach PR2-Merge starten wir den **Batched Sprint**:

**Batch A — Fast Tricks (serial, ~5min rebuild):**
- A1: Trick 16 Auto-bpw solver (CLI + common)
- A2: Trick 10 Block-variable bpw (extension zu PR2)
- A3: Trick 7 Shared LUT optimization (cleanup)
- A4: llama-cli -no-cnv fix (#139) — unblockt 35B Profile
- A5: Profile-Hook perplexity-path fix

**Batch B — CUDA Tricks (parallel agents, 1 gemeinsamer Rebuild ~50min):**
- B1: Trick 4 Correction Overlay (design fertig)
- B2: Trick 6 CUDA receiver-side Viterbi
- B3: Trick 9 Precomputed V signs
- B4: fattn.cu split (Infrastructure)

Details im Sprint-Plan.

### LEGION

Distillery-Claude hat heute 2026-04-20_1910 die Qwen3.5-A3B Layer-Indices geliefert
(Attention bei 3,7,11,...,39). Alle Nachrichten auf `ack`. Bei neuen Messages:
`/legion` check.

### Offene Fragen für nächste Session

1. PR2 Mid-Layer-Test Ergebnis — lohnt merge?
2. Batch A starten oder erst noch polish-Items (docs, README)?
3. Trick 4 307MB Cost — akzeptabel für opt-in Feature?

### Wichtige Feedback-Points (aus Memory)

- Deploy IMMER via git push + pull, NIEMALS scp
- VOR deploy: `git status` prüfen, ALLE Files committen
- Keine Co-Authored-By Zeile in Commits
- Folder-Docs lesen VOR Arbeit, aktualisieren NACH Änderungen
- Performance ist oberste Priorität, von Anfang an mitdesignen
- Dokumentation nicht vergessen (Docs + LEGION + Memory + BookStack)
- Code muss für Laien intuitiv lesbar sein (OidaNice Philosophie)

### Production Running

- gpu00:8791 = **gestoppt** (Dev-Mode für Tests, GPUs frei)
- Production-Recipe wenn reactivate: `-c 400000 --parallel 1 --cache-type-k ktq2_1 --cache-type-v vtq3_2 --tq-deferred-k --tq-deferred-v --tq-protect-sinks 4`

---

Starte mit Check ob Build fertig ist. Dann autonomous weiter nach Sprint-Plan.
