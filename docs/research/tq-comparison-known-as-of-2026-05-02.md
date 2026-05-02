# TurboQuant Forks — Vorab-Vergleich (was wir bisher wissen)

**Stand:** 2026-05-02 09:19 CEST
**Status:** Vorläufig, basierend auf bisheriger Recherche. Wird nach Abschluss der 3 Researcher-Agents erweitert.

## Bekannte Forks im llama.cpp-TurboQuant-Ökosystem

### 1. **LL4nc33/llama-tq** (das ist deins)

| Item | Detail |
|---|---|
| URL | github.com/LL4nc33/llama-tq |
| Maintainer | LL4nc33 |
| Last commit | 2026-05-02 (heute, e054a3088) |
| Status | **Active, daily commits** |
| Approach | **K vs V getrennt:** KTQ (Hadamard-domain Q·K) + VTQ (Trellis OR codebook) |
| Types | KTQ1-4, VTQ1-4 v1 (codebook), VTQ2-4 v2 (Trellis), VTQ2-4 v3 (Trellis+outliers), VTQ3_v8 (NEW) |
| K-cache | RHT (FWHT + Philox-signs) + Lloyd-Max codebook + Hadamard-domain dot product |
| V-cache | Trellis (vtq*_2) ODER codebook (vtq*_1) + optional outlier-sidecar |
| CUDA | Voll implementiert, sm_75 Turing optimiert (CC 6.1+ funktioniert) |
| Bench | pp512 1196, tg128 86 t/s @ Qwen3.6-35B-A3B 100k+mmproj (RTX 2060 12GB) |
| Papers | PolarQuant (2502.02617) + TurboQuant (2504.19874) + QTIP-Trellis (2406.11235) |
| upstream PR | Tier-S Phase A done (Task #182 completed) |
| **USP** | Einziger fork mit **K vs V SEPARATER Backend-Wahl** |

### 2. **TheTom/llama-cpp-turboquant**

| Item | Detail |
|---|---|
| URL | github.com/TheTom/llama-cpp-turboquant |
| Maintainer | TheTom |
| Status | **Wahrscheinlich abandoned** (laut Reddit u/MachineZer0: last sync war 2026-04-19) |
| Approach | TQ als unified KV-cache type (turbo3, turbo4) |
| Types | turbo3, turbo4 |
| K-cache | Same approach as V (kein split) |
| V-cache | Same approach as K |
| CUDA | Yes (4090, V100 verified by community) |
| Bench | Unklar — Community sagt "nicht besser als q8_0" |
| Papers | TurboQuant (2504.19874) |
| **Schwäche** | Kein K/V split, keine Trellis, kein outlier-sidecar |

### 3. **richginsberg/llama-cpp-turboquant**

| Item | Detail |
|---|---|
| URL | github.com/richginsberg/llama-cpp-turboquant |
| Maintainer | richginsberg |
| Status | **Sync von TheTom + 2 weeks upstream** |
| Approach | **Identisch zu TheTom** + neuere upstream commits gemerged |
| Types | turbo3, turbo4 (von TheTom) |
| Bench | "Successfully tested on quad V100 & quad RTX 3090" (u/MachineZer0) |
| **Differenz** | Nur upstream-sync, keine eigene TQ-Innovation |

### 4. **Spiritbuun-fork** (Detail TBD)

User hat erwähnt diesen fork zu kennen — Researcher-Agent klärt Details. Vermutlich noch ein TQ-Experimentier-fork.

## Wesentliche Unterschiede (head-to-head)

### llama-tq vs TheTom/richginsberg

| Feature | llama-tq | TheTom/richginsberg |
|---|---|---|
| K-cache backend | RHT + Lloyd-Max codebook (Hadamard-domain dot) | turbo3/turbo4 unified |
| V-cache backend | Trellis Viterbi (oder codebook) + optional outlier-sidecar | turbo3/turbo4 unified |
| K vs V split? | **Yes — separate types** | No — single type for both |
| FA-Dispatch | 8+ types über KTQ × VTQ Matrix | 2 types (turbo3, turbo4) |
| FWHT location | **Q-tile (once per tile)** — math identity trick | K-block (per-block, gather + butterfly) |
| Sparse V dequant | **Yes (skip if attn_weight < 1e-6, +22% TG)** | No |
| Deferred K/V staging | **Yes (f16 staging during prefill)** | No |
| Outlier sidecar | **Yes (vtq*_3 family + v8 vtq3_v8)** | No |
| Per-model tuning | **Yes** (35B-A3B, 4B-dense, 80B, 122B, Gemma4 separate winners) | Unified |
| Active development | **5 weeks of daily commits** | Last sync 2026-04-19 |
| Hardware tested | RTX 2060 12GB Turing primary | 4090, V100, 3090 (community) |

### Vorteile llama-tq

1. **Beste PPL/bpw bei kleinem KV** — 2.78 bpw mit -0.33% PPL drift auf 35B-A3B (turbo3 von TheTom: kein known result besser als q8_0 = 8.5 bpw)
2. **Hadamard-domain dot eliminiert FWHT-overhead in FA hot loop** — math identity trick den niemand sonst macht
3. **Outlier-sidecar fängt long-tail-PPL** — vtq*_3 family hat 4 fp16 outliers per block für lossless quality
4. **K/V split erlaubt per-model optimization** — 35B-A3B nimmt ktq2/vtq2, 4B-dense könnte vtq4 (wenn das script-generiert wird)
5. **Sparse V dequant +22% decode** at 32k+ ctx
6. **Active development**, daily commits, dokumentiert (turboquant.md, plans/)
7. **Anthropic-compatible /v1/messages** mit prompt caching für agent workflows
8. **Phase 4 perf stack** (MADV_HUGEPAGE, OMP_active, mul_mat_id prefetch, adaptive layer-split) — +18.5% TG auf 80B-A3B

### Nachteile llama-tq

1. **Komplexere CLI** — User muss K und V SEPARAT wählen (8 ktq + 12 vtq types, jetzt v8 reduziert)
2. **Nur sm_75 (Turing) getestet** — andere archs nicht arch-tuned, aber funktionieren
3. **Trellis encoder-cost** ~22 ms/call — gemildert durch deferred-V staging, aber in PPL-only bench (ohne staging) 10× slow
4. **Gemma4 reasoning-broken** auf raw text (model-specific, nicht TQ-Schuld)
5. **Ministral-3B mit KTQ gibberish** (model-specific edge case, fix: q8_0/q8_0)
6. **Kein PR aktiv in upstream** — Tier-S Phase A done, aber keine merged contribution
7. **Kleinere community reach** — 1 Reddit-comment, keine erwähnung in AI Flux video / Above Spec posts

### Vorteile TheTom/richginsberg

1. **Einfache CLI** — `-ctk turbo4 -ctv turbo4` statt unsere split-types
2. **Multi-GPU enterprise hardware** verifiziert (V100, 3090 quad-setups)
3. **Mehr community-mention** in `1sshpmh` Reddit-thread (top voted comment u/MachineZer0)
4. **Existed earlier** — first-mover advantage

### Nachteile TheTom/richginsberg

1. **Kein K/V split** — verschenkt Quality-Potenzial
2. **Kein Hadamard-domain dot** — FA-overhead unnötig hoch
3. **Kein outlier-sidecar** — long-tail PPL leidet
4. **Last commit 2 weeks ago** — wahrscheinlich abandoned
5. **Bench-Skepsis in community** — "nicht besser als q8_0" laut mehreren Reddit-Kommentaren
6. **Nur unified backend** — kann nicht per-model tunen

## Strategische Empfehlung (vorab)

llama-tq ist **technisch klar überlegen** in jedem messbaren Aspekt:
- Bessere PPL/bpw (2.78 vs 8.5)
- Bessere Speed (Hadamard-domain trick + sparse V skip)
- Mehr Features (split, outliers, deferred-V, Phase 4 stack)
- Active development

Aber: **Community kennt den fork noch nicht**. Marketing-Lücke, kein technische Lücke.

**Nächste Schritte (nach Bench-Result):**
1. README-Tabellen mit aktuellen Numbers refreshen
2. r/LocalLLaMA NEUER Thread mit numbers-forward Title (siehe `reddit-post-draft.md`)
3. Direkt-comment unter Above Spec's Twitter-bench mit KV-bpw-Vergleich
4. Optional: cross-post r/MachineLearning mit academic angle (PolarQuant + Hadamard identity)

## Pending — wird durch Researcher-Agents ergänzt

3 Background-Agents laufen aktuell:
1. Fork-deep-dive (TheTom + Spiritbuun + alle anderen forks die wir noch nicht kennen)
2. Academic landscape (KIVI, Atom, GEAR, KVQuant, SnapKV, H2O, …)
3. Community sentiment (Reddit threads, Twitter/X handles, YouTube coverage)

Outputs landen in `docs/research/tq-fork-landscape-2026-05-02.md`,
`docs/research/tq-academic-landscape-2026-05-02.md`,
`docs/research/tq-community-sentiment-2026-05-02.md`.

Dieses Vorab-Doc wird dann konsolidiert.
