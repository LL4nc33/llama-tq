# VTQ_1_3 — Research + Rejection

Stand: 2026-04-25. Phase 3.5 der post-Gemma4 Roadmap. Analog zur KTQ_3 Untersuchung wird hier evaluiert, ob ein 1-bit Pendant zur VTQ_3-Familie (`vtq1_3`) die Pareto-Lücke zwischen `vtq1_1` (1.06 bpw, PPL +16%) und `vtq2_3` (3.00 bpw, +tail-protection) füllen kann.

**Result: rejected.** Der Outlier-Split funktioniert in der VTQ_1-Familie nicht, weil die Block-Größe zu klein ist und der relevante PQ-Code-Pfad keinen Trellis-Header hat, an dem sich der Outlier-Overhead amortisieren ließe.

---

## 1. Familien-Architektur — was unterscheidet `_1` von `_3`?

### VTQ_1-Familie (`vtq{1..4}_1`) — PolarQuant
- Block-Größe `QK_VTQ = 32` (siehe `ggml/src/ggml-common.h:348`)
- Pfad: `set_rows_cuda_pq` in `ggml/src/ggml-cuda/set-rows.cu:466`
- Layout: `{ ggml_half d; uint8_t qs[QK_VTQ * b / 8]; }` — kein Trellis-Header, nur Scale + Codebook-Indizes
- bpw = `b + 0.5` (1.5 / 2.5 / 3.5 / 4.5 bpw für b=1..4)

### VTQ_3-Familie (`vtq{2,3,4}_3`) — Trellis + Outlier-Channel-Split
- Block-Größe `QK_VTQ_TRELLIS = 128`
- Pfad: `vtq_cuda_encode_set_rows` (Trellis-Encoder)
- Layout: `{ d, start_state, outlier_pos[4], outlier_val[4], qs[128 * b / 8] }`
  - Header `(d + start_state)` = 4 B, Outlier-Block = 12 B, Datenblock skaliert mit `b`
- bpw = `(4 + 12 + 16·b) / 128 * 8 = (16 + 16b) / 128` ≈ `b + 1.0` für b=2,3,4 → 3.0 / 4.0 / 5.0 bpw

Der Outlier-Overhead beträgt in der VTQ_3-Familie **konstant 12 B / 128 samples = 0.75 bpw**, unabhängig von `b`. Das ist tragbar bei b=2,3,4 weil Trellis-Header (4 B) und Datenblock (16·b B) ohnehin amortisieren.

---

## 2. Was würde `vtq1_3` kosten?

Wir müssten den Outlier-Mechanismus in den **PolarQuant-Pfad** (block_size 32) integrieren — der Roadmap-Hinweis war explizit, dass VTQ_1 nicht trellisbasiert ist und kein `start_state`-Feld kennt.

Mögliche Layouts bei `QK_VTQ = 32`, `b = 1` und Outlier-Anzahl K:

| K outliers | Layout | Größe | bpw | Vergleich |
|---:|---|---:|---:|---|
| 0 | `{d, qs[4]}` | 6 B | **1.50** | = `vtq1_1` (das ist es bereits) |
| 1 | `{d, qs[4], pos[1], val[1]}` | 9 B | **2.25** | = `vtq2_2` (3.0 bpw avg in Code), aber 1 outlier dünn |
| 2 | `{d, qs[4], pos[2], val[2]}` | 12 B | **3.00** | = `vtq2_3`, dominiert durch existierenden Type |
| 4 | `{d, qs[4], pos[4], val[4]}` | 18 B | **4.50** | > `vtq3_3` (4.0 bpw), schlechter |

Roadmap-Annahme war "1.81 bpw" — das hätte vorausgesetzt, dass Outlier-Overhead über 128 samples amortisiert (wie bei VTQ_3). Bei `QK_VTQ = 32` ist der Overhead 4× so dicht. Die zentrale Pareto-Lücken-Begründung kollabiert.

---

## 3. Drei Alternativen — alle verworfen

### 3.1 Mit 1 outlier (~2.25 bpw)
Pareto-dominiert von `vtq2_2` (Trellis, 2.25 bpw, ~6× bessere rel-MSE wegen 2-bit + Trellis-Korrelation). Ein einzelner Outlier-Slot pro 32 samples kann den 1-bit-Codebook-Tail-Collapse nicht zuverlässig fangen — Wahrscheinlichkeit, dass der "echte" Maximum-|x|-Sample erfasst wird, ist etwa Top-1/32, aber 1-bit-Codebook scheitert typischerweise bei den top-3-bis-4 Samples.

### 3.2 Mit 4 outliers @ QK_VTQ=32 (4.50 bpw)
4 von 32 Samples = 12.5% sind keine Outlier mehr, das ist ein kompletter Codebook-Switch. Bei 4.50 bpw ist `vtq3_3` (4.00 bpw, Trellis + 4 outlier) strikt überlegen.

### 3.3 Neuen Block-Typ `QK_VTQ_PQ_LARGE = 128` für `vtq1_3`
- Layout: `{ d (2B), qs[16], pos[4], val[8] }` = 30 B / 128 = **1.875 bpw** ✓ (matches roadmap target)
- Aber: 1-bit PolarQuant über 128 samples mit nur **einer** scale `d`. Standard PolarQuant-Annahme (RHT-rotated → near-Gaussian) hält über 32 lanes; bei 128 lanes wird die per-block-Varianz signifikant. Die scale `d` müsste größer werden um Tail nicht zu clippen → mehr Mid-range-Quantisierungsfehler.
- Encoder + Decoder + FA-Dispatch + KV-Cache-Sizing + Bench-Harness sind **kein** drop-in von VTQ_1. Es wäre eine eigene Familie, kein "_3"-Variant.
- Zusätzlich: ohne Trellis ist 1-bit "_3" konzeptuell schwach — der größte Gewinn der `_3`-Familie auf VTQ_2/_3/_4 kam aus der Kombination Trellis-Korrelation + Tail-Handling. 1-bit Trellis wäre quasi greedy (zu wenig Info-per-Step).

---

## 4. Was tatsächlich helfen würde

Der eigentliche Grund warum `vtq1_1` auf D=512 schwächelt ist nicht "fehlende Outlier-Slots", sondern **single-scale per block** + **kein Trellis-Kontext**. Die richtige Antwort ist eine der folgenden, alle außerhalb von Phase 3.5:

1. **Phase 7 (imatrix-aware calibration)** — model-specific Lloyd-Max-Codebook-Refit. Würde `vtq1_1` direkt verbessern bei +0 bpw cost.
2. **Phase 5 (per-head adaptive bpw)** — high-variance heads bekommen `vtq2_3`, low-variance heads `vtq1_1`. Erreicht den gleichen "fülle die Pareto-Lücke"-Effekt auf System-Ebene.
3. **VTQ_1 D=512 dedicated kernel** (im "Nicht-Pareto" tracker bereits) — das Problem ist das Kernel-Layout, nicht die Block-Encoding.

---

## 5. Decision

**vtq1_3 wird nicht implementiert.** Phase 3.5 in der Roadmap auf "rejected, see this research doc" gesetzt.

Begründung kompakt:
- PolarQuant-Familie hat `QK_VTQ = 32` — Outlier-Overhead amortisiert nicht (analog zu KTQ_3 bei `QK_KTQ = 32`)
- Mit ehrlichen 4 outliers landet das Layout bei 4.5 bpw (worse than `vtq3_3`)
- Alternative mit größerem Block (1.875 bpw) verlässt die VTQ_1-Familie und ist 1-bit-PolarQuant über 128 samples — eigene Forschungsfrage, keine Phase-3.5-Erweiterung
- Pareto-Lücke 1.06 → 3.00 bpw wird besser von Phase 5 (adaptive bpw) und Phase 7 (calibration) adressiert

Pattern matches KTQ_3-Rejection (`docs/plans/2026-04-25-ktq3-research.md`): kleine Block-Größen + Outlier-Overhead funktioniert nicht. Die "_3"-Generation ist auf Trellis-Block-Größen beschränkt.
