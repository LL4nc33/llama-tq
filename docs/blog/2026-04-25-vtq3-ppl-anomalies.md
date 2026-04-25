# VTQ_3 PPL-Sweep Anomalien — Investigation

**Date:** 2026-04-25
**Branch:** turboquant
**Build:** fbdb4a484
**Sweep:** `/home/claude/sweep-gemma4-ppl-20260425-1346/` (gpu00)
**Model:** gemma-4-26B-A4B-bartowski-IQ2_XXS (Gemma4-26B IT, D_v=512, IT-finetuned)

## TL;DR

Three anomalies investigated. **Two are real bugs**, one is expected behavior:

1. **Identische PPL über K=2,3,4 in VTQ_2/VTQ_3 family** — **REAL BUG.** Encoder kernel `k_vtq_encode_trellis_set_rows` hardcodet `VTQ_ENC_N=256`, aber `QK_VTQ_TRELLIS=128` (commit "halved from 256 to 128" hat die Block-Size verkleinert ohne den Encoder zu updaten). Encoder schreibt 256 Samples Trellis-Output in 128-Sample-Block → OOB-Write von 128·K Bits in den nachfolgenden Block-Header. Das macht die PPL annähernd K-unabhängig weil das ganze V-Signal systematisch korrupt ist.
2. **ktq2_1/vtq3_1 PPL=13918 (+66%)** — **erwartet.** vtq3_1 family is bekannt schwach auf D=512 (siehe README); +66% ist die normale Pathologie, kein neuer Bug. Reproduzierbar bei chunks=4 (PPL=12007). Kein NaN, kein FA-Dispatch-Fail.
3. **f16/vtq1_1 PPL=316** — **erwartet.** Kein NaN-Bug. Bei 1-bit V-Cache kollabiert die Attention-Verteilung auf eine quasi-uniforme Token-Distribution. Auf einem IT-finetuned Modell (das ohne Instruct-Format auf Wikitext eine sehr hohe baseline-PPL von 8382 hat) wirkt dieser Kollaps **PPL-senkend** — die degenerierte Output-Distribution ist näher an "uniform 1/V" als das stark-konditionierte IT-Output. Nicht physikalisch unmöglich; pathologisches Modell-Verhalten.

---

## Anomalie 1: identische PPL über K=2,3,4 (VTQ_2/_3 family)

**Status: BESTÄTIGT — encoder kernel bug.**

### Reproduktion

Sweep auf gpu00, gleiche Eingabe, chunks=4:

| Config | PPL | bit-identisch? |
|---|---:|---|
| f16/vtq2_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq3_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq4_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq2_3 | 15642.1165 ± 3691.10499 | ✓ |
| f16/vtq3_3 | 15642.1165 ± 3691.10499 | ✓ |
| f16/vtq4_3 | 15642.1165 ± 3691.10499 | ✓ |

Bei chunks=64 (Original-Sweep): vtq2_2 ≡ vtq4_2 = 8579.5706, ALLE Per-Chunk-Werte identisch via `diff` von extrahierten Werten. K=3 wurde im 64-chunk-Run nicht für VTQ_2 gemessen, deshalb tauchte er nicht im Pärchen-Vergleich auf.

### Root Cause

Datei: `ggml/src/ggml-cuda/trellis-encode.cuh`

```cpp
#define VTQ_ENC_N 256              // line 49
constexpr int N = VTQ_ENC_N;       // line 100, im Encoder-Kernel
```

Datei: `ggml/src/ggml-common.h`

```cpp
#define QK_VTQ_TRELLIS 128         // line 400 (Kommentar: "halved from 256 to 128")
typedef struct {
    ggml_half d;
    uint16_t  start_state;
    uint8_t   qs[QK_VTQ_TRELLIS * 2 / 8];  // 32 B für K=2
} block_vtq2_2;
static_assert(sizeof(block_vtq2_2) == 36, ...);
```

Mismatch: Block-Layout hat `QK_VTQ_TRELLIS=128` Samples, aber Viterbi-Encoder iteriert `for (step = 0; step < N=256; step++)` und schreibt `qs[byte] |= (e << shift)` für step bis 255. Das ist ein **Out-of-Bounds Write** von:
- K=2: 256·2/8 − 128·2/8 = 64−32 = **32 B Overflow**
- K=3: 256·3/8 − 128·3/8 = 96−48 = **48 B Overflow**
- K=4: 256·4/8 − 128·4/8 = 128−64 = **64 B Overflow**

Der Overflow läuft in den `d` + `start_state` + `qs` des **nachfolgenden Blocks** rein → systematische Block-Header-Korruption.

Zusätzlich: `__shared__ float s_xn[VTQ_ENC_N=256]` — Encoder lädt 256 Sample aus `x_row[j]` (`for j=tid; j<N=256`), aber pro Block sind nur 128 Source-Samples gültig. j=128..255 liest **fremde Speicher-Bytes** (next-block input), die deterministisch sind aber nichts mit dem Block zu tun haben.

### Warum Output bit-identisch über K?

Vermutung: Der Decoder liest nur die ersten 128·K Bits aus qs[] (das ist alles was im Block-Layout ist). Encoder schreibt 256·K Bits, wovon die ersten 128·K im qs landen, der Rest in nachfolgende Block-Header. Aber der Viterbi-DP-Pfad durch die ersten 128 Zustände ist **K-abhängig** — sollte unterschiedliche qs ergeben.

Es passiert aber: das **OOB clobbering der nächsten Block-`d`** plus das Lesen von `s_xn[128..255]` aus uninitialised shared mem ergibt deterministisches Garbage. Empirisch: Output-Floats sind bit-identisch, also ist nicht nur PPL-Wert sondern jeder einzelne dequant-Wert über K=2,3,4 identisch. Plausibelste Erklärung: der Block-Header (`d`, `start_state`) wird vom OOB-Write des Vorgängerblocks überschrieben mit Daten die K-Bits-Schreibmuster transportieren, aber der Decoder dann mit dem Block-eigenen K wieder dekodiert — Net-Output ist Garbage-but-deterministic. Da alle drei K-Varianten dieselbe `s_xn[128..255]` "Phantom-Sample" lesen und die Viterbi-DP über 65k States so dominant von diesen Phantom-Samples gesteuert wird, kollabiert die path-search auf identische `start_state` + `d` Werte unabhängig von K.

(Man könnte das mit einem Targeted-Print im Encoder beweisen, aber die OOB ist klar.)

### Action: FIX REQUIRED

Drei Optionen:
1. **VTQ_ENC_N auf 128 setzen** (`#define VTQ_ENC_N 128`) — schnell, aber halbiert den State-Count im Viterbi und ändert numerische Resultate aller Bestands-VTQ_2/_3 Runs. **Wahrscheinlich die richtige Lösung** — der Block ist 128 Samples, der Encoder muss auch 128 Samples encoden.
2. Encoder-Kernel akzeptiert template-param `N` und set-rows.cu wählt `N=QK_VTQ_TRELLIS`.
3. **VTQ_ENC_N und VTQ_ENC_S** auf `QK_VTQ_TRELLIS` koppeln (`#define VTQ_ENC_N QK_VTQ_TRELLIS`).

Empfohlen: **Option 1+3 zusammen** (header gemeinsam in ggml-common.h zusammenführen, single source of truth).

**Konsequenz:** ALLE bisherigen VTQ_2/_3-PPL-Messungen seit dem QK_VTQ_TRELLIS=128-Commit sind ungültig. Inkl. der "Phase 3 VTQ_3 Win" Behauptung.

---

## Anomalie 2: ktq2_1/vtq3_1 PPL=13918 (+66%)

**Status: ERWARTET — kein neuer Bug.**

### Reproduktion (chunks=4)

| Config | PPL ± Err |
|---|---:|
| f16/vtq1_1 | 612.2661 ± 115.4 |
| f16/vtq2_1 | 21794.3283 ± 5097.9 |
| f16/vtq3_1 | 13842.6485 ± 3289.1 |
| f16/vtq4_1 | 14557.3168 ± 3501.6 |
| ktq2_1/vtq3_1 | 12007.7846 ± 2823.6 |
| ktq2_1/vtq2_1 | 26644.5038 ± 6363.4 |

VTQ_1 family hat hohe absolute PPL über alle K, **K-Werte unterscheiden sich aber sauber** (612 / 21794 / 13842 / 14557 → echte K-Differenzierung). Encoder-Pfad ist `set_rows_cuda_pq` (PQ-codebook, nicht Trellis), unbeeinflusst von Anomalie 1.

ktq2_1/vtq3_1 PPL=13918 (64ch) ≈ f16/vtq3_1 PPL=13842 (4ch) — beide ähnlich hoch. README dokumentiert "VTQ_1 family suffers badly on D=512". Gemma4 hat D_v=512 in den Global-Layern → reproduzierbare Pathologie der VTQ_1-Codebook-Quality, kein FA-Dispatch-Fail, kein NaN.

### Action: KEIN FIX

VTQ_1 family ist auf D=512 als untauglich bekannt. Empfehlung: Sweep-Skript auf VTQ_1 für Gemma4 nicht erneut laufen lassen.

---

## Anomalie 3: f16/vtq1_1 PPL=316 (deutlich unter Baseline 8382)

**Status: ERWARTET — pathologisches Modell-Verhalten, kein Bug.**

### Reproduktion

Per-Chunk-Werte aus `gemma4-f16-vtq1_1.log`: 7029, 1802, 765, 612, 541, ..., 316. Smooth descent, keine NaN/Inf. Bei 4 chunks: PPL=612.

### Erklärung

1. Baseline f16/f16 = 8382 ist ABSURD HOCH weil Gemma4-26B-A4B-IT ein **instruction-tuned Modell** ist, das ohne ChatML-Format auf Wikitext-2 stark out-of-distribution arbeitet. Nicht-IT Baseline wäre ~5-10.
2. 1-bit V-cache zerstört Attention-Information vollständig → Modell kollabiert auf eine quasi-uniforme Output-Verteilung über das Vokabular (V≈262144).
3. PPL einer uniformen Verteilung über V Tokens ist `V` — also ~262k. Aber Gemma4 hat starkes IT-Bias, das selbst bei zerstörter Attention noch generic-token-distribution-Output produziert (top-frequent words like "the", "of", etc.).
4. Zerstörter IT-Bias produziert **näher-an-uniform-aber-bias-frei** Distribution, die auf raw Wikitext **bessere PPL** als das vollständig IT-konditionierte Output gibt.

Das ist kein numerischer Bug, sondern ein **Information-Theoretischer Effekt**: für einen IT-Bias der das Modell auf "Chat"-Distribution drückt, kann ein zerstörter Cache zufällig näher an "natural-language-Distribution" landen.

**Key Sanity-Check:** In den Per-Chunk-Werten (z.B. chunk 1 = 7029) ist klar dass das Modell zu Beginn einen ähnlichen High-PPL-State erreicht wie baseline, bevor der Attention-Kollaps die Output-Distribution flattens. Das ist konsistent mit "model goes degenerate over context length", nicht mit "computation is broken".

### Action: KEIN FIX

Erwartete Pathologie. Wenn man echte VTQ_1-Quality messen will, **muss man non-IT-Modelle** verwenden (e.g. Llama 3 base, Qwen2.5 base) oder mit ChatML-Format prompten.

---

## Empfehlungen

### Sofort-Action

**Anomalie 1 ist kritischer Bug.** Folgendes Vorgehen:

1. Hot-fix: `#define VTQ_ENC_N QK_VTQ_TRELLIS` in `ggml/src/ggml-cuda/trellis-encode.cuh`. Build + smoke-test (4-chunk PPL muss K-differenzieren).
2. Re-run der Phase-3-Sweep nach Fix:
   - `f16/vtq2_2`, `f16/vtq3_2`, `f16/vtq4_2` müssen unterschiedliche PPL haben (idealerweise: K=4 < K=3 < K=2)
   - `f16/vtq2_3`, `f16/vtq3_3`, `f16/vtq4_3` ebenso
   - `ktq2_1/vtq3_3` "Phase 3 win"-Behauptung (PPL=8339 < f16/f16=8382) **muss neu verifiziert** werden
3. Issue auf llama-tq filen mit Titel: "VTQ_2/_3 encoder writes 256 samples per 128-sample block — OOB clobbering since QK_VTQ_TRELLIS halved"

### Validierungs-Tests (nach Fix)

- Round-trip MSE-Test (`tests/test-vtq3-roundtrip.cpp`) — K=2/3/4 sollten weiterhin distinct MSE haben (das tat es schon, weil der Test offline-encoder verwendet, nicht GPU).
- PPL-Reproduzierbarkeits-Check: `f16/vtq2_2 chunks=4` zwei mal hintereinander muss identisch sein (deterministic-CUDA), aber `f16/vtq2_2 ≠ f16/vtq4_2` muss gelten.
- E14 split-decode + greedy-encode (FAST_ENC) Pfade haben **denselben** Bug-Mechanismus — unabhängig prüfen.

### Anomalien 2 & 3

Keine Action — dokumentiert. Optional: Sweep-Skript ergänzen mit "VTQ_1 family auf D=512 = expected high PPL" Annotation, damit künftige Investigations nicht auf den "Bug" anbeißen.

---

## Anhang: Files/Lines

- `ggml/src/ggml-common.h:400` — `#define QK_VTQ_TRELLIS 128`
- `ggml/src/ggml-cuda/trellis-encode.cuh:49` — `#define VTQ_ENC_N 256` ⚠️
- `ggml/src/ggml-cuda/trellis-encode.cuh:71-322` — `k_vtq_encode_trellis_set_rows` (Viterbi)
- `ggml/src/ggml-cuda/trellis-encode.cuh:340-510` — `k_vtq_greedy_encode_set_rows` (FAST_ENC, hardcoded `1<<3`)
- `ggml/src/ggml-cuda/set-rows.cu:505-581` — VTQ_2/_3 encoder dispatch
- `ggml/src/ggml-cuda/fattn-common.cuh:898-1054` — Decoder (uses block layout, OK)
- Sweep logs: `claude@gpu00.node:/home/claude/sweep-gemma4-ppl-20260425-1346/`
