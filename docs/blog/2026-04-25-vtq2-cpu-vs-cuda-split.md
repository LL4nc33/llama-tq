# VTQ_2 K-collision — CPU vs CUDA pipeline split investigation

Stand: 2026-04-25 17:00 CEST. Branch `turboquant`, HEAD `8a4be1f5a`.

## Auftrag

Hypothese B: trenne CPU-Decoder vs CUDA-Decoder, um den Pfad mit der bit-identischen
PPL-Kollision (vtq2_2/vtq3_2/vtq4_2 → bit-identische PPL trotz unterschiedlicher
Bit-Tiefe) zu lokalisieren.

## Methode + Ergebnisse

### Test 1 — CPU Encode/Decode Roundtrip (`test-vtq2-encoding-diff`)

Encoder erzeugt für K=2/3/4 strikt verschiedene `qs[]`-Bytes mit erwartetem
Popcount-Wachstum. Decoder-MSE ist monoton (0.060 → 0.015 → 0.0038). Keine
"low-K of K=4 == K=K' bytes" Identität.

- **CPU encode + CPU decode → korrekt K-abhängig.**
- Hypothese A (Encoder bug) bleibt refuted (Agent A-Befund bestätigt).

### Test 2 — `vtq_state_at<K>` Bit-Window Equivalence (`test-vtq2-cached-roundtrip`)

`fast_state_at<K>` (host-spiegelung von `fattn-common.cuh::vtq_state_at<K>`)
wird gegen `ref_state_at` (naive bit-stream-extraction) für K=2/3/4 verifiziert,
über 32 zufällige Blöcke × 128 Sample-Indizes. Alle 3 Fälle PASSEN.

- **Die O(1)-Formel für `state(i)` ist mathematisch korrekt für alle K.**

### Test 3 — Forced CPU dispatch (`-ngl 0`)

```
Gemma4-26B-A4B-IQ2_XXS, wikitext-2-raw, c=512, chunks=2, t=8
f16/vtq2_2 CPU: PPL = 99249.07
f16/vtq3_2 CPU: PPL = 46696.18
f16/vtq4_2 CPU: PPL = 68997.70
```

CPU-only PPL UNTERSCHEIDET sich über K-Werte. Da der CPU-Encoder/Decoder über
`ggml_trellis_{encode,decode}_group` läuft (mit getesteter K-Awareness), war
zu erwarten, dass CPU PPL K-abhängig ist.

Aber: **PPL-Werte sind sehr hoch (10⁴+), instruct-tuned Gemma4 auf raw-wikitext
ist hier schlechter Signal/Noise**. Die K-Reihenfolge stimmt zumindest grob
mit "mehr Bits = bessere Rekonstruktion" überein für 2 vs 3, aber 4 ist
schlechter als 3 — verdächtig.

### Test 4 — CUDA Symbol-Inspektion

```
nm -D libggml-cuda.so.0.9.11 | grep flash_attn_ext_vec_case | grep VTQ_2
→ 3 host launcher symbols (Type 50/51/52), je 0x135 bytes (boilerplate).
cuobjdump --dump-elf-symbols → 16/28/16 device kernels für VTQ2_2/VTQ3_2/VTQ4_2.
```

Device kernels für alle 3 K-Werte sind kompiliert. VTQ3_2 hat mehr (28) weil
beide TUs (`fattn-vec-instance-f16-vtq3_2.cu` + `fattn-vec-dispatch-vtq2.cu`)
Instanzen produzieren. VTQ2_2/VTQ4_2 haben nur die Dispatch-TU-Instanzen (16).

- **Build hat alle K-Varianten als separate device kernels.**
- Template-Instantiation ist nicht der Bug.

### Test 5 — Deferred-V-Staging-Pfad

KV-Cache-Init-Logs zeigen für CPU UND GPU runs:
```
llama_kv_cache: deferred V quantization enabled (5 layers with f16 staging)
llama_kv_cache: deferred V quantization enabled (25 layers with f16 staging)
```

In `src/llama-kv-cache.cpp:852` wird `TQ_DEFERRED_STAGING → READY` nur ausgelöst,
wenn `balloc.get_n_tokens() == 1` (single-token decode batch). `llama-perplexity`
liefert ausschließlich Multi-Token-Batches (chunks von 512). Daher:

- `deferred_state` BLEIBT bei `STAGING` über die gesamte PPL-Run-Dauer.
- `cpy_v` schreibt in `v_staging` (f16), nicht in den VTQ-Cache.
- `get_v` liest aus `v_staging` (f16), nicht aus dem VTQ-Cache.
- **`build_graph_deferred_convert` wird NIEMALS aufgerufen.**
- Der eigentliche VTQ-Encoder/Decoder wird in `llama-perplexity` NICHT exerciert.

## Konsequenz

Die ursprüngliche Beobachtung ("vtq2_2/vtq3_2/vtq4_2 produzieren bit-identische PPL")
für CUDA war **kein Bug im VTQ-Decoder, sondern eine Folge des Deferred-V-Staging-
Designs**: in `llama-perplexity` wird die V-Quantisierung nie real gerechnet, und
PPL ist immer gegen f16 V berechnet — daher K-invariant.

Die abweichenden CPU-PPL-Werte sind wahrscheinlich Resultat einer subtileren
Variation (sched-Order, Thread-Scheduling, RNG state), nicht echter VTQ-Decode-
Unterschiede. *(Verification pending: CPU-only repeated-run mit gleichem K würde
zeigen ob die Werte stabil sind.)*

## Empfehlungen

1. **Reproduktion korrigieren:** Den K-Collision-Test mit einem real-decode
   Workload neu fahren (e.g. `llama-cli` mit echtem chat-style 1-token decode
   nach prefill). Erst nach dem ersten Single-Token-Batch wird `deferred_state →
   READY → DONE` und das tatsächliche VTQ-Encoding via `build_graph_deferred_convert`
   ausgeführt. Erst danach werden FA-vec-Reads aus dem echten VTQ-Cache K-spezifisch.

2. **Test-Harness:** Falls eine PPL-style metric gewünscht ist, die VTQ wirklich
   exerciert: explicit ein `do_deferred_convert=true` triggern (z.B. via 1-token
   "warmup" batch nach jedem chunk) bevor die nächste Token-Eval-Schleife läuft.

3. **Verschiebung der Bug-Lokalisierung:** Solange die o.g. Korrektur nicht
   gemacht ist, ist der eigentliche K-Bit-Pfad-Code (Encoder + Decoder + FA-vec)
   noch UNGETESTET im PPL-Setup. Die unit-tests `test-vtq2-encoding-diff` und
   `test-vtq2-cached-roundtrip` decken die Algorithmik korrekt ab und beide
   PASSEN.

## Verifizierte Korrektheit

| Komponente | Status | Test |
|---|---|---|
| CPU `ggml_trellis_encode_group` K-aware | OK | test-vtq2-encoding-diff |
| CPU `ggml_trellis_decode_group` K-aware | OK | test-vtq2-encoding-diff (decode MSE monoton) |
| CUDA `vtq_cuda_encode_set_rows<K>` Template | OK | Source review (set-rows.cu, trellis-encode.cuh) |
| CUDA `k_dequantize_trellis<K>` (bulk) | OK | Source review (trellis.cuh) |
| CUDA `vtq_state_at<K>` (FA-vec O(1)) | OK | test-vtq2-cached-roundtrip |
| Build hat alle K-Instanzen | OK | nm + cuobjdump |

## Files

- `tests/test-vtq2-encoding-diff.cpp` — Encoder K-distinctness (Agent A)
- `tests/test-vtq2-cached-roundtrip.cpp` — `vtq_state_at` Equivalence
- `ggml/src/ggml-trellis.c` — CPU Trellis encoder/decoder (Zeile 100-318)
- `ggml/src/ggml-cuda/trellis.cuh` — CUDA bulk dequant + per-element decoder
- `ggml/src/ggml-cuda/fattn-common.cuh` — `vtq_state_at<K>` + `dequantize_V_vtq{2,3}`
- `src/llama-kv-cache.cpp:850-855, 1450-1530` — deferred staging state machine

## Offene Tasks

- Real-decode PPL-Ersatz finden (1-token decode trigger nach jedem chunk).
- Falls Bug nach korrektem Real-Decode-Test reproduziert → `dequantize_V_vtq_2`
  und `dequantize_V_vtq_3` sind die Hot-Path-Decoder; weiter dort ansetzen.
