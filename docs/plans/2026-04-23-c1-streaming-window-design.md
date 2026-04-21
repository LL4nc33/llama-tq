# Option C1: Streaming-Window Last-N fp16 V-cache — Design

**Datum:** 2026-04-23
**Priorität:** HIGH (größter einzelner TG-Gain bei existierender infrastructure)
**Effort:** 2-3 Tage
**Risiko:** MITTEL

## Motivation

Attention-patterns in Decoder-LLMs sind **skewed toward recent tokens**:
- Erste 4-16 tokens (sinks) haben hohes Gewicht
- Letzte ~128-256 tokens haben hohes Gewicht
- Mittlerer Kontext (z.B. tokens 100-20000 bei 400k ctx) hat geringes Gewicht

Aktuell quantisieren wir **alle** tokens sofort. Das verursacht dequant-cost bei jeder decode-iteration auch für die letzten ~256 tokens die fast immer zugegriffen werden.

## Design

### Ring-Buffer-Struktur

```c
struct v_streaming_window {
    int64_t capacity;           // z.B. 256 tokens
    int64_t write_pos;          // ring-buffer head (modulo capacity)
    int64_t n_tokens_in_window; // current count (0..capacity)
    int64_t oldest_abs_pos;     // absolute token position of window's tail

    // Layer-parallele Arrays
    half *      fp16_buf_v[N_LAYERS];    // [capacity * n_kv_heads * head_dim * sizeof(half)]
    block_vtq2_1 * vtq_buf_v[N_LAYERS];  // quantized store for evicted tokens
};
```

### Lifecycle

**Auf KV-write (prefill + decode):**
1. Schreibe K in quantized-buffer direkt (K bleibt ungeändert — nur V hat Streaming-Window)
2. Schreibe V in fp16 ring-buffer am `write_pos`
3. Wenn `n_tokens_in_window >= capacity`:
   - Nimm das älteste token aus dem window
   - Quantize es und schreibe in `vtq_buf_v` bei dessen absolute position
4. Advance `write_pos` und `n_tokens_in_window`

**Auf Attention-Read:**
1. Layer führt FA-Kernel aus auf:
   - K: vollständig quantized (keine Änderung)
   - V: **Split** zwischen `vtq_buf_v[abs_pos < oldest_abs_pos]` und `fp16_buf_v[rest]`
2. FA-Kernel dispatch:
   - Option a: Zwei separate FA-Calls, merge outputs via softmax-reweighting
   - Option b: Single FA-Call der intern V-type-switch per token macht (mehr register-pressure)

**Option a ist einfacher, Option b ist schneller.**

### FA-Kernel Integration

Option a (sequential dispatch):
```cpp
// Layer forward:
auto out_quant = fa_kernel(Q, K_quant, V_quant, mask=pos<oldest_abs);  // existing path
auto out_fp16  = fa_kernel(Q, K_quant, V_fp16, mask=pos>=oldest_abs);  // new path, faster
// Combine via log-sum-exp softmax merger
auto out = softmax_merge(out_quant, out_fp16, lse_quant, lse_fp16);
```

Wir exponieren log-sum-exp aus FA (bereits vorhanden für numerical stability), dann können wir zwei partial-outputs kombinieren.

### Memory-Budget

```
Pro layer:
  fp16_buf_v: 256 tokens × 16 KV-heads × 128 head_dim × 2 bytes = 1 MB

Bei 60 layers (Qwen3.5-35B-A3B):
  60 × 1 MB = 60 MB extra VRAM

Plus die existierende VTQ storage.
```

60 MB bei 12 GiB VRAM = **0.5% extra** — vernachlässigbar.

### Performance-Modell

Decode-iteration auf 4k ctx:
- Aktuell: dequant 4k × V-blocks = ~0.4 ms pro layer
- Mit streaming: dequant (4k - 256) blocks + fp16 copy 256 tokens = ~0.38 ms pro layer
- **~5% pro layer × 60 layers = ~5% TG speedup**

Decode auf 400k ctx:
- Aktuell: dequant 400k × V-blocks = ~40 ms pro layer (!)
- Mit streaming: dequant 399.7k blocks + fp16 direct 256 = ~39 ms pro layer
- **~2.5% TG speedup**

Hmm — der Gewinn scaliert nicht so gut wie erhofft. Der dominante Cost-Faktor ist tatsächlich die Anzahl der Blocks, nicht per-iteration dequant für recent-window.

**Revised hypothesis:** C1 hilft vor allem bei **Batch-Inference** oder **Parallel Slots** wo der L1/L2-cache-hit-rate für recent tokens der Gewinn ist, nicht die eliminierte dequant-arithmetic.

### Alternative Interpretation: Cache-Locality

Bei 256 tokens fp16:
- 256 × 16 × 128 × 2 = 1 MB pro layer → fits in L2 cache
- Random-access to 1 MB L2 vs 8 MB (400k × 2bpw) VRAM = **~2× speedup** on V-access for recent tokens
- Recent tokens get **high softmax-weight** → dominate the output

**Net revised gain:** 5-15% TG auf typische Chat-Workloads (mostly recent tokens), less on RAG-heavy (uniform attention).

## Risiken

1. **Attention-merge correctness:** softmax-merge von zwei partial-outputs mit log-sum-exp ist Standard-Trick (siehe FlashAttention), aber muss sauber implementiert werden. Bug hier = silent wrong inference.

2. **Eviction-timing:** Bei burst-generation könnte der eviction-path alle tokens gleichzeitig quantisieren müssen. Brauche rate-limit oder async path.

3. **Prefill-interaction:** Auf prefill (ganze prompt in einem Pass) ist window halb gefüllt. Edge-case muss handled sein.

## Phasing

1. **Design-doc** (dieses doc) ✓
2. **Prototyp: Ring-Buffer struct in llama-kv-cache.h** — 4h
3. **FA-Split-Dispatch in CPU-path** (reference impl) — 1d
4. **CUDA FA-Split-Dispatch** — 1-1.5d
5. **Softmax-merge-math validation** (unit-test) — 4h
6. **Integration-Test auf Qwen3.5-35B-A3B** — 4h
7. **Production-Deploy + TG-Measurement vs baseline** — 2h

Gesamt: **2-3 Tage**.

## Abhängigkeiten

- Keine externen
- Sollte auf Trick-1 Pattern aufbauen (erste-N-tokens-fp16 ist ähnliches Konzept, nur am Anfang statt Ende)

## Code-Reuse aus existing Deferred-V Pattern

**Wichtige Entdeckung (2026-04-23):** Das existierende `tq_deferred_state`-Pattern mit `v_staging` buffer (src/llama-kv-cache.cpp:338) ist **90% der benötigten Infrastruktur**.

Aktueller `TQ_DEFERRED_STAGING` → `TQ_DEFERRED_READY` → `TQ_DEFERRED_DONE` lifecycle:
- STAGING: Prefill schreibt fp16 in `v_staging` (full-kv-size)
- READY: Am Prefill→Decode Transition
- DONE: Bulk-Viterbi-convert, `v_staging` released, writes direkt ins VTQ-Cache

**Streaming-Window lifecycle (neu):**
- STREAMING (neuer state): Continuous ring-buffer fp16 für last-N tokens
- Eviction-trigger: when `n_tokens_in_window >= capacity` at KV-write
- Kein terminierter Transition — läuft permanent

**Kritischer Unterschied:** `v_staging` ist **full-kv-size allocated** (400k tokens × fp16 = 120 GB für Qwen3.5-35B-A3B @ 400k ctx — deshalb ist deferred-V nur bei kleinen Kontexten aktiv). Streaming-Window braucht **fixed-size ring** (256 × fp16 = 60 MB für 60 Layers), unabhängig von total ctx.

**Ergo:** Wir können nicht einfach `v_staging` recyclen, aber das Pattern (`deferred_state` switch in `cpy_v`, build_graph_deferred_convert als Vorbild für eviction-convert) ist direkt nutzbar.

### Revised Effort Estimate

Mit Code-Reuse:
- Ring-buffer struct allocation: 4h
- `cpy_v` dispatch logic extend: 2h
- Eviction-trigger + incremental quantize graph: 1d
- FA-Split-Dispatch (attention-read liest teils ring, teils quant): 1-1.5d
- Softmax-merge validation + CUDA FA path: 1d

**Gesamt: 2-2.5 Tage statt 2-3.**

## Decision-Point

Diesen Design erst starten nach Result von A (VTQ3_1 PPL sweep). Falls A zeigt "VTQ3_1 ist deutlich besser als VTQ2_1" → implementiere VTQ3_1 als new production default. Falls A zeigt marginal gain → C1 ist die richtige Priorität.

**Aktueller Status:** PPL sweep läuft auf gpu00 (CPU, 0.8B q8_0). ETA ~45min.
