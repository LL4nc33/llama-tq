# Option C: Memory & Dispatch-Optimierungen (bpw-unabhängig)

**Datum:** 2026-04-23
**Kontext:** v6-Compression-Ideen haben in echten Daten begrenzten ROI. Alternative Optimierungsrichtung.

## Prämisse

VTQ_1 production liefert:
- 70 tok/s TG auf Qwen3.5-35B-A3B
- -1% PPL vs f16 (durch Regularization auf großen Modellen)
- 400k context in 12 GiB VRAM
- Stabil in production seit 2026-03

**Das ist bereits sehr gut.** Die Low-Hanging-Fruits sind nicht mehr "weniger bpw" sondern:

1. **Decode-Latency**: erste Token nach Eingabe, p50/p99 latency
2. **Memory-Effizienz**: context-window bei fixed VRAM
3. **Concurrency**: mehrere parallel-slots ohne linear mehr VRAM
4. **Dispatch-Overhead**: FA kernel register pressure, L2 cache misses

## Idee C1: Streaming-Window Last-N fp16 V-cache

**Beobachtung:** Token N+1 attention liest fast ausschließlich die **letzten ~128-256 tokens** stark gewichtet. Aktuell quantisieren wir sofort bei KV-write.

**Vorschlag:** Ring-Buffer mit den letzten N=256 Tokens als fp16. Quantize nur wenn Token aus dem Window rausfällt.

**Gewinne:**
- Decode hat direktem fp16-access auf heißen Tokens — kein dequant-cost
- Für den seltenen Fall "attention greift weit zurück" trägt Quantization-Cost nur bei, nicht beim Normalfall
- Potentieller 10-20% TG improvement für typische Chat-Workloads

**Kosten:**
- 256 tokens × 128 head_dim × 40 heads × 2 bytes × 2 (K+V) = **5 MB** extra pro layer
- 60 layers × 5 MB = **300 MB** — akzeptabel bei 12 GiB
- Ring-Buffer-Logic in llama-kv-cache.cpp, ~200 LOC
- Dispatch-Logic: FA kernel muss "first N from fp16, rest from vtq" handlen — Code-Duplication

**Risiko MITTEL:** KV-Cache-Struktur wird komplexer. Wir haben Trick-1 (attention-sink first-4-fp16) als Pattern-Vorbild, Idee ist ähnlich aber am **Ende** statt Anfang.

**Effort:** 2-3 Tage.

## Idee C2: Async KV-Quantize (hide latency behind compute)

**Beobachtung:** Quantization-Cost tritt auf kritischem Path auf. Der GPU-Stream-State ist aktuell synchron: matmul → KV-write (includes quantize) → next matmul.

**Vorschlag:** KV-quantize auf **separatem CUDA stream**. Der next-layer matmul startet sofort nach K/V compute, quantize läuft parallel.

**Gewinne:**
- ~5-10% TG on decode path (quantize wird von matmul gehidet)
- Größer bei quantize-heavier paths (VTQ_2 wäre hier nützlich, aber wir nutzen VTQ_1)

**Kosten:**
- Dual-stream Synchronisation komplex
- Eine async-miss kann hang verursachen (stream-dependency bugs)

**Risiko HOCH:** Konkurrenz mit bestehendem FA-Stream, nicht-triviale stream-merge am Layer-Ende.

**Effort:** 3-5 Tage, hohe Debugging-Kosten.

## Idee C3: Dynamic Per-Token Precision (importance-weighted)

**Beobachtung:** Attention-weights sind heavy-tailed. Wenige Tokens dominieren die Softmax-Sum.

**Vorschlag:** Nach dem ersten forward-pass, pro Token den "average attention weight across subsequent tokens" tracken. Niedrige-importance Tokens später **stärker quantisieren** (re-quantize von VTQ_1 → hypothetisches VTQ2_2 mit 2× mehr Compression).

**Gewinne:**
- Dynamic bpw adjustment based on actual importance
- Could push effective bpw below 2.5 without PPL cost

**Kosten:**
- Re-quantization-logic (delete + rewrite KV slot)
- Importance-tracking overhead (extra memory per token)
- Attention-weight access in kernel (currently discarded after softmax)

**Risiko HOCH:** Kompletter redesign der KV-cache-semantik. Könnte subtle bugs introducen.

**Effort:** 1-2 Wochen.

## Idee C4: KV-Cache Compression via Paged Attention

**Beobachtung:** Wir haben continuous KV-allocation. Paged attention (vLLM-style) nutzt block-level memory-manager.

**Vorschlag:** Port paged-attention aus vLLM zu llama.cpp. Block-size typisch 16 tokens, allows "gaps" für sparse KV-cache ohne VRAM-Waste.

**Gewinne:**
- Bessere VRAM-Effizienz: n concurrent slots sharen KV-blocks
- Easier multi-slot batching

**Kosten:**
- MASSIVER Rewrite — touches llama-kv-cache, llama-context, llama-decode
- Upstream llama.cpp prüft schon paged-attention (siehe recent PRs)

**Risiko SEHR HOCH:** Nicht zu unserer derzeitigen Kapazität.

**Verdict: Skip.** Warte auf Upstream.

## Idee C5: KV-cache Prefetching (L2-aware)

**Beobachtung:** VTQ_1 dequant loads codebook indices + centroid LUT random-access. L2-cache miss rate ist vermutlich hoch.

**Vorschlag:** Restrukturiere block-layout um spatial locality zu erhöhen. Z.B. inter-leave `qs[0..15]` across tokens sodass 128 threads sequential access haben.

**Gewinne:**
- 10-30% decode TG bei memory-bound kernels
- Benefits scalen mit context-length

**Kosten:**
- Block-format breaking change — incompatible mit existierenden saved KV-checkpoints
- Encoder + Decoder kernel rewrite
- CPU reference impl muss matchen

**Risiko MITTEL:** Systematische Arbeit, aber self-contained.

**Effort:** 2-3 Tage testing + tuning. Real measurement nötig um Gain zu bestimmen.

## Idee C6: Attention-Sink Expansion (Trick-1 auf 16-32 Tokens)

**Beobachtung:** Trick-1 hat erste **4** Tokens als fp16. Arxiv "Streaming LLM" (2023) zeigt attention-sinks sind oft die ersten **4-16 Tokens**. Einige Papers gehen bis **32-64**.

**Vorschlag:** Expand attention-sink-window auf 32 Tokens. Kosten: 32 tokens × 128 × 40 × 2 × 2 × 60 = **40 MB** zusätzlich (trivial).

**Gewinne:**
- Bessere PPL auf attention-sink-sensitive Modellen (vermutlich marginal auf Qwen3.5)
- Backup gegen Edge-Cases wo Quant-Fehler in ersten Tokens cascaden

**Risiko NIEDRIG.** Einzelner Zahlen-Parameter-Change plus testing.

**Effort:** 2h.

## Priorisierung

| Idee | Effort | Gewinn | Risiko | Priorität |
|---|---|---|---|---|
| **C1 Streaming-Window Last-N** | 2-3d | **10-20% TG** | MITTEL | **#1** — highest TG-gain |
| **C6 Attention-Sink Expansion** | 2h | ~0.2% PPL | NIEDRIG | **#2** — easiest quick-win |
| **C5 L2 Prefetch / Layout** | 2-3d | **10-30% TG** | MITTEL | **#3** — block-format breaking |
| C2 Async KV-Quantize | 3-5d | 5-10% TG | HOCH | skip for now |
| C3 Dynamic Per-Token Precision | 1-2w | unklar | HOCH | skip for now |
| C4 Paged Attention | 2-3w | unklar | SEHR HOCH | skip (wait upstream) |

## Empfehlung

**Start C6** (quick-win, 2h). Dann **C1** (big-win, 2-3d). **C5** als optional follow-up.

C6 ist parallelisierbar mit dem PPL-Test der gerade läuft.

## Nächste Schritte (abhängig von A-Result)

- Falls A zeigt "VTQ3_1 deutlich besser als VTQ2_1" → outlier-split lohnt sich. Go to B (activation extraction).
- Falls A zeigt "VTQ3_1 nur marginal besser" → Compression ist nicht die bottleneck. Full pivot zu C.

Aktuelles TG production (70 tok/s) kann realistisch zu **~85 tok/s** mit C1+C5 getrieben werden. Das ist messbar besser als weiter an Compression zu feilen.
