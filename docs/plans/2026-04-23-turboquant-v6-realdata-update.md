# TurboQuant v6 — Real-Data Validation Update

**Datum:** 2026-04-23
**Follow-up zu:** `2026-04-23-turboquant-v6-poc-results.md`

## Update: Synthetic vs Real Data Gap

Die synthetischen PoC-Ergebnisse (22-24% Improvement) gelten für **artificial outlier channels mit 3-8x Variance-Ratio**. Auf echten Qwen3.5-27B V-Projection-Weights (post-RHT, seed 42) liefert das calibration script:

```
Max/Min variance ratio:           1.50x
Outlier/Regular var separation:   1.01
Verdict: WEAK separation
```

**Bedeutung:** RHT rotiert outliers in V-**weights** so effektiv weg, dass keine brauchbare outlier-separation bleibt. Die Paper-Claim (32 outlier channels auf Llama-3.1-8B) funktioniert vermutlich nur auf pre-RHT raw data oder auf V-**activations** (nicht V-weights).

## Hypothese

Es gibt zwei verschiedene "V" im KV-Cache-Kontext:

1. **V-projection-weights** (static, in GGUF gespeichert) — durch Training regularisiert, uniform nach RHT
2. **V-cache-activations** (dynamic, per-inference) — kontextabhängig, zeigen oft outliers durch:
   - Attention-sinks (first tokens accumulate importance)
   - Burst-patterns (specific content triggers high-variance responses)
   - Layer-specific patterns (early layers have more outliers)

Unser extract_v_samples.py extrahiert aus **V-weights**. Das ist falsch für outlier-calibration.

## Plan-Korrektur

Um Idee 2 (Outlier Channel Split) fair zu testen, brauchen wir **activation-extraction**:

1. Real model inference (llama-cli oder llama-perplexity) auf 1000-token text
2. Hook in KV-cache write path zum Dump der V-activations per-layer
3. Analyze post-RHT per-channel variance

Das ist 4-6 Stunden Engineering, kein quick check.

**Alternative Strategien** (günstiger):
- **A)** PPL-sweep mit synthetischem outlier-mask auf V-cache — Blackbox-Test. Mask based on channel-index patterns (z.B. every 4th channel als outlier). Wenn PPL nicht leidet → RHT-space ist uniform, Idee 2 nicht produktiv.
- **B)** Direkt die Paper-Claim replicieren: Llama-3.1-8B @ 2.5 bpw mit hardcoded outlier-mask aus Paper. Wenn PPL auf wikitext ~f16 erreicht → Paper-Technik works, Idee 2 proceed. Wenn nicht → RHT+Lloyd-Max auf Qwen (unsere Basis) ist schon besser als Paper-Basis.

## Empfehlung

**Option A ist schneller und aussagekräftiger.**

Mit aktueller VTQ2_1 infrastructure können wir in 1-2 Stunden:
1. Patch in ggml-cpu/ggml.c um bei outlier-channels andere centroids zu nutzen
2. PPL run auf Qwen3.5-27B mit verschiedenen outlier-configs:
   - baseline VTQ2_1 (aktuell, uniform 2-bit)
   - VTQ2_1_OUT_8  (8 outlier @ 3-bit, 120 @ 2-bit)
   - VTQ2_1_OUT_32 (32 outlier @ 3-bit, 96 @ 2-bit)
3. Outlier-Mask-Pattern: every-4th-channel (simpelster naive pattern, no calibration needed)

Falls "every-4th" schon 1-2% PPL-Verbesserung bringt → Calibration lohnt sich, proceed to full impl.
Falls nicht → RHT rotation ist für unseren Qwen workload schon optimal, Idee 2 skip.

## Decision Point

**User entscheidet:** Option A (1-2h PPL test) oder Option B (Llama-3.1-8B paper replication, 4-6h) oder **pivot zu neuer Richtung** (z.B. streaming-generation-window Idee 6, per-layer RHT seed Idee 7, oder fresh brainstorm)?

## Alternatives Next Step

Unsere **existierende VTQ2_1 liefert schon -1% PPL** auf Qwen3.5-35B-A3B. Das bedeutet:
- Für Accuracy ist wenig Raum nach oben
- Performance-Optimierungen bringen mehr (decode-throughput bei sehr langem Kontext)
- Memory-Optimierungen könnten 400k-context erweitern (z.B. 800k)

Möglicherweise ist der wahre Gewinn **nicht mehr Compression** sondern **bessere Dispatch/Memory-Strategy** — z.B.:
- Streaming-window Idee 6 (last N fp16) für bessere decode-latency
- Async KV-writes to hide quantize-cost
- Per-token dynamic precision based on attention-weight (compress unimportant tokens harder)
