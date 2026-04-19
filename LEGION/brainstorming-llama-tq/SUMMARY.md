# Brainstorming Summary: TP+TQ für llama-tq

Stand: 2026-04-15, 5 Researcher-Agents + 2 Follow-Up-Agents

---

## Must-Do (Kritisch, sofort umsetzen)

### 1. Gemma4 TQ Garbage-Bug fixen — Shared-KV-Layer Rotation Mismatch
- **Root Cause**: Gemma4 hat Layers ohne eigenen KV-Cache (`has_kv(il)==false`) die den Cache früherer Layer wiederverwenden. In `build_attn` wird Q mit der Rotation des AKTUELLEN Layer-Typs rotiert, aber K wurde mit der Rotation des CACHE-QUELL-Layers gespeichert. Bei Mixed SWA/Global → Rotation-Mismatch → Garbage.
- **Datei**: `src/llama-graph.cpp:2258-2266` (Rotation-Dispatch in build_attn)
- **Datei**: `src/models/gemma4-iswa.cpp:107-109` (shared-KV build_attn Aufruf)
- **Fix**: Q muss die Rotation des Cache-Quell-Layers bekommen, nicht des aktuellen Layers
- **Verify**: `LLAMA_ATTN_ROT_DISABLE=1` + TQ2_1 → wenn Output korrekt, Bug bestätigt

### 2. TP+TQ Validierung deployen (DONE)
- **Status**: Erledigt in `src/llama-context.cpp:2962-3004`
- Ersetzt den pauschalen Block durch Head-Divisibility + TQ-Block-Alignment Checks
- Upstream TP-Infrastruktur (Meta-Backend, NCCL AllReduce, Weight Split) existiert bereits

---

## Should-Do (Hoher Impact)

### 3. PCIe 3.0 AllReduce-Heuristik anpassen
- **Problem**: BF16-Kompression-Schwellwert (32768 Elemente) ist für RTX 4090 + PCIe 4.0 kalibriert
- **Fix**: Schwellwert auf ~16384 senken für PCIe 3.0 (halbe Bandbreite)
- **Datei**: `ggml/src/ggml-cuda/ggml-cuda.cu:1146`
- **Aufwand**: ~10 LOC
- **Impact**: Bessere Bandbreitenausnutzung bei TP

### 4. Layer-Split als Default für asymmetrisches PCIe empfehlen
- **Hardware-Constraint**: B450A Pro Max hat x16 + x4 Slots
- **Ergebnis**: Layer-Split ist ~30-50x schneller als TP auf diesem Board
- **TP-Overhead**: ~2-3 ms/Token (96 AllReduces über x4-Link)
- **Layer-Split-Overhead**: ~0.06 ms/Token (2 Transfers, eine Grenze)
- **Empfehlung**: `-sm layer -ts 1.2,1.0` (mehr Last auf GPU0 mit x16-Link)
- **Dokumentation**: In docs/tp-tq-design.md aufnehmen

### 5. BF16-Kompression auch im P2P-Fallback (ohne NCCL)
- **Datei**: `ggml/src/ggml-backend-meta.cpp:1758-1836`
- **Problem**: Ohne NCCL werden volle FP32-Tensoren kopiert, kein BF16
- **Aufwand**: ~30 LOC

---

## Nice-to-Have (Optional)

### 6. NCCL-Stream-Overlap (AllReduce parallel zu nächstem Compute)
- **Datei**: `ggml/src/ggml-backend-meta.cpp:1840-1870`
- **Impact**: <2% bei 2x RTX 2060 (AllReduce ~10-20µs vs Compute ~1ms)
- **Aufwand**: Hoch (gesamte Graph-Execution umstrukturieren)
- **Empfehlung**: Erst bei 4+ GPUs oder größerer Hardware relevant

### 7. FA Kernel Occupancy bei wenig Heads (TP=2, GQA)
- **Problem**: Qwen3.5-14B TP=2 → nur 4 KV-Heads pro GPU → 4 FA-Blöcke → GPU underutilized
- **Datei**: `ggml/src/ggml-cuda/fattn-vec.cuh` Launch-Konfiguration
- **Aufwand**: Mittel, braucht Profiling

### 8. Reduce-Scatter + All-Gather statt AllReduce
- **Ergebnis**: Bei 2 GPUs kein Vorteil (identisches Kommunikationsvolumen)
- **Empfehlung**: Nicht implementieren

---

## Konflikte / Offene Fragen

### K1: `-sm row` vs `-sm tensor` — Welches ist TP?
- Agent #5 (Benchmarking) sagt: `-sm row` ist TP (NCCL AllReduce)
- Agent #3 (Upstream) sagt: `-sm tensor` ist das volle Meta-Backend TP
- **Klärung**: `-sm tensor` (SPLIT_MODE_TENSOR=3) nutzt das Meta-Backend mit vollständigem TP. `-sm row` (SPLIT_MODE_ROW=2) nutzt die ältere Row-Split-Infrastruktur. Für TP+TQ: **`-sm tensor` ist korrekt**.

### K2: TP+TQ nicht bit-identisch zu Single-GPU+TQ
- Block-Indizes für `tq_derive_seed` sind lokal pro GPU, nicht global
- Ergebnisse sind KORREKT aber DIFFERENT zu Single-GPU
- **Impact**: Regressions-Tests müssen PPL-basiert sein (nicht bit-compare)

### K3: Deferred K-Quant + TP + iSWA
- Funktioniert laut Analyse korrekt (jede GPU konvertiert lokal)
- Risiko: SWA-Cache Eviction während Base-Cache noch im Staging
- **Empfehlung**: Monitoring-Log wenn State-Mismatch zwischen base/swa

---

## Benchmark-Strategie (Ready to Execute)

### Speed
```bash
llama-bench -m $MODEL -ngl 99 \
  -sm none,layer,tensor \
  -ctk f16,tq2_1,tq3_1,tq4_1 \
  -ctv f16,tq2_1,tq3_1,tq4_1 \
  -fa 1 -p 512,2048,8192 -n 128 -r 3 -o json
```

### Accuracy (3-Stufen-Test)
1. F16 KV Referenz: `llama-perplexity -sm none -ctk f16 -ctv f16`
2. Single GPU + TQ: `llama-perplexity -sm none -ctk tq2_1 -ctv tq2_1`
3. TP + TQ: `llama-perplexity -sm tensor -ctk tq2_1 -ctv tq2_1`
- delta_tp sollte ~0 sein (AllReduce verlustfrei in FP32)
- Wenn delta_tp > 0: BF16-Kompression im AllReduce prüfen

### Memory
```bash
watch -n 0.5 nvidia-smi --query-gpu=index,memory.used --format=csv
```
Plus `llama_memory_breakdown_print()` (automatisch am Ende von llama-perplexity)

---

## Architektur-Kompatibilität

| Modell | TQ allein | TQ + TP | Status |
|--------|-----------|---------|--------|
| Llama-3.1-8B | OK | OK | Ready |
| Qwen3.5-35B (Hybrid/MoE) | OK | OK (nicht in Blocklist) | Ready |
| Gemma4-26B (iSWA) | BROKEN (Garbage) | Blocked by Bug #1 | Fix nötig |
| DeepSeek2 (MLA) | N/A | GEBLOCKT in Blocklist | Nicht supportet |

---

## Nächste Aktionen (priorisiert)

1. **LEGION-Nachricht** an distillery-claude: `LLAMA_ATTN_ROT_DISABLE=1` Test für Gemma4
2. **Gemma4 Rotation-Bug** fixen in llama-graph.cpp
3. **docs/tp-tq-design.md** schreiben (Correctness Proof, Hardware-Empfehlung)
4. **PCIe-Heuristik** anpassen (ggml-cuda.cu:1146)
5. **Benchmark-Suite** auf gpu00 laufen lassen
