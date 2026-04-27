# TG-Optimization Session — Reality Check

**Datum:** 2026-04-23
**Scope:** Versuch TG über 67.65 tok/s Baseline zu heben auf Qwen3.5-35B-A3B-IQ2_XS / 2× RTX 2060
**Verdict:** Baseline ist nahe am Hardware-Limit. "Schnellster llama fork" ist kein realistisches Ziel mehr.

---

## Die Hoffnung

Session startete mit der Annahme dass es noch deutliche TG-Verbesserungen gibt:
- C1 Streaming-Window (5-15% erhofft)
- FA-Kernel Quick-Wins (launch_bounds, EXP2 softmax, Q-prescale — zusammen 10-30% erhofft)
- CUDA Graphs (25-40% erhofft wenn nicht aktiv)
- Speculative Decoding (2-3× erhofft)
- Custom 2060-Kernel (YouTube-inspiriert, 2× erhofft)

## Die Realität (durch echte Messungen)

### Was NICHT der Bottleneck ist

| Kandidat | Tatsächlicher Anteil | Verdict |
|---|---|---|
| **FA-vec Kernel** | 3.7% GPU / 6.4% kernel time | **NICHT bottleneck** — weitere FA-Opt sinnlos |
| **Launch overhead** | ~2.8% wall (108k launches/s × 4µs) | Marginal |
| **CUDA Graphs disabled** | FALSE — **bereits enabled by default** | Kein hidden gain |
| **V-cache dequant** (C1 Target) | 0% — wir nutzen f16 V in prod | C1 wäre für VTQ-V, nicht f16 |

### Was DER Bottleneck ist

| Kernel | % kernel time | Note |
|---|---|---|
| **mmvq (IQ2_XS expert matmuls)** | **~28%** | A3B MoE expert matmuls |
| concat_f32 (KV append) | 4.4% | |
| k_get_rows (MoE gather) | 2.6% | 8060 calls/s |
| FA-vec | 6.4% | klein |

**28% mmvq ist der einzig wirkliche Hebel.** Und mmvq ist schon **hoch optimierter upstream code** — die Luft nach oben ist begrenzt auf sm_75.

### Was Speculative Decoding betrifft

Public benchmark (2026-04-19, RTX 3090, Qwen3.6-35B-A3B + Qwen3.5-0.8B draft):
**135.7 → 121.1 tok/s = REGRESSION -11%** bei 100% accept rate.

Ursache: MoE expert-saturation pathology. Draft batches bleiben <94 tokens, triggern neue expert loads die mehr kosten als verify-savings. Auf 2× 2060 (weniger memory bw als 3090) wäre das noch schlechter.

**→ Speculative decoding ist auf A3B MoE architekturbedingt tot.**

## Was wir heute gemacht haben

1. ✅ 4 parallele Research-Agents dispatched
2. ✅ Video-Transcripts (Tone B Studio custom kernel, Meta torch.compile) → keine anwendbaren Tricks
3. ✅ HuggingFace Kernels Hub geprüft → nicht für C++/ggml nutzbar
4. ✅ Profile gelaufen (nvprof) → Bottleneck-Map erstellt
5. ✅ 3 dead-ends identifiziert und dokumentiert:
   - FA-vec micro-ops (zu klein)
   - CUDA Graphs disable-fix (schon an)
   - Speculative decoding (MoE-broken)
6. 🔄 mmvq optimization läuft (der einzig verbleibende 28% Hebel)

## Was noch versucht werden kann

### Realistische Optionen

1. **mmvq IQ2_XS tuning** — agent running, expected: +5-10% TG wenn was findbar
2. **ncu mit sudo** — würde L2 hit-rate + DRAM BW zeigen, könnte neue hebel aufdecken
3. **Wechsel weg von A3B zu dense Qwen3** — dann würde speculative decoding wieder funktionieren, aber anderer Trade-off (langsamere aktive weights, schnellere TG)
4. **IQ3_XXS statt IQ2_XS** — andere expert quant, andere mmvq path, könnte schneller sein

### Unrealistische Optionen (für 2060)

- **"Schnellster Fork"** — ist wahrscheinlich **schon erreicht** für diese Hardware + Model-Config. Upstream llama.cpp auf gleichem Stack wäre nicht schneller.
- **Flash Attention 3** — sm_80+ only (Ampere+), nicht für Turing
- **Custom mega-kernel** — würde Wochen kosten, realistisch 10-20% upside, nicht die erhofften 2×

## Der Quality-Angle

Was wir **tatsächlich besser haben** als upstream:

1. **TurboQuant KV-cache v5** — KTQ2_1/KTQ3_1/KTQ4_1 (K-side) messbar besser als upstream q-cache
2. **Deferred-V quantization** — prefill speedup infrastructure
3. **Attention sinks protection** — hybrid cache layout
4. **VTQ V-cache family** — V-side RHT+Lloyd-Max (deployment blocked by 5.5× TG regression)
5. **Research infrastructure** — Python harness, trellis framework, Laplace codebook optimization

**Das ist Quality, nicht Speed.** Für den Speed-Claim braucht's:
- Entweder andere Hardware-Generation (Ampere+)
- Oder einen komplett anderen Architecture-Angle (speculative + dense model, oder DeepSeek-V3 style MLA)

## Ehrliches Framing für README / Marketing

**Nicht mehr:**
- "Fastest llama fork"
- "2× speedup over upstream"

**Ja:**
- "Quality-focused: lower memory footprint at same/higher PPL"
- "Production-validated TurboQuant KV-cache for Turing+"
- "Research-grade: reproducible MSE-to-PPL pipeline, Laplace-codebook calibration"
- "Matches upstream speed, beats on memory + quality"

## Next Session

Wenn mmvq-agent keine signifikanten Gains findet:
1. Session closen, realistic re-framing in README
2. Pivot zu **quality expansion**: dense model support (Qwen3 dense) für speculative
3. Oder: pivot zu **broader hardware**: port TQ zu Ampere/Hopper wo mehr Kernel-Opts möglich sind
