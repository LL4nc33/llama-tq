# README Re-Framing Proposal

**Datum:** 2026-04-23
**Ziel:** README ehrlich neu positionieren — "Quality-focused fork" statt "Fastest llama fork"
**Prinzipien:**
- Kein Bragging
- Keine anderen Projekte schlecht machen
- Nur sagen: was nutzt es, warum, für wen
- Ehrliche Trade-offs zeigen
- Nicht versprechen was nicht messbar ist

---

## Vorgeschlagener Header (ersetzt die Top-Banner)

```markdown
# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

KV-cache Forschungs-Fork von [llama.cpp](https://github.com/ggml-org/llama.cpp). Fokus: **asymmetrische
Quantisierung** von K- und V-cache mit separaten Dequant-Pfaden im FlashAttention-Kernel.

**Für wen:** Nutzer die mit begrenzter VRAM große Modelle mit langem Kontext fahren wollen und
bereit sind minimale PPL-Kosten für signifikante KV-Einsparung zu akzeptieren. Speziell getuned
für NVIDIA Turing (CC 7.5) und später.

**Was es nutzt:**
- **TurboQuant** (Zandieh et al., [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) —
  Random Hadamard Transform + Lloyd-Max Codebook für K- und V-cache
- **Flash Attention** (standard llama.cpp Pfad, erweitert um asymmetrische K/V Dispatch)
- **Lloyd-Max Optimized Codebooks** — 1D-optimale Centroids für Laplace-verteiltes post-RHT Data
- **Deferred-V Quantization** — f16 staging bei Prefill, Bulk-Viterbi bei Prefill→Decode Transition
- **Attention Sinks Protection** — erste 4 Tokens in f16 (Streaming-LLM Methode)
- **ggml** Infrastruktur (type_traits, CPU fallback, CUDA FA Kernels)

**Was es nicht ist:**
- Kein Performance-Fork — TG-Durchsatz entspricht upstream llama.cpp. Der Fork spart VRAM, nicht Zeit.
- Keine Alternative zu vLLM/SGLang — Single-Node, keine Paged-Attention, kein Batching-Focus.
- Kein Tool für Consumer-GPUs neuerer Generationen — ungetestet auf Ampere/Hopper. sm_75 ist primary target.

**Aktueller Status:** Research-Grade. Produktion auf Qwen3.5-35B-A3B + 400k ctx funktional, aber
V-cache-Konfiguration ist workload-abhängig (siehe Trade-offs unten).
```

## Zweiter Abschnitt: Measured Results

```markdown
## Gemessene Ergebnisse

**Hardware:** 2× RTX 2060 12GB (CC 7.5, PCIe 3.0)
**Methodology:** wikitext-2 Perplexity, llama-bench TG128/PP512, nvprof Kernel-Profile

### Quality (KV-Cache Configs)

Auf 4 getesteten Qwen3.5/3.6 Modellen (0.8B bis 35B-A3B):

| Config | K | V | bpw | PPL Δ Range | Use Case |
|--------|---|---|:---:|---|---|
| Quality | q8_0 | vtq3_1 | 6.25 | +0.6% bis +2.5% | Default-Empfehlung |
| Balanced | q4_0 | vtq3_1 | 4.25 | (siehe Anmerkung) | Allgemein |
| Compact | q8_0 | vtq2_1 | 5.5 | +5.1% bis +10.0% | VRAM-knapp |
| Aggressive | q4_0 | vtq2_1 | 3.5 | +7.2% bis +10.4% | Langer Kontext |

*Anmerkung:* PPL-Messungen bei 3 wikitext-2 Chunks (statistisch rauschig). 64+ Chunk Re-runs pending.

### Throughput

Kein signifikanter Unterschied zu upstream llama.cpp bei gleichen K/V-Typen.
**Ausnahme:** VTQ V-cache auf 400k Kontext zeigt 5.5× TG-Regression gegenüber f16 V
(per-token V-dequant skaliert mit Kontextlänge im FA-Kernel). **Empfehlung:** VTQ V-cache
nur wenn VRAM-constrained — für VRAM-reichliche Deployments f16 V beibehalten.

### Bottleneck-Analyse (Turing sm_75, TG)

Per nvprof Profiling auf Qwen3.5-35B-A3B:
- mmvq (IQ2_XS expert matmuls): **28%** der Kernel-Zeit — primärer Bottleneck
- concat_f32 (KV append): 4.4%
- FA-vec Kernel: **nur 6.4%** der Kernel-Zeit
- Launch overhead: ~2.8% wall time

Das heißt: **weitere FA-Kernel-Optimierung bringt wenig**, der Hebel liegt bei den
Expert-Matmuls (MoE-Architektur), die bereits hoch-optimiert upstream sind.
```

## Dritter Abschnitt: Wann sollte ich das NICHT nutzen

```markdown
## Wann dieser Fork NICHT die richtige Wahl ist

- **Wenn VRAM kein Constraint ist:** upstream llama.cpp mit f16 KV ist einfacher und gleich schnell.
- **Wenn Latenz < 50ms/token wichtig ist:** VTQ V-cache erhöht per-token Dequant-Overhead bei
  langem Kontext. Nutze f16 V oder andere Forks.
- **Wenn Multi-GPU Skalierung priorität ist:** Dieser Fork macht keine speziellen Änderungen
  an der llama.cpp Split-Logik. vLLM/SGLang sind für Multi-Node-Inferenz ausgelegt.
- **Auf Ampere+ (CC 8.0+):** Ungetestet. Die sm_75-spezifischen launch_bounds und FA-Tuning
  sind für Turing kalibriert, nicht für neuere Tensor-Core-Generationen.
```

## Vierter Abschnitt: Roadmap-Realität

```markdown
## Roadmap-Realität

Stand 2026-04-23:

**Ausgeliefert:**
- KTQ1_1/2_1/3_1/4_1 — K-cache Quant-Typen
- VTQ1_1/2_1/3_1/4_1 — V-cache Quant-Typen (asymmetrische K/V via FA Dispatch)
- Deferred-V Quantization Infrastruktur
- Attention Sinks Protection
- Laplace-optimierte 2-bit Codebooks

**Active Research (ohne Garantie):**
- mmvq Tuning für IQ2_XS auf sm_75
- Trellis v2 (VTQ_2 Familie) — derzeit broken auf D=256
- C1 Streaming-Window — designed, nicht implementiert

**Verworfen nach Measurement:**
- Speculative Decoding — funktioniert nicht auf A3B MoE (expert saturation)
- VTQ_MIXED — dominated by VTQ3_1, nicht CUDA-portiert
- Calibrated Outlier Selection — marginal gain nach RHT

**Bewusst nicht auf Roadmap:**
- FA3-Port (sm_80+ Hardware nötig)
- Paged-Attention (Scope-Mismatch mit Fork-Ziel)
- Multi-Node-Inferenz
```

## Fünfter Abschnitt: Inspirationen

```markdown
## Inspirationen & verwendete Methoden

Dieser Fork ist eine Forschungs-Sammlung die mehrere Bausteine kombiniert.
Was davon wo einfließt:

### Kern-Foundation

- **llama.cpp** — Upstream-Fork. Unveränderte Runtime, Server, GGML-Infrastruktur.
  Alle nicht-KV-Cache-bezogenen Änderungen kommen von upstream und werden regelmäßig
  zurück-gemergt.

- **ggml** — Tensor-Library die type_traits erlaubt: unser VTQ-Dispatch geht über den
  Standard-ggml-Mechanismus, sodass CPU-Fallback, Quantize, Dequantize und set-rows
  automatisch funktionieren sobald die Typen registriert sind.

### Primäre Research-Methoden

- **TurboQuant** — Random Hadamard Transform (RHT) zur Gleichverteilung von Outliers,
  Lloyd-Max-Codebook für 1D-optimale Quantisierung. Wir implementieren die MSE-optimale
  Stage 1 (PolarQuant). Stage 2 (QJL Residual) haben wir evaluiert und für
  Attention-Workloads als ineffektiv befunden.

- **Flash Attention 2** — Der FA-Kernel auf dem unsere asymmetrische K/V-Dispatch-Erweiterung
  aufsetzt. Wir modifizieren die vec-Path für VTQ V-Cache Unterstützung.

- **Streaming-LLM / Attention Sinks** — Die ersten 4 Tokens einer Sequenz haben
  disproportionale Attention-Gewichte. Unser "Trick 1" hält diese in f16 statt quantisiert.

- **Laplace-optimale Codebooks** — Post-RHT Daten sind Laplace-verteilt, nicht Gauß.
  Wir fitten Centroids direkt an Laplace.

### Erwogene & dokumentierte Inspirationen

- **Trellis-Coded Quantization (TCQ)** — klassische Signalverarbeitungs-Methode.
  Wir haben eine Phase-1-Harness gebaut, VTQ_2 Familie daraus abgeleitet.
  Derzeit broken auf D=256, funktional auf D=128.

- **Paged Attention** — KV-Cache-Management via fixed-size Pages. Nicht direkt nutzbar
  weil Python/Triton-based und unser Stack reines C++ ist. Dokumentiert als mögliche
  Portierungs-Source.

- **Triton Autoresearch** — Autoresearch-Methode angewendet auf 2060 FA-Kernel.
  8 Experimente, E11 cached-decode erreicht 112 GB/s (14× über naive).

- **CUDA Graphs** — für Launch-Overhead-Reduktion. llama.cpp hat das bereits upstream,
  default enabled.

- **Speculative Decoding** — Bereits in upstream llama.cpp implementiert.
  Wir haben geprüft ob es auf unsere A3B MoE Config passt — expert-saturation pathology
  macht es auf dieser Architektur ineffektiv.

### Measurement-Infrastruktur

- **nvprof / Nsight Systems** für Kernel-Profiling
- **wikitext-2** Dataset für PPL-Messung
- **sentence-transformers** für MSE→PPL Pipeline Validation

### Was dieser Fork selbst beisteuert

Der eigenständige Beitrag beschränkt sich auf:
1. Asymmetrische K/V-Cache-Dispatch im FA-Kernel (K und V mit unterschiedlichen Quant-Typen)
2. VTQ V-Cache Familie — VTQ1_1/2_1/3_1/4_1 als registrierte ggml-Typen
3. Deferred-V Quantization Infrastruktur — f16 staging bei Prefill, Bulk-Viterbi Transition
4. Reproduzierbare MSE→PPL Pipeline — Python-Harness, real-data Validation
5. Measurement-first Methodology — jede Optimierung profile-gated vor merge

Alles andere steht auf den Schultern der oben genannten Arbeiten.
```

---

## Umsetzungs-Schritte

Wenn User die Richtung absegnet:

1. `README.md` top banner ersetzen (Zeilen 1-30) mit neuem Header
2. Neuen Abschnitt "Wann NICHT nutzen" nach Quick Start einfügen
3. Bestehende Benchmark-Tabellen behalten — sie sind schon ehrlich
4. "Known Issues" Sektion bleibt, sie ist transparent
5. Neuen "Roadmap-Realität" Abschnitt unten einfügen
6. Credits-Sektion am Ende erweitern

**Nicht zu entfernen:**
- Die detaillierten PPL-Ranges (ehrlich und wertvoll)
- Die Deployment-Warnung zu VTQ3_1 Regression
- Die technischen Deep-Dive Abschnitte

**Zu entfernen:**
- Jegliche vergleichenden Speed-Claims
- Marketing-Terme wie "Fast" / "Optimal" ohne Belegzahl
- Sätze die andere Forks/Tools abwerten

## Offene Frage

Soll ich direkt den README umbauen oder auf OK warten?
