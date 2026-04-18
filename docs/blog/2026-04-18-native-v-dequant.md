# Native V-Dequant auf der GPU — Weg vom CPU-Fallback

**Datum:** 2026-04-18
**Branch:** `trellis-v2-phase1`
**Status:** Build in progress, smoke-test pending

## TL;DR (für alle)

Wir quantisieren den V-Cache (Value-Teil der Attention) auf ~3 bit statt 16 bit.
Bisher musste das Modell die Daten dafür kurz zurück auf die CPU schicken —
langsam. Jetzt läuft der komplette Pfad auf der GPU. Erwartete
Beschleunigung: **1.5-2× bei langen Chats** (wo der V-Cache groß wird).

## Was? (What)

Phase-2c des Trellis v2 Projekts: native Flash-Attention-vec Kernel für
die neuen Typen `VTQ2_2`, `VTQ3_2`, `VTQ4_2`. Diese dequantizen den
Trellis-kodierten V-Cache direkt im GPU-Attention-Kernel, ohne
Roundtrip über den Host.

## Warum? (Why)

Das alte Setup hatte einen Bypass: wenn V-Cache vom Typ `VTQ_2` war,
fiel `flash-attention` auf den CPU-Path zurück. Jeder Attention-Step
brauchte dann GPU → CPU → GPU Datentransfer. Auf langen Kontexten
(4k+) wird das zum dominanten Bottleneck.

## Wie? (How)

Drei Änderungen:

1. **LUT Refactor** — die 256 KiB Trellis-Lookup-Tabelle war als
   `static __device__` pro Translation-Unit definiert. Bei ~120
   fattn-vec-instance TUs bekam jeder seine eigene uninitialisierte
   Kopie → garbage output. Fix: `extern __device__` mit einer zentralen
   Definition in `trellis.cu`.

2. **Per-Element Decoder** — `trellis_decode_element<K>(start_state, d, qs, j)`
   repliziert das Shift-Register für beliebige Indizes, damit FA-vec
   pro Thread beliebige V-Elemente lesen kann.

3. **Dispatch Matrix** — 18 Template-Instances pro VTQ_2 Type
   (K ∈ {F16, Q8_0, KTQ1-4_1} × D ∈ {64, 128, 256, 512}).

## Wann? (When)

- `7b2f94975` — Phase-2c scaffolding (gated bypass aktiv)
- `9c06bdceb` — Trick 6: receiver-side Viterbi encoder (atomic-free)
- `82d35aacf` — Trick 1: attention-sink fp16 protection
- `1520cb117` — **LUT extern refactor — unblocks native path**

## Wo? (Where)

- `ggml/src/ggml-cuda/trellis.cuh` — decoder, LUT declaration
- `ggml/src/ggml-cuda/trellis.cu` — LUT definition, encoder pool
- `ggml/src/ggml-cuda/fattn.cu` — dispatch matrix
- `ggml/src/ggml-cuda/fattn-common.cuh` — `dequantize_V_*` templates

## Wer? (Who)

- Maintainer: **LL4nc33** (llama-tq fork)
- Paper-Basis: **TurboQuant** (Google Research, ICLR 2026)
- Trellis-Design: **QTIP** (arXiv:2406.11235)
- Upstream: **llama.cpp** team (MIT)

## Wie viel? (How much) — bisher gemessen

### Qwen3.5-0.8B (dev-loop, wikitext-2)

| Config | bpw  | PPL   | Δ f16   |
|--------|------|-------|---------|
| f16    | 16.0 | 15.59 | —       |
| vtq3_2 | 3.06 | 16.02 | +2.76%  |
| vtq2_2 | 2.06 | 16.76 | +7.50%  |

*Source: `tests/trellis-phase1/results/run19_08b_dev_sweep.csv`*

### TQW2 Python Validation (weights, separater Branch)

| Variant | bpw    | MSE vs IQ2_XXS |
|---------|--------|----------------|
| TQW2    | 2.0625 | **−62-65%**    |
| TQW3    | 3.0625 | **−89%**       |

*Source: `tests/trellis-phase1/results/run20_tqw2_mse.csv`*

### Qwen3.5-35B MoE (deployment baseline, TQ v5)

| K type   | V type   | KV VRAM | PPL Δ f16 |
|----------|----------|---------|-----------|
| f16      | f16      | 100%    | —         |
| TQ2_1    | q4_0     | ~22%    | +1.2%     |

*Deployed on gpu00:8791 with 200K context/slot.*

## Warum nicht anders? (Alternatives rejected)

- **`__ldg` auf LUT** — Turing RO-cache ist 48 KiB, LUT ist 256 KiB.
  `__ldg` forciert ~20% hit rate mit L2-refill-stalls auf atomicMin
  critical path. Plain global loads gehen direkt in L2 (~100% hit nach
  warmup). **Reverted in `9d526db23`**, 2-3× Encoder-speedup zurück.

- **KTQ3_1 als Weight-Alias** — KTQ speichert Hadamard-Domain, nutzt
  FA-kernel "push FWHT to Q" trick. Für mul_mat mit physical-domain
  activations fehlt inverse-FWHT dequant. **Abandoned**, auf separaten
  Branch mit eigenem physical-domain TQW3 format ausgelagert.

- **Per-model RHT seed calibration (Trick 3)** — Kurtosis-sweep zeigte
  std 0.0044 ≈ noise floor 0.0048. Post-RHT marginals sind bereits
  near-Gaussian. Per-model salt cal würde unter Messauflösung liegen.
  **Abandoned** mit dokumentiertem null-result.

## Was kommt danach?

1. Smoke-test Phase-2c auf 0.8B (gate: PPL ±0.05 vs run19 baseline)
2. TG-benchmark nativer vs CPU-fallback (echte ms/token Zahlen)
3. Trick 1 validation (attention-sink fp16 in layer 0)
4. TP `-sm row` combined mit VTQ_2 auf 2× RTX 2060
5. 35B MoE full-stack validation
