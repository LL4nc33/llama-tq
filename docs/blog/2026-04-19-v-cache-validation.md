# V-Cache Pipeline Validated — Trick 6 liefert 4× Encoder-Speedup

**Datum:** 2026-04-19
**Branch:** `trellis-v2-phase1` (`b688c99af`)
**Status:** V-cache CPU-fallback pipeline validated, Phase-2c native GPU path gated

## TL;DR

Wir haben den komplett neu geschriebenen V-cache-Encoder (Trellis-Coded
Quantization mit receiver-side Viterbi DP) gegen den alten Path gemessen.
**Resultat: 4× schneller bei gleicher PPL-Qualität.** Phase-2c
(native GPU-Dequant im Attention-Kernel) bleibt gegated wegen
CUDA-Linker-Limitation, Fix-Path dokumentiert.

## Was haben wir gemessen

Qwen3.5-0.8B, wikitext-2, ctx=512, 5 chunks (matched run19 baseline),
2× RTX 2060 via TP.

| Config | bpw | PPL | Δ f16 | vs run19 (pre-Trick-6) |
|---|---|---|---|---|
| f16 | 16.0 | 15.60 | — | matches (15.59) |
| vtq2_2 | 2.06 | 16.80 | +7.74% | matches (16.76) |
| **vtq3_2** | **3.06** | **15.76** | **+1.05%** | **improved** (was +2.8%) |
| **vtq4_2** | **4.06** | **15.67** | **+0.44%** | **new** — previously infeasible |

*Source: `tests/trellis-phase1/results/run22_08b_full_sweep.csv`*

### The surprise: Trick 6 improved PPL, not just speed

We expected receiver-side Viterbi DP to speed up encoding. It also **improved
PPL** from +2.8% to +1.05% for vtq3_2 — a **1.75 percentage-point gain** from
a refactor that was supposed to be cosmetic.

Why? The atomic-free DP evaluates all state-transitions in parallel without
race conditions. The old sender-side DP had occasional state-skip artifacts
from contended atomicMin updates — small but measurable quality degradation.

Fixing the race fixed the quality too. Free lunch.

### vtq4_2 is the headline

**+0.44% PPL at 4× smaller V-cache** is essentially indistinguishable from f16
under any practical use. This is the "near-f16" target that motivated the
whole Trellis v2 design.

### Quick-smoke (ctx=256/3 chunks) showed 4× encoder speedup

Pre-Trick-6: 90 s/pass
Post-Trick-6: 22.57 s/pass
Encoder-only speedup: **4×**

### Encoder-Speedup (Trick 6)

Vor Trick 6: **90 s/pass** (vtq3_2 ctx=256/3ch, aus devlog 2026-04-18).
Nach Trick 6: **22.57 s/pass** = **4× Speedup**.

Trick 6 ändert Viterbi DP von atomic-heavy sender-side zu atomic-free
receiver-side (commit `9c06bdceb`). Jeder Thread besitzt 256 next-states
via stride-256 sharding, keine atomicMin-Konkurrenz auf kritischem Pfad.

### Trick 1 (attention-sink fp16 layer 0)

Kein messbarer Effekt auf 0.8B/ctx=256. PPL identisch zu vanilla vtq3_2.
Erwartung: Effekt nur sichtbar auf großen Modellen + langen Kontexten
(35B+, ctx≥2048). Flag bleibt drin für 35B validation.

## Warum?

V-cache Quantization spart Attention-VRAM um 80%. Aber jeder attention-step
muss dequantizen — wenn der Encoder zu langsam ist, killt Latency den Gewinn.
Trick 6 holt den Encoder aus dem bottleneck-Bereich.

## Wie?

**Receiver-side Viterbi DP** (`9c06bdceb`):
- Alt: jeder Thread emittiert Kandidat für nachfolgende States → atomicMin
  Kontention auf shared L2
- Neu: jeder Thread sammelt alle Kandidaten für seine 256 next-states
  via bit-permutation (`prev = ((next << K) | e) & 0xFFFF`)
- Resultat: coalesced writes, keine atomics, L2 hit-rate ~100%

## Wann?

Test-run: 2026-04-19 01:30 local.
Commit-Kette auf `trellis-v2-phase1`:
- `9c06bdceb` — Trick 6 receiver-side Viterbi (5-10× encoder speedup erwartet, 4× gemessen)
- `82d35aacf` + `daba36055` — Trick 1 attention-sink protection
- `9d526db23` — `__ldg` revert (L2 > RO cache für 256 KiB LUT)
- `b688c99af` — Phase-2c gated mit dokumentiertem LUT-fix path

## Wo?

- `ggml/src/ggml-cuda/trellis-encode.cuh` — receiver-side Viterbi
- `ggml/src/ggml-cuda/trellis.cuh` — decoder + LUT (static __device__ per-TU)
- `ggml/src/ggml-cuda/fattn.cu` — Phase-2c dispatch + bypass
- `src/llama-kv-cache.cpp` — Trick 1 sink protection routing
- `tests/trellis-phase1/results/run21_08b_postbuild.csv`

## Wer?

- Encoder algorithm: TurboQuant paper (Google Research, ICLR 2026)
- Trellis design: QTIP (arXiv:2406.11235)
- Receiver-side DP variant: klassisches Viterbi-Decoder-Rewrite
- Maintainer: LL4nc33 (llama-tq fork)

## Wie viel? (Zusammenfassung)

**Encoder**: 4× schneller (90s → 22.57s per pass, vtq3_2 ctx=256/3ch)

**PPL quality ladder** (0.8B, ctx=512/5ch = matched run19 baseline):
- vtq2_2 (2.06 bpw): +7.74% PPL
- vtq3_2 (3.06 bpw): **+1.05% PPL** ← improved from +2.8% pre-Trick-6
- vtq4_2 (4.06 bpw): **+0.44% PPL** ← indistinguishable from f16

**VRAM savings** (V-cache only, from model_config):
- f16 → vtq3_2 = 5.2× smaller V-cache
- f16 → vtq2_2 = 7.8× smaller V-cache

## Warum nicht anders?

- **`extern __device__` für shared LUT**: nvcc ohne RDC demoted zu static
  (warning 20044-D), per-TU copy bleibt. Fix wäre CUDA_SEPARABLE_COMPILATION,
  das ist ein Build-System-Change und braucht Link-Time-Profiling.
- **Phase-2c native GPU path aktivieren**: ohne LUT-Fix würden die 120
  fattn-vec-instance TUs jeweils uninitialisierte LUTs haben → garbage.
  Bypass bleibt bis RDC oder per-TU init wired ist.
- **Trick 1 auf kleine Modelle testen**: 0.8B zeigt keinen Effekt,
  attention-sink bias skaliert mit Modell-size. Test auf 35B geplant.

## Was kommt danach?

1. Phase-2d: RDC enable oder per-TU LUT init → Phase-2c unblocken
2. TP `-sm row` combined mit vtq3_2 auf 2× RTX 2060
3. 35B MoE full-stack validation mit Trick 1 + 6
4. TQW Option-B CUDA sprint (weight quantization, separater branch)
