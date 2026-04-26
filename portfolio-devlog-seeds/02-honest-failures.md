---
slug: "honest-failures-qjl-pie-spec-decode"
title: "Honest Failures: QJL, Pie-PCIe, Spec-Decode auf MoE"
date: "2026-04-26"
status: "shipped"
tags: ["turboquant", "failure-log", "kv-cache", "negative-results"]
hero_number: "3 dead ends"
---

# Was nicht funktioniert hat (und warum)

Der llama-tq Fork hat eine "what works"-Liste — aber genauso wichtig ist die "what didn't"-Liste. Das ist was uns von Marketing-Claims trennt: wir wissen wo die Wände sind weil wir dagegen gelaufen sind.

## 1. QJL — eingebaut, dann wieder entfernt

**Was:** QJL (1-Bit Quantized JL Transform, [arxiv:2406.03482](https://arxiv.org/abs/2406.03482)) als 1-bit-residual-correction zusätzlich zu PolarQuant K + Trellis V.

**Warum probiert:** Theoretisch sollte QJL inner-product unbiasedness sicherstellen — wichtig bei aggressiven sub-3-bit Quants, wo softmax-Verteilungen schief werden können.

**Status v1-v4:** drin, mit eigenem r_norm Field im KTQ Block-Struct + extra Philox-Round.

**Warum rausgenommen in v5:** Drei Gründe.

1. **+1 bit pro coordinate** ist teuer. 3.5 bpw KTQ2_1 wird 4.5 bpw — da kann man auch direkt KTQ3_1 nehmen ohne den QJL-Overhead.
2. **Norm correction scalar** (closed-form, deterministic) liefert vergleichbare quality-recovery ohne extra-bit. v5 macht das.
3. **Empirisch absorbiert softmax die Bias.** HellaSwag-200 auf Qwen3.6-35B-A3B mit reinem KTQ2_1+VTQ2_2 (kein QJL) = 83.5% [77.7%, 88.0%]. Vergleichbar mit f16 baselines auf kleineren Modellen.

**Was vLLM uns demonstriert:** [vLLM PR #38280](https://github.com/vllm-project/vllm/pull/38280) baute QJL ein und musste feststellen dass ihre Implementierung kaputt war (PyTorch/Triton butterfly mismatch beim Hadamard). Ohne QJL produzieren ihre 3-bit/2-bit Modi "garbage output". PR ist closed.

**Lesson:** QJL ist theoretisch elegant, praktisch fragil. Wenn 2-bit gut genug performt ohne, ist `simpler` besser als `correcter`.

**Code:** Removal happened in v5 commit history under tag `polarquant-v5`. Search `git log --grep=QJL`.

---

## 2. Pie — performance-transparent KV swapping über PCIe

**Was:** [Pie paper (arxiv:2411.09317)](https://arxiv.org/abs/2411.09317) — FIFO-Queue-basierter KV-Cache-Swap zwischen GPU und CPU RAM. Nutzt Layer-by-Layer-Vorhersagbarkeit für prefetch. Behauptet 9.4× über FlexGen, 1.9× über vLLM.

**Warum probiert:** Phase 5 sollte CPU-RAM für KV-Cache nutzbar machen. Das würde 200K-context auch auf 12GB single-GPU ermöglichen.

**Reality-Check Microbench:** Wir hatten asymmetric PCIe (GPU0 x16, GPU1 x4). Microbench mit `cudaMemcpyAsync`, 256MB chunks, 20 iter avg:

| GPU | PCIe link | Bandwidth | 256MB transfer time |
|---|---|---|---|
| GPU0 | x16 | 13.14 GB/s | 20.4 ms |
| GPU1 | x4 | **1.44 GB/s** | **186.5 ms** |

Decode-Budget @ 36 t/s auf 80B = 27.7 ms/token.

**GPU1 prefetch eines 256MB Experts braucht 186 ms — 6.7× über decode budget.** Pie's fundamentale Annahme `tswap ≤ tcompute` ist auf unserer Hardware verletzt.

**Pie braucht NVLink (419 GB/s)** wie auf GH200 spezifiziert. PCIe-x4 ist 290× langsamer.

**Lesson:** Hardware-Mathematik beachten bevor man Software-Architektur baut. Pie ist nicht "auf Consumer-GPUs portable" — Pie ist "auf NVLink portable". Microbench in 30 Minuten hätte uns ein Wochenende Implementierung erspart.

---

## 3. Speculative Decoding auf MoE A3B

**Was:** Standard speculative decoding mit kleinem Draft-Model (Qwen3.5-0.8B) das tokens speculativ generiert die das große Target-Model (Qwen3.6-35B-A3B) in batched verify-step prüft. Theoretischer Speedup 2-3× bei hoher accept-rate.

**Status:** llama.cpp upstream hat `--model-draft` flag bereits implementiert. **Unser Target hatte fertige integration ready.**

**Warum es nicht funktioniert:**

- **Public benchmark** (thc1006, RTX 3090, post-PR#19493, 2026-04-19) auf **Qwen3.6-35B-A3B** mit Qwen3.5-0.8B draft bei **100% accept-rate**: **-10.8% Regression**, kein Speedup.
- **Root cause:** A3B routes 8-of-256 Experts pro Token. Draft-Batch K=4-8 ist unter expert-saturation threshold (~94 tokens). Jeder zusätzlich akzeptierte Token = mehr unique experts geladen = mehr Memory-Traffic der die Verification-Savings cancelt.
- **Hardware:** Wir haben 2× RTX 2060 12GB, schlechter als die RTX 3090 in dem Benchmark → noch tieferer Regress erwartet.

**Auf welchen Modellen funktioniert spec-decode trotzdem:**
- Dense Modelle (Qwen3.6-27B): theoretisch ja, separate Untersuchung
- A10B+ MoE (122B class): expert-saturation threshold höher, vielleicht
- Compute-bound regimes (große batches, prefill): andere physics

**Lesson:** "Spec-decode = 2× speedup" ist ein **dense-model claim**, kein MoE claim. Für unseren primary deploy (Qwen3.6-35B-A3B) ist das eine Sackgasse, kein Quality-Issue.

---

## Was diese Failures lehren

Drei Patterns:

1. **Hardware-Realität schlägt Algorithm-Eleganz** (Pie). Microbench first.
2. **Architecture-Specific Pathologies** (spec-decode auf MoE). "It works on dense models" ≠ "it works on your model".
3. **Simpler beats correcter wenn quality-gate bestanden** (QJL). Norm correction ohne extra bit reichte.

Was übrig bleibt: PolarQuant K + Trellis V + Sparse V Dequant + Asymmetric Pareto + Adaptive Layer Split. Alles deployed live, alles verifiziert.

---

## Sources

- [QJL paper (arxiv:2406.03482)](https://arxiv.org/abs/2406.03482)
- [Pie paper (arxiv:2411.09317)](https://arxiv.org/abs/2411.09317)
- [Speculative decoding spec in llama-tq](https://github.com/LL4nc33/llama-tq/blob/master/docs/plans/2026-04-23-speculative-decoding-spec.md)
- [vLLM TurboQuant PR #38280 (closed)](https://github.com/vllm-project/vllm/pull/38280)
