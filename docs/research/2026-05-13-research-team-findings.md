# Research Team Findings — 2026-05-13 morning

Drei spezialisierte agents wurden parallel beauftragt während ich an performance-tests
gearbeitet habe. Hier die zusammenfassung der drei zentralen erkenntnisse.

## Agent 1: q4_0 KV dispatch path (researcher)

**Frage:** Warum ist KTQ+VTQ -40% PP slower als q4_0/q8_0 obwohl alle quantisiert?

**Antwort:** Es gibt **KEINEN magischen direkt-kernel-pfad** für q4_0. Beide nutzen
**denselben pre-pass dequant** in `launch_fattn` (`fattn-common.cuh:929`):
- q4_0/q8_0: allociert f16 scratch + `ggml_get_to_fp16_nc_cuda(K->type)` ruft
  `dequantize_block_q4_0_nc_cuda`
- KTQ-split-wrapper (`fattn-mma-ktq.cu:23-44`) macht **GENAU das gleiche** —
  duplikation ohne mehr-cost
- Echter unterschied: der **dequant-kernel selbst** ist teurer:
  - q4_0: 1 thread → 8 outputs (256 elems/warp), 1 vectorized load
  - KTQ: 1 thread → 1 output (32 elems/warp), FWHT shuffle, sign-decode

**Implikation:** der bottleneck ist nicht "scratch vs inline", sondern **dequant-kernel-throughput**.

## Agent 2: SageAttention SM_75 migration potential (ask)

**Frage:** Kann SageAttention's INT8 QK^T helfen unsere KTQ-K effizienz zu steigern?

**Antwort:** **HOCHRISKANT, nicht realistic** unter 4-6 wochen.
- SageAttention's K-smoothing kollidiert mit unserer RHT (Hadamard-rotation) —
  beide spreaden outliers, also macht smoothing nach RHT keinen sinn
- Würde dual-precision Q-tensor brauchen (f16 für PV, int8 für QK)
- INT8-FWHT verliert precision → KTQ-quality bricht
- Realistischer end-to-end speedup: nur **+15-25%**, nicht der claim'd 2.1×

**Bessere empfehlung des agents:** Inline VTQ-V dequant in fattn-mma-f16 kernel.
**Erwartete recovery:** 60-70% des f16-gaps ohne accuracy-risk.

**Quote des agents:** "INT8 attention auf einem fork der schon hadamard-rotiert
ist, ist wie Tupperdosen in eine Tupperdose stecken."

## Agent 3: Tensor Parallelism realistic? (ask)

**Frage:** Warum ist `-sm tensor` -61% PP schlechter als layer-split bei uns?

**Antwort:** **Tensor-Parallelism ist DEAD-END für unsere hardware**:

1. **AllReduce pro layer × token** — bei dual-GPU mit x16+x4 PCIe asym ist der
   ring durch x4 limitiert (~6 GB/s real). Layer-split braucht nur 1× P2P pro
   layer-boundary
2. **PR maintainer Johannes Gaessler selbst** sagt: "synchronization overhead may
   be prohibitive for fast GPUs with slow interconnect running small, sparse models"
3. **Issue #22391** zeigt EXAKT unser symptom: TP auf Qwen3.6-35B-A3B erzeugt
   "endless `/////`" output — verified by other users
4. **Auf 2× RTX 4090 PCIe 4.0 x16** war TP nur **0.39-1.72×** layer-split, also
   selbst NVLink-äquivalent nicht garantierten win

**Empfehlung:** Layer-split bleiben. Unser +50-74% PP layer-split ist **besser
als die meisten reports**. Tensor-parallelism nicht weiter pursuen.

**Quote:** "Tensor Parallelism ohne NVLink ist wie ein Ferrari mit Anhängerkupplung."

## Konvergenz der findings

Alle drei agents kommen zum gleichen schluss aus verschiedenen perspektiven:

> **Der KTQ-K dequant ist nicht das problem. Der bottleneck ist VTQ-V.**

Verifiziert durch bench (siehe `2026-05-13-vtq-v-is-bottleneck.md`):
- KTQ2_1 K + f16 V = -0.8% vs f16 (essentially free)
- KTQ2_1 K + VTQ2_1 V = -26% PP (massive regression)

## Action plan

1. ✅ Eliminate split-wrapper duplication (commit `f46e9626f`) — +1.9% measured
2. ✅ Multi-warp-per-CTA KTQ dequant (commit `ff917bc33`) — flat (not the bottleneck)
3. ⏳ Multi-warp-per-CTA VTQ dequant + pre-scaled CB (commits `e1f27e9b5`, `356042ff9`) — BUILDING
4. 📋 Future: 8-elements/thread VTQ dequant kernel (RPCS3-style "auto" amortization)
5. 📋 Future: vectorized 8-byte qs-load in decoder (eliminate 32× 1-byte gather)

## Sources

- [llama.cpp PR #19378 — backend-agnostic tensor parallelism](https://github.com/ggml-org/llama.cpp/pull/19378)
- [Issue #22391 — Qwen3.6-35B-A3B TP crash](https://github.com/ggml-org/llama.cpp/issues/22391)
- [SageAttention paper (arxiv 2410.02367v5)](https://arxiv.org/html/2410.02367v5)
- [SageAttention-SM75 Turing fork](https://github.com/XUANNISSAN/SageAttention-SM75-path)
- [POD-Attention ASPLOS 25](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf)
