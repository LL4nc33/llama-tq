# Qwen3.6-35B-A3B Architecture-Findings für llama-tq

## TL;DR — alles bisherige neu kalibrieren

**Bisherige annahme:** 35B-A3B ist standard MoE transformer mit 40 attention layers.
**Realität:** Hybrid SSM/Attention. Nur **10/40 layers** haben KV-cache.

## Architektur-essentials

| Parameter | Wert |
|---|---|
| Layers | 40 total |
| Layer-mix | 30× DeltaNet (SSM) + 10× GatedAttention |
| Pattern | 3:1 ratio, also `10 × (3×Delta + 1×Attn)` |
| Active params | 3B per token (35B sparse) |
| Experts | 256 routed + 1 shared, top-8 routed + 1 shared = 9 aktiv |
| moe_intermediate_size | 512 (winzig) |
| head_dim (attention) | **256** (ungewöhnlich, doppelt so groß wie Qwen3) |
| GQA-ratio | 16:2 = 8 |
| Partial RoPE | nur 64 von 256 dims rotated (factor 0.25) |
| Native ctx | 262K |
| Extended ctx (YaRN) | 1M |

## Was das für unsere VTQ-V optimierung heißt

**Reality check:** wir optimieren seit gestern VTQ2_1 V-dequant für 35B-A3B PP@4k. Aber:
- KV-cache existiert nur in 10/40 layers
- VTQ-V dequant kommt nur 25% der zeit vor
- **Wenn wir VTQ-V auf -5% statt -25% bringen, bringt das maximal +5% PP@4k** (nicht +25%)

→ wir haben **3 stunden** in einen pfad investiert der maximal +5% liefern kann.

## Die echten bottlenecks (basierend auf upstream issues)

### 1. MoE GEMM `ggml_mul_mat_id` — vermutlich >60% compute bei prefill
- 256 expert routing, top-8 select, gather, dispatch
- Issue #19672: crash bei großen MoE
- Issue #19345: llama.cpp 40% langsamer als vLLM bei MoE Coder Next
- Issue #20757: expert-streaming memory-bandwidth-bound bei TG

### 2. DeltaNet kernel — 75% der layers
- Custom kernel, jung (PR #19408)
- Issue #22320: 78-81% GPU utilization auf RTX 4090 mit MoE+SSM
- Audit-target: vergleichen mit rasbt/LLMs-from-scratch reference

### 3. Hybrid checkpoint broken — multi-turn killer
- Issue #22384: jeder multi-turn = full re-prefill
- 2-line patch in server-context.cpp

### 4. Partial-RoPE not exploited
- 64 dims rotated + 192 dims static
- Aktuell quantisieren wir alle 256 dims identisch
- Non-rotated dims sind "stiller" → niedrigere effective bpw möglich

### 5. MoE-pin-experts auf 35B-A3B
- Existing flag bringt +3.3% TG auf 80B-IQ2
- Auf 35B-A3B noch nicht gemessen / etabliert

## Strategie-revision

Statt VTQ-V kernel weiter zu polish (max +5% bei 25% der layers):

### Hebel A — MoE-expert-cache + pinning (1-2 tage)
- Two-tier GPU+RAM cache mit LRU/LFU für hot experts
- 20-30% experts decken 80% activations
- Expected: **+10-20% TG** (memory-bandwidth bound befreit)
- Issue #20757 als roadmap

### Hebel B — Checkpoint-fix portieren (30 min)
- 2-line fix aus Issue #22384
- Eliminiert re-prefill bei multi-turn
- Expected: **+200-500% UX** für chat-flows
- Trivialer win, sofort umsetzbar

### Hebel C — Partial-RoPE-aware KTQ (2-3 tage)
- Erste 64 dims (rotated) → normales KTQ2_1
- Letzte 192 dims (non-rotated) → niedrigere bpw quant (KTQ1_1?)
- Expected: **+5-10% PP/TG** plus kleinerer KV-cache

### Hebel D — DeltaNet kernel audit (1-2 tage)
- 75% der layers, vermutlich unoptimiert
- Wenn dort 20% drin ist → riesig
- Untested, hohes risiko

### Hebel E — MoE routing fusion (1 tag)
- 256-softmax + top-8 + gather in 1 kernel
- Eliminiert kernel-launch-overhead
- Expected: **+3-5% prefill**

## Konkrete next steps

**Reihenfolge nach ROI:**
1. Hebel B (checkpoint-fix) — 30 min, massive UX-win, sofort
2. **Hebel A (expert-pinning bench + tune)** — 2-3 stunden bench-sweep
3. **Hebel D (DeltaNet audit)** — wo sitzt die zeit wirklich?
4. Hebel C (partial-RoPE quant) — long-term gewinn
5. Hebel E (routing fusion) — kleiner aber sicher

**Sofort als experiment:** `--moe-pin-experts` auf 35B-A3B bench-sweep mit verschiedenen N=experts-pinned.

## Verworfen

- ❌ DeltaNet state quantisierung (accuracy-sensitiv, recurrent error)
- ❌ Speculative decoding auf A3B (RTX3090 19-config-bench zeigt -3 bis -12%)
- ❌ MTP (multi-token prediction) — sub-case von spec-decode, gleicher pfad
- ❌ Weitere VTQ-V kernel-polish ohne reality-check der hebel-größe

Sources: [research-agent-output 2026-05-13](memory/agent_acb3938296c67af45)
