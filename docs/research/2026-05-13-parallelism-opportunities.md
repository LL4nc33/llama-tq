# Parallelismus-Opportunities im llama.cpp/llama-tq Flow — 2026-05-13

User's "Wien-analogie" als optimization-framework:
> Manchmal ist zu Fuss/öffis (parallel kleine wege) schneller als das auto
> (sequenziell schneller theoretisch, aber praktisch gestaut)

## Aktueller flow (überwiegend sequenziell)

```
CPU: token-prep
    ↓
GPU: prefill batch (one CUDA stream)
    ↓
  layer 0 → layer 1 → ... → layer N (sequenziell pro batch)
  RMSnorm → QKV-proj → attn → out-proj → MoE-gate → expert FFN → reduce
    ↓
GPU: KV-write
    ↓
CPU: sampler (top-k / top-p / min-p / temp / repetition-penalty seriell)
    ↓
CPU: next-batch-prep
    ↓
[loop back to GPU prefill]
```

## Parallelismus-kandidaten (sortiert nach ROI)

### Tier 1: HIGH ROI (tage, nicht wochen)

#### 1.1 Mmproj-encoding parallel zu text-prefill
**Wo:** Vision-input → CLIP-style image-encoder → token-injection → text-prefill
**Aktuell:** sequentiell auf einem stream
**Idee:** image-encode auf GPU1 || text-context-prep auf GPU0
**Impact:** vision-heavy workloads (chat mit bildern) → -100ms latency
**Aufwand:** ggml-scheduler stream-hinting

#### 1.2 Fused GPU-sampler kernel
**Wo:** CPU-sampler nach jeder generation
**Aktuell:** top-k → top-p → min-p → temp → repetition-penalty seriell auf CPU
**Idee:** single fused CUDA kernel auf logits-vector
**Impact:** TG +5-15% wenn sampling stack aggressiv (alle 5 filters aktiv)
**Aufwand:** 1 neuer kernel, dispatch-changes
**Status:** `--backend-sampling` ist erste version davon, könnte vollständiger sein

#### 1.3 Speculative cache-write
**Wo:** nach token T generiert, KV-cache write blockt T+1 compute-start
**Aktuell:** synchroner write
**Idee:** async-write auf separatem stream, T+1 compute startet sofort
**Impact:** TG @ langem ctx +3-7%
**Aufwand:** ein cudaStream split + sync-token zwischen write und next-read

### Tier 2: MEDIUM ROI (1-2 wochen)

#### 2.1 Layer-internal stream-overlap (out-proj || next-layer RMSnorm)
**Wo:** innerhalb eines transformer layer
**Idee:** während current-layer out-proj fertig läuft, kann next-layer RMSnorm vom previous output starten (input-pipeline)
**Impact:** prefill +10-20% wenn out-proj nicht trivial
**Aufwand:** ggml graph-scheduler refactor
**Konflikt:** überlappt mit upstream tensor-parallelism work (b8738)

#### 2.2 MoE expert-prefetch
**Wo:** MoE-layer in Qwen3.6-A3B
**Aktuell:** gate → softmax → top-4 → fetch expert weights → matmul
**Idee:** pessimistic prefetch der top-N expert weights während gate läuft
**Impact:** PP@MoE-layers +5-10% wenn expert-weight fetches L2/DRAM-bound
**Aufwand:** custom MoE-dispatch kernel

#### 2.3 Async slot-save/restore
**Wo:** llama-server slot-save bei context-switch zwischen sessions
**Aktuell:** save blockt next session start
**Idee:** save auf separate stream während next session prefill starts
**Impact:** multi-user-server latency improvement

### Tier 3: LOW ROI (eigene projekte)

#### 3.1 Speculative decoding (multi-model parallel)
**Status:** ist eigener track. Memory zeigt: Ministral hat schon hohe TG, spec-decode marginal
**Mehrwert für MoE:** könnte größer sein weil 35B-A3B TG nur 76 t/s

#### 3.2 KV-cache compaction parallel zu generation
**Wo:** bei context-overflow muss old context kompaktiert werden
**Aktuell:** blockt
**Idee:** compaction-thread on idle SMs während generation läuft
**Aufwand:** memory-management refactor, niche use-case

## Wichtige beobachtung: Wann auto vs öffis

Direkt aus user's analogie:

| Bottleneck-art | Optimum | Beispiel aus unseren benches |
|----------------|---------|------------------------------|
| Memory/latency-bound, viele small operations | öffis (parallel small batches) | Ministral-3B ub=128 = +18% PP |
| Compute/throughput-bound, tensor cores hungry | auto (large batches) | Phase 5 MMA inline @ ncols1=8 |
| Per-batch fixed cost (MoE routing, kernel-launch) | auto (amortize) | 35B-A3B ub=1024 = +13% PP |
| Pipeline-stall bottleneck | öffis (multiple streams) | dual-GPU layer-split = +50-78% PP |
| Single-track resource (single GPU memory) | sequenziell | TG decode (one token at a time) |

## Konflikt mit upstream tensor-parallelism

Viele tier-2 items überlappen mit was upstream's `d6f303004` (b8738, April 2026)
schon macht. **Empfehlung:** statt eigenständig tier-2 anzugehen, eher die
upstream-TP sync planen — bringt 3-4× und ist battle-tested.

## Konkrete next steps (ROI-rangordnung)

1. **Tier 1.2 (fused GPU sampler)** — low-risk, isolated kernel, measurable
2. **Tier 1.1 (mmproj parallel)** — scheduling-änderung, kein refactor
3. **Tier 1.3 (async KV-write)** — kleine stream-tuning
4. **Tier 2 oder upstream-TP-sync** — größere refactor-entscheidung
5. **SageAttention** — separate research-track

## Sources

- llama.cpp #20252 discussion: pipeline parallelism status
- POD-Attention ASPLOS 25: prefill-decode overlap
- FlashInfer: cp.async pipelining primitives
