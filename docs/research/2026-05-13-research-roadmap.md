# Research Roadmap — Future Optimization Levers — 2026-05-13

Sammlung von research-findings die heute nicht umgesetzt wurden, aber als
potentielle zukünftige optimization-richtungen dokumentiert sind.

## 1. Upstream Tensor Parallelism (PR #19378, b8738) — HIGH IMPACT

Upstream llama.cpp commit `d6f303004` (April 2026, b8738) bringt **echtes
tensor-parallelism** (statt nur layer-split). Benchmark-claims: **3-4×**
weiter on top über layer-parallel, besonders bei MoE.

### Was es bringen würde

- **Qwen3.6-A35-A3B / OidaNice-GPT-34B:** MoE expert routing distribuiert
  ungleich auf layers → layer-split leidet, TP nicht. Potentiell **4×
  prefill-speedup** zusätzlich zu unseren bisherigen wins
- GPUs auf 100% utilization (statt pipelined sequential)
- NCCL support für multi-machine

### Warum nicht jetzt

- Diff: 3197 insertions, 48 files
- Konflikte mit unserem TQ-code in genau diesen files:
  `ggml-cuda.cu`, `ggml-alloc.c`, `llama-context.cpp`, `llama-model.cpp`
- Cherry-pick = mehrere tage merge-resolution + regression-testing
- Wir sind aktuell bei b8303, master ist b8738+ → **rebase-strategy** wäre
  besser als cherry-pick

### Empfohlene approach (zukunft)

1. Branch `feature/upstream-tp-sync` von `master` starten
2. Subset von TQ-commits cherry-picken auf rebased upstream-base
3. Konflikte einzeln resolven, jeden commit testen
4. Phase 5 kernel (KTQ+VTQ MMA inline) muss re-validiert werden post-rebase

## 2. SageAttention SM_75 — MEDIUM IMPACT

[SageAttention](https://github.com/thu-ml/SageAttention) (ICLR/ICML/NeurIPS 2025
spotlight) claims **2.1-3.1× speedup über FA2** auf Turing durch:

- QK^T in INT8 via `mma.u8.u8.s32` (2× tensor-throughput vs fp16)
- PV bleibt fp16 mit fp16-accumulator
- "K smoothing" pre-pass für quality preservation (<0.2% PPL impact)

### Konzeptuelle compatibility mit unserer arbeit

- KTQ stores K als 2.5-bit hadamard-quantized → bereits compact
- SageAttention quantisiert Q to INT8 für QK^T multiplication
- Würde brauchen: Q→int8 pre-pass + KTQ-K→int8 conversion path

### Warum nicht jetzt

- Größerer kernel-refactor, keine quick win
- Quality-pipeline (smoothing) muss validiert werden gegen unsere KTQ-codebooks
- Sonst risiko von "speedup ohne quality"

## 3. POD-Attention (ASPLOS 25) — RESEARCH ONLY

[POD-Attention](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf)
unlock prefill-decode overlap. Relevant für server mit gemischtem workload
(streaming-decode während prefill neuer request).

### Anwendung

- llama-server `--parallel N` workload
- Aktuell: prefill blockt alle slots, decode wartet
- Mit POD: prefill + decode laufen auf gleichen SMs gleichzeitig

### Status

- Concept-paper, keine offene CUDA-implementation für sm_75
- Würde ggml-scheduler änderungen brauchen

## 4. FlashInfer prefill kernel

[FlashInfer](https://homes.cs.washington.edu/~arvind/papers/flashinfer.pdf)
customizable attention engine. Mehr als research-curiosity — production-quality
mit cp.async pipelining für KV-cache loads.

### Insights nutzbar

- Pipeline-depth tuning (cp.async stages)
- Cache-line-aware tile sizes
- Mixed precision accumulation patterns

Diese sind alle in unserem inline kernel **fragmentarisch implementiert** (nstages,
ldmatrix), aber nicht systematisch.

## 5. Sage attention sm_75 dependency check

Wenn wir eines tages SageAttention probieren:
- Triton 2.1+ benötigt
- CUDA 12+ (aktuell auf gpu00 verifizieren)
- Python integration für quantization-prep

## Empfohlene priorität

1. **Erst stabilisieren was wir haben** — Phase 5 + dual-GPU + ub=128 ist stack
   der schon 13.6× über pre-Phase-5 ist
2. **Tensor Parallelism (#1) ist der nächste sprung** — aber braucht eine
   eigene branch + 1-2 wochen ungestörte arbeit
3. SageAttention (#2) wenn jemand das parallel als research macht
4. POD/FlashInfer als laufendes monitoring (papers lesen, ideas
   inkrementell adoptieren)

## Sources

- llama.cpp PR #19378 / commit `d6f303004` (April 2026): backend-agnostic tensor parallelism
- SageAttention2 ICLR 2025: https://github.com/thu-ml/SageAttention
- POD-Attention ASPLOS 25: https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf
- FlashInfer: https://homes.cs.washington.edu/~arvind/papers/flashinfer.pdf
