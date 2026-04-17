# Run 7 — CPU-path PPL validation (VTQ2_2/3_2/4_2 on Qwen3.5-0.8B)

**Date:** 2026-04-17
**Model:** Qwen3.5-0.8B (q8_0 weights)
**Data:** wikitext-2-raw/wiki.test.raw, ctx=512, 5 chunks
**Branch:** trellis-v2-phase1 @ b59c443d7
**Binary:** build-cpu/bin/llama-perplexity (GGML_CUDA=OFF)
**Baseline:** f16 V-cache PPL = 15.5588 ± 1.318

## Configuration

QK_VTQ_TRELLIS was lowered from 512 → 256 for compatibility with
Qwen3.5-0.8B (head_dim=256, n_head_kv=1 → row_size=256). 512 is not
a divisor and would fail the context init gate.

Block layouts (all three types, 256 samples per block):
- VTQ2_2: 68 B/block → **2.125 bpw**
- VTQ3_2: 100 B/block → **3.125 bpw**
- VTQ4_2: 132 B/block → **4.125 bpw**

vs Phase-1 Q256_G4_group = 2.031 bpw. Overhead +0.094 bpw is the
per-block header (fp16 d + uint16 start_state). Group structure is
effectively G=1 — each ggml-block is one Trellis group.

## Results

(Data will be appended as sweep completes — see run11_ppl_0.8b.csv)

### Measured PPL (5 chunks, ctx=512) — COMPLETE

| Config | PPL | Delta % vs f16 | bpw | Time |
|--------|-----|----------------|-----|------|
| f16 (baseline) | 15.559 ± 1.32 | 0.0% | 16.0 | 11s |
| VTQ2_1 | **FAIL** (CPU path null from_float) | — | 2.5 | — |
| VTQ2_2 | 16.693 ± 1.43 | **+7.29%** | 2.125 | 483s |
| VTQ3_2 | 16.092 ± 1.37 | **+3.43%** | 3.125 | 892s |
| **VTQ4_2** | **15.667 ± 1.33** | **+0.69%** | 4.125 | 3758s |

**Caveat**: 5 chunks × 512 tokens = 2560 samples per PPL — CI is huge
(±1.3–1.4 PPL). The +0.69% VTQ4_2 delta is **well within noise of f16**
— statistically indistinguishable. VTQ2_2/3_2 deltas exceed noise
threshold but confidence intervals overlap with f16 (15.56 ± 1.32
vs 16.09 ± 1.37 vs 16.69 ± 1.43).

**Surprise**: VTQ4_2 is nearly lossless (+0.69%) at 4.125 bpw. That's
3.88× compression over f16 with no perceptible quality cost on 0.8B.
Linear MSE projection predicted +0.2% — measured within noise of that.

**Production implication**: For users who want f16-equivalent quality
at minimum VRAM cost, VTQ4_2 is the sweet spot. For aggressive
compression (2-bit), VTQ2_2 matches vtq2_1's +5.10% only within
CI overlap; real ranking needs 27B + 200+ chunks to resolve.

**VTQ2_1 failure:** Existing VTQ1_1/2_1/3_1/4_1 types have no
`from_float` in type_traits_cpu[] — they only work in FA CUDA kernels.
With `-ngl 0`, set_rows tries to call a null from_float pointer and
segfaults. The new VTQ{2,3,4}_2 types register
quantize_row_vtq{K}_2_ref as from_float, so they're the **first VTQ
family that runs end-to-end on CPU**.

## Observations

### Viterbi encoder is slow on CPU
Full L=16 Viterbi over 256 samples costs ~90ms/block on 8 threads.
For a 512-token forward pass on 6-layer Qwen3.5-0.8B, that's:
  6 layers × 2 V-cache rows × 90ms = ~1.1s per token
Realistic only for reference PPL measurement, not production serving.

Beam-pruned Viterbi (infrastructure exists in Phase-1 but not
integrated into the ggml API) could give 10-50× speedup at ~1% MSE
cost. Necessary for Qwen3.5-27B PPL measurement (otherwise each run
takes hours × n_chunks).

### CPU as reference only
`-ngl 0` forces all compute on CPU. For GPU users (`-ngl 99`) on
VTQ{2,3,4}_2 today, the result would be:
- set_rows on CUDA: `default: nullptr` in convert.cu → failure
- Fallback to CPU set_rows: works but cross-boundary copies slow

Phase 2 adds the CUDA kernels so GPU path is native.

## Phase-2 Decision Criteria

Projection from Phase-1 harness was:
- VTQ2_2: +2.9% PPL vs f16 (linear MSE scaling)
- VTQ3_2: +0.73%
- VTQ4_2: +0.2% (extrapolated)

Measurement (5 chunks, 0.8B model):
- VTQ2_2: +7.3% → **2.5× higher than projection**

### Interpretation

The linear MSE→PPL projection was calibrated on `vtq2_1 @ 2.500 bpw
→ +5.10% measured`. Our Phase-1 harness gave MSE ratio 0.574 for
Q256_G4_group, translating (linearly) to +2.9% expected delta.

Measured +7.3% means the linear map underestimates low-bpw
distortion impact by ~2.5×. Two possible explanations:

1. **Small-model sensitivity**: 0.8B Qwen3.5 has fewer, narrower
   attention heads — each V-value carries more signal, so quant
   errors hurt more. Larger models typically tolerate more KV
   compression. Expected outcome on 27B: **lower measured delta**.

2. **Nonlinear attention sensitivity**: softmax amplifies quant
   errors in top-k values. Not model-size dependent. Expected
   on 27B: similar 2.5× factor.

Without a 27B run, we can't discriminate (1) vs (2). But Phase-2
GPU port is low-risk either way:
- CPU path already works; Phase-2 adds speed, not capability.
- If 27B shows the 2.5× factor, we know VTQ3_2 (projected +0.73%)
  will measure ~+1.8% — still better than vtq2_1 @ 2.5 bpw, but
  likely worse than buun turbo3_tcq.
- If 27B shows closer-to-projection, VTQ3_2 is ~+0.73-+1.0% and
  beats buun at 7.4% lower bpw.

### Recommendation

**Do Phase-2a first (struct + CUDA convert for VTQ{2,3}_2 only)**.
This unlocks fast (GPU-path) PPL measurement on 27B. Then decide
Phase-2b (FA dispatch + production path) based on real numbers.

Skip VTQ4_2 for Phase-2a — overkill if VTQ3_2 comes close to f16
already.
