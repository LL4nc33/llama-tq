# llama-tq: Deployed sub-3 bpw KV-cache quantization on consumer GPUs

**Date:** 2026-04-26
**Repo:** [llama-tq](https://github.com/LL4nc33/llama-tq) (fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp))
**Target audience:** ML inference engineers, llama.cpp contributors, fork users

---

## 1. Hero claim

llama-tq runs **Qwen3.6-35B-A3B at 2.78 bpw average KV cache, 200K context per slot with parallel=2, on 24 GB of consumer-class VRAM (2× RTX 2060 12 GB, Turing sm_75)**, decoding at **75.39 t/s** with **HellaSwag 83.5% (200 tasks)** and **wikitext-2 perplexity within +0.27%–+0.47% of f16/f16**. To our knowledge, this is the only end-to-end deployed combination of randomized-Hadamard K-cache and trellis-coded V-cache currently shipping in any open-source inference engine, and the only sub-3 bpw KV deployment with verified real-model quality on consumer hardware.

---

## 2. The 2.78 bpw deploy

**What it is.** A two-flag configuration on top of llama.cpp:

```
--cache-type-k ktq2_1 --cache-type-v vtq2_2
```

- `ktq2_1`: 3.5 bpw K-cache. Randomized Hadamard Transform (FWHT + per-block sign diagonal, Philox 6r-keyed) followed by Lloyd-Max scalar quantization on the Beta(15.5, 15.5) coordinate marginal of d=32 unit vectors. Q·K is computed in the Hadamard domain (FWHT applied once to Q per tile), so K is never explicitly dequantized in the FA vec path. PolarQuant-class method, inspired by Zandieh et al. (arXiv:2504.19874, ICLR 2026).
- `vtq2_2`: 2.06 bpw V-cache. Group-Viterbi trellis encoder with 16-state shift-register decoder, inverse-Gaussian CDF code table, shared scale per block. Bulk-encoded once at the prefill→decode boundary (f16 staging during prefill avoids the per-token encoder cost and the K-quantization repetition pathology).
- **Average:** (3.5 + 2.06) / 2 = **2.78 bpw**. Compared to the f16 baseline (16 bpw), this is an 82.6% reduction in KV-cache bytes.

**Hardware.** test box, KVM guest VM:
- 12 vCPUs from a Ryzen 7 3700X host (Zen 2, 8C/16T, 2 CCDs × 2 CCXs)
- 40 GB DDR4-3200 (~40 GB/s real)
- 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4)
- Linux 6.8 / Ubuntu 24.04, transparent_hugepage=madvise

**Model.** Qwen3.6-35B-A3B (MoE, 8 active experts of 128, head_dim=128) at bartowski IQ2_XXS (~10 GB weights). Full-GPU, dual-GPU split, parallel=2 slots, 200K context per slot, total reach 400K tokens of KV-cache addressable concurrently.

**Numbers (verified 2026-04-25/26).**
- Decode: **75.39 t/s** at tg128
- Prefill: **1005 t/s** at pp512
- Perplexity: **6.018** wikitext-2 (4-chunk, ctx=2048)
- HellaSwag: **83.5%** at 200 tasks, 95% CI [77.7%, 88.0%]
- Δ vs f16/f16 K=ktq2_1 alone: +0.27% PPL; full 2.78 bpw: +0.47% PPL — both well below the chunk=4 stderr.

---

## 3. What we actually run (stack)

The five layers, top to bottom:
1. **PolarQuant K** (RHT + Lloyd-Max + norm correction + precomputed sign bits).
2. **Trellis V** (group-Viterbi encoder, shift-register decoder, deferred quantization at prefill→decode).
3. **Sparse V Dequant** (skip codebook lookup + scale multiply for any position with attention weight < 1e-6 — at 32K+ context this is >90% of positions).
4. **Asymmetric K/V Pareto** (K-type and V-type chosen independently; the FA dispatch covers all 11×8 combinations under `GGML_CUDA_FA_ALL_QUANTS`).
5. **Adaptive expert/layer routing** (per-context-length deploy scripts switch between an aggressive split that maximizes TG at short contexts and a safe split that preserves 200K reach).

---

## 4. Verified numbers

All measurements 2026-04-25/26 on the hardware in §2. KV cache is `ktq2_1 + vtq2_2` unless noted. `OMP_WAIT_POLICY=active`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, prefetch enabled.

| Model | Weights | tg128 short | tg128 long | pp512 | PPL (4-chunk wikitext-2) | HellaSwag-200 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B (dense, IQ2) | 9.4 GB | **15.26** | n/a | 400 | 7.67 (3-chunk) | 78.0% [71.8%, 83.2%] |
| Gemma4-26B-A4B (MoE, IQ2_XXS) | 9.0 GB | **80.91** | — | 1331 | reasoning-broken on raw text | TBD |
| **Qwen3.6-35B-A3B** (MoE, IQ2_XXS) | 10 GB | **75.39** | 75.39 | 1005 | **6.018** | **83.5% [77.7%, 88.0%]** |
| Qwen3-Next-80B-A3B (MoE) | 25 GB | **~36.5** | 32.62 | 415–463 | 5.0817 | TBD |
| Qwen3.5-122B-A10B (MoE) | 34 GB | **18.24** | 17.43 | 196 | 4.0379 | TBD |

**Phase-4 win-stack on 80B (ctx ≤ 8192):**

| Layer | tg128 | Δ |
|---|---:|---:|
| Phase-3 baseline (vtq2_2 alone) | 30.80 | — |
| + `OMP_WAIT_POLICY=active` | 32.62 | +5.9% |
| + `__builtin_prefetch` in `mul_mat_id` | 32.88 | +0.8% |
| + 18/18/12 adaptive layer split | ~36.5 | +11.0% |
| **Cumulative** | **~36.5** | **+18.5%** |

**Reference points (literature, full-split HellaSwag):** LLaMA-7B Q4 ~76–78%, LLaMA-2-13B Q4 ~80–82%, Mistral-7B Q4 ~82%, Llama-3-8B Q4 ~80%. Our 35B-A3B at 2.78 bpw KV scores 83.5% on the 200-task subset.

**Caveats.**
- HellaSwag-200 is a sample; 95% CI ±5%. Use full 10042-task split for headline-paper comparisons.
- PPL is 4-chunk wikitext-2 (chunk=512). 64-chunk runs budgeted, pending.
- Numbers self-measured on a single host. No third-party reproduction yet.
- test-box is a KVM guest VM; vCPU↔CCX pinning at guest level only.

Single source of truth: [`docs/bench/LIVE_NUMBERS.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/bench/LIVE_NUMBERS.md), updated on each phase ship.

---

## 5. What we did differently

Three contributions distinguish this fork:

### 5.1 Combined RHT-K + Trellis-V in one shipped CUDA + CPU dispatch

PolarQuant (RHT + codebook) for K and group-Viterbi trellis for V are independently published, but no other deployed engine combines them in the same FA dispatch. K and V have different attention-time access patterns: K is read in Q·K (rotation can be folded into Q, eliminating K dequant), V is read in a weighted sum (register-light inline-able dequant matters more than rotation). The fork picks the right tool per cache.

The KTQ-vs-VTQ split was forced by a register-spilling bug — full-RHT V dequant in the FA inner loop blew the register budget, NVCC fell back to `__noinline__` and spilled the 32-element FWHT staging to local memory, LMEM traffic interleaved with FA accumulator state and produced visible numerical corruption. Splitting rotation out of V path fixed corruption and dropped V-dequant from ~40 floats and ~160 FLOPs to ~8 floats and one multiply.

### 5.2 Sparse V Dequant

In FA's V-accumulation loop, attention weights are heavily concentrated — at 32K+ context, >90% of positions have softmax weight below 1e-6. We skip dequant for those positions outright. Single `if` against precomputed `KQ_max`-anchored weight; saved work is codebook table read + scale multiply + FMA into VKQ. Measured: **+22% decode** on Qwen3.6-35B-A3B at 32K+ ctx. Literature search: no analogous optimization in vLLM, SGLang, TensorRT-LLM, or exllamav3.

### 5.3 Asymmetric K/V Pareto + adaptive expert routing

K-types (`ktq{1..4}_1`, `q8_0`, `f16`) and V-types (`vtq{1..4}_1`, `vtq{2,3,4}_2`, `vtq{2,3,4}_3`, `q8_0`, `f16`) chosen independently; FA dispatch covers all combinations. V quality is largely "absorbed" by softmax-weighted sum — `vtq2_2`/`vtq3_2`/`vtq4_2` produced bit-identical PPL on same model despite 16× MSE difference — while K quality affects every dot product. Pareto-optimal lives at high-bit K + low-bit V — opposite of symmetric-quant engines.

Deploy scripts pick layer split adaptively by ctx length: short-ctx (≤ 8192) aggressive (e.g. 18/18/12 on 80B), long-ctx (> 8192) safe to fit 200K KV. Net: **+18.5% TG** at short ctx vs Phase-3 single-split baseline.

---

## 6. Honest limitations

- **Turing-tuned (sm_75).** Launch bounds, FWHT shuffle counts, FA tile sizes calibrated for CC 7.5. Builds + runs on Ampere/Ada/Blackwell but not arch-tuned. FP8 tensor cores, WGMMA, Blackwell tensor-memory paths unused.
- **GGUF/ggml ecosystem only.** No PyTorch, no HF Transformers, no vLLM. Algorithms portable, implementation not.
- **Tuned on MoE.** Phase-4 prefetch + adaptive split shaped by Qwen3.5/3.6 MoE characteristics. Dense models work (Qwen3.6-27B verified) but the 5× speedup vs old Q4 partial-offload is MoE+offload-specific.
- **KVM guest VM.** Host-level cpuset CCX-aware pinning not yet applied; 5–10% throughput likely behind that work item.
- **Self-measured, single-host.** No third-party reproduction. HellaSwag-200 wide CI; full-split runs pending.
- **One PPL pathology fix is load-bearing.** KTQ K-cache, when quantized per-token during prefill, hits a repetition-loop pathology (just-quantized rows accumulate RHT round-trip noise → output collapses into loops). f16 staging during prefill + bulk-convert at prefill→decode boundary is **mandatory**, auto-enabled for all KTQ/VTQ_2 types.

---

## 7. Reproduce

```bash
git clone https://github.com/LL4nc33/llama-tq
cd llama-tq
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server llama-perplexity llama-bench

# 35B-A3B deploy as measured:
./build/bin/llama-server -m Qwen3.6-35B-A3B-IQ2_XXS.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    -c 400000 --parallel 2

# HellaSwag-200:
./build/bin/llama-perplexity -m Qwen3.6-35B-A3B-IQ2_XXS.gguf \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 -fa on -ngl 99 \
    --hellaswag --hellaswag-tasks 200 -f hellaswag_val_full.txt
```

GGUF model files: bartowski IQ2_XXS quants on HuggingFace work as-is.

---

## 8. Roadmap

The deployed 2.78 bpw is not the floor. Phase-5 targets **1.69 bpw average KV** by stacking cross-layer KV reuse:

- **XQuant cross-layer KV port** (~1165 LOC, ~2 weeks). Training-free reuse from brinenick511/XQuant. K-only v1 reduces 200K KV from 13.6 GB to 9.16 GB on 35B-A3B (-32%).
- **Sink + FP16 recent-token window** (~400 LOC, 2 days).
- **RULER + LongBench harness** (~500 LOC). Required gate for any further sub-3 bpw spike.
- **QTIP weight-trellis** (~2345 LOC, ~6 weeks, stretch). Lands 122B in 24 GB at Q3_K_M-equivalent quality. Risk-rated 5/5.
- **Upstream PR to ggml-org/llama.cpp** (Discussion #20969). Sparse V Dequant + adaptive expert routing remain in this fork.

End-state: 122B MoE on 24 GB consumer VRAM at 200K ctx, decode ≥ 25 t/s, HellaSwag full-split within 1% of f16.

Phase-5 plan: [`docs/plans/2026-04-26-phase5-master-plan.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/plans/2026-04-26-phase5-master-plan.md).

---

## 9. Citation and references

```bibtex
@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {ICLR},
  year      = {2026},
  eprint    = {2504.19874},
  archivePrefix = {arXiv}
}

@misc{llamatq2026,
  title  = {llama-tq: Deployed sub-3 bpw KV-cache quantization for llama.cpp},
  author = {LL4nc33},
  year   = {2026},
  howpublished = {\url{https://github.com/LL4nc33/llama-tq}},
  note   = {Fork of ggml-org/llama.cpp at b8303}
}
```

**Primary references.**
- TurboQuant: arXiv:[2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026.
- PolarQuant: arXiv:[2502.02617](https://arxiv.org/abs/2502.02617).
- QJL: arXiv:[2406.03482](https://arxiv.org/abs/2406.03482) (used v1–v4, removed v5).
- Philox PRNG: Salmon et al., SC 2011.
- llama.cpp upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), forked at b8303.
- Live numbers: [`docs/bench/LIVE_NUMBERS.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/bench/LIVE_NUMBERS.md).
- Design doc: [`docs/turboquant.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/turboquant.md).
- Phase-5 roadmap: [`docs/plans/2026-04-26-phase5-master-plan.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/plans/2026-04-26-phase5-master-plan.md).
- Competitive landscape: [`docs/plans/2026-04-26-competitive-landscape.md`](https://github.com/LL4nc33/llama-tq/blob/master/docs/plans/2026-04-26-competitive-landscape.md).

License: MIT (inherited from llama.cpp).
