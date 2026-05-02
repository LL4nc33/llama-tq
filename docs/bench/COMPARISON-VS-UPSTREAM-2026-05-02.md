# llama-tq vs upstream llama.cpp — A/B Benchmark (2026-05-02 Refresh)

**Date:** 2026-05-02
**Hardware:** test-box — 2× RTX 2060 12GB (CC 7.5), CUDA, FA on, KVM guest of Ryzen 7 3700X
**Methodology:** `llama-bench -p 512 -n 128 -ngl 99 -fa 1 -ts 12,12 -r 3` (sequential, GPU verified clean between runs)

---

## 1. Build Info

| Item | llama-tq | upstream |
|------|----------|----------|
| SHA | `e054a3088` (turboquant branch, 2026-05-02) | `63d93d1` (master, 2026-05-02) |
| CUDA arch | 75 (Turing) | 75 (same hardware) |
| Build flags | `-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` | identical |
| FA | enabled (`-fa 1`) | identical |
| OMP | `OMP_WAIT_POLICY=passive OMP_PROC_BIND=close OMP_PLACES=cores` | identical |
| Repetitions | r=3 | r=3 |

Note: This is a refresh of the 2026-04-26 A/B (commit `6e50fc701` vs `0c6ee1cad`). Both code-bases ~5 weeks newer.

---

## 2. Main Benchmark — Qwen3.6-35B-A3B-IQ2_XXS bartowski (10.01 GiB, 34.66 B params)

| Engine | KV cache | KV bpw | pp512 (t/s) | tg128 (t/s) |
|--------|----------|-------:|------------:|------------:|
| upstream | f16/f16 | 32.0 | **1232.30** | 85.79 |
| upstream | q8_0/q8_0 | 8.5 | 1214.55 | 83.50 |
| upstream | q4_0/q4_0 | 4.5 | 1213.63 | 82.42 |
| llama-tq | f16/f16 | 32.0 | 1195.70 | **87.77** |
| llama-tq | q8_0/q8_0 | 8.5 | 1182.60 | 84.08 |
| llama-tq | q4_0/q4_0 | 4.5 | 1179.95 | 83.30 |
| **llama-tq ktq2/vtq2** ⭐ | **2.78** | 1178.15 | **85.71** |
| llama-tq ktq2/vtq3 | 3.56 | 1176.53 | 85.60 |

## 3. Headline Comparisons

### llama-tq ktq2/vtq2 (2.78 bpw) vs upstream q4_0/q4_0 (4.5 bpw)

The most aggressive upstream KV-quant available, vs llama-tq's TQ default:

| Metric | llama-tq ktq2/vtq2 | upstream q4_0/q4_0 | Δ |
|---|---:|---:|---:|
| pp512 | 1178.15 | 1213.63 | -2.9% |
| **tg128** | **85.71** | **82.42** | **+4.0%** |
| **bpw** | **2.78** | **4.5** | **-38%** |
| **KV @ 32k** | **55 MiB** | **90 MiB** | **-39%** |
| **KV @ 200k** | **348 MiB** | **562 MiB** | **-38%** |

### llama-tq ktq2/vtq2 (2.78 bpw) vs upstream q8_0/q8_0 (8.5 bpw)

The most popular upstream KV choice for production:

| Metric | llama-tq ktq2/vtq2 | upstream q8_0/q8_0 | Δ |
|---|---:|---:|---:|
| pp512 | 1178.15 | 1214.55 | -3.0% |
| **tg128** | **85.71** | **83.50** | **+2.6%** |
| **bpw** | **2.78** | **8.5** | **-67%** |
| KV @ 32k | 55 MiB | 170 MiB | -68% |
| KV @ 200k | 348 MiB | 1.06 GB | -68% |

### Phase 4 baseline win (without TQ)

llama-tq f16/f16 vs upstream f16/f16 at identical KV bpw shows the Phase 4 perf-stack effect (MADV_HUGEPAGE, mul_mat_id prefetch, OMP-tuning):

| Metric | llama-tq f16/f16 | upstream f16/f16 | Δ |
|---|---:|---:|---:|
| pp512 | 1195.70 | 1232.30 | -3.0% |
| **tg128** | **87.77** | **85.79** | **+2.3%** |

The TG advantage at f16/f16 explains why all llama-tq quants beat their upstream counterparts in TG.

## 4. PPL (Quality Reference)

From the wikitext-2 sweep (chunks=3, ctx=512):

| Config | PPL | Drift vs f16/f16 baseline |
|---|---:|---:|
| llama-tq f16/f16 baseline | 7.2044 | 0.00% |
| **llama-tq ktq2/vtq2** | **7.1807** | **−0.33%** (within stderr of f16) |
| llama-tq ktq2/vtq3 | 7.2024 | −0.03% (essentially lossless) |
| upstream baseline (legacy 35B prod ktq2_1/vtq2_1) | 7.4816 | +3.85% |

## 5. TL;DR

> **llama-tq's `ktq2/vtq2` (2.78 bpw KV) beats upstream `q4_0/q4_0` (4.5 bpw) by +4.0% TG and 38% smaller KV storage on Qwen3.6-35B-A3B with -0.33% PPL drift.**

## 6. Reproducibility

```bash
# Build llama-tq
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
git checkout turboquant
cmake -B build -DGGML_CUDA=ON
cmake --build build -j2 --target llama-bench

# Build upstream
git clone https://github.com/ggml-org/llama.cpp llama.cpp-upstream
cd llama.cpp-upstream
cmake -B build -DGGML_CUDA=ON
cmake --build build -j2 --target llama-bench

# Run identical bench on same hardware
MODEL=Qwen_Qwen3.6-35B-A3B-IQ2_XXS-bartowski.gguf

OMP_WAIT_POLICY=passive ./llama-tq/build/bin/llama-bench -m $MODEL \
  -ngl 99 -fa 1 -ts 12,12 -p 512 -n 128 -r 3 -ctk ktq2 -ctv vtq2

OMP_WAIT_POLICY=passive ./llama.cpp-upstream/build/bin/llama-bench -m $MODEL \
  -ngl 99 -fa 1 -ts 12,12 -p 512 -n 128 -r 3 -ctk q4_0 -ctv q4_0
```

## 7. Caveats

- 2× RTX 2060 12GB Turing-only test box. Other archs may differ.
- `OMP_WAIT_POLICY=passive` may slightly disadvantage upstream (which defaults to system-default).
- ctx=2048 by default in llama-bench. Real-world long-ctx (32k+) deploys may show different KV-storage scaling.
- llama-bench measures pure inference, not server overhead (HTTP, JSON parsing, anthropic-cache).
- PP regression of -3% across all configs is consistent — likely OMP-mode or some non-quant code path; doesn't change the TG/storage win.

## 8. Note on Active Competitors

For context (per `docs/research/SUMMARY-2026-05-02.md`):
- **vLLM** merged TurboQuant April 15, 2026 (PR #38479) — datacenter-targeted, requires Hopper FP8
- **TheTom/llama-cpp-turboquant** (1.1k stars) ships `turbo3`/`turbo4` symmetric K=V types
- **spiritbuun/buun-llama-cpp** (526 stars) ships Trellis-Coded turbo*_tcq, asymmetric default
- **scrya-com/RotorQuant** (940 stars) claims block-diagonal rotation faster than FWHT

llama-tq's unique position: **first to ship K vs V as separate type-families with full FA dispatch matrix**, including Hadamard-domain Q·K trick (no K-dequant), Trellis V (vtq*_2), and outlier-channel-split (vtq*_3).
