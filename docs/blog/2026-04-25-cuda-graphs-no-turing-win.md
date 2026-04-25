# CUDA Graphs on Turing — measured: no TG win

Stand: 2026-04-25 20:11 CEST. A/B benchmark of `GGML_CUDA_FORCE_GRAPHS=1` on RTX 2060.

## Setup

- Model: Qwen3.5-2B Q4_K_M, fa=1, ngl=99, ktq2_1 + vtq2_2
- Hardware: gpu00, single RTX 2060 (CC 7.5, Turing)
- Build: `1ebd59e8c` (FORCE_GRAPHS env var added)
- Test: tg128, 3 reps each

## Results

| Config | TG t/s |
|---|---:|
| Baseline (graphs disabled, default Turing) | 139.17 ± 2.16 |
| `GGML_CUDA_FORCE_GRAPHS=1` | 136.97 ± 2.50 |

**Δ: −1.6%, within stderr — no measurable win.**

## Why upstream disables graphs pre-Ampere is correct

The decode-path on Turing is mmvq-bandwidth-bound. CUDA Graph capture+replay overhead (~2-5 µs per replay) does not amortize over a single 128-token TG run when each step already takes ~7 ms (decode + sampling). The launch-overhead reduction graphs are designed to eliminate is a smaller fraction of total time on pre-Ampere than on Ampere+, where Tensor-Core math is fast enough to make launch latency relatively bigger.

For the OidaNice fork's production deployment (Qwen3.6-35B-A3B on 2× RTX 2060), the situation is identical or worse — at 35B with MoE-routing, kernel-launch frequency is similar but per-kernel time is much larger.

## Decision

Keep `1ebd59e8c` as opt-in env var (low risk, useful for future Ampere/Hopper deployments), but:

- Do NOT enable by default
- Do NOT add to production launch script
- Mark CUDA Graphs as **researched and benchmarked, no TG win on Turing**

## Files

- `ggml/src/ggml-cuda/ggml-cuda.cu` (commit `1ebd59e8c`): FORCE_GRAPHS env var
- This blog: `docs/blog/2026-04-25-cuda-graphs-no-turing-win.md`
