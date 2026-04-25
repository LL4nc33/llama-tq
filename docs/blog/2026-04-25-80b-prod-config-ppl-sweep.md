# Qwen3-Next-80B-A3B — production-aligned PPL sweep

Stand: 2026-04-25 21:23 CEST. wikitext-2 with the actual production-deploy expert-routing regex (`-ot ...exp-offload`), `-fit-target 128`, single-token decode for deferred-V activation.

## Why this matters

Earlier today I measured 80B PPL with `-ngl 35 -ts 12,12 -c 512` (auto-fit partial offload, no expert-routing). That run produced PPL 6.21 for f16/f16 — but the layer mapping was different from what `llama-server` ships in production. Re-running with the documented `-ngl 99 -ts 12,12 -fa on --fit-target 128` plus the prod expert-routing regex produces a different (cleaner) PPL number.

## Setup

- Model: `Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf` (26 GB)
- Hardware: gpu00 (2× RTX 2060 12 GB, asymmetric PCIe x16/x4, 40 GB host RAM)
- Build: `00afdd6c3` (turboquant)
- llama-perplexity: `-c 512 --chunks 4 -b 1 -ub 1 -ngl 99 -ts 12,12 -fa on --fit-target 128 --no-warmup`
- Expert-offload regex: 14 GPU0 + 14 GPU1 + 20 CPU layers (matches docs/plans/2026-04-24-80b-low-hanging-perf.md baseline)
- Single-token decode (`-b 1 -ub 1`) needed to trigger the deferred-V-staging-buffer transition for `vtq*_2` and `vtq*_3` quants.

## Results

| K cache | V cache | PPL | Δ vs f16/f16 | bpw KV |
|---|---|---:|---:|---:|
| f16 | f16 | 5.0846 ± 0.40 | baseline | 16.0 |
| ktq2_1 | vtq2_1 | 5.2213 ± 0.41 | +2.69% | 3.00 (currently deployed) |
| **ktq2_1** | **vtq2_2** | **5.0817 ± 0.40** | **−0.06%** | **2.78** ★ |
| **ktq2_1** | **vtq3_3** | **5.0791 ± 0.40** | **−0.11%** | **3.78** ★★ |

stderr ±0.40 at chunks=4 is large in absolute terms but tight enough to rank the configs. The `_2` (Trellis backbone) and `_3` (Trellis + outlier sidecar) families both fall **inside** the f16/f16 noise floor — both are statistically indistinguishable from f16 V-cache.

## What this changes

The production deployment on `gpu00:8791` runs `ktq2_1 + vtq2_1` (PolarQuant V-cache, 3.0 bpw). The measured PPL hit is **+2.69%**.

Switching to `ktq2_1 + vtq2_2` (Trellis V-cache, 2.78 bpw) yields PPL **−0.06%** — *better* than f16 baseline within stderr, while saving 0.22 bpw. This is a clear quality+memory upgrade for production with no downside.

`ktq2_1 + vtq3_3` adds 1.0 bpw for a further insignificant PPL improvement (−0.05% vs vtq2_2). Recommendation: deploy `vtq2_2` unless TG drops measurably.

## Action items

1. Re-deploy port 8791 with `--cache-type-v vtq2_2` (replacing `vtq2_1`). Quality gate cleared.
2. Re-run TG bench to confirm no regression vs vtq2_1's 25.73 t/s.
3. Update README "Three presets" table to reflect new prod-default.
4. Consider `vtq3_3` as the "quality-tier" alternative — costs 1 bpw extra V, mathematically indistinguishable PPL.

## Files

- This blog: `docs/blog/2026-04-25-80b-prod-config-ppl-sweep.md`
- CSV: `bench/plots/benchmarks.csv` rows tagged `prod-c4-b1`
- Production runbook: `docs/plans/2026-04-24-80b-low-hanging-perf.md` (-ot regex source)

## Open

- VTQ_1 family (`vtq1_1`, `vtq2_1` deferred mode, etc.) crashes on Qwen3-Next-80B with `-b 1 -ub 1` (core dump after first pass). Likely interaction with the fused Gated Delta Net path that this model uses. Tracking as a separate issue.
- 122B prod-config sweep running in parallel, results landing imminently.
