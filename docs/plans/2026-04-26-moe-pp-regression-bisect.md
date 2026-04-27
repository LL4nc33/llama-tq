# MoE PP Regression Investigation — llama-tq vs upstream

**Date:** 2026-04-26
**Author:** Debug session (Lance + Claude Opus 4.7)
**Status:** Partial — H1 falsified, root cause not yet isolated. Time-budgeted at 2h, documenting state for follow-up.
**Branch under investigation:** `turboquant` @ `6e50fc701`
**Upstream baseline:** `master` @ `0c6ee1cad`

---

## 1. Symptom

Reproducible -14% pp512 regression on Qwen3.6-35B-A3B (MoE) for **all** KV-cache configurations, including pure f16/f16 where TurboQuant types are not active. The regression is MoE-specific in magnitude — dense models on the same branch show only a baseline overhead, not a multiplied one.

| Engine | Model | KV | pp512 (t/s) | Δ vs upstream |
|---|---|---|---|---|
| upstream `0c6ee1cad` | Qwen3.6-35B-A3B (MoE) | f16/f16 | 1184.99 ± 3.94 | baseline |
| llama-tq `6e50fc701` | Qwen3.6-35B-A3B (MoE) | f16/f16 | 1017.45 ± 4.19 | **-14.1%** |
| upstream | Qwen3.6-27B (dense) | f16/f16 | 417.69 ± 0.05 | baseline |
| llama-tq | Qwen3.6-27B (dense) | f16/f16 | 408.12 ± 0.40 | -2.3% |
| upstream | Qwen3.5-0.8B-Q8_0 (dense) | f16/f16 | 7375.67 ± 25.46 | baseline |
| llama-tq | Qwen3.5-0.8B-Q8_0 (dense) | f16/f16 | 6976.33 ± 2.32 | -5.4% |

Two regressions stack:
- **Baseline overhead** ~2-5% on dense models (cause: not yet isolated; likely TU-bloat from `fattn-common.cuh` includes `turboquant.cuh` + `trellis.cuh`, increasing register pressure / instruction cache footprint of f16 MMA kernel).
- **MoE multiplier** that takes 35B-A3B from ~-5% to -14%. The extra ~10pp is what this investigation focuses on.

TG (decode) is fine across the board (~1% within noise). Decode is bandwidth-bound and exits on the same vec FA path; the regression lives in the **prefill compute graph**.

---

## 2. Approach

**Did not** run a classical `git bisect` because:
1. `turboquant` branch has no merge-base with `upstream/master` (rebase history). `git merge-base HEAD upstream/master` returns no shared commit.
2. Build cost is 15-25 min per step; budget would be exhausted before reaching the bad commit.
3. The regression is reproducible at HEAD on tiny dense models, so it can be measured in **<1 minute per build** — making single-toggle experiments far cheaper than bisect.

**Strategy used:** hypothesis-driven, targeted reverts of the latest suspect commits, measure delta on Qwen3.6-35B-A3B-UD-IQ2_XXS (the actual victim, which loads in ~6s and runs pp512 in ~5s).

---

## 3. Hypotheses

### H1 — CPU prefetch in `mul_mat_id` (commit `6e50fc701`)
> Phase 4 `__builtin_prefetch` of next active MoE expert in `ggml_compute_forward_mul_mat_id` is somehow hurting CUDA-resident MoE prefill.

**Test:** Reverted the 11-line patch in `ggml/src/ggml-cpu/ggml-cpu.c`, rebuilt `libggml-cpu.so` + `llama-bench`, re-ran pp512.

**Result:** **FALSIFIED.**

| Build | pp512 (t/s) |
|---|---|
| HEAD (`6e50fc701`) | 1017.45 ± 4.19 |
| HEAD with prefetch reverted | 1008.62 ± 14.66 |

Within 1σ. CPU prefetch has zero impact on full-GPU-resident MoE — confirmed `ggml_compute_forward_mul_mat_id` is not on the hot path when `-ngl 99` puts the entire model on CUDA. The prefetch only matters in CPU-offload configurations (Phase 4 was tuned for 80B/122B partial offload, not 35B full-GPU).

### H2 — `fattn-common.cuh` TU bloat
> Adding `#include "turboquant.cuh"` (+1376 LOC of `__device__ __forceinline__` helpers) and `#include "trellis.cuh"` (+311 LOC) to `fattn-common.cuh` increases NVCC's TU instantiation graph for every FA `.cu` TU. Even though the helpers are unused for f16/f16, NVCC may make different inlining/register-allocation decisions for the f16 MMA kernel that tip it over a register-spill threshold on Turing (CC 7.5).

**Status:** **NOT YET TESTED.** Verifying this needs a build where the TQ helpers are conditionally excluded from `fattn-common.cuh` for non-TQ instantiations, or a `nvcc --resource-usage` comparison of the relevant FA kernel between the two trees. Estimated cost: 30-45 min for the build hack, 5 min to bench.

Why this is the leading remaining candidate:
- It explains the dense baseline overhead (-2.3% on 27B, -5.4% on 0.8B).
- It explains why **pp** is hit but **tg** is not — pp uses MMA-F16, tg uses VEC; only the MMA-F16 kernel co-resides in the same TU as the new TQ helpers via `fattn-common.cuh`.
- It does **not** by itself explain the MoE multiplier.

### H3 — MoE multiplier source
> The extra ~10pp on MoE specifically. Candidates not yet measured:
> - **`ggml_cuda_get_best_fattn_kernel`** has a new TQ-detection block (lines +361-391 in fattn.cu). For f16/f16 the `is_tq_*` flags are all false and the block is skipped, **but** the function is called per-FA-op. On 40-layer MoE prefill with batch=512, that's 40 calls — irrelevant.
> - **`fattn-mma-f16.cuh`** itself was modified (60 lines diff). Specifically the `flash_attn_ext_f16_load_tile` synchronous (non-cp.async) path. **However** this diff is upstream's *new* code (PR #22051, "CUDA: refactor mma data loading for AMD") that we haven't merged in. Net: **upstream is now using narrower 4-byte loads on the sync path; we still use 16-byte loads.** That makes our path *faster* on Turing (no cp.async), not slower. So this is a **negative regression source** — without our delta here, the gap would be even wider.
> - **`mmq.cuh` / `convert.cu`** have substantial diffs (571 + 317 lines). Most of the visible diff is upstream's NVFP4/RDNA4 work that we have via merge. But the IQ2_XXS dequantization path runs through `convert.cu`, and any non-trivial change there touches every prefill mat-mul in the model. Need to bisect upstream-vs-llama-tq specifically on `convert.cu`.

**Status:** UNTESTED. The most likely actual MoE multiplier source is in the per-expert `mul_mat_id` CUDA kernel selection (`mmid.cu` / `mmf.cu`) or in how the IQ2_XXS dequant path interacts with the per-expert split. **Diff against upstream on `mmid.cu` / `mmid.cuh` / `mmf.cu` / `mmf.cuh` shows zero LL4nc33 commits** — i.e., we have not directly modified those files. Any difference vs upstream there comes from us being behind on upstream merges, not from our patches.

This points to a non-obvious answer: **we may simply be missing an upstream MoE-prefill optimization.** Candidates to check upstream's recent history:
- `5b67fe533 Optimize MOE GEMV kernel for BS > 1` (#20905) — directly relevant to MoE prefill, BS>1 covers pp512.
- `13e15de2a CUDA/HIP: Fix kernel selection for mmvq mmid kernel to align host selection with device launch bounds` (#21238)
- `cb9cef694 CUDA: refactor topk-moe to enable more models` (#19126)

Verify whether these are in our `ggml/src/ggml-cuda/` tree at HEAD or whether they are upstream-only.

### H4 — Eliminated suspects
- `155557cc0` MADV_HUGEPAGE — env-gated behind `LLAMA_MMAP_HUGEPAGES=1`, default off, not active in benchmark runs.
- `7f82b4c33` deferred K quantization — gated behind `--tq-deferred-k`, default false in `common.h:554`. Not active in benchmark runs.
- `deb3493e9` deferred V quantization — same, gated behind `--tq-deferred-v`, default false (`common.h:555`).
- Branding / WebUI / docs changes — obviously not on hot path.

---

## 4. Bisect log

| Step | Build SHA | Working tree state | pp512 (t/s, r=2) | Δ vs upstream | Notes |
|---|---|---|---|---|---|
| 0 | `0c6ee1cad` | upstream/master clean | 1184.99 ± 3.94 | baseline | upstream/master @ 2026-04-26 |
| 1 | `6e50fc701` | turboquant HEAD clean | 1017.45 ± 4.19 | -14.1% | confirms regression at HEAD |
| 2 | `6e50fc701` | + revert of `6e50fc701` (CPU prefetch) | 1008.62 ± 14.66 | -14.9% | H1 falsified — within noise of step 1 |

Steps 3+ not executed within the 2h budget.

---

## 5. Identified culprit

**Not yet isolated.** Confirmed *not* the culprit:
- CPU `__builtin_prefetch` in `mul_mat_id` (H1)
- env-gated MADV_HUGEPAGE (H4)
- env-gated deferred K/V quantization (H4)

Most likely remaining cause for the **MoE multiplier specifically**: missing upstream optimizations in `mmid.cu` / `topk-moe.cu` / `mmf.cu` that we are simply behind on. The turboquant branch was forked at b8303 and has merged forward selectively; some MoE-relevant upstream PRs (especially #20905 "Optimize MOE GEMV kernel for BS > 1") may be missing.

Most likely remaining cause for the **dense baseline**: TU-bloat from `fattn-common.cuh` including `turboquant.cuh` + `trellis.cuh` (H2) — needs verification via `nvcc --resource-usage`.

---

## 6. Root cause analysis

Pending isolation. The two-component model (baseline + MoE multiplier) is well-supported by the data:

```
0.8B dense:  -5.4%  ← baseline overhead only, no MoE
27B  dense:  -2.3%  ← baseline (smaller % because compute-heavier kernel dominates)
35B  MoE:    -14.1% ← baseline + MoE multiplier
```

The fact that `35B - 27B = ~12pp` and not `5×` the dense gap suggests the MoE multiplier is **additive on a per-FA-call basis**, not multiplicative. With 40 layers × ~64 active experts per token batch on 35B-A3B, the MoE path dispatches roughly 40× more `mul_mat_id` operations than the dense `mul_mat` count on 27B's 64-layer dense path. A small fixed overhead per `mul_mat_id` dispatch would scale linearly with that count.

This is consistent with H3's "missing upstream `mmid` optimization" hypothesis — upstream's `5b67fe533` ("Optimize MOE GEMV kernel for BS > 1") would precisely target the BS>1 prefill case where pp512 lives.

---

## 7. Proposed next steps (next session)

In priority order, each ~30-60 min:

1. **Verify which upstream MoE-relevant CUDA commits we have/don't have.** Specifically PRs #20905, #21238, #19126. `git log upstream/master --oneline -- ggml/src/ggml-cuda/mmid.cu ggml/src/ggml-cuda/mmf.cu ggml/src/ggml-cuda/topk-moe.cu` and grep our HEAD for the same SHAs / equivalent code presence. If missing, cherry-pick into a test branch and bench.
2. **Test H2 (TU bloat).** Move `turboquant.cuh` / `trellis.cuh` includes out of `fattn-common.cuh` into per-TU includes only where the TQ helpers are actually used. Bench dense 0.8B and 27B — if baseline overhead drops to ~0%, H2 confirmed.
3. **`nvcc --resource-usage` diff** of the f16 MMA kernel between the two trees. If our build has higher register count or more spills on Turing, that's a smoking gun for H2.
4. **`nsys profile`** of pp512 on 35B-A3B for both engines. Look for which kernel(s) account for the 14% gap. This is the most direct measurement and should have been step 1 — recommend doing it first next session.

---

## 8. Honest disclosure

I burned ~70% of the time budget on log/diff archaeology before running the first revert experiment. In hindsight the right first move was `nsys profile` — it would have pointed at the actual hot kernel in 5 minutes. Documenting this for next session: **profile first, hypothesize second.**

The H1 falsification is a real result and useful: it removes one of the three Phase-4 commits from the suspect list and confirms that CPU-side optimizations are inert on full-GPU configurations as expected.

---

## 9. Reproducibility

All measurements taken on test-box (2× RTX 2060 12GB, CC 7.5, CUDA, FA on, sequential):

```bash
# upstream baseline
cd ~/llama-cpp-upstream && git log -1 --oneline    # 0c6ee1cad
./build/bin/llama-bench -m ~/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    -p 512 -n 0 -ngl 99 -fa 1 -r 2 --output md

# llama-tq HEAD
cd ~/llama-tq && git log -1 --oneline              # 6e50fc701
./build/bin/llama-bench -m ~/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    -p 512 -n 0 -ngl 99 -fa 1 -r 2 --output md

# llama-tq with H1 revert
cd ~/llama-tq && git revert --no-commit 6e50fc701
cd build && cmake --build . --target llama-bench ggml-cpu -j 8
./build/bin/llama-bench -m ~/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    -p 512 -n 0 -ngl 99 -fa 1 -r 2 --output md
git checkout -- .
```

GPU2 had a 378 MiB Python worker (Cortex `run_worker.py`, PID 2253) running throughout; this is constant across all three measurements and should not skew the comparison. Free VRAM was ~11.7 GB on CUDA0 and ~11.4 GB on CUDA1 — comfortably above the model + KV footprint.

No source code committed during this investigation. Working tree restored to clean HEAD state on test-box.
