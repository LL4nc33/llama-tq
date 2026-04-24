# Night Session Summary — 2026-04-18/19

**Date:** 2026-04-18 23:00 → 2026-04-19 02:00 local
**Branch:** `trellis-v2-phase1`
**Commits:** `9c06bdceb` → `8c03e39cb` (17 commits ahead of master)

## TL;DR

Three milestones in one night:
1. **Build green** after OOM + ptxas-zombie debugging
2. **Receiver-side encoder validated**: 4× encoder speedup + 1.75 pp PPL improvement (bonus)
3. **vtq4_2 first measurement**: +0.44% PPL vs f16, essentially indistinguishable

## Timeline

### 23:00–00:30 — Build instability debugging

- fattn-vec template expansion blew 32 GB RAM budget
- gpu00 went into swap-death, ssh timeout for 40min
- VM restart revealed ptxas zombie process holding 10 GB
- Reduced VTQ_2 dispatch matrix (72 TUs → 0) per bypass-is-active state
- Build completed at `-j2` with RAM guard triggering at 1.2 GB avail

### 00:30–01:30 — Validation phase

- Phase-2c LUT extern attempts reverted (`b688c99af`) — nvcc warning 20044-D
  showed extern was silently demoted to static anyway
- Smoke tests on ctx=256/3ch confirmed correctness
- Full sweep on ctx=512/5ch (matching run19 baseline config)

### 01:30–02:00 — Documentation + analysis

- Blog post `2026-04-19-v-cache-validation.md` with real numbers
- Trick 1 ctx=512 null-result confirmed (needs bigger models)
- TP `-sm row` discovered upstream-broken on Qwen3.5 (not our bug)

## Key Measurements

### V-Cache Quality (Qwen3.5-0.8B, wikitext-2, ctx=512/5ch)

| Config | PPL | Δ f16 | V-cache size |
|---|---|---|---|
| f16 | 15.60 | — | 100% |
| vtq2_2 | 16.80 | +7.74% | 13% (7.8× smaller) |
| vtq3_2 | 15.76 | +1.05% | 19% (5.2× smaller) |
| vtq4_2 | 15.67 | +0.44% | 25% (4× smaller) |

### Encoder Speed (Trick 6)

- Pre: 90 s/pass (ctx=256/3ch)
- Post: 22.57 s/pass
- Speedup: **4×**

### Flash Attention + VTQ integration

- CPU-fallback path still active (Phase-2c gated)
- Native GPU path blocked on CUDA relocatable device code (RDC)
- Fix path: enable `CUDA_SEPARABLE_COMPILATION` in cmake — scope for
  separate sprint due to link-time impact

## Null Results (honest reporting)

- **Trick 1** (attention-sink fp16 layer 0): no measurable PPL effect on
  0.8B at ctx=256 or ctx=512. Flag is wired correctly (verified by
  identical PPL with/without). Expected to show on 35B+ at ctx≥2048.
- **Trick 3** (per-model RHT seed calibration): abandoned earlier session
  (kurtosis noise floor, see 2026-04-18 devlog).
- **Phase-2c native path**: still gated — nvcc without RDC cannot share
  `extern __device__` across TUs.

## What's next

### High priority (next session)
1. Enable `CUDA_SEPARABLE_COMPILATION` for ggml-cuda target
2. Activate Phase-2c (remove bypass, restore 36 VTQ_2 FATTN_VEC_CASES)
3. Measure native GPU-path TG tok/s vs CPU-fallback

### Medium priority
4. 35B full-stack validation (Trick 1 + 6 + VTQ_2)
5. TQW Option-B CUDA sprint (weight quantization, ~600 LOC on separate branch)

### Upstream reports
6. File bug: Qwen3.5 + `-sm row` hits
   `GGML_ASSERT(!(split && ne02 < ne12))` in ggml-cuda.cu:1622

## Commits on trellis-v2-phase1 ahead of master (17 total)

Safe for cherry-pick to master:
- `9c06bdceb` perf(vtq-enc): receiver-side Viterbi DP (atomic-free)
- `9d526db23` perf(cuda): revert __ldg on trellis LUT — Turing RO cache trap
- `b688c99af` revert: Phase-2c LUT extern attempts — back to static + bypass
- `82d35aacf`, `daba36055` Trick 1 attention-sink protection (flag, layer-0 routing)
- `8c03e39cb`, `bdbdf8cee` run21/22 CSV measurements + blog posts

NOT merge-worthy (broken TQW3 alias chain):
- `dd2c01a50` → `a24cfa2bb` — KTQ3_1 alias for weights was structurally wrong
  (Hadamard-domain storage incompatible with mul_mat), agent report dated
  2026-04-18 18:00
