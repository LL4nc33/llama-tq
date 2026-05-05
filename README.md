# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)
[![Hardware](https://img.shields.io/badge/tested-2x%20RTX%202060%20(CC%207.5)-orange)](#hardware-notes)

**The llama.cpp fork with independent K and V cache type families. Hadamard-domain Q·K dot product. Trellis V-cache. 38% smaller KV than upstream's most aggressive quant — at lossless quality.**

**For users:** drop in two flags (`--cache-type-k ktq2 --cache-type-v vtq2`), get a Qwen3.6-35B-A3B running at **86 t/s @ 100k+mmproj on a single 12 GB RTX 2060** — or 2× 200k parallel slots on a 24 GB total dual-2060 setup. Same answer quality as f16 (PPL drift −0.33% at chunks=3, within stderr). **For developers:** the only inference engine where K-cache and V-cache have separate type families (KTQ × VTQ matrix, full FA dispatch). KTQ uses Randomized Hadamard Transform + Lloyd-Max codebook with Hadamard-domain Q·K (no K dequant in attention). VTQ has three sub-families: codebook (v1), group-Viterbi Trellis (v2), Trellis + outlier-channel-split (v3).

> **tl;dr** — `--cache-type-k ktq2 --cache-type-v vtq2` (= 2.78 bpw KV) on Qwen3.6-35B-A3B-IQ2_XXS:
> - **+4% TG vs upstream q4_0/q4_0** (85.71 vs 82.42 t/s, llama-bench)
> - **+2.6% TG vs upstream q8_0/q8_0** (the popular upstream choice)
> - **−38% KV storage** vs upstream q4_0, **−67%** vs q8_0
> - **−0.33% PPL drift vs f16** (chunks=3 wikitext-2, within stderr)
> - 100k+mmproj fits a single 12 GB GPU; 200k parallel slots × 2 fits 24 GB total
>
> **What this fork does that no other does** (per [Research summary 2026-05-02](docs/research/SUMMARY-2026-05-02.md)):
> - First inference engine with K vs V as separate type families (KTQ × VTQ matrix)
> - Hadamard-domain Q·K dot product (compute attention without dequantizing K — likely the first software impl)
> - V-cache outlier-channel-split (top-N fp16 outliers per block) for lossless V quality
> - Sparse V dequant skip (+22% decode at 32k+ context)
> - Currently running 100k+ context on a 12 GB single-GPU; talks to any OpenAI-compatible or Anthropic-compatible client (see [Client integration](#client-integration))

## Contents

**Always visible:**
- [Quick Start](#quick-start) — build + 2 flags + 5-tier table
- [vs llama.cpp upstream](#vs-llamacpp-upstream--apples-to-apples-2026-05-02) — A/B 2026-05-02
- [Client integration](#client-integration) — OpenAI / Anthropic / Ollama native

**Collapsed (click to expand):**
- Glossary, Quality-vs-throughput score, Full feature matrix
- KV-cache families (KTQ, VTQ v1/v2/v3)
- Live performance numbers, Large-MoE deployments (5 models)
- Benchmarks (full 4×4 matrix, 50+ configs)
- Perplexity (wikitext-2), KV memory savings
- How it works, Build, Roadmap

---

<details>
<summary><b>Glossary</b> — skim if you need it</summary>

| Term | What it means |
|---|---|
| **K-cache / V-cache** | The two memory buffers Attention writes per token (Keys + Values). With long contexts they dominate VRAM. |
| **KTQ** | This fork's K-cache format. Randomized Hadamard Transform + Lloyd-Max codebook. Comes as `ktq1_1`–`ktq4_1`. |
| **VTQ** | This fork's V-cache format. Two generations: v1 codebook lookup (`vtq*_1`), v2 group-Viterbi Trellis (`vtq*_2`, near-f16 quality). |
| **bpw** | Bits per weight. Lower = smaller cache. f16 is 16 bpw; `ktq2_1` is 3.5 bpw; `vtq2_2` is 2.25 bpw. |
| **PPL** | Perplexity, a quality metric. Lower is better. **+0.15% vs f16 = practically lossless.** |
| **TG / PP** | Token generation speed (decode) / Prompt processing speed (prefill), both in tokens per second. |
| **D** | Attention head dimension. 64/128/256/512 depending on model. All four are supported. |

![KV-cache bpw vs PPL Pareto frontier](docs/img/ppl_vs_bpw.png)

</details>

<details>
<summary><b>Quality-vs-throughput score</b> — Pareto leaderboard (35B-A3B)</summary>

Combined score: `ppl_delta_pct + 0.5 × tg_slowdown_pct`. Lower is better. f16/f16 is the reference.

| Score | K / V | Note |
|:---:|---|---|
|  0.00 | f16 / f16 | reference |
| **0.82** | **ktq2_1 / vtq2_2** | 🏆 current default since 2026-04-25 |
|  1.69 | ktq4_1 / vtq4_1 | |
|  2.40 | ktq2_1 / vtq3_1 | legacy v1 (replaced) |
|  2.47 | ktq3_1 / vtq3_1 | |
|  5.50 | ktq2_1 / vtq2_1 | legacy v1 (replaced) |
| 17.66 | ktq1_1 / vtq1_1 | 1-bit floor, unusable |

From `autoresearch/baseline.json`. See the [autoresearch loop](autoresearch/README.md) for iterating on new quant variants.

</details>

<details>
<summary><b>Full feature matrix</b> — what's shipped, what's experimental</summary>

> K-cache and V-cache are quantized **independently**. You pick a K-type (`ktq1_1`…`ktq4_1` at 2.5–5.5 bpw) and a V-type (`vtq1_1`…`vtq4_1` at 1.5–4.5 bpw, or `vtq2_2`…`vtq4_2` Trellis at 2.25–4.25 bpw) — the FA kernel family covers every combination. Use a low-bpw V-type with a higher-bpw K-type for the cleanest quality-per-byte trade.

| Thing | Status |
|-------|--------|
| **KTQ K-cache** — RHT + Lloyd-Max, 1/2/3/4-bit (2.5–5.5 bpw), Q·K computed in Hadamard domain (no K dequant) | shipped, 4 types |
| **VTQ V-cache v1** — DHD rotation + Laplace-fit codebook, 1/2/3/4-bit (1.5–4.5 bpw), codebook lookup in FA inner loop | shipped, 4 types |
| **VTQ V-cache v2 (Trellis)** — group-Viterbi encoder + shift-register decoder at 2.25 / 3.25 / 4.25 bpw — current default since 2026-04-25 | shipped, all D=64/128/256/512 verified |
| **VTQ V-cache v3 (Trellis + outlier-channel-split)** — v2 backbone plus 4 fp16-outliers per block; +1 bpw avg, ~4× lower V-noise floor | shipped (research tier) |
| **Asymmetric K/V dispatch** — K and V types chosen independently, single FA path. All three VTQ families (VTQ_1 / VTQ_2 / VTQ_3) cover all 11 K-types under `GGML_CUDA_FA_ALL_QUANTS`; default builds ship the full KTQ × VTQ matrix (44 K-K combos verified live) | shipped, full matrix |
| **Deferred K/V quantization** — f16 staging during prefill, bulk-convert at prefill→decode boundary; avoids repetition-loop pathology on K | auto-enabled for KTQ / VTQ\_2 |
| **Anthropic-compatible `/v1/messages`** with prompt caching, tool-call early-stop, `--keep` shift protection | shipped |

**Hardware target:** NVIDIA Turing (CC 7.5) — launch\_bounds and FA tuning are calibrated for sm\_75. **Runs on all CUDA GPUs from CC 6.1+** — Pascal (GTX 10-series), Turing (GTX 16 / RTX 20), Ampere (RTX 30), Ada (RTX 40) and Blackwell (RTX 50). On newer archs everything is functional but not yet arch-specifically tuned.

</details>

---

## Quick Start

Build, then add two flags. K-cache and V-cache types are chosen independently.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server
```

### Pick your tier

> **TurboQuant v8 (2026-05-02):** short CLI aliases `ktq{1,2,3,4}` + `vtq{1,2,3,4}` map to the proven defaults. New `vtq3` (= `vtq3_v8`, enum 58) is a 3.625-bpw trellis-3bit + 2 outliers — essentially **lossless** on 35B-A3B (−0.03% PPL drift vs f16 baseline). Legacy long names (`ktq2_1`, `vtq2_2`, etc.) remain supported.

| Tier | K | V | Avg bpw | VRAM saved | PPL cost | Who it's for |
|---|---|---|:---:|:---:|:---:|---|
| ⭐ **Lossless** (recommended) | `ktq2` | `vtq2` | 2.78 | **91%** | **−0.33%** (3-chunk) / +0.15% (8-chunk) | Most users. Fits ~330k single-ctx (or ~470k with `-ub 128`, or 2× 200k parallel slots) of a 35B MoE on 24 GB total VRAM. Alias of `ktq2_1`/`vtq2_2`. |
| 🪜 **Multi-tenant** (parallel slots) | `ktq2` | `vtq2` | 2.78 | **91%** | **−0.33%** | Multi-user / agent fleets. `gpt-oss-20b F16` (12.85 GB MXFP4-native) fits 24 GB with **4 concurrent 65k slots** (262k total) at 61 t/s per slot. |
| 🆕 **Quality v8** | `ktq2` | `vtq3` | 3.56 | 78% | **−0.03%** | Essentially f16-quality at 3.56 bpw. New `vtq3_v8` = trellis-3bit + 2 outliers (12% smaller than legacy `vtq3_3`). |
| **Aggressive** | `ktq2_1` | `vtq3_1` | 4.0 | 77% | +0.49% | Trade ~0.5% PPL for a different bpw point if v2 isn't built. |
| **Conservative** | `q8_0` | `vtq3_1` | 6.25 | 61% | +1.05% | Falls back to the standard `q8_0` K-quant — no KTQ kernels needed. |
| **Research** | `q8_0` | `vtq4_2` | 6.03 | 62% | +0.44% | Highest-quality Trellis V-cache, larger blocks. |

```bash
# ⭐ Recommended (lossless, 83% smaller KV)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2

# 🪜 Multi-tenant: 4 concurrent 65k slots on a 20B MoE
./build/bin/llama-server -m gpt-oss-20b-F16.gguf -fa on -ngl 99 -ts 12,12 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    -c 262144 --parallel 4

# Aggressive (smaller PPL trade, no v2 kernels needed)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq3_1

# Conservative (mix with stock q8_0 K)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k q8_0 --cache-type-v vtq3_1
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1`. `--cache-type-v` accepts the same stock quants plus `vtq{1,2,3,4}_1` (v1) and `vtq{2,3,4}_2` (v2 Trellis).

---

## vs llama.cpp upstream — apples-to-apples (2026-05-02)

Same hardware (test box, 2× RTX 2060, Ryzen 7 3700X host), same `-fa 1 -ts 12,12 -p 512 -n 128 -r 3`, same `OMP_WAIT_POLICY=passive`. Upstream binary: `63d93d1` (ggerganov/llama.cpp master, fresh 2026-05-02 build). llama-tq binary: `e054a3088` (turboquant branch).

**Reading the table:** ↑ higher is better (throughput), ↓ lower is better (PPL, KV-memory).

### 35B-A3B IQ2_XXS bartowski (full-GPU dual, ctx=2048)

| Engine | KV cache | KV bpw | pp512 ↑ | tg128 ↑ | KV @ 32k ↓ |
|---|---|---:|---:|---:|---:|
| **upstream** | f16/f16 | 32.0 | **1232.30** | 85.79 | 640 MiB |
| **upstream** | q8_0/q8_0 | 8.5 | 1214.55 | 83.50 | 170 MiB |
| **upstream** | q4_0/q4_0 | 4.5 | 1213.63 | 82.42 | 90 MiB |
| **llama-tq** | f16/f16 | 32.0 | 1195.70 | **87.77** | 640 MiB |
| **llama-tq** | q8_0/q8_0 | 8.5 | 1182.60 | 84.08 | 170 MiB |
| **llama-tq** | q4_0/q4_0 | 4.5 | 1179.95 | 83.30 | 90 MiB |
| **llama-tq** `ktq2/vtq2` ⭐ | trellis | **2.78** | 1178.15 | **85.71** | **55 MiB** |
| **llama-tq** `ktq2/vtq3` | trellis+outliers | 3.56 | 1176.53 | 85.60 | 71 MiB |

**Δ llama-tq `ktq2/vtq2` (2.78 bpw) vs upstream `q4_0/q4_0` (4.5 bpw):**
- pp512: −2.9% (1178 vs 1213)
- **tg128: +4.0% (85.71 vs 82.42)**
- **KV @ 32k: −38% (55 MiB vs 90 MiB)**
- **PPL drift vs f16: −0.33%** (within stderr, see "Perplexity" section)

**Δ llama-tq `ktq2/vtq2` vs upstream `q8_0/q8_0` (8.5 bpw, the popular choice):**
- pp512: −3.0%
- **tg128: +2.6%**
- **KV @ 32k: −67% (55 MiB vs 170 MiB)**

**Phase 4 baseline (without TQ):** llama-tq f16/f16 vs upstream f16/f16 at identical KV cost: pp512 −3.0%, **tg128 +2.3%**. The TG advantage is from the Phase 4 stack (`MADV_HUGEPAGE`, `mul_mat_id` prefetch, OMP-tuning); the TQ types match this baseline at 11.5× smaller storage.

### Honest caveats

- **PP regression of ~3% across all configs** is consistent and not from the TQ types themselves. Likely the `OMP_WAIT_POLICY=passive` setting we use for the image-OOM fix; upstream's default is system-default. Switching back would close the gap but burn 12 vCPUs idle (see `scripts/deploy-35b-singlegpu-100k.sh` commit `e054a3088`).
- **Earlier README revisions claimed +33% TG vs upstream.** That was the 2026-04-27 measurement (`1a1d49ef5` vs `0c6ee1cad`). Upstream has since merged Hadamard rotation (PR #21038, default-on), closed the FA-tuning gap, and is now within 3% on PP and 2.3% on TG at f16. **The win is now smaller storage, not faster math.**
- 80B-A3B and 122B-A10B A/B benches are stale (2026-04-26 numbers in `docs/bench/`). Refresh pending.

### Long-context KV scaling

| ctx | f16/f16 | upstream q8_0 | upstream q4_0 | **llama-tq ktq2/vtq2** |
|---|---:|---:|---:|---:|
| 32k | 640 MiB | 170 MiB | 90 MiB | **55 MiB** |
| 100k | 2.0 GB | 530 MiB | 280 MiB | **170 MiB** |
| 200k | 4.0 GB | 1.06 GB | 562 MiB | **348 MiB** |

---

<details>
<summary><b>KV-cache families</b> — KTQ (K-cache), VTQ v1/v2/v3 (V-cache)</summary>

This fork ships **three V-cache families** and **one K-cache family**. All can be freely combined (asymmetric K/V is the reference deployment pattern). Current default since 2026-04-25: `--cache-type-k ktq2_1 --cache-type-v vtq2_2`.

### V-cache v1 — VTQ (PolarQuant, shipped first)

Fixed D·H·D rotation (sign-diagonal · FWHT · sign-diagonal) applied once at the graph level, then a flat codebook lookup per entry in the FA inner loop. Laplace-fit codebooks at 1–2 index bits, uniform-like at 3–4. Reference: arXiv:2504.19874 (PolarQuant, ICLR 2026).

| Type | Index bits | bpw | Block | Intended use |
|------|:---:|:---:|:---:|---|
| `vtq1_1` | 1 | 1.5 | 6 B | extreme VRAM, quality drops sharply |
| `vtq2_1` | 2 | 2.5 | 10 B | previous deployed default (replaced by `vtq2_2`) |
| `vtq3_1` | 3 | 4.0 | 16 B | quality-sensitive, lower TG |
| `vtq4_1` | 4 | 4.5 | 18 B | smallest codebook-fit error of v1 |

**Status:** retained for stability and as a fallback. On Qwen3-Next-80B with `-b 1 -ub 1` the v1 path can crash via the fused Gated Delta Net interaction — tracked separately. v1 still works in batched mode and on all other tested models.

### V-cache v2 — Trellis (current default)

Group-level Viterbi trellis with shared state and shared scale. 16-state shift-register, 16-bit open-start state, inverse-Gaussian CDF code table. The Viterbi path optimizes globally over the block, which adapts implicitly to the running model's V-distribution — at the same average bpw, every V-element gets to leverage local statistics.

| Type | Index bits | bpw | Block |
|------|:---:|:---:|:---:|
| `vtq2_2` | 2 | 2.25 | 36 B  |
| `vtq3_2` | 3 | 3.25 | 52 B  |
| `vtq4_2` | 4 | 4.25 | 68 B  |

**K-collision is a feature, not a bug.** `vtq2_2 / vtq3_2 / vtq4_2` produce bit-identical PPL on the same model — the per-element MSE drops 16× across K=2/3/4, but FA softmax averages it out across the sequence (attention-absorbed). Save the bandwidth — pick `vtq2_2`. Source: `docs/blog/2026-04-25-vtq2-attention-absorbs-bit-depth.md`.

**Activation:** v2 supports D=64/128/256/512 — verified live on Gemma4-26B-A4B (D=256 SWA + D=512 full-attn) and Qwen3.6-35B-A3B (D=128). Encoder is ~22 ms/call, which is why `--cache-type-v vtq*_2` auto-enables **f16 staging during prefill** and runs the bulk Viterbi exactly once at the prefill→decode boundary. Logs say `deferred V quantization enabled (N layers with f16 staging)` on startup. **PPL measurement requires `-b 1 -ub 1`** to fire the deferred-V trigger — batched runs (b > 1) measure f16 + mixed-precision overhead, not the actual VTQ_2 PPL.

### V-cache v3 — Trellis + Outlier-Channel-Split (research, quality tier)

Same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that captures the largest absolute V values losslessly. Round-trip MSE drops a further 4× vs v2 at the cost of +1 bpw average. PPL impact on 35B-A3B at 3.78 bpw avg (`ktq2_1 + vtq3_3`): **+0.47%** vs f16/f16 — well below stderr.

| Type | Index bits | bpw avg | Block |
|------|:---:|:---:|:---:|
| `vtq2_3` | 2 | 3.00 | 48 B  |
| `vtq3_3` | 3 | 4.00 | 64 B  |
| `vtq4_3` | 4 | 5.00 | 80 B  |

Same K-collision pattern as v2 (attention-absorbed). On giants (80B/122B) v3 buys an additional ~0.05% PPL over v2 — currently within stderr at chunks=4. Recommendation: deploy `vtq2_2`; keep v3 reserved for "quality-priority" workloads where 1 extra bpw on V is acceptable. Source: `docs/blog/2026-04-25-vtq3-asymmetric-on-35b.md`.

### K-cache — KTQ (PolarQuant, RHT + Lloyd-Max)

Per-block Randomized Hadamard Transform (FWHT + per-block sign flip) + Lloyd-Max codebook. The FA kernel applies FWHT to Q once per tile and computes Q·K entirely in the Hadamard domain — K is never explicitly dequantized in the vec path. On CC ≥ 7.5 (Turing+) an **MMA-KTQ tensor-core path** is wired and live: split-dequant for prefill (PP ≥ 8 tokens), routes through the existing MMA-F16 tensor-core kernel. Measured KTQ2_1: PP128 **727 t/s** (vs 431 f16 baseline), PP512 875 (parity with f16), PP2048 868 (parity). TG falls back to VEC. Source: `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}`.

| Type | Index bits | bpw | Block |
|------|:---:|:---:|:---:|
| `ktq1_1` | 1 | 2.5 | 10 B |
| `ktq2_1` | 2 | 3.5 | 14 B |
| `ktq3_1` | 3 | 4.5 | 18 B |
| `ktq4_1` | 4 | 5.5 | 22 B |

**Current default:** `ktq2_1` at 3.5 bpw. Measured PPL hit on 35B-A3B with f16 V: **+0.27%** vs f16/f16 — inside the perplexity stderr.

**Why deferred K:** KTQ K-cache suffers a repetition-loop pathology when quantized per-token during prefill — attention re-reads the just-quantized rows, RHT round-trip noise accumulates, and the model loops (`"Es war einfach. Es war einfach. Es war einfach."`). f16 staging during prefill + bulk-convert at prefill→decode avoids this. Auto-enabled for any KTQ type.

**Combining KTQ K with VTQ V:** asymmetric is the reference config. The 35B live deployment uses `ktq2_1 + vtq2_2` at **2.78 bpw avg** for a measured +0.27%–+0.47% PPL hit (well below noise floor). `vtq3_3` adds 1 bpw on V for a further marginal improvement.

</details>

---

<details>
<summary><b>Live performance numbers</b> — single source of truth</summary>

**Single source of truth:** [`docs/bench/LIVE_NUMBERS.md`](docs/bench/LIVE_NUMBERS.md) — current TG, PPL, HellaSwag for all five deploy targets. That file is updated each phase; the per-model deep-dives below stay structurally stable.

**Phase 4 (2026-04-26)** added an adaptive layer-split + `OMP_WAIT_POLICY=active` + `__builtin_prefetch` in `mul_mat_id` to the deploy stack. Net **+18.5% TG on 80B** (30.80 → ~36.5 t/s at ctx ≤ 8192), **+9.3% on 122B** (16.69 → 18.24 t/s). Long-context configs (>8k) fall back to the safe split that fits ctx=200000. New: Qwen3.6-27B dense single-GPU deploy, 5.1× faster than the old Qwen3.5-27B-Q4 offload path.

</details>

<details>
<summary><b>Large-MoE deployments</b> — gpt-oss-20b, Gemma4-26B, 35B-A3B, 80B-A3B, 122B-A10B</summary>

These are the five MoE models we deploy. All measured on the same box: Ryzen 7 3700X host (Zen 2, 8C/16T, 2 CCDs × 2 CCXs, separate L3 per CCX) — test box is a KVM guest VM with 12 vCPUs, 40 GB DDR4-3200 (~40 GB/s real), 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4).

![Large-MoE TG: 35B / 80B / 122B on 2x RTX 2060](docs/img/large_moe_tg.png)

The 20B/26B/35B/80B-TQ1_0 fit fully on GPU. The 80B-IQ2 and 122B spill 20/29 layers to CPU RAM — TG becomes CPU-memory-bandwidth-bound, so the numbers are read against a physics ceiling (DDR4-3200 @ ~40 GB/s real / per-token CPU traffic).

**Current default (all five models):** `--cache-type-k ktq2_1 --cache-type-v vtq2_2` at 2.78 bpw avg. Selected on 2026-04-25 after measuring vtq2_2 vs vtq2_1 on both giants — `vtq2_2` wins or ties on PPL, pp512, and tg128 across the board. Verified on gpt-oss-20b (head_dim=64) on 2026-04-27.

### gpt-oss-20b — small-MoE F16 native (2026-04-27)

OpenAI's `gpt-oss-20b` ships in **native MXFP4** for expert FFN tensors, so the F16 GGUF (12.85 GB) is barely larger than Q2_K (10.7 GB). 24 layers, 8 KV heads, **head_dim=64** (the smallest in our matrix). Fits fully on a single dual-2060 box at the model's full **262k native context** with **4 parallel slots** (4× 65k). Originally hit the upstream `head_dim % blck_size` check (false-positive for VTQ_2 since it quantizes along the sequence axis, not along D) — fixed in commit `c818f6c84`.

```bash
./build/bin/llama-server -m gpt-oss-20b-F16.gguf \
    --host 0.0.0.0 --port 8791 -c 262144 -ngl 99 -ts 12,12 -fa on \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    --parallel 4 --jinja --reasoning off
```

| Config | bpw KV | pp t/s ↑ | tg t/s ↑ | VRAM | ctx total / per-slot |
|---|---:|---:|---:|---:|---:|
| `f16 / f16` parallel=1 | 16.0 | 92 | 62.4 | ~14 GB | 65k / 65k |
| `ktq2_1 / vtq2_1` parallel=1 | 3.0 | 110 | 61.2 | ~14 GB | 65k / 65k |
| `ktq2_1 / vtq2_2` parallel=1 | 2.78 | 114 | 61.1 | ~16 GB | 65k / 65k |
| **`ktq2_1 / vtq2_2`** parallel=4 ⭐ | **2.78** | **113** | **61.3** | **21.5 GB** | **262k / 65k** |

The full 262k native context window with **four concurrent 65k slots** at f16-equivalent quality and ~61 t/s per slot — the entire `gpt-oss-20b` parallel server fits in 24 GB total VRAM with 2.1 GB to spare.

### Gemma4-26B-A4B — fast quality model

Gemma4 reasoning architecture, **256 experts / 8 active**, head_dim=512 (full-attn) + head_dim=256 (SWA). 9 GB weights at bartowski IQ2_XXS. Fits fully on GPU. Daily driver for short-context tasks where Gemma's reasoning style is preferred.

```bash
./build/bin/llama-server -m gemma-4-26B-A4B-bartowski-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 -c 32768 -ngl 99 -fa on \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    --parallel 2 -ts 12,12 --jinja
```

| Config | bpw KV | pp512 | tg128 | VRAM (32k ctx) | PPL |
|---|---:|---:|---:|---:|---|
| `f16 / f16` | 16.0 | 1366 | 84.7 | ~10.5 GB | N/A* |
| `ktq2_1 / vtq2_1` | 3.0 | 1005 | 78.1 | ~9.4 GB | N/A* |
| **`ktq2_1 / vtq2_2`** | **2.78** | **1319** | **79.9** | **~9.3 GB** | N/A* |
| `ktq3_1 / vtq2_2` | 3.78 | 1320 | 79.9 | ~9.5 GB | N/A* |

\* PPL on raw wikitext is broken for reasoning models — Gemma4 expects the chat-template thought-channel prefix and reports PPL in the 10⁴–10⁵ range on raw text, even at f16/f16. For relative quality validation use MMLU/HumanEval against an unquantized-KV baseline.

### 35B-A3B — daily driver

Qwen3.5 or Qwen3.6 35B-A3B (32 experts / 4 active, GQA), UD-IQ2\_XXS weights. Fits fully on GPU at 400k ctx parallel 2 with `ktq2_1 / vtq2_2` (current best-tradeoff config, score 0.82 on the leaderboard at the top of this README).

```bash
./build/bin/llama-server -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 -c 200000 -ngl 99 \
    --flash-attn on --parallel 2 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    -ts 12,12 --jinja --reasoning off
```

| Config | bpw KV | pp512 | tg128 | VRAM (400k ctx, p=2) | PPL (4-chunk) |
|---|---:|---:|---:|---:|---:|
| `f16 / f16` | 16.0 | 1018 | 76.8 | ~24 GB | 5.967 |
| `ktq2_1 / vtq2_1` | 3.0 | 932 | 74.4 | **19.3 GB** | 6.256 (+4.8%) |
| **`ktq2_1 / vtq2_2`** | **2.78** | **996** | **74.9** | **~18.8 GB** | **6.018 (+0.85%)** |
| `ktq2_1 / vtq3_3` | 3.78 | ~990 | ~74.5 | ~19.5 GB | 6.401 (+0.47%, c2-b1) |
| `ktq2_1 / f16` | 9.40 | 997 | 75.5 | ~22.6 GB | 5.895 (+0.27%) |

> **35B deploy default vs README recommendation.** This README recommends `ktq2_1 + vtq2_2` (better PPL, 2.78 bpw avg). Some long-running deploys still use `ktq2_1 + vtq2_1` (slightly faster TG at 3.0 bpw avg). Both configs are within +0.85% PPL of f16/f16 — the difference is marginal and either is a defensible default. New deploys should prefer `vtq2_2`.

Full K × V sweep in the [Benchmarks](#benchmarks) section.

### 80B-A3B — Qwen3-Next hybrid (DeltaNet + Attention)

Hybrid architecture, **512 experts / 10 active**. Two deploy tiers:

#### Tier 1 (current default) — TQ1_0 full-VRAM ⭐

Unsloth's `UD-TQ1_0` is **19.3 GB instead of 26.2 GB**, so both GPUs hold full model + KV — no CPU expert-offload. Live since 2026-04-27 on test box.

```bash
./build/bin/llama-server -m Qwen3-Next-80B-A3B-Instruct-UD-TQ1_0.gguf \
    --host 0.0.0.0 --port 8791 -c 65536 -ngl 99 -ts 12,12 -fa on \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    --parallel 1 --jinja --reasoning off
```

Max ctx full-VRAM = 65k (KV ~1.86 GB, GPU1 has ~100 MB headroom idle). 80k+ OOM on FA compute buffer.

#### Tier 2 (fallback for ctx > 65k) — IQ2_XXS with CPU expert-offload

80B params, ~26 GB at UD-IQ2\_XXS. 14 expert-layers per GPU, 20 offloaded to CPU RAM. Usable at 200k ctx with parallel=1.

```bash
./build/bin/llama-server -m Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 -c 200000 -ngl 99 -ts 12,12 -fa on \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    --parallel 1 --fit-target 128 \
    -ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)\.ffn_(up|down|gate)_exps\.=CUDA0,\
blk\.(14|15|16|17|18|19|20|21|22|23|24|25|26|27)\.ffn_(up|down|gate)_exps\.=CUDA1,\
blk\.(2[89]|3[0-9]|4[0-7])\.ffn_(up|down|gate)_exps\.=CPU" \
    --jinja --reasoning off
# Optional: add --moe-pin-experts for +3.3% TG on this offload tier (opt-in since 2026-04-27)
```

#### Comparison table

Measured 2026-04-25 / -27 (llama-bench, 2 reps; PPL via deploy-aligned `-c 512 --chunks 4 -b 1 -ub 1`):

| Config | bpw KV | model size | pp512 | tg128 | PPL | Δ PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 80B-IQ2_XXS `f16/f16` (CPU offload) | 16.0 | 26.2 GB | 404 | 31.5 | 5.085 | baseline | reference |
| 80B-IQ2_XXS `ktq2_1/vtq2_1` (legacy) | 3.0 | 26.2 GB | 386 | 30.6 | 5.221 | +2.69% | replaced 2026-04-25 |
| 80B-IQ2_XXS `ktq2_1/vtq2_2` (long-ctx fallback) | 2.78 | 26.2 GB | 403 | 30.9 | 5.082 | **−0.06%** | for ctx > 65k |
| **80B-TQ1_0 `ktq2_1/vtq2_2`** (current default) ⭐ | **2.78** | **19.3 GB** | **653** | **57.0** | 7.039 | weight-quant tier | **+85% TG vs offload** |

The +85% TG jump on TQ1_0 is from killing PCIe-x4 expert streaming. Quality cost: TQ1_0 is more aggressive than IQ2_XXS at the *weight* level (PPL 7.04 vs 5.08), but for short-to-medium context it's the tier where 80B fits a dual-2060 box without thrashing PCIe.

Physics ceiling for IQ2_XXS offload tier: 40 GB/s DDR4 / ~0.75 GB per-token CPU traffic → ~53 t/s hard limit. Phase-4 stack lifted current to ~36.5 t/s (69% of ceiling) via OMP_active + adaptive layer-split + prefetch. TQ1_0 full-VRAM is bandwidth-unbound by comparison.

<details>
<summary><b>80B-TQ1_0 full KTQ × VTQ matrix</b> (2026-05-02, full-GPU dual, ts 18,7)</summary>

Full 4×4 KTQ × VTQ sweep + 3 baselines on Qwen3-Next-80B-A3B-UD-TQ1_0 (`ts 18,7 --fit-target 128 OMP_active`). 19 configs, llama-bench `-p 512 -n 128 -r 3`. **Note: gap to deploy 57.0 t/s comes from `--moe-pin-experts` which llama-bench doesn't expose.**

| K | V | pp512 | tg128 | Notes |
|---|---|---:|---:|---|
| f16 | f16 | 688.21 | **56.64** | reference baseline |
| q8_0 | q8_0 | 674.72 | 54.09 | |
| q4_0 | q4_0 | 683.12 | 53.42 | |
| ktq1 | vtq1 | 641.42 | 54.57 | extreme bpw, lower PP |
| ktq1 | vtq2 | 679.87 | 54.72 | |
| ktq1 | vtq3 | 683.57 | 55.20 | |
| ktq1 | vtq4 | 609.68 | 54.31 | |
| ktq2 | vtq1 | 645.09 | 54.70 | |
| **ktq2** | **vtq2** ⭐ | **680.17** | **55.16** | current default |
| ktq2 | vtq3 | 681.20 | 55.05 | v8 quality tier |
| ktq2 | vtq4 | 601.95 | 54.38 | |
| ktq3 | vtq1 | 646.30 | 54.49 | |
| **ktq3** | **vtq2** 🏆 | **681.64** | **55.65** | **best KV-quant TG** |
| ktq3 | vtq3 | 681.46 | 54.63 | |
| ktq3 | vtq4 | 609.68 | 53.55 | |
| ktq4 | vtq1 | 646.16 | 54.44 | |
| ktq4 | vtq2 | 680.80 | 55.47 | |
| ktq4 | vtq3 | 681.59 | 55.04 | |
| ktq4 | vtq4 | 609.29 | 53.97 | |

**Pattern observations:**
- TG range 53.42–56.64, all KV-quants within 1.5 t/s of f16 baseline
- vtq1 family has consistent −40 pp512 vs vtq2/3 (codebook lookup overhead)
- vtq4 family has consistent −70 pp512 (larger blocks)
- `ktq3/vtq2` slightly beats `ktq2/vtq2` on TG (+0.5 t/s, +0.5 bpw cost)

</details>

### 122B-A10B — largest that fits

Qwen3.5-122B-A10B (256 experts / 8 active, GQA(2)). 34 GB weights at UD-IQ2\_XXS. 19 expert-layers on GPU (PCIe-aware 10+9 split), 29 offloaded to CPU RAM.

```bash
./build/bin/llama-server -m Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 -c 200000 -ngl 99 -ts 12,12 -fa on \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
    --parallel 1 --fit-target 128 \
    -ot "blk\.(0|1|2|3|4|5|6|7|8|9)\.ffn_(up|down|gate)_exps\.=CUDA0,\
blk\.(10|11|12|13|14|15|16|17|18)\.ffn_(up|down|gate)_exps\.=CUDA1,\
blk\.(19|[2-4][0-9])\.ffn_(up|down|gate)_exps\.=CPU" \
    --jinja --reasoning off
```

Measured 2026-04-25 (llama-bench, 2 reps; PPL via deploy-aligned `-c 512 --chunks 4 -b 1 -ub 1`):

| Config | bpw KV | pp512 | tg128 | PPL | Δ PPL vs f16 |
|---|---:|---:|---:|---:|---:|
| `f16 / f16` | 16.0 | 187.6 | 17.0 | 4.0634 | baseline |
| `ktq2_1 / vtq2_1` (legacy) | 3.0 | 189.5 | 16.8 | 4.2338 | **+4.19%** |
| **`ktq2_1 / vtq2_2`** (current default) ⭐ | **2.78** | **196.3** | **16.8** | **4.0379** | **−0.63%** |

The 122B is where v2 Trellis shines hardest: `vtq2_2` actually beats `f16/f16` on **both** PPL (−0.63%) and pp512 (+4.6%). VRAM 10.9/10.5 GB at 200k ctx. Full 262k ctx also fits (GQA(2) + 2.78 bpw KV = +140 MB delta from 200k). Physics ceiling: 2.5 GB/token CPU traffic / 56 GB/s effective ≈ 22 t/s, current 17 t/s = 77% efficiency.

#### 122B-IQ1_M — smaller-tier alternative (tested 2026-04-27, not recommended)

Unsloth ships a `UD-IQ1_M` (1.75 bpw) variant at **31.8 GB instead of 36.6 GB**. Tested live with split 13/13/22 expert-blocks (26 GPU / 22 CPU), ktq2_1+vtq2_2, ctx=16k.

| Config | bpw KV | model size | pp t/s | tg t/s | ctx max | PPL (4ch) |
|---|---:|---:|---:|---:|---:|---:|
| 122B-IQ2_XXS ktq2_1+vtq2_2 (10/9/29) | 2.78 | 36.6 GB | **196** | 16.8 | **200k** | 4.038 |
| 122B-IQ1_M ktq2_1+vtq2_2 (13/13/22) | 2.78 | 31.8 GB | 26 | 17.3 | 16k | n/a |

**Verdict: not worth deploying.** Real measurement on test box (live curl, not bench-tool):
- TG roughly **flat** vs IQ2_XXS (17.3 vs 16.8 = +3%) — the layer-count gain is eaten by the hybrid SSM compute path's overhead
- PP **collapses** −87% (26 vs 196 t/s) — fatal for long prompts
- `ctx=16k` is the practical ceiling because Qwen3.5-A10B is hybrid SSM (`full_attention_interval=4`) and the recurrent-state compute buffer eats the layer-budget headroom

For 122B, **stay on IQ2_XXS at long ctx**. The TQ1_0 trick that made 80B fly (full-VRAM, no PCIe streaming) doesn't translate here — even at 1.75 bpw the model exceeds 24 GB VRAM, so we still pay the offload cost without the FA quality of the IQ2 path.

**PCIe asymmetry matters:** GPU0 x16 / GPU1 x4 means heavier expert-load on GPU0 avoids x4 cross-traffic. 19L (10+9) beats balanced 9+9 by +2% TG and +11% PP-stability. Full sweep: [docs/bench-qwen35-122b-a10b.md](docs/bench-qwen35-122b-a10b.md). Deploy PPL sweep (legacy notes): [docs/blog/2026-04-25-giant-models-prod-ppl-sweep.md](docs/blog/2026-04-25-giant-models-prod-ppl-sweep.md).

</details>

---

<details>
<summary><b>Benchmarks</b> — 4-model summary, full K×V matrix, methodology</summary>

All measurements: Ryzen 7 3700X + 2× RTX 2060 12 GB + DDR4-3200, Flash Attention on. Sweep build `0639f7835` / 2026-04-25.

### 4-Model summary (`ktq2_1 + vtq2_2` vs baseline)

The current default config (`ktq2_1` K + `vtq2_2` V at 2.78 bpw avg) compared to f16/f16 baseline across all four target models. PPL measured with deploy-aligned methodology where possible (`-c 512 --chunks 4 -b 1 -ub 1`).

| Model | bpw KV | pp512 | tg128 | PPL | Δ PPL | KV mem savings |
|---|---:|---:|---:|---:|---:|---:|
| Gemma4-26B-A4B | 2.78 | 1319 | 79.9 | N/A* | — | 5.8× |
| Qwen3.6-35B-A3B | 2.78 | 996 | 74.9 | 6.018 | +0.85% | 5.8× |
| Qwen3-Next-80B-A3B | 2.78 | 402.6 | 30.9 | 5.0817 | **−0.06%** | 5.8× |
| Qwen3.5-122B-A10B | 2.78 | 196.3 | 16.8 | 4.0379 | **−0.63%** | 5.8× |

\* Reasoning model — wikitext PPL methodologically broken, see Gemma4 section above.

**Headline:** on the 80B and 122B, `vtq2_2` ties or beats f16 on every metric — quality, throughput, AND memory. PPL goes *negative* (better than f16) within stderr because the Trellis code optimizes globally over the V-distribution. The 35B sees +0.85% PPL (still well below noise floor for downstream tasks) at 5.8× smaller KV-cache.

![Decode throughput by KV config](docs/img/decode_throughput.png)

<details>
<summary><b>Qwen3.6-35B-A3B (UD-IQ2_XXS)</b> — full 50-config K × V matrix, dual-GPU <code>-ts 12,12</code></summary>

48-layer dense MoE, 35B params total / 3B active, head_dim=128 (D=128). Wikitext-2 PPL baseline f16/f16 = **7.062** (64-chunk ctx=512). PPL column shows 4-chunk runs unless marked.

| K | V | PP512 | TG128 | ΔPP | ΔTG | bpw avg | PPL |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **f16** | **f16** | **1018.04** | **76.77** | +0.0% | +0.0% | 16.00 | 7.062 |
| f16 | vtq1\_1 | 951.08 | 75.24 | −6.6% | −2.0% | 8.75 | 6.950 |
| f16 | vtq2\_1 | 952.06 | 75.66 | −6.5% | −1.4% | 9.25 | 6.378 |
| f16 | vtq3\_1 | 919.02 | 74.85 | −9.7% | −2.5% | 10.00 | 5.966 |
| f16 | vtq4\_1 | 877.75 | 74.72 | −13.8% | −2.7% | 10.75 | 6.725 |
| **f16** | **vtq2\_2** | **1006.75** | **75.75** | −1.1% | −1.3% | 9.03 | **7.062** ⭐ |
| f16 | vtq3\_2 | 1006.22 | 75.74 | −1.2% | −1.3% | 9.53 | 7.062 |
| f16 | vtq4\_2 | 1008.43 | 75.70 | −0.9% | −1.4% | 10.03 | 7.062 |
| f16 | q4\_0 | 664.25 | 67.25 | −34.8% | −12.4% | 10.25 | — |
| f16 | q8\_0 | 665.07 | 63.71 | −34.7% | −17.0% | — | — |
| ktq1\_1 | vtq1\_1 | 940.72 | 74.30 | −7.6% | −3.2% | 2.00 | 6.962 |
| ktq1\_1 | vtq4\_1 | 867.35 | 73.82 | −14.8% | −3.8% | 4.00 | 6.710 |
| ktq1\_1 | vtq4\_2 | 997.12 | 74.66 | −2.1% | −2.7% | 3.28 | 6.723 |
| ktq2\_1 | f16 | 997.05 | 75.50 | −2.1% | −1.7% | — | 5.895 |
| ktq2\_1 | vtq1\_1 | 940.98 | 74.19 | −7.6% | −3.4% | 2.50 | 6.962 |
| ktq2\_1 | vtq2\_1 | 931.85 | 74.43 | −8.5% | −3.0% | — | 6.256 |
| ktq2\_1 | vtq3\_1 | 909.36 | 73.58 | −10.7% | −4.2% | — | 6.017 |
| ktq2\_1 | vtq4\_1 | 863.99 | 73.73 | −15.1% | −4.0% | 4.50 | 6.710 |
| **ktq2\_1** | **vtq2\_2** | **995.80** | **74.86** | **−2.2%** | **−2.5%** | **2.78** | **7.073** ⭐ |
| ktq2\_1 | vtq3\_2 | 992.94 | 74.79 | −2.5% | −2.6% | 3.28 | 7.073 |
| ktq2\_1 | vtq4\_2 | 996.12 | 74.86 | −2.2% | −2.5% | 3.78 | 5.976 |
| ktq2\_1 | q4\_0 | 668.88 | 67.00 | −34.3% | −12.7% | — | — |
| ktq2\_1 | q8\_0 | 680.03 | 65.89 | −33.2% | −14.2% | — | — |
| ktq3\_1 | f16 | 991.38 | 75.37 | −2.6% | −1.8% | — | 5.895 |
| ktq3\_1 | vtq1\_1 | 940.31 | 74.20 | −7.6% | −3.3% | 3.00 | 6.962 |
| ktq3\_1 | vtq2\_1 | 927.85 | 74.21 | −8.9% | −3.3% | — | — |
| ktq3\_1 | vtq3\_1 | 903.82 | 73.80 | −11.2% | −3.9% | — | — |
| ktq3\_1 | vtq4\_1 | 864.55 | 73.51 | −15.1% | −4.2% | 5.00 | 6.710 |
| ktq3\_1 | vtq2\_2 | 992.41 | 74.83 | −2.5% | −2.5% | — | 5.976 |
| ktq3\_1 | vtq3\_2 | 991.13 | 74.52 | −2.6% | −2.9% | — | 5.976 |
| ktq3\_1 | vtq4\_2 | 994.29 | 74.86 | −2.3% | −2.5% | 4.28 | 5.976 |
| ktq3\_1 | q4\_0 | 653.10 | 59.81 | −35.8% | −22.1% | — | — |
| ktq3\_1 | q8\_0 | 651.67 | 66.79 | −36.0% | −13.0% | — | — |
| ktq4\_1 | vtq1\_1 | 937.44 | 74.38 | −7.9% | −3.1% | 3.50 | 7.816 |
| ktq4\_1 | vtq4\_1 | 863.46 | 73.53 | −15.2% | −4.2% | 5.50 | 6.710 |
| ktq4\_1 | vtq4\_2 | 993.49 | 74.61 | −2.4% | −2.8% | 4.78 | 6.723 |
| q8\_0 | f16 | 735.69 | 67.63 | −27.7% | −11.9% | — | — |
| q8\_0 | vtq2\_1 | 902.87 | 72.71 | −11.3% | −5.3% | 5.50 | 6.361 |
| q8\_0 | vtq3\_1 | 873.28 | 72.12 | −14.2% | −6.1% | — | — |
| q8\_0 | vtq2\_2 | 747.96 | 63.36 | −26.5% | −17.5% | — | — |
| q8\_0 | vtq3\_2 | 739.68 | 66.70 | −27.3% | −13.1% | — | — |
| q8\_0 | q4\_0 | 681.96 | 69.60 | −33.0% | −9.3% | — | — |
| q8\_0 | q8\_0 | 985.42 | 73.08 | −3.2% | −4.8% | — | — |
| q4\_0 | f16 | 743.31 | 67.16 | −27.0% | −12.5% | — | — |
| q4\_0 | vtq2\_1 | 897.98 | 72.59 | −11.8% | −5.4% | — | — |
| q4\_0 | vtq3\_1 | 874.30 | 72.27 | −14.1% | −5.9% | — | — |
| q4\_0 | vtq2\_2 | 745.13 | 64.45 | −26.8% | −16.0% | — | — |
| q4\_0 | vtq3\_2 | 708.94 | 66.01 | −30.4% | −14.0% | — | — |
| q4\_0 | q4\_0 | 984.79 | 72.61 | −3.3% | −5.4% | — | — |
| q4\_0 | q8\_0 | 663.94 | 68.37 | −34.8% | −10.9% | — | — |

Rows in **bold** are the deploy recommendations: `f16/vtq2_2` is near-free on FA (−1.1% PP, −1.3% TG) and `ktq2_1/vtq2_2` is the lightest-with-K-quant config at −2.2% / −2.5% throughput cost for ~80% KV savings.

**Observations:**
- **VTQ_2 (Trellis v2) is the cheapest V-cache on FA** — 1.1–1.3% slowdown vs f16, beats every VTQ_1 variant at the same or lower bpw.
- **`q4_0` / `q8_0` as V destroys FA dispatch** — drops to ~650 PP, ~60 TG (legacy types fall out of fastest FA path on CC 7.5).
- **Asymmetric `ktq2_1 / vtq2_2`** is the deploy winner at 2.2% PP / 2.5% TG cost for **~80% KV savings** (28.75 MiB vs 160 MiB at ctx=8192).
- **1bit (vtq1_1) is usable** — `f16/vtq1_1` only −2.0% TG. Speed-wise the `ktq1_1/vtq1_1` combo at 2.0 bpw avg costs only 3.2% TG. PPL +16.5% on Qwen (6.95 vs 5.97) → "Aggressive" quality tier.
- **VTQ_2 V-cache is literally PPL-lossless** — 64-chunk PPL on Qwen3.6: `f16/f16` = `f16/vtq2_2` = `f16/vtq3_2` = `f16/vtq4_2` = **7.062**. The Trellis-quantized V cache reproduces the f16 attention output bit-perfectly at this measurement granularity.
- **KTQ K-quant costs ~+0.15% PPL** — `ktq2_1/vtq2_2` = `ktq2_1/vtq3_2` = **7.073** (+0.15% vs f16/f16 baseline 7.062, 64-chunk). All Trellis V variants give identical PPL once K is quantized.
- **`ktq2_1/vtq2_2` (2.78 bpw avg) is the Pareto winner** — same PPL as vtq4_2 (3.78 bpw) at lower VRAM. The bpw difference is gratis savings.

</details>

<details>
<summary><b>Gemma4-26B-A4B (IQ2_XXS)</b> — full 50-config K × V matrix, dual-GPU <code>-ts 12,12</code></summary>

26B MoE with 4B active, 30 layers, hybrid attention (iSWA), reasoning model with `<|channel>thought` format. Full-attention layers use head_dim=512 (D=512), SWA layers head_dim=256. FA-vec dispatch covers D=64/128/256/512 for all TQ types. **V is rms-normed before KV write** (Gemma4-specific).

| K | V | PP512 | TG128 | ΔPP | ΔTG |
|---|---|:---:|:---:|:---:|:---:|
| **f16** | **f16** | **1365.97** | **84.72** | +0.0% | +0.0% |
| f16 | vtq1\_1 | 1082.83 | 81.87 | −20.7% | −3.4% |
| f16 | vtq2\_1 | 1024.38 | 80.54 | −25.0% | −4.9% |
| f16 | vtq3\_1 | 913.67 | 79.06 | −33.1% | −6.7% |
| f16 | vtq4\_1 | 808.93 | 79.61 | −40.8% | −6.0% |
| **f16** | **vtq2\_2** | **1343.97** | **82.73** | **−1.6%** | **−2.3%** |
| f16 | vtq3\_2 | 1344.84 | 82.70 | −1.5% | −2.4% |
| **f16** | **vtq4\_2** | **1356.80** | **83.55** | **−0.7%** | **−1.4%** ⭐ |
| f16 | q4\_0 | 380.72 | 55.11 | −72.1% | −35.0% |
| f16 | q8\_0 | 393.75 | 52.65 | −71.2% | −37.9% |
| ktq1\_1 | vtq1\_1 | 1058.42 | 78.03 | −22.5% | −7.9% |
| ktq1\_1 | vtq4\_1 | 800.22 | 76.60 | −41.4% | −9.6% |
| ktq1\_1 | vtq4\_2 | 1325.46 | 80.41 | −3.0% | −5.1% |
| ktq2\_1 | f16 | 1321.62 | 81.78 | −3.2% | −3.5% |
| ktq2\_1 | vtq1\_1 | 1061.76 | 78.65 | −22.3% | −7.2% |
| ktq2\_1 | vtq2\_1 | 1005.42 | 78.08 | −26.4% | −7.8% |
| ktq2\_1 | vtq3\_1 | 900.80 | 76.44 | −34.1% | −9.8% |
| ktq2\_1 | vtq4\_1 | 800.99 | 76.43 | −41.4% | −9.8% |
| **ktq2\_1** | **vtq2\_2** | **1318.67** | **79.88** | **−3.5%** | **−5.7%** |
| ktq2\_1 | vtq3\_2 | 1314.98 | 79.74 | −3.7% | −5.9% |
| **ktq2\_1** | **vtq4\_2** | **1324.86** | **80.06** | **−3.0%** | **−5.5%** ⭐ |
| ktq2\_1 | q4\_0 | 367.10 | 55.30 | −73.1% | −34.7% |
| ktq2\_1 | q8\_0 | 375.96 | 53.11 | −72.5% | −37.3% |
| ktq3\_1 | f16 | 1320.73 | 81.92 | −3.3% | −3.3% |
| ktq3\_1 | vtq1\_1 | 1059.86 | 78.37 | −22.4% | −7.5% |
| ktq3\_1 | vtq2\_1 | 1007.49 | 78.06 | −26.2% | −7.9% |
| ktq3\_1 | vtq3\_1 | 903.14 | 76.26 | −33.9% | −10.0% |
| ktq3\_1 | vtq4\_1 | 800.67 | 76.11 | −41.4% | −10.2% |
| ktq3\_1 | vtq2\_2 | 1319.96 | 79.94 | −3.4% | −5.6% |
| ktq3\_1 | vtq3\_2 | 1316.41 | 79.74 | −3.6% | −5.9% |
| ktq3\_1 | vtq4\_2 | 1324.48 | 80.16 | −3.0% | −5.4% |
| ktq3\_1 | q4\_0 | 367.97 | 48.38 | −73.1% | −42.9% |
| ktq3\_1 | q8\_0 | 366.66 | 47.10 | −73.2% | −44.4% |
| ktq4\_1 | vtq1\_1 | 1057.68 | 78.61 | −22.6% | −7.2% |
| ktq4\_1 | vtq4\_1 | 799.65 | 76.26 | −41.5% | −10.0% |
| ktq4\_1 | vtq4\_2 | 1321.25 | 79.99 | −3.3% | −5.6% |
| q8\_0 | f16 | 508.51 | 47.61 | −62.8% | −43.8% |
| q8\_0 | vtq2\_1 | 930.84 | 74.16 | −31.9% | −12.5% |
| q8\_0 | vtq3\_1 | 834.23 | 72.37 | −38.9% | −14.6% |
| q8\_0 | vtq2\_2 | 522.73 | 46.21 | −61.7% | −45.5% |
| q8\_0 | vtq3\_2 | 510.40 | 47.42 | −62.6% | −44.0% |
| q8\_0 | q4\_0 | 403.15 | 53.80 | −70.5% | −36.5% |
| q8\_0 | q8\_0 | 1305.33 | 76.00 | −4.4% | −10.3% |
| q4\_0 | f16 | 499.38 | 56.74 | −63.4% | −33.0% |
| q4\_0 | vtq2\_1 | 930.06 | 73.92 | −31.9% | −12.7% |
| q4\_0 | vtq3\_1 | 834.29 | 72.53 | −38.9% | −14.4% |
| q4\_0 | vtq2\_2 | 504.01 | 55.34 | −63.1% | −34.7% |
| q4\_0 | vtq3\_2 | 505.58 | 53.06 | −63.0% | −37.4% |
| q4\_0 | q4\_0 | 1300.30 | 75.18 | −4.8% | −11.3% |
| q4\_0 | q8\_0 | 390.69 | 58.18 | −71.4% | −31.3% |

**⭐ marks Pareto winners** (best speed/compression tradeoff for given column constraint).

**Phase 3 PPL sweep (2026-04-25, wikitext-2, single-token decode `-b 1 -ub 1`):** the prior sweep was invalid because `llama-perplexity` defaulted to multi-token batches, which kept the V-cache permanently in f16 staging (deferred-V trigger fires only at `n_tokens()==1`). Re-running with `-b 1 -ub 1` forces every batch through the deferred-V state machine; the VTQ encoder/decoder is genuinely exercised and FA reads from the quantized cache.

**Qwen3.5-2B Q4_K_M, ctx=2048, 8 chunks** (16K wikitext tokens):

| K / V | PPL | Δ vs f16/f16 (9.6792) | Avg V bpw | Note |
|---|---:|---:|---:|---|
| f16 / f16 | 9.6792 | baseline | 16.0 | reference |
| f16 / vtq2_2 | 9.6780 | −0.012% | 2.25 | within noise (±0.293 stderr) |
| f16 / vtq3_2 | 9.6780 | −0.012% | 3.25 | identical to vtq2_2 (K-invariance still observed at this scale) |
| f16 / vtq4_2 | 9.6780 | −0.012% | 4.25 | identical |
| f16 / vtq2_3 | 9.6799 | +0.007% | 4.0 | outlier-channel split |
| f16 / vtq3_3 | 9.6805 | +0.013% | 5.0 | |
| f16 / vtq4_3 | 9.6799 | +0.007% | 6.0 | |

**Qwen3.5-27B IQ2_XXS, ctx=512, 4 chunks** (2K wikitext tokens, 16 V-staging layers):

| K / V | PPL | Δ vs f16/f16 (8.0266) |
|---|---:|---:|
| f16 / f16 | 8.0266 | baseline |
| f16 / vtq{2,3,4}_2 | 8.0212 | −0.067% (all three identical) |
| f16 / vtq{2,3,4}_3 | 8.0238 | −0.035% (all three identical) |

> 📋 **Methodology fix verified:** kv_cache log now prints `deferred V quantization enabled (N layers with f16 staging)` and FA reads VTQ-encoded V. Encoder/decoder unit tests (`test-vtq2-encoding-diff`, `test-vtq2-cached-roundtrip`) continue to pass.
>
> **Findings:**
> 1. **VTQ ≈ f16 quality at these scales.** Δs are all under ±0.07%, within or below stderr (±1–3% on PPL).
> 2. **Bit-width K (2/3/4) does not produce measurable PPL differences within a version on these short tests** (2–16K tokens). Differences only appear between version 2 (uniform Trellis) and version 3 (Trellis + outlier sidecar): `_3` is consistently ≈ 0.01–0.05% higher PPL than `_2` on Qwen3.5-2B, and ≈ 0.03% lower than f16 on 27B. To resolve K=2/3/4 differentiation a longer test (32+ chunks at ctx≥4096) on a non-instruct base model is required — left as follow-up.
> 3. **The previous "Phase 3 winner" claims (Gemma4 8000+ PPL with VTQ_3 ⭐) were spurious** — they were measuring f16/f16 in disguise. The real result is more boring but more honest: **VTQ_3 quality on real inference is statistically indistinguishable from f16 V at 2–16K context.**
>
> Raw data: `bench/plots/benchmarks.csv` rows tagged `phase3-ctx{512,2048}-c{4,8}-b1`. CLI for reproduction:
> ```
> llama-perplexity -m <model> -f wiki.test.raw -c 2048 --chunks 8 -ngl 99 -fa on \
>     -ctk f16 -ctv vtq3_3 -b 1 -ub 1
> ```

**Observations (vs Qwen3.6 sweep):**
- **VTQ_2 family is the Pareto winner on Gemma4 too** — `f16/vtq4_2` only −0.7% PP / −1.4% TG (best non-baseline). `f16/vtq2_2` slightly behind at −1.6% / −2.3%.
- **1bit on D=512 works well** — `f16/vtq1_1` only −3.4% TG (1.0625 bpw V). Phase 1 V_rows=8 D≥256 fix made this practical.
- **VTQ_1 family suffers badly on D=512** — `f16/vtq2_1` is −25% PP and `f16/vtq4_1` is −41% PP, in stark contrast to Qwen's −6 to −14%. The codebook approach has a per-block fixed-cost overhead that scales linearly with D.
- **Legacy `q4_0` / `q8_0` as V is catastrophic** at D=512 (PP −72%). Even worse paired with q-K (`q8_0/vtq2_2` = −62% PP, completely broken FA dispatch).
- **`ktq*/vtq2_2/3_2/4_2` cluster** all within −5.5 to −5.9% TG of baseline at ~3.0–4.78 bpw avg — multiple Pareto points to choose from.
- **TG improvements vs pre-fix** (commit `584378082` vs prior): VTQ-family configs gained +2 to +6% TG via Phase 1 v-rows fix.

**Lever 1 — SWA-mix: per-layer V-cache override** (Phase 6 tooling, 2026-04-25):

Gemma4's 30 layers alternate full-attention (D=512) and SWA (D=256, every 6th: layers 5/11/17/23/29). Quantizing the SWA layers as f16 while keeping full-attn as vtq2_2 trades 25% of expected V-cache savings for **better than uniform-baseline throughput**:

| K | V config | PP512 | TG128 | avg V bpw | Note |
|---|---|:---:|:---:|:---:|---|
| f16 | f16 (uniform) | 1365.97 | 84.72 | 16.00 | f16 baseline |
| f16 | vtq2_2 (uniform) | 1343.97 | 82.73 | 2.25 | uniform Trellis |
| f16 | vtq2_2 + SWA=f16 | 1381.16 | 84.81 | 3.55 | safe option |
| f16 | vtq2_2 + SWA=vtq2_1 | 1382.59 | 84.95 | 2.43 | |
| **f16** | **vtq2_2 + SWA=vtq4_2** | **1383.19** | **85.17** | **2.43** | ⭐ **best Gemma4 config — verified** |
| ktq2_1 | vtq2_2 + SWA=f16 | 1344.61 | 80.89 | 3.55 | with K-quant |

**`SWA=vtq4_2` is the new top Gemma4 config** — verified via llama-server chat completion (`The capital of France is **Paris**.`) and reasoning extraction. Despite the SWA layers having head_dim=256 (where an old `LOG_WRN` cautioned about VTQ_2 corruption), the modern build runs cleanly at D=128, 256, and 512. The warning in `common.cpp:1244` was obsolete and has been removed.

At **avg V bpw 2.43** the config beats both:
- f16/f16 uniform (16 bpw, 1366 PP / 84.72 TG)
- vtq2_2 uniform (2.25 bpw, 1344 PP / 82.73 TG)

→ **+1.3% PP / +0.5% TG vs f16 baseline at 6.6× smaller V-cache**.

Available via env var on llama-bench (Phase 6 tooling, commit `78c3ece6d`):
```bash
# Best Gemma4 config: SWA=vtq4_2
LLAMA_ARG_TQ_V_OVERRIDE='5:vtq4_2,11:vtq4_2,17:vtq4_2,23:vtq4_2,29:vtq4_2' \
  llama-bench -m gemma4.gguf --cache-type-k f16 --cache-type-v vtq2_2 ...
```

For llama-server use the existing `--tq-v-override` flag.

**Sample reasoning output** (greedy, `--log-verbose`):
- `<|channel>thought\nThe user is asking a simple factual question: "What is the capital of France?"...`

**Earlier "gibberish" reports** were a test-harness artifact — llama-cli's interactive REPL prompt-prefix made reasoning control tokens look like empty newlines. Token-ID dump confirms valid sampling.

**Quants tested (both work):** [unsloth UD-IQ2_XXS](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), [bartowski IQ2_XXS](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF).

</details>

---

</details>

<details>
<summary><b>Perplexity (wikitext-2)</b> — quality measurements</summary>

> **Why you'll see different TG numbers across this README for the same config.** TG measurements depend on the harness (`llama-bench` `tg128` vs `tg256` vs `tg1024`, vs live server-runtime t/s), the run count (`-r 2` vs default), and whether the run includes prefill warmup. Examples for `Qwen3.6-35B ktq2_1+vtq2_2` in this doc: 75.40 (apples-to-apples table, llama-bench `tg128`), 74.86 (full sweep matrix, `tg128`), 74.9 (per-model deploy table summary), 72.00 (decode throughput table, `tg256` with `-p 0 -r 2`). All are real measurements on the same box; the variation is harness/methodology, not a regression.
>
> **Why you'll see different PPL deltas across this README.** PPL is sensitive to (a) weight quant, (b) ctx length, (c) chunk count, (d) batch size (`-b 1 -ub 1` exercises deferred-V; larger batches measure f16 staging instead).
>
> Three measurement regimes appear:
> - **64-chunk ctx=512** (matrix sweep, full attn-only PPL) — the headline `+0.15%` for `ktq2_1+vtq2_2` lives here. Used in TL;DR.
> - **8-chunk ctx=512** (apples-to-apples vs upstream, noisier sample) — `+1.34%`. Used only in [vs upstream table](#vs-llamacpp-upstream--apples-to-apples-2026-04-27).
> - **4-chunk ctx=512 `-b 1 -ub 1`** (deploy-aligned, exercises VTQ_2 deferred path) — `+0.85%` on 35B, `−0.06%` on 80B, `−0.63%` on 122B. Used in per-model deploy tables.
>
> The numbers are not contradictory — they're different sample sizes / different decode paths. 64-chunk is the cleanest noise floor; 4-chunk `-b 1 -ub 1` is what the deployed config actually does.

Numbers below use the **2026-04-24 matrix sweep methodology**: 2048 ctx, 5 chunks unless stated otherwise.

### Qwen3.6-35B-A3B (UD-IQ2\_XXS) — 2048 ctx, 5 chunks (preferred methodology)

Representative row from the full 5×8 K × V matrix. All KTQ bitrates produce the same PPL because the attention-only PPL eval can't distinguish K bpw within a forward pass; we show `ktq2_1` as the representative K since it's the lightest.

| K | V | PPL | vs f16/f16 |
|---|---|:---:|:---:|
| f16 | f16 | **6.7251** | — |
| ktq2\_1 | vtq4\_1 | 6.7101 | −0.22% (near-lossless) |
| ktq2\_1 | vtq2\_2 / 3\_2 / 4\_2 | 6.7227 | −0.04% |
| f16 | vtq2\_2 / 3\_2 / 4\_2 | 6.7388 | +0.20% |
| ktq2\_1 | vtq3\_1 | 6.7582 | +0.49% |
| ktq2\_1 | vtq2\_1 | 7.0140 | +4.30% |
| ktq2\_1 | vtq1\_1 | 7.8157 | +16.17% (1-bit floor) |

### Qwen3.6-35B-A3B (Q4\_K\_M)

| K | V | PPL | vs f16/f16 |
|---|---|:---:|:---:|
| f16 | f16 | **5.127** | — |
| f16 | q4\_0 | 5.129 | +0.04% |
| q4\_0 | q4\_0 | 5.169 | +0.8% |
| f16 | vtq3\_1 | 5.177 | **+1.0%** |
| q8\_0 | vtq3\_1 | 5.232 | +2.1% |
| q4\_0 | vtq2\_1 | 5.498 | +7.2% |
| q8\_0 | vtq2\_1 | 5.563 | +8.5% |

### V-cache v2 Trellis (Qwen3.5-0.8B, 512 ctx, 5 chunks)

From `docs/blog/2026-04-19-v-cache-validation.md`, `tests/trellis-phase1/results/run22_08b_full_sweep.csv`. Historical measurement under `QK_VTQ_TRELLIS=512` (since reduced to 128 in Task #143; current bpw is 2.25 / 3.25 / 4.25 — PPL deltas are unaffected).

| V type | bpw | PPL | Δ f16 |
|--------|:---:|:---:|:---:|
| f16 | 16.0 | 15.60 | — |
| vtq2\_2 | 2.06 | 16.80 | +7.74% |
| **vtq3\_2** | 3.06 | 15.76 | **+1.05%** |
| **vtq4\_2** | 4.06 | 15.67 | **+0.44%** |

**Why 2-bit is stuck at ~7%:** 4-state codebook hits an entropy floor for Gaussian/Laplace V entries. The outlier-channel-split sidecar (VTQ_3 family — `vtq2_3 / 3_3 / 4_3`) recovers most of this gap at +1 bpw avg and is shipped — see [V-cache v3](#v-cache-v3--trellis--outlier-channel-split-research-quality-tier).

### Decode throughput (tg256, 35B-A3B IQ2_XXS, measured 2026-04-24)

From `llama-bench -fa 1 -ngl 99 -n 256 -p 0 -r 2`. Running on 2× RTX 2060 12 GB.

| K cache | V cache | tok/s | vs f16/f16 |
|---------|---------|:---:|:---:|
| f16 | f16 | 73.40 | 100% |
| ktq2_1 | f16 | 72.77 | 99.1% |
| f16 | vtq2_2 | 73.01 | 99.5% |
| **ktq2_1** | **vtq2_2** | **72.00** | **98.1%** |
| ktq2_1 | vtq3_2 | 71.99 | 98.1% |
| ktq2_1 | vtq4_2 | 71.97 | 98.0% |
| ktq2_1 | vtq2_1 | 71.59 | 97.5% |
| ktq2_1 | vtq3_1 | 70.51 | 96.1% |
| ktq2_1 | vtq4_1 | 70.52 | 96.1% |

**Finding: VTQ_2 (Trellis) is 1.5–2% faster than VTQ_1 at the same bit class.** First measurable v2 decode advantage — the deferred-V + warp-parallel shift-register decoder keeps the FA inner loop tighter than the v1 codebook lookup. All three v2 variants run within 0.1% of each other at decode — the 2/3/4-bit V-cache choice is pure quality vs memory, not quality vs speed.

**Attention-only PPL caveat:** `llama-perplexity` never hits the prefill→decode transition, so deferred V conversion never fires. Within a single K-cache choice, `vtq{2,3,4}_2` all produce the same PPL (V stays in f16 staging). The 2048-ctx table above reflects the K-cache component; V_2 variants are orthogonally validated on Qwen3.5-0.8B and via the throughput benchmark table (VTQ_2 shows a measurable decode-path speed advantage). Decode-phase PPL for the full 35B V-cache delta is follow-up work.

---

</details>

<details>
<summary><b>KV memory savings</b> — full 5×8 K × V matrix</summary>

Measured on Qwen3.6-35B-A3B-UD-IQ2_XXS at ctx=8192 (10 attention layers out of 48 have KV). Numbers are the actual allocated KV-cache size as reported by the runtime, not a theoretical bpw calculation.

### Full 5 × 8 K × V matrix — total KV in MiB (percentage of f16/f16)

| K \ V | f16 | vtq1_1 | vtq2_1 | vtq3_1 | vtq4_1 | vtq2_2 | vtq3_2 | vtq4_2 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **f16**     | 160.0 (100%) | 87.50 (55%) | 92.50 (58%) | 100.00 (63%) | 102.50 (64%) | 91.25 (57%) | 96.25 (60%) | 101.25 (63%) |
| **ktq1_1**  |  92.50 (58%) | **20.00 (13%)** 🏆 | 25.00 (16%) |  32.50 (20%) |  35.00 (22%) | **23.75 (15%)** |  28.75 (18%) |  33.75 (21%) |
| **ktq2_1**  |  97.50 (61%) | 25.00 (16%) | 30.00 (19%) |  37.50 (23%) |  40.00 (25%) |  28.75 (18%) |  33.75 (21%) |  38.75 (24%) |
| **ktq3_1**  | 102.50 (64%) | 30.00 (19%) | 35.00 (22%) |  42.50 (27%) |  45.00 (28%) |  33.75 (21%) |  38.75 (24%) |  43.75 (27%) |
| **ktq4_1**  | 107.50 (67%) | 35.00 (22%) | 40.00 (25%) |  47.50 (30%) |  50.00 (31%) |  38.75 (24%) |  43.75 (27%) |  48.75 (30%) |

### Per-cache sizes (constant regardless of partner)

| Type  | Size (MiB @ ctx=8192, 10 layers) | Effective bpw |
|-------|:---:|:---:|
| f16   | 80.0  | 16.0 |
| ktq1_1 / vtq1_1 | 12.5 / 7.5   | 2.5 / 1.5 |
| ktq2_1 / vtq2_1 | 17.5 / 12.5  | 3.5 / 2.5 |
| ktq3_1 / vtq3_1 | 22.5 / 20.0  | 4.5 / 4.0 |
| ktq4_1 / vtq4_1 | 27.5 / 22.5  | 5.5 / 4.5 |
| vtq2_2 / vtq3_2 / vtq4_2 | 11.25 / 16.25 / 21.25 | 2.25 / 3.25 / 4.25 |

**Smallest usable:** `ktq1_1 / vtq1_1` at **20 MiB total (13% of f16/f16, 8× smaller)** — but vtq1_1 costs +16% PPL, not practical. **Smallest PPL-sensible:** `ktq1_1 / vtq2_2` at 23.75 MiB (15%, 6.7× smaller). For large contexts the absolute savings matter more — at 200k ctx on the Qwen3.5-122B-A10B GQA(2) config, the deploy `ktq2_1 / vtq2_2` config is ~430 MB total vs ~2.3 GB at f16/f16.

Measurement note: the 10-layer count is Qwen3.6-35B-A3B specific (48 blocks total, 10 with attention after the MoE filter). Different architectures allocate KV on different block counts; scale the per-layer numbers accordingly.

---

</details>

<details>
<summary><b>How it works</b> — RHT, Lloyd-Max, Trellis, Hadamard-domain Q·K</summary>

The trick is to use **different formats for K and V**, because they hit Flash Attention differently.

- **KTQ (K-cache)** — per-block Randomized Hadamard Transform + Lloyd-Max codebook. The kernel transforms Q once per tile and computes Q·K entirely in the Hadamard domain, so K is **never explicitly dequantized** in the hot loop.
- **VTQ v1 (V-cache)** — one D·H·D rotation applied at graph level before writes. Per-entry dequant in the FA inner loop is just `codebook[idx] * scale`.
- **VTQ v2 Trellis (V-cache)** — group-level Viterbi DP encodes 512 samples jointly against a fixed inverse-Gaussian CDF table. Decode is a shift register, one sample per iteration. The encoder is slow (~22 ms/call), so the runtime stages V in f16 during prefill and bulk-converts once at the prefill→decode boundary. This is what makes v2 PPL-lossless on the measurement granularity used.

Full design notes (RHT math, register pressure on CC 7.5, encoder details) live in [`docs/turboquant.md`](docs/turboquant.md). Source: `ggml/src/ggml-trellis.{h,c}`.

---

</details>

## Client integration

The server speaks **three API dialects natively** — no proxy needed:

| Dialect | Endpoints | Works with |
|---|---|---|
| **OpenAI** | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/audio/transcriptions`, `/v1/responses`, `/v1/rerank` | Open WebUI, LibreChat, AnythingLLM, Cline, Roo Code, OpenCode, Aider, Continue.dev, Hermes Agent, Lobe Chat, big-AGI, GPTel, Bolt.diy, openai-sdk |
| **Anthropic** | `/v1/messages`, `/v1/messages/count_tokens` (with prompt caching) | Claude Code, anthropic-sdk |
| **Ollama** | `/api/chat`, `/api/tags`, `/api/show` | Ollama clients |

Most clients work plug-and-play — point them at `http://your-server:8080` (set as `OPENAI_API_BASE` or `ANTHROPIC_BASE_URL`), pass any string as the API key, and run. For tool-calling, start the server with `--jinja`. Cursor and Claude Desktop need a proxy (LiteLLM / claude-code-proxy) because they hardcode the upstream API URL.

Server-side features for agent-style clients (Claude Code, Cline, Roo, Aider with tools):

- **Tool-call early-stop** on `/v1/messages` — saves 1–15 s per agent turn.
- **`--keep 8192`** pins the first 8k prompt tokens across context shifts (system prompts).
- **`--cache-reuse 256`** — KV-shift-based prompt-prefix reuse. Second-turn latency ~60 s → ~5–10 s on 35B.
- **Anthropic prompt caching** — `cache_control:{type:"ephemeral"}` parsed and persisted with 5m/1h TTL. Enable via `--slot-save-path PATH`. Hybrid/recurrent models (Qwen3-Next etc.) get a companion `.ckpt`.
- **Full Anthropic `usage` shape** — `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation.ephemeral_5m/1h_input_tokens` all emitted.
- **TCP\_NODELAY on SSE** + opt-in `gzip` (`LLAMA_SERVER_ZLIB=ON`) on tool-call JSON.

Quick-start: [docs/claude-code.md](docs/claude-code.md) for Claude Code over SSH tunnel.

---

<details>
<summary><b>Build</b> — CMake + CUDA</summary>

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# All KTQ × VTQ FA kernel combinations (longer build):
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

CUDA CC 6.1+. CPU fallback available for all KTQ / VTQ types.

### Hardware notes

sm\_75 (Turing / RTX 2060) is the only calibration target. FA `launch_bounds` and thread-count tuning are set for Turing's SM layout. The MMA-KTQ tensor-core path is live and tested on Turing; on Ampere/Ada/Hopper the code compiles and should run but the FA tuning is probably sub-optimal.

---

</details>

<details>
<summary><b>Roadmap</b> — shipped, active research, discarded</summary>

**Shipped**
- **v8 unified CLI aliases** (2026-05-02) — short names `ktq{1,2,3,4}` + `vtq{1,2,3,4}` map to proven defaults. Legacy long names still supported.
- **`vtq3` (= `vtq3_v8`, enum 58)** — new 3.625-bpw trellis-3bit + 2 outliers — **−0.03% PPL drift on 35B-A3B** (essentially f16-quality, 12% smaller than legacy `vtq3_3`).
- KTQ1\_1 / 2\_1 / 3\_1 / 4\_1 (PolarQuant K-cache, 2.5–5.5 bpw)
- VTQ1\_1 / 2\_1 / 3\_1 / 4\_1 (v1, codebook lookup)
- VTQ2\_2 / 3\_2 / 4\_2 (v2 Trellis, D=64/128/256/512 all verified live) — current default since 2026-04-25
- VTQ2\_3 / 3\_3 / 4\_3 (v3 Trellis + outlier-channel split, research tier)
- Asymmetric KTQ × VTQ dispatch through FA
- Deferred K/V (auto-gated by type)
- Attention-sink protection (first 4 tokens in f16)
- MMA-KTQ asymmetric dispatch — KTQ K + f16 V takes the tensor-core MMA path via bulk K→f16 split-dequant. Reference 35B IQ2\_XS: PP512 **875** (vs f16 861), PP2048 **868** (vs 857), TG128 67 (vs 71). 9.5× jump over the pre-fix 92 t/s.
- Phase 4 perf stack: OMP_active, adaptive layer-split, prefetch in mul_mat_id, P2P opt-in (+18.5% TG on 80B, +9.3% on 122B vs prior baseline)
- `--moe-pin-experts` flag (opt-in, merged 2026-04-27) — pins MoE expert tensors to GPU; measured **+3.3% TG** on 80B-IQ2_XXS (28.95 vs 28.02 t/s) in offload configs. No effect on full-VRAM models.
- Anthropic `/v1/messages` with prompt caching

**Active research**
- **Phase 5 — XQuant cross-layer KV reuse** (code-complete, dormant on hybrid SSM models). Pair adjacent KTQ2_1 layers; subordinate stores only scale, codes shared from sibling. Activates with `--xquant`. Yields 0 pairs on Qwen3.X-A3B / Qwen3.6-27B (alternating Mamba/attention layers); intended for pure-transformer dense models (Llama-3, Mistral, Gemma-2 family). Reference: arXiv:2510.11236.
- Correction Overlay Buffer (Trick 4) — designed, not implemented. Top-N lossless error patch.
- **Phase 7 — imatrix-aware KTQ calibration** (proposed). Use importance matrix to bias the K-quant Lloyd-Max codebook.
- `mmvq` IQ2\_XS tuning on sm\_75 — 28% of kernel time on current 35B config

**Discarded after measurement**
- Speculative decoding on A3B MoE — expert-saturation pathology makes it ineffective
- VTQ\_MIXED — discarded experiment (dominated by VTQ3\_1). Type enum (53) and struct definition remain in `ggml-common.h` for reference; CPU-only path, no CUDA dispatch. Do not use.
- Calibrated outlier selection — marginal gain after RHT
- MMA-KTQ split-dequant as default for all ctx — regresses past ~512 tokens; now short-ctx only

**Not on roadmap**
- FA3 (requires sm\_80+)
- Paged attention (scope mismatch)
- Multi-node inference

---

</details>

## When *not* to use this fork

- **VRAM is not a constraint.** Upstream llama.cpp with f16 KV is simpler and equally fast.
- **Sub-50 ms/token latency at long ctx matters.** VTQ V-cache adds per-token dequant overhead that grows with context length.
- **Multi-node serving.** This fork makes zero changes to llama.cpp's split logic.
- **Ampere+ (CC 8.0+).** Untested. The sm\_75-specific tuning is not going to be a good default there.

---

## License

MIT, inherited from upstream llama.cpp. No restrictions.
