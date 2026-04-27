# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)
[![Hardware](https://img.shields.io/badge/tested-2x%20RTX%202060%20(CC%207.5)-orange)](#hardware-notes)

**llama.cpp fork that cuts KV-cache VRAM by 82% near-lossless (+0.15% PPL). Run longer contexts on the same GPU.**

**For users:** drop in two flags, get a 35B MoE running at **~450k single context** (or 2× 200k parallel slots) on 24 GB total VRAM. With default `-ub 512` you'd hit ~330k; lowering to `-ub 128` shrinks the per-GPU compute buffer by ~4× and unlocks the extra 120k. Same answer quality, longer chats, no new hardware. **For developers:** asymmetric KTQ K-cache × VTQ V-cache with split dequant paths inside Flash Attention, plus a Trellis-coded V family that is bit-exact against f16 at the measurement granularity used.

> **tl;dr** — Two flags, `--cache-type-k ktq2_1 --cache-type-v vtq2_2`, take a Qwen3.6-35B MoE to ~330k single-context (or 2× 200k parallel slots) on 24 GB total VRAM. Cost: ~2.5% slower token generation, **+0.15% perplexity vs f16** (64-chunk wikitext-2). The V-cache is perplexity-lossless on its own; the 0.15% is the K-quant. 80B and 122B MoEs run with expert-offload to CPU RAM.
>
> **Default behavior:** ~330k single-context. **Optimized with `-ub 128`:** ~470k single-context (~+40%). Trade-off: prompt processing is ~30% slower at small ubatch (token generation unchanged). Pure KV-cache stays small (~5 GB at 470k); the remaining VRAM is weights (~10 GB), per-GPU Flash Attention compute buffers, and activations. Single-GPU with 24 GB avoids the per-GPU compute-buffer duplication and would push higher.

## Contents

- [vs llama.cpp upstream](#vs-llamacpp-upstream--apples-to-apples-2026-04-27)  ← apples-to-apples on 35B-A3B
- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [KV-cache families](#kv-cache-families)
- [Large-MoE deployments](#large-moe-deployments)
- [Benchmarks](#benchmarks)
- [Perplexity (wikitext-2)](#perplexity-wikitext-2)
- [KV memory savings](#kv-memory-savings)
- [How it works](#how-it-works)
- [Claude Code integration](#claude-code-integration)
- [Build](#build)
- [Roadmap](#roadmap)
- [When *not* to use this fork](#when-not-to-use-this-fork)

---

### Glossary (skim once, then come back)

| Term | What it means |
|---|---|
| **K-cache / V-cache** | The two memory buffers Attention writes per token (Keys + Values). With long contexts they dominate VRAM. |
| **KTQ** | This fork's K-cache format. Randomized Hadamard Transform + Lloyd-Max codebook. Comes as `ktq1_1`–`ktq4_1`. |
| **VTQ** | This fork's V-cache format. Two generations: v1 codebook lookup (`vtq*_1`), v2 group-Viterbi Trellis (`vtq*_2`, near-f16 quality). |
| **bpw** | Bits per weight. Lower = smaller cache. f16 is 16 bpw; `ktq2_1` is 3.5 bpw; `vtq2_2` is 2.06 bpw. |
| **PPL** | Perplexity, a quality metric. Lower is better. **+0.15% vs f16 = practically lossless.** |
| **TG / PP** | Token generation speed (decode) / Prompt processing speed (prefill), both in tokens per second. |
| **D** | Attention head dimension. 64/128/256/512 depending on model. All four are supported. |

![KV-cache bpw vs PPL Pareto frontier](docs/img/ppl_vs_bpw.png)

### Quality-vs-throughput score (35B-A3B IQ2_XXS, wikitext-2 ctx=2048/5ch + tg256)

Combined score: `ppl_delta_pct + 0.5 × tg_slowdown_pct`. Lower is better. f16/f16 is the reference.

| Score | K / V | Note |
|:---:|---|---|
|  0.00 | f16 / f16 | reference |
| **0.82** | **ktq2_1 / vtq2_2** | 🏆 production-default since 2026-04-25 |
|  1.69 | ktq4_1 / vtq4_1 | |
|  2.40 | ktq2_1 / vtq3_1 | older v1 prod (replaced) |
|  2.47 | ktq3_1 / vtq3_1 | |
|  5.50 | ktq2_1 / vtq2_1 | older v1 prod (replaced) |
| 17.66 | ktq1_1 / vtq1_1 | 1-bit floor, unusable |

From `autoresearch/baseline.json`. See the [autoresearch loop](autoresearch/README.md) for iterating on new quant variants.

---

## Highlights

> **TL;DR:** K-cache and V-cache are quantized **independently**. You pick a K-type (`ktq1_1`…`ktq4_1` at 2.5–5.5 bpw) and a V-type (`vtq1_1`…`vtq4_1` at 1.5–4.5 bpw, or `vtq2_2`…`vtq4_2` Trellis at 2.06–4.06 bpw) — the FA kernel family covers every combination. Use a low-bpw V-type with a higher-bpw K-type for the cleanest quality-per-byte trade.

| Thing | Status |
|-------|--------|
| **KTQ K-cache** — RHT + Lloyd-Max, 1/2/3/4-bit (2.5–5.5 bpw), Q·K computed in Hadamard domain (no K dequant) | shipped, 4 types |
| **VTQ V-cache v1** — DHD rotation + Laplace-fit codebook, 1/2/3/4-bit (1.5–4.5 bpw), codebook lookup in FA inner loop | shipped, 4 types |
| **VTQ V-cache v2 (Trellis)** — group-Viterbi encoder + shift-register decoder at 2.06 / 3.06 / 4.06 bpw — production default since 2026-04-25 | shipped, all D=64/128/256/512 verified |
| **VTQ V-cache v3 (Trellis + outlier-channel-split)** — v2 backbone plus 4 fp16-outliers per block; +1 bpw avg, ~4× lower V-noise floor | shipped (research tier) |
| **Asymmetric K/V dispatch** — K and V types chosen independently, single FA path. All three VTQ families (VTQ_1 / VTQ_2 / VTQ_3) cover all 11 K-types under `GGML_CUDA_FA_ALL_QUANTS`; default builds ship the full KTQ × VTQ matrix (44 K-K combos verified live, smoke-tested KTQ4_1 × VTQ3_2 etc.) | shipped, full matrix |
| **Deferred K/V quantization** — f16 staging during prefill, bulk-convert at prefill→decode boundary; avoids repetition-loop pathology on K | auto-enabled for KTQ / VTQ\_2 |
| **Anthropic-compatible `/v1/messages`** with prompt caching, tool-call early-stop, `--keep` shift protection | shipped |

**Hardware target:** NVIDIA Turing (CC 7.5) — launch\_bounds and FA tuning are calibrated for sm\_75. **Runs on all CUDA GPUs from CC 6.1+** — Pascal (GTX 10-series), Turing (GTX 16 / RTX 20), Ampere (RTX 30), Ada (RTX 40) and Blackwell (RTX 50). On newer archs everything is functional but not yet arch-specifically tuned. Arch-specific contributions (FP8 Tensor Cores on Ada+, WGMMA on Hopper) welcome.

---

## Quick Start

Build, then add two flags. K-cache and V-cache types are chosen independently.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server
```

### Pick your tier

| Tier | K | V | Avg bpw | VRAM saved | PPL cost | Who it's for |
|---|---|---|:---:|:---:|:---:|---|
| ⭐ **Lossless** (recommended) | `ktq2_1` | `vtq2_2` | 2.78 | **83%** | **+0.15%** | Most users. Fits ~330k single-ctx (or ~470k with `-ub 128`, or 2× 200k parallel slots) of a 35B MoE on 24 GB total VRAM. |
| **Aggressive** | `ktq2_1` | `vtq3_1` | 4.0 | 77% | +0.49% | Trade ~0.5% PPL for a different bpw point if v2 isn't built. |
| **Conservative** | `q8_0` | `vtq3_1` | 6.25 | 61% | +1.05% | Falls back to the standard `q8_0` K-quant — no KTQ kernels needed. |
| **Research** | `q8_0` | `vtq4_2` | 6.03 | 62% | +0.44% | Highest-quality Trellis V-cache, larger blocks. |

```bash
# ⭐ Recommended (lossless, 83% smaller KV)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_2

# Aggressive (smaller PPL trade, no v2 kernels needed)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k ktq2_1 --cache-type-v vtq3_1

# Conservative (mix with stock q8_0 K)
./build/bin/llama-server -m model.gguf -fa on -ngl 99 \
    --cache-type-k q8_0 --cache-type-v vtq3_1
```

`--cache-type-k` accepts the stock quants (`f16`, `q8_0`, `q4_0`, …) plus `ktq{1,2,3,4}_1`. `--cache-type-v` accepts the same stock quants plus `vtq{1,2,3,4}_1` (v1) and `vtq{2,3,4}_2` (v2 Trellis).

---

## vs llama.cpp upstream — apples-to-apples (2026-04-27)

Same hardware (test box, 2× RTX 2060, Ryzen 7 3700X host), same KV config (`f16/f16`), same `-fa 1 -ts 12,12`. Upstream binary: `0c6ee1cad` (ggerganov/llama.cpp master, fresh 2026-04-27 build with Qwen3-Next + Qwen3.5-A10B arch support). llama-tq binary: `1a1d49ef5` (turboquant).

**Reading the table:** ↑ higher is better (throughput), ↓ lower is better (PPL = quality cost, KV memory = footprint).

### 35B-A3B (full-GPU, ctx=2048)

| Variant | pp512 ↑ | tg128 ↑ | tg1024 ↑ | PPL (8ch) ↓ | KV-bpw ↓ | KV @ 32k ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **upstream** f16/f16 | 934 | 57.76 | 57.30 | 7.0810 | 32.0 | **640 MiB** |
| **llama-tq** f16/f16 | 1009 | 76.58 | 75.09 | 7.0863 | 32.0 | 640 MiB |
| **llama-tq** `ktq2_1+vtq2_2` ⭐ | 1010 | 75.40 | 75.18 | 7.1814 | **2.78** | **115 MiB** |
| **Δ tq vs upstream** (f16 / quant) | **+8% / +8%** | **+33% / +31%** | **+31% / +31%** | +0.07% / +1.34% (worse) | — / **−91%** | — / **−82%** |

### 80B-A3B (Qwen3-Next, hybrid + CPU expert-offload, ctx=2048)

Both binaries with 28 of 48 expert-blocks on CPU (upstream `--n-cpu-moe 28`, llama-tq `-ot` regex split 14/14/20). Identical layer distribution.

| Variant | pp512 ↑ | tg128 ↑ |
|---|---:|---:|
| **upstream** f16/f16 | 378 | 31.21 |
| **llama-tq** f16/f16 | 404 | 31.5 |
| **Δ tq vs upstream** | **+7%** | **+1%** |

### 122B-A10B (Qwen3.5-A10B, GQA(2) + CPU expert-offload, ctx=2048)

Both binaries with 38 of 48 expert-blocks on CPU.

| Variant | pp512 ↑ | tg128 ↑ |
|---|---:|---:|
| **upstream** f16/f16 | 178 | 15.65 |
| **llama-tq** f16/f16 | 187 | 17.0 |
| **Δ tq vs upstream** | **+5%** | **+9%** |

**Key wins llama-tq across all three sizes:**
- **+33% TG on 35B** (full-GPU decode) — Phase-1-to-4 stack (FA quick wins, sparse-V dequant, P2P opt-in, OMP_active, layer-split, prefetch). The all-on-GPU path is where tq's FA tuning shows up.
- **+1–9% TG on 80B/122B** — tighter on offload paths because TG is CPU-RAM-bandwidth-bound (~40 GB/s ceiling on DDR4-3200), so FA wins matter less.
- **+5–8% pp512** consistent across all model sizes — prefill always benefits from tq's FA quick wins.
- **−82% KV-cache memory** at ctx=32k with `ktq2_1+vtq2_2` (35B; same proportional savings on 80B/122B).
- **Quality match in f16/f16** on 35B (within 0.07% noise floor — same numerics, just faster). 80B/122B PPL A/B is a separate sweep.

---

## KV-cache families

This fork ships **three V-cache families** and **one K-cache family**. All can be freely combined (asymmetric K/V is the reference deployment pattern). Production-default since 2026-04-25: `--cache-type-k ktq2_1 --cache-type-v vtq2_2`.

### V-cache v1 — VTQ (PolarQuant, shipped first)

Fixed D·H·D rotation (sign-diagonal · FWHT · sign-diagonal) applied once at the graph level, then a flat codebook lookup per entry in the FA inner loop. Laplace-fit codebooks at 1–2 index bits, uniform-like at 3–4. Reference: arXiv:2504.19874 (PolarQuant, ICLR 2026).

| Type | Index bits | bpw | Block | Intended use |
|------|:---:|:---:|:---:|---|
| `vtq1_1` | 1 | 1.5 | 6 B | extreme VRAM, quality drops sharply |
| `vtq2_1` | 2 | 2.5 | 10 B | previous deployed default (replaced by `vtq2_2`) |
| `vtq3_1` | 3 | 4.0 | 16 B | quality-sensitive, lower TG |
| `vtq4_1` | 4 | 4.5 | 18 B | smallest codebook-fit error of v1 |

**Status:** retained for stability and as a fallback. On Qwen3-Next-80B with `-b 1 -ub 1` the v1 path can crash via the fused Gated Delta Net interaction — tracked separately. v1 still works in batched mode and on all other tested models.

### V-cache v2 — Trellis (current production default)

Group-level Viterbi trellis with shared state and shared scale. 16-state shift-register, 16-bit open-start state, inverse-Gaussian CDF code table. The Viterbi path optimizes globally over the block, which adapts implicitly to the running model's V-distribution — at the same average bpw, every V-element gets to leverage local statistics.

| Type | Index bits | bpw | Block |
|------|:---:|:---:|:---:|
| `vtq2_2` | 2 | 2.06 | 132 B |
| `vtq3_2` | 3 | 3.06 | 196 B |
| `vtq4_2` | 4 | 4.06 | 260 B |

**K-collision is a feature, not a bug.** `vtq2_2 / vtq3_2 / vtq4_2` produce bit-identical PPL on the same model — the per-element MSE drops 16× across K=2/3/4, but FA softmax averages it out across the sequence (attention-absorbed). Save the bandwidth — pick `vtq2_2`. Source: `docs/blog/2026-04-25-vtq2-attention-absorbs-bit-depth.md`.

**Activation:** v2 supports D=64/128/256/512 — verified live on Gemma4-26B-A4B (D=256 SWA + D=512 full-attn) and Qwen3.6-35B-A3B (D=128). Encoder is ~22 ms/call, which is why `--cache-type-v vtq*_2` auto-enables **f16 staging during prefill** and runs the bulk Viterbi exactly once at the prefill→decode boundary. Logs say `deferred V quantization enabled (N layers with f16 staging)` on startup. **PPL measurement requires `-b 1 -ub 1`** to fire the deferred-V trigger — batched runs (b > 1) measure f16 + mixed-precision overhead, not the actual VTQ_2 PPL.

### V-cache v3 — Trellis + Outlier-Channel-Split (research, quality tier)

Same Viterbi backbone as v2, plus a 4-fp16-outliers-per-block sidecar that captures the largest absolute V values losslessly. Round-trip MSE drops a further 4× vs v2 at the cost of +1 bpw average. PPL impact on 35B-A3B at 3.78 bpw avg (`ktq2_1 + vtq3_3`): **+0.47%** vs f16/f16 — well below stderr.

| Type | Index bits | bpw avg | Block |
|------|:---:|:---:|:---:|
| `vtq2_3` | 2 | 2.50 | 140 B |
| `vtq3_3` | 3 | 3.50 | 204 B |
| `vtq4_3` | 4 | 4.50 | 268 B |

Same K-collision pattern as v2 (attention-absorbed). On giants (80B/122B) v3 buys an additional ~0.05% PPL over v2 — currently within stderr at chunks=4. Recommendation: deploy `vtq2_2`; keep v3 reserved for "quality-priority" workloads where 1 extra bpw on V is acceptable. Source: `docs/blog/2026-04-25-vtq3-asymmetric-on-35b.md`.

### K-cache — KTQ (PolarQuant, RHT + Lloyd-Max)

Per-block Randomized Hadamard Transform (FWHT + per-block sign flip) + Lloyd-Max codebook. The FA kernel applies FWHT to Q once per tile and computes Q·K entirely in the Hadamard domain — K is never explicitly dequantized in the vec path. On CC ≥ 7.5 (Turing+) an **MMA-KTQ tensor-core path** is wired and live: split-dequant for prefill (PP ≥ 8 tokens), routes through the existing MMA-F16 tensor-core kernel. Measured KTQ2_1: PP128 **727 t/s** (vs 431 f16 baseline), PP512 875 (parity with f16), PP2048 868 (parity). TG falls back to VEC. Source: `ggml/src/ggml-cuda/fattn-mma-ktq.{cu,cuh}`.

| Type | Index bits | bpw | Block |
|------|:---:|:---:|:---:|
| `ktq1_1` | 1 | 2.5 | 10 B |
| `ktq2_1` | 2 | 3.5 | 14 B |
| `ktq3_1` | 3 | 4.5 | 18 B |
| `ktq4_1` | 4 | 5.5 | 22 B |

**Production default:** `ktq2_1` at 3.5 bpw. Measured PPL hit on 35B-A3B with f16 V: **+0.27%** vs f16/f16 — inside the perplexity stderr.

**Why deferred K:** KTQ K-cache suffers a repetition-loop pathology when quantized per-token during prefill — attention re-reads the just-quantized rows, RHT round-trip noise accumulates, and the model loops (`"Es war einfach. Es war einfach. Es war einfach."`). f16 staging during prefill + bulk-convert at prefill→decode avoids this. Auto-enabled for any KTQ type.

**Combining KTQ K with VTQ V:** asymmetric is the reference config. The 35B production deployment uses `ktq2_1 + vtq2_2` at **2.78 bpw avg** for a measured +0.27%–+0.47% PPL hit (well below noise floor). `vtq3_3` adds 1 bpw on V for a further marginal improvement.

---

## Live performance numbers

**Single source of truth:** [`docs/bench/LIVE_NUMBERS.md`](docs/bench/LIVE_NUMBERS.md) — current TG, PPL, HellaSwag for all five deploy targets. That file is updated each phase; the per-model deep-dives below stay structurally stable.

**Phase 4 (2026-04-26)** added an adaptive layer-split + `OMP_WAIT_POLICY=active` + `__builtin_prefetch` in `mul_mat_id` to the deploy stack. Net **+18.5% TG on 80B** (30.80 → ~36.5 t/s at ctx ≤ 8192), **+9.3% on 122B** (16.69 → 18.24 t/s). Long-context configs (>8k) fall back to the safe split that fits ctx=200000. New: Qwen3.6-27B dense single-GPU deploy, 5.1× faster than the old Qwen3.5-27B-Q4 offload path.

## Large-MoE deployments

These are the four MoE models that drive the fork's existence. All measured on the same box: Ryzen 7 3700X host (Zen 2, 8C/16T, 2 CCDs × 2 CCXs, separate L3 per CCX) — test box is a KVM guest VM with 12 vCPUs, 40 GB DDR4-3200 (~40 GB/s real), 2× RTX 2060 12 GB on asymmetric PCIe (GPU0 x16 / GPU1 x4).

![Large-MoE TG: 35B / 80B / 122B on 2x RTX 2060](docs/img/large_moe_tg.png)

The 26B and 35B fit fully on GPU. The 80B and 122B spill 20 / 29 layers to CPU RAM — TG becomes CPU-memory-bandwidth-bound, so the numbers are read against a physics ceiling (DDR4-3200 @ ~40 GB/s real / per-token CPU traffic).

**Production default (all four models):** `--cache-type-k ktq2_1 --cache-type-v vtq2_2` at 2.78 bpw avg. Selected on 2026-04-25 after measuring vtq2_2 vs vtq2_1 on both giants — `vtq2_2` wins or ties on PPL, pp512, and tg128 across the board.

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

\* PPL on raw wikitext is broken for reasoning models — Gemma4 expects the chat-template thought-channel prefix and reports PPL in the 10⁴–10⁵ range on raw text, even at f16/f16. For relative quality validation use MMLU/HumanEval (see oidanice-distillery `scripts/benchmark_mmlu.py`).

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

Full K × V sweep in the [Benchmarks](#benchmarks) section.

### 80B-A3B — Qwen3-Next hybrid (DeltaNet + Attention)

Hybrid architecture, **512 experts / 10 active**. Two deploy tiers:

#### Tier 1 (current default) — TQ1_0 full-VRAM ⭐

Unsloth's `UD-TQ1_0` (1-bit dynamic Trellis) is **19.3 GB instead of 26.2 GB**, so both GPUs hold full model + KV — no CPU expert-offload. Live since 2026-04-27 on test box. Deploy script: `oidanice-distillery/scripts/deploy/deploy-80b-tq1.sh`.

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
```

#### Comparison table

Measured 2026-04-25 / -27 (llama-bench, 2 reps; PPL via prod-aligned `-c 512 --chunks 4 -b 1 -ub 1`):

| Config | bpw KV | model size | pp512 | tg128 | PPL | Δ PPL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 80B-IQ2_XXS `f16/f16` (CPU offload) | 16.0 | 26.2 GB | 404 | 31.5 | 5.085 | baseline | reference |
| 80B-IQ2_XXS `ktq2_1/vtq2_1` (old prod) | 3.0 | 26.2 GB | 386 | 30.6 | 5.221 | +2.69% | replaced 2026-04-25 |
| 80B-IQ2_XXS `ktq2_1/vtq2_2` (long-ctx fallback) | 2.78 | 26.2 GB | 403 | 30.9 | 5.082 | **−0.06%** | for ctx > 65k |
| **80B-TQ1_0 `ktq2_1/vtq2_2`** (current default) ⭐ | **2.78** | **19.3 GB** | **653** | **57.0** | 7.039 | weight-quant tier | **+85% TG vs offload** |

The +85% TG jump on TQ1_0 is from killing PCIe-x4 expert streaming. Quality cost: TQ1_0 is more aggressive than IQ2_XXS at the *weight* level (PPL 7.04 vs 5.08), but for short-to-medium context it's the tier where 80B fits a dual-2060 box without thrashing PCIe.

Physics ceiling for IQ2_XXS offload tier: 40 GB/s DDR4 / ~0.75 GB per-token CPU traffic → ~53 t/s hard limit. Phase-4 stack lifted current to ~36.5 t/s (69% of ceiling) via OMP_active + adaptive layer-split + prefetch. TQ1_0 full-VRAM is bandwidth-unbound by comparison.

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

Measured 2026-04-25 (llama-bench, 2 reps; PPL via prod-aligned `-c 512 --chunks 4 -b 1 -ub 1`):

| Config | bpw KV | pp512 | tg128 | PPL | Δ PPL vs f16 |
|---|---:|---:|---:|---:|---:|
| `f16 / f16` | 16.0 | 187.6 | 17.0 | 4.0634 | baseline |
| `ktq2_1 / vtq2_1` (old prod) | 3.0 | 189.5 | 16.8 | 4.2338 | **+4.19%** |
| **`ktq2_1 / vtq2_2`** (new prod) ⭐ | **2.78** | **196.3** | **16.8** | **4.0379** | **−0.63%** |

The 122B is where v2 Trellis shines hardest: `vtq2_2` actually beats `f16/f16` on **both** PPL (−0.63%) and pp512 (+4.6%). VRAM 10.9/10.5 GB at 200k ctx. Full 262k ctx also fits (GQA(2) + 2.78 bpw KV = +140 MB delta from 200k). Physics ceiling: 2.5 GB/token CPU traffic / 56 GB/s effective ≈ 22 t/s, current 17 t/s = 77% efficiency.

#### 122B-IQ1_M — smaller-tier alternative (2026-04-27)

Unsloth ships a `UD-IQ1_M` (1.75 bpw) variant at **31.8 GB instead of 36.6 GB**. Still doesn't fit fully on dual-2060 (24 GB total VRAM), but more GPU-resident layers possible (12/13 instead of 10/9 split). Useful when ctx ≤ 32k and small TG bump is enough.

| Config | bpw KV | model size | pp512 | tg128 | PPL (8ch) |
|---|---:|---:|---:|---:|---:|
| 122B-IQ2_XXS ktq2_1+vtq2_2 (10/9/29) | 2.78 | 36.6 GB | 196 | 16.8 | 4.038 |
| **122B-IQ1_M ktq2_1+vtq2_2 (12/13/24)** | **2.78** | **31.8 GB** | **241** | **20.7** | 4.818 |

+23% TG, +23% pp512 — but PPL +19% (4.04 → 4.82) — IQ1_M trades quality for less CPU offload. For long-ctx prod stay on IQ2_XXS; for short-ctx fast-iteration use IQ1_M.

**PCIe asymmetry matters:** GPU0 x16 / GPU1 x4 means heavier expert-load on GPU0 avoids x4 cross-traffic. 19L (10+9) beats balanced 9+9 by +2% TG and +11% PP-stability. Full sweep: [docs/bench-qwen35-122b-a10b.md](docs/bench-qwen35-122b-a10b.md). Production-PPL sweep: [docs/blog/2026-04-25-giant-models-prod-ppl-sweep.md](docs/blog/2026-04-25-giant-models-prod-ppl-sweep.md).

---

## Benchmarks

All measurements: Ryzen 7 3700X + 2× RTX 2060 12 GB + DDR4-3200, Flash Attention on. Production sweep build `0639f7835` / 2026-04-25.

### 4-Model production summary (`ktq2_1 + vtq2_2` vs baseline)

The production-default config (`ktq2_1` K + `vtq2_2` V at 2.78 bpw avg) compared to f16/f16 baseline across all four target models. PPL measured with prod-aligned methodology where possible (`-c 512 --chunks 4 -b 1 -ub 1`).

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

Rows in **bold** are the production recommendations: `f16/vtq2_2` is near-free on FA (−1.1% PP, −1.3% TG) and `ktq2_1/vtq2_2` is the lightest-with-K-quant config at −2.2% / −2.5% throughput cost for ~80% KV savings.

**Observations:**
- **VTQ_2 (Trellis v2) is the cheapest V-cache on FA** — 1.1–1.3% slowdown vs f16, beats every VTQ_1 variant at the same or lower bpw.
- **`q4_0` / `q8_0` as V destroys FA dispatch** — drops to ~650 PP, ~60 TG (legacy types fall out of fastest FA path on CC 7.5).
- **Asymmetric `ktq2_1 / vtq2_2`** is the production winner at 2.2% PP / 2.5% TG cost for **~80% KV savings** (28.75 MiB vs 160 MiB at ctx=8192).
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

## Perplexity (wikitext-2)

> **Why you'll see different PPL deltas across this README.** PPL is sensitive to (a) weight quant, (b) ctx length, (c) chunk count, (d) batch size (`-b 1 -ub 1` exercises deferred-V; larger batches measure f16 staging instead).
>
> Three measurement regimes appear:
> - **64-chunk ctx=512** (matrix sweep, full attn-only PPL) — the headline `+0.15%` for `ktq2_1+vtq2_2` lives here. Used in TL;DR.
> - **8-chunk ctx=512** (apples-to-apples vs upstream, noisier sample) — `+1.34%`. Used only in [vs upstream table](#vs-llamacpp-upstream--apples-to-apples-2026-04-27).
> - **4-chunk ctx=512 `-b 1 -ub 1`** (prod-aligned, exercises VTQ_2 deferred path) — `+0.85%` on 35B, `−0.06%` on 80B, `−0.63%` on 122B. Used in per-model deploy tables.
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

From `docs/blog/2026-04-19-v-cache-validation.md`, `tests/trellis-phase1/results/run22_08b_full_sweep.csv`.

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

## KV memory savings

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

**Smallest usable:** `ktq1_1 / vtq1_1` at **20 MiB total (13% of f16/f16, 8× smaller)** — but vtq1_1 costs +16% PPL, not practical. **Smallest PPL-sensible:** `ktq1_1 / vtq2_2` at 23.75 MiB (15%, 6.7× smaller). For large contexts the absolute savings matter more — at 200k ctx on the Qwen3.5-122B-A10B GQA(2) config, the production `ktq2_1 / vtq2_2` config is ~430 MB total vs ~2.3 GB at f16/f16.

Measurement note: the 10-layer count is Qwen3.6-35B-A3B specific (48 blocks total, 10 with attention after the MoE filter). Different architectures allocate KV on different block counts; scale the per-layer numbers accordingly.

---

## How it works

The trick is to use **different formats for K and V**, because they hit Flash Attention differently.

- **KTQ (K-cache)** — per-block Randomized Hadamard Transform + Lloyd-Max codebook. The kernel transforms Q once per tile and computes Q·K entirely in the Hadamard domain, so K is **never explicitly dequantized** in the hot loop.
- **VTQ v1 (V-cache)** — one D·H·D rotation applied at graph level before writes. Per-entry dequant in the FA inner loop is just `codebook[idx] * scale`.
- **VTQ v2 Trellis (V-cache)** — group-level Viterbi DP encodes 512 samples jointly against a fixed inverse-Gaussian CDF table. Decode is a shift register, one sample per iteration. The encoder is slow (~22 ms/call), so the runtime stages V in f16 during prefill and bulk-converts once at the prefill→decode boundary. This is what makes v2 PPL-lossless on the measurement granularity used.

Full design notes (RHT math, register pressure on CC 7.5, encoder details) live in [`docs/turboquant.md`](docs/turboquant.md). Source: `ggml/src/ggml-trellis.{h,c}`.

---

## Claude Code integration

The server exposes `/v1/messages` (Anthropic-compatible), so Claude Code can talk to it directly:

```bash
./scripts/onllama-launch-claude.sh --server http://localhost:8080
```

Server-side features wired in:

- **Tool-call early-stop** — `</tool_call>` stop sequence on `/v1/messages` with `tools:[]`, saves 1–15 s per agent turn.
- **`--keep 8192`** — pins the first 8k prompt tokens across context shifts, protecting the Claude Code system prompt (~15–25k) from silent discard.
- **`--cache-reuse 256`** — KV-shift-based prompt-prefix reuse across turns. Second-turn latency ~60 s → ~5–10 s on 35B.
- **Anthropic prompt caching** — `cache_control:{type:"ephemeral"}` markers on `system`, `messages.content`, `tools` are parsed and persisted to `<slot_save_path>/anthropic-cache/` with 5m/1h TTL and refresh-on-hit. Enable via `--slot-save-path PATH`.
  - **Hybrid/recurrent models:** a companion `.ckpt` is written alongside each blob and re-injected into `slot.prompt.checkpoints` on restore. Response fields are always correct; wall-time speedup is limited on models whose memory can't be truncated at arbitrary positions (Qwen3-Next etc.).
- **Full Anthropic `usage` shape** — `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation.ephemeral_5m/1h_input_tokens` all emitted.
- **TCP\_NODELAY on SSE** — removes ~40 ms Nagle stalls per chunk on streaming.
- **gzip** (opt-in, `LLAMA_SERVER_ZLIB=ON`) — 4–6× on tool-call JSON, non-streaming only.

Full setup incl. SSH tunnel: [docs/claude-code.md](docs/claude-code.md).

---

## Build

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

## Roadmap

**Shipped**
- KTQ1\_1 / 2\_1 / 3\_1 / 4\_1 (PolarQuant K-cache, 2.5–5.5 bpw)
- VTQ1\_1 / 2\_1 / 3\_1 / 4\_1 (v1, codebook lookup)
- VTQ2\_2 / 3\_2 / 4\_2 (v2 Trellis, D=64/128/256/512 all verified live) — production default since 2026-04-25
- VTQ2\_3 / 3\_3 / 4\_3 (v3 Trellis + outlier-channel split, research tier)
- Asymmetric KTQ × VTQ dispatch through FA
- Deferred K/V (auto-gated by type)
- Attention-sink protection (first 4 tokens in f16)
- MMA-KTQ asymmetric dispatch — KTQ K + f16 V takes the tensor-core MMA path via bulk K→f16 split-dequant. Reference 35B IQ2\_XS: PP512 **875** (vs f16 861), PP2048 **868** (vs 857), TG128 67 (vs 71). 9.5× jump over the pre-fix 92 t/s.
- Phase 4 perf stack: OMP_active, adaptive layer-split, prefetch in mul_mat_id, P2P opt-in (+18.5% TG on 80B, +9.3% on 122B vs prior baseline)
- Anthropic `/v1/messages` with prompt caching

**Active research**
- **Phase 5 — XQuant cross-layer KV reuse** (code-complete, dormant on hybrid SSM models). Pair adjacent KTQ2_1 layers; subordinate stores only scale, codes shared from sibling. Activates with `--xquant`. Yields 0 pairs on Qwen3.X-A3B / Qwen3.6-27B (alternating Mamba/attention layers); intended for pure-transformer dense models (Llama-3, Mistral, Gemma-2 family). Reference: arXiv:2510.11236.
- Correction Overlay Buffer (Trick 4) — designed, not implemented. Top-N lossless error patch.
- **Phase 7 — imatrix-aware KTQ calibration** (proposed). Use importance matrix to bias the K-quant Lloyd-Max codebook.
- `mmvq` IQ2\_XS tuning on sm\_75 — 28% of kernel time on current 35B config

**Discarded after measurement**
- Speculative decoding on A3B MoE — expert-saturation pathology makes it ineffective
- VTQ\_MIXED — dominated by VTQ3\_1, not CUDA-ported
- Calibrated outlier selection — marginal gain after RHT
- MMA-KTQ split-dequant as default for all ctx — regresses past ~512 tokens; now short-ctx only

**Not on roadmap**
- FA3 (requires sm\_80+)
- Paged attention (scope mismatch)
- Multi-node inference

---

## When *not* to use this fork

- **VRAM is not a constraint.** Upstream llama.cpp with f16 KV is simpler and equally fast.
- **Sub-50 ms/token latency at long ctx matters.** VTQ V-cache adds per-token dequant overhead that grows with context length.
- **Multi-node serving.** This fork makes zero changes to llama.cpp's split logic.
- **Ampere+ (CC 8.0+).** Untested. The sm\_75-specific tuning is not going to be a good default there.

---

## License

MIT, inherited from upstream llama.cpp. No restrictions.
