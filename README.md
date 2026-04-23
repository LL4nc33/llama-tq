# llama-tq

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Personal llama.cpp fork focused on KV-cache quantization. K-cache and V-cache can use different quant types with different dequant paths inside the Flash Attention kernel.

**For whom:** Users running large models with long context on limited VRAM who accept some PPL cost for significant KV memory savings. Tested on NVIDIA Turing (CC 7.5, 2× RTX 2060 12 GB).

**What it does:**
- **Asymmetric K/V** — `KTQ{1..4}_1` for K (RHT + Lloyd-Max + FWHT inside FA), `VTQ{1..4}_1` for V (codebook lookup in FA inner loop). CLI: `--cache-type-k` / `--cache-type-v`.
- **Random Hadamard Transform (RHT) + Lloyd-Max codebooks** — outlier diffusion + 1D-optimal quantization (PolarQuant / TurboQuant Stage 1).
- **Trellis-Coded V family** (`VTQ{2,3,4}_2`) — group-Viterbi encoder + shift-register decoder at 2.06 / 3.06 / 4.06 bpw. CPU reference + CUDA dequant. Currently functional on D=128, broken on D=256.
- **Laplace-fit codebooks** (post-RHT data is Laplace-distributed, not Gaussian).
- **Deferred-V quantization** — f16 staging during prefill, bulk Viterbi at prefill→decode transition.
- **Attention-sink protection** — first 4 tokens kept in f16.
- **Flash Attention 2 vec-path** extended for VTQ V-cache dispatch.
- **MMA-KTQ path** — Tensor-Core K-cache dispatch for 3/4-bit KTQ on CC ≥ 8.0 (untested locally).
- Standard **ggml infrastructure** (type_traits, CPU fallback, CUDA FA kernels).

**What it is not:**
- Not faster than upstream. TG throughput matches upstream llama.cpp; this fork trades some perf for VRAM.
- Not a serving system — single-node, no paged attention, no batching focus.
- Not tested on Ampere/Hopper. sm_75 is the only calibration target.

**Status:** Works day-to-day on Qwen3.5-35B-A3B with 400k context on the author's setup. V-cache choice is workload-dependent — see the trade-offs below before picking a type.

> **Measured on 2x RTX 2060 12GB (CC 7.5), 5 model/quant pairs:**
> - `q8_0` K + `vtq2_1` V (5.5 bpw avg): **+5.1% to +10.0%** PPL Δ depending on model, ~65% KV VRAM vs f16
> - `q8_0` K + `vtq3_1` V (6.25 bpw avg): **+0.6% to +2.5%** PPL Δ ← **recommended for fitting models**
> - TG128 overhead: **−1% to −4%** with `vtq*` V-cache on short context
> - CUDA only. PPL measurements are at 3 wikitext-2 chunks (noisy), proper 64+ chunk reruns pending.

> **⚠ Long-context TG (historical):** an earlier 2026-04 build measured a 5.5× TG regression
> (12.4 vs 67.7 tok/s) for `vtq3_1` V-cache at 400k ctx on 35B. Not re-measured against the
> phase3 FA-path fixes. **`vtq2_1` at 400k is tested at −3% TG** (see
> [Quick Start](#quick-start)); short-context `vtq3_1` on the current build is stable at −3% to −4% TG.

> **Bottleneck note (2026-04-23):** nvprof on the current config shows
> FA-vec kernel is only 6.4% of kernel time. The real bottleneck at 67 tok/s baseline is
> `mmvq` (IQ2_XS expert matmuls at 28%), which is already upstream-optimized. Further TG
> improvements on this hardware + model combination are likely small.

![PPL vs KV bpw](docs/img/ppl_vs_bpw.png)

---

## Quick Start

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc) --target llama-server

# Recommended: high-quality K + balanced V (3.07% rel MSE)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq3_1 \
    -fa on -ngl 99

# Memory-extreme (for context > 200k on 12 GB VRAM): vtq2_1 (13.25% rel MSE)
./build/bin/llama-server -m model.gguf \
    --cache-type-k q8_0 --cache-type-v vtq2_1 \
    -fa on -ngl 99

# Asymmetric KTQ2_1 K + VTQ2_1 V + 400k ctx on 2x RTX 2060 12GB
# (the setup this fork is developed on, Qwen3.6-35B-A3B)
./build/bin/llama-server -m /path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 \
    -c 400000 -ngl 99 --flash-attn on --no-mmap --parallel 2 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
    --cache-reuse 256 \
    -ub 512 -ts 12,12 --jinja --reasoning off
# Notes:
#  - Both K *and* V now quantized with TurboQuant family — maximum VRAM savings.
#    Qwen3.6-35B-A3B UD-IQ2_XXS at 400k ctx parallel 2:
#      KTQ2_1 K + f16 V   → 22.6 GB total VRAM, 68 tok/s TG
#      KTQ2_1 K + VTQ2_1 V → 19.3 GB total VRAM, 66 tok/s TG  ← this config
#    Saves ~3.3 GB (-15%) at 400k parallel 2; ~6 GB (-27%) at 200k parallel 1.
#    Only 2–3% TG regression, no measurable quality loss on Qwen3.6-35B.
#  - Deferred K quantization auto-enables for KTQ types (f16 staging during
#    prefill, bulk-convert at prefill→decode). Avoids repetition-loop pathology.
#  - --parallel 2 gives two 200k-ctx slots on this config.
#  - --cache-reuse 256 reuses KV cache across turns for agent clients with
#    stable system prompts (Claude Code, Open WebUI). Cuts multi-turn
#    prefill latency dramatically — see docs/claude-code.md.
#  - -ts 12,12 splits weights evenly across two 12 GB GPUs.
#  - WARNING: On very small models (<1B params) the 2+2-bit combo loses too
#    much signal and produces garbage output. Use ≥7B for asymmetric KTQ+VTQ.
```

### Use it from Claude Code

llama-tq's server exposes `/v1/messages` (Anthropic-compatible), so Anthropic's
official Claude Code CLI can talk to it directly:

```bash
./scripts/onllama-launch-claude.sh --server http://localhost:8080
```

Server-side optimizations for Claude Code already wired into the reference config:

- **Tool-call early-stop** — injects `</tool_call>` stop sequence on `/v1/messages`
  requests carrying `tools:[]`, skipping the 2-10 dribble tokens Qwen3 emits after
  a finished tool call. Saves 1-15 seconds per agent turn.
- **`--keep 8192`** — pins the first 8k prompt tokens across context shifts,
  protecting the Claude Code system prompt (~15-25k tokens) from silent discard.
- **`--cache-reuse 256`** — KV-shift-based prompt-prefix reuse across turns.
  Second-turn latency drops from ~60s to ~5-10s on a 35B model.
- **Full Anthropic `usage` response shape** — `cache_read_input_tokens`,
  `cache_creation_input_tokens`, and `cache_creation.ephemeral_5m/1h_input_tokens`
  fields are all emitted (strict clients no longer error on missing fields).
- **Anthropic prompt caching (`cache_control`)** — `cache_control:{type:"ephemeral"}`
  markers on `system`, `messages.content`, and `tools` blocks are parsed, validated
  against the 4-breakpoint spec cap, and their KV state is persisted to disk under
  `<slot_save_path>/anthropic-cache/`. A second request with the same cached prefix
  restores KV from file and skips prefill on that segment. TTL 5m/1h with lazy
  delete + refresh-on-hit. Enable with `--slot-save-path PATH` (auto-enables
  `--anthropic-cache`). 400 on >4 breakpoints per spec.
  - **Hybrid/recurrent model caveat:** for models classified as hybrid memory
    (e.g. Qwen3-MoE), a companion `<filepath>.ckpt` is written alongside the
    main KV blob and injected into `slot.prompt.checkpoints` on restore. This
    passes the hybrid-arch gate in the prefix-match path. On these models, a
    subsequent `memory_seq_rm` during the new user turn may still trigger a
    re-prefill because the hybrid memory cannot be truncated at arbitrary
    positions — response fields are correct, but wall-time speedup is limited.
    Pure SWA / dense models get the full prefill-skip benefit.
- **TCP_NODELAY on SSE** — disables Nagle buffering so streaming token deltas
  flush immediately (no ~40ms stalls per chunk).
- **gzip response compression** (opt-in, `LLAMA_SERVER_ZLIB=ON` at build time) —
  4-6× ratio on tool-call JSON. Non-streaming responses only.

Full setup (including SSH tunnel for a remote server): [docs/claude-code.md](docs/claude-code.md).

## Recommended Configurations

Measured on Qwen3.6-35B-A3B (UD-IQ2_XXS & Q4_K_M); PPL is sensitive to weight quant.

| Config | K | V | Avg bpw | PPL Δ | Notes |
|--------|---|---|:---:|:---:|---|
| Safe | `q8_0` | `vtq3_1` | 6.25 | +1.1% to +2.1% | quality-first |
| Compact | `q8_0` | `vtq2_1` | 5.5 | +6.6% to +8.5% | VRAM/quality tradeoff |
| Asymmetric (reference) | `ktq2_1` | `vtq2_1` | 3.5 | not yet re-measured on 64-chunk PPL | the reference config |

---

## Benchmarks

All benchmarks on **2× NVIDIA RTX 2060 12 GB** (CC 7.5, PCIe 3.0), Flash Attention on, 1 rep, pp512 / tg128, build `bc7c2e3d3`. Full K × V matrix in [docs/bench-safe-2026-04-23.md](docs/bench-safe-2026-04-23.md).

### Dual-GPU 35B-A3B MoE (24 GB VRAM, no offload, `-ts 12,12`)

Qwen3.6-35B-A3B-UD-IQ2_XXS (10 GB weights, doesn't fit on one 12 GB card → real dual-GPU split).

| K | V | PP512 tok/s | TG128 tok/s | ΔPP | ΔTG |
|---|---|:---:|:---:|:---:|:---:|
| f16 | f16 | **986** | **72.8** | baseline | baseline |
| ktq2_1 | f16 | 938 | 70.6 | −5% | −3% |
| f16 | vtq2_1 | 909 | 71.4 | −8% | −2% |
| **ktq2_1** | **vtq2_1** | **877** | **69.4** | **−11%** | **−5%** |
| q8_0 | q8_0 | 974 | 70.0 | −1% | −4% |
| q4_0 | q4_0 | 961 | 69.2 | −3% | −5% |
| q8_0 | vtq2_1 | 874 | 69.6 | −11% | −4% |

Notes:
- KTQ K + f16 V loses ~5% PP for ~40% KV-cache savings. Good default for dual-GPU users.
- Full asymmetric `ktq2_1 + vtq2_1` = 3.5+2.5 bpw averaged over K+V, smallest footprint at ~11% PP cost.
- `q8_0` as *V*-cache hits the non-MMA FA dispatch on CC 7.5 — it's not broken, but `vtq2_1` is measurably faster at similar quality cost.

### Single-GPU 35B-A3B MoE (12 GB VRAM, expert-offload to CPU RAM)

Same model, single RTX 2060, `-ot blk.1[5-9].ffn_.*_exps.=CPU` (5 MoE layers to CPU). This is the "light offload" configuration — most layers stay on GPU, only the first few MoE experts move to CPU. Full 30-layer offload drops PP to ~200 tok/s as expected.

| K | V | PP512 tok/s | TG128 tok/s | ΔPP | ΔTG |
|---|---|:---:|:---:|:---:|:---:|
| f16 | f16 | 547 | 57.1 | baseline | baseline |
| ktq2_1 | f16 | 521 | 55.6 | −5% | −3% |
| f16 | vtq2_1 | 517 | 51.3 | −6% | −10% |
| ktq2_1 | vtq2_1 | 509 | 52.6 | −7% | −8% |

PP scales ~55% of dual-GPU because PCIe traffic to CPU experts is the bottleneck. Reference 200k-ctx config:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
    -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --host 0.0.0.0 --port 8791 --jinja --flash-attn on \
    -c 200000 -ngl 99 --no-mmap \
    -ot 'blk.1[5-9].ffn_.*_exps.=CPU' \
    -ot 'blk.[2-4][0-9].ffn_.*_exps.=CPU' \
    --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
    --cache-reuse 256 \
    --parallel 1 -ub 512 --reasoning off
```

### Dense model — Qwen3.5-27B (single-GPU)

Qwen3.5-27B-UD-IQ2_XXS (7.97 GiB, fits on one 12 GB card). TG is CPU-bound because the dense weights already saturate memory bandwidth — KV-cache quant barely moves it.

| K | V | PP512 tok/s | TG128 tok/s | ΔPP | ΔTG |
|---|---|:---:|:---:|:---:|:---:|
| f16 | f16 | **409** | **15.0** | baseline | baseline |
| ktq2_1 | f16 | 392 | 14.7 | −4% | −2% |
| f16 | vtq2_1 | 375 | 14.7 | −8% | −2% |
| ktq2_1 | vtq2_1 | 374 | 14.6 | −9% | −3% |
| q8_0 | q8_0 | 393 | 14.6 | −4% | −3% |
| q4_0 | q4_0 | 391 | 14.5 | −4% | −3% |
| q8_0 | vtq2_1 | 360 | 14.5 | −12% | −3% |

Notable: dual-GPU split on 27B-Dense buys nothing (the model fits on one GPU and llama.cpp's `-ts` heuristic skips the split). The GPU1 only holds ~90 MiB CUDA context and stays idle. See the appendix bench for confirmation.

### Takeaways

- **If the model fits on one GPU, single-GPU wins.** The llama.cpp `-ts` split only kicks in when weights don't fit. Dual-RTX-2060 is only worth it for >12 GB models.
- **KTQ × f16 V is the lowest-overhead compression.** 4–5% PP cost, 2–3% TG cost, ~40% KV savings.
- **Full asymmetric `ktq2_1 + vtq2_1` is the maximum-compression config.** 7–11% PP cost, 3–5% TG cost, ~80% KV savings vs f16.
- **Skip `q8_0` as V-cache on CC 7.5.** Falls out of the fastest FA dispatch; VTQ is both smaller and faster.

### Extreme case — Qwen3.5-122B-A10B (34 GB weights, expert-offload)

This model has **256 experts / 8 active per token** and **2 KV heads (GQA)** — the expert sparsity is enough that even a 122B-parameter model runs on 2× RTX 2060 with all 48 FFN experts offloaded to CPU RAM.

Config: `-ts 12,12`, `-ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(up|down|gate)_exps\.=CPU'` (all 48 layers' expert FFNs on CPU, attention + shared FFN on GPU), `--no-mmap`.

| K | V | ctx | VRAM used (GPU0/GPU1) | PP tok/s | TG tok/s |
|---|---|---|---|:---:|:---:|
| f16 | f16 | 2k (bench) | — | ~56 (pp128) | ~13.6 (tg32) |
| ktq2_1 | vtq2_1 | 2k (bench) | — | **148** (pp512) | **12.6** (tg128) |
| ktq2_1 | vtq2_1 | **200k** | 4.4 / 4.8 GB | — | — |
| ktq2_1 | vtq2_1 | **262k** (max) | 6.7 / 6.4 GB | — | — |

A full `262144` context fits in ~13 GB total VRAM across both cards because the tiny 2-head KV + KTQ/VTQ compression makes per-token KV ~9 KB. Most of the remaining VRAM is the compute buffer, not the cache itself. The active 10B parameters per token means TG is CPU-RAM-bandwidth-bound rather than GPU-bound.

### Perplexity (wikitext-2, 512 ctx, 3 chunks)

#### Qwen3.6-35B-A3B (UD-IQ2_XXS)

| K-Cache | V-Cache | KV bpw | PPL | vs baseline |
|---------|---------|:---:|:---:|:---:|
| f16 | f16 | 16.0 | **5.967** | -- |
| q8_0 | q8_0 | 8.5 | 6.006 | +0.65% |
| q4_0 | q4_0 | 4.5 | 6.001 | +0.57% |
| f16 | vtq3_1 | 10.0 | 6.030 | **+1.05%** |
| q8_0 | vtq2_1 | 5.5 | **6.361** | **+6.6%** |
| f16 | vtq2_1 | 9.3 | 6.378 | +6.9% |

On IQ2_XS/IQ2_XXS weights, `vtq3_1` sits at +1.05–1.8% PPL and `q8_0 + vtq2_1` at +6.6–7.2%. The Q4_K_M results below show larger deltas. See the [Observation](#perplexity-wikitext-2-512-ctx-3-chunks) block after the Q4_K_M tables.

#### Qwen3.6-35B-A3B (Q4_K_M)

| K-Cache | V-Cache | PPL | vs baseline |
|---------|---------|:---:|:---:|
| f16 | f16 | **5.127** | -- |
| f16 | q4_0 | 5.129 | +0.04% |
| q4_0 | q4_0 | 5.169 | +0.8% |
| f16 | vtq3_1 | 5.177 | **+1.0%** |
| q8_0 | vtq3_1 | 5.232 | +2.1% |
| q4_0 | vtq2_1 | 5.498 | +7.2% |
| q8_0 | vtq2_1 | 5.563 | +8.5% |

**Caveat:** 3-chunk wikitext-2 measurements are noisy. A proper 64-chunk re-run is pending. `vtq3_1` stays at +1.0–2.1% across both configs.

### KV-Cache Memory (4096 ctx)

| Config | KV Size | Savings vs f16 |
|--------|:---:|:---:|
| f16 / f16 | 40.0 MiB | -- |
| q8_0 / vtq2_1 | 13.8 MiB | **65%** |
| q4_0 / vtq3_1 | 10.6 MiB | **73%** |
| q4_0 / vtq2_1 | 8.7 MiB | **78%** |
| ktq2_1 / vtq2_1 | 7.5 MiB | **81%** |

### Comparison with other approaches

Other KV-quant forks exist (TheTom's symmetric TurboQuant, buun's TCQ trellis). They run on different hardware, different weights, and different metrics, so a direct side-by-side table would be misleading. If you want cross-project numbers, run them yourself on the same model + hardware.

---

## Available Cache Types

<details>
<summary><strong>KTQ (K-Cache TurboQuant)</strong></summary>

Per-block Randomized Hadamard Transform (FWHT + per-block sign flip) + Lloyd-Max codebook. The FA kernel applies FWHT to Q once per tile and computes the Q·K dot product in the Hadamard domain, avoiding a per-K inverse FWHT.

| Type | Index bits | bpw | Block | Notes |
|------|:---:|:---:|:---:|---|
| `ktq1_1` | 1 | 2.5 | 10 B | extreme compression |
| `ktq2_1` | 2 | 3.5 | 14 B | good quality |
| `ktq3_1` | 3 | 4.5 | 18 B | near-lossless |
| `ktq4_1` | 4 | 5.5 | 22 B | lowest-PPL KTQ |

**Note:** Combining KTQ K with VTQ V at low bit-widths showed super-additive PPL degradation in my tests (not isolated to a single cause; likely softmax-sensitivity to correlated K and V noise). I recommend `q8_0` or `q4_0` for K when pairing with VTQ V.

**Deferred K quantization (auto-enabled):** KTQ K-cache types are subject to a repetition-loop pathology when K is quantized per-token during prefill — the same attention step reads back the just-quantized rows, so stochastic-rounding and RHT round-trip noise accumulate on every layer of a long prompt, the softmax collapses onto a few tokens, and the model starts looping (`"Es war einfach. Es war einfach. Es war einfach."`). To avoid this, the K cache is staged as f16 during prefill and bulk-converted to KTQ exactly once at the prefill→decode transition. This runs automatically as soon as `--cache-type-k` is a KTQ type; no flag needed. The legacy `--tq-deferred-k` CLI flag is retained as a no-op for backwards compat. Log line `deferred K quantization enabled (N layers with f16 staging)` confirms the path on startup.

</details>

<details>
<summary><strong>VTQ (V-Cache TurboQuant)</strong></summary>

A fixed D·H·D rotation (sign-diagonal · FWHT · sign-diagonal) is applied once at the graph level, before values enter the cache. The FA V-dequant reduces to `codebook[idx] * scale` and is `__forceinline__`. Codebooks at 1–2 index bits are fit with Lloyd-Max to a Laplace(0, 1) prior, which matches the empirical marginal distribution of rotated V entries in my measurements.

| Type | Index bits | bpw | Block | Notes |
|------|:---:|:---:|:---:|---|
| `vtq1_1` | 1 | 1.5 | 6 B | maximum compression |
| `vtq2_1` | 2 | 2.5 | 10 B | recommended, Laplace codebook |
| `vtq3_1` | 3 | 4.0 | 16 B | near-lossless (see PPL tables) |
| `vtq4_1` | 4 | 4.5 | 18 B | smallest codebook-fit error |

**Deferred V quantization (auto-enabled for VTQ_2 types):** `vtq2_2`/`vtq3_2`/`vtq4_2` use a Trellis encoder whose per-token Viterbi is ~21.7ms/call and would stall the decode loop. To avoid that, V-writes are staged as f16 during prefill and bulk-converted once at the prefill→decode transition. This auto-enables whenever `--cache-type-v` is a VTQ_2 type; the legacy `--tq-deferred-v` CLI flag is retained as a no-op. VTQ_1 types (`vtq1_1`/`vtq2_1`/`vtq3_1`/`vtq4_1`) don't need this — their codebook lookup is cheap enough for per-token writes.

</details>

---

## How It Works

### The Problem

A TurboQuant-style per-block V-dequant requires a 32-element FWHT butterfly inside the FA kernel's inner loop. In this CUDA implementation on CC 7.5, this pushed the kernel over the 255-register/thread limit on `vec_dot_KQV`, producing register spills to local memory, and in one observed case, corruption of the FA accumulator for some head/tile combinations. I did not fully characterize the corruption; the split-path design was chosen to sidestep it rather than fix it in place.

### The Solution: Split K and V

```
KTQ K-path (FA inner loop):        VTQ V-path (FA inner loop):
  // no dequant of K at all         float val = codebook[idx] * scale;
  // Q is FWHT'd once per tile      // done
  dot = <Q_hat, K_indices>          
```

- **KTQ** keeps per-block RHT on K. The FA kernel applies FWHT to Q once per tile and computes the Q·K dot product entirely in the Hadamard domain, so K is never explicitly dequantized.
- **VTQ** moves the randomization out of the FA inner loop: a fixed D·H·D (sign-diagonal · FWHT · sign-diagonal) rotation is applied once at the graph level, before the cache write. Per-entry dequant is then just `codebook[idx] * scale`.

The D·H·D rotation is *not* a fully random orthogonal rotation (it uses deterministic sign diagonals), but the Hadamard mixing decorrelates the D-dimensional value vector enough that its coordinate marginals match a Laplace(0, 1) prior closely in my measurements. That property, not strict i.i.d.-ness, is what the 2-bit Lloyd-Max codebook relies on.

---

## Build

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# For all KTQ x VTQ combinations:
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
```

CUDA CC 6.1+. CPU fallback available.

## When This Fork Is Not the Right Choice

- **When VRAM is not a constraint:** upstream llama.cpp with f16 KV is simpler and equally fast.
- **When sub-50ms/token latency matters:** VTQ V-cache increases per-token dequant overhead at long context. Use f16 V or other forks.
- **For multi-node inference:** this fork makes no changes to llama.cpp's split logic. Large-scale serving systems are designed for that use case.
- **On Ampere+ (CC 8.0+):** untested. The sm_75-specific launch_bounds and FA tuning are calibrated for Turing, not newer tensor-core generations.

## Roadmap Reality

As of 2026-04-23:

**Shipped:**
- KTQ1_1/2_1/3_1/4_1 — K-cache quant types
- VTQ1_1/2_1/3_1/4_1 — V-cache quant types (asymmetric K/V via FA dispatch)
- Deferred-V quantization infrastructure
- Attention Sinks protection
- Laplace-optimized 2-bit codebooks

**Active research (no guarantees):**
- **MMA-KTQ asymmetric dispatch (shipped)** — KTQ K + f16 V now takes the tensor-core MMA path via bulk K→f16 split-dequant. A silent early `supports_op` guard in `fattn.cu` was rejecting `K.type != V.type` for all KTQ+f16V shapes, so every prior "KTQ PP regression" number was actually the FA op falling out of the CUDA graph and splitting to a non-FA CPU-fallback. After the fix:
  - **Qwen3.5-35B-A3B IQ2_XS** (reference config): PP128 **727 t/s** (vs f16 431 — KTQ *faster*), PP512 **875** (vs 861), PP2048 **868** (vs 857), TG128 67 (vs 71). That's a **9.5× jump over the pre-fix 92 t/s** PP512 number.
  - **Ministral-3-14B IQ2_M**: PP128 674 (96% of f16), PP512 687 (93%), TG128 24.4 (96%).
- **MMA-KTQ inline tile-load (compiled, dormant)** — warp-cooperative KTQ dequant inside the MMA tile-load. Wired and building but the split path wins on current shapes; kept for future use.
- **When to pick MMA-KTQ:** only if your V-cache stays f16 (you only compress K). For the more common asymmetric config `--cache-type-k ktq2_1 --cache-type-v vtq2_1` the VEC path is used and MMA-KTQ does not apply.
- mmvq tuning for IQ2_XS on sm_75
- Trellis v2 (VTQ_2 family) — currently broken on D=256
- C1 streaming window — designed, not implemented

**Discarded after measurement:**
- Speculative decoding — does not work on A3B MoE (expert saturation)
- VTQ_MIXED — dominated by VTQ3_1, not CUDA-ported
- Calibrated outlier selection — marginal gain after RHT
- MMA-KTQ split-dequant *as default for all ctx* — regresses beyond ~512 tokens when the K→f16 bulk pass starts dominating. Now gated to short-ctx only (see above).

**Deliberately not on roadmap:**
- FA3 port (requires sm_80+ hardware)
- Paged attention (scope mismatch with fork goal)
- Multi-node inference

## Known Limitations

- **KTQ + VTQ at low bits:** combined 2-bit K + 2-bit V shows super-additive PPL degradation in my tests. Pair `vtq*_1` with `q8_0` or `q4_0` for K.
- **Platform:** CUDA only. CC 6.1+ required. A CPU reference path exists for correctness tests but is not performance-tuned. No Metal or ROCm port.
- **PPL measurement scale:** tables above use 3 wikitext-2 chunks at 512 ctx. Chunk-to-chunk variance matters at this scale; treat sub-1% deltas as noise.
- **Model coverage:** all benchmarks are on Qwen3.5 / Qwen3.6 (MoE and Dense). Behavior on Llama, Mistral, or Phi architectures has not been measured.

## Documentation

| Doc | Content |
|-----|---------|
| [docs/turboquant.md](docs/turboquant.md) | Architecture, CUDA kernels, codebooks |
| [docs/plans/2026-04-16-vtq-design.md](docs/plans/2026-04-16-vtq-design.md) | VTQ design spec, math proofs |
| [docs/plans/2026-04-17-trellis-v2-design.md](docs/plans/2026-04-17-trellis-v2-design.md) | Trellis v2 design — trellis-coded quantization for KTQ/VTQ |
| [docs/plans/2026-04-17-trellis-v2-phase1-report.md](docs/plans/2026-04-17-trellis-v2-phase1-report.md) | Trellis v2 Phase-1 report — experiments, bugs, results, Phase-2 candidates |
| [docs/claude-code.md](docs/claude-code.md) | Using Anthropic's Claude Code CLI against a llama-tq server (`/v1/messages`) |

## Related Projects

| Project | Focus |
|---------|-------|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Symmetric TQ, Metal + CUDA, extensive benchmarks (NIAH, long ctx) |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ (Trellis-Coded Quantization), Viterbi encoding, best 3-bit PPL |
| [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | TurboQuant community thread |

## Inspirations & Methods Used

This fork is a research collection combining several building blocks. Where each method fits:

### Core Foundation

- **llama.cpp** — Upstream fork. Unchanged runtime, server, GGML infrastructure. Non-KV-cache changes come from upstream and are merged back regularly.
- **ggml** — Tensor library enabling type_traits: the VTQ dispatch goes through the standard ggml mechanism, so CPU fallback, quantize, dequantize, and set-rows work automatically once types are registered.

### Primary Research Methods

- **TurboQuant** — Random Hadamard Transform (RHT) for outlier diffusion, Lloyd-Max codebook for 1D-optimal quantization. The MSE-optimal Stage 1 (PolarQuant) is implemented. Stage 2 (QJL residual) was evaluated and found ineffective for attention workloads.
- **Flash Attention 2** — The FA kernel on which the asymmetric K/V dispatch extension is built. The vec-path is modified for VTQ V-cache support.
- **Streaming-LLM / Attention Sinks** — The first 4 tokens in a sequence have disproportionate attention weights. Trick 1 keeps these in f16 instead of quantized.
- **Laplace-optimal codebooks** — Post-RHT data is Laplace-distributed, not Gaussian. Centroids are fit directly to Laplace.

### Considered & Documented Inspirations

- **Trellis-Coded Quantization (TCQ)** — classical signal processing method. A Phase-1 harness was built and the VTQ_2 family was derived from it. Currently broken on D=256, functional on D=128.
- **Paged Attention** — KV-cache management via fixed-size pages. Not directly usable because Python/Triton-based while this stack is pure C++. Documented as a possible porting source.
- **Triton Autoresearch** — autoresearch methodology applied to the 2060 FA kernel. 8 experiments, E11 cached-decode reaches 112 GB/s (14× over naive).
- **CUDA Graphs** — for launch overhead reduction. llama.cpp has this already upstream, default enabled.
- **Speculative Decoding** — already implemented upstream in llama.cpp. Verified whether it fits the A3B MoE config — expert-saturation pathology makes it ineffective on this architecture.

### Measurement Infrastructure

- **nvprof / Nsight Systems** for kernel profiling
- **wikitext-2** dataset for PPL measurement
- **sentence-transformers** for MSE→PPL pipeline validation

### What This Fork Contributes

The independent contribution is limited to:

1. Asymmetric K/V cache dispatch in the FA kernel (K and V with different quant types)
2. VTQ V-cache family — VTQ1_1/2_1/3_1/4_1 as registered ggml types
3. Deferred-V quantization infrastructure — f16 staging at prefill, bulk Viterbi transition
4. Reproducible MSE→PPL pipeline — Python harness, real-data validation
5. Measurement-first methodology — every optimization profile-gated before merge

Everything else stands on the shoulders of the works named above.

## References

This implementation is inspired by but deviates from the TurboQuant paper. Concrete deviations:

- **Rotation (K):** paper uses a Haar-random orthogonal rotation (sampled via QR of a Gaussian matrix). KTQ uses a Randomized Hadamard Transform (FWHT with a per-block sign diagonal). RHT is cheaper to apply (O(D log D), in-place, no stored matrix) and empirically preserves the i.i.d.-like coordinate distribution needed for Lloyd-Max codebooks, but is not strictly Haar.
- **Rotation (V):** VTQ does not use a per-block random rotation. Instead, a single fixed D·H·D rotation is applied at graph-construction time, outside the FA kernel. This is a design specific to this fork choice, not from the paper, and trades some randomization quality for a substantially simpler FA V-dequant path.
- **QJL:** earlier llama-tq versions (v1–v4) used QJL (1-bit Quantized Johnson-Lindenstrauss) for the sign stream. v5 removed QJL; sign information is now carried by the rotation itself.
- **Codebook:** Lloyd-Max fit to a Laplace(0, 1) target (matching the measured marginal distribution after rotation), rather than the paper's Gaussian-fit codebook.

| Paper | Authors | arXiv | Relevance |
|-------|---------|-------|-----------|
| **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** | Zandieh, Daliri, Hadian, Mirrokni | [2504.19874](https://arxiv.org/abs/2504.19874) (April 2025) | Primary inspiration: random rotation + Lloyd-Max codebooks |
| **PolarQuant: Quantizing KV Cache via Polar Coordinate Transformation** | Han, Kacham, Karbasi, Mirrokni, Zandieh | [2502.02617](https://arxiv.org/abs/2502.02617) (Feb 2025) | Different method (polar coordinates), not used here |
| **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization** | Zandieh, Daliri, Han | [2406.03482](https://arxiv.org/abs/2406.03482) (June 2024) | Used in v1-v4, removed in v5 |

## License

MIT (inherited from [llama.cpp](https://github.com/ggml-org/llama.cpp))
