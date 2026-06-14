# Full context on 12 GB — running modern models on Turing-class GPUs

This fork is tuned for one thing: **running specific modern models at full context
on 12 GB GPUs, with a special focus on Turing (sm_75)** — both the consumer cards
(RTX 2060 / 2070 / 2080) and the datacenter **T4**. The KTQ/VTQ KV-cache quantization,
the Gemma-4 SWA handling, and the kernel tuning all exist to make that work where
stock llama.cpp runs out of VRAM or context.

This document is the *applied* layer: practical, verified `llama-server` recipes plus
the **reasoning** behind every KV-cache choice — which combination to pick for which
model and workload, and why. For the full type reference (every `ktq*`/`vtq*` variant,
bit widths, kernels, PPL) see [`turboquant.md`](turboquant.md).

All commands use placeholders: replace `/path/to/models` and pick your own `--port`.
`-ngl 99` offloads all layers; drop it only if a model genuinely doesn't fit. These
recipes were verified on RTX 2060 12 GB (Turing sm_75); they apply directly to the T4
and scale up cleanly to 16 GB Turing.

---

## The five principles

Everything below follows from these. If a recipe surprises you, one of these explains it.

### 1. Decode is bandwidth-bound — KV size *is* context length

On a single GPU the model weights are fixed, but the KV cache grows linearly with
context. At long ctx the KV cache, not the weights, is what pushes you over 12 GB.
**Shrinking the KV cache is how you buy context.** That is what KTQ (K) and VTQ (V)
quantization do: `ktq2_1 + vtq2_2` is ~2.78 bpw vs 16 bpw for f16 — an 83% smaller KV.

### 2. KV-quant aggressiveness depends on the *model* quant, not just the KV type

A coarse model quant (Q4) produces "rougher" K/V activations than a fine one (Q8).
The same KV-quant that is clean on a Q8 model can collapse into garbage on the Q4
version of the same model. **Rule of thumb:** Q8/Q6 models tolerate `ktq2/vtq2`;
Q4/IQ models often need a higher-precision V (`vtq2_3`/`vtq3`) or f16 K. Always
verify with a *long* generation, not a one-liner — degradation shows up late.

### 3. Gemma-style SWA models have two KV sub-caches — quantize them separately

Gemma-3/4 use interleaved sliding-window attention: a few **global** layers
(head_dim 512, KV scales with ctx — the expensive part) and many **SWA** layers
(head_dim 256, window-capped — small and *constant*). The SWA layers run a
windowed softmax that is far more sensitive to quantization error. Quantizing them
aggressively is what produces the classic `enene…`/`t-t-t…` collapse (issue #179).

Fix: keep the global KV aggressive (it's the big, ctx-scaling part) and **protect
the SWA sub-cache** with higher precision via the dedicated flags:

```
--cache-type-k ktq2_1 --cache-type-v vtq2_1   # base = global layers (cheap, ctx-scaling)
--cache-type-k-swa f16 --cache-type-v-swa f16 # SWA  = fragile layers (protected, capped-small)
```

The SWA cache is window-capped, so f16 there costs only a few hundred MB — a bargain
for the coherence it restores. **The `*-swa` flags are essential on Gemma-4; without
them, aggressive KV is unusable.**

### 4. Exact output (code) needs outlier-robust V; prose does not

`vtq2_1` uses a fixed L2-norm block scale with no outlier protection. Code
activations are *spiky* — one outlier inflates the scale and the bulk values
collapse to the nearest centroid. In prose you never notice; in code you get
`rgba(255-255-255)` instead of `rgba(255,255,255)` and truncated identifiers.

Fix: use **`vtq2_3`** (Trellis + 4 fp16 outlier slots — literally built to "absorb
the long-tail V-distribution outliers"). It is ~0.5 bpw larger but makes code-grade
output exact. **For coding/agentic workloads pick `vtq2_3`; for chat/prose, `vtq2_1`
is fine and smaller.**

### 5. Single-GPU beats dual-GPU for decode when the model fits

With layer-split across two GPUs and no P2P, each token runs the layer chain
serially — one GPU computes while the other idles (a pipeline bubble). At
single-stream decode this is *unavoidable* (no microbatch to overlap). If a model
fits on one card, prefer that: no bubble, full bandwidth, and the second GPU is
free for a draft model / another instance. Only go dual-GPU when VRAM forces it.

---

## Flag glossary (the ones that matter for fitting + coherence)

| Flag | What it does | When you tune it |
|------|--------------|------------------|
| `-c N` | context length (KV size scales with this) | lower it to fit; raise it for headroom |
| `-ngl 99` | offload all layers to GPU | always, unless OOM |
| `-fa on` | FlashAttention (**required** for KTQ/VTQ) | always |
| `--cache-type-k / -v` | KV quant type for the *base* (global) layers | the main size/quality lever |
| `--cache-type-k-swa / -v-swa` | KV quant type for **SWA** layers (Gemma only) | f16 here = coherence on Gemma-4 |
| `-ub N` | micro-batch size; shrinks the compute buffer | drop to 256/128 to claw back the last few hundred MB |
| `-ts a,b` | tensor split across GPUs | balance the *expensive* (global) layers, not just count |
| `--spec-type ngram-cache` | n-gram speculative decode (lossless) | free speedup on repetitive/code output |
| `--no-context-shift` | disable rolling-window context shift | keep full-ctx semantics |
| `env -u LLAMA_ARG_SWA_FULL` | **unset** the full-SWA-cache env var | Gemma-4: its mere presence balloons the SWA cache → OOM |

> **Gemma-4 footgun:** never *set* `LLAMA_ARG_SWA_FULL` (even `=0` activates it — it's
> a no-arg flag). Launch with `env -u LLAMA_ARG_SWA_FULL` so the SWA cache stays
> windowed (capped ~765 MiB instead of full-ctx-sized).

---

## Recipe table (verified, single 12 GB GPU unless noted)

| Model | Quant | ctx | K / V | SWA K / V | GPUs | Workload | ~t/s |
|-------|-------|----:|-------|-----------|:---:|----------|:---:|
| Gemma-4-12B | Q4_K_M | 180k | `ktq2_1` / `vtq2_3` | `f16` / `vtq3` | 1 | **code** (exact) | ~34 |
| Gemma-4-12B | Q4_K_M | 256k | `ktq2_1` / `vtq2_1` | `f16` / `f16` | 1 | chat/prose | ~34 |
| Gemma-4-12B | Q8_K_XL | 256k | `ktq2_1` / `vtq2_1` | `f16` / `vtq3` | 2 | max quality | ~22 |
| Gemma-4-26B-A4B | Q4_K_M | 256k | `ktq2_1` / `vtq2_1` | `f16` / `vtq3` | 2 | fast MoE | ~59 |
| Qwen3.6-35B-A3B | IQ4_XS | 48k | `ktq2` / `vtq3` | — | 2 | coding (Qwen) | ~69 |

(*Speeds are single-stream decode on RTX 2060-class cards; bandwidth, not the recipe,
is the ceiling. Larger ctx with the same recipe simply needs more KV headroom.*)

---

## Recipes, explained

### Gemma-4-12B Q4 — code-exact, single GPU, ~180k ctx

```bash
env -u LLAMA_ARG_SWA_FULL CUDA_VISIBLE_DEVICES=0 llama-server \
  -m /path/to/models/gemma-4-12b-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 --jinja --flash-attn on \
  -c 180224 -ngl 99 --parallel 1 -ub 256 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_3 \
  --cache-type-k-swa f16 --cache-type-v-swa vtq3 \
  --no-context-shift --spec-type ngram-cache --draft-max 8 --draft-min 4
```

Why this combo:
- **`vtq2_3` (V)** — principle #4: the model is Q4, so we need outlier-robust V or
  long code corrupts. This is the whole reason ctx is 180k not 256k (vtq2_3 is bigger).
- **`f16`/`vtq3` SWA** — principle #3: protect the fragile sliding-window layers.
  K on f16 (cheap, capped), V on vtq3 (still robust, a touch smaller than f16).
- **`ktq2_1` (K, base)** — the global K is the ctx-scaling cost; keep it small.
- **`-ub 256`** — shaves the compute buffer to fit the last few hundred MB.
- **ngram-cache spec** — lossless; code has lots of repeated structure, so it helps.

For **chat instead of code**, swap `vtq2_3 → vtq2_1` and `-v-swa vtq3 → f16`, and you
get the full **256k** ctx (prose doesn't expose the outlier problem).

### Gemma-4-12B Q8 — max quality, full 256k, dual GPU

```bash
env -u LLAMA_ARG_SWA_FULL CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -m /path/to/models/gemma-4-12b-it-Q8_K_XL.gguf \
  --host 0.0.0.0 --port 8080 --jinja --flash-attn on \
  -c 262144 -ngl 99 -ts 1,1 --parallel 1 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --cache-type-k-swa f16 --cache-type-v-swa vtq3 \
  --no-context-shift --spec-type ngram-cache
```

Why: Q8 (principle #2) absorbs KV-quant error, so even `vtq2_1` stays clean — no
need for vtq2_3 here. The 13 GB Q8 model doesn't fit one 12 GB card, so it's dual-GPU
(principle #5 conceded for VRAM). f16-K @256k nearly OOMs; `ktq2_1` K shrinks it.

### Gemma-4-26B-A4B (MoE) — fast, full 256k, dual GPU

```bash
env -u LLAMA_ARG_SWA_FULL CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -m /path/to/models/gemma-4-26b-a4b-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 --jinja --flash-attn on \
  -c 262144 -ngl 99 -ts 16,14 --parallel 1 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --cache-type-k-swa f16 --cache-type-v-swa vtq3 \
  --no-context-shift --spec-type ngram-map-k4v --draft-max 16 --draft-min 2
```

Why `-ts 16,14` (not 12,12): Gemma-4-26B has 5 global layers at indices 5/11/17/23/29.
At an even split, 3 of them land on GPU1 → its KV is 50% bigger → OOM. `-ts 16,14`
moves more layers (and global-layer KV) onto GPU0, which carries the compute buffer,
balancing the load. **The global-layer distribution, not the layer count, is what
matters for KV balance on SWA MoEs.**

`ngram-map-k4v` (not `ngram-cache`): on this MoE it hits ~95% draft acceptance on
repetitive output → ~140 t/s on code/repeat (vs ~59 baseline), lossless. It stores
4 continuations per key, so it finds far more matches. (On dense models like the 35B
plain `ngram-cache` is the safer pick — k4v's bigger draft batch can OOM tight setups.)

---

## Debugging checklist

- **Gibberish (`enene…`, `t-t-t…`) on Gemma-4** → SWA layers are over-quantized. Add
  `--cache-type-k-swa f16 --cache-type-v-swa f16`. (And confirm you used
  `env -u LLAMA_ARG_SWA_FULL`.)
- **Subtle corruption in *code* only** (`rgba(255-255-255)`, truncated names) →
  V-cache lacks outlier protection. Switch `--cache-type-v` to `vtq2_3` (or `vtq3`).
- **OOM by a few hundred MB** → drop `-ub` to 256 or 128; or lower `-c` slightly.
  Note: the big `allocating … MiB … cudaMalloc failed` numbers are often *retried*
  smaller — the real compute buffer is in the `sched_reserve:` log line.
- **OOM only at high ctx on dual-GPU SWA models** → rebalance `-ts` so the global
  layers split evenly (see the 26B recipe).
- **Slower than expected on dual-GPU** → it's the pipeline bubble (principle #5). If
  the model fits one card, go single-GPU.

---

See [`turboquant.md`](turboquant.md) for the underlying KTQ/VTQ type reference,
bit widths, kernels, and PPL benchmarks.
