# Qwen3.5-122B-A10B Bench — Expert-Offload Sweep

Date: 2026-04-23
Model: `Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf` (34.11 GiB)
Hardware: 2× RTX 2060 12 GB (CC 7.5, PCIe 3.0), 40 GB host RAM
Build: `turboquant` branch, commit `bc7c2e3d3`

## Architecture

From GGUF metadata:

| Key | Value |
|---|---|
| `block_count` | 48 |
| `expert_count` | 256 |
| `expert_used_count` | 8 |
| `attention.head_count` | 32 |
| `attention.head_count_kv` | **2** (GQA) |
| `embedding_length` | 3072 |
| `context_length` | 262144 |
| `expert_feed_forward_length` | 1024 |
| `expert_shared_feed_forward_length` | 1024 |

The 2 KV heads + 48 layers makes per-token KV ~9 KB at f16, so even 262k ctx needs only ~2.3 GB f16 KV or ~0.4 GB at `ktq2_1/vtq2_1`.

## Baseline — all FFN experts on CPU

Command (`-r 2`):

```bash
./build/bin/llama-bench -m Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
  -ngl 99 -ts 12,12 -fa 1 -mmp 1 \
  -ot "blk\.([0-9]|[1-4][0-9])\.ffn_(up|down|gate)_exps\.=CPU" \
  -p 512 -n 128
```

| K | V | pp512 tok/s | tg128 tok/s |
|---|---|:---:|:---:|
| f16 | f16 | 151.94 – 152.87 | 12.41 – 12.62 |
| ktq2_1 | f16 | 152.11 – 151.61 | 13.09 – 12.23 |
| f16 | vtq2_1 | 147.97 – 147.47 | 13.44 – 12.66 |
| ktq2_1 | vtq2_1 | 148.52 – 145.80 | 12.71 – 12.49 |

KV quant cost: ~−3% PP, ~0% TG. The tiny GQA KV makes compression almost free here.

## Expert-Offload Sweep (distillery-claude contribution)

Hypothesis: moving a few layers' experts back on GPU should speed up TG (less CPU-RAM traffic per token).

Phase 1 — coarse sweep at 0, 8, 16, 24, 32 GPU-expert layers:

| GPU-expert layers | pp512 | tg128 | Status |
|---:|:---:|:---:|---|
| 0 (baseline) | 151.67 – 152.28 | 12.70 – 13.02 | ✅ |
| 8 | 170.73 – 171.18 | 14.02 – 14.27 | ✅ (+13%) |
| 16 | — | — | ❌ OOM (model load) |
| 24 | — | — | ❌ OOM |
| 32 | — | — | ❌ OOM |

Cliff between 8 and 16. Fine-sweep (`-r 2`) to find the exact edge:

| GPU-expert layers | pp512 ±σ | tg128 ±σ | Status |
|---:|:---:|:---:|---|
| 8 | 170.73 – 171.42 | 13.74 – 14.29 | ✅ |
| **10** | **174.59 – 176.06** | **14.64 – 14.74** | ✅ **sweet spot** |
| 12 | — | — | ❌ context-create OOM with `-ub 512` (may work with `-ub 256`) |
| 13–15 | — | — | ❌ model-load OOM |

At 10 GPU-expert layers:
- **+15% PP** vs all-CPU baseline (175 vs 152)
- **+16% TG** vs all-CPU baseline (14.7 vs 12.7)

## Asymmetric Tensor-Split (distillery-claude contribution)

With 10-GPU-expert config, GPU0 is near-full while GPU1 has headroom. Asymmetric `-ts` to re-balance:

| `-ts` | pp512 | tg128 |
|---|:---:|:---:|
| 12,12 | 175.37 – 176.66 | 14.64 – 14.75 |
| 11,12 | 175.12 – 176.16 | 14.30 – 14.52 |
| 10,12 | 175.04 – 175.24 | 14.44 – 14.72 |
| **9,12** | 174.35 – 174.50 | **14.74 – 15.12** |
| 8,12 | 174.01 – 174.30 | 14.64 – 14.88 |
| 12,10 (reverse) | 174.80 | 14.94 |

Delta from `12,12` to `9,12` is ~0.4 tok/s, within the σ≈0.8 noise. Not yet confirmed as real (precision-validation pending, `-r 5 -n 256`).

## Production Config

Tested on `llama-server` at live load (prompts 10–30k tokens, streaming):

| Config | VRAM GPU0 / GPU1 | Live TG |
|---|---|:---:|
| 200k ctx, `ts 12,12`, 10 GPU-expert layers | 10.9 / 4.7 GB | 13.54 tok/s |
| 262k ctx, same | 11.3 / 5.2 GB | 13.50 tok/s |
| 262k ctx + `--reasoning off` | 11.3 / 5.2 GB | 12.54 tok/s* |

\* Lower number because reasoning-off skips the 64-token thinking-warmup, producing direct answers. For single-shot prompts this is the real user-facing latency win.

Full 262k ctx costs only +1 GB over 200k ctx — the `ktq2_1/vtq2_1` + GQA(2) combination keeps KV growth minimal.

### Recommended on-llm-122b service config

```bash
./build/bin/llama-server \
  -m Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
  --host 0.0.0.0 --port 8794 \
  -c 262144 -ngl 99 -ts 12,12 -fa on \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --parallel 1 \
  -ot "blk\.(1[0-9]|[2-4][0-9])\.ffn_(up|down|gate)_exps\.=CPU" \
  --jinja --reasoning off
```

Pattern matches layers 10–47 (38 layers) CPU-offloaded, layers 0–9 (10 layers) fully on GPU. Live: ~13.5 tok/s TG, ~28 tok/s PP (single-user streaming).

## Gotchas

1. **`llama-bench` does not accept `-ctk ktq2_1`** — only `llama-server` wires up the KTQ/VTQ types via `--cache-type-k` / `--cache-type-v`. Bench sweeps have to use f16 KV; the relative expert-offload ordering is unaffected because bench defaults to 4k ctx (low KV impact).
2. **`--cache-type-k tq2_1` is rejected** on `llama-server` — the types are K/V-specific: use `ktq2_1` for K and `vtq2_1` for V.
3. **Thinking-mode defaults to on** for Qwen3.5-122B. `--reasoning off` is critical for prod (the 64-token reasoning warmup balloons short-answer latency — "count to 10" emits 64 thinking tokens before the 15 answer tokens).
4. **12 GPU-expert layers would fit the model** but the compute buffer at `-ub 512` pushes total past 12 GB. Not tested with `-ub 256` — might recover the 12-layer point.
5. **`--n-cpu-moe N`** is a cleaner alternative to `-ot` regex (it's native in llama.cpp master). Not yet A/B-compared against the regex path in this bench.

## Open Work

- Expert-activation profiling — needs a source patch to log which experts fire per token during a typical workload. With that data, "hot" experts could be pinned to GPU layer-by-layer, likely beating the 10-layer contiguous config.
- Shared-experts-only on GPU — Qwen3.5 has `expert_shared_feed_forward_length=1024`, meaning every token hits the shared expert. Isolating just the shared expert on GPU (not the 8 routed experts) hasn't been tested.
- Precision validation of `-ts 9,12` vs `12,12` — running at `-r 5 -n 256` to confirm or reject the 0.4 tok/s delta.

## Credit

Coarse/fine/asymmetric sweep methodology and prod-smoke numbers by distillery-claude
(`LEGION/2026-04-23_2152_distillery_122b-sweep-results-and-method.md`).
