# 400K ctx + parallel=2 OOM — Root Cause Analysis

**Date:** 2026-04-20
**Scope:** Qwen3.5-35B-A3B IQ2_XS on gpu00 (2× RTX 2060 12 GB), TQ KV cache (ktq2_1 + vtq3_2 + deferred + sink), flash-attn.
**Symptom:** `sched_reserve` succeeds (barely), first forward triggers `ggml_cuda_compute_forward: SCALE failed` → `CUDA error: out of memory`.
**Status:** READ-ONLY diagnosis. No code changes.

---

## 1. Why is the CUDA0 compute buffer larger than CUDA1?

Asymmetric buffer sizes (1852 MiB vs 1292 MiB, delta ≈ 560 MiB) come from **three overlapping causes** in llama.cpp's layer-split scheduler:

### a) Output head ("lm_head") lives on a single device
`llama_model::dev_output()` (`src/llama-model.cpp:8519`) returns one device — the one holding the last block of layers. Under LAYER split mode, device 0 typically owns the **input embedding + first N layers**, device 1 owns the **last layers + output head + logits**. But the compute buffer for the *input path* (token embed, embedding matmul intermediates, positional data) plus a large slice of the graph reside on CUDA0.

More importantly: in `sched_reserve()` (`src/llama-context.cpp:562-603`) the graph is reserved with `n_tokens = min(n_ctx, n_ubatch)` for PP. All intermediate tensors of layers assigned to device 0 are counted toward that device. With 64 layers on Qwen3.5-35B, even a 50/50 layer split doesn't translate to 50/50 compute buffer — the **first-layer attention + input path activations** sit on device 0 only.

### b) Pipeline Parallelism replicates input buffers (GGML_SCHED_MAX_COPIES = 4)
Relevant code: `ggml/src/ggml-backend.cpp:1744`:
```cpp
sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
```
`GGML_SCHED_MAX_COPIES` defaults to 4 (`ggml/CMakeLists.txt:194`, `ggml/src/ggml-backend.cpp:761`). Pipeline parallelism is auto-enabled when:
- `n_devices > 1`
- `n_gpu_layers > n_layer` (full offload)
- split mode = LAYER
- `offload_kqv` = true
- no tensor overrides
(`src/llama-context.cpp:320-325`)

With 2 GPUs + full offload that's always true here. Every input tensor per ubatch is replicated **4×** in the pipeline input buffer. The input copies land on the **first device (CUDA0)** because that's where the input path begins. That's a big contributor to the 560 MiB delta.

### c) Host buffer duplication (`CUDA_Host compute buffer = 1573 MiB`)
`src/llama-context.cpp:302-309`: when a CPU backend is registered alongside GPUs, the **first device's** host buffer type is used for pinned-memory intermediate state. This 1.57 GB pinned buffer is NOT on GPU, but it competes with pageable system RAM; more importantly, it's also scaled by `n_copies=4`.

### Math sanity check (back-of-envelope)
- KQ mask per ubatch: `n_kv × (n_ubatch / n_stream) × n_stream × fp32`
  For n_ctx=400K, parallel=2 → n_stream=2, n_kv per stream ≈ 200K, n_ubatch=512:
  `200,000 × 256 × 2 × 4 B ≈ 400 MiB` per replicated copy. With FA enabled this mask is largely consumed inside the fused kernel (no materialized softmax), but **scratch tensors for the "fake" reservation path and non-FA fallbacks still allocate a chunk.**
- Token embedding ubatch tensor: `n_embd × n_ubatch × fp32` = `5120 × 512 × 4 B = 10 MiB` × 4 copies = 40 MiB (small).
- The dominant 500 MiB delta is **layer-0 attention scratch + RoPE temporaries + pipeline input-copy ring**.

**Refs:**
- `src/llama-context.cpp:320-347` (pipeline_parallel auto-enable)
- `src/llama-context.cpp:562-615` (reserve + per-backend buffer print)
- `ggml/src/ggml-backend.cpp:1744` (n_copies = 4 when parallel)
- `src/llama-graph.cpp:31` (KQ mask shape: `n_kv × n_tokens/n_stream × 1 × n_stream`, F32)

---

## 2. What determines compute buffer size?

Three multiplicative factors:
1. **Per-ubatch activations:** `n_ubatch × n_embd × dtype` for every intermediate tensor the worst-case graph touches.
2. **Attention scratch:** dominated by KQ mask (`n_kv × n_ubatch/n_stream × F32`) and, if FA disabled, the materialized `Q·K^T` (`n_head × n_ubatch × n_kv × F32`). With FA enabled the mask stays but the score matrix collapses into a streaming softmax.
3. **Pipeline copies:** whole input subgraph tensors × `n_copies` (up to 4).

It is **not** simply `ubatch × n_embd × dtype`. The KQ-mask piece scales with **n_ctx** (via n_kv), so context IS in the compute buffer. Halving `n_ubatch` halves the per-ubatch activations but only partially reduces the KQ-mask-dependent tensors (mask itself is n_kv × n_tokens/n_stream, so yes linear in ubatch too). So **ubatch 512 → 256 should cut ~40-50% of compute buffer**, not 50% flat. See point 4.

**Refs:** `src/llama-context.cpp:2091-2146` (graph_reserve), `src/llama-graph.cpp:22-33` (mask build).

---

## 3. Asymmetric tensor split (`-ts`)

Yes: `common/arg.cpp:2429-2447` parses `-ts N0,N1,...` into `params.tensor_split[128]`. It's passed to the model at load (`common/common.cpp:1150`, `common/common.h:438`). **Caveat:** `tensor_split` affects **model weight distribution across GPUs at load time**, NOT the compute buffer split. The scheduler still assigns graph nodes to devices by layer ownership. Moving layers to CUDA1 via `-ts 10,14` (or similar) will shift both model weights AND attached compute buffer to CUDA1 — because layer 0 (input-side) stays on CUDA0, but fewer total layers on CUDA0 means less accumulated activation memory there.

Estimate: shifting ~4 layers from CUDA0 to CUDA1 should move roughly 4/32 × (layer compute) ≈ 200-300 MiB from CUDA0 to CUDA1, possibly enough to clear the 39 MiB deficit with headroom.

**Refs:** `common/arg.cpp:2429`, `common/common.cpp:1421`.

---

## 4. Does `--ubatch 256` halve the compute buffer?

**Approximately yes, but not exactly 50%.**

- Per-ubatch activation tensors: **linear in n_ubatch** → halved.
- KQ mask: shape `n_kv × n_tokens/n_stream × n_stream`, F32. n_tokens=n_ubatch → also **linear in n_ubatch** → halved.
- Fixed overheads (weights intermediates, routing tables, output buffer, pipeline ring metadata): **constant**.
- Pipeline input copies (4×): scale with ubatch → halved.

Realistic gain on CUDA0: `1852 MiB → ~1000-1150 MiB` (−40 to −45%).
Realistic gain on CUDA1: `1292 MiB → ~700-800 MiB`.

**Throughput cost:** PP bandwidth roughly linear in ubatch for compute-bound ops; 256 vs 512 ≈ 10-25% slower prompt processing, decode unaffected.

**Refs:** `src/llama-graph.cpp:31`, `src/llama-context.cpp:564` (uses `n_tokens` = ubatch for reserve).

---

## 5. Alternatives to fit 400K ctx + parallel=2 without OOM

Ranked by effort × impact:

| # | Change | CUDA0 saving | Cost | Risk |
|---|--------|--------------|------|------|
| 1 | `--ubatch 256` (from 512) | −700 to −850 MiB | PP −15-25% | none |
| 2 | `-ts 10,14` (shift ~4 layers to CUDA1) | −200 to −300 MiB | minimal; CUDA1 gets +300 MiB weights but has 149 MiB free → need combine with #1 | low |
| 3 | Set `GGML_SCHED_MAX_COPIES=2` at build time (recompile ggml) | −400 to −600 MiB (input copies) | Reduces pipeline overlap → PP −5-10% | requires rebuild; gated by env at cmake |
| 4 | Disable pipeline parallelism: `cparams.pipeline_parallel = false` (no CLI flag yet, requires `--no-op-offload` or hack) | large (−500 MiB CUDA0, −1.5 GB host) | PP −20-40% on 2-GPU | medium — no clean CLI switch; see `src/llama-context.cpp:347,567-571` |
| 5 | `--ubatch 384` (gentler than 256) | −400 to −500 MiB | PP −8-12% | none |
| 6 | `--ctx 350K` instead of 400K | −100 to −150 MiB | user-visible context reduction | none |
| 7 | Drop sink (`--sink` off) — attention extra buffers | −50-100 MiB | quality regression on long ctx | low |
| 8 | KV offload off on layer 0 (`-nkvo` / explicit override) | −150-300 MiB | decode slower | medium |
| 9 | vtq3_2 → vtq4_1 (more compact) — wait, vtq3_2 is already compact; ktq2_1 → smaller is ktq1_1 if present | variable | quality regression | high |
| 10 | Larger host RAM and move `CUDA_Host compute buffer` to true CPU: not a knob | — | — | — |

### Recommended stack (conservative):
**`--ubatch 256` + `-ts 10,14`** — combined saving on CUDA0 ≈ 900-1150 MiB. That pushes the 39 MiB deficit to ~900 MiB free headroom. PP throughput cost ~15-25%. **Zero rebuilds, zero quality compromise.**

### Recommended stack (aggressive, if above not enough):
Add **`GGML_SCHED_MAX_COPIES=2`** via cmake at next rebuild. Saves further ~400 MiB on both GPUs AND halves the 1.57 GB host buffer to ~800 MiB.

---

## File references (consolidated)
- `src/llama-context.cpp:302-309` — CPU host buffer type = first device's host buft
- `src/llama-context.cpp:320-347` — pipeline_parallel auto-enable conditions
- `src/llama-context.cpp:353,393,1282,1616` — `sched_reserve()` call sites
- `src/llama-context.cpp:562-615` — PP/TG graph reservation, per-backend buffer logging
- `src/llama-context.cpp:2091-2146` — `graph_reserve()` implementation
- `src/llama-graph.cpp:22-33` — `build_attn_inp_kq_mask()` — shape `n_kv × n_tokens/n_stream × 1 × n_stream`, F32
- `src/llama-model.cpp:8515-8521` — `dev_layer()` / `dev_output()` device assignment
- `ggml/src/ggml-backend.cpp:1744` — `sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`
- `ggml/src/ggml-backend.cpp:760-761` — `#define GGML_SCHED_MAX_COPIES 4`
- `ggml/CMakeLists.txt:194` — cache var `GGML_SCHED_MAX_COPIES`
- `common/arg.cpp:2429-2447` — `-ts/--tensor-split` parsing
- `common/arg.cpp:1276-1279` — `-ub/--ubatch-size`
- `common/common.cpp:1150,1421` — tensor_split passthrough to llama_model_params

---

## Why did `SCALE` fail and not earlier?
`sched_reserve()` allocates the **reservation buffer** based on worst-case graph with `no_alloc=true` paths — it reports −39 MiB (fits in CUDA's internal rounding slack and reserved block headroom). The first **real** forward pass allocates additional temporary workspace (FA scratch, softmax temp, norm temp) that was not counted in the reservation pre-estimate. The `SCALE` op (used in RMSNorm epsilon scale or attention `1/sqrt(d_k)` scaling) is typically an early small allocation in the layer graph — it fails because the preceding larger allocations consumed the last remaining headroom. In other words: the reservation estimate was within 40 MiB of truth, and the runtime needed just a few more MiB than estimated. This is a classic off-by-slack failure, not a logic bug.

---

## TL;DR
- Asymmetry = layer-0 sits on CUDA0 + pipeline-parallel 4× input copies land on the first device + host buffer auto-binds to first device.
- Compute buffer is NOT purely `ubatch × n_embd`; KQ-mask term scales with n_ctx as well.
- Best no-rebuild fix: **`--ubatch 256 -ts 10,14`**.
- Best with rebuild: add **`-DGGML_SCHED_MAX_COPIES=2`** to cmake.
