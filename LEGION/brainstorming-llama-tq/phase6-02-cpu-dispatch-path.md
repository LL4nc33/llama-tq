# Phase 6 — CPU Dispatch Path Findings

## Executive Summary

The MoE expert selection in llama-tq is **fully graph-level**: top-k is hardcoded as `n_expert_used` (an `int64_t` from `hparams.n_expert_used`) baked into `ggml_argsort_top_k` at graph build time. The `(expert_idx, weight)` pair vector materializes as **two separate ggml tensors** (`selected_experts` I32 + `weights` F32), both consumed by `ggml_mul_mat_id` / `ggml_add_id` ops. There is **no host-side runtime decision point** — k is fixed per layer at graph construction.

Phase 6 (dynamic per-token k from cumulative softmax mass) requires a **graph-level redesign**: introduce a new op (e.g. `ggml_argsort_top_k_dynamic`) that takes a probability tensor + threshold, emits an `(indices, weights, k_per_token)` triple, and wire downstream `mul_mat_id` to honor a per-token k mask (zero-masked tail slots OR scatter-style dispatch).

## Key Files & Lines

### 1. Router → top-k construction (graph builder)

**File:** `/mnt/d/repos/llama-tq/src/llama-graph.cpp`

- **L1226-1266**: `build_moe_ffn` overload #1 (no-bias) — delegates to overload #2.
- **L1268-1622**: `build_moe_ffn` overload #2 (canonical) — full implementation.
  - **L1296-1303**: router logits = `build_lora_mm(gate_inp, cur)` → `[n_expert, n_tokens]`. Tensor name `ffn_moe_logits`.
  - **L1311-1326**: gating op switch (`SOFTMAX` / `SIGMOID` / `SOFTMAX_WEIGHT`) producing `probs`.
  - **L1331-1346**: bias / arch-specific selection adjustments → `selection_probs`.
  - **L1350-1371**: optional expert-group masking (DeepSeek-V3 style, uses `ggml_argsort_top_k` with k=2 and k=`n_group_used`).
  - **L1374**: `ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);` → `[n_expert_used, n_tokens]` I32. **THIS IS THE PHASE 6 PRIMARY HOOK.**
  - **L1387**: `weights = ggml_get_rows(ctx0, probs, selected_experts)` → `[1, n_expert_used, n_tokens]`.
  - **L1391-1412**: weight post-processing (softmax-on-topk, norm, scale).
  - **L1419**: `ggml_build_forward_expand(gf, weights)` — early commit so kernel can fuse with topk.
  - **L1435, L1459, L1477, L1567**: four `build_lora_mm_id(*, cur, selected_experts)` calls — these become `ggml_mul_mat_id` ops in the graph. Output shape `[n_ff, n_expert_used, n_tokens]`.
  - **L1439, L1463, L1484, L1571**: `ggml_add_id(... selected_experts)` for biases.
  - **L1585**: `experts = ggml_mul(ctx0, experts, weights)` — final per-expert weighting.
  - **L1591-1612**: aggregation loop using `hparams.n_expert_used` (NOT the local `n_expert_used` parameter) as upper bound — note this iterates `n_expert_used` views and adds them. **A dynamic-k variant must replace this with masked-add or scatter-reduce.**

### 2. Header / signatures

**File:** `/mnt/d/repos/llama-tq/src/llama-graph.h`

- **L813-831**: `build_moe_ffn` overload #1 declaration.
- **L833-856**: `build_moe_ffn` overload #2 declaration.
- Both take `int64_t n_expert_used` as a fixed scalar — no per-token vector path.

### 3. CPU dispatch consumption

**File:** `/mnt/d/repos/llama-tq/ggml/src/ggml-cpu/ops.h`

- **L32**: `ggml_compute_forward_add_id` — consumes `selected_experts` (src[2]) for bias add.
- The `mul_mat_id` CPU forward (in `ops.cpp` / `llamafile/sgemm.cpp`) iterates `dst->ne[2] = n_tokens` × `dst->ne[1] = n_expert_used` slots; for each slot reads `selected_experts[token, slot]` to pick the expert weight matrix row block. **Currently this loop bound is the static tensor dim, not a per-token k.**

The `ne[1]` dimension of `selected_experts` IS the static k. To support dynamic k, options:
- **(A) Padding approach**: keep `ne[1] = max_k` (8); fill unused tail slots with sentinel index (e.g. -1 or duplicate of slot 0 with zero weight). Kernel skips sentinel OR weight=0 zeroes contribution. Minimal kernel change (early-skip on -1).
- **(B) New op**: `ggml_mul_mat_id_dyn` with extra `k_per_token` I32 tensor `[n_tokens]`. Requires CPU + CUDA op variants.

### 4. CUDA path (for reference / parity)

Files (per directory listing): `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/argsort.cu`, `mmid.cu` (or `mmq.cu` / `mmvq.cu` for mmid). The `selected_experts` tensor flows in identically; CUDA `mul_mat_id` kernels also iterate `ne[1]` slots per token. Same options A/B apply.

### 5. Tensor dataflow summary

Router logits → `probs` → **graph tensor** `selected_experts` (I32 `[k=n_expert_used, n_tokens]`) + **graph tensor** `weights` (F32 `[1, k, n_tokens]`). Both flow as **first-class ggml tensors through the compute graph**, NOT as host-side arrays at dispatch. There is no `ggml_backend_tensor_get` / CPU readback before mmid — everything stays on-device.

**Implication for Phase 6:** A pure runtime decision is impossible without graph rebuild. Two viable designs:

1. **Static-max + mask (recommended for v1)**: keep k=8 at graph level, compute `k_per_token` and weight-mask in a new fused op `ggml_topk_dynamic_mask` placed between L1374 and L1387. Tail slots get weight=0; existing mmid kernels run unchanged but waste compute on zero-weight slots. Optional kernel-side optimization: peek weight, skip mmid contribution.
2. **True dynamic k (v2)**: new ops `ggml_argsort_top_k_cumsum(probs, threshold)` + `ggml_mul_mat_id_dyn` accepting `k_per_token`. Requires CPU + CUDA + (Metal/Vulkan) kernel variants. Higher payoff but ~6 kernels to write.

## Recommended Phase 6 landing points

| Change | File:Line |
|---|---|
| Replace `ggml_argsort_top_k` call with dynamic variant | `src/llama-graph.cpp:1374` |
| Add `cparams.moe_dynamic_k_threshold` plumbing | `src/llama-cparams.h` + `common/arg.cpp` |
| New op declaration | `ggml/include/ggml.h` (near `ggml_argsort_top_k`) |
| CPU forward | `ggml/src/ggml-cpu/ops.cpp` (new `ggml_compute_forward_argsort_top_k_dyn`) |
| CUDA forward | `ggml/src/ggml-cuda/argsort.cu` |
| mmid kernel mask handling | `ggml/src/ggml-cpu/ops.cpp` (mul_mat_id loop) + `ggml/src/ggml-cuda/mmid.cu` |
| Aggregation loop fix | `src/llama-graph.cpp:1591-1612` (use weight-mask, not static `hparams.n_expert_used`) |

## Caveats

- `selected_experts` consumption inside CPU/CUDA mmid kernels was inferred from ggml conventions and the graph-side usage, not directly read. Verify by inspecting `ggml_compute_forward_mul_mat_id` body in `ggml-cpu/ops.cpp` before committing to design.
- Aggregation at L1591-1612 uses `hparams.n_expert_used` (not the parameter `n_expert_used`) — this is a deliberate warmup-stability hack per PR #14753. Dynamic-k must NOT regress this behavior; padding approach (A) preserves it natively.
- Llama4 weight-before-ffn path (L1423-1428) multiplies `cur` by `weights` before mmid — masking at weight tensor still works.

Relevant absolute paths:
- `/mnt/d/repos/llama-tq/src/llama-graph.cpp` (lines 1226-1622)
- `/mnt/d/repos/llama-tq/src/llama-graph.h` (lines 813-856)
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cpu/ops.h`
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cpu/ops.cpp` (verify mul_mat_id forward)
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/argsort.cu`
- `/mnt/d/repos/llama-tq/ggml/src/ggml-cuda/mmid.cu`
