# Phase 6f — Per-Layer Hot-Expert L3 Cache Pinning + Software Prefetcher

**Status:** SPEC (ready for builder)
**Date:** 2026-04-27
**Owner:** llama-tq fork
**Predecessors:** Phase 6 (adaptive top-k, ABORTED), Phase 6a (router profiler, SHIPPED)
**Reference:** `LEGION/brainstorming-llama-tq/phase6-06-real-leverage.md` (Lever A)

---

## 1. Motivation

Phase 6 adaptive-k died on Qwen3 because `SOFTMAX_WEIGHT` routing produces a near-flat distribution — cumulative softmax mass is uninformative. **However**, the same calibration data (`/tmp/router-80b.bin`) revealed that *top-k selection* itself is highly skewed:

- **80B-A3B:** 118/512 experts ever appear in top-8 (23%); top-10 carry 45.3% of dispatch.
- **Per-layer top-20 share:** 30–65% (mid-layers strongest).
- **Adjacent-layer overlap:** mean 0.000 — every layer has its own private hot set, so the prefetcher must be **per-layer**, not global.

Bandwidth on the deploy target (Ryzen 7 3700X, DDR4-3200, ~40 GB/s real) is the dominant ceiling for CPU-offloaded MoE layers in 80B/122B deploys. Hot-pinning ~10 MB of expert weights per active layer fits inside one CCX's 16 MB L3 and lifts the effective bandwidth ceiling substantially.

## 2. Bandwidth Math

| Quantity | Value |
|---|---|
| Expert size (IQ2_XXS, 80B) | ~0.5 MB |
| Top-20 hot set per layer | 20 × 0.5 = **10 MB** |
| L3 per CCX (3700X) | 16 MB |
| L3 BW (estimate) | ~600 GB/s |
| DDR4-3200 BW (real) | ~40 GB/s |
| Hit rate at top-20 | 55–65% |
| Effective BW | 0.60 × 600 + 0.40 × 40 = **~376 GB/s** |
| Ceiling lift | ~9× theoretical |
| Realized TG win (bandwidth-bound regime, with FA/norm/residual contention) | **+25–40% expected** |

Working set fits with 6 MB headroom for FA scratch + norm + residual on the same CCX.

## 3. Implementation Phases

| Phase | Scope | Effort |
|---|---|---|
| 6f-1 | Profiler extension: dump `selected_experts` (I32 top-k indices) | 1 d |
| 6f-2 | Offline analyzer → `expert-hotness.json` | 1 d |
| 6f-3 | Runtime prefetch hook in CPU `mul_mat_id` path | 1–2 d |
| 6f-4 | Validation matrix + perf gate | 0.5 d |

---

## 4. Phase 6f-1 — Profiler Extension

**Existing:** `common/router-profile.{h,cpp}` already dumps router logits via `--log-router-stats`. Reuse the same flag and binary file format.

**Add a second record type per layer per token:**

```c
struct router_record {
    uint32_t magic;          // 'RPRF' for logits, 'RPRS' for selected
    uint32_t layer_idx;
    uint32_t n_expert;       // logits only
    uint32_t n_expert_used;  // selected only
    uint32_t n_tokens;
    // payload: float[n_expert * n_tokens]   (RPRF)
    //          int32_t[n_expert_used * n_tokens] (RPRS)
};
```

**Hook point:** Tap the `selected_experts` tensor produced at `src/llama-graph.cpp:1374` (already named `ffn_moe_topk-N` via the cb chain). Add an `RPRS` filter in the same `router_profile_cb_eval`.

**Builder grep targets:**
- `common/router-profile.h` — add `record_selected()` API.
- `src/llama-graph.cpp` — already named, just extend filter regex.

## 5. Phase 6f-2 — Offline Analyzer

Tool: `tools/analyze-expert-hotness.py` (extend existing `tools/profile-router.py`).

**Input corpora (concat):**
- WikiText-2 train (full)
- StarCoder-Python sample (~5 MB)
- Optional: agentic chat traces from `/tmp/agentic-trace.bin`

**Algorithm:**
1. Read every `RPRS` record from `router-*.bin`.
2. Per `(layer_idx, expert_id)` count dispatch frequency across all tokens.
3. Sort descending; take top-20 per layer.
4. Compute model identity hash: SHA-256 of (n_layers, n_expert, n_expert_used, n_embd, GGUF tensor name list digest).

**Output: `expert-hotness.json`**
```json
{
  "schema_version": 1,
  "model_hash": "sha256:...",
  "model_name": "Qwen3-Next-80B-A3B-IQ2_XXS",
  "n_layers": 48,
  "n_expert": 512,
  "top_k": 20,
  "layers": {
    "0":  [42, 17, 301, ...],
    "1":  [88, 412, ...]
  },
  "stats": {
    "tokens_analyzed": 32768,
    "mean_top20_share": 0.58
  }
}
```

Ship to deploy alongside the GGUF: `models/<model>/expert-hotness.json`.

## 6. Phase 6f-3 — Runtime Prefetch Hook

### 6f-3a. Loader

At server startup (after model load, before first forward):
- New CLI flag: `--expert-hotness <path>`
- New file: `src/llama-expert-hotness.{h,cpp}`
  - `struct expert_hotness { std::vector<std::vector<int32_t>> per_layer; std::string model_hash; };`
  - `bool load(const char * path, const llama_model & model);` — verifies `model_hash` matches; logs warn + disables on mismatch.
- Owned by `llama_context`.

### 6f-3b. Pointer resolution

Need raw `void *` for each expert's weight block. GGML stores tensor data either:
- mmap'd (default for GGUF) → `tensor->data` is a stable VA into the file mapping.
- copied into backend buffer → `tensor->data` is into `ggml_backend_buffer`.

Both cases: `tensor->data` is the right pointer for `__builtin_prefetch` provided we cap stride to `ggml_nbytes(tensor) / n_expert`. For grouped MoE tensors (`ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`), expert N's slice starts at:
```c
char * base = (char *)tensor->data;
size_t stride = ggml_row_size(tensor->type, tensor->ne[0]) * tensor->ne[1]; // per-expert
void * expert_ptr = base + stride * expert_id;
```
**Open question (6f-3 task):** confirm row-major layout vs interleaved by reading `llama-model.cpp` MoE tensor packing. If interleaved, prefetch becomes scatter — still works, just multiple `__builtin_prefetch` calls per expert.

### 6f-3c. Prefetch dispatch

**Landing point:** CPU `mul_mat_id` operator in `ggml/src/ggml-cpu/ops.cpp` — function `ggml_compute_forward_mul_mat_id`. The early prelude (before per-expert thread fan-out) is where to issue prefetch for *this layer's* hot set.

**Pseudocode (insert at top of `ggml_compute_forward_mul_mat_id`, ithread==0 branch):**

```c
// llama-tq Phase 6f: hot-expert L3 prime
const int32_t layer_idx = ggml_get_op_params_i32(dst, 0);
if (g_expert_hotness && layer_idx >= 0) {
    const auto & hot = g_expert_hotness->per_layer[layer_idx];
    const struct ggml_tensor * src0 = dst->src[0];
    const size_t per_expert = ggml_nbytes(src0) / src0->ne[2];
    char * base = (char *)src0->data;
    for (int32_t eid : hot) {
        char * p   = base + per_expert * (size_t)eid;
        char * end = p + per_expert;
        for (char * q = p; q < end; q += 64) {
            __builtin_prefetch(q, 0, 3);
        }
    }
}
```

### 6f-3d. Layer-idx threading

`ggml_tensor` has no native "which transformer layer" field. Two options:

1. **Op params:** stash `layer_idx` in `tensor->op_params[0]` at graph-build time in `build_moe_ffn`.
2. **Name parse:** parse `tensor->name` like `"blk.17.ffn_gate_exps.weight"`.

**Recommended:** option 1 (op_params). Set once when emitting `MUL_MAT_ID` in `src/llama-graph.cpp:1374` area.

### 6f-3e. Alternative: `madvise(MADV_WILLNEED)`

Not a substitute. `MADV_WILLNEED` schedules background readahead from disk → page cache; it does NOT promote into L3. For mmap'd GGUF that's already fully resident in page cache, `MADV_WILLNEED` is a no-op. Stick with `__builtin_prefetch`.

## 7. Validation Plan (6f-4)

**A/B harness:**
- Baseline branch: `turboquant` (current tip).
- Test branch: `phase6f-prefetch`.
- Deploy: `deploy-80b.sh` — 80B-IQ2_XXS, 20 expert layers offloaded to CPU, ctx 200k×1, partial offload mode.

**Metrics:**
| Metric | Tool | Pass criterion |
|---|---|---|
| TG t/s @ 4k context | `llama-bench -p 0 -n 128` | **≥ +25%** vs baseline |
| TG t/s @ 32k context | same with `-c 32768` | ≥ +20% |
| PPL on wikitext-2 | `llama-perplexity` | Δ ≤ 0.005 (numerically zero) |
| L3 miss rate | `perf stat -e LLC-loads,LLC-load-misses` | hot expert layers show ≥ 30 pp miss-rate drop |
| Cross-CCX traffic | nsys / `perf c2c` | no increase |

**Negative test:** load with `--expert-hotness` pointing to a mismatched JSON → loader rejects, server runs unmodified, TG matches baseline.

**Per fork policy:** if any benchmark regresses outside noise band (±1.5%), do not merge.

## 8. Risks

1. **L3 thrashing:** FA scratch + KV reads + residual on the same CCX may evict prefetched lines. Mitigation: nsys profile in 6f-4; if thrashing, reduce hot-set to top-15 (7.5 MB).
2. **Per-deploy hotness file:** Different model = different IDs. Loader hash check enforces this. Document in README.
3. **mmap precondition:** If `--no-mmap` is set, `tensor->data` is in a backend buffer — still a valid VA, prefetch still works. No regression risk.
4. **Non-contiguous expert layout:** if `ffn_*_exps` tensors are interleaved rather than block-stacked, prefetch becomes scatter. Confirm layout in `llama-model.cpp` MoE tensor builder before coding 6f-3b.
5. **TLB pressure:** 10 MB across 4 KB pages = 2560 PTEs. Mitigate by enabling THP on host: `echo always > /sys/kernel/mm/transparent_hugepage/enabled`. Document in deploy-80b.sh.
6. **CCX pinning required:** Without `taskset`/cpuset to one CCX, threads bounce → L3 effect halves. Deploy script must pin CPU threads (`--threads 8 --cpu-mask 0xFF`) to a single CCX. gpu00 is a KVM guest; host-side cpuset may also be needed for full effect.

## 9. Open Questions

1. **GGML allocator path:** is the MoE expert tensor mmap'd file or copied into `ggml_backend_cpu_buffer`?
2. **Layout of grouped expert tensors:** block-stacked `[ne0, ne1, n_expert]` or interleaved?
3. **Are prefetch addresses in physically contiguous memory?**
4. **Cross-arch coverage:** spec targets Qwen3 MoE first. Mixtral / DeepSeek-MoE need same `op_params[0]` setter in their `build_moe_ffn` paths.

## 10. Deliverables Checklist

- [ ] `common/router-profile.cpp` — `record_selected()` added (6f-1)
- [ ] `src/llama-graph.cpp` — selected-experts profiler hook (6f-1)
- [ ] `tools/analyze-expert-hotness.py` — JSON exporter (6f-2)
- [ ] `models/<model>/expert-hotness.json` for 80B and 122B deploys (6f-2)
- [ ] `src/llama-expert-hotness.{h,cpp}` — loader + hash verify (6f-3a)
- [ ] `src/llama-graph.cpp` — `op_params[0] = layer_idx` on MUL_MAT_ID emit (6f-3d)
- [ ] `ggml/src/ggml-cpu/ops.cpp` — prefetch loop in `ggml_compute_forward_mul_mat_id` prelude (6f-3c)
- [ ] CLI flag `--expert-hotness` plumbed in `common/arg.cpp` and `tools/server/server.cpp`
- [ ] `deploy-80b.sh` — CCX pin + THP enable + `--expert-hotness` arg
- [ ] Bench report under `docs/benchmarks/2026-04-XX-phase6f.md`
- [ ] Update `FORK_CHANGES.md`
