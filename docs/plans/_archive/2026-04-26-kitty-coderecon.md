# KITTY Hybrid KV Cache — Code Reconnaissance

**Date:** 2026-04-26
**Repo:** `/mnt/d/repos/llama-tq` (branch: `turboquant`)
**Mode:** READ-ONLY recon — no edits.
**Goal:** Map the file:line surface needed to implement KITTY-style hybrid cache:
*sink (first N tokens, fp16) + Q-buffer (recent N tokens, fp16) + low-bit middle (existing KTQ/VTQ)*.

---

## 1. KV slot write path

The "where does a token's K/V land in the cache" pipeline:

### 1a. Graph-level write call sites

`build_attn()` overloads in `src/llama-graph.cpp` all funnel into the same two MCTX calls:

```cpp
// src/llama-graph.cpp:2127-2133  (the canonical KV-attn variant)
{
    const auto & k_idxs = inp->get_k_idxs();
    const auto & v_idxs = inp->get_v_idxs();

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
}
```

Other build_attn variants do the same: `:2131-2132` (kv-attn), `:2217` (kv-attn k-only), `:2293-2299` (cross-attn), `:2358-2367` (iSWA / hybrid-recurrent attn). **All token writes go through `cpy_k`/`cpy_v`.**

### 1b. K quantize + store

```cpp
// src/llama-kv-cache.cpp:1484
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur,
                                    ggml_tensor * k_idxs, int32_t il,
                                    const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    // deferred K quantization: during prefill, write to f16 staging buffer
    ggml_tensor * k = (deferred_state == TQ_DEFERRED_STAGING && layers[ikv].k_staging)
                      ? layers[ikv].k_staging : layers[ikv].k;
    ...
    return ggml_set_rows(ctx, k, k_cur, k_idxs);   // line 1518
}
```

The actual quantization happens implicitly inside `ggml_set_rows()` — the destination tensor's `type` (KTQ*/VTQ*/F16/Q8_0…) drives the kernel selected by the backend (CUDA/CPU). For KITTY: a per-token destination *type* would mean different rows of the same cache go to different kernels — not supported by the current `ggml_set_rows` design. **Practical implication:** a KITTY hybrid would need either separate physical buffers (sink-buffer FP16 + middle-buffer KTQ) selected by token index *before* the call, or a custom "set_rows_dispatch" op.

### 1c. V quantize + store

```cpp
// src/llama-kv-cache.cpp:1521
ggml_tensor * llama_kv_cache::cpy_v(...) const {
    auto * v = (deferred_state == TQ_DEFERRED_STAGING && layers[ikv].v_staging)
               ? layers[ikv].v_staging : layers[ikv].v;
    ...
    if (!v_trans) {                                     // line 1544 (FA path)
        ...
        return ggml_set_rows(ctx, v, v_cur, v_idxs);    // line 1557
    }
    // transposed-V path:
    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);   // line 1578
}
```

Same staging-pointer pattern as K (`use_deferred_v` at `:357-371`).

### 1d. Slot-index computation (positional? sequential?)

`slot_info` declared at `src/llama-kv-cache.h:41`:

```cpp
struct slot_info {
    using idx_vec_t = std::vector<uint32_t>;
    uint32_t s0, s1;                        // stream range
    std::vector<llama_seq_id> strm; // [ns]
    std::vector<idx_vec_t>    idxs; // [ns]   ← per-stream cell indices
    ...
    bool is_contiguous() const { ... }   // line 84
};
```

Indices are **cell positions in a ring buffer**, not absolute token positions. They are produced by `find_slot()` (declared `src/llama-kv-cache.h:207`) and converted to a host I64 buffer that the GPU consumes:

```cpp
// src/llama-kv-cache.cpp:1648  set_input_k_idxs
for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
    const int64_t offs = sinfo.strm[s]*get_size();
    for (uint32_t i = 0; i < sinfo.size(); ++i) {
        data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
    }
}
```

Crucially, `cpy_k()` calls `GGML_UNUSED(sinfo)` (line 1485) — it only consumes `k_idxs` (the I64 tensor), not the per-stream metadata. The "absolute token position" (needed for `pos < N_sink` decisions) lives in `llama_ubatch::pos[]` upstream of the call. To route by token-position you must thread `ubatch.pos` (or a derived "is-sink" mask) into `cpy_k`/`cpy_v`.

---

## 2. KV cache type detection (`is_vtq_v` etc.)

The lines mentioned in the request (288-291, 489-492) are *local constants* inside the constructor — **not function symbols.** There is no centralized `is_vtq()` helper today. Every site re-derives the predicate.

### 2a. All in-tree occurrences (verified via `grep is_vtq_v|is_ktq|is_tq_k|is_tq_v` in `src/`)

| File:line | Context |
|---|---|
| `src/llama-kv-cache.cpp:269-270` | `is_tq_v` for FA-V-trans guard (forces V to f16 when not transposed) |
| `src/llama-kv-cache.cpp:290-297` | `is_vtq_v` + `is_ktq_v` for **attention-sink protection** branch |
| `src/llama-kv-cache.cpp:305-307` | `is_tq_k` / `is_tq_v` / `is_vtq_v` for **boundary-layer protection** |
| `src/llama-kv-cache.cpp:340-341` | `layer_uses_tq` for K staging buffer (deferred K) |
| `src/llama-kv-cache.cpp:358-361` | `layer_uses_vtq2` for V staging buffer (deferred V) |
| `src/llama-kv-cache.cpp:495-498` | `is_vtq_v` for D*H*D rotation matrix (ZeroQuant-V) |
| `src/llama-kv-cache.cpp:521` | `is_vtq_v` for `attn_rot_v` activation gate |

**Dispatch flow:** Type is decided once per layer in the constructor (`:240-330`), baked into `eff_type_k`/`eff_type_v`, and tensors allocated at `:329-330`. After that, `cpy_k`/`cpy_v` are type-agnostic — the *tensor type* drives kernel selection. **No runtime type switch per token** exists.

### 2b. Recommended refactor before KITTY

Pull these scattered predicates into `src/llama-kv-cache.h` as inline helpers:

```cpp
inline bool is_ktq_type(ggml_type t);
inline bool is_vtq_type(ggml_type t);
inline bool is_tq_type(ggml_type t) { return is_ktq_type(t) || is_vtq_type(t); }
```

This isolates the long `t == X || t == Y || ...` chains and prepares for the KITTY "is this slot fp16 or KTQ?" check.

---

## 3. Existing attention-sink support

### 3a. Two unrelated "sink" concepts coexist in this repo

**(A) GPT-OSS learned per-head attention sinks** (upstream-llama.cpp feature, not StreamingLLM):

- Tensor: `LLM_TENSOR_ATTN_SINKS = "blk.%d.attn_sinks"` in `src/llama-arch.cpp:532`.
- Loaded as `layer.attn_sinks` of shape `{n_head}`: `src/llama-model.cpp:7197` (required) and `:7859` (optional).
- Threaded through `build_attn(..., ggml_tensor * sinks, ...)` — see `src/llama-graph.h:884,899,914,929,945,960` (six overloads).
- Applied at attention-output time:
  - `src/llama-graph.cpp:1897` → `ggml_flash_attn_ext_add_sinks(cur, sinks)` (FA path)
  - `src/llama-graph.cpp:1954` → `ggml_soft_max_add_sinks(kq, sinks)` (eager softmax)

This is a learned bias added *after* attention scores. **It does NOT preserve the first-N tokens at higher precision** — different mechanism entirely.

**(B) TurboQuant `tq_protect_sinks` (StreamingLLM-style, this fork's own work):**

- CLI: `--tq-protect-sinks N` at `common/arg.cpp:2061-2072`.
- Plumbed: `common/common.h:553` → `cparams.tq_protect_sinks` (`common/common.cpp:1520`) → `llama_kv_cache` ctor (`src/llama-kv-cache.cpp:102,110`) → iSWA wrapper (`src/llama-kv-cache-iswa.cpp:27,70,77`).
- Effect at `src/llama-kv-cache.cpp:289-303`:

```cpp
// Attention-sink protection (StreamingLLM, arXiv:2309.17453):
// first tokens carry outsized attention weight. When tq_protect_sinks > 0,
// protect the first **attention** KV layer's V-cache at f16.
if (tq_protect_sinks > 0 && kv_layer_idx_sink == 0 && hparams.has_kv(il)) {
    if (is_vtq_v || is_ktq_v) {
        eff_type_v = GGML_TYPE_F16;
        ...
    }
}
```

**Important:** today's `tq_protect_sinks` protects the **entire first attention layer's V-cache**, NOT the first-N **tokens**. The variable name is misleading. KITTY's actual ask — "first N tokens at fp16 across all layers" — does **not** exist yet.

### 3b. What needs to be added for KITTY

| Need | Status |
|---|---|
| First-N-tokens fp16 region | **Missing.** Needs new dual-buffer or position-aware dispatch. |
| Per-token "is-sink" classification at write time | **Missing.** `cpy_k`/`cpy_v` ignore `sinfo` and `ubatch.pos`. |
| Per-token "is-recent-Q-buffer" sliding window | **Missing.** No rolling fp16 region exists. |
| Per-layer fp16 sink (current `tq_protect_sinks`) | Done — but *layer-granular*, not what KITTY wants. |
| GPT-OSS learned per-head sink | Done — orthogonal to KITTY, leave alone. |

---

## 4. CLI parser for cache types

### 4a. Existing parse pipeline

```cpp
// common/arg.cpp:384-409  — type whitelist
const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_KTQ1_1, GGML_TYPE_KTQ2_1, GGML_TYPE_KTQ3_1, GGML_TYPE_KTQ4_1,
    GGML_TYPE_VTQ1_1, GGML_TYPE_VTQ2_1, GGML_TYPE_VTQ3_1, GGML_TYPE_VTQ4_1,
    GGML_TYPE_VTQ2_2, GGML_TYPE_VTQ3_2, GGML_TYPE_VTQ4_2, GGML_TYPE_VTQ_MIXED,
    GGML_TYPE_VTQ2_3, GGML_TYPE_VTQ3_3, GGML_TYPE_VTQ4_3,
};

// common/arg.cpp:411-418
static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types)
        if (ggml_type_name(type) == s) return type;
    throw std::runtime_error("Unsupported cache type: " + s);
}

// common/arg.cpp:2024-2049  — --cache-type-k / --cache-type-v
add_opt(common_arg({"-ctk", "--cache-type-k"}, "TYPE", ...,
    [](common_params & params, const std::string & value) {
        params.cache_type_k = kv_cache_type_from_str(value);
    }).set_env("LLAMA_ARG_CACHE_TYPE_K"));
```

### 4b. Adding `--kv-sink-tokens N` and `--kv-recent-tokens M`

Three coordinated edits, mirroring the pattern at `common/arg.cpp:2061-2072` (`--tq-protect-sinks`):

1. **`common/common.h`** (~line 553 area, near `tq_protect_sinks`): add fields
   ```cpp
   uint32_t kv_sink_tokens   = 0;   // KITTY: first N tokens kept at fp16
   uint32_t kv_recent_tokens = 0;   // KITTY: rolling Q-buffer (last M tokens)
   ```
2. **`common/common.cpp:1519-1520`**: thread to cparams (new fields needed in `llama_cparams`).
3. **`common/arg.cpp`** after line 2072: clone the `--tq-protect-sinks` block twice for the two new flags. Use `set_examples({SERVER, COMPLETION, CLI, ...})` and `.set_env("LLAMA_ARG_KV_SINK_TOKENS")` etc.

Plumb-through chain after that:
- `src/llama-cparams.h` (declare fields)
- `src/llama-context.cpp:299` (already does this for `tq_protect_sinks` — copy the line)
- `src/llama-kv-cache.cpp:102` ctor signature; `src/llama-kv-cache-iswa.cpp:27,70,77` matching changes.

**LOC for CLI plumbing alone: ~30-40 lines across 6 files.**

---

## 5. Per-token metadata

**There is no per-token metadata array today.** Verified:

- `slot_info` (`src/llama-kv-cache.h:41-99`) carries only cell indices and stream ids. No type/precision/protection fields.
- `tq_deferred_state` (`src/llama-kv-cache.h:16-21`) is a *whole-cache* state machine, not per-token.
- `llama_ubatch` carries `pos[]` (token absolute positions) but is not consulted in `cpy_k`/`cpy_v`.
- The cache uses a ring buffer (see `head` comment at `src/llama-kv-cache.h:304-305`), so cell index ≠ token position. Any KITTY classification ("is this cell currently a sink-cell?") needs a parallel array indexed by **cell**, not token.

### Suggested metadata layout for KITTY

Add to `class llama_kv_cache` (private members near `:274 deferred_state`):

```cpp
// KITTY hybrid cache — per-cell precision tag
//   0 = low-bit (KTQ/VTQ middle), 1 = fp16 sink, 2 = fp16 Q-buffer
std::vector<uint8_t> cell_kind;   // size = kv_size * n_stream
uint32_t kv_n_sink   = 0;
uint32_t kv_n_recent = 0;
```

Update sites:
- `apply_ubatch()` (declared `src/llama-kv-cache.h:210`) — write the tag when assigning a slot.
- Eviction logic in `find_slot()` — never evict a sink cell; rotate Q-buffer cells correctly.
- `cpy_k`/`cpy_v` — branch on `cell_kind[idxs[i]]` to pick destination buffer.

**~80-150 LOC + new fp16 shadow buffers (parallel `layers[].k_sink`, `layers[].v_sink`, `layers[].k_recent`, `layers[].v_recent`).**

---

## 6. Build system

### 6a. CMake

`src/CMakeLists.txt:23-24` lists `llama-kv-cache.cpp` and `llama-kv-cache-iswa.cpp` already. **No CMake change needed** if KITTY logic stays in those files. New helper headers (e.g. `llama-kv-kitty.h`) typically don't need explicit listing if pulled via `#include`.

### 6b. Header updates required (minimum)

| File | Reason |
|---|---|
| `src/llama-kv-cache.h` | new ctor params, new private members, new `slot_info` fields or new method signatures |
| `src/llama-kv-cache-iswa.h:29` | mirror new ctor params |
| `src/llama-cparams.h` | new fields `kv_sink_tokens`, `kv_recent_tokens` |
| `common/common.h` | CLI struct fields |
| `include/llama.h` | only if a public API/struct (e.g. `llama_context_params`) gets new fields |

### 6c. Tests

Existing TQ-relevant test:
- `tests/test-vtq2-cached-roundtrip.cpp`

No KV-cache integration test exists. New tests advisable:
- `tests/test-kitty-sink-protection.cpp` — perplexity smoke at long ctx (sink fp16 should beat plain KTQ on PG19/Long-PPL).
- `tests/test-kitty-cellkind-rotate.cpp` — unit test for ring-buffer eviction never dropping sink cells.

CMake side: add to `tests/CMakeLists.txt` following the existing test pattern (typically `llama_test()` macro).

---

## Estimated patch surface (LOC delta)

| File | Δ LOC | Note |
|---|---|---|
| `common/arg.cpp` | +30 | two new CLI flags (clone of `--tq-protect-sinks`) |
| `common/common.h` | +4 | two `uint32_t` fields |
| `common/common.cpp` | +4 | propagate to cparams |
| `src/llama-cparams.h` | +4 | two `uint32_t` fields |
| `src/llama-context.cpp` | +4 | propagate from params |
| `src/llama-kv-cache.h` | +30 | ctor sig, new members, helper enum |
| `src/llama-kv-cache.cpp` | +250 | dual-buffer alloc, `cpy_k`/`cpy_v` dispatch, `apply_ubatch` tagging, `find_slot` eviction guard |
| `src/llama-kv-cache-iswa.{h,cpp}` | +20 | thread params through |
| `src/llama-graph.cpp` | +10 | optionally pass `ubatch.pos` into mctx |
| `tests/test-kitty-*.cpp` | +200 (new) | two unit tests |
| `tests/CMakeLists.txt` | +10 | register tests |
| **Total** | **~570 LOC** | bulk in `llama-kv-cache.cpp` |

---

## Risks (3 things that could break)

### Risk 1 — ggml_set_rows is type-monolithic per call

`cpy_k`/`cpy_v` ultimately call `ggml_set_rows(ctx, dest, src, idxs)`. The destination tensor has **one** type, and the kernel selected by the backend (CUDA in `ggml/src/ggml-cuda/set-rows.cu`) handles that type. Routing different rows of a single ubatch to different destination types means **multiple** `ggml_set_rows` calls per layer per step (one per region). That increases per-step kernel-launch count — non-trivial on tiny models / decode steps. Mitigation: batch by region pre-graph; emit at most 3 ops per layer (sink + recent + middle) only when both regions are non-empty.

### Risk 2 — Ring-buffer eviction violates KITTY invariants

`find_slot()` (declaration at `src/llama-kv-cache.h:207`, plus `head` cursor at `:304-305`) treats the cache as a wrap-around ring. Without careful changes, eviction will overwrite sink cells once `head` wraps. Worse, the **iSWA** cache (`src/llama-kv-cache-iswa.cpp`) layers a sliding window on top — interactions between SWA window, Q-buffer window, and sink region need explicit testing. Mitigation: explicit `cell_kind`-aware eviction, and add an assertion that `cell_kind[evicted] != KIND_SINK`.

### Risk 3 — Deferred-quantization state machine clash

The existing `tq_deferred_state` machine (`src/llama-kv-cache.h:16-21`) flips destination between `k`/`k_staging` based on prefill→decode transition (`cpy_k:1490`). Layered onto KITTY's per-token dispatch, this becomes a **2-D state** (deferred-state × cell-kind) — easy to introduce subtle bugs where, e.g., a "sink cell" is written to the staging buffer during prefill and never reaches the fp16 sink-buffer. Mitigation: KITTY sink/recent buffers should be plain fp16 (skip staging entirely for those regions), and the dispatch should branch on cell-kind **before** consulting `deferred_state`.

---

## Open questions for spike kickoff

1. **Q-buffer eviction semantics:** when token N+M+1 arrives, where does the oldest Q-buffer cell go — into the low-bit middle, or just discard? Production LM systems usually quantize-and-keep.
2. **Sink/Q-buffer per-layer or shared?** Allocating per-layer fp16 buffers for `n_sink + n_recent = 256` tokens at 80 layers × 5120 dim × 2 byte ≈ 200 MB. Affordable — confirm with Lance.
3. **Interaction with VTQ randomized Hadamard** (`src/llama-kv-cache.cpp:495-498`): sink cells stored as fp16 should **not** apply the D*H*D rotation. Need a separate non-rotated path for sink writes/reads.
4. **iSWA + KITTY:** iSWA layers already re-use a sliding window. Does KITTY's Q-buffer subsume that, or are they independent? If subsumed, large simplification possible.

---

*End of recon. Total reading: ~12 files, ~80 grep hits. No edits performed.*
