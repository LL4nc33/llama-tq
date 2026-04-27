# KITTY-Pattern Hybrid KV Cache: Sink + Q-Buffer Spec

**Date:** 2026-04-26
**Branch:** turboquant
**Reference paper:** arXiv:2511.18643 (KITTY)
**Author:** Builder agent (spec only, do NOT implement yet)
**Related work in repo:** `tq_protect_sinks` (StreamingLLM, layer-0 V→f16) is a precursor — this spec generalises it across all layers and adds a sliding high-bit recency window.

---

## 1. Problem & Motivation

The current TQ stack quantises every K/V slot uniformly with `--cache-type-k/v` (modulo two existing exceptions: `tq_protect_layers` for boundary layers, and `tq_protect_sinks` which only protects **layer 0** V-cache to f16). Two attention-pattern observations from the literature are not yet exploited at the slot level:

1. **Attention sinks** (Xiao 2023, arXiv:2309.17453) — the first ~4 tokens absorb a disproportionate amount of attention mass. Quantising them at low bit (KTQ2_1, ~3.5 bpw) measurably degrades long-context recall.
2. **Recency window** — the most recent ~128 tokens get high attention weight in the next step. The KITTY paper shows that holding the *recency* window at ≥4 bpw and the *bulk middle* at ≤3.5 bpw matches FP16 RULER scores at <40% of the FP16 KV footprint.

Goal: per-token bit-width — sink (FP16) / mid (current `--cache-type-k/v`) / qbuffer (KTQ4_1 by default), with a sliding qbuffer that follows the write head.

---

## 2. CLI Surface (new flags, additive)

Add to `common/arg.cpp` immediately after the existing `--tq-protect-sinks` block (line 2072) and before `--tq-profile-heads` (line 2073):

```cpp
add_opt(common_arg(
    {"--kv-sink-tokens"}, "N",
    string_format(
        "KITTY hybrid KV: hold the first N tokens of each sequence in FP16\n"
        "(StreamingLLM attention sinks). 0 disables the sink region.\n"
        "(default: %u)",
        params.kv_sink_tokens),
    [](common_params & params, int value) {
        params.kv_sink_tokens = value >= 0 ? (uint32_t)value : 0;
    }
).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI,
                LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BENCH})
 .set_env("LLAMA_ARG_KV_SINK_TOKENS"));

add_opt(common_arg(
    {"--kv-qbuffer-tokens"}, "N",
    string_format(
        "KITTY hybrid KV: hold the most recent N tokens at high bit\n"
        "(see --kv-qbuffer-type). 0 disables the recency window.\n"
        "(default: %u)",
        params.kv_qbuffer_tokens),
    [](common_params & params, int value) {
        params.kv_qbuffer_tokens = value >= 0 ? (uint32_t)value : 0;
    }
).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI,
                LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BENCH})
 .set_env("LLAMA_ARG_KV_QBUFFER_TOKENS"));

add_opt(common_arg(
    {"--kv-qbuffer-type"}, "TYPE",
    string_format(
        "KITTY hybrid KV: cache type used in the qbuffer (recency) region.\n"
        "Must be a higher-bit type than --cache-type-k/v. allowed: %s\n"
        "(default: %s)",
        get_all_kv_cache_types().c_str(),
        ggml_type_name(params.kv_qbuffer_type)),
    [](common_params & params, const std::string & value) {
        params.kv_qbuffer_type = kv_cache_type_from_str(value);
    }
).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI,
                LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BENCH})
 .set_env("LLAMA_ARG_KV_QBUFFER_TYPE"));
```

Mid region (everything between the sink and the qbuffer) inherits `--cache-type-k`/`--cache-type-v` — no new flag.

### Defaults (in `common/common.h` near `tq_protect_sinks` at line 553)

```cpp
uint32_t  kv_sink_tokens    = 4;                    // KITTY sink region (FP16)
uint32_t  kv_qbuffer_tokens = 128;                  // KITTY recency window
ggml_type kv_qbuffer_type   = GGML_TYPE_KTQ4_1;     // recency bit-width
```

Mirror in `include/llama.h` `llama_context_params` near `tq_protect_sinks` (line 368), wire through `common.cpp:1520` block, and `src/llama-context.cpp:299` and `:3125`.

### Interaction with existing flags

* `--tq-protect-sinks N` is **deprecated** (continues to work, but logs a warning suggesting `--kv-sink-tokens N`). Internally, when `kv_sink_tokens > 0` we no longer need the layer-0-only special case at `llama-kv-cache.cpp:289-303` — the sink region applies uniformly to all attention layers via the per-token mechanism described in §4.
* `--tq-protect-layers` is orthogonal and continues to apply (boundary layers always q8_0 regardless of token region).

---

## 3. Memory Layout: Per-Token Bit-Width Tracking

The current `kv_layer` struct (`src/llama-kv-cache.h:238-254`) holds **one** quantised tensor per layer plus an **optional** f16 staging buffer used for deferred K/V conversion. To support three concurrent bit-widths per slot we need three storage tiers, but we want to avoid tripling VRAM. Two layout options were evaluated:

### Option A — Three fixed tensors per layer (rejected)

Allocate `k_sink` (`[n_embd_k_gqa, kv_sink_tokens]` FP16), `k_qbuffer` (`[n_embd_k_gqa, kv_qbuffer_tokens]` `kv_qbuffer_type`), `k_mid` (`[n_embd_k_gqa, kv_size - sink - qbuffer]` `type_k`). Same for V.

* Pro: trivial dispatch in `cpy_k`/`cpy_v` based on absolute slot index.
* Con: requires re-quantising data when the qbuffer slides (every decode token shifts the boundary). Cost = one full-row dequant + requant per decode step → unacceptable.

### Option B — Tiled "ring" with bit-width metadata (chosen)

Keep the existing `k`/`v` tensors at the **mid** quant type for the bulk of `kv_size`, plus two small **co-located** auxiliary tensors per layer: `k_sink_buf` (FP16, fixed `kv_sink_tokens` rows, never written after the first prefill) and `k_qbuffer_buf` (qbuffer-type, `kv_qbuffer_tokens` rows, ring-buffer indexed).

```cpp
// new fields in struct kv_layer (src/llama-kv-cache.h:238)
ggml_tensor * k_sink_buf    = nullptr;  // [n_embd_k_gqa, kv_sink_tokens]   FP16
ggml_tensor * v_sink_buf    = nullptr;  // [n_embd_v_gqa, kv_sink_tokens]   FP16
ggml_tensor * k_qbuffer_buf = nullptr;  // [n_embd_k_gqa, kv_qbuffer_tokens] kv_qbuffer_type
ggml_tensor * v_qbuffer_buf = nullptr;  // [n_embd_v_gqa, kv_qbuffer_tokens] kv_qbuffer_type
std::vector<ggml_tensor *> k_sink_stream, v_sink_stream;
std::vector<ggml_tensor *> k_qbuffer_stream, v_qbuffer_stream;
```

Per-token region tag in `llama_kv_cells` (file `src/llama-cells.h` / `llama-kv-cells.h`, near pos_set):

```cpp
enum kv_region : uint8_t { KV_REGION_SINK = 0, KV_REGION_MID = 1, KV_REGION_QBUFFER = 2 };
std::vector<uint8_t> region;   // size == kv_size; only populated when KITTY enabled
```

`get_k`/`get_v` (`llama-kv-cache.cpp:1428` / `:1450`) become **gather views**: for each slot in `sinfo.idxs[s]` look up `region[idx]` and dispatch to one of the three tensors. Implementation note: instead of branching per-row, build three sub-views (sink prefix, qbuffer suffix, mid middle) via `ggml_view_4d` + `ggml_concat` along the sequence axis. Order in the concat must match `sinfo.idxs` so the attention mask remains correct.

### VRAM delta (per attention layer, `n_embd_gqa = 1024`, ctx = 64k)

| Region   | Tokens | Type     | Bytes (K)             |
|----------|--------|----------|-----------------------|
| Sink     | 4      | FP16     | 4 × 1024 × 2 = 8 KiB  |
| QBuffer  | 128    | KTQ4_1   | 128 × 1024 × 5.5/8 ≈ 88 KiB |
| Mid      | 65404  | KTQ2_1   | 65404 × 1024 × 3.5/8 ≈ 28.6 MiB |
| **Total**|        |          | **~28.7 MiB** vs uniform KTQ2_1 at **28.7 MiB** |

Net overhead is **<0.5%** because the sink+qbuffer regions are already counted toward `kv_size` (they replace mid slots, not add to them). See §6 for the slot-eviction interaction.

---

## 4. Slot Write Logic (`cpy_k` / `cpy_v`)

Modify `llama_kv_cache::cpy_k` (`src/llama-kv-cache.cpp:1484-1519`) and `cpy_v` (`:1521-1565`) to fan out writes by region:

```cpp
// pseudo-code added to cpy_k
const uint32_t sink_n  = cparams.kv_sink_tokens;
const uint32_t qbuf_n  = cparams.kv_qbuffer_tokens;

if (sink_n == 0 && qbuf_n == 0) {
    // KITTY disabled — original code path
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

// Build three index sub-tensors per ubatch token based on absolute pos
// computed from sinfo + ubatch.pos[i]:
//   pos < sink_n                           -> sink_buf
//   pos >= seq_pos_max - qbuf_n            -> qbuffer_buf
//   else                                   -> k (mid)
//
// ggml_set_rows is called once per region with the appropriate sub-tensor.
```

The split is computed **on host** during `apply_ubatch` (`:1283-1354`) — that function already iterates per-token and has direct access to `ubatch.pos[i]`. Add region tagging there:

```cpp
// after cells.pos_set(idx, ubatch.pos[i]); at line 1312
if (kitty_enabled) {
    cells.region[idx] = (ubatch.pos[i] < sink_n) ? KV_REGION_SINK
                      : KV_REGION_QBUFFER;   // mid-promotion happens at slide time
}
```

For each ubatch we also build three index vectors (`k_idxs_sink`, `k_idxs_qbuf`, `k_idxs_mid`) passed alongside the existing `k_idxs` into `cpy_k`. The `build_input_k_idxs` graph builder (`src/llama-kv-cache.cpp` companion of `:1484`) will need a sibling `build_input_k_idxs_sink/qbuf/mid` — see §7 for LOC estimate.

### When a token leaves the qbuffer (slide event)

When a new token at position `p` enters the qbuffer, the token at position `p - qbuf_n` must be **demoted** from the qbuffer-type tensor to the mid-type tensor. Implementation:

1. Dequantise that one row from `k_qbuffer_buf` to a temporary FP16 row using the existing dequant kernel (already present for `KTQ4_1`).
2. Re-quantise into the mid tensor `k` using the standard KTQ2_1 row writer (already present — same path as deferred-K bulk convert at `llama-kv-cache.cpp:846-900`, `build_graph_deferred_convert`).
3. Update `cells.region[old_idx] = KV_REGION_MID`.

The slide is **at most one token per decode step** (per-stream). Re-using `build_graph_deferred_convert`'s row-level codepath keeps the implementation small. For prefill batches > 1 the slide is a small ggml graph appended after `cpy_k`/`cpy_v` (one row dequant + requant per excess token).

---

## 5. Decision Tree: Slot Eviction When Ctx Wraps the QBuffer Window

Three timelines matter:
* **t < sink_n** — All writes go to `*_sink_buf`. No demotion ever happens.
* **sink_n ≤ t < sink_n + qbuf_n** — Writes go to `*_qbuffer_buf`. No demotion yet.
* **t ≥ sink_n + qbuf_n** — Steady state. Each new token write also demotes the token at `t - qbuf_n` from qbuffer → mid.
* **Cache full (ring wraparound)** — When the ring buffer in `v_cells` wraps and overwrites a sink slot (i.e. `find_slot` selects a slot whose `region == KV_REGION_SINK`), that is a **bug condition** in pure ring-buffer mode (sink slots must be pinned). Two policies:

```
                    ┌─────────────────────┐
                    │  find_slot picks    │
                    │  idx with region X  │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   X = SINK                X = QBUFFER             X = MID
   ┌──────────┐            ┌──────────┐            ┌──────────┐
   │ REJECT — │            │ Demote   │            │ Standard │
   │ search   │            │ row to   │            │ overwrite│
   │ next idx │            │ mid then │            │ via      │
   │ (pin     │            │ overwrite│            │ ggml_    │
   │  sinks)  │            │          │            │ set_rows │
   └──────────┘            └──────────┘            └──────────┘
```

Implementation hook in `find_slot` (`src/llama-kv-cache.cpp:1084`): add a guard that skips slots where `cells.region[i] == KV_REGION_SINK`. Sink slots are pinned for the lifetime of the sequence — they are released only by `seq_rm` or `clear`.

For SWA caches the sink pin must coexist with the SWA window logic; we should only pin sinks in the **non-SWA** unified cache for the first iteration. Document SWA as out-of-scope for this spec (see Risk #3).

---

## 6. Bench Gate Criteria

Before merging:

| Test                                | Target                                     | Tooling                           |
|-------------------------------------|--------------------------------------------|-----------------------------------|
| RULER 64k passkey/MK-NIAH           | ≥ 95% of FP16 baseline                     | `tools/perplexity` + RULER subset |
| RULER 64k VT (variable tracking)    | ≥ 95% of FP16 baseline                     | same                              |
| Wikitext PPL @ ctx 4k               | ≤ +0.5% vs uniform KTQ2_1                  | `llama-perplexity`                |
| Decode TG @ 64k                     | ≥ 98% of uniform KTQ2_1 t/s                | `llama-bench` `-d 65536 -n 64`    |
| Prefill PP @ 64k                    | ≥ 95% of uniform KTQ2_1 t/s                | `llama-bench` `-d 65536 -p 4096`  |
| VRAM delta vs uniform KTQ2_1        | ≤ +1% at ctx 64k                           | `nvidia-smi` snapshot             |
| Existing TQ regression suite        | All pass                                   | `ctest -R tq`                     |

Promotion path: gate the feature behind `--kv-sink-tokens > 0 || --kv-qbuffer-tokens > 0` (off by default in initial release). Promote to default after one production-deploy week with no quality regressions.

---

## 7. Risks & Mitigations

### Risk 1 — Three-way gather in attention path tanks decode TG

`get_k`/`get_v` currently return a single contiguous `ggml_view_4d`. Fanning out to three sub-views + `ggml_concat` adds graph nodes and (worst case) an extra device-side copy per layer per decode step. On a 64-layer model this could cost 5–10% TG.

**Mitigation:** prototype with `ggml_concat` first; if TG drops below the §6 gate, fall back to a **fused gather kernel** that reads from three source tensors directly (one new CUDA kernel `kitty_gather_kv` — adds ~150 LOC but eliminates the concat copy). Existing `cpy_v` deferred-V codepath already proves that custom small kernels integrate cleanly into the cgraph.

### Risk 2 — Slide-time row dequant/requant introduces decode-step latency spikes

Demoting one row from KTQ4_1 → KTQ2_1 per decode step is cheap (~5µs on RTX 2060) but it adds a synchronisation point. On batched decode (parallel slots > 1) this multiplies.

**Mitigation:** batch the demotion across all layers into a single graph node (one fused kernel per slide event). Re-use `build_graph_deferred_convert` (`llama-kv-cache.cpp:341-343`) which already does layer-batched bulk convert — extract its inner row-loop into a callable that takes `(src_buf, dst_buf, src_idx, dst_idx, src_type, dst_type)`. If still too slow, allow `--kv-qbuffer-slide-stride N` to slide every N tokens at the cost of qbuffer width drift.

### Risk 3 — Interaction with SWA, deferred K/V, and per-layer V types

The repo already has three orthogonal slot-level mechanisms: `swa_type` (sliding-window attention masking), `tq_deferred_k/v` (f16 staging during prefill), and `user_type_v_layers` (per-layer V quant from `llama-kv-cache.cpp:261-264`). KITTY adds a fourth axis. Worst combinations:

* SWA + KITTY: SWA already evicts old tokens; sink pinning conflicts with the SWA window (sinks may fall outside the window but must remain attended to). Out of scope for this spec — gate KITTY off when `swa_type != LLAMA_SWA_TYPE_NONE` and log a warning.
* Deferred K + KITTY: during prefill, all writes go to the f16 staging buffer (`k_staging`). The region tagging still applies (sink/qbuffer/mid), but the actual quantisation into the right tensor only happens at `TQ_DEFERRED_READY` bulk-convert time. The bulk-convert routine (`build_graph_deferred_convert`) must be extended to dispatch per-region.
* Per-layer V types (`user_type_v_layers`): KITTY sets the *region* type, the per-layer override sets the *mid* type. They compose naturally — `kv_qbuffer_type` is uniform across layers, mid type follows the per-layer vector.

**Mitigation:** explicit refusal in `llama-kv-cache.cpp` constructor when SWA is on and `kv_sink_tokens || kv_qbuffer_tokens > 0`. Add unit test `tests/test-kv-cache-kitty.cpp` covering all three interaction matrices.

---

## 8. Estimated LOC Delta

| Area                                                           | Files                                                | Add | Mod |
|----------------------------------------------------------------|------------------------------------------------------|-----|-----|
| CLI + params plumbing                                          | `common/arg.cpp`, `common/common.{h,cpp}`            |  60 |  10 |
| Public API surface                                             | `include/llama.h`                                    |  10 |   2 |
| Context wiring                                                 | `src/llama-context.cpp`                              |  15 |   4 |
| `kv_layer` struct + ctor (sink/qbuffer tensor allocation)      | `src/llama-kv-cache.h`, `:cpp` (around `:240-380`)   | 130 |  20 |
| `llama_kv_cells::region` storage + accessors                   | `src/llama-kv-cells.h`                               |  40 |   5 |
| `apply_ubatch` region tagging                                  | `src/llama-kv-cache.cpp:1283`                        |  20 |   5 |
| `cpy_k` / `cpy_v` three-region fan-out                         | `src/llama-kv-cache.cpp:1484` / `:1521`              | 110 |  30 |
| `get_k` / `get_v` gather views                                 | `src/llama-kv-cache.cpp:1428` / `:1450`              |  90 |  20 |
| `find_slot` sink-pin guard                                     | `src/llama-kv-cache.cpp:1084`                        |  15 |   5 |
| Slide-time row demotion (extends `build_graph_deferred_convert`) | `src/llama-kv-cache.cpp:341-343, :846-900`         |  70 |  15 |
| ISWA / hybrid plumbing (constructor params)                    | `src/llama-kv-cache-iswa.{cpp,h}`, `llama-memory-hybrid*` | 20 |  10 |
| Tests                                                          | `tests/test-kv-cache-kitty.cpp` (new)                | 250 |   0 |
| Docs                                                           | `docs/turboquant.md` KITTY section                   |  60 |   5 |
| **Total**                                                      |                                                      | **~890** | **~131** |

Roughly **1000 LOC net add**, ~130 LOC modified. Fits in one PR. Estimated implementation effort: **3 focused engineering days** for code + 2 days for benchmarking against the §6 gate.

---

## 9. Open Questions for Implementer (resolve before coding)

1. Should `kv_qbuffer_type` be allowed to differ per K/V (`--kv-qbuffer-type-k`, `--kv-qbuffer-type-v`)? Current spec uses one type for both; matches KITTY paper but loses VTQ asymmetry that we currently exploit.
2. For multi-stream/multi-seq deployments, should sink slots be **per-sequence** or **shared**? Per-sequence is correct (each seq has its own first 4 tokens) but multiplies pinned VRAM by `n_seq_max`. Spec assumes per-sequence; revisit if VRAM tight.
3. Does the FA path (`!v_trans`) need a separate review? KITTY paper assumes non-FA attention; FA kernels in this repo have known TQ-V dispatch limits (`fattn.cu`). Recommend gating KITTY off when FA is active for V at first cut, mirroring the existing TQ-V→f16 fallback at `llama-kv-cache.cpp:266-278`.

---

## 10. Out-of-Scope (Follow-ups)

* Adaptive qbuffer sizing based on observed attention-mass (KITTY paper §5.2)
* Per-head sink/qbuffer (currently per-layer uniform)
* Integration with Markov Document Store residual routing (Phase 19)
* SWA-aware sink pinning (see Risk 3)
