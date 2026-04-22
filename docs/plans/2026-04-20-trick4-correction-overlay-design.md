# Trick 4 — Correction Overlay Buffer for VTQ V-cache

**Status:** Proposed
**Date:** 2026-04-20
**Branch target:** `trick4-correction-overlay` (from `phase2`)
**Scope:** VTQ{2,3,4}_2 only (Trellis-coded V-cache); read-only design, no K-side work.

---

## 1. Motivation

Trick 2 PR1 error profiling confirmed VTQ quant-error is **heavy-tailed**
(per-block kurtosis 90–7421). A tiny fraction of samples accounts for most of
the PPL regression. VTQ2_2 (2.06 bpw) currently costs ~+8% PPL vs f16; VTQ3_2
costs ~+1.9%. A per-layer sidecar storing only the top-N highest-error positions
at f16 can cheaply recover most of that loss.

Memory budget: at `N=16` and `sizeof(entry)=6 B` (fp16 value + uint32 position),
~96 B per layer-row, or ~24 KB across all V-layers for Qwen3.5-35B-A3B (48
layers) — **negligible vs the ~11 GiB V-cache**.

---

## 2. Target

| Config | PPL regression | Overlay storage | Target PPL |
|--------|----------------|------------------|------------|
| VTQ2_2 baseline | +8.0% | 0 | — |
| VTQ2_2 + N=8 | ? | +0.03% | ≤+4.0% |
| VTQ2_2 + N=16 | ? | +0.06% | ≤+2.5% |
| VTQ2_2 + N=32 | ? | +0.13% | ≤+1.5% |
| VTQ3_2 baseline | +1.9% | 0 | — |
| VTQ3_2 + N=16 | ? | +0.06% | ≤+0.5% |
| Decode overhead (wall) | — | — | ≤+5% tok/s |

**Auto-enable default:** overlay is ON for VTQ2_2 with `N=16`, OFF otherwise
(diminishing returns on VTQ3_2/VTQ4_2 without measurement).

---

## 3. Storage Layout

### 3.1 Granularity: per-block, not per-layer

Per-layer top-N would require a 256k-entry priority queue over the entire
prefill V-tensor (`n_embd_v_gqa × kv_size`). Per-block top-N is:
- **Local** — each 256-sample trellis block picks its own top-N.
- **Cache-friendly** — overlay for block `b` co-locates with the block.
- **Hash-free** — position index is a 0..255 byte, stored in a small sorted
  or bitmap array alongside `qs[]`.

Chosen: **per-trellis-block top-N** with `N_PER_BLOCK = 2` (→ amortised `N=2`
per 256 samples ≈ `N≈32` per 4k-sample row). Adjust via CLI.

### 3.2 Block struct extension

Decision: **do NOT modify `block_vtq{2,3,4}_2`** — they have `static_assert`-locked
sizes (68/100/132 B) used across GGUF, CPU, CUDA, disk. Instead, add a
**parallel overlay tensor** per layer.

```
layer struct (llama-kv-cache.cpp:360):
  { il, k, v, k_staging, v_staging, k_stream, v_stream,
    k_staging_stream, v_staging_stream,
    v_overlay,            // NEW: ggml_tensor *
    v_overlay_stream }    // NEW: std::vector<ggml_tensor *>
```

Overlay layout (single flat tensor, `GGML_TYPE_I16` viewed as packed entries):

```
v_overlay: [n_overlay_bytes_per_row , kv_size , n_stream]

Per-row (one V vector, length n_embd_v_gqa):
  n_blocks = n_embd_v_gqa / 256
  n_entries = n_blocks * N_PER_BLOCK
  bytes = n_entries * 4              // 1B pos + 1B flags + 2B fp16 value
```

Entry (packed, 4 B):
```
struct vtq_overlay_entry {
    uint8_t   pos;    // 0..255, position within block
    uint8_t   flags;  // bit0=valid; bits1-7 reserved
    ggml_half value;  // true fp16 value (pre-quant, post-D*H*D rotation)
};
```

Storage overhead for Qwen3.5-35B-A3B (48 V-layers, `n_embd_v_gqa=4096`, `kv_size=200k`, `N_PER_BLOCK=2`):
```
= 48 * 4096/256 * 2 * 4 B * 200000 * 1 stream
= 48 * 16 * 2 * 4 * 200k
= 48 * 25.6 MB
= 1.23 GB                         ← NOT negligible at 200k ctx
```

**Mitigation:** overlay size scales with `kv_size`. Three options:
- **(A)** Sparse overlay: only allocate for populated rows (hard with ggml tensors).
- **(B)** Lower `N_PER_BLOCK=1` → 615 MB. Still steep.
- **(C)** Store overlay only for prefill phase (bulk Viterbi), discard on decode
  transition. Contradicts the point.
- **(D)** `N_PER_BLOCK=1` + 2-byte packed entry (pos_lo + val_fp16, flag=fp16 sign):
  307 MB. Acceptable.

**Chosen:** **(D)** — 2 B/entry × 1 entry/block × 16 blocks/row × 200k rows × 48 layers = **307 MB**.
Still a real cost. Flag in CLI; disabled by default. Reconsidered relative to
the 11 GiB V-cache: ~2.7% overhead for potentially 6–7% PPL recovery on VTQ2_2.

### 3.3 File-line hook for allocation

Add at `src/llama-kv-cache.cpp:348` (after `v_staging` block, same gating):

```cpp
// correction overlay: per-block top-N sidecar for VTQ_2 layers
ggml_tensor * v_overlay = nullptr;
std::vector<ggml_tensor *> v_overlay_stream;
if (use_correction_overlay && has_v && layer_uses_vtq2) {
    const int64_t bytes_per_row = (n_embd_v_gqa / 256) * N_PER_BLOCK * 2; // 2B packed
    v_overlay = ggml_new_tensor_3d(ctx, GGML_TYPE_I16,
                                    bytes_per_row / 2, kv_size, n_stream);
    ggml_format_name(v_overlay, "cache_v_overlay_l%d", il);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_overlay_stream.push_back(
            ggml_view_2d(ctx, v_overlay, bytes_per_row/2, kv_size,
                         v_overlay->nb[1], s*v_overlay->nb[2]));
    }
}
```

Update `layers.push_back({...})` at :360 to include the new members.

---

## 4. Encode Path

### 4.1 Where

Bulk Viterbi happens at `TQ_DEFERRED_READY → bulk convert` transition in
`build_graph_deferred_convert()` — currently at **`src/llama-kv-cache.cpp:2181`**
(`if (layer.v_staging) { ... ggml_set_rows(ctx, dst, src_f32, inp->idxs) }`).

The `ggml_set_rows` operation dispatches to the VTQ2_2 encoder (CUDA Viterbi
in `ggml/src/ggml-cuda/trellis-encode.cuh`). The encoder has the *ground-truth*
f32 values and produces both the packed `qs[]` and (now) the overlay.

### 4.2 Encoder kernel extension

In `trellis_encode_group_kernel<K>` (CUDA):

1. Do normal Viterbi, produce `qs[]`, `start_state`, `d`.
2. **Side-channel pass:** dequantize inline (`trellis_decode_block<K>`) into
   shared-mem `decoded[256]`.
3. Compute per-sample residual `err[i] = fabsf(src[i] - decoded[i])`.
4. `argmax_N(err)` — for `N_PER_BLOCK=1`, a single warp-reduce
   (`__shfl_down_sync` max-with-index). For `N_PER_BLOCK ≤ 4`, a small
   tournament in shared memory. No global sort needed.
5. Write `pos` + `fp16(src[argmax])` to `v_overlay` at the block's entry slot.

Cost: one extra decode pass + one warp reduction per 256-sample block. In the
Phase-2b CUDA Viterbi encoder, group-encode is ~15µs/block; overlay adds
<0.5µs. Prefill is a one-off, budget is fine.

**Threshold option:** if `|err[argmax]| < threshold * d` (e.g. 0.25), set
`flags.valid=0` so decode skips it. Avoids wasting a slot on already-accurate
blocks.

### 4.3 File-line hook points

- **CUDA encoder:** `ggml/src/ggml-cuda/trellis-encode.cuh` — extend
  `trellis_encode_group_kernel` to take optional `overlay_out` pointer, emit
  top-1 error per block. Dispatch: adjacent to `ggml_set_rows` path.
- **Graph build:** `src/llama-kv-cache.cpp:2181` — attach `v_overlay` as a
  second `dst` to the encoder op (requires a new ggml op variant
  `GGML_OP_SET_ROWS_VTQ_OVERLAY` or carry overlay as op param).
  **Simpler:** add a separate `ggml_trellis_extract_overlay(src_f32, v_ref, v_overlay)`
  op that runs after `set_rows`, re-dequantizes and compares. Slower but zero
  change to set_rows semantics. Recommend path A long-term, path B for MVP.

---

## 5. Decode Path (Dequant Integration)

### 5.1 CUDA hook

File: `ggml/src/ggml-cuda/trellis.cuh:152` (`k_dequantize_trellis_nc`).

```cpp
// AFTER trellis_decode_block writes `decoded[]`:
if (overlay_ptr) {
    const int block_id = ib;                                  // linear block idx
    const vtq_overlay_entry_packed * e = overlay_ptr + block_id * N_PER_BLOCK;
    #pragma unroll
    for (int k = 0; k < N_PER_BLOCK; ++k) {
        if (tid == 0 && (e[k].pos_and_flags & 0x80)) {         // valid bit
            decoded[e[k].pos_and_flags & 0x7F] =
                __half2float(*(__half*)&e[k].value);
        }
    }
    __syncthreads();
}
```

**Cost analysis:**
- Current kernel: 128 threads × 1 block, `tid==0` decodes serial 256 samples.
- Overlay check: 1–4 global-memory reads from `overlay_ptr`, one scalar
  assignment. L1-cached after first block. ~20 cycles additional.
- Block decode currently ~3 µs; overlay adds ~3 ns. **<0.1% overhead** on the
  decode hot path — well under the 100 ns budget.

### 5.2 CPU hook

File: `ggml/src/ggml-quants.c:6240` (`dequantize_row_vtq2_2`).

```cpp
void dequantize_row_vtq2_2(const block_vtq2_2 * x, float * y, int64_t k,
                           const vtq_overlay_entry * overlay /* NEW, optional */) {
    const int nb = k / QK_VTQ_TRELLIS;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        ggml_trellis_decode_group(x[i].start_state, 2, d, x[i].qs,
                                  y + i * QK_VTQ_TRELLIS);
        if (overlay) {
            for (int j = 0; j < N_PER_BLOCK; ++j) {
                const vtq_overlay_entry * e = &overlay[i*N_PER_BLOCK + j];
                if (e->flags & 1) {
                    y[i*QK_VTQ_TRELLIS + e->pos] = GGML_FP16_TO_FP32(e->value);
                }
            }
        }
    }
}
```

ABI: add overloaded signature to avoid breaking callers. New call site passes
overlay ptr, old `dequantize_row_vtq*` wrappers pass `nullptr`.

### 5.3 Lookup structure choice

- **Hash table** — rejected. Overhead (~30–50 ns/lookup), collisions, not
  GPU-friendly in __shared__.
- **Binary search** — rejected. 2 comparisons/block is noise, but branching
  hurts warp coherence.
- **Direct-indexed bitmap** — rejected. 256 bits/block = 32 B/block *just
  for the flag*; wasteful.
- **Small sorted array, linear scan** — ✅ chosen. `N_PER_BLOCK=1 or 2`
  → 1–2 iterations, fully unrolled, branch-predictable.

### 5.4 FA-vec path

Per-element decoder variant at `ggml/src/ggml-cuda/trellis.cuh:~215` — needs
the same overlay check. Since it's called per-sample, overlay read has to be
structured to allow random position lookup. With `N_PER_BLOCK=1`, a single
compare `if (pos == overlay.pos) return overlay.value;` is 2 instructions.

---

## 6. CLI / API

Single flag, opt-in, VTQ2_2-only default:

```
--tq-correction-overlay [N]
    Enable per-block correction overlay for VTQ_2 V-cache.
    N = entries per 256-sample block (default: 1 if flag present, 0 if absent).
    Storage cost: ~0.5% of V-cache size per entry.
    Effective only on VTQ2_2 layers (VTQ3_2/VTQ4_2 see <0.5% PPL gain).
```

In `common/common.h` (near line 555 alongside `tq_deferred_v`):

```cpp
uint32_t tq_correction_overlay = 0;   // 0 = off, N = entries per block
```

In `include/llama.h` (near line 364):
```cpp
uint32_t tq_correction_overlay;
```

In `common/arg.cpp` (near line 2077 alongside `--tq-deferred-v`).

**GGUF:** overlay is **runtime-only** for MVP. Persisting into GGUF makes the
file format incompatible and adds a ~300 MB blob; not worth it. Re-extract on
reload (one-shot, part of bulk Viterbi).

**Interaction with `--tq-protect-sinks`:** boundary layers are Q8_0 (no trellis),
no overlay allocated. Cleanly handled by the `layer_uses_vtq2` guard at
allocation time.

---

## 7. Validation Plan

### 7.1 Unit tests

- `tests/test-vtq-overlay.cpp`: round-trip a 256-sample block with synthetic
  heavy-tailed noise; verify top-1 argmax matches brute-force; verify decode
  recovers the exact fp16 at `pos`.

### 7.2 PPL matrix (wikitext-2, 512 ctx, Qwen3.5-35B-A3B, TP=1)

| Run | K-cache | V-cache | N_PER_BLOCK | Expected PPL Δ | Decode tok/s Δ |
|-----|---------|---------|-------------|----------------|----------------|
| A | f16 | f16 | — | baseline | baseline |
| B | TQ2_1 | VTQ2_2 | 0 | +8.0% | +0% |
| C | TQ2_1 | VTQ2_2 | 1 | +3.0% (hope) | −1% |
| D | TQ2_1 | VTQ2_2 | 2 | +1.8% (hope) | −2% |
| E | TQ2_1 | VTQ3_2 | 0 | +1.9% | +0% |
| F | TQ2_1 | VTQ3_2 | 1 | +1.0% | −1% |

Acceptance: **C** must hit PPL Δ ≤ +4.0% with decode regression ≤ 2%.
Otherwise overlay is not worth the 300 MB.

### 7.3 Error-distribution sanity

Log `histogram(|err[argmax]|/d)` over first 1000 blocks — expect long-tail,
mean ~2σ of residual. Confirms the kurtosis hypothesis translates to useful
corrections.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Encoder overhead at bulk-Viterbi transition | Low | Overhead budget <3%; already measured Phase-2b encoder at ~15µs/block. |
| 300 MB overlay too costly at 200k ctx | Med | Default OFF; flag opt-in; optionally `N_PER_BLOCK=1` only |
| Decode-path branch breaks FA-vec coherence | Med | Benchmark run E/F first; if hit, gate to non-FA dequant path only |
| Overlay desync after shift/copy (kv layer ops) | High | **Must** copy overlay alongside `v_stream` at `:984` (ggml_backend_tensor_copy for v_staging_stream). Add identical call for v_overlay_stream. |
| Overlay with `--tq-deferred-v` OFF (incremental encode) | High | Incremental encode does 1 token/step with no bulk pass — overlay extraction would need per-step hook. **MVP:** require `--tq-deferred-v` and error otherwise. |
| GGUF save/load of cache | Low | State save/restore paths at `:2272`/`:2355`/`:2381`; must serialize overlay tensor. Cache restore needs it. |
| Interaction with Sparse V Dequant (v4) | Med | Sparse V skips zero-value positions at decode; overlay may force a non-zero read. Verify sparsity check happens **after** overlay apply. |

---

## 9. Open Questions (defer to implementation)

1. **Path A vs Path B encoder integration** — new ggml op, or post-set_rows
   extraction op? Path B is MVP-safe but reads src_f32 twice. Decision after
   prototyping.
2. **Adaptive N per layer** — early layers may need more corrections than late
   layers. Profile before complicating the schema.
3. **Hybrid with Trick 3 (block-scale refinement)** — if Trick 3 already
   rescales `d` per block using the same error stats, Trick 4 marginal gain
   shrinks. Run C standalone *and* C+Trick3 to measure orthogonality.

---

## 10. File Summary (hook points)

| File | Line | Change |
|------|------|--------|
| `include/llama.h` | ~364 | Add `tq_correction_overlay` to `llama_context_params` |
| `common/common.h` | ~555 | Add field |
| `common/common.cpp` | ~1485 | Wire param |
| `common/arg.cpp` | ~2077 | Add `--tq-correction-overlay` |
| `src/llama-kv-cache.h` | layer struct | Add `v_overlay`, `v_overlay_stream` |
| `src/llama-kv-cache.cpp` | 348 | Allocate `v_overlay` |
| `src/llama-kv-cache.cpp` | 360 | Extend `layers.push_back` |
| `src/llama-kv-cache.cpp` | 984 | Copy overlay on stream copy |
| `src/llama-kv-cache.cpp` | 2181 | Emit overlay alongside VTQ encode |
| `src/llama-kv-cache.cpp` | 2272/2355/2381 | Save/restore overlay |
| `ggml/src/ggml-common.h` | ~406 | Define `vtq_overlay_entry` (no block struct change) |
| `ggml/src/ggml-quants.h` | 84–86 | Add overlay-aware overload or new entry point |
| `ggml/src/ggml-quants.c` | 6240 | CPU dequant: overlay apply |
| `ggml/src/ggml-cuda/trellis.cuh` | 152 | CUDA dequant: overlay apply in `k_dequantize_trellis_nc` |
| `ggml/src/ggml-cuda/trellis.cuh` | ~215 | FA-vec per-element overlay check |
| `ggml/src/ggml-cuda/trellis-encode.cuh` | — | Encoder extension: top-N argmax + write |
| `tests/test-vtq-overlay.cpp` | new | Unit test |

---

## 11. Corrections to the Ticket

- `QK_VTQ_TRELLIS = 256`, not 512 (confirmed
  `ggml/src/ggml-common.h:385` and `ggml/src/ggml-cuda/trellis.cuh:23`).
  All budgets above recalculated on 256-sample blocks.
- "~512 B per layer × 48 layers ≈ 24 KB" in the concept undershoots by ~4
  orders of magnitude at realistic context sizes; real estimate is 307 MB at
  200k ctx with `N_PER_BLOCK=1`. Still <3% of V-cache but not free.
