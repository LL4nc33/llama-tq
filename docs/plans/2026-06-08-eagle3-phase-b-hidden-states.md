# Eagle3 Phase B — Hidden-state-extraction infrastructure

**Status:** Design + architecture, code starts after review.
**Dry-run target:** Qwen3.5-9B-MTP-IQ4_XS (5.1 GB, fits single 12 GB GPU with headroom for training).
**Final target:** Qwen3.6-35B-A3B-IQ2_XXS (10 GB, single-GPU inference + dual-GPU training).

## What Eagle3 needs from the base model

The Eagle3 draft head takes **3 hidden-state-streams** from the frozen base:
- `h_low` — from a layer near the input (typically layer 2 or `n_layer / 8`)
- `h_mid` — from a middle layer (typically `n_layer / 2`)
- `h_high` — from a layer near the output (typically `n_layer - 2` or `n_layer * 7/8`)

These are concatenated, projected through a 1-layer FC (`3·n_embd → n_embd`),
and fed as input to the 1-layer Eagle3 decoder.

The paper doesn't fix the indices. Common choice for a model with N layers:
```
h_low  = layer N/4 - 1     (e.g. 9 for 36-layer)
h_mid  = layer N/2 - 1     (e.g. 17 for 36-layer)
h_high = layer N - 2       (e.g. 34 for 36-layer, second-to-last)
```

For **Qwen3.5-9B** (n_layer = 36): layers 8, 17, 34.
For **Qwen3.6-35B-A3B-MTP** (n_layer = 41, last is MTP): use 9, 20, 38 (one before MTP).

## Current state in llama-tq

Existing single-layer MTP infrastructure (Phase 2G+) extracts ONE hidden state row:
- `embd_nextn` buffer in `llama_context` (declared `src/llama-context.h:277`)
- `t_h_nextn` tensor named `h_pre_norm` (see `src/models/qwen35.cpp` graph builder)
- Async backend copy in `src/llama-context.cpp:1508` and `1964`
- Public API: `llama_get_embeddings_nextn_ith(ctx, i)` in `src/llama-ext.h:106`

That's the seed pattern. Eagle3 needs the **same mechanism replicated three times**
with three different layer indices.

## Architecture plan

### 1. hparams extension

Add to `src/llama-hparams.h`:
```c
// Eagle3 hidden-state-extraction layer indices.
// 0 means feature is disabled. Set by Eagle3-arch loader.
uint32_t eagle3_layer_low  = 0;
uint32_t eagle3_layer_mid  = 0;
uint32_t eagle3_layer_high = 0;
```

Add `LLM_KV_EAGLE3_LAYER_{LOW,MID,HIGH}` to `src/llama-arch.{h,cpp}` so the
GGUF can carry them.

### 2. Three new buffers in llama_context

In `src/llama-context.h` next to `embd_nextn`:
```c
buffer_view<float> embd_eagle3_low  = {nullptr, 0};
buffer_view<float> embd_eagle3_mid  = {nullptr, 0};
buffer_view<float> embd_eagle3_high = {nullptr, 0};
```

Sizing: same as `embd_nextn` (n_embd × max_outputs × sizeof(float)).
For Qwen3.5-9B n_embd=5120, 2048 ctx: 3 × 5120 × 2048 × 4 = **120 MB** total.
Acceptable on single 12 GB GPU.

### 3. Graph builder taps

The base-model graph builder needs to mark the three target layers and copy
their pre-norm hidden states to dedicated output tensors.

For Qwen3.5 in `src/models/qwen35.cpp`, after each `build_attn` + FFN block
in the main decoder loop:

```cpp
if (il == (int)hparams.eagle3_layer_low) {
    ggml_tensor * h_l_copy = ggml_dup(ctx0, h_residual);
    cb(h_l_copy, "h_eagle3_low", il);
    ggml_build_forward_expand(gf, h_l_copy);
}
// repeat for _mid and _high
```

Note: `ggml_dup` (not view) so the tensor stays valid after later layers
overwrite the residual stream.

### 4. Async copy in context.cpp

Mirror the existing `embd_nextn` copy pattern (lines 1508, 1964 in
`src/llama-context.cpp`):

```cpp
if (embd_eagle3_low.data && t_h_eagle3_low) {
    ggml_backend_tensor_get_async(backend, t_h_eagle3_low, embd_eagle3_low.data, ...);
}
// repeat for mid, high
```

### 5. Public API

In `src/llama-ext.h`:
```c
float * llama_get_embeddings_eagle3_low (llama_context * ctx, int32_t i);
float * llama_get_embeddings_eagle3_mid (llama_context * ctx, int32_t i);
float * llama_get_embeddings_eagle3_high(llama_context * ctx, int32_t i);
```

Implementation in `llama-context.cpp` analogous to `llama_get_embeddings_nextn_ith`.

### 6. cparams flag

Enabled per-context via `cparams.embeddings_eagle3` (similar to existing
`cparams.embeddings_nextn`). Default off — only the Eagle3 draft-impl
turns it on.

## Implementation order (incremental, testable)

**Step B.1** — hparams + KV + cparams plumbing (no graph changes).
Build verifies, no runtime effect. ~30 LOC.

**Step B.2** — three buffers in llama_context + reserve in opt-allocate path.
Build verifies, server starts, no extra memory used (size stays 0 until flag set).
~50 LOC.

**Step B.3** — graph builder taps in qwen35.cpp + qwen35moe.cpp.
Build verifies, output unchanged (taps are no-ops if flag off). ~80 LOC.

**Step B.4** — async copies + public API.
Build verifies. Manual test: write a small CLI that loads model, generates 10
tokens with `embeddings_eagle3=true`, prints norm of each h-stream. ~100 LOC.

**Step B.5** — GGUF KV writer in `convert_hf_to_gguf.py` so the 3 indices
travel with the model. ~20 LOC.

## Dry-run test plan (Qwen3.5-9B)

1. Convert Qwen3.5-9B base from HF → GGUF with `eagle3_layer_low=8`,
   `_mid=17`, `_high=34` baked in
2. Load via patched llama-tq, enable `--embd-eagle3`, generate 100 tokens
3. Verify three buffers have non-zero norms
4. Verify byte-identical output to a baseline run (taps must not change forward path)
5. Measure overhead: extra memory + extra ms per token from the dup+copy

If all 5 pass: Phase B done. Move to Phase C (Eagle3 draft-head graph).
If overhead > 5%: revisit — maybe use views instead of dups, accept the
"stays valid" constraint differently.

## Risks

- **ggml_dup overhead**: each tap is a memcpy of n_embd floats per token.
  For batch=1 TG: 5120 × 4 = 20 KB × 3 taps = 60 KB extra per token.
  At 80 t/s that's 4.8 MB/s sustained — negligible.
- **Tensor lifetime**: must verify ggml_build_forward_expand keeps the dup
  tensor alive across the rest of the graph build. If not, allocate from
  a longer-lived ctx.
- **Layer indices off-by-one**: paper notation 0-indexed vs 1-indexed
  varies. Choose 0-indexed and document.

## Next steps after Phase B

- **Phase C**: Eagle3-head graph (1 decoder layer + FC fusion), GGUF tensor
  types, loader. Equivalent to extending `graph_mtp` in qwen35.cpp.
- **Phase D**: Training script in distillery (frozen base, trainable head,
  training-time-test loss). 600-1000 LOC.
- **Phase E**: `state_draft_eagle3` impl in `common/speculative.cpp`,
  clone of `state_draft_mtp` with 3-stream carryover. 400-700 LOC.

Total scope: 6-8 weeks engineering, $0 cloud budget. Single RTX 2060 for
training the 9B head (frozen base 5 GB IQ4 + trainable head 200 MB + grad
buffers fits 12 GB).
