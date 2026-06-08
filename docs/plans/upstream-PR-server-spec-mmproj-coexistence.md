# server: enable speculative decoding for text-only requests on mmproj-loaded servers

## Problem

Currently when a server is started with `--mmproj`, three features are globally disabled at init:
- `--cache-reuse N` → warn + set to 0
- `--ctx-shift` → warn + set false
- Speculative decoding works at init but asserts mid-execution

This forces users to choose: vision OR (spec/cache-reuse/ctx-shift). One server can't serve both.

The blocks exist because vision requests have image-chunks that occupy multiple tokens, breaking the math in those paths. But text-only requests on the same server (which `server_tokens.has_mtmd=true` flags as "multimodal-capable" but actually contain no media chunks) could safely use all three features.

## Fix

Distinguish slot-capability (`has_mtmd`, set when mmproj is loaded) from per-request content (`has_media()`, new accessor that returns `!map_idx_to_media.empty()`).

Convert global blocks to per-request gates:

1. **Init paths**: stop auto-disabling `ctx_shift` + `cache_reuse` + `speculative.type` when mmproj is loaded
2. **`server_tokens::get_tokens()`, `set_token()`**: relax `GGML_ASSERT(!has_mtmd)` to `GGML_ASSERT(!has_media())`
3. **`can_speculate()`**: return `!!spec && !prompt.tokens.has_media()` (gate per-request)
4. **ctx_shift abort path**: replace `GGML_ABORT` with `send_error` + `slot.release()` for vision requests
5. **cache_reuse abort path**: replace `GGML_ABORT` with `continue` for vision requests

## Effect

- **Text-only requests** on mmproj-loaded servers get the full feature set (spec, cache-reuse, ctx-shift) — same behavior as text-only deploys
- **Vision requests** skip spec/cache-reuse/ctx-shift automatically per-request — fail gracefully with `send_error` instead of `GGML_ABORT` server crash
- **No regression** for non-mmproj deploys

## Test plan

Tested with `Qwen3.6-35B-A3B-IQ2_XXS` + mmproj loaded:
- Text-only request 1 (cold): 80.57 t/s — spec active, lossless
- Text-only request 2 (cache-reuse hit): 81.41 t/s
- Vision request: "What color is this image?" → "red" (correct, spec skipped)
- Text-only request after vision: 175.61 t/s (ngram-cache spec on repeat-prompt, 2.2x)

Server stable across mixed-mode sequences. No crashes. Lossless output verified byte-identical vs text-only-only deploys for the same prompts.

## Files

- `tools/server/server-common.h`: add `has_media()` accessor (7 lines)
- `tools/server/server-common.cpp`: relax 2 asserts (4 lines changed)
- `tools/server/server-context.cpp`: gate per-request + lift global blocks (24 lines)

Net diff: ~32 insertions, ~16 deletions.
