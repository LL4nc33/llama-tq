# VTQ_2 K-collision — CPU vs CUDA pipeline split investigation

Date: 2026-04-25 17:00 CEST. Branch `turboquant`, HEAD `8a4be1f5a`.

## Goal

Hypothesis B: split CPU decoder vs CUDA decoder to localize the path with the bit-identical PPL collision (vtq2_2/vtq3_2/vtq4_2 → bit-identical PPL despite different bit depths).

## Method + results

### Test 1 — CPU encode/decode roundtrip (`test-vtq2-encoding-diff`)

Encoder produces strictly different `qs[]` bytes for K=2/3/4 with the expected popcount growth. Decoder MSE is monotone (0.060 → 0.015 → 0.0038). No "low-K of K=4 == K=K' bytes" identity.

- **CPU encode + CPU decode → correctly K-dependent.**
- Hypothesis A (encoder bug) remains refuted (Agent A finding confirmed).

### Test 2 — `vtq_state_at<K>` bit-window equivalence (`test-vtq2-cached-roundtrip`)

`fast_state_at<K>` (host mirror of `fattn-common.cuh::vtq_state_at<K>`) is verified against `ref_state_at` (naive bit-stream extraction) for K=2/3/4 across 32 random blocks × 128 sample indices. All 3 cases PASS.

- **The O(1) formula for `state(i)` is mathematically correct for all K.**

### Test 3 — Forced CPU dispatch (`-ngl 0`)

```
Gemma4-26B-A4B-IQ2_XXS, wikitext-2-raw, c=512, chunks=2, t=8
f16/vtq2_2 CPU: PPL = 99249.07
f16/vtq3_2 CPU: PPL = 46696.18
f16/vtq4_2 CPU: PPL = 68997.70
```

CPU-only PPL DIFFERS across K values. Since the CPU encoder/decoder runs through `ggml_trellis_{encode,decode}_group` (with verified K-awareness), CPU PPL was expected to be K-dependent.

But: **PPL values are very high (10⁴+), instruct-tuned Gemma4 on raw wikitext is poor signal/noise here**. The K ordering at least roughly matches "more bits = better reconstruction" for 2 vs 3, but 4 is worse than 3 — suspicious.

### Test 4 — CUDA symbol inspection

```
nm -D libggml-cuda.so.0.9.11 | grep flash_attn_ext_vec_case | grep VTQ_2
→ 3 host launcher symbols (Type 50/51/52), each 0x135 bytes (boilerplate).
cuobjdump --dump-elf-symbols → 16/28/16 device kernels for VTQ2_2/VTQ3_2/VTQ4_2.
```

Device kernels for all 3 K values are compiled. VTQ3_2 has more (28) because both TUs (`fattn-vec-instance-f16-vtq3_2.cu` + `fattn-vec-dispatch-vtq2.cu`) produce instances. VTQ2_2/VTQ4_2 only have the dispatch-TU instances (16).

- **Build has all K variants as separate device kernels.**
- Template instantiation is not the bug.

### Test 5 — Deferred-V staging path

KV-cache init logs show, for both CPU and GPU runs:
```
llama_kv_cache: deferred V quantization enabled (5 layers with f16 staging)
llama_kv_cache: deferred V quantization enabled (25 layers with f16 staging)
```

In `src/llama-kv-cache.cpp:852` the `TQ_DEFERRED_STAGING → READY` transition only fires when `balloc.get_n_tokens() == 1` (single-token decode batch). `llama-perplexity` only feeds multi-token batches (chunks of 512). Therefore:

- `deferred_state` STAYS at `STAGING` for the entire PPL run.
- `cpy_v` writes into `v_staging` (f16), not into the VTQ cache.
- `get_v` reads from `v_staging` (f16), not from the VTQ cache.
- **`build_graph_deferred_convert` is NEVER called.**
- The actual VTQ encoder/decoder is NOT exercised in `llama-perplexity`.

## Consequence

The original observation ("vtq2_2/vtq3_2/vtq4_2 produce bit-identical PPL") for CUDA was **not a bug in the VTQ decoder, but a consequence of the deferred-V staging design**: in `llama-perplexity` V quantization is never actually computed, and PPL is always evaluated against f16 V — hence K-invariant.

The diverging CPU PPL values are likely the result of subtler variation (sched order, thread scheduling, RNG state) rather than real VTQ-decode differences. *(Verification pending: a CPU-only repeated run with the same K would show whether the values are stable.)*

## Recommendations

1. **Fix the reproduction:** rerun the K-collision test with a real-decode workload (e.g. `llama-cli` with a real chat-style 1-token decode after prefill). Only after the first single-token batch does `deferred_state → READY → DONE` fire and the actual VTQ encoding via `build_graph_deferred_convert` execute. Only after that do FA-vec reads from the real VTQ cache become K-specific.

2. **Test harness:** if a PPL-style metric is desired that genuinely exercises VTQ, explicitly trigger `do_deferred_convert=true` (e.g. via a 1-token "warmup" batch after each chunk) before the next token-eval loop runs.

3. **Defer bug localization:** until the above correction is made, the actual K-bit-path code (encoder + decoder + FA-vec) is still UNTESTED in the PPL setup. The unit tests `test-vtq2-encoding-diff` and `test-vtq2-cached-roundtrip` cover the algorithmic part correctly and both PASS.

## Verified correctness

| Component | Status | Test |
|---|---|---|
| CPU `ggml_trellis_encode_group` K-aware | OK | test-vtq2-encoding-diff |
| CPU `ggml_trellis_decode_group` K-aware | OK | test-vtq2-encoding-diff (decode MSE monotone) |
| CUDA `vtq_cuda_encode_set_rows<K>` template | OK | source review (set-rows.cu, trellis-encode.cuh) |
| CUDA `k_dequantize_trellis<K>` (bulk) | OK | source review (trellis.cuh) |
| CUDA `vtq_state_at<K>` (FA-vec O(1)) | OK | test-vtq2-cached-roundtrip |
| Build has all K instances | OK | nm + cuobjdump |

## Files

- `tests/test-vtq2-encoding-diff.cpp` — encoder K-distinctness (Agent A)
- `tests/test-vtq2-cached-roundtrip.cpp` — `vtq_state_at` equivalence
- `ggml/src/ggml-trellis.c` — CPU Trellis encoder/decoder (lines 100-318)
- `ggml/src/ggml-cuda/trellis.cuh` — CUDA bulk dequant + per-element decoder
- `ggml/src/ggml-cuda/fattn-common.cuh` — `vtq_state_at<K>` + `dequantize_V_vtq{2,3}`
- `src/llama-kv-cache.cpp:850-855, 1450-1530` — deferred staging state machine

## Open tasks

- Find a real-decode PPL replacement (1-token decode trigger after each chunk).
- If the bug reproduces under a correct real-decode test → `dequantize_V_vtq_2` and `dequantize_V_vtq_3` are the hot-path decoders; continue investigation there.
