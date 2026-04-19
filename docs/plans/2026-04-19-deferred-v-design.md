# Deferred V Quantization Design (Phase-3)

## Problem Statement

Measured 2026-04-19: VTQ3_2 tg64 on Qwen3.5-0.8B runs at **7.4 tok/s**
versus f16 baseline **199 tok/s** — 27× slowdown. Root cause isolated
via CUDA events instrumentation on the `set_rows` dispatch:

```
[vtq3_2 enc] calls=192 total=4162ms avg=21.7ms ne11=1
```

The Viterbi encoder kernel takes **21.7ms per call** at `ne11=1` (single
new V token per layer per decode step). With 6 KV layers × 32 tokens =
192 calls, encoder alone consumes 4.2s of 4.3s wall-clock.

Viterbi is necessary for quality: greedy fallback yields +60% PPL
(15.76 → 26.3 on wikitext-2). Beam-search B=16 on single thread with
parent/edge arrays spilled to local memory was still 2× slower than
Viterbi (9.88 vs 7.4 tok/s).

## Solution: Deferred V Quantization

Mirror the existing deferred K quantization (`tq_deferred_k`) for V:
stage V-cache writes in a f16 buffer during prefill, then bulk-convert
to VTQ at the prefill → decode transition. During decode, continue
writing in f16 to avoid per-token Viterbi overhead, trigger periodic
bulk conversions when enough tokens accumulate.

### Reference Implementation (Deferred K)

The KTQ deferred path already exists end-to-end. Files and key sites:

| File | Purpose |
|------|---------|
| `include/llama.h:363` | public API bool |
| `common/common.h:554` | common_params bool |
| `common/arg.cpp:2068` | `--tq-deferred-k` CLI flag |
| `src/llama-kv-cache.h:117,238` | ctor param + `k_staging` tensor |
| `src/llama-kv-cache.cpp:126,309,329` | allocate staging + layer struct |
| `src/llama-kv-cache.cpp:784,830,979` | state machine `TQ_DEFERRED_*` |
| `src/llama-kv-cache.cpp:1345,1401` | `get_k` returns staging when STAGING |
| `src/llama-kv-cache.cpp:2078` | `build_graph_deferred_convert` |
| `src/llama-kv-cache.cpp:2029` | stream-copy reads staging |
| `src/llama-context.cpp:283,2916` | wire cparams |

### Deferred V Adapter (Phase-3 scope)

1. **llama.h**: add `bool tq_deferred_v;`
2. **common.h**: add `bool tq_deferred_v = false;`
3. **arg.cpp**: add `--tq-deferred-v` CLI flag (copy KTQ block, s/k/v/g)
4. **llama-kv-cache.h**:
   - ctor signature: add `bool tq_deferred_v`
   - `kv_layer`: add `ggml_tensor * v_staging`, `v_staging_stream`
5. **llama-kv-cache.cpp** — this is the big one:
   - `is_tq_type_v = VTQ{2,3,4}_2`
   - `use_deferred_v` flag + tensor budget `n_tensors_per_layer`
   - allocate `v_staging` (f16, same shape as v) per layer
   - route `get_v`/`cpy_v` to staging during STAGING
   - state machine: STAGING when `n_tokens > 1`, READY when `n_tokens == 1`
   - `build_graph_deferred_convert`: add V set_rows f32→VTQ stage
   - `state_write`/`state_read`: persist staging buffers
6. **llama-context.cpp**: wire `cparams.tq_deferred_v`
7. **llama-model.cpp**: 4 call sites, add the param

### Testing

Primary:
```bash
# Without deferred V (current — slow)
./build-cuda/bin/llama-bench -m qwen3.5-0.8b-q8_0.gguf \
    -ctv vtq3_2 -fa 1 -ngl 99 -p 0 -n 64
# Expected: ~7 tok/s

# With deferred V
./build-cuda/bin/llama-bench -m qwen3.5-0.8b-q8_0.gguf \
    -ctv vtq3_2 -fa 1 -ngl 99 -p 0 -n 64 \
    --tq-deferred-v
# Expected: ~180 tok/s (close to f16 = 199)
```

Quality check:
```bash
# PPL with deferred V must match non-deferred (Viterbi bulk-encode is
# identical, just batched differently).
./build-cuda/bin/llama-perplexity -m ... -ctv vtq3_2 --tq-deferred-v
# Expected: within ±0.1% of baseline 15.76
```

### Risk / Open Questions

- **Memory overhead**: f16 staging = 2× bpw temporarily during prefill.
  For 200k ctx at 3bpw this adds ~5GB until convert. Acceptable on
  24GB GPUs, tight on 12GB. Gate to TQ V-types like K is.
- **Streaming writes during decode**: current deferred K converts once
  at prefill-end. For continuous decode we'd need periodic re-deferral,
  which the KTQ path doesn't do — K stays converted after first batch.
  For V in autoregressive gen, every new token is ne11=1: need a
  "staging → convert at chunk boundary" policy.
- **State serialization**: save/load must handle partial STAGING state.

## Status

- 2026-04-19: Root cause identified + measured. Greedy/Beam fast paths
  added as opt-in (`GGML_VTQ_FAST_ENC=1`) for A/B testing.
- Phase-3 (deferred V): ~400-500 LOC, proposed next session.
- Phase-1 PPL numbers (`run22_08b_full_sweep.csv`) are unaffected — PPL
  test uses bulk Viterbi even without deferred flag.

## Measurements Archive (reference)

| config | tg64 tok/s | encoder ms/call | PPL (5-chunk) |
|--------|-----------|-----------------|----------------|
| f16 | 199 | — | 16.39 |
| vtq2_1 | 195 | — | — |
| vtq3_2 Viterbi | 7.4 | 21.7 | 15.95 ✓ |
| vtq3_2 greedy | 170 | 0.08 | 26.3 ✗ |
| vtq3_2 beam-B16 | 9.9 | ~15 | untested |

Standalone encoder bench: `tests/trellis-phase1/bench_encoder_gpu.cu`.
Greedy encoder: 57 μs/call isolated = 264× faster than Viterbi in
kernel-only measurement, confirms encoder launch overhead is real
but much of the 21.7ms is actual DP work in full runtime.
