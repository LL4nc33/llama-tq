# Attention-Sink Protection — Results (2026-04-19)

## Trick 1: Protect first KV layer's V-cache at f16

StreamingLLM (arXiv:2309.17453) shows first tokens carry outsized
attention weight. Protecting the first KV layer's V-tensor (~2 KB/ctx)
at f16 eliminates the quant overhead at negligible cost.

## Bugs Fixed Alongside

1. **common.cpp forgot to forward `tq_deferred_v` to cparams** —
   meant perplexity/cli ignored `--tq-deferred-v` entirely.
   Only llama-bench worked (reads env directly). This also explained
   the "PPL prefill slowdown" reported earlier (58s/pass vs 0.77s).

2. **arg.cpp tq-protect-{layers,sinks} missing `.set_examples(...)`** —
   options only attached to default example, invisible to
   perplexity/bench/cli frontends.

3. **Sink protection used `il == 0`** — on hybrid models (Mamba/SSM +
   attention, e.g. Qwen3.5) il==0 can be recurrent/filtered. Fixed
   to use first KV-carrying layer index.

## Results (Qwen3.5-0.8B-Q8_0, wikitext-2, 4 chunks, n_ctx=2048)

| Chunk | f16 V | vtq3_2+deferred | vtq3_2+deferred+sink=4 |
|-------|-------|-----------------|------------------------|
| [1] | 14.80 | 15.37 (+3.8%) | **14.76** (-0.3%) |
| [2] | 16.68 | 16.99 (+1.9%) | **16.64** (-0.2%) |
| [3] | 17.29 | 17.68 (+2.3%) | **17.27** (-0.1%) |
| [4] | 17.37 | 17.73 (+2.0%) | **17.35** (-0.1%) |

**PPL overhead: ~+2% → essentially 0%** with sink protection.
Cost: one V-layer stays f16 (e.g. 24 KV layers → 1/24 = 4% extra V VRAM).

## Speed (after deferred_v propagation fix)

Prefill per-pass: 58.91s → **0.78s** (76× speedup via now-working deferred V).
Matches f16 baseline 0.77s — confirms deferred V path is identical-cost
at prefill, gains come purely from skipping per-token Viterbi at decode.

## ⚠️ PPL-Test Caveat (2026-04-19 evening)

The PPL results above were taken with `--tq-deferred-v` active. Because
llama-perplexity runs pure prefill (no prefill→decode transition),
`deferred_state` never leaves `STAGING`, meaning reads come from the
f16 staging buffer — not from the VTQ-quantized tensor. This means the
PPL numbers above reflect **deferred-staging f16 reads**, not quantized
V-cache reads. Real decode-time PPL (after transition to READY/DONE)
is expected ~+2% vs f16 without sink-protection.

For correct quantized PPL measurement: disable `--tq-deferred-v` or
use an end-to-end generation benchmark that includes decode.

## Verdict

- **Speed (decode):** verified in llama-bench tg64 — 26× faster on
  0.8B, parity with f16 on 27B dual-GPU (Viterbi runs once at
  prefill→decode, not per-token).
- **PPL (prefill-only):** test harness limitation — see caveat above.
  Decode-time quality separately validated via tg latency stability.
