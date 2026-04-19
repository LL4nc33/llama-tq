# CUDA Stability Validation — VTQ_2 Production Readiness

Status: **in progress** (2026-04-20)

## Context

After fixing 3 flag-propagation bugs yesterday (common.cpp missed
tq_deferred_v, arg.cpp missed set_examples for protect-{sinks,layers},
kv-cache counter logic ignored filter() on hybrid models), we need
to confirm CUDA path is production-ready before starting the Trick-17
research series.

## Stage 1 — Extended tg512 speed sweep (Qwen3.5-0.8B-Q8_0)

12 runs × `llama-bench -p 512 -n 512 -r 1` on single RTX 2060.

| type | baseline | +deferred | +sink | +both |
|------|----------|-----------|-------|-------|
| vtq2_2 | (timeout ~7 t/s) | — | 12.67 | **196.20** |
| vtq3_2 | 7.29 | 195.62 | 7.28 | **196.22** |
| vtq4_2 | (timeout) | 196.26 | (timeout) | **196.05** |

Key: deferred alone gives full 27× speedup; sink is orthogonal
(quality-only, no speed effect). Combined both is production default.

## Stage 2 — True quantized PPL (wikitext-2, 10 chunks)

`llama-perplexity` w/o `--tq-deferred-v` (deferred never exits STAGING
in pure prefill, as discovered yesterday). Sink protection measured
by contrast to no-sink baseline.

Baseline f16: **14.39**

| type | no sink | sink=4 | Δ sink | overhead vs f16 (sink=4) |
|------|---------|--------|--------|--------------------------|
| vtq2_2 (2.06 bpw) | 15.69 | **15.54** | -1.0pp | **+8.0%** |
| vtq3_2 (3.06 bpw) | 14.74 | **14.66** | -0.5pp | **+1.9%** |
| vtq4_2 (4.06 bpw) | pending | pending | — | — |

Sink reliably trims overhead; effect stronger at lower bpw where
sinks dominate quality. vtq3_2 at +1.9% is production-acceptable.

## Stage 3 — 35B 2-GPU stability (pending)

Planned: Qwen3.5-27B-IQ2_XXS on 2× RTX 2060, vtq3_2 + deferred + sink,
long-run tg1024, watch for OOM/crashes.

## Bugs Fixed Alongside Validation

1. **common.cpp** forgot to forward `tq_deferred_v` → cparams
2. **arg.cpp** `tq-protect-{layers,sinks}` missed `set_examples(...)` →
   invisible to perplexity/bench/cli frontends
3. **kv-cache.cpp** boundary + sink counters incremented on recurrent
   layers because they check `has_kv()` without `filter()`; on hybrid
   models (Qwen3.5 = Mamba+attention) this caused counter drift so
   sink never fired on first attention-KV layer

## Verdict (tentative, pending vtq4_2 + 35B)

VTQ3_2 + `--tq-deferred-v --tq-protect-sinks 4` is the production
recipe: +1.9% PPL, parity tg speed, 3.06 bpw V-cache (62% VRAM save
vs f16). VTQ2_2 viable for VRAM-constrained deployments at +8.0% PPL.
