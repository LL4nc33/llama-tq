# VTQ_2 Production Ready — CUDA Stability Validated (2026-04-20)

After a 2-day debugging session that peeled back three layers of
flag-propagation bugs, the VTQ_2 Trellis-coded V-cache is production-
ready on CUDA. Here's what shipped.

## The Production Recipe

```bash
--cache-type-v vtq3_2 --tq-deferred-v --tq-protect-sinks 4
```

At 3.06 bpw (vs f16 = 16 bpw), this gives:
- +1.9% PPL overhead on wikitext-2
- 98% of f16 generation speed on 27B dual-GPU
- 62% V-cache VRAM savings

## Three Silent Bugs

All three were flag-propagation issues. Nothing crashed. Everything
*looked* like it worked. Only when you measured carefully did you see
the flags weren't actually doing anything.

**Bug 1 — `common.cpp` forgot `tq_deferred_v`.** The field existed in
the CLI parser, existed in `cparams`, but the `common_params →
cparams` translation line was missing. Result: llama-perplexity and
llama-cli silently ignored `--tq-deferred-v`. llama-bench worked
because I had added a direct env-var reader there. Lesson: never trust
one frontend's behaviour to validate flag wiring.

**Bug 2 — `arg.cpp` missed `set_examples(...)`.** The
`--tq-protect-{sinks,layers}` options were registered but without
specifying which frontends they apply to. Default scope apparently
maps to nothing. Both flags silently no-op'd in perplexity/bench/cli.

**Bug 3 — hybrid-model counter drift.** This was the subtle one.
The sink-protection code said *"apply to first KV layer"* via:

```cpp
for (j = 0; j < il; j++) if (hparams.has_kv(j)) kv_layer_idx++;
if (kv_layer_idx == 0 && hparams.has_kv(il)) { ... }
```

On non-hybrid models this works. On Qwen3.5 (Mamba+attention hybrid),
`hparams.has_kv(j)` returns **true for recurrent layers too** — the
`filter()` callback is what distinguishes attention from recurrent.
Without honouring filter, the counter incremented on every layer, so
it was never 0 when we hit the first attention-KV layer at il=3.
Fix: `&& (!filter || filter(j))`.

## Why PPL Measurements Looked Perfect (But Weren't)

Yesterday's report showed vtq3_2 at "~0% PPL overhead with
sink=4+deferred." That was wrong, and it's a beautiful case of
measurement artifact.

`--tq-deferred-v` uses a state machine: STAGING during prefill,
READY at prefill→decode transition, DONE after convert. llama-
perplexity runs **pure prefill** — 4-chunk sweep with no decode.
So state stays in STAGING, reads come from f16-staging, and PPL
matches f16. It's not measuring the quantized cache; it's measuring
f16.

Removing `--tq-deferred-v` exposed the true numbers:

| Type | no sink | sink=4 |
|------|---------|--------|
| vtq2_2 (2.06 bpw) | +9.0% | +8.0% |
| vtq3_2 (3.06 bpw) | +2.4% | +1.9% |
| vtq4_2 (4.06 bpw) | +0.4% | +0.6% (noise) |

Sink protection reliably trims ~0.5-1pp at low bpw where sinks
dominate. At 4 bpw the baseline is already f16-parity so sink has
no effect (within measurement noise).

## Speed Validation

Extended tg512 sweep on 0.8B:

| type | baseline | +both |
|------|----------|-------|
| vtq2_2 | ~7 t/s | **196.20** |
| vtq3_2 | 7.29 | **196.22** |
| vtq4_2 | — | **196.05** |

27B dual-GPU tg1024:

| config | tg1024 | vs f16 |
|--------|--------|--------|
| f16 | 14.89 | — |
| vtq3_2+both | 14.62 | -1.8% |

No crashes, no OOM, no NaN across ~6h of accumulated runs.

## What's Next

Gate for the "Trick 17" research series is now open. The plan:

1. **Trick 2** — per-head precision mixing (high-variance heads get
   vtq4_2, low-variance get vtq2_2, average stays at 3 bpw)
2. **Trick 4** — correction overlay buffer (top-N quant errors patched
   lossless)
3. **Deploy test-box:8791** — update production stack with the fixed
   flags

Plus one question worth debating: whether to break the perplexity
measurement loop (by moving the STAGING→DONE transition to also fire
at end-of-prefill, not just at decode). That would give real quantized
PPL from perplexity runs, but changes semantics for the inference
path too.

Files: `docs/plans/2026-04-20-cuda-stability-validation.md`
Commits: e40416d8f, 99ec97e96, c9d95e747
