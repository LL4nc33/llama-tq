# Production VTQ3_1 Deploy-Test — REVERTED

**Datum:** 2026-04-23
**Dauer:** ~30min (PID 2749407 lifetime)
**Verdict:** **REVERT** — Performance-Regression zu groß

## Configuration Change

Production gpu00:8791 running Qwen3.5-35B-A3B-IQ2_XS + tq2_1 K + 400k ctx + parallel 2.

Tested: `--cache-type-v tq3_1` (was: `--cache-type-v f16`).
Note: Production build uses old naming `tq3_1` = current branch `vtq3_1`.

## Measurements

### Memory (GPU VRAM used)
| State | GPU0 | GPU1 | Total | Δ |
|---|---|---|---|---|
| f16 V (baseline) | 11542 MiB | 11270 MiB | **22812 MiB** | — |
| tq3_1 V | 10284 MiB | 9262 MiB | **19546 MiB** | **-3266 MiB (-14%)** |

Memory savings moderate — f16 V wasn't fully allocated at idle (lazy ctx).

### Token Generation (TG)
| Test | Config | n_tokens | TG tok/s |
|---|---|---|---|
| Short generation | tq3_1 V | 30 | 46.8 |
| **Long generation** | **tq3_1 V** | **297** | **12.37** |
| Long generation | f16 V (restored) | 300 | **67.65** |

**TG regression: 5.5× slower on long generation.** Short generation (~30 tokens)
shows only 30% regression; scaling gets dramatically worse with length.

### Output Quality
Smoke test output grammatisch + faktisch korrekt:
> "Wien ist die Hauptstadt Österreichs und bekannt als „Stadt der Musik" sowie
> für seine historischen Paläste und die Kaffeehauskultur."

No quality regression visible in short samples. Longer generation also coherent.

## Root Cause Analysis

TG regression is **length-dependent**, which strongly suggests **per-token
V-dequant cost** scales with context length in the FA kernel:
- Short context + short generation → dequant cost masked by compute
- Long context (or long generation → growing attention span) → dequant dominates

Compare to v6 research TG-benchmark plan (which we couldn't run on CPU): this
matches the hypothesis that VTQ V-cache saves memory at a TG cost. On 35B prod:
- **f16 V** is 60-70 tok/s because V-reads are direct memory loads (no dequant)
- **tq3_1 V** needs FWHT-free dequant per attention read; at ~400k ctx this is a
  significant fraction of decode time

## Conclusion

**The "VTQ3_1 as recommended default" guidance in README is correct for new deployments**
where memory is the binding constraint. **But for existing f16-V production that fits
in VRAM, switching to VTQ3_1 is a 5× TG regression for modest memory savings.**

Updated mental model:
- **VTQ3_1 recommended ↔ memory-constrained deployments** (can't fit f16 V)
- **f16 V is preferable ↔ memory-abundant deployments** (fits with headroom)
- Our 35B prod at 400k ctx with 2 slots currently fits f16 V in 24GB total → stay f16

## Actions Taken

1. Stopped PID 2749407 (tq3_1 V test)
2. Restarted with original `--cache-type-v f16` config → PID 2753213
3. Verified health endpoint returns `{"status":"ok"}`
4. Smoke-tested TG on restored config: **67.65 tok/s @ 300 tokens** (baseline confirmed)
5. Production back to steady state.

## Follow-up Docs Update

README.md currently says VTQ3_1 is "recommended default". We should refine:
- Use VTQ3_1 when **memory-constrained** (large ctx, smaller GPUs)
- Use f16 V when **memory allows** (small ctx or large-VRAM setups)

This is not a reversal — it's a clarification. The rel-MSE data still stands
(VTQ3_1 = 3.07% vs VTQ2_1 = 13.25%), but TG cost on large models/long-context
is more substantial than we'd estimated.
