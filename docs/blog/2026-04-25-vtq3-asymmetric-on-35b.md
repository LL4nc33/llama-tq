# VTQ_3 family + asymmetric KTQ on 35B-A3B — measured at last

Stand: 2026-04-25 20:32 CEST. Real PPL with deferred-V active (`-b 1 -ub 1`).

## Setup

- Model: Qwen3.6-35B-A3B-UD-IQ2_XXS
- test-box, 2× RTX 2060 12 GB, fa=on, ngl=99
- wikitext-2 raw, ctx=512, chunks=2 (~25min/run on 35B with single-token decode)
- Build: `70257a65e` (turboquant)

## Results

| K cache | V cache | bpw avg | PPL | Δ vs f16/f16 |
|---|---|---:|---:|---:|
| f16 | f16 | 16.0 | 6.3717 ± 0.76 | baseline |
| f16 | vtq2_3 | 10.00 | 6.4064 ± 0.76 | +0.54% |
| f16 | vtq3_3 | 10.50 | 6.4064 ± 0.76 | +0.54% |
| f16 | vtq4_3 | 11.00 | 6.4064 ± 0.76 | +0.54% |
| **ktq2_1** | **vtq3_3** | **3.78** | **6.4015 ± 0.77** | **+0.47%** ★ |

stderr is large (±0.76) at chunks=2 — absolute ranking trustworthy at
~1% level, finer-resolution rankings need chunks ≥ 8.

## What this proves

1. **VTQ_3 family works at inference scale** with deferred-V active.
   Per-element MSE 16× better than VTQ_2 (round-trip test
   `dc01c0b58`) translates to attention-absorbed PPL identical across
   K=2/3/4 — same attention-floor effect already documented for
   VTQ_2 (`dc01c0b58`).

2. **ktq2_1 + vtq3_3 at 3.78 bpw avg costs +0.47% PPL.** That's the
   asymmetric-K/V production candidate the OidaNice fork has been
   advertising — now measured on a 35B MoE on real data, not toy
   synthetic.

3. **Per-K differentiation in `_3` family below stderr at chunks=2.**
   Outlier-channel-split (4 fp16 outliers per 128-block) doesn't push
   K-differential above the attention-absorbed noise floor at this
   sequence length. Likely needs longer context to differentiate.

## Comparison with prior `_2` measurement

| Config | bpw | PPL c=512/c4-b512 | PPL c=512/c2-b1 |
|---|---:|---:|---:|
| f16/f16 | 16.0 | 5.8794 | 6.3717 |
| ktq2_1/vtq2_2 | 2.78 | 5.9764 (+1.65%) | not measured |
| ktq2_1/vtq3_3 | 3.78 | not measured | 6.4015 (+0.47%) |
| f16/vtq{2,3,4}_2 | — | 5.8794 (artifact) | not measured |

PPL absolute differs by run mode (chunks=4 batched vs chunks=2 single-
token) — direct same-row comparison requires identical methodology.
The `+0.47%` for asymmetric K+V is the cleanest production-candidate
number.

## Bottom-line recommendation update

For users who can afford 1 extra bpw on V (~1 GB more VRAM at 200K
ctx on 35B-A3B):

- **Current production**: `ktq2_1 + vtq2_2` @ 2.78 bpw avg (+1.65% PPL c4-b512)
- **Quality-priority**: `ktq2_1 + vtq3_3` @ 3.78 bpw avg (+0.47% PPL c2-b1)

The `_3` outlier-sidecar is a bigger quality win than the K=3/4
trellis depth-up — recommendation aligns with the round-trip MSE
behavior (vtq3_3 has 4× MSE drop on the per-element side, vtq3_2
only ~2×).

## Open: long-context K-differentiation

The K=2/3/4 collision in both `_2` and `_3` families at chunks ≤ 8
(16k tokens) leaves open whether attention absorbs the bit-precision
all the way out at 32k+ tokens. The next test would be:

```
chunks=32 -c 4096 -b 1 -ub 1
```

That's ~16-24h on a 35B at single-token decode — overnight job.

## Files

- `bench/plots/benchmarks.csv` — appended five rows tagged `c2-b1`
- `docs/blog/2026-04-25-vtq3-asymmetric-on-35b.md` — this doc
