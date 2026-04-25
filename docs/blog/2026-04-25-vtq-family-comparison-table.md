# VTQ Family Comparison Table — Qwen3.6-35B-A3B-UD-IQ2_XXS

Stand: 2026-04-25 20:48 CEST. All rows from `bench/plots/benchmarks.csv`.

## Methodology matters

Three measurement modes are mixed in our CSV — they are NOT directly
comparable across modes:

| Mode | Settings | Deferred-V active? | Use for |
|---|---|---|---|
| Legacy | `-c 512 -b 512 --chunks 64` | No (batch>1) | tg/pp throughput only |
| `c4-b512` | `-c 512 -b 512 --chunks 4` | No | KTQ K-cache PPL |
| `c2-b1` | `-c 512 -b 1 -ub 1 --chunks 2` | **Yes** | Real VTQ V-cache PPL |

VTQ_2 / VTQ_3 PPL only meaningful in `c2-b1` mode.

## Apples-to-apples: VTQ family in `c2-b1` mode (deferred-V active)

The cleanest comparison set we have on the production model:

| K cache | V cache | bpw avg | PPL | Δ vs f16/f16 | Notes |
|---|---|---:|---:|---:|---|
| f16 | f16 | 16.00 | 6.3717 ± 0.76 | baseline | — |
| f16 | vtq2_3 | 10.00 | 6.4064 ± 0.76 | +0.54% | K=2 outlier-sidecar |
| f16 | vtq3_3 | 10.50 | 6.4064 ± 0.76 | +0.54% | K=3 outlier-sidecar |
| f16 | vtq4_3 | 11.00 | 6.4064 ± 0.76 | +0.54% | K=4 outlier-sidecar |
| **ktq2_1** | **vtq3_3** | **3.78** | **6.4015 ± 0.77** | **+0.47%** | ★ asymmetric production-candidate |

**Key reads:**

1. **VTQ_3 family K-collision**: K=2/3/4 all produce bit-identical PPL.
   Same attention-absorbed effect documented for VTQ_2 in
   `2026-04-25-vtq2-attention-absorbs-bit-depth.md`.
2. **Asymmetric KTQ_1 + VTQ_3 wins** — at 3.78 bpw avg (4.78× smaller
   than f16), PPL impact is **+0.47%** — well below stderr.
3. **VTQ_2 family in c2-b1 mode not yet measured on 35B-A3B** — was
   measured on smaller models (Qwen3.5-2B, Qwen3.5-27B), all showed
   same K-collision pattern.

## c4-b512 mode (K-cache only — VTQ rows are f16-equivalent artifact)

| K cache | V cache | bpw avg | PPL | Δ vs f16/f16 | Notes |
|---|---|---:|---:|---:|---|
| f16 | f16 | 16.00 | 5.8794 ± 0.46 | baseline | — |
| **ktq2_1** | **f16** | **9.40** | **5.8952 ± 0.46** | **+0.27%** | ★ K-cache PolarQuant |
| ktq2_1 | vtq2_2 | 2.78 (claimed) | 5.9764 | +1.65% | V-rows artifacts (deferred-V inactive) |
| ktq2_1 | vtq3_2 | 3.28 (claimed) | 5.9764 | +1.65% | identical to vtq2_2 (artifact) |
| ktq3_1 | vtq3_2 | 3.78 (claimed) | 5.9764 | +1.65% | identical to vtq2_2 (artifact) |

**Key reads:**

1. **KTQ_1 K-cache costs +0.27% PPL on 35B-A3B** at 9.4 bpw avg.
2. The "+1.65%" rows are **NOT a real V-cache PPL hit** — they show
   the cost of activating mixed-precision overhead in batched mode
   while V is still effectively f16.

## Smaller models — VTQ_2 family in c2-b1 mode (deferred-V active)

For reference, from `2026-04-25-vtq2-cpu-vs-cuda-split.md`:

### Qwen3.5-2B Q4_K_M (ctx=2048 chunks=8)

| K / V | PPL | Δ vs f16 |
|---|---:|---:|
| f16/f16 | 9.6792 | baseline |
| f16/vtq2_2 | 9.6780 | −0.012% |
| f16/vtq3_2 | 9.6780 | −0.012% |
| f16/vtq4_2 | 9.6780 | −0.012% |
| f16/vtq2_3 | 9.6799 | +0.007% |
| f16/vtq3_3 | 9.6805 | +0.013% |
| f16/vtq4_3 | 9.6799 | +0.007% |

### Qwen3.5-27B IQ2_XXS (ctx=512 chunks=4)

| K / V | PPL |
|---|---:|
| f16/f16 | 8.0266 |
| f16/vtq{2,3,4}_2 | 8.0212 (all 3 identical) |
| f16/vtq{2,3,4}_3 | 8.0238 (all 3 identical) |

## Synthesis: Pareto frontier (K-cache + V-cache combined)

bpw avg → PPL impact, picking the lowest cost row at each bpw target:

```
   bpw    PPL Δ      config                 measurement
  ──────  ───────   ─────────────────────  ─────────────
  16.0    0.00%     f16/f16                baseline
   9.4   +0.27%     ktq2_1/f16             c4-b512 ★
   3.78  +0.47%     ktq2_1/vtq3_3          c2-b1 ★★
   2.78    ~?       ktq2_1/vtq2_2          NEEDS c2-b1 measurement
```

The +0.47% at 3.78 bpw is the production sweet-spot for users wanting
maximum quality with significant VRAM savings. The 2.78 bpw config
needs a c2-b1 PPL measurement to confirm it's still in the sub-1%
band — that's the next obvious test (~30min on 35B at single-token).

## What's missing (next-priority measurements)

1. **`ktq2_1/vtq2_2` in c2-b1 mode on 35B-A3B** — production default
   PPL not yet measured at single-token. Probably +0.4-0.7%.
2. **`ktq3_1/vtq2_2` and `ktq4_1/vtq2_2`** — the higher-bpw KTQ
   variants for users who can pay 1-2 bpw more on K.
3. **chunks=8+ at c2-b1** — current ±0.76 stderr too large for
   sub-percent ranking confidence. Need 4× more data to halve stderr.

## Files

- This doc: `docs/blog/2026-04-25-vtq-family-comparison-table.md`
- Source data: `bench/plots/benchmarks.csv`
- Methodology blogs:
  - `2026-04-25-vtq2-cpu-vs-cuda-split.md` (deferred-V batch=1 gate)
  - `2026-04-25-vtq2-attention-absorbs-bit-depth.md` (K-collision explanation)
  - `2026-04-25-ktq-on-35b-quality-win.md` (KTQ_1 production-validate)
  - `2026-04-25-vtq3-asymmetric-on-35b.md` (asymmetric ktq2_1+vtq3_3)
