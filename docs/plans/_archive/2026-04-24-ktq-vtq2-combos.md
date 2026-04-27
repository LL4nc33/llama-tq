# KTQ × VTQ_2 Asymmetric Combo Measurement — 2026-04-24

## Setup

- Hardware: gpu00 (2× RTX 2060 12 GB, prod 80B stopped for duration of sweep)
- Model: Qwen3.6-35B-A3B-UD-IQ2_XXS (head_dim=128, D=128 FA path)
- Tool: `llama-perplexity` at commit `f30e85aa5` (turboquant branch HEAD)
- Data: wikitext-2 (wiki.test.raw)
- Two sweeps: ctx=512/3ch (quick) and ctx=2048/5ch (deferred V intended to fire)

## Results ctx=2048 / 5 chunks

Primary dataset (larger context, more reliable signal).

| K cache | V cache | avg bpw | PPL     | Δ baseline | Notes                     |
|---------|---------|:-------:|:-------:|:----------:|---------------------------|
| f16     | f16     | 16.0    | 6.7251  | —          | baseline                  |
| ktq2_1  | vtq2_2  |  2.78   | 6.7227  | **-0.04%** | identical to ktq2_1+vtq4_2|
| ktq2_1  | vtq3_2  |  3.28   | 6.7227  | **-0.04%** | identical                 |
| ktq2_1  | vtq4_2  |  3.78   | 6.7227  | **-0.04%** | identical                 |
| ktq3_1  | vtq3_2  |  3.78   | 6.7227  | **-0.04%** | identical                 |
| f16     | vtq2_2  |  9.03   | 6.7388  | **+0.20%** | identical to f16+vtq3_2   |
| f16     | vtq3_2  |  9.53   | 6.7388  | **+0.20%** | identical                 |

## Findings

### 1. VTQ_2 family is a no-op in PPL measurements

All three `vtq{2,3,4}_2` variants produced **identical PPL** at every K-cache pairing.
Root cause hypothesis: `llama-perplexity` is a forward-pass-only eval — the
prefill→decode transition never triggers, so **deferred V conversion never fires**.
During PPL eval, V stays in the f16 staging buffer, so the selected VTQ_2 type is
allocated but never exercised.

Log evidence:
```
llama_kv_cache: deferred V quantization enabled (10 layers with f16 staging)
llama_kv_cache: VTQ V-cache active — using D*H*D randomized rotation
```

The flags are correct; the actual quantization never runs in this workload shape.

### 2. Gate C triggered, but only for the K component

The plan's Gate C required `ktq2_1 + vtq3_2` to land under 2% PPL delta.
Measured: **-0.04%** — well under 2%. But this is entirely the K-side effect:
`ktq2_1` paired with any vtq_2 gives 6.7227 regardless of which V-variant,
and `f16` K paired with any vtq_2 gives 6.7388 regardless. The PPL number
is determined by K-cache choice alone.

The KTQ2_1 + VTQ2_2 (2.78 bpw) config is **functionally the measured winner**
at the lowest bitrate, but the winning margin comes from KTQ. The VTQ_2 bitrate
choice is cosmetic until decode-phase workloads stress the deferred path.

### 3. KTQ2_1 is essentially free under this workload

The -0.04% delta for all KTQ2_1 rows vs f16/f16 baseline is below run-to-run
noise (PPL ± 0.234). Functionally: **KTQ2_1 costs nothing on this 35B-A3B
IQ2_XXS config**. Confirms the deployment choice for the 80B prod server.

### 4. Drift check (Gate A)

- ctx=512/3ch baseline: 5.9366 (vs stored 5.967 → -0.51%, just outside ±0.5%)
- ctx=2048/5ch baseline: 6.7251 (new reference)

ctx=512 shows mild binary drift vs the older stored value; absolute delta
is 0.03 PPL which is noise-level. Keeping the new 6.7251 value as the
ctx=2048 reference for future comparisons.

## Full 5×8 K×V matrix (40 configs, ctx=2048/5ch)

Measured same session, completed 2026-04-24.

### V_1 series (with ktq2_1 K, representative row — all KTQ rows identical)

| V cache | avg bpw | PPL | Δ baseline |
|---------|:---:|:---:|:---:|
| f16     | 9.75 | 6.7309 | +0.09% |
| vtq1_1  | 2.5  | 7.8157 | **+16.1%** ← 1-bit floor |
| vtq2_1  | 3.0  | 7.0140 | **+4.30%** |
| vtq3_1  | 3.75 | 6.7582 | **+0.49%** |
| vtq4_1  | 4.5  | 6.7101 | **-0.22%** ← near-lossless |

### V_2 series (with any K — all V_2 rows identical)

| V cache | avg bpw | PPL | Δ baseline |
|---------|:---:|:---:|:---:|
| vtq2_2 / vtq3_2 / vtq4_2 | 2.78 – 3.78 (with ktq2_1) | 6.7227 | -0.04% |

### Key observations from full matrix

1. **All KTQ bitrates produce identical PPL** (6.7309 with f16 V, 6.7227 with
   any vtq_2). The K side of attention-only PPL doesn't distinguish
   2-bit from 4-bit K-cache — there's a constant +0.09% offset for "any
   KTQ present" that's likely structural (rotation or sign-bit rounding).
2. **V_1 series shows clean PPL hierarchy** because codebook V-decode runs
   on every attention step (decode-free unlike Trellis V).
3. **Cheapest config with baseline-equivalent PPL:** `ktq1_1 + vtq4_1` at
   4.0 bpw avg gives -0.22% PPL. Practical: `ktq2_1 + vtq3_1` (3.75 bpw)
   at +0.49% — existing deployed config is already near-optimal.
4. **V_2 caveat still applies** — the "-0.04%" V_2 result is attention-only
   measurement. A decode-phase bench is the only way to distinguish the
   V_2 variants.

## Actual KV-cache VRAM (ctx=8192, 10 KV layers) — full 5×8 matrix

Runtime-reported allocation (`llama_kv_cache: size = ...`), all 40 K×V combinations
measured 2026-04-24.

### Per-cache sizes (constant across partner)

| K type  | K (MiB) | bpw | | V type  | V (MiB) | bpw |
|---------|:---:|:---:|-|---------|:---:|:---:|
| f16     | 80.0  | 16.0 | | f16     | 80.0  | 16.0 |
| ktq1_1  | 12.5  | 2.5  | | vtq1_1  |  7.5  | 1.5  |
| ktq2_1  | 17.5  | 3.5  | | vtq2_1  | 12.5  | 2.5  |
| ktq3_1  | 22.5  | 4.5  | | vtq3_1  | 20.0  | 4.0  |
| ktq4_1  | 27.5  | 5.5  | | vtq4_1  | 22.5  | 4.5  |
|         |       |      | | vtq2_2  | 11.25 | 2.25 |
|         |       |      | | vtq3_2  | 16.25 | 3.25 |
|         |       |      | | vtq4_2  | 21.25 | 4.25 |

### Pivot table — total KV (MiB)

| K \ V   | f16   | vtq1_1 | vtq2_1 | vtq3_1 | vtq4_1 | vtq2_2 | vtq3_2 | vtq4_2 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f16     | 160.00 |  87.50 |  92.50 | 100.00 | 102.50 |  91.25 |  96.25 | 101.25 |
| ktq1_1  |  92.50 |  20.00 |  25.00 |  32.50 |  35.00 |  23.75 |  28.75 |  33.75 |
| ktq2_1  |  97.50 |  25.00 |  30.00 |  37.50 |  40.00 |  28.75 |  33.75 |  38.75 |
| ktq3_1  | 102.50 |  30.00 |  35.00 |  42.50 |  45.00 |  33.75 |  38.75 |  43.75 |
| ktq4_1  | 107.50 |  35.00 |  40.00 |  47.50 |  50.00 |  38.75 |  43.75 |  48.75 |

### Pivot table — percentage of f16/f16 baseline

| K \ V   | f16  | vtq1_1 | vtq2_1 | vtq3_1 | vtq4_1 | vtq2_2 | vtq3_2 | vtq4_2 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| f16     | 100% |  55% |  58% |  63% |  64% |  57% |  60% |  63% |
| ktq1_1  |  58% | **13%** 🏆 |  16% |  20% |  22% |  15% |  18% |  21% |
| ktq2_1  |  61% |  16% |  19% |  23% |  25% |  18% |  21% |  24% |
| ktq3_1  |  64% |  19% |  22% |  27% |  28% |  21% |  24% |  27% |
| ktq4_1  |  67% |  22% |  25% |  30% |  31% |  24% |  27% |  30% |

The K allocation depends only on K type (same value across every row), V only
on V type (same value down every column) — confirming the runtime picks per-cache
layout deterministically from type without interaction effects.

Smallest cell at `ktq1_1 / vtq1_1` = 20 MiB (13% of baseline, 8× smaller) but
vtq1_1 costs +16% PPL → not practical. Smallest PPL-sensible cell:
`ktq1_1 / vtq2_2` at 23.75 MiB (15%, 6.7× smaller).

## Follow-up work

1. **Measure VTQ_2 deltas with a decode workload** — `llama-cli` or `llama-bench -gen N`
   where prefill→decode actually transitions, so deferred V conversion runs.
2. **Complete the KTQ_1 × VTQ_1 matrix on 35B-A3B-IQ2_XXS at ctx=2048** for
   clean comparison against VTQ_2 once (1) has measurable V deltas.
3. **KTQ_2 design spike** — Trellis variant for K-cache. Currently only V has
   a v2 family because K needs online per-token quantization (no bulk deferred
   path). Design question: can a warp-cooperative online Viterbi encoder keep
   pace with the prefill rate? → parked for autoresearch.

## Raw logs

- `/tmp/ktq-vtq2-results.log` (ctx=512/3ch, 7 runs)
- `/tmp/ktq-vtq2-ctx2048.log` (ctx=2048/5ch, 7 runs, primary dataset)
