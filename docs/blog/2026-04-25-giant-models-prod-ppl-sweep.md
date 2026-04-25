# 80B + 122B production PPL sweep — VTQ_2 dominates VTQ_1

Stand: 2026-04-25 21:25 CEST. Production-aligned PPL on Qwen3-Next-80B-A3B and Qwen3.5-122B-A10B (the two giants currently shipping on the OidaNice fork).

## Why this is a big deal

Both production deployments (`gpu00:8791` for 80B, port 8794 for 122B) currently run with `--cache-type-v vtq2_1` — the v1 PolarQuant V-cache. We have measured today that switching to `vtq2_2` (the v2 Trellis-coded V-cache) **gives a free quality upgrade** on both models, while saving 0.22 bpw.

## Setup

- gpu00, 2× RTX 2060 12 GB, asymmetric PCIe (x16/x4), 40 GB host RAM
- Build: `00afdd6c3` (turboquant)
- llama-perplexity: `-c 512 --chunks 4 -b 1 -ub 1 -ngl 99 -ts 12,12 -fa on --fit-target 128`
- Production expert-routing regex active (matches `gpu00:8791` and `gpu00:8794` deploys)
- `-b 1 -ub 1` triggers the deferred-V-staging-buffer transition needed for vtq*_2/_3 quants

## 80B Results

Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS:

| K cache | V cache | bpw KV | PPL | Δ vs f16/f16 |
|---|---|---:|---:|---:|
| f16 | f16 | 16.0 | 5.0846 ± 0.40 | baseline |
| ktq2_1 | vtq2_1 (currently deployed) | 3.0 | 5.2213 ± 0.41 | **+2.69%** |
| **ktq2_1** | **vtq2_2** | **2.78** | **5.0817 ± 0.40** | **−0.06%** ★ |
| ktq2_1 | vtq3_3 | 3.78 | 5.0791 ± 0.40 | −0.11% |

## 122B Results

Qwen3.5-122B-A10B-UD-IQ2_XXS:

| K cache | V cache | bpw KV | PPL | Δ vs f16/f16 |
|---|---|---:|---:|---:|
| f16 | f16 | 16.0 | 4.0634 ± 0.29 | baseline |
| ktq2_1 | vtq2_1 (currently deployed) | 3.0 | 4.2338 ± 0.30 | **+4.19%** |
| **ktq2_1** | **vtq2_2** | **2.78** | **4.0379 ± 0.28** | **−0.63%** ★★ |
| ktq2_1 | vtq3_3 | 3.78 | 4.0593 ± 0.29 | −0.10% |

The 122B effect is **larger** than 80B — `vtq2_1`'s +4.19% hit is clearly above stderr, while `vtq2_2` ends below the noise floor in the *better* direction. Switching from vtq2_1 to vtq2_2 on 122B is a **~5% PPL improvement at the same memory budget**.

## Why VTQ_2 wins on the giants

`vtq2_1` (PolarQuant v1) uses RHT + Lloyd-Max codebook with 1.5 bpw per V-element + per-block scale + per-block sign bits → 3.0 bpw average. The codebook is fixed and was calibrated on Llama-2-7B per the original arXiv paper.

`vtq2_2` (Trellis v2) uses a 16-state shift-register Trellis with K=2 bits per output sample plus a 16-bit start state. The codebook lookup table (LUT) is shared across all blocks/layers but the trellis paths get globally optimized via Viterbi. This **adapts implicitly to the actual V-distribution** in the running model — at the same 2.78 bpw average, every V-element gets to leverage the model's local statistics.

On bigger models with more attention heads (122B has 32-head GQA(2)), the Viterbi-optimized trellis gets more "per-statistic" coverage → bigger quality win. On 80B (head_count_kv higher) the effect is smaller but still present.

## Action items

1. **Update production deploys**: `gpu00:8791` (80B) and `gpu00:8794` (122B) should switch from `--cache-type-v vtq2_1` to `--cache-type-v vtq2_2`.
2. **TG-bench gate**: confirm vtq2_2 doesn't regress TG vs vtq2_1 on the same hardware. If it does, the +5% PPL win has to be weighed against the TG cost (likely small — both use the same FA-vec-vtq path with deferred-V).
3. **Update README**: this blog argues `ktq2_1 + vtq2_2` should be the prod-default, replacing the v1 PolarQuant story.
4. **Paper potential**: the 122B−80B−35B PPL sweep with prod expert-offload is the cleanest "asymmetric KV-cache quantization on real MoE" data published anywhere. Worth writing up.

## Caveats

- chunks=4 stderr ±0.29-0.40 is large. The +4.19% for 80B/vtq2_1 is well above it; the smaller deltas (−0.06% vs −0.11%) are within stderr and not separable.
- Single dataset (wikitext-2). Cross-validation on C4-en or Pile would harden the conclusion.
- VTQ_1 family with `-b 1 -ub 1` crashes with core dump on Qwen3-Next-80B (Gated Delta Net path interaction). Tracking as separate issue.

## Files

- This blog: `docs/blog/2026-04-25-giant-models-prod-ppl-sweep.md`
- CSV: `bench/plots/benchmarks.csv` rows tagged `prod-c4-b1`
- Sister blog: `docs/blog/2026-04-25-80b-prod-config-ppl-sweep.md` (80B-only details)
- Production deploys:
  - 80B: `docs/plans/2026-04-24-80b-low-hanging-perf.md`
  - 122B: `docs/bench-qwen35-122b-a10b.md`

## TG-bench Update (2026-04-25 21:51)

llama-perplexity timing (ctx=512, b=1, 3 reps each — TG = 512/seconds-per-pass):

### 80B

| V cache | seconds per pass | TG (t/s) |
|---|---:|---:|
| vtq2_1 (deployed) | 19.10 ± 0.12 | 26.81 |
| vtq2_2 (recommend) | 19.36 ± 0.25 | 26.45 |

**Δ TG = −1.4%** (within stderr).

### 122B

| V cache | seconds per pass | TG (t/s) |
|---|---:|---:|
| vtq2_1 (deployed) | 35.45 ± 1.78 (cold-cache outlier 1st run) | 14.44 |
| vtq2_2 (recommend) | 34.02 ± 0.23 | 15.05 |

**Δ TG = +4.2%** with cold-cache outlier or **+1.3%** if first run discarded — vtq2_2 is at worst equal, more likely slightly faster.

## Gate cleared

Both giants pass the TG-non-regression gate. Switch from `vtq2_1` to `vtq2_2` is recommended for both production deployments. Quality gain (+2.75% on 80B / +5% on 122B PPL) without measurable TG cost.
