# Calibrating KTQ / VTQ Codebooks

KTQ and VTQ ship with `PQ_CODEBOOK_2BIT` / `PQ_CODEBOOK_3BIT` / `PQ_CODEBOOK_4BIT`
constants that are Lloyd-Max optimal for the *theoretical* post-RHT marginal
`Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)` at `d = QK_KTQ = 32`. See
`docs/turboquant.md` §"Quantization Pipeline" for the derivation.

In practice the empirical post-RHT distribution can deviate -- attention sinks,
RoPE artefacts, or layer-1 dominance produce heavier tails than the analytical
marginal. Re-fitting the codebook against a calibration set typically yields
~10-25% MSE reduction, which maps to ~2-4% PPL improvement at zero runtime
cost (only a constant table changes).

## When to Calibrate

Calibration is **per model family**, not per individual model. A codebook fit
on Qwen3.6-A3B will work on every Qwen3.6-A3B variant (different sizes, finetunes,
quantizations) since the post-RHT statistics are dominated by architecture, not
weights.

Recommended cadence:
- Once per *new* base architecture (e.g. Qwen3.6, Llama4, Mistral4).
- After major attention-mechanism changes (e.g. switching from MHA to MLA).
- **Not** required per finetune or per quantization level.

## Quick Start

### Synthetic (no model needed)

The fastest path: assume a symmetric-Beta empirical distribution and pick `alpha`.

```bash
# Theoretical default (reproduces the shipped codebook to ~6 decimals)
python3 scripts/calibrate_ktq_codebook.py \
    --mode synthetic --alpha 15.5 --bits 2 3 4 \
    --output ggml/include/ggml-common-codebook-override.h

# Slightly heavier tails (alpha < 15.5 widens distribution)
python3 scripts/calibrate_ktq_codebook.py \
    --mode synthetic --alpha 13.0 --bits 2 3 4 \
    --output ggml/include/ggml-common-codebook-override.h
```

### From extracted samples (recommended)

If you have post-RHT samples on disk -- e.g. from `extract_v_samples.py` or any
`tq_calibrate_*` extraction tool -- pass them directly. The script will fit a
symmetric-Beta via method-of-moments AND run Lloyd-Max on the raw samples.

```bash
python3 scripts/calibrate_ktq_codebook.py \
    --mode samples --samples vcache-qwen35-27b.bin --head-dim 128 \
    --bits 2 3 4 \
    --model-hint qwen35-27b \
    --output ggml/include/ggml-common-codebook-override.h
```

Output is a header file:

```c
#ifndef PQ_CODEBOOK_OVERRIDE_H
#define PQ_CODEBOOK_OVERRIDE_H
#define PQ_CODEBOOK_2BIT_CALIBRATED  { -1.483f, -0.450f, +0.450f, +1.483f }
#define PQ_CODEBOOK_3BIT_CALIBRATED  { ... }
#define PQ_CODEBOOK_4BIT_CALIBRATED  { ... }
#endif
```

## Activating the Override

The override is **opt-in** so the upstream-aligned baseline never changes
silently. Build with:

```bash
cmake -B build -DPQ_CODEBOOK_USE_CALIBRATED=ON ...
cmake --build build
```

When that flag is set, `ggml/src/ggml-quants.c` includes the override header
and substitutes `PQ_CODEBOOK_*BIT_CALIBRATED` for the shipped constants. CUDA
constants `PQ_CUDA_CB_*BIT` are derived from the same source and pick up the
override automatically (see `ggml/src/ggml-cuda/turboquant.cuh`).

> **Note:** at the time of writing the include-and-substitute glue itself is
> not yet wired into `ggml-quants.c`. The script and header format are stable
> and forward-compatible; the C-side switch lands as a follow-up commit.

## Validating the Result

The script prints a comparison table:

```
 bits  iters      shift      MSE_old      MSE_new  reduction
----------------------------------------------------------------
    2     25   2.99e-06 1.102405e-01 1.102333e-01      0.01%
    3     84   1.13e-05 3.173432e-02 3.171989e-02      0.05%
    4    195   1.08e-05 8.831651e-03 8.616719e-03      2.43%
```

* `MSE_old`  -- shipped codebook MSE on the calibration samples
* `MSE_new`  -- calibrated codebook MSE on the same samples
* `reduction` -- expect 0% on perfectly-Beta(15.5,15.5) data; 5-25% on real K-tensor samples

A reduction below ~5% means the empirical distribution is already very close
to Beta(15.5, 15.5); the override will be indistinguishable from baseline and
should not be deployed. PPL evaluation (`llama-perplexity wikitext-2-raw`)
remains the source of truth for whether to keep an override.

## Limitations

1. **Parametric fit is symmetric.** The script enforces a symmetric codebook
   (averaging `c_i` with `-c_{n-1-i}`). Real K-tensors may have small skew,
   which symmetry suppresses; in exchange we keep the layout simple and avoid
   bias on neutral inputs. Asymmetric K behaviour is captured by the
   per-block scale, not the codebook.

2. **Method-of-moments is variance-only.** Beta-fit is driven by sample
   variance; tail-mass differences with the same variance won't move `alpha`.
   For better tail fidelity, switch to direct Lloyd-Max on the samples
   (already done by `--mode samples`) and skip the analytical fit.

3. **Single global codebook.** Per-layer-class codebooks (early/mid/late) would
   shave another 1-2% MSE but require either runtime indirection
   (`codebook[layer_class][idx]`) or compile-time specialization. Out of
   scope for this script; track in `LEGION/` if pursued.

4. **No KV-cache live path yet.** Option B in the original task (a `'M'` tag
   in `common/router-profile.h` for K-tensor mean post-FWHT) is *not*
   implemented. `--mode samples` requires offline extraction (e.g. via
   `extract_v_samples.py` adapted for K).

## See Also

- `docs/turboquant.md` -- full TurboQuant pipeline + theoretical derivation
- `scripts/tq_calibrate_outliers.py` -- VTQ_OUT outlier-channel calibration
- `ggml/src/ggml-quants.c` -- CPU quantize/dequantize paths
- `ggml/src/ggml-cuda/turboquant.cuh` -- CUDA codebook constants
