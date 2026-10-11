# Calibrating KTQ / VTQ Codebooks

KTQ and VTQ ship with `PQ_CODEBOOK_2BIT` / `PQ_CODEBOOK_3BIT` / `PQ_CODEBOOK_4BIT`
constants that are Lloyd-Max optimal for the *theoretical* post-RHT marginal
`Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)` at `d = QK_KTQ = 32`. See
[turboquant.md, How It Works](turboquant.md#how-it-works) for the pipeline.

**Status:** the script below fits calibrated codebooks and writes them as a header; using that header in the build is
not implemented yet, so the shipped codebooks are always used. Since 2026-10-11 the encoder also searches the block scale,
which already takes up much of what a re-fitted codebook would gain.

In practice the empirical post-RHT distribution can deviate -- attention sinks,
RoPE artefacts, or layer-1 dominance produce heavier tails than the analytical
marginal. Re-fitting the codebook against a calibration set lowers the
reconstruction error on those samples (the example below: 0.01-2.4 %); the
effect on perplexity has not been measured.

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

If you have post-RHT samples on disk (the repository has no extraction tool for
them; dump the rotated blocks from your own build), pass them directly. The script will fit a
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

Planned, not implemented: an opt-in build switch so the shipped codebooks never
change silently:

```bash
cmake -B build -DPQ_CODEBOOK_USE_CALIBRATED=ON ...
cmake --build build
```

With that switch, `ggml/src/ggml-quants.c` would include the override header and use `PQ_CODEBOOK_*BIT_CALIBRATED`
instead of the shipped constants, and the CUDA constants `PQ_CUDA_CB_*BIT` would follow
(`ggml/src/ggml-cuda/turboquant.cuh`). Neither part exists yet.

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
   scope for this script.

4. **No KV-cache live path yet.** Option B in the original task (a `'M'` tag
   in the former router profiler for K-tensor mean post-FWHT) is *not*
   implemented. `--mode samples` requires offline extraction of the samples.

## See Also

- `docs/turboquant.md` -- full TurboQuant pipeline + theoretical derivation
- `scripts/tq_calibrate_outliers.py` -- VTQ_OUT outlier-channel calibration
- `ggml/src/ggml-quants.c` -- CPU quantize/dequantize paths
- `ggml/src/ggml-cuda/turboquant.cuh` -- CUDA codebook constants
