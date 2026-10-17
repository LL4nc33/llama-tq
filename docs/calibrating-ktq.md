# Calibrating KTQ / VTQ Codebooks

KTQ and VTQ ship with `PQ_CODEBOOK_2BIT` / `PQ_CODEBOOK_3BIT` / `PQ_CODEBOOK_4BIT`
constants that are Lloyd-Max optimal for the *theoretical* post-RHT marginal
`Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)` at `d = QK_KTQ = 32`. See
[turboquant.md, How It Works](turboquant.md#how-it-works) for the pipeline.

**Status:** the script below fits calibrated codebooks and writes them as a header; using that header in the build is
not implemented yet, so the shipped codebooks are always used. Since 2026-10-11 the encoder also searches the block scale,
which already takes up much of what a re-fitted codebook would gain. The codebooks only affect the codebook types
`ktq*_1` / `vtq*_1`; the trellis types behind the short names `ktq2`-`ktq4` and `vtq2`-`vtq4` use the trellis code
table (`ggml/src/ggml-trellis.c`) and are not changed by this script.

In practice the empirical post-RHT distribution can deviate -- attention sinks,
RoPE artefacts, or layer-1 dominance produce heavier tails than the analytical
marginal. Re-fitting the codebook against a calibration set lowers the
reconstruction error on those samples (the example below: 0.01-2.4 %); the
effect on perplexity has not been measured.

## When to Calibrate

Not measured yet. The assumption behind the script is that the post-RHT statistics depend mostly on the architecture, so
one codebook per model family would do (not one per finetune or quantization level). Check any calibrated codebook by
perplexity or KL divergence before using it.

## Quick Start

### Synthetic (no model needed)

The fastest path: assume a symmetric-Beta empirical distribution and pick `alpha`.

```bash
# Theoretical default (alpha 15.5, the distribution the shipped codebooks were fitted to)
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
* `reduction` -- 0% on data that follows Beta(15.5, 15.5) exactly

A small reduction means the samples are already close to Beta(15.5, 15.5). Perplexity or KL divergence decides whether a
calibrated codebook is worth keeping.

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

3. **Single global codebook.** Per-layer-class codebooks (early/mid/late) are not tried; they would need either runtime
   indirection (`codebook[layer_class][idx]`) or compile-time specialization.

4. **No sample extraction.** `--mode samples` needs post-RHT samples extracted offline; the repository has no tool for
   that.

## See Also

- `docs/turboquant.md` -- full TurboQuant pipeline + theoretical derivation
- `scripts/tq_calibrate_outliers.py` -- outlier-channel calibration (research script)
- `ggml/src/ggml-quants.c` -- CPU quantize/dequantize paths
- `ggml/src/ggml-cuda/turboquant.cuh` -- CUDA codebook constants
