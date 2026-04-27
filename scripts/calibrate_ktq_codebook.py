#!/usr/bin/env python3
"""Calibration-aware Lloyd-Max codebook generator for KTQ / VTQ.

Re-derives the shared `PQ_CODEBOOK_2BIT` / `PQ_CODEBOOK_3BIT` / `PQ_CODEBOOK_4BIT`
constants used by `ggml/src/ggml-quants.c` against an *empirical* post-RHT
distribution rather than the theoretical Beta((d-1)/2, (d-1)/2) marginal.

Theory background (see docs/turboquant.md, §"Quantization Pipeline"):
    For unit-norm vectors in R^d with d=32, post-D*H*D rotation each coordinate
    `x_i` satisfies `(x_i + 1) / 2 ~ Beta((d-1)/2, (d-1)/2) = Beta(15.5, 15.5)`.
    The shipped codebooks are Lloyd-Max optimal for that ideal Beta.

In practice, K-tensor blocks after RHT show *heavier tails* than Beta(15.5, 15.5)
predicts (attention sinks, RoPE artefacts, layer-1 dominance). A Lloyd-Max
codebook fitted against the empirical distribution typically reduces MSE by
~10-25%, which maps to ~2-4% PPL improvement -- with zero runtime cost since
only the centroid table changes.

Two operating modes:
    1. --mode synthetic  (default)
       Sample directly from Beta(alpha, alpha) with user-supplied alpha
       (default alpha = (QK_KTQ-1)/2 = 15.5). Use a smaller alpha to model
       heavier tails -- e.g. alpha=12 widens the distribution noticeably.

    2. --mode samples
       Load pre-extracted post-RHT samples from a `.bin` file (float32,
       packed contiguously). This is what tq_calibrate_outliers.py emits.
       The script will fit a symmetric Beta(alpha, alpha) to those samples
       (matching their variance) AND additionally run Lloyd-Max directly on
       the raw samples for the *most* faithful codebook.

Output: a C header `ggml/include/ggml-common-codebook-override.h` defining
`PQ_CODEBOOK_*BIT_CALIBRATED`. Include path is opt-in via the CMake flag
`-DPQ_CODEBOOK_USE_CALIBRATED=ON` (see docs/calibrating-ktq.md).

Usage:
    # Pure synthetic, theoretical Beta(15.5, 15.5)
    python3 scripts/calibrate_ktq_codebook.py \
        --mode synthetic --alpha 15.5 --bits 2 3 4 \
        --output ggml/include/ggml-common-codebook-override.h

    # Heavier-tailed assumption (alpha=12 ~ realistic for Qwen3 mid layers)
    python3 scripts/calibrate_ktq_codebook.py \
        --mode synthetic --alpha 12.0 --bits 2 3 4 \
        --output ggml/include/ggml-common-codebook-override.h

    # From extracted V samples (re-uses extract_v_samples.py output)
    python3 scripts/calibrate_ktq_codebook.py \
        --mode samples --samples vcache-qwen35-27b.bin --head-dim 128 \
        --bits 2 3 4 \
        --output ggml/include/ggml-common-codebook-override.h
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# Must match QK_KTQ in ggml/src/ggml-common.h
QK_KTQ = 32

# Shipped codebooks (must match ggml/src/ggml-quants.c, lines ~5762/5766/6168).
# These are stored at "Gaussian-like" scale; runtime applies cb_scale = 1/sqrt(QK_KTQ).
SHIPPED_CODEBOOKS = {
    2: np.array([-1.489560, -0.451428,  0.451428,  1.489560], dtype=np.float64),
    3: np.array([-2.071926, -1.314996, -0.745325, -0.242405,
                  0.242405,  0.745325,  1.314996,  2.071926], dtype=np.float64),
    4: np.array([-2.732590, -2.069017, -1.618046, -1.256231,
                 -0.942340, -0.656759, -0.388048, -0.128395,
                  0.128395,  0.388048,  0.656759,  0.942340,
                  1.256231,  1.618046,  2.069017,  2.732590], dtype=np.float64),
}


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

def gen_beta_samples(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate samples from Beta(alpha, alpha) mapped to [-1, 1], then rescaled
    to unit-variance domain to align with the shipped codebook scale.

    The shipped codebooks live in a "1/sqrt(QK_KTQ) is applied at runtime" frame,
    which means they are quantizers for the *unscaled* coordinate of a vector
    that has been normalized away. Concretely: the per-coordinate marginal is
    Beta(15.5, 15.5) on [-1, 1], variance = 1 / (2*alpha + 1). To present the
    quantizer with samples whose variance equals the shipped codebook's design
    variance (~1.0), divide by std.
    """
    raw = rng.beta(alpha, alpha, size=n)               # in [0, 1]
    coord = 2.0 * raw - 1.0                            # in [-1, 1]
    # Rescale so std matches the canonical quantizer domain (std ~= 1).
    std = coord.std()
    if std < 1e-12:
        return coord
    return coord / std


def fit_alpha_from_samples(samples: np.ndarray) -> float:
    """Method-of-moments fit: for symmetric Beta(alpha, alpha) on [-1,1],
    Var = 1 / (2*alpha + 1)  =>  alpha = (1/Var - 1) / 2.

    `samples` is expected to be in the canonical (unit-std-ish) domain. We first
    rescale to [-1, 1] using max(|samples|), then fit moments. This is biased
    for heavy tails, but stable on small N.
    """
    s = np.asarray(samples, dtype=np.float64)
    smax = np.max(np.abs(s))
    if smax < 1e-12:
        return float("inf")
    s_unit = s / smax  # roughly in [-1, 1]
    var = float(s_unit.var())
    if var <= 0 or var >= 0.5:
        # Variance out of range for a symmetric Beta on [-1,1]; clip
        return 0.5
    alpha = (1.0 / var - 1.0) / 2.0
    return float(alpha)


def load_samples_bin(path: str, head_dim: int) -> np.ndarray:
    """Load packed float32 samples and reshape to [N, head_dim]."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % head_dim != 0:
        raise ValueError(
            f"Sample count {raw.size} not divisible by head-dim {head_dim}"
        )
    return raw.reshape(-1, head_dim).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Lloyd-Max
# ---------------------------------------------------------------------------

def lloyd_max(
    samples: np.ndarray,
    n_levels: int,
    init: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[np.ndarray, dict]:
    """Scalar Lloyd-Max on `samples`. Returns (centroids, info).

    Uses sorted-search assignment for O(N log K) per iteration; converges when
    the largest centroid shift drops below `tol`.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    if init is None:
        # Quantile init -- robust and fast
        qs = np.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
        c = np.quantile(s, qs)
    else:
        c = np.array(init, dtype=np.float64).copy()
        if c.size != n_levels:
            raise ValueError("init centroid count != n_levels")
    c.sort()

    prev_mse = float("inf")
    for it in range(max_iter):
        # Boundaries between centroids (decision thresholds)
        bounds = (c[:-1] + c[1:]) * 0.5
        # Each sample -> nearest centroid index
        idx = np.searchsorted(bounds, s)
        # Update centroids = mean of assigned samples
        new_c = np.empty_like(c)
        # Accumulate via bincount weighted sum (fast, vectorized)
        sums = np.bincount(idx, weights=s, minlength=n_levels)
        counts = np.bincount(idx, minlength=n_levels).astype(np.float64)
        empty = counts == 0
        new_c[~empty] = sums[~empty] / counts[~empty]
        # Heuristic: if a cell is empty, leave it where it was (rare for
        # well-conditioned data with quantile init).
        new_c[empty] = c[empty]
        new_c.sort()

        shift = float(np.max(np.abs(new_c - c)))
        c = new_c

        # MSE for diagnostic
        recon = c[idx]
        mse = float(np.mean((s - recon) ** 2))

        if shift < tol or abs(prev_mse - mse) < tol * 1e-3:
            return c, {"iters": it + 1, "mse": mse, "final_shift": shift}
        prev_mse = mse

    return c, {"iters": max_iter, "mse": prev_mse, "final_shift": shift}


def codebook_mse(samples: np.ndarray, codebook: np.ndarray) -> float:
    """MSE of nearest-centroid encoding under a fixed codebook."""
    s = np.asarray(samples, dtype=np.float64).ravel()
    cb = np.sort(np.asarray(codebook, dtype=np.float64))
    bounds = (cb[:-1] + cb[1:]) * 0.5
    idx = np.searchsorted(bounds, s)
    recon = cb[idx]
    return float(np.mean((s - recon) ** 2))


# ---------------------------------------------------------------------------
# Header emission
# ---------------------------------------------------------------------------

def fmt_array(c: Sequence[float]) -> str:
    return "{ " + ", ".join(f"{v:+.6f}f" for v in c) + " }"


def emit_header(
    output_path: str,
    codebooks: dict[int, np.ndarray],
    meta: dict,
) -> None:
    lines = []
    lines.append("// Auto-generated by scripts/calibrate_ktq_codebook.py")
    lines.append("// DO NOT EDIT BY HAND -- re-run the calibration script instead.")
    lines.append("//")
    for k, v in meta.items():
        lines.append(f"// {k}: {v}")
    lines.append("//")
    lines.append("// To activate, build with `-DPQ_CODEBOOK_USE_CALIBRATED=ON`.")
    lines.append("// See docs/calibrating-ktq.md for details.")
    lines.append("")
    lines.append("#ifndef PQ_CODEBOOK_OVERRIDE_H")
    lines.append("#define PQ_CODEBOOK_OVERRIDE_H")
    lines.append("")
    for bits in sorted(codebooks.keys()):
        cb = codebooks[bits]
        lines.append(
            f"#define PQ_CODEBOOK_{bits}BIT_CALIBRATED  {fmt_array(cb)}"
        )
    lines.append("")
    lines.append("#endif // PQ_CODEBOOK_OVERRIDE_H")
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Calibration-aware Lloyd-Max codebook generator for KTQ/VTQ."
    )
    p.add_argument("--mode", choices=["synthetic", "samples"], default="synthetic",
                   help="`synthetic`: draw from Beta(alpha, alpha); "
                        "`samples`: load post-RHT samples from a .bin file.")
    p.add_argument("--alpha", type=float, default=(QK_KTQ - 1) / 2.0,
                   help="Beta shape parameter for synthetic mode "
                        f"(default = (QK_KTQ-1)/2 = {(QK_KTQ-1)/2:.1f}).")
    p.add_argument("--n-samples", type=int, default=1_000_000,
                   help="Number of samples to draw / use (default 1M).")
    p.add_argument("--samples", type=str, default=None,
                   help="Path to packed float32 sample file (mode=samples).")
    p.add_argument("--head-dim", type=int, default=128,
                   help="Head dim used to reshape the samples file.")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4],
                   choices=[2, 3, 4], help="Bit widths to calibrate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str,
                   default="ggml/include/ggml-common-codebook-override.h")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--model-hint", type=str, default="generic")
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Acquire samples
    # ------------------------------------------------------------------
    fitted_alpha: float | None = None
    sample_source: str
    if args.mode == "synthetic":
        if args.alpha <= 0:
            print(f"ERROR: alpha must be > 0, got {args.alpha}", file=sys.stderr)
            return 1
        print(f"[synthetic] Drawing {args.n_samples:,} samples from "
              f"Beta(alpha={args.alpha:.4f}, beta={args.alpha:.4f})")
        samples = gen_beta_samples(args.alpha, args.n_samples, rng)
        sample_source = f"synthetic Beta({args.alpha:.4f}, {args.alpha:.4f})"
    else:
        if not args.samples:
            print("ERROR: --mode samples requires --samples PATH", file=sys.stderr)
            return 1
        if not os.path.exists(args.samples):
            print(f"ERROR: samples file not found: {args.samples}", file=sys.stderr)
            return 1
        print(f"[samples] Loading {args.samples} (head_dim={args.head_dim})")
        flat = load_samples_bin(args.samples, args.head_dim).ravel()
        if flat.size > args.n_samples:
            sub = rng.choice(flat.size, size=args.n_samples, replace=False)
            flat = flat[sub]
        # Normalize to unit-std for codebook frame parity with the shipped values.
        std = flat.std()
        if std > 1e-12:
            flat = flat / std
        samples = flat
        fitted_alpha = fit_alpha_from_samples(samples)
        sample_source = (
            f"empirical samples (N={samples.size:,}, fitted alpha={fitted_alpha:.4f})"
        )
        print(f"[samples] Method-of-moments fit: Beta(alpha={fitted_alpha:.4f}, "
              f"beta={fitted_alpha:.4f})")

    # Diagnostic moments
    mu = float(samples.mean())
    sigma = float(samples.std())
    if HAVE_SCIPY:
        kurt = float(_scipy_stats.kurtosis(samples, fisher=True))
    else:
        m = samples - mu
        kurt = float(np.mean(m**4) / (np.mean(m**2) ** 2) - 3.0)
    print(f"  moments: mean={mu:+.4e}  std={sigma:.4f}  excess-kurtosis={kurt:+.4f}")

    # ------------------------------------------------------------------
    # 2. Lloyd-Max per bit width
    # ------------------------------------------------------------------
    new_codebooks: dict[int, np.ndarray] = {}
    print()
    print(f"{'bits':>5} {'iters':>6} {'shift':>10} {'MSE_old':>12} "
          f"{'MSE_new':>12} {'reduction':>10}")
    print("-" * 64)
    for bits in sorted(set(args.bits)):
        n_levels = 1 << bits
        shipped = SHIPPED_CODEBOOKS[bits]
        mse_old = codebook_mse(samples, shipped)
        cb_new, info = lloyd_max(
            samples, n_levels, init=shipped, max_iter=args.max_iter
        )
        mse_new = info["mse"]
        # Force exact symmetry: the design distribution is symmetric, so a
        # symmetric codebook minimizes worst-case bias and also matches the
        # shipped layout. We average c_i with -c_{n-1-i}.
        sym = 0.5 * (cb_new - cb_new[::-1])
        cb_new = sym
        mse_new_sym = codebook_mse(samples, cb_new)
        reduction = (mse_old - mse_new_sym) / mse_old * 100.0 if mse_old > 0 else 0.0
        new_codebooks[bits] = cb_new
        print(f"{bits:>5} {info['iters']:>6} {info['final_shift']:>10.2e} "
              f"{mse_old:>12.6e} {mse_new_sym:>12.6e} {reduction:>9.2f}%")

    # ------------------------------------------------------------------
    # 3. Emit header
    # ------------------------------------------------------------------
    meta = {
        "mode": args.mode,
        "model_hint": args.model_hint,
        "sample_source": sample_source,
        "n_samples": int(samples.size),
        "QK_KTQ": QK_KTQ,
        "alpha_input": f"{args.alpha:.4f}",
        "alpha_fitted_MoM": f"{fitted_alpha:.4f}" if fitted_alpha is not None else "n/a",
        "moments": f"mean={mu:+.4e} std={sigma:.4f} excess_kurt={kurt:+.4f}",
        "seed": args.seed,
    }
    emit_header(args.output, new_codebooks, meta)
    elapsed = time.time() - t0
    print()
    print(f"Wrote {args.output}")
    print(f"Bits emitted: {sorted(new_codebooks.keys())}")
    print(f"Wall time:    {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
