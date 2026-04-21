#!/usr/bin/env python3
"""VTQ_OUT Outlier-Channel Calibration.

Consumes packed V-samples (output of extract_v_samples.py) and identifies
the top-N outlier channels per layer after RHT. Outputs a JSON file
describing which channels need higher bit precision.

Usage:
    python3 tq_calibrate_outliers.py \
        --samples vcache-qwen35-27b.bin \
        --head-dim 128 \
        --n-outliers 32 \
        --output outlier-mask-qwen35-27b.json

Output format:
    {
        "model_hint": "qwen35-27b",
        "head_dim": 128,
        "n_outliers": 32,
        "outlier_channels": [3, 17, 42, ...],  # global (or per-layer if --per-layer)
        "stats": {"max_var_ratio": 5.2, "outlier_var_mean": 3.1, "regular_var_mean": 0.92}
    }
"""
import argparse
import json
import numpy as np
import os
import sys


def load_samples(path: str, head_dim: int) -> np.ndarray:
    """Load packed float32 V-samples, reshape to [N, head_dim]."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % head_dim != 0:
        raise ValueError(f"Sample count {raw.size} not divisible by head_dim {head_dim}")
    return raw.reshape(-1, head_dim)


def calibrate(samples: np.ndarray, n_outliers: int) -> dict:
    """Identify top-n-outlier channels by post-sample variance.

    NOTE: samples are expected to already be post-RHT (extract_v_samples.py
    applies RHT). If they're raw V, this still works but outlier-channel
    candidates are less stable.
    """
    N, d = samples.shape
    channel_var = samples.var(axis=0)
    order = np.argsort(channel_var)[::-1]  # highest variance first
    outlier_idx = np.sort(order[:n_outliers])
    regular_idx = np.sort(order[n_outliers:])

    stats = {
        "n_samples": int(N),
        "head_dim": int(d),
        "max_var_ratio": float(channel_var.max() / channel_var.min()),
        "outlier_var_mean": float(channel_var[outlier_idx].mean()),
        "regular_var_mean": float(channel_var[regular_idx].mean()),
        "outlier_var_min": float(channel_var[outlier_idx].min()),
        "regular_var_max": float(channel_var[regular_idx].max()),
        "separation_quality": float(channel_var[outlier_idx].min() /
                                    channel_var[regular_idx].max()),
    }
    return {
        "outlier_channels": outlier_idx.tolist(),
        "stats": stats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="Packed float32 samples from extract_v_samples.py")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--n-outliers", type=int, default=32, help="Number of outlier channels")
    p.add_argument("--model-hint", default="unknown")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not os.path.exists(args.samples):
        print(f"ERROR: samples file not found: {args.samples}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading samples from {args.samples}...")
    samples = load_samples(args.samples, args.head_dim)
    print(f"Loaded {samples.shape[0]} samples × {samples.shape[1]} channels")

    result = calibrate(samples, args.n_outliers)
    result["model_hint"] = args.model_hint
    result["head_dim"] = args.head_dim
    result["n_outliers"] = args.n_outliers

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nOutlier-Channel Calibration Result:")
    print(f"  Model:                 {args.model_hint}")
    print(f"  head_dim:              {args.head_dim}")
    print(f"  n_outliers:            {args.n_outliers}")
    print(f"  Outlier indices (first 10): {result['outlier_channels'][:10]}...")
    print(f"\nVariance Statistics:")
    print(f"  Outlier channel mean var:  {result['stats']['outlier_var_mean']:.4f}")
    print(f"  Regular channel mean var:  {result['stats']['regular_var_mean']:.4f}")
    print(f"  Max/Min variance ratio:    {result['stats']['max_var_ratio']:.2f}x")
    print(f"  Separation quality (outlier_min/regular_max): {result['stats']['separation_quality']:.2f}")
    print(f"\n  {'GOOD' if result['stats']['separation_quality'] > 1.2 else 'WEAK'} separation")
    print(f"  → {'Outlier split is worthwhile' if result['stats']['separation_quality'] > 1.2 else 'Consider fewer/more outliers or uniform quant'}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
