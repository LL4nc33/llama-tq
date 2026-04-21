"""Compare random vs calibrated outlier-channel selection for VTQ_MIXED-style mixing.

Current VTQ_MIXED uses fixed stride-4 positions (0,4,8,...,28) — random w.r.t. data.

Paper Table 1 uses STATISTICALLY-CALIBRATED outlier channels. Question: on real
Qwen3.5-27B V-weights (post-RHT, Laplace-distributed), does calibration help?

Approach:
1. Load samples, reshape to blocks of QK=32
2. Compute per-position variance across all blocks (128 positions but QK=32...)
   Actually: per-position variance across blocks within the QK=32 window
3. Select top-8 highest-variance positions as "calibrated outliers"
4. Run VTQ_MIXED-style mix (8 @ 3-bit, 24 @ 2-bit) with random vs calibrated
"""
import numpy as np
import sys


VTQ_2BIT_LAPLACE = np.array([-1.810, -0.395, 0.395, 1.810])
VTQ_3BIT_GAUSS = np.array([-2.15195, -1.34407, -0.75244, -0.245,
                           0.245, 0.75244, 1.34407, 2.15195])


def vtq_mixed_roundtrip(samples, hi_positions):
    """Quantize blocks with 3-bit at hi_positions, 2-bit elsewhere."""
    N, QK = samples.shape
    cb_scale = 1.0 / np.sqrt(QK)
    cb_hi = VTQ_3BIT_GAUSS * cb_scale
    cb_lo = VTQ_2BIT_LAPLACE * cb_scale

    is_hi = np.zeros(QK, dtype=bool)
    is_hi[hi_positions] = True

    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    x_norm = np.where(norms > 1e-30, samples / norms, 0)

    recon = np.empty_like(x_norm)
    edges_hi = (cb_hi[:-1] + cb_hi[1:]) / 2
    edges_lo = (cb_lo[:-1] + cb_lo[1:]) / 2
    idx_hi = np.searchsorted(edges_hi, x_norm[:, is_hi])
    recon[:, is_hi] = cb_hi[idx_hi]
    idx_lo = np.searchsorted(edges_lo, x_norm[:, ~is_hi])
    recon[:, ~is_hi] = cb_lo[idx_lo]

    recon_sq = np.linalg.norm(recon, axis=1, keepdims=True)
    corrected = np.where(recon_sq > 1e-30, norms / recon_sq, norms)
    return recon * corrected


def bench(samples_path):
    raw = np.fromfile(samples_path, dtype=np.float32)
    QK = 32
    raw = raw[:raw.size - (raw.size % QK)]
    samples = raw.reshape(-1, QK)
    N = samples.shape[0]

    # Per-position variance across blocks (within QK=32 window)
    # Note: after D*H*D rotation these should be uniform; outliers within a block
    # are less common than outliers across blocks. Check anyway.
    pos_variance = samples.var(axis=0)
    print(f"Per-position variance in {QK}-window: "
          f"min={pos_variance.min():.4f} max={pos_variance.max():.4f} "
          f"ratio={pos_variance.max()/pos_variance.min():.2f}x")

    pos_order = np.argsort(pos_variance)[::-1]  # highest variance first
    top8_calibrated = np.sort(pos_order[:8])
    print(f"Top-8 calibrated positions: {top8_calibrated.tolist()}")

    var_total = samples.var()
    configs = [
        ("stride-4 (current VTQ_MIXED)", np.array([0, 4, 8, 12, 16, 20, 24, 28])),
        ("top-8 calibrated (by variance)", top8_calibrated),
        ("first-8 (0..7)", np.arange(8)),
        ("last-8 (24..31)", np.arange(24, 32)),
    ]
    # Random baselines
    rng = np.random.default_rng(42)
    for i in range(3):
        configs.append((f"random 8 (seed {42+i})",
                       np.sort(rng.choice(QK, 8, replace=False))))

    print(f"\n{'configuration':>36} {'MSE':>12} {'rel MSE':>10}")
    print("-" * 62)
    for name, positions in configs:
        recon = vtq_mixed_roundtrip(samples, positions)
        mse = ((samples - recon) ** 2).mean()
        rel = mse / var_total
        print(f"{name:>36} {mse:>12.5e} {rel:>9.3%}")

    print("\nReference (uniform):")
    print(f"{'VTQ2_1 (all 2-bit)':>36}   {'':>12} {13.252:>9.3f}%")
    print(f"{'VTQ3_1 (all 3-bit)':>36}   {'':>12} {3.069:>9.3f}%")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vcache-qwen35-27b.bin"
    bench(path)
