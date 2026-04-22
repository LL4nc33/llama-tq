"""Test Laplace vs Gaussian centroids for 3-bit and 4-bit VTQ codebooks.

Current ggml-quants.c:
- VTQ_CODEBOOK_2BIT: Laplace-optimal [-1.81, -0.395, 0.395, 1.81]
- VTQ_CODEBOOK_3BIT: Gaussian (PQ shared)
- VTQ_CODEBOOK_4BIT: Gaussian (PQ shared)

Hypothesis: Post-D*H*D rotation is Laplace-distributed, so 3/4-bit should
also benefit from Laplace-optimized centroids.

Method:
1. Fit Lloyd-Max centroids to Laplace distribution empirically
2. Run roundtrip bench with both codebook variants
3. Compare MSE
"""
import numpy as np
import sys


def lloyd_max_fit(data, k, n_iter=200):
    """Lloyd-Max k-means 1D."""
    data = np.sort(data)
    # Init with percentiles
    centroids = np.percentile(data, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(n_iter):
        edges = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(edges, data)
        new = np.array([data[idx == i].mean() if np.any(idx == i) else centroids[i]
                       for i in range(k)])
        if np.allclose(new, centroids, atol=1e-8):
            break
        centroids = new
    return np.sort(centroids)


def quantize_symmetric_1d(x, cb):
    """Quantize 2d array to nearest centroid in cb (shape [N,Q] → [N,Q])."""
    edges = (cb[:-1] + cb[1:]) / 2
    return cb[np.searchsorted(edges, x)]


def vtq_roundtrip(samples, cb):
    """VTQ-style quantize+dequantize with norm correction."""
    cb_scale = 1.0 / np.sqrt(samples.shape[1])
    cb_s = cb * cb_scale
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    x_norm = np.where(norms > 1e-30, samples / norms, 0)
    recon = quantize_symmetric_1d(x_norm, cb_s)
    recon_sq = np.linalg.norm(recon, axis=1, keepdims=True)
    corrected = np.where(recon_sq > 1e-30, norms / recon_sq, norms)
    return recon * corrected


def fit_and_bench(samples_path):
    # Load post-RHT samples
    raw = np.fromfile(samples_path, dtype=np.float32)
    QK = 32
    raw = raw[:raw.size - (raw.size % QK)]
    samples = raw.reshape(-1, QK)
    N = samples.shape[0]

    # Fit centroids empirically on full distribution (flattened, already post-RHT)
    flat = samples.flatten()
    # Normalize per-block first (like real VTQ): divide each sample by its block's L2 norm
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    flat_norm = (samples / (norms + 1e-30)).flatten()

    print(f"Loaded {N} blocks × {QK} samples = {flat.size} values")
    print(f"Raw stats: mean={flat.mean():.4f} std={flat.std():.4f} "
          f"kurt={(((flat - flat.mean())/flat.std())**4).mean() - 3:.2f}")
    print(f"Per-block-normalized: mean={flat_norm.mean():.4f} std={flat_norm.std():.4f}")

    # Existing codebooks from ggml-quants.c (pre-scaling by cb_scale=1/sqrt(QK))
    cb_2bit_laplace = np.array([-1.810, -0.395, 0.395, 1.810])
    cb_3bit_gauss = np.array([-2.15195, -1.34407, -0.75244, -0.245,
                              0.245, 0.75244, 1.34407, 2.15195])
    cb_4bit_gauss = np.array([-2.73321, -2.06942, -1.61813, -1.25635,
                              -0.94449, -0.66423, -0.40432, -0.15758,
                              0.15758, 0.40432, 0.66423, 0.94449,
                              1.25635, 1.61813, 2.06942, 2.73321])

    # Fit Laplace-optimal centroids empirically on per-block-normalized data.
    # Scale up by sqrt(QK) because existing codebooks are "unscaled" (will be multiplied
    # by cb_scale=1/sqrt(QK) in vtq_roundtrip).
    cb_3bit_fit = lloyd_max_fit(flat_norm, 8) * np.sqrt(QK)
    cb_4bit_fit = lloyd_max_fit(flat_norm, 16) * np.sqrt(QK)

    print("\nFitted Laplace-optimal centroids (empirical, multiplied by sqrt(QK)):")
    print(f"  3-bit: {cb_3bit_fit.tolist()}")
    print(f"  4-bit: {cb_4bit_fit.tolist()}")

    # Benchmark all variants
    var_total = samples.var()
    print(f"\n{'codebook':>30} {'MSE':>12} {'rel MSE':>10}")
    print("-" * 60)
    for name, cb in [
        ("2-bit Laplace (current)", cb_2bit_laplace),
        ("3-bit Gaussian (current)", cb_3bit_gauss),
        ("3-bit Laplace (fitted)", cb_3bit_fit),
        ("4-bit Gaussian (current)", cb_4bit_gauss),
        ("4-bit Laplace (fitted)", cb_4bit_fit),
    ]:
        recon = vtq_roundtrip(samples, cb)
        mse = ((samples - recon) ** 2).mean()
        rel = mse / var_total
        print(f"{name:>30} {mse:>12.5e} {rel:>9.3%}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vcache-qwen35-27b.bin"
    fit_and_bench(path)
