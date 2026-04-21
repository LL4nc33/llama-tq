"""VTQ{2,3,4}_1 roundtrip MSE benchmark on real post-RHT V-samples.

Uses the Lloyd-Max codebooks from ggml-quants.c (ported to Python)
to quantize-then-dequantize the real samples and measure MSE.

This gives us a fast proxy for PPL-improvement between bit-widths,
without needing a full llama-perplexity run.
"""
import numpy as np
import sys


# Codebooks from ggml-common.h (VTQ_CODEBOOK_{2,3,4}BIT)
# These are Lloyd-Max centroids for N(0, 1/d) distribution,
# scaled by 1/sqrt(d) in the actual impl via cb_scale.
VTQ_CODEBOOK_2BIT = np.array([-1.51096, -0.4534,  0.4534,  1.51096])
VTQ_CODEBOOK_3BIT = np.array([-2.15195, -1.34407, -0.75244, -0.245,
                              0.245, 0.75244, 1.34407, 2.15195])
VTQ_CODEBOOK_4BIT = np.array([-2.73321, -2.06942, -1.61813, -1.25635,
                              -0.94449, -0.66423, -0.40432, -0.15758,
                              0.15758, 0.40432, 0.66423, 0.94449,
                              1.25635, 1.61813, 2.06942, 2.73321])


def quantize_vtq(x: np.ndarray, codebook: np.ndarray, QK: int = 32) -> np.ndarray:
    """Replicate VTQ_1 quantize+dequantize with norm correction.
    x: [N, QK] already in post-RHT space.
    Returns reconstructed x (same shape)."""
    N, Q = x.shape
    assert Q == QK
    cb_scale = 1.0 / np.sqrt(QK)
    cb = codebook * cb_scale

    # Per-row norm
    norms = np.linalg.norm(x, axis=1, keepdims=True)  # [N, 1]
    mask = (norms > 1e-30).squeeze()

    # Normalize
    x_norm = np.where(norms > 1e-30, x / norms, 0)

    # Nearest-centroid (broadcast)
    edges = (cb[:-1] + cb[1:]) / 2
    idx = np.searchsorted(edges, x_norm)  # [N, QK]
    recon = cb[idx]

    # Norm correction (as in ggml-quants.c)
    recon_sq_norm = np.linalg.norm(recon, axis=1, keepdims=True)
    corrected_norms = np.where(recon_sq_norm > 1e-30,
                                norms / recon_sq_norm, norms)

    return recon * corrected_norms


def quantize_vtq_mixed(x: np.ndarray, QK: int = 32) -> np.ndarray:
    """VTQ_MIXED: positions {0,4,8,12,16,20,24,28} use VTQ_CODEBOOK_3BIT, rest use VTQ_CODEBOOK_2BIT.
    Per-block L2 norm + norm correction same as other VTQ_1 types."""
    N, Q = x.shape
    assert Q == QK
    cb_scale = 1.0 / np.sqrt(QK)
    cb_hi = VTQ_CODEBOOK_3BIT * cb_scale
    cb_lo = VTQ_CODEBOOK_2BIT * cb_scale

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = np.where(norms > 1e-30, x / norms, 0)

    edges_hi = (cb_hi[:-1] + cb_hi[1:]) / 2
    edges_lo = (cb_lo[:-1] + cb_lo[1:]) / 2

    is_hi = (np.arange(QK) % 4 == 0)

    recon = np.empty_like(x_norm)
    idx_hi = np.searchsorted(edges_hi, x_norm[:, is_hi])
    recon[:, is_hi] = cb_hi[idx_hi]
    idx_lo = np.searchsorted(edges_lo, x_norm[:, ~is_hi])
    recon[:, ~is_hi] = cb_lo[idx_lo]

    recon_sq = np.linalg.norm(recon, axis=1, keepdims=True)
    corrected = np.where(recon_sq > 1e-30, norms / recon_sq, norms)
    return recon * corrected


def benchmark(samples_path: str, QK: int = 32):
    """Load samples, benchmark each VTQ codebook."""
    raw = np.fromfile(samples_path, dtype=np.float32)
    N_total = raw.size
    if N_total % QK != 0:
        raw = raw[:N_total - (N_total % QK)]
    samples = raw.reshape(-1, QK)
    print(f"Loaded {samples.shape[0]} blocks of {QK} samples from {samples_path}")
    print(f"  per-sample mean: {samples.mean():.4f}, std: {samples.std():.4f}")
    print()

    print(f"{'type':>12} {'MSE':>12} {'rel MSE':>10} {'bpw':>6}")
    print("-" * 52)

    var_total = samples.var()

    for name, bits, cb in [("VTQ2_1", 2, VTQ_CODEBOOK_2BIT),
                           ("VTQ3_1", 3, VTQ_CODEBOOK_3BIT),
                           ("VTQ4_1", 4, VTQ_CODEBOOK_4BIT)]:
        recon = quantize_vtq(samples, cb, QK=QK)
        mse = ((samples - recon) ** 2).mean()
        rel_mse = mse / var_total
        bpw = bits + 16.0 / QK
        print(f"{name:>12} {mse:>12.5e} {rel_mse:>9.3%} {bpw:>6.2f}")

    # VTQ_MIXED: 8 hi + 24 lo slots
    recon = quantize_vtq_mixed(samples, QK=QK)
    mse = ((samples - recon) ** 2).mean()
    rel_mse = mse / var_total
    # 12 bytes / 32 samples = 3.0 bpw (with 1B straddle pad)
    print(f"{'VTQ_MIXED':>12} {mse:>12.5e} {rel_mse:>9.3%} {3.00:>6.2f}  ← new")

    print()
    print("Paper Theorem-1 per-coord: D_mse ≈ 0.117/d, 0.03/d, 0.009/d for b=2,3,4")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vcache-qwen35-27b.bin"
    benchmark(path, QK=32)
