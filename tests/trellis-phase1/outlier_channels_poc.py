"""Outlier-Channel-Split PoC (Idea 2 from v6 brainstorm)

Paper section 4.3, line 1095-1103:
    "32 outlier channels at 3 bits, 96 regular at 2 bits = effective 2.5 bpw"

Premise: After RHT rotation, channels SHOULD be uniform Gaussian.
In practice, real LLM activations have outlier channels that survive rotation
(especially for first 4 layers / attention sinks).

This script:
1. Generate realistic V-cache vectors (heavy-tailed mixture of Gaussians)
2. Apply RHT
3. Measure per-channel variance after rotation
4. Identify top-N outlier channels
5. Quantize: outliers @ b+1 bits, regular @ b bits
6. Measure attention error vs uniform b-bit baseline
"""
import numpy as np
from tqprod_poc import (
    lloyd_max_gaussian_centroids, quant_mse, dequant_mse, random_rotation
)


def make_realistic_v(N, d, n_outlier_channels, seed):
    """Heavy-tailed V-cache simulation. Some channels have 5-10x higher variance."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((N, d))
    # Inject outlier channels
    outlier_idx = rng.choice(d, n_outlier_channels, replace=False)
    V[:, outlier_idx] *= rng.uniform(3.0, 8.0, n_outlier_channels)
    # Normalize per row
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, outlier_idx


def quantize_with_outliers(V, Pi, centroids_outlier, centroids_regular,
                           outlier_channels):
    """Apply different quantizers to outlier vs regular channels.
    Per-channel scaling: each column normalized by its std before quantization
    (matches real VTQ block-scale that ggml uses)."""
    N, d = V.shape
    Y = V @ Pi.T

    # Per-channel scale (this is what real impl needs to store)
    scale = Y.std(axis=0) + 1e-9

    Y_quant = np.empty_like(Y)
    is_outlier = np.zeros(d, dtype=bool)
    is_outlier[outlier_channels] = True

    out_cols = Y[:, is_outlier] / scale[is_outlier]
    edges_o = (centroids_outlier[:-1] + centroids_outlier[1:]) / 2
    idx_o = np.searchsorted(edges_o, out_cols)
    Y_quant[:, is_outlier] = centroids_outlier[idx_o] * scale[is_outlier]

    reg_cols = Y[:, ~is_outlier] / scale[~is_outlier]
    edges_r = (centroids_regular[:-1] + centroids_regular[1:]) / 2
    idx_r = np.searchsorted(edges_r, reg_cols)
    Y_quant[:, ~is_outlier] = centroids_regular[idx_r] * scale[~is_outlier]

    return Y_quant @ Pi


def quantize_uniform(V, Pi, centroids):
    """Apply uniform b-bit codebook to all channels (with per-channel scaling
    to match real VTQ block-scale behavior)."""
    Y = V @ Pi.T
    scale = Y.std(axis=0) + 1e-9
    Y_norm = Y / scale
    edges = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(edges, Y_norm)
    Y_quant = centroids[idx] * scale
    return Y_quant @ Pi


def attention_error(V_ref, V_quant, Q):
    """Compute cosine-error of attention output, weighted by softmax(Q·K)."""
    N = V_ref.shape[0]
    K = V_ref  # use V as K for simplicity (real test would use separate K)
    scores = (Q @ K.T) / np.sqrt(Q.shape[0])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    O_ref = weights @ V_ref
    O_quant = weights @ V_quant
    return 1 - (O_ref @ O_quant) / (np.linalg.norm(O_ref) * np.linalg.norm(O_quant))


def run_experiment(d=128, N=1024, n_trials=20, seed=0, n_outlier_channels=8):
    rng = np.random.default_rng(seed)

    print(f"\n{'='*78}")
    print(f"Outlier-Channel-Split: {n_outlier_channels} channels @ b+1 bits, rest @ b bits")
    print(f"d={d}, N={N}, {n_outlier_channels} outlier channels (real outliers)")
    print(f"{'='*78}\n")

    print(f"{'b':>3} {'Uniform b':>14} {'Mixed (b/b+1)':>16} {'Effective bpw':>14} {'Improvement':>14}")

    for b in (2, 3, 4):
        cb_b = lloyd_max_gaussian_centroids(b)
        cb_bp1 = lloyd_max_gaussian_centroids(b + 1)
        # NOTE: scale by sqrt(d) is canceled by rotation, so use unscaled centroids here

        err_uniform = []
        err_mixed = []
        for trial in range(n_trials):
            Pi = random_rotation(d, seed=1000 + trial)
            V, true_outliers = make_realistic_v(N, d, n_outlier_channels, seed=trial)

            # Per-channel variance after rotation (offline calibration)
            Y = V @ Pi.T
            channel_var = Y.var(axis=0)
            calibrated_outliers = np.argsort(channel_var)[-n_outlier_channels:]

            V_uniform = quantize_uniform(V, Pi, cb_b)
            V_mixed = quantize_with_outliers(V, Pi, cb_bp1, cb_b, calibrated_outliers)

            Q = rng.standard_normal(d); Q /= np.linalg.norm(Q)
            err_uniform.append(attention_error(V, V_uniform, Q))
            err_mixed.append(attention_error(V, V_mixed, Q))

        eff_bpw = (n_outlier_channels * (b + 1) + (d - n_outlier_channels) * b) / d
        eu, em = np.mean(err_uniform), np.mean(err_mixed)
        improvement = (eu - em) / eu * 100
        print(f"{b:>3} {eu:>14.3e} {em:>16.3e} {eff_bpw:>13.3f} {improvement:>13.1f}%")


if __name__ == '__main__':
    # Match Paper config: 8 of 128 channels outlier
    run_experiment(d=128, N=1024, n_outlier_channels=8)
    # Match Paper Table 1: 32 of 128 = 25% outlier (their 2.5 bpw config)
    run_experiment(d=128, N=1024, n_outlier_channels=32)
