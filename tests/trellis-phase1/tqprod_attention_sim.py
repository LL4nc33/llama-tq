"""TQ_prod Attention-Aggregation PoC

Wenn Stage-1 minimalen Bias hat aber per-vector D_prod niedriger ist (siehe tqprod_poc.py),
muss man die kritische Frage stellen: was passiert bei Attention-Aggregation über N tokens?

Attention output O = Σ_t softmax(Q·K_t)·V_t

Wenn (Q·K_t̃) biased ist mit per-token bias b, dann akkumuliert sich der Fehler in der softmax-distribution.
Stage-1: bias konstant, akkumuliert linear über tokens
Stage-2: bias 0, error mittelt sich raus, akkumuliert √N

Hypothese: Bei N=4096 tokens (typischer KV-cache) sollte Stage-2 Vorteil sichtbar werden.
"""
import numpy as np
from tqprod_poc import (
    lloyd_max_gaussian_centroids, quant_mse, dequant_mse,
    quant_prod, dequant_prod, random_rotation, random_gaussian
)


def attention_test(d=128, N=4096, b=2, n_trials=20, seed=0):
    """Simulate attention output O = Σ softmax(Q·K)V comparing TQ_mse vs TQ_prod V-cache.

    Q is fp16 query, K is fp16 keys (we don't quantize K here, focus on V).
    V is quantized either by Stage-1 only (current VTQ) or Stage-2 (TQ_prod).
    """
    rng = np.random.default_rng(seed)
    centroids_b = lloyd_max_gaussian_centroids(b) / np.sqrt(d)
    centroids_bm1 = lloyd_max_gaussian_centroids(max(b - 1, 1)) / np.sqrt(d)

    err_s1_total = []
    err_s2_total = []

    for trial in range(n_trials):
        # Same Π for all tokens in a layer (per-model RHT seed)
        Pi = random_rotation(d, seed=1000 + trial)
        S = random_gaussian(d, seed=2000 + trial)

        # Generate Q, K, V (Q-K dotprod is fp16; V is what gets quantized)
        Q = rng.standard_normal((d,)); Q /= np.linalg.norm(Q)
        K = rng.standard_normal((N, d))
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.standard_normal((N, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        # Quantize all V with Stage-1
        V_s1 = np.array([dequant_mse(quant_mse(V[t], Pi, centroids_b), Pi, centroids_b)
                         for t in range(N)])

        # Quantize all V with Stage-2 (TQ_prod)
        V_s2 = np.empty_like(V)
        for t in range(N):
            idx, qjl, gamma = quant_prod(V[t], Pi, S, centroids_bm1)
            V_s2[t] = dequant_prod(idx, qjl, gamma, Pi, S, centroids_bm1)

        # Attention scores (Q·K^T)
        scores = Q @ K.T  # shape (N,)
        # Softmax with sqrt(d) normalization (standard attention)
        scores = scores / np.sqrt(d)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()

        # Reference output
        O_ref = weights @ V         # shape (d,)
        O_s1 = weights @ V_s1
        O_s2 = weights @ V_s2

        # Cosine similarity to reference (1 = perfect)
        err_s1 = 1 - (O_ref @ O_s1) / (np.linalg.norm(O_ref) * np.linalg.norm(O_s1))
        err_s2 = 1 - (O_ref @ O_s2) / (np.linalg.norm(O_ref) * np.linalg.norm(O_s2))
        err_s1_total.append(err_s1)
        err_s2_total.append(err_s2)

    return {
        'b': b, 'N': N,
        'err_s1_mean': np.mean(err_s1_total), 'err_s1_std': np.std(err_s1_total),
        'err_s2_mean': np.mean(err_s2_total), 'err_s2_std': np.std(err_s2_total),
    }


if __name__ == '__main__':
    print(f"\n{'='*72}")
    print(f"Attention-Output Cosine-Distance vs Reference")
    print(f"(lower = better, both methods at same bpw budget)")
    print(f"{'='*72}\n")

    print(f"{'b':>3} {'N':>6} {'TQ_mse (s1)':>22} {'TQ_prod (s2)':>22} {'s2/s1 ratio':>14}")
    for b in (2, 3, 4):
        for N in (256, 1024, 4096):
            r = attention_test(d=128, N=N, b=b, n_trials=10)
            ratio = r['err_s2_mean'] / max(r['err_s1_mean'], 1e-12)
            verdict = "← s2 better" if ratio < 0.95 else ("← s1 better" if ratio > 1.05 else "← equal")
            print(f"{b:>3} {N:>6} {r['err_s1_mean']:>10.2e} ± {r['err_s1_std']:>7.1e} "
                  f"{r['err_s2_mean']:>10.2e} ± {r['err_s2_std']:>7.1e} {ratio:>10.2f}x  {verdict}")
