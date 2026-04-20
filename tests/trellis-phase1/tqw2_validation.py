#!/usr/bin/env python3
"""
TQW2 weight quantization validation vs IQ2_XXS baseline.

Numerical experiment (Python-only, numpy/sklearn) to decide whether to
green-light a CUDA sprint for TurboQuant weight quantization (TQW2).

Compares:
  - IQ2_XXS-style baseline: k-means over 8-D sub-vectors (256 codewords -> 2.0625 bpw).
  - TQW2 K=2 candidate:     per-group RHT (Philox-seeded signs + FWHT) then
                            1-D Lloyd-Max 4-centroid on rotated marginals (2 bpw).
  - TQW2 K=3 candidate:     same, 8-centroid (3 bpw).

RHT seed derivation mirrors `kktq_derive_seed()` (FNV-1a) and `tq_random_signs()`
(Philox 6-round) in ggml-quants.c, so signs match exactly what the CUDA impl
would produce.

Decision gate:
  if TQW2_K2 MSE <= IQ2_XXS_MSE * 0.9 for ALL tensor types -> GREEN.
  else RED.

Outputs:
  - CSV: per (tensor, group_size, quant) MSE
  - stdout: summary table + verdict
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import warnings
import numpy as np

# Philox/FNV-1a rely on uint32 wrap-around, which numpy warns about.
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "gguf-py"))


# ----------------------------------------------------------------------------
# Philox 6-round (matches ggml-quants.c:5509)
# ----------------------------------------------------------------------------
def philox_6r(counter, key):
    """Vectorized; inputs uint32 arrays (same shape). Returns uint32 lo."""
    lo = np.asarray(counter, dtype=np.uint32).copy()
    hi = np.asarray(key,     dtype=np.uint32).copy()
    M  = np.uint64(0xD2511F53)
    W  = np.uint32(0x9E3779B9)
    for i in range(6):
        lo_old = lo.copy()
        prod   = lo_old.astype(np.uint64) * M
        lo_hi  = (prod >> np.uint64(32)).astype(np.uint32)
        lo     = lo_hi ^ hi ^ (W * np.uint32(i + 1))
        hi     = (lo_old.astype(np.uint64) * M).astype(np.uint32)  # low32
    return lo


def derive_seed(block_index):
    """FNV-1a over 4 low bytes, returns uint16 (matches kktq_derive_seed)."""
    h = np.uint32(2166136261)
    P = np.uint32(16777619)
    bi = np.uint32(block_index & 0xFFFFFFFF)
    for shift in (0, 8, 16, 24):
        h = h ^ np.uint32((bi >> np.uint32(shift)) & np.uint32(0xFF))
        h = h * P
    return int(h) & 0xFFFF


def tq_random_signs(seed, n):
    idx = np.arange(n, dtype=np.uint32)
    key = np.full(n, seed, dtype=np.uint32)
    raw = philox_6r(idx, key)
    return np.where((raw & 1) == 1, 1.0, -1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (in-place, normalized 1/sqrt(n))
# ----------------------------------------------------------------------------
def fwht(x):
    """In-place FWHT along last axis. x must be float32, last dim = power of 2."""
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, "n must be power of 2"
    h = 1
    y = x.copy()
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*y.shape[:-3], n)
        h <<= 1
    y *= (1.0 / np.sqrt(n))
    return y


def rht_forward(x, seeds):
    """x: [B, n], seeds: [B] uint16. Returns RHT(x)."""
    B, n = x.shape
    signs = np.stack([tq_random_signs(int(seeds[i]), n) for i in range(B)], axis=0)
    return fwht(x * signs), signs


def rht_inverse(y, signs):
    return fwht(y) * signs


# ----------------------------------------------------------------------------
# Quantizers
# ----------------------------------------------------------------------------
def lloyd_max_1d(samples, K, n_iter=40, seed=0):
    """1-D Lloyd-Max on symmetric distribution. Returns K centroids (sorted)."""
    rng = np.random.default_rng(seed)
    # Init: quantiles
    qs = np.linspace(0.5 / K, 1 - 0.5 / K, K)
    c  = np.quantile(samples, qs).astype(np.float32)
    for _ in range(n_iter):
        # Assign
        d  = np.abs(samples[:, None] - c[None, :])
        a  = np.argmin(d, axis=1)
        # Update
        for k in range(K):
            mask = (a == k)
            if mask.any():
                c[k] = samples[mask].mean()
    c.sort()
    return c


def quantize_1d(x, centroids):
    """Nearest-centroid assignment. Returns reconstruction."""
    d = np.abs(x[..., None] - centroids[None, ...])
    a = np.argmin(d, axis=-1)
    return centroids[a]


def tqw2_quantize_group(group, K, seed):
    """Apply RHT, 1-D Lloyd-Max K-centroid on rotated coeffs, inverse RHT.
    Per-group norm is preserved (scale factor).
    group: [n] float32. Returns reconstructed [n]."""
    n = group.shape[0]
    norm = np.linalg.norm(group) + 1e-30
    x_hat = group / norm
    y, signs = rht_forward(x_hat[None, :], np.array([seed], dtype=np.uint16))
    y = y[0]
    # Fit centroids on THIS group (could be shared globally, but fit-per-group
    # is what 1-D Lloyd-Max would do if given local data; for fairness with
    # IQ2 baseline which uses global codebook, we use a pre-fit codebook instead).
    raise NotImplementedError("use tqw2_quantize_batch for efficiency")


def tqw2_quantize_batch(groups, K, codebook):
    """Batch RHT + quantize with shared codebook, inverse RHT.
    groups: [B, n] float32. Returns reconstructions [B, n]."""
    B, n = groups.shape
    norms = np.linalg.norm(groups, axis=1, keepdims=True) + 1e-30
    x_hat = groups / norms
    seeds = np.array([derive_seed(i) for i in range(B)], dtype=np.uint16)
    y, signs = rht_forward(x_hat, seeds)
    y_q = quantize_1d(y, codebook)
    rec_hat = rht_inverse(y_q, signs)
    return rec_hat * norms


def fit_tqw_codebook(groups, K, max_samples=200_000, seed=0):
    """Fit shared 1-D Lloyd-Max codebook on RHT-rotated samples."""
    B, n = groups.shape
    norms = np.linalg.norm(groups, axis=1, keepdims=True) + 1e-30
    x_hat = groups / norms
    seeds = np.array([derive_seed(i) for i in range(B)], dtype=np.uint16)
    y, _ = rht_forward(x_hat, seeds)
    flat = y.reshape(-1)
    if flat.size > max_samples:
        rng = np.random.default_rng(seed)
        flat = rng.choice(flat, max_samples, replace=False)
    return lloyd_max_1d(flat, K, n_iter=30, seed=seed)


# ----------------------------------------------------------------------------
# IQ2_XXS-style baseline: k-means on 8-D sub-vectors, 256 codewords
# ----------------------------------------------------------------------------
def iq2_baseline_quantize(groups, n_codewords=256, subvec_dim=8,
                          max_fit=50_000, seed=0):
    """Split each group into 8-D subvectors, learn single shared codebook
    via mini-batch k-means on a sample, reconstruct.
    Per-group scale (max-abs) is applied (like IQ2_XXS's d scale).
    Returns reconstructions [B, n]."""
    B, n = groups.shape
    assert n % subvec_dim == 0, f"group size {n} not divisible by {subvec_dim}"

    # Per-group scale (max-abs, mimics IQ2_XXS's d = max(|x|) style)
    scales = np.max(np.abs(groups), axis=1, keepdims=True) + 1e-30
    x_s    = groups / scales

    # Reshape to subvectors
    sv = x_s.reshape(B, n // subvec_dim, subvec_dim)
    sv_flat = sv.reshape(-1, subvec_dim)

    # Fit codebook via k-means
    rng = np.random.default_rng(seed)
    if sv_flat.shape[0] > max_fit:
        sample_idx = rng.choice(sv_flat.shape[0], max_fit, replace=False)
        sample = sv_flat[sample_idx]
    else:
        sample = sv_flat

    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=n_codewords, random_state=seed,
                             batch_size=4096, n_init=3, max_iter=50)
        km.fit(sample)
        codebook = km.cluster_centers_.astype(np.float32)
    except ImportError:
        # Fallback: random init + Lloyd iterations
        idx = rng.choice(sample.shape[0], n_codewords, replace=False)
        codebook = sample[idx].astype(np.float32)
        for _ in range(20):
            # assign
            d = np.linalg.norm(sample[:, None, :] - codebook[None, :, :], axis=2)
            a = np.argmin(d, axis=1)
            for k in range(n_codewords):
                m = (a == k)
                if m.any():
                    codebook[k] = sample[m].mean(axis=0)

    # Assign all subvectors
    # Chunked to save RAM
    rec = np.zeros_like(sv_flat)
    chunk = 8192
    for i in range(0, sv_flat.shape[0], chunk):
        d = np.linalg.norm(sv_flat[i:i+chunk, None, :] - codebook[None, :, :], axis=2)
        a = np.argmin(d, axis=1)
        rec[i:i+chunk] = codebook[a]
    rec = rec.reshape(B, n // subvec_dim, subvec_dim).reshape(B, n)
    return rec * scales


# ----------------------------------------------------------------------------
# Dequantize q8_0 from GGUFReader tensor.data
# ----------------------------------------------------------------------------
def dequantize_q8_0(raw_bytes, n_elements):
    """q8_0 block: [fp16 d][32 int8 qs]. Returns fp32 array of length n_elements."""
    QK = 32
    block_size = 2 + QK            # 2 bytes fp16 + 32 int8 = 34
    n_blocks   = n_elements // QK
    data       = np.frombuffer(raw_bytes, dtype=np.uint8)[:n_blocks * block_size]
    data       = data.reshape(n_blocks, block_size)
    d          = np.frombuffer(data[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    qs         = np.frombuffer(data[:, 2:].tobytes(), dtype=np.int8).reshape(n_blocks, QK).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


# ----------------------------------------------------------------------------
# Tensor extraction
# ----------------------------------------------------------------------------
TENSOR_PATTERNS = {
    "attn_q":   ".attn_q.weight",
    "attn_k":   ".attn_k.weight",
    "attn_v":   ".attn_v.weight",
    "ffn_gate": ".ffn_gate.weight",
    "ffn_up":   ".ffn_up.weight",
    "ffn_down": ".ffn_down.weight",
}


def extract_tensors(model_path, max_layers=4, max_elements_per_type=1_000_000):
    """Yield (type_name, flat_fp32_array) once we have enough per type."""
    import gguf
    reader = gguf.GGUFReader(model_path, "r")

    collected = defaultdict(list)
    counts    = defaultdict(int)

    for tensor in reader.tensors:
        name = tensor.name
        try:
            layer_idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        if layer_idx >= max_layers:
            continue

        ttype = None
        for k, pat in TENSOR_PATTERNS.items():
            if pat in name:
                ttype = k
                break
        if ttype is None:
            continue
        if counts[ttype] >= max_elements_per_type:
            continue

        qtype = tensor.tensor_type
        n_elements = int(np.prod(tensor.shape))

        if qtype == gguf.GGMLQuantizationType.Q8_0:
            raw = bytes(tensor.data)
            arr = dequantize_q8_0(raw, n_elements)
        elif qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = np.array(tensor.data, copy=False).astype(np.float32).reshape(-1)
        else:
            # Skip types we can't dequant here
            continue

        collected[ttype].append(arr)
        counts[ttype] += arr.size
        print(f"  {name}: qtype={qtype.name} shape={tensor.shape} -> {arr.size} elems (type {ttype} total {counts[ttype]})")

    for ttype, chunks in collected.items():
        yield ttype, np.concatenate(chunks)


# ----------------------------------------------------------------------------
# Synthetic fallback
# ----------------------------------------------------------------------------
def synthetic_weights(n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    # Laplace (matches transformer weight marginals)
    return rng.laplace(0.0, 0.12, size=n).astype(np.float32)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def evaluate(flat, ttype, group_sizes, max_groups=4000, seed=0):
    """For each group_size and quant method, compute MSE. Returns rows."""
    rows = []
    for gs in group_sizes:
        n_groups = min(flat.size // gs, max_groups)
        if n_groups < 32:
            continue
        g = flat[:n_groups * gs].reshape(n_groups, gs).astype(np.float32)

        t0 = time.time()
        rec_iq2  = iq2_baseline_quantize(g, n_codewords=256, subvec_dim=8, seed=seed)
        mse_iq2  = float(np.mean((g - rec_iq2) ** 2))
        t_iq2 = time.time() - t0

        t0 = time.time()
        cb_k2 = fit_tqw_codebook(g, K=4, seed=seed)
        rec_k2  = tqw2_quantize_batch(g, K=4, codebook=cb_k2)
        mse_k2  = float(np.mean((g - rec_k2) ** 2))
        t_k2 = time.time() - t0

        t0 = time.time()
        cb_k3 = fit_tqw_codebook(g, K=8, seed=seed)
        rec_k3  = tqw2_quantize_batch(g, K=8, codebook=cb_k3)
        mse_k3  = float(np.mean((g - rec_k3) ** 2))
        t_k3 = time.time() - t0

        print(f"  {ttype} gs={gs} n_groups={n_groups}: "
              f"iq2={mse_iq2:.6g} ({t_iq2:.1f}s) "
              f"tqw2_K2={mse_k2:.6g} ({t_k2:.1f}s) "
              f"tqw2_K3={mse_k3:.6g} ({t_k3:.1f}s)")

        rows.append({"tensor": ttype, "group_size": gs, "n_groups": n_groups,
                     "iq2_baseline": mse_iq2, "tqw2_K2": mse_k2, "tqw2_K3": mse_k3,
                     "improvement_K2_pct": 100.0 * (mse_iq2 - mse_k2) / mse_iq2,
                     "improvement_K3_pct": 100.0 * (mse_iq2 - mse_k3) / mse_iq2})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/claude/models/qwen3.5-0.8b-q8_0.gguf")
    ap.add_argument("--out-csv", default=os.path.join(SCRIPT_DIR, "results", "run20_tqw2_mse.csv"))
    ap.add_argument("--max-layers", type=int, default=4)
    ap.add_argument("--max-elements", type=int, default=300_000)
    ap.add_argument("--max-groups", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip model load, use synthetic Laplace weights")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Extract
    tensors = {}
    if args.synthetic or not os.path.exists(args.model):
        print(f"[synthetic mode] (model_exists={os.path.exists(args.model)})")
        for t in TENSOR_PATTERNS:
            tensors[t] = synthetic_weights(args.max_elements, seed=hash(t) & 0xFFFF)
    else:
        print(f"loading {args.model} ...")
        for ttype, arr in extract_tensors(args.model,
                                          max_layers=args.max_layers,
                                          max_elements_per_type=args.max_elements):
            tensors[ttype] = arr
            print(f"  collected {ttype}: {arr.size} elems, std={arr.std():.4f}, kurt={((arr-arr.mean())**4).mean()/((arr.var()+1e-30)**2):.2f}")

    if not tensors:
        print("No tensors collected. Falling back to synthetic.")
        for t in TENSOR_PATTERNS:
            tensors[t] = synthetic_weights(args.max_elements, seed=hash(t) & 0xFFFF)

    # Evaluate
    all_rows = []
    for ttype, flat in tensors.items():
        print(f"\n=== {ttype} ({flat.size} samples) ===")
        rows = evaluate(flat, ttype, group_sizes=[32, 256],
                        max_groups=args.max_groups)
        all_rows.extend(rows)

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tensor", "group_size", "n_groups",
                                          "iq2_baseline", "tqw2_K2", "tqw2_K3",
                                          "improvement_K2_pct", "improvement_K3_pct"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {args.out_csv}")

    # Summary + verdict
    print("\n================ SUMMARY ================")
    print(f"{'tensor':<10} {'gs':<5} {'iq2':>12} {'tqw2_K2':>12} {'tqw2_K3':>12} {'ΔK2%':>7} {'ΔK3%':>7}")
    per_tensor_k2_ratios = defaultdict(list)
    for r in all_rows:
        print(f"{r['tensor']:<10} {r['group_size']:<5} "
              f"{r['iq2_baseline']:>12.4g} {r['tqw2_K2']:>12.4g} {r['tqw2_K3']:>12.4g} "
              f"{r['improvement_K2_pct']:>7.1f} {r['improvement_K3_pct']:>7.1f}")
        per_tensor_k2_ratios[r["tensor"]].append(r["tqw2_K2"] / r["iq2_baseline"])

    # Decision gate: TQW2_K2 MSE <= IQ2 * 0.9 across ALL tensors (average group sizes)
    print("\n================ VERDICT ================")
    mean_ratios = {t: float(np.mean(rs)) for t, rs in per_tensor_k2_ratios.items()}
    all_pass = all(r <= 0.9 for r in mean_ratios.values())
    for t, r in mean_ratios.items():
        flag = "PASS" if r <= 0.9 else "FAIL"
        print(f"  {t:<10} TQW2_K2/IQ2 = {r:.3f} [{flag}]")

    if all_pass:
        print("\n[GREEN] proceed CUDA sprint — TQW2_K2 beats IQ2_XXS by >10% across all tensors")
    else:
        print("\n[RED] abandon TQW2 — does not beat IQ2_XXS at 2-bit equal bpw")

    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
