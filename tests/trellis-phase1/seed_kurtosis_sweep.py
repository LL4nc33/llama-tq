#!/usr/bin/env python3
"""
Trick 3: Per-model RHT seed calibration.

Sweeps a global seed-salt XORed against the per-block hashed seed
and measures post-RHT marginal kurtosis. The 1-D Lloyd-Max codebook
is near-optimal only for Gaussian marginals (excess kurtosis = 0).
Minimizing mean kurtosis = picking the salt whose rotation best
Gaussianizes this specific model's V-cache distribution.

If the kurtosis variation across salts is below measurement noise,
there is nothing to gain from per-model calibration and the feature
should be abandoned. Otherwise the argmin salt becomes a GGUF
metadata tensor (one uint16) and is XORed into every block seed
at RHT time.

Usage:
    PYTHONPATH=gguf-py python3 tests/trellis-phase1/seed_kurtosis_sweep.py \
        --model /path/to/model.gguf \
        --n-salts 256 \
        --max-blocks 4096
"""

import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "gguf-py"))

import gguf

QK_VTQ2 = 512  # trellis-block size


def fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform, normalized by 1/sqrt(n).
    Operates along last axis, in place on a copy."""
    x = x.copy()
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i:i + h].copy()
            b = x[..., i + h:i + 2 * h].copy()
            x[..., i:i + h] = a + b
            x[..., i + h:i + 2 * h] = a - b
        h *= 2
    return x / np.sqrt(n)


def philox_6r(counter: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Philox 2x32 6-round, matches ggml-quants.c ktq_philox_6r.

    counter: shape [..., n], uint32
    key:     shape [..., 1] or broadcastable to counter, uint32
    returns: shape [..., n], uint32
    """
    lo = counter.astype(np.uint32)
    hi = np.broadcast_to(key.astype(np.uint32), lo.shape).copy()
    M = np.uint32(0xD2511F53)
    W = np.uint32(0x9E3779B9)
    for i in range(6):
        lo_old = lo.copy()
        prod = (lo_old.astype(np.uint64) * M) >> np.uint64(32)
        lo = (prod.astype(np.uint32) ^ hi ^ (W * np.uint32(i + 1))).astype(np.uint32)
        hi = (lo_old * M).astype(np.uint32)
    return lo


def block_seeds(n_blocks: int, salt: int) -> np.ndarray:
    """Derive per-block seeds (FNV-1a hash) XOR salt.
    Matches kktq_derive_seed in ggml-quants.c."""
    idx = np.arange(n_blocks, dtype=np.uint32)
    h = np.full(n_blocks, 2166136261, dtype=np.uint32)
    for shift in (0, 8, 16, 24):
        h = h ^ ((idx >> np.uint32(shift)) & np.uint32(0xFF))
        h = (h.astype(np.uint64) * np.uint64(16777619)).astype(np.uint32)
    seeds = (h & np.uint32(0xFFFF)).astype(np.uint16)
    return (seeds ^ np.uint16(salt & 0xFFFF)).astype(np.uint16)


def rht_forward(blocks: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Forward RHT: y = H · (sign(seed) ⊙ x). blocks [N, QK_VTQ2]."""
    n = blocks.shape[-1]
    j = np.arange(n, dtype=np.uint32)
    # signs per block: shape [N, n] — counter [N,n], key [N,1]
    counter = np.broadcast_to(j[None, :], (seeds.size, n)).astype(np.uint32)
    key = seeds.astype(np.uint32)[:, None]  # shape [N, 1]
    rng = philox_6r(counter, key)
    signs = np.where((rng & 1) == 1, 1.0, -1.0).astype(np.float32)
    y = blocks * signs
    return fwht(y)


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis: 0 for Gaussian, positive for heavy-tailed."""
    mu = x.mean()
    s = x.std()
    if s < 1e-20:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)


def load_v_blocks(model_path: str, max_blocks: int) -> np.ndarray:
    """Load V-projection weights, split into QK_VTQ2-sized blocks."""
    reader = gguf.GGUFReader(model_path, "r")
    collected = []
    total = 0
    for tensor in reader.tensors:
        name = tensor.name
        if ".attn_v.weight" not in name and ".attn_qkv.weight" not in name:
            continue
        arr = np.array(tensor.data, copy=False).astype(np.float32)
        flat = arr.reshape(-1)
        n_usable = (flat.size // QK_VTQ2) * QK_VTQ2
        if n_usable == 0:
            continue
        blocks = flat[:n_usable].reshape(-1, QK_VTQ2)
        collected.append(blocks)
        total += blocks.shape[0]
        if total >= max_blocks:
            break
    if not collected:
        raise RuntimeError("no V-projection tensors found in model")
    all_blocks = np.concatenate(collected, axis=0)
    if all_blocks.shape[0] > max_blocks:
        all_blocks = all_blocks[:max_blocks]
    return all_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-salts", type=int, default=256)
    ap.add_argument("--max-blocks", type=int, default=4096)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print(f"Loading V blocks from {args.model} ...", flush=True)
    blocks = load_v_blocks(args.model, args.max_blocks)
    print(f"  {blocks.shape[0]} blocks × {blocks.shape[1]} samples",
          flush=True)

    # Per-block L2 normalize (matches how VTQ treats blocks)
    norms = np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-30
    blocks = blocks / norms * np.sqrt(QK_VTQ2)

    n_blocks = blocks.shape[0]
    results = []

    print(f"Sweeping {args.n_salts} salts ...", flush=True)
    for salt in range(args.n_salts):
        seeds = block_seeds(n_blocks, salt)
        rotated = rht_forward(blocks, seeds)
        kurt = excess_kurtosis(rotated)
        results.append((salt, kurt))
        if salt % 32 == 0:
            print(f"  salt={salt:4d}  excess_kurt={kurt:+.6f}", flush=True)

    arr = np.array(results)
    salts = arr[:, 0].astype(int)
    kurts = arr[:, 1]

    best = int(salts[np.argmin(np.abs(kurts))])
    worst = int(salts[np.argmax(np.abs(kurts))])

    print("")
    print(f"Results across {args.n_salts} salts:")
    print(f"  mean excess kurtosis = {kurts.mean():+.6f}")
    print(f"  std  excess kurtosis = {kurts.std():.6f}")
    print(f"  min |kurt| = {np.abs(kurts).min():.6f}  (salt={best})")
    print(f"  max |kurt| = {np.abs(kurts).max():.6f}  (salt={worst})")
    print(f"  spread = {np.abs(kurts).max() - np.abs(kurts).min():.6f}")

    # Decision heuristic: is the spread meaningful?
    # Noise floor ~ sqrt(2/N) on kurtosis estimator for N samples.
    n_total = blocks.shape[0] * QK_VTQ2
    noise = np.sqrt(24.0 / n_total)
    print(f"  noise floor (~sqrt(24/N)) = {noise:.6f}")
    if kurts.std() < noise:
        print("  VERDICT: variation below noise floor — abandon Trick 3")
    else:
        print(f"  VERDICT: meaningful variation — recommend salt={best}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("salt,excess_kurtosis\n")
            for s, k in results:
                f.write(f"{s},{k:.8f}\n")
        print(f"  CSV written to {args.csv}")


if __name__ == "__main__":
    main()
