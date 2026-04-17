#!/usr/bin/env python3
"""
Extract V-projection weights from a GGUF model and apply the VTQ fixed
rotation (D·H·D, Hadamard-based) to produce realistic post-rotation
samples that mimic V-cache distributions.

Usage:
    PYTHONPATH=gguf-py python3 extract_v_samples.py \
        --model /path/to/model.gguf \
        --out vcache.bin \
        --max-samples 65536 \
        --layers 4,8,12,16   (optional; defaults to every Nth layer)

The output is a packed float32 binary usable with:
    ./trellis_phase1 --mode real --data vcache.bin --n 32768
"""

import argparse
import sys
import os
import struct
import numpy as np

# Extend path to llama-tq gguf-py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "gguf-py"))

import gguf


def hadamard_matrix(n: int) -> np.ndarray:
    """Build normalized Hadamard matrix H_n (n must be power of 2)."""
    assert n & (n - 1) == 0, "n must be power of 2"
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def rht(x: np.ndarray, seed: int = 42) -> np.ndarray:
    """Randomized Hadamard Transform: y = D·H·D·x along last axis.
    D is random ±1 diagonal (seeded). Matches VTQ's fixed rotation
    structure. Input shape [..., n]; n must be power of 2."""
    n = x.shape[-1]
    rng = np.random.default_rng(seed)
    d1 = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    d2 = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    H = hadamard_matrix(n)
    # y = d1 · H · (d2 · x)
    return (x * d2) @ H * d1


def extract_v_weights(model_path: str, layer_filter=None):
    """Yield (layer_idx, tensor_name, f32 array) for each V-projection weight.

    Supports two layouts:
      - Separate: blk.N.attn_v.weight
      - Fused:    blk.N.attn_qkv.weight (V is the last kv_dim slice)
    """
    reader = gguf.GGUFReader(model_path, "r")

    # Pull attention metadata to split the fused qkv layout
    def field(key):
        for fk, f in reader.fields.items():
            if fk.endswith(key):
                try:
                    return int(f.parts[-1][0])
                except Exception:
                    return int(f.parts[-1])
        return None

    n_head    = field("attention.head_count") or 1
    n_kv_head = field("attention.head_count_kv") or n_head
    v_len     = field("attention.value_length")  # None = embed / n_head

    for tensor in reader.tensors:
        name = tensor.name
        try:
            layer_idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        if layer_filter is not None and layer_idx not in layer_filter:
            continue

        arr = np.array(tensor.data, copy=False).astype(np.float32)

        if ".attn_v.weight" in name:
            yield layer_idx, name, arr

        elif ".attn_qkv.weight" in name:
            # Fused QKV: split out the V slice.
            # Typical layouts (columns = output dim):
            #   [n_embd, q_out + k_out + v_out]
            # where q_out = n_head * head_dim, k_out = v_out = n_kv_head * head_dim
            # We take the last v_out columns.
            if arr.ndim != 2:
                continue
            head_dim_est = v_len if v_len else (arr.shape[0] // n_head)
            v_out = n_kv_head * head_dim_est
            if v_out > arr.shape[1]:
                # fallback: last 1/3 slice
                v_out = arr.shape[1] // 3
            v_slice = arr[:, -v_out:]
            yield layer_idx, name + "[V-slice]", v_slice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=65536)
    ap.add_argument("--layers", type=str, default=None,
                    help="comma-sep layer indices; default = every 4th")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layer_filter = None
    if args.layers:
        layer_filter = set(int(x) for x in args.layers.split(","))

    collected = []
    total = 0
    for lidx, name, arr in extract_v_weights(args.model, layer_filter):
        # Default: sample every 4th layer
        if layer_filter is None and lidx % 4 != 0:
            continue
        # arr shape: [n_embd_v_gqa, n_embd] typically, or transposed
        # We want rows of length head_dim. Flatten and reshape so each row is
        # a head_dim-sized chunk, then apply RHT along the last axis.
        flat = arr.reshape(-1)
        head_dim = args.head_dim
        # Trim to multiple of head_dim
        n_usable = (flat.size // head_dim) * head_dim
        rows = flat[:n_usable].reshape(-1, head_dim)
        # Shuffle rows (deterministic) to mix layers/heads
        rng = np.random.default_rng(args.seed + lidx)
        idx = rng.permutation(rows.shape[0])
        rows = rows[idx]
        # Apply RHT row-wise
        rotated = rht(rows, seed=args.seed + lidx).astype(np.float32)
        # Per-row L2 normalize (matches VTQ's per-block norm)
        norms = np.linalg.norm(rotated, axis=1, keepdims=True) + 1e-30
        normed = rotated / norms
        # Rescale to approximately unit variance per element
        # (after row-norm, each element has variance ~ 1/n; undo)
        normed = normed * np.sqrt(head_dim)
        collected.append(normed.reshape(-1))
        total += normed.size
        print(f"  layer {lidx:3d} {name}: {normed.size} samples (total {total})")
        if total >= args.max_samples:
            break

    if not collected:
        sys.exit("no V-weights found")

    data = np.concatenate(collected)[: args.max_samples].astype(np.float32)
    with open(args.out, "wb") as f:
        f.write(data.tobytes())

    # Diagnostics
    print(f"\nwrote {args.out}: {data.size} samples")
    print(f"  mean = {data.mean():+.4f}")
    print(f"  var  = {data.var():.4f}")
    print(f"  std  = {data.std():.4f}")
    print(f"  min  = {data.min():+.4f}")
    print(f"  max  = {data.max():+.4f}")
    p1, p99 = np.percentile(data, [1, 99])
    print(f"  1%-99% = [{p1:+.3f}, {p99:+.3f}]")


if __name__ == "__main__":
    main()
