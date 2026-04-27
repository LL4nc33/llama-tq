#!/usr/bin/env python3
"""
Phase 6a — Router confidence analyzer.

Consumes the binary dump produced by `llama-perplexity --log-router-stats <path>`
and computes the decision-gate metrics for adaptive top-k MoE routing:

    mean_k @ tau       — average minimum k such that cumsum(sorted_probs) >= tau
    mean_k_per_layer   — same metric, broken out by layer
    p99_k, max_k       — tail behavior
    histogram(k)       — distribution of chosen k values
    histogram(top1_p)  — router peakedness

Decision gate (exit code 0 = proceed, 1 = abort):
    PASS  if mean_k < 5 AND p99_k < n_expert/2

See `docs/plans/2026-04-27-phase6-adaptive-topk-moe.md`.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


HEADER_FMT = "<4sIIIf12s"  # magic, version, n_expert, reserved, tau, pad
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_HDR_FMT = "<IHH"  # token_idx, layer_idx, n_expert
RECORD_HDR_SIZE = struct.calcsize(RECORD_HDR_FMT)


def parse_dump(path: Path):
    """Parse the binary dump and return (probs[N, n_expert], layer_idx[N], header)."""
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"dump too small: {len(raw)} bytes")

    magic, version, n_expert, _reserved, tau, _pad = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != b"TQRP":
        raise ValueError(f"bad magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported version: {version}")

    record_size = RECORD_HDR_SIZE + 4 * n_expert
    body = raw[HEADER_SIZE:]
    n_records = len(body) // record_size
    if n_records == 0:
        raise ValueError("no records in dump")

    # Parse layer_idx column (we don't need token_idx for histograms).
    layer_idx = np.empty(n_records, dtype=np.uint16)
    probs = np.empty((n_records, n_expert), dtype=np.float32)

    for i in range(n_records):
        off = i * record_size
        _tok, lay, n_e = struct.unpack(RECORD_HDR_FMT, body[off : off + RECORD_HDR_SIZE])
        if n_e != n_expert:
            raise ValueError(f"record {i}: n_expert mismatch ({n_e} != {n_expert})")
        layer_idx[i] = lay
        # Slice the float row directly.
        row_off = off + RECORD_HDR_SIZE
        probs[i] = np.frombuffer(body, dtype=np.float32, count=n_expert, offset=row_off)

    header = {
        "version": version,
        "n_expert": n_expert,
        "tau_recorded": tau,
        "n_records": n_records,
    }
    return probs, layer_idx, header


def compute_k_tau(sorted_desc: np.ndarray, tau: float) -> np.ndarray:
    """
    sorted_desc: [N, n_expert] probabilities sorted descending per row.
    Returns k_tau[N] = smallest k such that cumsum[k-1] >= tau.
    """
    cumsum = np.cumsum(sorted_desc, axis=-1)
    # argmax of bool returns first True index; +1 for 1-indexed k.
    reaches = cumsum >= tau
    # If a row never reaches tau (numerical edge), fall back to n_expert.
    k = np.where(reaches.any(axis=-1), reaches.argmax(axis=-1) + 1, sorted_desc.shape[-1])
    return k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="binary dump from --log-router-stats")
    ap.add_argument("--tau", type=float, default=0.85, help="cumulative-mass threshold (default: 0.85)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report to this path")
    ap.add_argument("--gate-mean-k", type=float, default=5.0, help="abort if mean_k >= this (default: 5.0)")
    args = ap.parse_args()

    logits, layer_idx, header = parse_dump(args.dump)
    n_expert = header["n_expert"]

    # The dump contains pre-gating router logits (ffn_moe_logits-N).
    # We apply softmax here so the analyzer is gating-op-agnostic — works for
    # SOFTMAX, SIGMOID and SOFTMAX_WEIGHT (Qwen3-Next) routers identically.
    logits = logits - logits.max(axis=-1, keepdims=True)  # numerical stability
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=-1, keepdims=True)

    sorted_desc = np.sort(probs, axis=-1)[:, ::-1]
    k_tau = compute_k_tau(sorted_desc, args.tau)

    mean_k = float(k_tau.mean())
    p99_k  = float(np.percentile(k_tau, 99))
    max_k  = int(k_tau.max())
    top1_p = sorted_desc[:, 0]

    # Per-layer aggregation.
    layers = np.unique(layer_idx)
    per_layer = {}
    for lid in layers:
        mask = layer_idx == lid
        per_layer[int(lid)] = {
            "n_tokens": int(mask.sum()),
            "mean_k": float(k_tau[mask].mean()),
            "p99_k": float(np.percentile(k_tau[mask], 99)),
            "mean_top1_p": float(top1_p[mask].mean()),
        }

    # k histogram (1..n_expert).
    k_hist = np.bincount(k_tau, minlength=n_expert + 1)[1:]

    # Decision gate.
    gate_pass = mean_k < args.gate_mean_k and p99_k < (n_expert / 2.0)
    verdict = "PASS" if gate_pass else "ABORT"

    print(f"=== Phase 6a router-confidence report ===")
    print(f"  records         : {header['n_records']}")
    print(f"  n_expert        : {n_expert}")
    print(f"  layers seen     : {len(layers)}")
    print(f"  tau (analysis)  : {args.tau:.2f}")
    print(f"  tau (recorded)  : {header['tau_recorded']:.2f}")
    print(f"")
    print(f"  mean_k          : {mean_k:.3f}")
    print(f"  p99_k           : {p99_k:.1f}")
    print(f"  max_k           : {max_k}")
    print(f"  mean top-1 prob : {float(top1_p.mean()):.4f}")
    print(f"  median top-1 p  : {float(np.median(top1_p)):.4f}")
    print(f"")
    print(f"  GATE {verdict}: mean_k={mean_k:.2f} (threshold < {args.gate_mean_k:.1f}), "
          f"p99_k={p99_k:.1f} (threshold < {n_expert / 2:.0f})")
    print(f"")
    print(f"  per-layer mean_k:")
    for lid, st in sorted(per_layer.items()):
        bar = "#" * int(st["mean_k"] * 4)
        print(f"    layer {lid:3d}: mean_k={st['mean_k']:5.2f}  top1={st['mean_top1_p']:.3f}  {bar}")

    if args.out:
        report = {
            "header": header,
            "tau_analysis": args.tau,
            "global": {
                "mean_k": mean_k,
                "p99_k": p99_k,
                "max_k": max_k,
                "mean_top1_p": float(top1_p.mean()),
                "median_top1_p": float(np.median(top1_p)),
            },
            "per_layer": per_layer,
            "k_histogram": k_hist.tolist(),
            "gate_pass": bool(gate_pass),
            "gate_threshold_mean_k": args.gate_mean_k,
        }
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\n  → wrote {args.out}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
