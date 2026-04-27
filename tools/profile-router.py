#!/usr/bin/env python3
"""
Phase 6a/6f — Router profile analyzer.

Consumes the binary dump produced by `llama-perplexity --log-router-stats <path>`
(format TQR2). The dump interleaves two record types per (token, layer):

  tag 'L'  pre-gating logits (F32, n_expert wide)
  tag 'K'  selected expert IDs (I32, n_expert_used wide)

Produces two analyses:

  --mode=routing  (default)
      Decision-gate metrics for adaptive top-k:
        mean_k @ tau, p99_k, max_k, per-layer mean_k.
      Exit code 0 if mean_k < 5 AND p99_k < n_expert/2.

  --mode=hotness
      Per-layer expert-hotness ranking from selected-expert records:
        top-N hot experts per layer, share-of-dispatch, unique-touched count.
      Output: JSON file consumable by Phase 6f prefetcher.

See `docs/plans/2026-04-27-phase6f-hot-expert-prefetch.md`.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HEADER_FMT = "<4sIIIf12s"  # magic, version, n_expert, n_expert_used, tau, pad
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_HDR_FMT = "<IHBB"   # token_idx u32, layer_idx u16, tag u8, pad u8
RECORD_HDR_SIZE = struct.calcsize(RECORD_HDR_FMT)


def parse_dump(path: Path):
    """Stream-parse the binary dump and bucket records by tag."""
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"dump too small: {len(raw)} bytes")

    magic, version, n_expert, n_expert_used, tau, _pad = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != b"TQR2":
        if magic == b"TQRP":
            raise ValueError("legacy dump (TQRP). Re-run with new llama-perplexity.")
        raise ValueError(f"bad magic: {magic!r}")
    if version != 2:
        raise ValueError(f"unsupported version: {version}")

    body = raw[HEADER_SIZE:]
    pos = 0
    logits_records = []
    topk_records = []
    while pos + RECORD_HDR_SIZE <= len(body):
        tok, lay, tag, _pad = struct.unpack_from(RECORD_HDR_FMT, body, pos)
        pos += RECORD_HDR_SIZE
        if tag == ord("L"):
            n = n_expert
            row = np.frombuffer(body, dtype=np.float32, count=n, offset=pos)
            pos += 4 * n
            logits_records.append((lay, row))
        elif tag == ord("K"):
            n = n_expert_used or 8  # fallback if header was patched late
            row = np.frombuffer(body, dtype=np.int32, count=n, offset=pos)
            pos += 4 * n
            topk_records.append((lay, row))
        else:
            raise ValueError(f"unknown tag {tag} at offset {pos}")

    header = {
        "version": version,
        "n_expert": n_expert,
        "n_expert_used": n_expert_used,
        "tau_recorded": tau,
        "n_logits_records": len(logits_records),
        "n_topk_records": len(topk_records),
    }
    return logits_records, topk_records, header


# ---------------------------------------------------------------------------
# Mode: routing — decision gate for adaptive top-k (Phase 6a)
# ---------------------------------------------------------------------------
def analyze_routing(logits_records, header, tau: float, gate_mean_k: float, out: Path | None) -> int:
    if not logits_records:
        print("ERROR: no logits records in dump.", file=sys.stderr)
        return 2
    n_expert = header["n_expert"]
    layers = np.array([r[0] for r in logits_records], dtype=np.uint16)
    logits = np.stack([r[1] for r in logits_records])

    logits = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=-1, keepdims=True)
    sorted_desc = np.sort(probs, axis=-1)[:, ::-1]

    cumsum = np.cumsum(sorted_desc, axis=-1)
    reaches = cumsum >= tau
    k_tau = np.where(
        reaches.any(axis=-1), reaches.argmax(axis=-1) + 1, sorted_desc.shape[-1]
    )

    mean_k = float(k_tau.mean())
    p99_k = float(np.percentile(k_tau, 99))
    max_k = int(k_tau.max())
    top1_p = sorted_desc[:, 0]

    per_layer = {}
    for lid in np.unique(layers):
        mask = layers == lid
        per_layer[int(lid)] = {
            "n_tokens": int(mask.sum()),
            "mean_k": float(k_tau[mask].mean()),
            "p99_k": float(np.percentile(k_tau[mask], 99)),
            "mean_top1_p": float(top1_p[mask].mean()),
        }

    gate_pass = mean_k < gate_mean_k and p99_k < (n_expert / 2.0)
    verdict = "PASS" if gate_pass else "ABORT"

    print("=== Phase 6a router-confidence report ===")
    print(f"  records         : {len(logits_records)}")
    print(f"  n_expert        : {n_expert}")
    print(f"  layers seen     : {len(per_layer)}")
    print(f"  tau (analysis)  : {tau:.2f}")
    print()
    print(f"  mean_k          : {mean_k:.3f}")
    print(f"  p99_k           : {p99_k:.1f}")
    print(f"  max_k           : {max_k}")
    print(f"  mean top-1 prob : {float(top1_p.mean()):.4f}")
    print()
    print(f"  GATE {verdict}: mean_k={mean_k:.2f} (threshold < {gate_mean_k:.1f}), "
          f"p99_k={p99_k:.1f} (threshold < {n_expert / 2:.0f})")

    if out:
        out.write_text(json.dumps({
            "header": header,
            "mode": "routing",
            "tau_analysis": tau,
            "global": {"mean_k": mean_k, "p99_k": p99_k, "max_k": max_k,
                       "mean_top1_p": float(top1_p.mean())},
            "per_layer": per_layer,
            "gate_pass": gate_pass,
        }, indent=2))
        print(f"  → wrote {out}")

    return 0 if gate_pass else 1


# ---------------------------------------------------------------------------
# Mode: hotness — per-layer expert ranking (Phase 6f)
# ---------------------------------------------------------------------------
def analyze_hotness(topk_records, header, top_n: int, out: Path | None,
                    model_name: str | None) -> int:
    if not topk_records:
        print("ERROR: no topk records in dump (need RPRS/'K' records).", file=sys.stderr)
        return 2
    n_expert = header["n_expert"]
    n_expert_used = header["n_expert_used"]

    per_layer_counter: dict[int, Counter] = {}
    per_layer_tokens: dict[int, int] = {}
    for lay, ids in topk_records:
        c = per_layer_counter.setdefault(int(lay), Counter())
        per_layer_tokens[int(lay)] = per_layer_tokens.get(int(lay), 0) + 1
        c.update(ids.tolist())

    layer_summary = {}
    overall_top_share = []
    for lid, c in sorted(per_layer_counter.items()):
        total = sum(c.values())
        top = c.most_common(top_n)
        top_ids = [int(eid) for eid, _ in top]
        top_share = sum(cnt for _, cnt in top) / total if total else 0.0
        layer_summary[str(lid)] = {
            "tokens": per_layer_tokens[lid],
            "unique_touched": len(c),
            f"top_{top_n}": top_ids,
            f"top_{top_n}_share": round(top_share, 4),
        }
        overall_top_share.append(top_share)

    n_layers = len(per_layer_counter)
    mean_share = float(np.mean(overall_top_share)) if overall_top_share else 0.0

    print("=== Phase 6f expert-hotness report ===")
    print(f"  records         : {len(topk_records)}")
    print(f"  n_expert        : {n_expert}")
    print(f"  n_expert_used   : {n_expert_used}")
    print(f"  layers seen     : {n_layers}")
    print(f"  top-N kept      : {top_n}")
    print(f"  mean top-{top_n} dispatch share: {mean_share:.3f}")
    print()
    for lid, st in sorted(layer_summary.items(), key=lambda x: int(x[0])):
        print(f"  layer {int(lid):3d}: unique={st['unique_touched']:3d} "
              f"top_{top_n}_share={st[f'top_{top_n}_share']:.3f}")

    if out:
        # Schema designed for runtime loader: sha256(model) verification + per-layer arrays.
        report = {
            "schema_version": 1,
            "model_name": model_name or "unknown",
            "n_expert": n_expert,
            "n_expert_used": n_expert_used,
            "top_k": top_n,
            "stats": {
                "n_layers": n_layers,
                "tokens_analyzed": sum(per_layer_tokens.values()),
                f"mean_top_{top_n}_share": mean_share,
            },
            "layers": {
                lid: st[f"top_{top_n}"] for lid, st in layer_summary.items()
            },
            "layer_stats": layer_summary,
        }
        out.write_text(json.dumps(report, indent=2))
        print(f"\n  → wrote {out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="binary dump from --log-router-stats")
    ap.add_argument("--mode", choices=["routing", "hotness"], default="routing")
    ap.add_argument("--tau", type=float, default=0.85,
                    help="(routing) cumulative-mass threshold (default: 0.85)")
    ap.add_argument("--gate-mean-k", type=float, default=5.0,
                    help="(routing) abort if mean_k >= this (default: 5.0)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="(hotness) keep top-N experts per layer (default: 20)")
    ap.add_argument("--model-name", type=str, default=None,
                    help="(hotness) friendly model name to embed in JSON output")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report to this path")
    args = ap.parse_args()

    logits, topk, header = parse_dump(args.dump)

    if args.mode == "routing":
        return analyze_routing(logits, header, args.tau, args.gate_mean_k, args.out)
    else:
        return analyze_hotness(topk, header, args.top_n, args.out, args.model_name)


if __name__ == "__main__":
    sys.exit(main())
