#!/usr/bin/env python3
"""
score_ruler.py — Stub wrapper around NVIDIA/RULER's evaluate.py.

Reads a preds JSONL written by pred_llama.py, computes per-task
accuracy against the gold `outputs` field, emits a summary CSV.

CURRENT STATE: stub-only.
The vendored RULER code (under bench/harness/vendor/ruler/) is NOT yet
checked in. See bench/harness/vendor/VENDOR.md for the target SHA.

Until vendor lands, we ship a *placeholder* string-contains scorer so the
end-to-end pipeline (pred → score → summary) can be smoke-tested on
niah_single_3, where the prediction must merely contain the gold string.
This is NOT correct for cwe / vt / niah_multikey_*; do not trust those
numbers until vendor is in.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def placeholder_score(pred: str, golds: list[str]) -> float:
    """Substring-match accuracy. TODO: replace with vendored RULER metric."""
    if not golds:
        return 0.0
    pred_l = (pred or "").lower()
    return 1.0 if any((g or "").lower().strip() in pred_l for g in golds) else 0.0


def vendored_score(task: str, pred: str, golds: list[str]) -> float:
    """Dispatch to vendored RULER scorers once available."""
    # TODO: from bench.harness.vendor.ruler.scripts.eval.synthetic.evaluate import score_fn
    # return score_fn(task, pred, golds)
    raise NotImplementedError(
        "Vendored RULER scorer not yet imported. "
        "See bench/harness/vendor/VENDOR.md for the target SHA."
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Score RULER preds JSONL → summary CSV")
    p.add_argument("--preds", required=True, help="preds.jsonl from pred_llama.py")
    p.add_argument("--summary", required=True, help="output summary CSV")
    p.add_argument(
        "--use-vendored",
        action="store_true",
        help="Use vendored RULER metric (errors until VENDOR.md SHA is pinned).",
    )
    args = p.parse_args()

    by_task: dict[str, list[float]] = defaultdict(list)
    with Path(args.preds).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec.get("task") or "unknown"
            golds = rec.get("outputs") or []
            pred = rec.get("pred", "")
            if args.use_vendored:
                acc = vendored_score(task, pred, golds)
            else:
                acc = placeholder_score(pred, golds)
            by_task[task].append(acc)

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "n", "accuracy"])
        for task in sorted(by_task):
            scores = by_task[task]
            acc = sum(scores) / max(len(scores), 1)
            w.writerow([task, len(scores), f"{acc:.4f}"])
            print(f"{task}\tn={len(scores)}\tacc={acc:.4f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
