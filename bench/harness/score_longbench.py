#!/usr/bin/env python3
"""
score_longbench.py — Score LongBench EN preds JSONL → results JSON.

Dispatches each task to its canonical metric from the vendored
LongBench/metrics.py. EN tasks only — ZH dispatch is intentionally
omitted; activate later by extending TASK2METRIC.

Usage:
    python3 score_longbench.py \\
        --preds runs/<ts>/longbench/<task>.preds.jsonl \\
        --task narrativeqa \\
        --results runs/<ts>/longbench/<task>.results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VENDOR_LB = Path(__file__).parent / "vendor" / "longbench"


def _load_metrics():
    """Lazy-import vendored metrics. fuzzywuzzy/rouge required for some tasks."""
    sys.path.insert(0, str(VENDOR_LB))
    try:
        import metrics as _m  # type: ignore
    except ImportError as e:
        print(f"[score_longbench] vendored metrics import failed: {e}\n"
              "Install: pip install fuzzywuzzy rouge", file=sys.stderr)
        raise
    return _m


# EN task → metric function name (in vendored metrics module).
TASK2METRIC = {
    "narrativeqa": "qa_f1_score",
    "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score",
    "hotpotqa": "qa_f1_score",
    "2wikimqa": "qa_f1_score",
    "musique": "qa_f1_score",
    "triviaqa": "qa_f1_score",
    "gov_report": "rouge_score",
    "qmsum": "rouge_score",
    "multi_news": "rouge_score",
    "samsum": "rouge_score",
    "trec": "classification_score",
    "lcc": "code_sim_score",
    "repobench-p": "code_sim_score",
    "passage_count": "count_score",
    "passage_retrieval_en": "retrieval_score",
}


def score_task(task: str, preds_path: Path) -> dict:
    metrics = _load_metrics()
    fn_name = TASK2METRIC.get(task)
    if fn_name is None:
        raise SystemExit(f"[score_longbench] no EN metric for task '{task}'")
    metric_fn = getattr(metrics, fn_name)

    total, n = 0.0, 0
    with preds_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pred = rec.get("pred", "") or ""
            golds = rec.get("outputs") or []
            kwargs = {}
            if "all_classes" in rec and rec["all_classes"] is not None:
                kwargs["all_classes"] = rec["all_classes"]
            if not golds:
                continue
            best = max(metric_fn(pred, g, **kwargs) for g in golds)
            total += best
            n += 1
    score = (total / n * 100) if n else 0.0
    return {"task": task, "metric": fn_name, "n": n, "score": round(score, 4)}


def main() -> int:
    p = argparse.ArgumentParser(description="Score LongBench EN preds → results JSON")
    p.add_argument("--preds", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--results", required=True)
    args = p.parse_args()

    result = score_task(args.task, Path(args.preds))
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[score_longbench] {result['task']}\tn={result['n']}\t"
          f"{result['metric']}={result['score']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
