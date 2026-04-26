#!/usr/bin/env python3
"""
prepare_longbench.py — Download THUDM/LongBench EN subset, write JSONL.

Uses HuggingFace `datasets` (lazy import). Each task pulled from
`THUDM/LongBench` config → mapped through vendored dataset2prompt.json
to build a ready-to-eval prompt.

Usage:
    python3 prepare_longbench.py \\
        --task narrativeqa --num-samples 50 \\
        --output runs/<ts>/longbench/narrativeqa.jsonl

EN subset only. ZH tasks are intentionally not wired here (the vendored
metrics.py still supports them; activate later by extending TASK_LANG).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VENDOR_CFG = Path(__file__).parent / "vendor" / "longbench" / "config"
EN_TASKS = {
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "lcc", "repobench-p", "passage_count", "passage_retrieval_en",
}


def load_prompts() -> dict:
    p = VENDOR_CFG / "dataset2prompt.json"
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare LongBench EN task JSONL.")
    p.add_argument("--task", required=True)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.task not in EN_TASKS:
        print(f"[prepare_longbench] '{args.task}' not in EN subset. "
              f"Allowed: {sorted(EN_TASKS)}", file=sys.stderr)
        return 2

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[prepare_longbench] huggingface `datasets` not installed. "
              "Run: pip install datasets", file=sys.stderr)
        return 3

    prompts = load_prompts()
    template = prompts.get(args.task)
    if template is None:
        print(f"[prepare_longbench] no prompt template for '{args.task}'",
              file=sys.stderr)
        return 2

    print(f"[prepare_longbench] loading THUDM/LongBench config={args.task}",
          file=sys.stderr)
    ds = load_dataset("THUDM/LongBench", args.task, split="test")
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, ex in enumerate(ds):
            prompt = template.format(**{k: ex.get(k, "") for k in
                                        ["context", "input", "question"]
                                        if "{" + k + "}" in template})
            rec = {
                "index": i,
                "task": args.task,
                "input": prompt,
                "outputs": ex.get("answers", []),
                "all_classes": ex.get("all_classes"),
                "length": ex.get("length"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"[prepare_longbench] wrote {written} samples → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
