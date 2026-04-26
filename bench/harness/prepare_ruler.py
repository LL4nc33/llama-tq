#!/usr/bin/env python3
"""
prepare_ruler.py — Generate RULER task JSONL for our smoke + full profiles.

Two modes:
  * --use-vendor: shell out to vendored NVIDIA/RULER generators
    (requires nltk, wonderwords, transformers, tqdm). Produces real RULER
    samples honoring tokenizer-aware length budgets.
  * default: built-in synthetic stand-in for niah_single_3 only.
    Stdlib only — exercises the pipeline without heavy deps. Other tasks
    error out with a clear message pointing at --use-vendor.

The vendored generators live under bench/harness/vendor/ruler/data/synthetic/.
See vendor/VENDOR.md for the upstream SHA pin.

Usage:
    python3 prepare_ruler.py \\
        --task niah_single_3 --length 4096 --num-samples 10 \\
        --output runs/<ts>/ruler/niah_single_3_4k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "vendor" / "ruler" / "data" / "synthetic"
SUPPORTED_BUILTIN = {"niah_single_3"}


def builtin_niah_single_3(num_samples: int, length: int, seed: int = 42) -> list[dict]:
    """Stdlib-only NIAH stand-in. Filler scaled by `length` (rough char heuristic).

    Not a real RULER sample — for wiring tests only. The vendored upstream
    generator is required for length-accurate, distribution-faithful tasks.
    """
    rng = random.Random(seed)
    # Roughly 1 token ~ 4 chars; pad filler so haystack approximates the budget.
    filler_chars = max(length * 3, 1024)
    base_filler = "Filler sentence about an unrelated topic. "
    haystack = (base_filler * (filler_chars // len(base_filler) + 1))[:filler_chars]
    out = []
    for i in range(num_samples):
        needle = f"key-{i:04d}-VALUE-{rng.randint(1000, 9999)}"
        prompt = (
            "You are given a long document. Find the secret key.\n\n"
            f"<document>\n{haystack}\nThe secret key is {needle}.\n{haystack}\n</document>\n\n"
            "What is the secret key? Answer with the key only."
        )
        out.append({
            "index": i,
            "task": "niah_single_3",
            "length": length,
            "input": prompt,
            "outputs": [needle],
        })
    return out


def run_vendored(task: str, length: int, num_samples: int, output: Path,
                 tokenizer_path: str) -> int:
    """Shell out to vendored generator. Heavy deps required."""
    # Map our task names → upstream save_name/script. niah_* all share niah.py.
    if task.startswith("niah"):
        script = VENDOR_DIR / "niah.py"
    elif task == "vt":
        script = VENDOR_DIR / "variable_tracking.py"
    elif task == "cwe":
        script = VENDOR_DIR / "common_words_extraction.py"
    elif task == "fwe":
        script = VENDOR_DIR / "freq_words_extraction.py"
    elif task in ("qa_1", "qa_2"):
        script = VENDOR_DIR / "qa.py"
    else:
        print(f"[prepare_ruler] unknown task: {task}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--save_dir", str(output.parent),
        "--save_name", output.stem,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(length),
        "--tokens_to_generate", "128",
        "--num_samples", str(num_samples),
    ]
    print(f"[prepare_ruler] vendored: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare RULER task JSONL.")
    p.add_argument("--task", required=True, help="e.g. niah_single_3, vt, cwe")
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--use-vendor", action="store_true",
                   help="Use vendored RULER generators (requires heavy deps).")
    p.add_argument("--tokenizer-path", default="",
                   help="HF tokenizer dir/repo, required with --use-vendor.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    if args.use_vendor:
        if not args.tokenizer_path:
            print("[prepare_ruler] --tokenizer-path required with --use-vendor",
                  file=sys.stderr)
            return 2
        return run_vendored(args.task, args.length, args.num_samples, out,
                            args.tokenizer_path)

    if args.task not in SUPPORTED_BUILTIN:
        print(f"[prepare_ruler] task '{args.task}' has no built-in stand-in. "
              f"Pass --use-vendor (and --tokenizer-path) to use the real generator.",
              file=sys.stderr)
        return 2

    samples = builtin_niah_single_3(args.num_samples, args.length, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in samples:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[prepare_ruler] wrote {len(samples)} samples → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
