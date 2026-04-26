#!/usr/bin/env python3
"""
pred_llama.py — Drive a llama-server /completion endpoint over a JSONL task file.

Reads tasks (one JSON object per line, with at least `index` and `input`),
POSTs each prompt to the configured llama-server, and writes prediction
records to an output JSONL.

Designed to plug in as a drop-in replacement for NVIDIA/RULER's
`pred/pred_*.py` and THUDM/LongBench's `pred.py`.

Usage:
    python3 pred_llama.py \\
        --server-url http://localhost:8080 \\
        --input  tasks/niah_single_3_4k.jsonl \\
        --output preds/niah_single_3_4k.preds.jsonl \\
        --max-new-tokens 128 \\
        --temperature 0.0

The harness intentionally disables prompt-cache reuse so KV-cache regressions
(e.g. TurboQuant) cannot be masked by warm-prefix shortcuts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error


def post_completion(
    server_url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    stop: list[str] | None,
    timeout_s: float,
) -> dict[str, Any]:
    """POST to /completion. Returns the parsed JSON response."""
    payload = {
        "prompt": prompt,
        "n_predict": max_new_tokens,
        "temperature": temperature,
        "cache_prompt": False,  # critical: do not warm-reuse KV across samples
        "stream": False,
    }
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{server_url.rstrip('/')}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def run(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"[pred_llama] {len(tasks)} samples → {args.server_url}", file=sys.stderr)

    written = 0
    t_start = time.time()
    with out_path.open("w", encoding="utf-8") as out_fh:
        for i, task in enumerate(tasks):
            prompt = task.get("input") or task.get("prompt") or ""
            if not prompt:
                print(f"[pred_llama] skip idx={i}: empty prompt", file=sys.stderr)
                continue

            t0 = time.time()
            try:
                resp = post_completion(
                    server_url=args.server_url,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop=args.stop,
                    timeout_s=args.timeout,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                print(f"[pred_llama] error idx={i}: {e}", file=sys.stderr)
                resp = {"content": "", "error": str(e)}

            dt = time.time() - t0
            record = {
                "index": task.get("index", i),
                "input": prompt if args.echo_input else None,
                "pred": resp.get("content", ""),
                "outputs": task.get("outputs"),  # gold answers, passed through
                "task": task.get("task"),
                "length": task.get("length"),
                "latency_s": round(dt, 3),
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            written += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                print(
                    f"[pred_llama] {i+1}/{len(tasks)} ({rate:.2f}/s)",
                    file=sys.stderr,
                )

    elapsed = time.time() - t_start
    print(
        f"[pred_llama] done: {written} preds in {elapsed:.1f}s → {out_path}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="POST tasks to llama-server /completion and dump predictions JSONL.",
    )
    p.add_argument("--server-url", required=True, help="Base URL of llama-server")
    p.add_argument("--input", required=True, help="Input tasks JSONL")
    p.add_argument("--output", required=True, help="Output preds JSONL")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--stop", nargs="*", default=None, help="Stop strings")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s)")
    p.add_argument("--limit", type=int, default=0, help="Cap number of samples (0=all)")
    p.add_argument("--echo-input", action="store_true", help="Include prompt in output")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
