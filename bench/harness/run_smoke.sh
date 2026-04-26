#!/usr/bin/env bash
# run_smoke.sh — End-to-end smoke test of the harness.
#
# Runs niah_single_3 at 4k, n=10, against a llama-server URL.
# Intended target for first pass: a locally-spun llama-server with
# qwen3.5-0.8b-q8_0.gguf (the smallest test model). DO NOT point this
# at the live deploy on gpu00:8791.
#
# Usage:
#   ./run_smoke.sh [SERVER_URL]
#
# Default SERVER_URL: http://localhost:8080
set -euo pipefail

SERVER_URL="${1:-http://localhost:8080}"
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${HARNESS_DIR}/work/smoke"
mkdir -p "${WORK_DIR}/tasks" "${WORK_DIR}/preds" "${WORK_DIR}/summary"

TASKS_JSONL="${WORK_DIR}/tasks/niah_single_3_4k.jsonl"
PREDS_JSONL="${WORK_DIR}/preds/niah_single_3_4k.preds.jsonl"
SUMMARY_CSV="${WORK_DIR}/summary/smoke.csv"

# --- Generate placeholder NIAH-single-3 tasks (vendor RULER prepare step pending) ---
# Real NIAH samples come from NVIDIA/RULER prepare.py. Until vendored, emit a
# tiny synthetic stand-in so the wiring (pred + score) is exercised end-to-end.
if [[ ! -s "${TASKS_JSONL}" ]]; then
    echo "[smoke] generating 10 placeholder niah_single_3 samples → ${TASKS_JSONL}" >&2
    python3 - <<PYEOF "${TASKS_JSONL}"
import json, random, sys
random.seed(42)
out = sys.argv[1]
needles = [f"key-{i:04d}-VALUE-{random.randint(1000,9999)}" for i in range(10)]
with open(out, "w", encoding="utf-8") as fh:
    for i, n in enumerate(needles):
        haystack = ". ".join(f"Filler sentence {j} about an unrelated topic." for j in range(80))
        prompt = (
            f"You are given a long document. Find the secret key.\n\n"
            f"<document>\n{haystack}\nThe secret key is {n}.\n{haystack}\n</document>\n\n"
            f"What is the secret key? Answer with the key only."
        )
        rec = {
            "index": i,
            "task": "niah_single_3",
            "length": 4096,
            "input": prompt,
            "outputs": [n],
        }
        fh.write(json.dumps(rec) + "\n")
PYEOF
fi

echo "[smoke] running pred_llama against ${SERVER_URL}" >&2
python3 "${HARNESS_DIR}/pred_llama.py" \
    --server-url "${SERVER_URL}" \
    --input "${TASKS_JSONL}" \
    --output "${PREDS_JSONL}" \
    --max-new-tokens 32 \
    --temperature 0.0 \
    --stop $'\n' \
    --limit 10

echo "[smoke] scoring → ${SUMMARY_CSV}" >&2
python3 "${HARNESS_DIR}/score_ruler.py" \
    --preds "${PREDS_JSONL}" \
    --summary "${SUMMARY_CSV}"

echo
echo "=== SMOKE SUMMARY ==="
cat "${SUMMARY_CSV}"
