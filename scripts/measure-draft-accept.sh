#!/bin/bash
# measure-draft-accept.sh — measure speculative-decoding token acceptance rate
# against a running llama-tq server that was started with --model-draft.
#
# Usage:
#   ./scripts/measure-draft-accept.sh [--server URL] [--samples N] [--max-tokens N]
#
# Defaults:
#   --server       http://localhost:8791
#   --samples      20
#   --max-tokens   128
#
# Output:
#   Per-prompt JSON line with accept-rate, aggregate at the end.
#
# Requires: server started with `--model-draft PATH --draft-max N --draft-min M`.
# If draft is not configured, all accept rates will be 0 (silent; no error).

set -euo pipefail

SERVER_URL="${SERVER_URL:-http://localhost:8791}"
SAMPLES=20
MAX_TOKENS=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)     SERVER_URL="$2";  shift 2;;
        --samples)    SAMPLES="$2";     shift 2;;
        --max-tokens) MAX_TOKENS="$2";  shift 2;;
        -h|--help)    grep '^#' "$0" | sed 's/^# \?//' | head -20; exit 0;;
        *)            echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# Prompt mix representative of Claude Code workload:
#  - tool-use JSON emission (high structure, high draft-accept)
#  - code completion (medium)
#  - free-form explanation (low-structure, lower accept)
PROMPTS=(
    'def fibonacci(n):\n    '
    '{"name": "Read", "input": {"path": "'
    'import numpy as np\nimport pandas as pd\n\ndef '
    '// TypeScript utility to merge two objects\nfunction merge<T>('
    'class BinaryTree:\n    def __init__(self, value):\n        '
    '{"tool_use": {"name": "Bash", "input": {"command": "'
    'SELECT users.id, COUNT(orders.id) AS n FROM users LEFT JOIN '
    'describe how to implement a red-black tree in rust'
    'what are the tradeoffs between gRPC and REST'
    '{"name": "Grep", "input": {"pattern": "'
    'public class HttpServer {\n    private final int port;\n    '
    'impl Display for Error {\n    fn fmt(&self, f: &mut '
    '- [ ] Review PR\n- [ ] Update docs\n- [ ] '
    'explain quicksort step by step for a 10-year-old'
    'async function fetchUser(id: string): Promise<User> {\n    '
    '{"tool_calls":[{"type":"function","function":{"name":"Edit","arguments":"'
    'package main\n\nimport (\n\t"context"\n\t"fmt"\n)\n\nfunc main() {\n\t'
    'CREATE TABLE messages (\n    id BIGSERIAL PRIMARY KEY,\n    '
    '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n    '
    'Dockerfile:\nFROM python:3.11-slim\nWORKDIR /app\n'
)

# Trim to SAMPLES
PROMPTS=("${PROMPTS[@]:0:$SAMPLES}")

printf '{"server":"%s","samples":%d,"max_tokens":%d,"results":[\n' \
    "$SERVER_URL" "${#PROMPTS[@]}" "$MAX_TOKENS"

TOTAL_DRAFT=0
TOTAL_ACCEPT=0
TOTAL_TOKENS=0
TOTAL_MS=0
FIRST=1

for prompt in "${PROMPTS[@]}"; do
    body=$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': $MAX_TOKENS, 'stream': False, 'cache_prompt': False}))" "$prompt")
    resp=$(curl -s -X POST "$SERVER_URL/v1/completions" \
        -H "Content-Type: application/json" \
        --data "$body")

    [ "$FIRST" -eq 1 ] && FIRST=0 || printf ',\n'

    python3 <<PYEOF
import json, sys
r = json.loads('''$resp''')
t = r.get('timings', {}) if isinstance(r, dict) else {}
draft_n      = int(t.get('draft_n', 0) or 0)
draft_accept = int(t.get('draft_n_accept', 0) or 0)
predicted    = int(t.get('predicted_n', 0) or 0)
predict_ms   = float(t.get('predicted_ms', 0) or 0)
tps          = float(t.get('predicted_per_second', 0) or 0)
accept_rate  = (100.0 * draft_accept / draft_n) if draft_n else 0.0
out = {
    "prompt_preview": '''$prompt'''[:50],
    "predicted_n":   predicted,
    "predict_ms":    round(predict_ms, 1),
    "tok_s":         round(tps, 2),
    "draft_n":       draft_n,
    "draft_accept":  draft_accept,
    "accept_rate":   round(accept_rate, 1)
}
print("  " + json.dumps(out), end="")
PYEOF

    # Accumulate via python for reliable JSON parsing
    read -r dn da pn pm <<< "$(python3 -c "
import json
r = json.loads('''$resp''')
t = r.get('timings', {}) if isinstance(r, dict) else {}
print(int(t.get('draft_n', 0) or 0), int(t.get('draft_n_accept', 0) or 0), int(t.get('predicted_n', 0) or 0), float(t.get('predicted_ms', 0) or 0))
")"
    TOTAL_DRAFT=$((TOTAL_DRAFT + dn))
    TOTAL_ACCEPT=$((TOTAL_ACCEPT + da))
    TOTAL_TOKENS=$((TOTAL_TOKENS + pn))
    TOTAL_MS=$(python3 -c "print($TOTAL_MS + $pm)")
done

printf '\n],\n'

# Aggregate
if [ "$TOTAL_DRAFT" -gt 0 ]; then
    AGG_ACCEPT=$(python3 -c "print(round(100 * $TOTAL_ACCEPT / $TOTAL_DRAFT, 2))")
else
    AGG_ACCEPT=0
fi
AGG_TPS=$(python3 -c "print(round($TOTAL_TOKENS * 1000 / max(1, $TOTAL_MS), 2))")

printf '"aggregate": {\n'
printf '  "total_draft_tokens":  %d,\n' "$TOTAL_DRAFT"
printf '  "total_accept_tokens": %d,\n' "$TOTAL_ACCEPT"
printf '  "accept_rate_pct":     %s,\n' "$AGG_ACCEPT"
printf '  "total_predicted":     %d,\n' "$TOTAL_TOKENS"
printf '  "total_predict_ms":    %s,\n' "$TOTAL_MS"
printf '  "mean_tok_s":          %s\n' "$AGG_TPS"
printf '}}\n'

# Human-readable summary to stderr
{
    echo ""
    echo "=== Aggregate ==="
    echo "  accept_rate:   ${AGG_ACCEPT}% (${TOTAL_ACCEPT}/${TOTAL_DRAFT} draft tokens accepted)"
    echo "  mean tok/s:    ${AGG_TPS}"
    echo "  total tokens:  ${TOTAL_TOKENS}"
    if [ "$TOTAL_DRAFT" -eq 0 ]; then
        echo ""
        echo "  WARNING: draft_n=0 for all samples. Server was probably started"
        echo "  without --model-draft. Restart with:"
        echo "    llama-server ... --model-draft PATH --draft-max 8 --draft-min 3"
    fi
} >&2
