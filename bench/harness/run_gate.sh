#!/usr/bin/env bash
# run_gate.sh — Orchestrator for RULER + LongBench harness.
#
# Reads profiles/<profile>.yaml, sequentially runs prepare → predict → score
# for each (task × length) combination, and compares results to a baseline
# JSON if one exists at baselines/<model_tag>.json.
#
# Usage:
#   ./run_gate.sh --profile smoke [--server-url URL] [--model-tag TAG]
#                 [--use-vendor] [--tokenizer PATH] [--dry-run]
#
# Examples:
#   ./run_gate.sh --profile smoke --dry-run
#   ./run_gate.sh --profile smoke --server-url http://localhost:8080 \
#                 --model-tag qwen35-0.8b-q8
#
# Constraints (see README.md / project memory):
#   - Never point --server-url at the live deploy on gpu00:8791.
#   - Never run two gates in parallel on the same gpu00 host.
#   - cache_prompt is forced false inside pred_llama.py — do not weaken.
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE=""
SERVER_URL="http://localhost:8080"
MODEL_TAG=""
USE_VENDOR=0
TOKENIZER=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)     PROFILE="$2"; shift 2 ;;
        --server-url)  SERVER_URL="$2"; shift 2 ;;
        --model-tag)   MODEL_TAG="$2"; shift 2 ;;
        --use-vendor)  USE_VENDOR=1; shift ;;
        --tokenizer)   TOKENIZER="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PROFILE}" ]]; then
    echo "ERROR: --profile required (smoke|full)" >&2
    exit 2
fi

PROFILE_YAML="${HARNESS_DIR}/profiles/${PROFILE}.yaml"
[[ -f "${PROFILE_YAML}" ]] || { echo "Missing profile: ${PROFILE_YAML}" >&2; exit 2; }

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${HARNESS_DIR}/runs/${TIMESTAMP}"

# --- Parse YAML profile (stdlib python; no pyyaml dep) ---
read_profile() {
    python3 - "$PROFILE_YAML" <<'PYEOF'
import sys, json, re
p = open(sys.argv[1], "r", encoding="utf-8").read()
# Minimal YAML: extract ruler.tasks, ruler.lengths, ruler.samples_per_length,
# longbench.enabled, longbench.tasks, longbench.samples_per_task.
def section(name):
    m = re.search(rf"^{name}:\s*\n((?:[ \t]+.*\n?)+)", p, re.MULTILINE)
    return m.group(1) if m else ""
def list_field(blk, key):
    m = re.search(rf"^[ \t]*{key}:\s*\n((?:[ \t]+-\s*.+\n?)+)", blk, re.MULTILINE)
    if not m: return []
    return [l.strip()[1:].strip() for l in m.group(1).splitlines() if l.strip().startswith("-")]
def scalar(blk, key, default=None):
    m = re.search(rf"^[ \t]*{key}:\s*([^\n#]+)", blk, re.MULTILINE)
    return m.group(1).strip() if m else default
ruler = section("ruler")
lb = section("longbench")
out = {
  "ruler_tasks": list_field(ruler, "tasks"),
  "ruler_lengths": [int(x) for x in list_field(ruler, "lengths")],
  "ruler_samples": int(scalar(ruler, "samples_per_length", "10")),
  "ruler_max_new_tokens": int(scalar(ruler, "max_new_tokens", "128")),
  "lb_enabled": (scalar(lb, "enabled", "false").lower() == "true"),
  "lb_tasks": list_field(lb, "tasks"),
  "lb_samples": int(scalar(lb, "samples_per_task", "50")),
}
print(json.dumps(out))
PYEOF
}

CFG_JSON="$(read_profile)"
RULER_TASKS=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(' '.join(json.load(sys.stdin)['ruler_tasks']))")
RULER_LENGTHS=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(' '.join(str(x) for x in json.load(sys.stdin)['ruler_lengths']))")
RULER_N=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['ruler_samples'])")
RULER_MAXTOK=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['ruler_max_new_tokens'])")
LB_ENABLED=$(echo "$CFG_JSON" | python3 -c "import sys,json;print('1' if json.load(sys.stdin)['lb_enabled'] else '0')")
LB_TASKS=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(' '.join(json.load(sys.stdin)['lb_tasks']))")
LB_N=$(echo "$CFG_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['lb_samples'])")

cat <<EOF
=== run_gate plan ===
profile        : ${PROFILE}
server_url     : ${SERVER_URL}
model_tag      : ${MODEL_TAG:-<none>}
use_vendor     : ${USE_VENDOR}
tokenizer      : ${TOKENIZER:-<none>}
dry_run        : ${DRY_RUN}
run_dir        : ${RUN_DIR}
ruler tasks    : ${RULER_TASKS:-<none>}
ruler lengths  : ${RULER_LENGTHS:-<none>}
ruler n        : ${RULER_N}
ruler max_new  : ${RULER_MAXTOK}
longbench on   : ${LB_ENABLED}
lb tasks       : ${LB_TASKS:-<none>}
lb n           : ${LB_N}
=====================
EOF

# Compute total inferences for budget visibility.
RULER_TASK_COUNT=$(echo "$RULER_TASKS" | wc -w)
RULER_LEN_COUNT=$(echo "$RULER_LENGTHS" | wc -w)
RULER_TOTAL=$(( RULER_TASK_COUNT * RULER_LEN_COUNT * RULER_N ))
LB_TASK_COUNT=$(echo "$LB_TASKS" | wc -w)
LB_TOTAL=0
[[ "${LB_ENABLED}" == "1" ]] && LB_TOTAL=$(( LB_TASK_COUNT * LB_N ))
echo "estimated inferences: ruler=${RULER_TOTAL} longbench=${LB_TOTAL} total=$(( RULER_TOTAL + LB_TOTAL ))"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] no work executed. Exit 0."
    exit 0
fi

mkdir -p "${RUN_DIR}/ruler" "${RUN_DIR}/longbench" "${RUN_DIR}/summary"

# --- RULER loop ---
for task in ${RULER_TASKS}; do
    for length in ${RULER_LENGTHS}; do
        tag="${task}_${length}"
        TASKS_JSONL="${RUN_DIR}/ruler/${tag}.jsonl"
        PREDS_JSONL="${RUN_DIR}/ruler/${tag}.preds.jsonl"
        SUMMARY_CSV="${RUN_DIR}/summary/ruler_${tag}.csv"

        echo "[run_gate] RULER prepare ${tag}" >&2
        VENDOR_FLAGS=""
        [[ "${USE_VENDOR}" == "1" ]] && VENDOR_FLAGS="--use-vendor --tokenizer-path ${TOKENIZER}"
        # shellcheck disable=SC2086
        python3 "${HARNESS_DIR}/prepare_ruler.py" \
            --task "${task}" --length "${length}" \
            --num-samples "${RULER_N}" --output "${TASKS_JSONL}" \
            ${VENDOR_FLAGS}

        echo "[run_gate] RULER predict ${tag}" >&2
        python3 "${HARNESS_DIR}/pred_llama.py" \
            --server-url "${SERVER_URL}" \
            --input "${TASKS_JSONL}" --output "${PREDS_JSONL}" \
            --max-new-tokens "${RULER_MAXTOK}" --temperature 0.0

        echo "[run_gate] RULER score ${tag}" >&2
        python3 "${HARNESS_DIR}/score_ruler.py" \
            --preds "${PREDS_JSONL}" --summary "${SUMMARY_CSV}"
    done
done

# --- LongBench loop ---
if [[ "${LB_ENABLED}" == "1" ]]; then
    for task in ${LB_TASKS}; do
        TASKS_JSONL="${RUN_DIR}/longbench/${task}.jsonl"
        PREDS_JSONL="${RUN_DIR}/longbench/${task}.preds.jsonl"
        RESULTS_JSON="${RUN_DIR}/summary/longbench_${task}.json"

        echo "[run_gate] LongBench prepare ${task}" >&2
        python3 "${HARNESS_DIR}/prepare_longbench.py" \
            --task "${task}" --num-samples "${LB_N}" --output "${TASKS_JSONL}"

        echo "[run_gate] LongBench predict ${task}" >&2
        python3 "${HARNESS_DIR}/pred_llama.py" \
            --server-url "${SERVER_URL}" \
            --input "${TASKS_JSONL}" --output "${PREDS_JSONL}" \
            --max-new-tokens 256 --temperature 0.0

        echo "[run_gate] LongBench score ${task}" >&2
        python3 "${HARNESS_DIR}/score_longbench.py" \
            --preds "${PREDS_JSONL}" --task "${task}" --results "${RESULTS_JSON}"
    done
fi

# --- Baseline comparison ---
if [[ -n "${MODEL_TAG}" ]]; then
    BASELINE="${HARNESS_DIR}/baselines/${MODEL_TAG}.json"
    if [[ -f "${BASELINE}" ]]; then
        echo "[run_gate] comparing against baseline ${BASELINE}" >&2
        python3 - "${RUN_DIR}" "${BASELINE}" <<'PYEOF'
import csv, glob, json, os, sys
run_dir, baseline_path = sys.argv[1], sys.argv[2]
with open(baseline_path) as fh: base = json.load(fh)

# Aggregate RULER per-length averages.
ruler_per_len = {}
for path in glob.glob(os.path.join(run_dir, "summary", "ruler_*.csv")):
    name = os.path.basename(path)[len("ruler_"):-len(".csv")]
    task, _, length = name.rpartition("_")
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try: acc = float(row["accuracy"]) * 100
            except Exception: continue
            ruler_per_len.setdefault(length, []).append(acc)

print("\n=== gate report ===")
for length, accs in sorted(ruler_per_len.items(), key=lambda x: int(x[0])):
    avg = sum(accs)/len(accs) if accs else 0.0
    base_key = f"ruler_{int(length)//1024}k_avg"
    base_val = (base.get("ruler") or {}).get(base_key)
    delta = (avg - base_val) if base_val is not None else None
    status = "PASS" if (delta is None or delta >= -2.0) else "FAIL"
    print(f"{base_key:24s} now={avg:6.2f}  base={base_val}  Δ={delta}  {status}")

lb_results = []
for path in glob.glob(os.path.join(run_dir, "summary", "longbench_*.json")):
    with open(path) as fh: lb_results.append(json.load(fh))
if lb_results:
    avg = sum(r["score"] for r in lb_results)/len(lb_results)
    base_val = (base.get("longbench") or {}).get("longbench_en_avg")
    delta = (avg - base_val) if base_val is not None else None
    status = "PASS" if (delta is None or delta >= -1.5) else "FAIL"
    print(f"longbench_en_avg         now={avg:6.2f}  base={base_val}  Δ={delta}  {status}")
PYEOF
    else
        echo "[run_gate] no baseline at ${BASELINE} — skipping gate comparison" >&2
    fi
fi

echo "[run_gate] done. Run dir: ${RUN_DIR}"
