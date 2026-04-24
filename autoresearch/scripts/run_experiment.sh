#!/usr/bin/env bash
# run_experiment.sh — apply a patch, build, measure, score, auto-revert if worse.
#
# Usage: run_experiment.sh <patch_file> <experiment_name> <k_type> <v_type> [score_threshold]
#
# score_threshold (default 0.9) — keep if score < threshold * best_previous_score.
set -euo pipefail

PATCH="${1:?need patch file}"
NAME="${2:?need experiment name}"
K="${3:?need K type}"
V="${4:?need V type}"
THRESHOLD="${5:-0.9}"

ROOT="$(realpath "$(dirname "$0")/../..")"
EXP_DIR="$ROOT/autoresearch/experiments/$(date +%Y-%m-%d-%H%M)-$NAME"
mkdir -p "$EXP_DIR"

cp "$PATCH" "$EXP_DIR/patch.diff"

cd "$ROOT"

# Snapshot HEAD so we can revert
pre_commit=$(git rev-parse HEAD)
echo "pre_commit=$pre_commit" > "$EXP_DIR/params.json"
echo "patch=$PATCH" >> "$EXP_DIR/params.json"

# Apply
if ! git apply --check "$PATCH"; then
    echo "REVERT: patch does not apply cleanly" > "$EXP_DIR/decision.txt"
    exit 2
fi
git apply "$PATCH"

# Build (build failure = auto-revert)
if ! cmake --build build -j2 --target llama-perplexity llama-bench 2>&1 | tee "$EXP_DIR/build.log"; then
    git checkout -- .
    echo "REVERT: build failed" > "$EXP_DIR/decision.txt"
    exit 3
fi

# Measure
bash "$(dirname "$0")/measure.sh" "$K" "$V" "$EXP_DIR"
score=$(python3 -c "import json; print(json.load(open('$EXP_DIR/metrics.json'))['score'])")

# Compare against best previous score (if exists)
best_file="$ROOT/autoresearch/best_score.txt"
if [ -f "$best_file" ]; then
    best=$(cat "$best_file")
    ratio=$(python3 -c "print($score / $best)")
    if python3 -c "import sys; sys.exit(0 if $score < $THRESHOLD * $best else 1)"; then
        echo "KEEP: score=$score < $THRESHOLD × best=$best" > "$EXP_DIR/decision.txt"
        echo "$score" > "$best_file"
        # Optional: commit the winning patch
        git add -A && git commit -m "autoresearch: $NAME (score $score)" || true
    else
        echo "REVERT: score=$score >= $THRESHOLD × best=$best (ratio=$ratio)" > "$EXP_DIR/decision.txt"
        git checkout -- .
    fi
else
    echo "KEEP: first experiment, score=$score" > "$EXP_DIR/decision.txt"
    echo "$score" > "$best_file"
fi

cat "$EXP_DIR/decision.txt"
