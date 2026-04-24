#!/usr/bin/env bash
# baseline.sh — generate baseline.json with reference PPL+TG for current
# deployable configs. Run once per major code change.
set -euo pipefail

OUT_DIR="$(dirname "$(realpath "$0")")/.."
mkdir -p "$OUT_DIR"

configs=(
    "f16 f16"
    "ktq1_1 vtq1_1"
    "ktq2_1 vtq2_1"
    "ktq3_1 vtq3_1"
    "ktq4_1 vtq4_1"
    "ktq2_1 vtq3_1"
    "ktq2_1 vtq2_2"
)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "[" > "$OUT_DIR/baseline.json"
first=1
for cfg in "${configs[@]}"; do
    K=${cfg% *}
    V=${cfg#* }
    sub="$tmp/${K}_${V}"
    mkdir -p "$sub"
    if ! bash "$(dirname "$0")/measure.sh" "$K" "$V" "$sub"; then
        echo "WARN: measure failed for $K/$V — skipping" >&2
        continue
    fi
    [ $first -eq 1 ] || echo "," >> "$OUT_DIR/baseline.json"
    cat "$sub/metrics.json" >> "$OUT_DIR/baseline.json"
    first=0
done
echo "]" >> "$OUT_DIR/baseline.json"

echo
echo "Wrote $OUT_DIR/baseline.json"
