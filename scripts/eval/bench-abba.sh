#!/usr/bin/env bash
# llama-bench of two builds in the order A, B, B, A per model and K/V pair.
#
# usage: BIN_A=dir-a BIN_B=dir-b MODEL=model.gguf scripts/eval/bench-abba.sh
#
# On the GPUs these numbers were taken on, the first run of a round is 2-3 % faster than the next one (warm-up and
# clocks), so every cell is the mean of one run of each build in first and in second position.
#
# Environment:
#   BIN_A, BIN_B  directories that contain llama-bench (required); A is the reference (e.g. upstream)
#   MODEL         one or more GGUF files, separated by spaces (required)
#   KV            K/V pairs run A,B,B,A (default "f16/f16 q8_0/q8_0")
#   KV_B_ONLY     K/V pairs run once on B only, after the A,B,B,A pairs of a model (types A does not have),
#                 e.g. "q8_0/vtq4_1 ktq4_1/vtq4_1 ktq4/vtq4 ktq3/vtq3" (default empty)
#   ARGS          llama-bench arguments (default "-p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1")
#   THREADS       -t (default 4); every run also gets -ngl 99 -fa 1
#   GPUS          CUDA_VISIBLE_DEVICES (default: inherited)
#   TIMEOUT       seconds per llama-bench run (default 3600)
#   OUT           result directory (default bench-abba-<date-time>)
#
# Output: OUT/A/*.jsonl and OUT/B/*.jsonl (raw llama-bench JSONL), OUT/*.err (stderr), OUT/summary.md (mean per
# build and test, ratio B / A; made by bench-table.py).
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${BIN_A:?set BIN_A to the directory with the reference llama-bench}"
: "${BIN_B:?set BIN_B to the directory with the llama-bench to compare}"
: "${MODEL:?set MODEL to one or more GGUF files}"
KV=${KV:-f16/f16 q8_0/q8_0}
KV_B_ONLY=${KV_B_ONLY:-}
ARGS=${ARGS:--p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1}
THREADS=${THREADS:-4}
TIMEOUT=${TIMEOUT:-3600}
OUT=${OUT:-bench-abba-$(date +%Y%m%d-%H%M%S)}
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$OUT/A" "$OUT/B"

for b in "$BIN_A" "$BIN_B"; do
    [ -x "$b/llama-bench" ] || { echo "no llama-bench in $b" >&2; exit 1; }
done

run() {  # run <A|B> <model> <K/V> <round>
    local bin k=${3%/*} v=${3#*/} name
    [ "$1" = A ] && bin=$BIN_A || bin=$BIN_B
    name=$(basename "$2" .gguf)-$k-$v-r$4
    echo "[$(date +%H:%M:%S)] $1 $(basename "$2") $k/$v round $4" >&2
    # shellcheck disable=SC2086  # ARGS is a word list on purpose
    LD_LIBRARY_PATH="$bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" timeout "$TIMEOUT" "$bin/llama-bench" -m "$2" \
        -ngl 99 -fa 1 -t "$THREADS" -ctk "$k" -ctv "$v" $ARGS -o jsonl \
        > "$OUT/$1/$name.jsonl" 2> "$OUT/$1-$name.err" || echo "  exit $? (see $OUT/$1-$name.err)" >&2
}

for m in $MODEL; do
    for kv in $KV; do
        run A "$m" "$kv" 1; run B "$m" "$kv" 1; run B "$m" "$kv" 2; run A "$m" "$kv" 2
    done
    for kv in $KV_B_ONLY; do run B "$m" "$kv" 1; done
done

python3 "$HERE/bench-table.py" "$OUT"/A/*.jsonl "$OUT"/B/*.jsonl | tee "$OUT/summary.md"
