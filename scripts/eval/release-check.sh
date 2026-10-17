#!/usr/bin/env bash
# Release check of a CUDA build on one GPU: TurboQuant op tests, then (optional) llama-bench and KL divergence.
#
# usage: BIN=build/bin scripts/eval/release-check.sh
#        BIN=build/bin MODEL=Qwen3-4B-Instruct-2507-Q4_K_M.gguf WIKITEXT=wikitext-2-raw/wiki.test.raw scripts/eval/release-check.sh
#
# Steps (each test-backend-ops group with its own time limit; a group that hits it is reported with exit=124):
#   1  test-backend-ops on the GPU: GET_ROWS f16 (smoke), SET_ROWS / GET_ROWS / CPY of the TurboQuant types,
#      FLASH_ATTN_EXT with head size 512, then
#      MODE=full: a flash-attention matrix at head size 128 and the GQA decode cases (time-capped),
#      MODE=ops:  every flash-attention case with a TurboQuant K or V, every flash-attention case, every op
#   2  MODE=full and MODEL set: llama-bench of the BENCH_KV pairs (-p 512 -n 128 -d 0,32768 -r 2), the list forwards
#      and then backwards, plus verify batches (-p 2,4,8,16 -n 0 -d 32768) of the VERIFY_KV pairs, both rounds
#      averaged in OUT/bench.md
#   3  MODE=full, MODEL and WIKITEXT set: KL divergence against an f16 cache, 40 chunks of 512 tokens, the cache
#      quantized during prefill (--no-tq-deferred-k --no-tq-deferred-v), for the KLD_KV pairs
# Every test log line with FAIL, a CUDA error, a misaligned or illegal address, an abort or GGML_ASSERT is printed.
# Exit code 1 if any test group reported a failure.
#
# Environment:
#   BIN        directory with test-backend-ops, llama-bench and llama-perplexity (required)
#   MODE       full (default) or ops
#   GPU        CUDA device to test (default 0; the backend is then CUDA0)
#   MODEL      GGUF for llama-bench and KLD (optional; the docs use Qwen3-4B-Instruct-2507 Q4_K_M)
#   WIKITEXT   text for KLD (optional; scripts/get-wikitext-2.sh downloads wikitext-2-raw/wiki.test.raw)
#   BENCH_KV, VERIFY_KV, KLD_KV  K/V pair lists (defaults below)
#   THREADS    -t (default 4), OUT  result directory (default release-check-<date-time>)
#   KLD_BASE   path of the f16 base file (default OUT/kld/base.kld; a few GB for a 150k vocabulary)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${BIN:?set BIN to the directory with test-backend-ops, llama-bench and llama-perplexity}"
MODE=${MODE:-full}
GPU=${GPU:-0}
MODEL=${MODEL:-}
WIKITEXT=${WIKITEXT:-}
THREADS=${THREADS:-4}
OUT=${OUT:-release-check-$(date +%Y%m%d-%H%M%S)}
KLD_BASE=${KLD_BASE:-$OUT/kld/base.kld}
BENCH_KV=${BENCH_KV:-f16/f16 q8_0/q8_0 q4_0/q4_0 ktq2_1/vtq2_1 ktq3_1/vtq3_1 ktq4_1/vtq4_1 q8_0/vtq4_1 ktq4/vtq4 ktq3/vtq3 ktq2/vtq2}
VERIFY_KV=${VERIFY_KV:-q8_0/q8_0 q8_0/vtq4_1 ktq4_1/vtq4_1 ktq3/vtq3}
KLD_KV=${KLD_KV:-q8_0/q8_0 ktq4/vtq4 ktq3/vtq3 ktq2/vtq2 ktq4_1/vtq4_1 ktq3_1/vtq3_1 ktq2_1/vtq2_1 q8_0/vtq4 q8_0/vtq4_1}
case $MODE in full|ops) ;; *) echo "MODE must be full or ops" >&2; exit 1 ;; esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU  # same numbering as nvidia-smi
export LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$OUT/bench" "$(dirname "$KLD_BASE")"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
command -v nvidia-smi > /dev/null && nvidia-smi -i "$GPU" --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv,noheader | tee -a "$OUT/run.log"
log "mode $MODE, $("$BIN/llama-perplexity" --version 2>&1 | grep -m1 -i version)"

FAILED=0
ops() {  # ops <name> <op|-> <pattern|-> <timeout s>
    local a=(-b CUDA0) t0 e
    [ "$2" != - ] && a+=(-o "$2")
    [ "$3" != - ] && a+=(-p "$3")
    t0=$(date +%s)
    timeout "$4" "$BIN/test-backend-ops" "${a[@]}" > "$OUT/test-$1.log" 2>&1
    e=$?
    log "$1 exit=$e $(( $(date +%s) - t0 ))s $(grep -aE 'tests passed' "$OUT/test-$1.log" | tail -1)" \
        "ok=$(grep -ac 'OK' "$OUT/test-$1.log") fail=$(grep -ac 'FAIL' "$OUT/test-$1.log")" \
        "notsup=$(grep -ac 'not supported' "$OUT/test-$1.log")"
    if grep -aqE 'FAIL|CUDA error|misaligned|illegal|Abort|GGML_ASSERT' "$OUT/test-$1.log"; then
        FAILED=1
        grep -aE 'FAIL|CUDA error|misaligned|illegal|Abort|GGML_ASSERT' "$OUT/test-$1.log" | head -8 | cut -c1-300 \
            | tee -a "$OUT/run.log"
    fi
}

# 1) op tests ('[kv]tq' matches every TurboQuant K and V type)
ops smoke    GET_ROWS       'type=f16' 300
ops setrows  SET_ROWS       '[kv]tq'   1500
ops getrows  GET_ROWS       '[kv]tq'   900
ops cpy      CPY            '[kv]tq'   900
ops fa-d512  FLASH_ATTN_EXT 'hsk=512'  1800
if [ "$MODE" = ops ]; then
    ops fa-tq   FLASH_ATTN_EXT 'type_[KV]=[kv]tq' 4500
    ops fa-all  FLASH_ATTN_EXT -                  3600
    ops all-ops -              -                  3600
    log "done, failures: $FAILED"
    exit $FAILED
fi
ops fa-matrix FLASH_ATTN_EXT 'hsk=128,hsv=128,nh=2,nr23=\[4,1\],kv=256,nb=(1|32),' 2400
ops fa-gqa    FLASH_ATTN_EXT 'kv=1024,nb=1,'                                        900
log "op tests done, failures: $FAILED"
[ -z "$MODEL" ] && { log "MODEL not set: no llama-bench and KLD"; exit $FAILED; }

# 2) llama-bench; round 2 runs the pair list backwards (the first runs on a GPU are a few % faster)
bench() {  # bench <prefix> <K/V> <round> <args...>
    local k=${2%/*} v=${2#*/} p=$1 r=$3
    shift 3
    "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -t "$THREADS" -ctk "$k" -ctv "$v" "$@" -r 2 -o jsonl \
        > "$OUT/bench/$p-$k-$v-r$r.jsonl" 2>> "$OUT/bench-err.log" || log "llama-bench $k/$v exit $?"
}
for r in 1 2; do
    L=$BENCH_KV; V=$VERIFY_KV
    [ $r = 2 ] && { L=$(tr ' ' '\n' <<< "$L" | tac | tr '\n' ' '); V=$(tr ' ' '\n' <<< "$V" | tac | tr '\n' ' '); }
    for kv in $L; do bench speed "$kv" $r -p 512 -n 128 -d 0,32768; done
    for kv in $V; do bench verify "$kv" $r -p 2,4,8,16 -n 0 -d 32768; done
done
python3 "$HERE/bench-table.py" "$OUT"/bench/*.jsonl > "$OUT/bench.md"
log "llama-bench done: $OUT/bench.md"

# 3) KL divergence against f16
[ -z "$WIKITEXT" ] && { log "WIKITEXT not set: no KLD"; exit $FAILED; }
P="$BIN/llama-perplexity"
KA=(-m "$MODEL" -f "$WIKITEXT" -ngl 99 -fa on -t "$THREADS" -c 512 -b 512 -ub 512 --chunks 40)
if [ ! -s "$KLD_BASE" ]; then
    "$P" "${KA[@]}" --kl-divergence-base "$KLD_BASE" > "$OUT/kld/base.log" 2>&1
    log "kld base exit=$? $(grep -oE 'Final estimate: PPL = [0-9.]+' "$OUT/kld/base.log")"
fi
for kv in $KLD_KV; do
    k=${kv%/*}; v=${kv#*/}
    "$P" "${KA[@]}" -ctk "$k" -ctv "$v" --no-tq-deferred-k --no-tq-deferred-v \
        --kl-divergence-base "$KLD_BASE" --kl-divergence > "$OUT/kld/$k-$v.log" 2>&1
    log "kld $k/$v exit=$? $(grep -E '^Mean +KLD' "$OUT/kld/$k-$v.log" | head -1 | tr -s ' ')" \
        "| $(grep -E '^Same top p' "$OUT/kld/$k-$v.log" | head -1 | tr -s ' ')"
done
log "done, failures: $FAILED"
exit $FAILED
