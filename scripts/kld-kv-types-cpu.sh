#!/usr/bin/env bash
# KL divergence of K-cache (and optionally V-cache) types against an f16 cache, on the CPU backend.
#
# usage: scripts/kld-kv-types-cpu.sh MODEL.gguf wiki.test.raw OUT_DIR
#
# environment (defaults in brackets):
#   BIN      directory with llama-perplexity [build/bin]
#   THREADS  CPU threads [all cores]
#   CHUNKS   chunks of CTX tokens [20]
#   CTX      context per chunk [512]
#   K_TYPES  K types to score, V stays f16 [q8_0 q4_0 ktq2_1 ktq3_1 ktq4_1 ktq2_2s ktq3_2s ktq4_2s]
#   V_TYPES  V types to score, K stays f16, for reference [empty = skip], e.g. "vtq2_1 vtq3_1 vtq2_2s vtq3_2s vtq4_2s"
#
# Every run quantizes the cache for the whole evaluation (--no-tq-deferred-k/-v: no f16 staging). The base logits
# come from an f16 K/V run; bpw is derived from the cache size that llama-perplexity logs for that run.
# Prints one line per type: bpw, mean KLD, same top p, PPL, wall time. Logs: OUT_DIR/<side>-<type>.log

set -euo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,15p' "$0"
    exit 1
fi

MODEL=$1
DATA=$2
OUT=$3
BIN=${BIN:-build/bin}
THREADS=${THREADS:-$(nproc)}
CHUNKS=${CHUNKS:-20}
CTX=${CTX:-512}
K_TYPES=${K_TYPES:-"q8_0 q4_0 ktq2_1 ktq3_1 ktq4_1 ktq2_2s ktq3_2s ktq4_2s"}
V_TYPES=${V_TYPES:-}

mkdir -p "$OUT"
BASE="$OUT/base-f16.kld"

# same settings for every run: CPU only, flash attention on
COMMON=(-m "$MODEL" -f "$DATA" -c "$CTX" --chunks "$CHUNKS" -t "$THREADS" -ngl 0 -fa on
        --no-tq-deferred-k --no-tq-deferred-v)

# cache size in MiB of one side (K or V) from the llama_kv_cache log line
cache_mib() { grep -o "$1 ([a-z0-9_]*): *[0-9.]* MiB" "$2" | head -1 | sed 's/.*: *\([0-9.]*\) MiB/\1/'; }

run() { # side type log extra-args...
    local log=$3
    shift 3
    local t0 t1
    t0=$(date +%s.%N)
    "$BIN/llama-perplexity" "${COMMON[@]}" "$@" > "$log" 2>&1
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" 'BEGIN { print b - a }'
}

echo "model $MODEL, $CHUNKS x $CTX tokens, $THREADS threads"
secs=$(run base f16 "$OUT/base-f16.log" -ctk f16 -ctv f16 --kl-divergence-base "$BASE")
f16_k=$(cache_mib K "$OUT/base-f16.log")
f16_v=$(cache_mib V "$OUT/base-f16.log")
printf '%-4s %-8s %6s %22s %18s %10s %9s\n' side type bpw "mean KLD" "same top p" PPL "wall s"
printf '%-4s %-8s %6.3f %22s %18s %10s %9.0f\n' - f16 16 "(base)" "" \
    "$(grep -o 'Final estimate: PPL = [0-9.]*' "$OUT/base-f16.log" | awk '{print $NF}')" "$secs"

score() { # side type
    local side=$1 type=$2 log="$OUT/$1-$2.log" secs mib ref
    if [ "$side" = K ]; then
        secs=$(run K "$type" "$log" -ctk "$type" -ctv f16 --kl-divergence --kl-divergence-base "$BASE")
        ref=$f16_k
    else
        secs=$(run V "$type" "$log" -ctk f16 -ctv "$type" --kl-divergence --kl-divergence-base "$BASE")
        ref=$f16_v
    fi
    mib=$(cache_mib "$side" "$log")
    printf '%-4s %-8s %6.3f %22s %18s %10s %9.0f\n' "$side" "$type" "$(awk -v m="$mib" -v r="$ref" 'BEGIN { print 16 * m / r }')" \
        "$(grep 'Mean    KLD:' "$log" | sed 's/Mean    KLD: *//')" \
        "$(grep 'Same top p:' "$log" | sed 's/Same top p: *//')" \
        "$(grep 'Mean PPL(Q)  ' "$log" | awk '{print $4}')" "$secs"
}

for t in $K_TYPES; do score K "$t"; done
for t in $V_TYPES; do score V "$t"; done
