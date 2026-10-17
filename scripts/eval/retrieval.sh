#!/usr/bin/env bash
# Long-context retrieval pass rate per KV cache configuration.
#
# usage: MODEL=model.gguf scripts/eval/retrieval.sh < configs.txt
#
# Each config line: LABEL CTK CTV ENV FLAGS
#   LABEL  name of the configuration (used for the log and result files)
#   CTK    K cache type (-ctk), CTV  V cache type (-ctv)
#   ENV    one VAR=value set for the server process, or -
#   FLAGS  the rest of the line is passed to llama-server, or -
# Example:
#   q8      q8_0   q8_0   -  -
#   k41     ktq4_1 vtq4_1 -  -
#   k2-prot ktq2_1 vtq2_1 -  --tq-protect-layers 2
#
# For every SEED x FRAC a prompt is generated offline with retrieval.py (same seed, fraction and token target give
# the same prompt on every machine). Then one llama-server is started per config line (never two at a time), every
# prompt is sent once (greedy, max_tokens MAX_TOKENS), and the number of exact copies is printed:
#   LABEL CTK/CTV [ENV] [FLAGS]: exact N/M  mean ratio R  failed <prompt hashes>
# The defaults give the 36-prompt test of docs/turboquant.md (6 documents x 6 positions, ~29k tokens, -c 40960).
#
# Environment:
#   MODEL          GGUF model (required)
#   LLAMA_SERVER   llama-server binary (default: llama-server on PATH)
#   GPUS           CUDA_VISIBLE_DEVICES for the server (default 0)
#   PORT           server port (default 8796)
#   SEEDS          document seeds (default "1234 2001 3001 4001 5001 6001")
#   FRACS          relative positions of the section to copy (default "0.10 0.25 0.40 0.55 0.70 0.85")
#   TARGET_TOKENS  prompt size (default 29000), CTX  context size (default 40960), THREADS  -t (default 4)
#   MAX_TOKENS     max_tokens per request (default 320), TIMEOUT  request timeout in s (default 300)
#   OUT            result directory (default retrieval-results/<date-time>), PROMPT_DIR (default OUT/prompts)
#   CPU_GUARD      kill the server when it uses more than this CPU % for 15 s in a row, which means the
#                  attention runs on the CPU instead of the GPU (default 200, 0 = off)
#   VRAM_IDLE_MIB  before each start wait until every GPU in GPUS uses less VRAM than this (needs nvidia-smi,
#                  default 600, 0 = do not wait)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT="$HERE/retrieval.py"
: "${MODEL:?set MODEL to the GGUF model}"
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
GPUS=${GPUS:-0}
PORT=${PORT:-8796}
SEEDS=${SEEDS:-1234 2001 3001 4001 5001 6001}
FRACS=${FRACS:-0.10 0.25 0.40 0.55 0.70 0.85}
TARGET_TOKENS=${TARGET_TOKENS:-29000}
CTX=${CTX:-40960}
THREADS=${THREADS:-4}
MAX_TOKENS=${MAX_TOKENS:-320}
TIMEOUT=${TIMEOUT:-300}
OUT=${OUT:-retrieval-results/$(date +%Y%m%d-%H%M%S)}
PROMPT_DIR=${PROMPT_DIR:-$OUT/prompts}
CPU_GUARD=${CPU_GUARD:-200}
VRAM_IDLE_MIB=${VRAM_IDLE_MIB:-600}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT" "$PROMPT_DIR"

# shared libraries of a build tree sit next to the binary
if [[ "$LLAMA_SERVER" == */* ]]; then
    export LD_LIBRARY_PATH="$(cd "$(dirname "$LLAMA_SERVER")" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA device numbers = nvidia-smi numbers

for seed in $SEEDS; do for frac in $FRACS; do
    f=$PROMPT_DIR/p-$seed-$frac.json
    [ -f "$f" ] || python3 "$RT" offline --out "$f" --seed "$seed" --target-frac "$frac" \
        --target-tokens "$TARGET_TOKENS" 2> /dev/null || { echo "prompt generation failed: $f" >&2; exit 1; }
done; done

wait_gpus_idle() {
    [ "$VRAM_IDLE_MIB" -gt 0 ] && command -v nvidia-smi > /dev/null || return 0
    local used
    for _ in $(seq 900); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | sort -n | tail -1)
        [ "${used:-0}" -lt "$VRAM_IDLE_MIB" ] && return 0
        sleep 2
    done
    echo "GPUs $GPUS still busy after 30 min" >&2
    return 1
}

while read -r LABEL CTK CTV ENVS FLAGS; do
    [ -z "${LABEL:-}" ] || [[ "$LABEL" == \#* ]] && continue
    [ "${ENVS:--}" = "-" ] && ENVS=""
    [ "${FLAGS:--}" = "-" ] && FLAGS=""
    SPEC=off; [[ " $FLAGS " == *" --spec-type "* ]] && SPEC=on
    wait_gpus_idle || exit 1

    # shellcheck disable=SC2086  # ENVS and FLAGS are word lists on purpose
    env CUDA_VISIBLE_DEVICES="$GPUS" $ENVS "$LLAMA_SERVER" -m "$MODEL" -ngl 99 -fa on -c "$CTX" --jinja -np 1 \
        --reasoning off --cache-ram 0 -t "$THREADS" -ctk "$CTK" -ctv "$CTV" $FLAGS \
        --host 127.0.0.1 --port "$PORT" > "$OUT/$LABEL.log" 2>&1 < /dev/null &
    PID=$!
    if ! python3 "$RT" health --url "$URL" --pid "$PID" --timeout 600 2>> "$OUT/$LABEL.run"; then
        echo "$LABEL $CTK/$CTV [$ENVS] [$FLAGS]: server did not start, see $OUT/$LABEL.log" | tee -a "$OUT/summary.txt"
        kill "$PID" 2> /dev/null; wait "$PID" 2> /dev/null
        continue
    fi

    GUARD=""
    if [ "$CPU_GUARD" -gt 0 ]; then
        ( n=0
          while kill -0 "$PID" 2> /dev/null; do
              c=$(ps -o %cpu= -p "$PID" | cut -d. -f1 | tr -d ' ')
              if [ "${c:-0}" -gt "$CPU_GUARD" ]; then n=$((n + 1)); else n=0; fi
              if [ $n -ge 15 ]; then echo "CPU GUARD: killed $LABEL (CPU fallback?)" >> "$OUT/summary.txt"; kill -9 "$PID"; fi
              sleep 1
          done ) &
        GUARD=$!
    fi

    for seed in $SEEDS; do for frac in $FRACS; do
        kill -0 "$PID" 2> /dev/null || break 2
        python3 "$RT" run --url "$URL" --prompt "$PROMPT_DIR/p-$seed-$frac.json" --out "$OUT/$LABEL.jsonl" \
            --build "$LABEL" --pair "$CTK/$CTV" --ctk "$CTK" --ctv "$CTV" --spec "$SPEC" \
            --max-tokens "$MAX_TOKENS" --timeout "$TIMEOUT" --server-log "$OUT/$LABEL.log" >> "$OUT/$LABEL.run" 2>&1
    done; done

    kill "$PID" 2> /dev/null; wait "$PID" 2> /dev/null
    [ -n "$GUARD" ] && { kill "$GUARD" 2> /dev/null; wait "$GUARD" 2> /dev/null; }

    python3 - "$OUT/$LABEL.jsonl" "$LABEL $CTK/$CTV [$ENVS] [$FLAGS]" "$(( $(wc -w <<< "$SEEDS") * $(wc -w <<< "$FRACS") ))" <<'PY' | tee -a "$OUT/summary.txt"
import json, os, sys
path, name, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
ok = [r["verbatim"]["exact"] for r in rows]
ratio = sum(r["verbatim"]["ratio"] for r in rows) / max(1, len(rows))
fails = [r["prompt_sha256"][:6] for r in rows if not r["verbatim"]["exact"]]
note = "" if len(rows) == expected else f"  ONLY {len(rows)} of {expected} prompts ran"
print(f"{name}: exact {sum(ok)}/{len(ok)}  mean ratio {ratio:.3f}  failed {' '.join(fails)}{note}")
PY
done
echo "results: $OUT/summary.txt"
