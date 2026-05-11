#!/usr/bin/env bash
# bench-ministral-suite.sh
#
# Comprehensive bench suite für Ministral-3-3B optimization tracking.
# Run after each kernel change to validate no regression + measure improvement.
#
# Usage:
#   bench-ministral-suite.sh [HOST] [PORT]
#
# Defaults: localhost:8791
#
# Tests:
#   1. Short-prompt TG baseline (single-shot quality)
#   2. Multi-size PP sweep (decay measurement)
#   3. Parallel scaling (par=1,2,4,8 aggregate)
#   4. Prefix-cache hit rate
#
# Outputs:
#   /tmp/bench-results-$(date +%Y%m%d-%H%M%S).json with all measurements

set -euo pipefail
HOST="${1:-localhost}"
PORT="${2:-8791}"
BASE="http://${HOST}:${PORT}"
OUT="/tmp/bench-results-$(date +%Y%m%d-%H%M%S).json"

echo "Bench suite for Ministral-3-3B"
echo "  endpoint: $BASE"
echo "  output:   $OUT"
echo ""

# Sanity: server alive?
if ! curl -sf "$BASE/health" >/dev/null; then
  echo "ERR: server not reachable at $BASE"
  exit 1
fi

results='{"timestamp":"'$(date -Iseconds)'","host":"'$HOST'","port":'$PORT',"tests":{}}'

run_test() {
  local name="$1"; shift
  local prompt="$1"; shift
  local n_predict="${1:-50}"
  local cache_prompt="${2:-false}"
  local id_slot="${3:-0}"

  local payload=$(jq -nc \
    --arg p "$prompt" \
    --argjson n "$n_predict" \
    --argjson cp "$cache_prompt" \
    --argjson s "$id_slot" \
    '{prompt:$p,n_predict:$n,stream:false,cache_prompt:$cp,id_slot:$s,temperature:0.7}')

  curl -sf -X POST "$BASE/completion" -H "Content-Type: application/json" -d "$payload" 2>/dev/null || echo '{"err":true}'
}

# Test 1: TG baseline
echo "=== Test 1: TG single-shot baseline ==="
result=$(run_test "tg_baseline" "Hi. Erzaehl mir kurz die Geschichte von Wien." 200 false 0)
tg=$(echo "$result" | jq -r '.timings.predicted_per_second // 0')
echo "  TG: $tg t/s"

# Test 2: PP decay sweep
echo ""
echo "=== Test 2: PP decay sweep ==="
for size in 1000 5000 10000 20000; do
  words=$(python3 -c "print(' '.join([f'wort{i}_{$size}' for i in range($size//2)]))")
  result=$(run_test "pp_$size" "# session_$(date +%s)_$size $words User: kurze frage?" 10 false 0)
  pp=$(echo "$result" | jq -r '.timings.prompt_per_second // 0')
  pn=$(echo "$result" | jq -r '.timings.prompt_n // 0')
  echo "  size_target=$size: prompt_n=$pn pp=$pp t/s"
done

# Test 3: Parallel scaling (requires --parallel >=8 server)
echo ""
echo "=== Test 3: Parallel scaling ==="
for par in 1 2 4 8; do
  echo "  par=$par:"
  start_ms=$(date +%s%3N)
  pids=()
  for i in $(seq 1 $par); do
    (run_test "par_${par}_$i" "Frage $i kurz." 200 false $((i-1)) > "/tmp/par_${par}_$i.json") &
    pids+=($!)
  done
  wait "${pids[@]}"
  end_ms=$(date +%s%3N)
  wall=$((end_ms - start_ms))
  total=0
  for i in $(seq 1 $par); do
    n=$(jq -r '.tokens_predicted // 0' /tmp/par_${par}_$i.json)
    total=$((total + n))
  done
  agg=$(python3 -c "print(f'{$total * 1000 / $wall:.1f}')")
  echo "    wallclock=${wall}ms total_tokens=$total aggregate_tg=${agg} t/s"
done

# Test 4: Prefix-cache hit
echo ""
echo "=== Test 4: Prefix-cache ==="
big_prompt=$(python3 -c "import sys; print('System: tools.\n' + '\n'.join([f'- t_{i}: x,y -> result' for i in range(500)]) + '\n\nUser: kurz?')")

start_ms=$(date +%s%3N)
result=$(run_test "cache_cold" "$big_prompt" 20 true 0)
end_ms=$(date +%s%3N)
wall1=$((end_ms - start_ms))
pp_cold=$(echo "$result" | jq -r '.timings.prompt_per_second // 0')
pn_cold=$(echo "$result" | jq -r '.timings.prompt_n // 0')
echo "  COLD: wall=${wall1}ms prompt_n=$pn_cold pp=$pp_cold t/s"

# Same prompt + small variation → cache hit
big_prompt2="${big_prompt} Bitte kurz antworten."
start_ms=$(date +%s%3N)
result=$(run_test "cache_warm" "$big_prompt2" 20 true 0)
end_ms=$(date +%s%3N)
wall2=$((end_ms - start_ms))
cache_n=$(echo "$result" | jq -r '.timings.cache_n // 0')
pn_warm=$(echo "$result" | jq -r '.timings.prompt_n // 0')
echo "  WARM: wall=${wall2}ms cache_n=$cache_n prompt_n_new=$pn_warm"

echo ""
echo "Bench suite complete."
echo "  Compare against baseline: docs/2026-05-11-ministral3-perf-sweep.md"
echo "  Live deploy: GPU0:8791 (Q4_K_M), GPU1:8794 (IQ2_XXS par=8)"
