#!/usr/bin/env bash
# ctx-aware TG bench matrix — the foundation for measuring MoE-IQ2 compute tweaks.
#
# WHY THIS EXISTS (perf-deep-dive 2026-07-13, the "PURSUE" calibration finding):
#   MoE-IQ2 (Qwen3.6-A35B-A3B) token-gen is NOT uniformly bandwidth-bound.
#     - short ctx  (512, 4k): bottleneck is FFN-GEMV launch/occupancy — active-weight
#       ceiling (~434 t/s) and KV ceiling (~2000 t/s) both sit far above the measured
#       ~80 t/s, so neither bandwidth binds. Compute/occupancy tweaks CAN show here.
#     - long ctx   (65k+):    the KTQ/VTQ KV read dominates (~1.4-4.3 GB/tok) -> genuinely
#       bandwidth-bound. FFN/compute tweaks measure ~0% here because the KV read masks them.
#   => A compute tweak MUST be measured at short ctx. Benching only at 200k (as a deploy
#      config) makes a real 2x FFN win look like noise. This is why past "code-tricks are
#      dead" conclusions (from dense bandwidth-bound Gemma-4) do NOT transfer to MoE-IQ2.
#
# The dense-Gemma-4 lesson (bandwidth-bound, code-tricks dead) is the CONTROL, not the rule.
#
# Usage:
#   bench-ctx-matrix.sh <model.gguf> [label]
#
# Env:
#   BIN=...            path to llama-bench (required)
#   CTXS="512 4096 65536"   ctx depths to sweep (short = tweak-visible, long = KV-bound control)
#   TG=128             tokens generated per run
#   NGL=99             layers offloaded
#   REPS=3             repetitions per point (llama-bench -r)
#   FA=1 TS="12,12"    canonical IQ2-MoE flags (feedback_bench_flags_canonical)
#   CTK=ktq2_1 CTV=vtq2_1   KV types (deploy config)
#   OUT=...            output dir

set -euo pipefail

MODEL="${1:?usage: bench-ctx-matrix.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"
BIN="${BIN:?set BIN= to the llama-bench path}"
CTXS="${CTXS:-512 4096 65536}"
TG="${TG:-128}"
NGL="${NGL:-99}"
REPS="${REPS:-3}"
FA="${FA:-1}"
TS="${TS:-12,12}"
CTK="${CTK:-ktq2_1}"
CTV="${CTV:-vtq2_1}"
OUT="${OUT:-$(pwd)/bench-ctx-$(date +%Y%m%d-%H%M)}"

[[ -x "$BIN" ]] || { echo "llama-bench not executable: $BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

mkdir -p "$OUT"
CSV="$OUT/${LABEL}.csv"
echo "ctx,tg_tps,regime" > "$CSV"

echo "=== ctx-matrix bench: $LABEL ==="
echo "model=$MODEL  bin=$BIN  ctk=$CTK ctv=$CTV  fa=$FA ts=$TS  tg=$TG reps=$REPS"
echo

# depth-marker: short ctx = tweak-visible (compute/occupancy), long ctx = KV-bandwidth control
regime_for() {
    local c="$1"
    if   (( c <= 8192 ));  then echo "compute-visible"
    else                        echo "KV-bandwidth-control"; fi
}

# single-GPU isolation for the compute-visible runs is deliberate: dual-GPU pipeline
# overhead adds noise that can swamp a small FFN win (project_27b_dual_vs_single).
export CUDA_VISIBLE_DEVICES=0

for CTX in $CTXS; do
    REGIME="$(regime_for "$CTX")"
    echo "--- ctx=$CTX ($REGIME) ---"
    # -d <ctx> primes the KV cache to depth ctx, then measures TG from there.
    # This is the key: TG@depth, not TG@0, so the KV-read cost is actually in the loop.
    TPS="$("$BIN" -m "$MODEL" -ngl "$NGL" -fa "$FA" -ts "$TS" \
                  -ctk "$CTK" -ctv "$CTV" \
                  -d "$CTX" -n "$TG" -p 0 -r "$REPS" -o csv 2>/dev/null \
           | awk -F',' 'NR>1 && $0 ~ /tg/ {gsub(/"/,"",$NF); print $NF; exit}')"
    # fallback parse: last numeric field of the last data row
    [[ -z "${TPS:-}" ]] && TPS="$("$BIN" -m "$MODEL" -ngl "$NGL" -fa "$FA" -ts "$TS" \
                  -ctk "$CTK" -ctv "$CTV" -d "$CTX" -n "$TG" -p 0 -r "$REPS" 2>/dev/null \
           | grep -oE '[0-9]+\.[0-9]+ ± ' | tail -1 | grep -oE '[0-9]+\.[0-9]+')"
    echo "  tg = ${TPS:-?} t/s"
    echo "$CTX,${TPS:-NA},$REGIME" >> "$CSV"
done

echo
echo "=== done. CSV: $CSV ==="
cat "$CSV"
echo
echo "READING THE RESULT for an A/B (baseline vs tweak):"
echo "  * compute-visible rows (ctx<=8192): a real FFN/occupancy tweak shows HERE."
echo "  * KV-bandwidth-control rows (ctx>=65536): should NOT move — if they do, the tweak"
echo "    touched the KV path, not FFN. If ONLY the control moves, you measured the wrong thing."
