# Phase 3A1 Validation Runbook

**Datum:** 2026-04-22
**Pre-req:** `build-e11/bin/llama-bench` + `llama-perplexity` gebaut mit `-DFATTN_VTQ2_CACHED=1`
**Ziel:** Gate-Entscheidung für Phase 3A2 (siehe `2026-04-22-e11-phase3a2-spec.md` §7)

## Gate-Targets (Hard)

| Metric | Baseline (master) | 3A1 Gate | 3A2 Ziel |
|---|---|---|---|
| TG128 tok/s (Qwen3.5-35B-A3B, KTQ2_1×VTQ3_2, D=128) | 4.32 | **≥ 25** | ≥ 50 |
| Regs/thread (flash_attn_ext_vec_vtq2_cached) | — | **≤ 150** | ≤ 150 |
| Blocks/SM (cuobjdump) | 1 | **≥ 2** (goal 4) | ≥ 4 |
| PPL delta vs legacy VTQ3_2 | 0% | ≤ **0.5%** | ≤ 0.5% |
| test-backend-ops FLASH_ATTN_EXT | PASS | PASS | PASS |

## Step-by-Step auf gpu00

### 1. Verify build

```bash
ssh claude@gpu00.node
cd ~/llama-tq
ls -la build-e11/bin/llama-bench build-e11/bin/llama-perplexity
# Expected: both present, size >20MB
```

### 2. Register-Usage Check

```bash
cd ~/llama-tq
find build-e11 -name "fattn-vec-dispatch-vtq2*.o" | \
  xargs -I{} cuobjdump --dump-resource-usage {} 2>/dev/null | \
  grep -A1 "vtq2_cached" | head -40
```

**Pass:** regs line shows ≤ 150. Blocks/SM derived as `65536 / (128 * regs)`.

**Fail:** if regs > 180 → spill likely. Check with `ncu --set full --kernel-regex vtq2_cached llama-bench ...` later, but proceed to TG measurement first.

### 3. Unit Test (test-backend-ops)

```bash
./build-e11/bin/test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT 2>&1 | \
  grep -E "ktq2_1.*vtq3_2|PASS|FAIL|\[OK\]|\[FAIL\]"
```

**Pass:** KTQ2_1 × VTQ3_2 combo reports OK. If combo not listed, cached path wasn't taken → check `FATTN_VTQ2_CACHED` was set at compile.

### 4. TG Measurement

```bash
cd ~/llama-tq
./build-e11/bin/llama-bench \
  -m ~lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf \
  -fa 1 \
  -ctk ktq2_1 -ctv vtq3_2 \
  -ngl 99 -mg 0 -sm none \
  -p 0 -n 128 -r 2 \
  2>&1 | tee /tmp/e11-tg-3a1.log | tail -20
```

**Single-GPU mode (`-sm none -mg 0`)** — isolates the E11 kernel path, avoids multi-GPU artifacts. Production server keeps using GPU 1.

**Pass:** TG128 ≥ 25 tok/s. Ideal ≥ 50 tok/s.

**Measured values:**
- Fill in: TG128 = __ tok/s

### 5. Cross-Check gegen Baseline

```bash
# Re-measure on same hardware, same conditions, legacy kernel:
./build-master/bin/llama-bench \
  -m ~lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf \
  -fa 1 -ctk ktq2_1 -ctv vtq3_2 \
  -ngl 99 -mg 0 -sm none -p 0 -n 128 -r 2 \
  2>&1 | tee /tmp/e11-tg-baseline.log | tail -20
```

**Expected:** ~4.32 tok/s (the regression baseline). Speedup = 3A1 / baseline.

### 6. PPL Delta (kurz)

Nur sanity check — 512 tokens reicht:

```bash
cd ~/llama-tq
./build-e11/bin/llama-perplexity \
  -m ~lance/models/Qwen3.5-35B-A3B-IQ2_XS.gguf \
  -f ~lance/wikitext-2-raw/wiki.test.raw \
  -fa 1 -ctk ktq2_1 -ctv vtq3_2 \
  -ngl 99 -c 512 --chunks 4 \
  2>&1 | tee /tmp/e11-ppl-3a1.log | grep -E "perplexity:|chunks"
```

**Pass:** PPL ≤ baseline × 1.005 (0.5% tolerance). Baseline ≈ 10.8 for IQ2_XS weights.

### 7. Gate Decision

Trage in `docs/plans/2026-04-22-e11-phase3a1-results.md` ein:

```yaml
tg_3a1: __ tok/s
tg_baseline: 4.32 tok/s
speedup: __×
regs: __
blocks_per_sm: __
ppl_delta: __%
test_backend: PASS/FAIL
gate: GREEN/YELLOW/RED
```

Ampeln:
- **GREEN (TG ≥ 25, regs ≤ 150, PPL ≤ 0.5%)** → merge 3A1 default-ON, start 3A2 per `2026-04-22-e11-phase3a2-spec.md`.
- **YELLOW (10 ≤ TG < 25)** → profile erst: `ncu --set full --kernel-regex vtq2_cached`. Check bank conflicts, LUT residency, Q_reg spill. Fix vor 3A2.
- **RED (TG < 10 OR regs > 180 OR PPL > 1%)** → revert 3A1 dispatch hook, keep kernel code for reference. Re-evaluate: E14 (split decode → fp16 buffer → cuBLAS GEMM) may be the correct pattern for GQA=8 instead.

## Parallel-Bench Optional (Produktion)

Falls Gate GREEN, folgende Combos ebenfalls messen (read-only — sind nicht im cached path, fallen auf legacy):

```bash
for V in vtq2_2 vtq4_2; do
  ./build-e11/bin/llama-bench -m ...A3B.gguf -fa 1 -ctk ktq2_1 -ctv $V \
    -ngl 99 -mg 0 -sm none -p 0 -n 128 -r 2 2>&1 | tail -3
done
```

Erwartet: diese bleiben auf ~4-7 tok/s (legacy path). Das bestätigt 3A2-Bedarf.

## Rollback

```bash
cd ~/llama-tq
git revert 31c6790c0   # Phase 3A1 dispatch + kernel
# Or nur dispatch deaktivieren:
cmake -B build -DFATTN_VTQ2_CACHED=OFF && cmake --build build -j4
```
