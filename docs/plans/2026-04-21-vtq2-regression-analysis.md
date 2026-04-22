# VTQ_2 TG Regression — Root Cause Analysis

**Datum:** 2026-04-21
**Related tasks:** #141, #143, #135
**Status:** Partial fix committed (QK=128), full root-cause identified, proper fix pending benchmark

## TL;DR

VTQ_2 decodes at **4.32 tok/s** on Qwen3.5-35B-A3B (D=256) vs **66.51 tok/s** for VTQ_1 —
a 15× regression. Multiple contributing factors identified:

1. **Register pressure (primary):** CUDA FA-vec at D=256 uses **249 regs/thread** →
   `__launch_bounds__(..., 1)` forces 1 block/SM occupancy. Measured via
   `cuobjdump --dump-resource-usage` on built kernel.
2. **Block size mismatch (secondary):** QK_VTQ_TRELLIS=256 meant 1 block/head on
   D=256 + total failure on D=128. Fixed in commit 1d92cfe5c.
3. **LUT-thrashing (tertiary):** Ceiling ~3× in Triton spike — not the dominant
   factor but real.

## Measurements

### TG128 bench (single RTX 2060, Qwen3.5-35B-A3B IQ2_XS, -ngl 99)

| K-cache | V-cache | tok/s | vs fp16 |
|---------|---------|-------|---------|
| f16 | f16 | 71.71 | 1.00× |
| q8_0 | q8_0 | 69.31 | 0.97× |
| ktq2_1 | vtq2_1 (VTQ v1) | 66.51 | 0.93× |
| ktq2_1 | **vtq2_2** | **7.37** | **0.10×** |
| ktq2_1 | **vtq3_2** | **4.32** | **0.06×** |
| ktq2_1 | **vtq4_2** | **1.27** | **0.02×** |

Pattern: more bits → slower. Not memory-bound (reads stay ~linear).
**Compute-bound, with register pressure dominating.**

### Register usage (cuobjdump)

FA-vec kernel `flash_attn_ext_vec<256, 2, F16, VTQ3_2, ...>`:
- REG: **249** per thread
- SHARED: 4608 B (standard FA shmem: KQ max/sum + Q tile)
- LOCAL: 0 (no spill to L1/L2, but 1 block/SM)

Turing SM compute 7.5:
- Max regs/thread: 255
- Register file/SM: 65536 (256 × 256 threads theoretical)
- With 249 regs: 65536 / 249 / 128 threads = 2.05 blocks/SM theoretical
- With `__launch_bounds__(D, 1)`: **forced 1 block/SM**

VTQ_1 equivalent expected ~128-150 regs (simple codebook lookup, no
state-machine arithmetic, no boundary-straddle branches).

### Triton Strategy A spike (RTX 2060, D=256 simulation)

Warp-cooperative block cache:
- Ceiling speedup: **3.27×**
- LUT overhead isolated: ~2-3× of total cost
- **Does not explain full 15× gap** — register pressure is separate factor.

See `docs/plans/2026-04-20-triton-spike-report.md` and
`docs/plans/2026-04-21-triton-strategy-a-spec.md` for details.

## Root Cause Decomposition

Estimated contribution to 15× slowdown (multiplicative):

| Factor | Contribution | Fix |
|--------|-------------|-----|
| Register spill / low occupancy | ~4-5× | Reduce regs in `vtq_state_at` or relax launch_bounds |
| LUT L2-thrashing | ~2-3× | Strategy A (warp-shmem block cache) |
| Sequential qs-byte loads (not coalesced) | ~1.5× | 32-bit aligned loads |
| fp16 cast overhead | ~1.2× | In-place fp16 accumulation |

Product: 4.5 × 2.5 × 1.5 × 1.2 ≈ **20×** (matches observed 15× with some noise).

## Implemented Fix (commit 1d92cfe5c)

**`QK_VTQ_TRELLIS` 256 → 128:**
- Enables D=128 head models (Qwen3.5-0.6B/0.8B previously failed at kv-cache load)
- Gives 2× block-level parallelism on D=256
- BPW overhead: +0.031 per K-level (2.25/3.25/4.25 vs 2.125/3.125/4.125)
- MSE *better* in Phase-1 harness (Q128 vs Q256 direct comparison):
  - 2-bit: 0.0668 (Q128) vs 0.0680 (Q256)
  - 3-bit: 0.01704 (Q128) vs 0.01722 (Q256)

**Expected impact on TG:** ~1.5-2× improvement (2× block parallelism alone).
Does **not** fix register pressure — that remains the dominant issue.

## Next Steps (ordered by priority)

### 1. Measure post-QK=128 TG (pending current build)
Run `llama-bench -ctk ktq2_1 -ctv vtq3_2 -p 0 -n 128` on Qwen3.5-35B-A3B.
Expected: 4.32 → 8-10 tok/s (still slow but 2× improvement proves block-parallelism fix).

### 2. Register-pressure reduction (high impact, medium effort)
Options:
- **Relax `__launch_bounds__(..., 1)` → `(..., 2)`** forces compiler to reduce
  regs. Measure if spill penalty < occupancy gain.
- **Refactor `vtq_state_at<K>`** to reduce live state. Specifically the
  `stream_bit < L` branch uses 3 extra registers for s0 masking.
- **Split FA-vec into "decode" + "compute"** phases. Current single-pass
  keeps decoded values live across queries = reg pressure.

### 3. Strategy A (CUDA port) (medium impact, high effort)
Port from Triton spike. ~6-10h effort + 3-5 CUDA builds. Ceiling 3× gain.
**Only do after (2)** — register fix cleans the playing field.

### 4. In-place fp16 accumulation (low impact, low effort)
`dequantize_V_vtq_2` currently computes `float val = ... ; cast to half`.
Eliminate intermediate float → direct `__float2half_rn(table[state] * ds)`
without intermediate. Saves 1 reg per sample.

## Decision Point

**Before deploying fix (3) Strategy A:** re-measure after (1) lands.
The Triton autoresearch loop running now (`T3` agent) will enumerate
candidate strategies (FP16, packed qs, shmem hot-states) and measure
which combinations give >5× cumulative.

If Triton proves >5× speedup achievable → justify 1-2 days CUDA port.
If Triton ceiling stays ~3-4× → accept VTQ_1 as production-best and
document VTQ_2 as experimental.
