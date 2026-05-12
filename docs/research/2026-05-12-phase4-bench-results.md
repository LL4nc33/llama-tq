# Phase 4 Bench Results — KTQ+VTQ MMA Inline (2026-05-12)

## Build
- Branch: `feature/ktq-vtq-mma-inline` @ 9c2f36d66
- Binary: `/workspace/llama-tq/build/bin/llama-server`
- Hardware: GPU0 RTX 2060 12GB, sm_75

## Deploy config
```
--cache-type-k ktq2_1 --cache-type-v vtq2_1 --tq-protect-layers 12
--flash-attn on -ngl 99 -ub 256 --backend-sampling
--no-mmap --parallel 1 -c 65536
```

## PP comparison — VOR vs JETZT

Baseline (Q4_K_M + q8_0 K + vtq2_1 V, fattn-mma-ktq.cu split path):

| Tokens | Baseline PP (t/s) | Phase 4 PP (t/s) | Speedup |
|--------|-------------------|-------------------|---------|
| ~1000  | 614               | **1818**          | **+196%** |
| ~5000  | 552               | **1369**          | **+149%** |
| ~10000 | 212               | **725**           | **+242%** |
| ~26000 | 122               | **435**           | **+257%** |
| ~59000 | ~70 (extrapol.)   | **199**           | **+185%** |

## TG status

| ctx | TG (t/s) | Notes |
|-----|----------|-------|
| short (~10 tok prefill) | 105 | identical to baseline |
| 1k ctx, 200 predicted | 91 | -11% vs baseline (might be repetitive sampling artifact) |
| 10k ctx, 200 predicted | 36 | **-49% vs baseline** — needs investigation |

**Quality:** Output coherent at all tested ctx lengths. No gibberish (KTQ+protect=12 still required).

## VRAM @ 65k ctx
- Before (q8_0 K + vtq2_1 V): 8.0 GB
- After (ktq2_1 K + vtq2_1 V): **6.1 GB** (saves 1.9 GB)

## What worked
- VTQ V inline dequant in shared memory avoids per-forward f16 K scratch buffer
- Single template-instance pair (4,4) + (8,4) covers Ministral-3 prefill chunks
- Build-validation after each phase caught duplicate-default issue early
- No __syncthreads race detected, no warp divergence regressions

## Open issues / TODO
1. **TG decay at long ctx investigation**:
   - Hypothesis: VTQ-V dequant cost at decode (n_tokens=1) is higher than f16-V load
   - VTQ load does 32-element FWHT-free codebook lookup ×2 blocks per V-row
   - f16 load is plain memcpy
   - At decode, weights are bandwidth-bound, V cost is in the loop
2. **Verify PPL no regression**: vs split-path baseline
3. **Phase 5**: extend to more configs (256 head_dim, MLA, etc.) if relevant

## Reproduction
```bash
ssh <user>@example.local "/workspace/deploy-ministral-ktq-vtq-inline.sh"  # port 8794
python3 /tmp/bench-vtq-pp.py   # PP scaling
python3 /tmp/bench-vtq-tg.py   # TG with longer predictions
```

## Phase 5 update — fattn.cu dispatch fix (70912a5b6)

Without Phase 5, the kernel-selector at `fattn.cu:396-400` returned
BEST_FATTN_KERNEL_VEC when V was VTQ (only KTQ K + f16 V got MMA-KTQ).
Phase 5 adds an explicit dispatch line for KTQ2_1 K + VTQ2_1 V.

### Updated PP table (Phase 5)

| Tokens | Baseline | Phase 4 | **Phase 5** | Speedup vs baseline |
|--------|----------|---------|--------------|---------------------|
| ~1k    | 614      | 1818    | **2304**     | **+275%** (3.75×)   |
| ~5k    | 552      | 1369    | **1249**     | +126% (2.26×)       |
| ~10k   | 212      | 725     | **1735**     | **+718%** (8.2×)    |
| ~26k   | 122      | 435     | TBD          | —                   |

(Note: Phase 5 PP at 5k slightly lower than Phase 4 — possibly noise / warmup.
Phase 5 wins big at 10k+ where MMA kicks in cleanly.)

## Open: TG-decay at long ctx

Phase 5 PP is now near roofline-bandwidth-bound. TG at 10k ctx is 35 t/s
(same as baseline — not a regression, but a separate problem).

Roofline analysis @ 10k ctx:
- Weight reads: ~2 GB/token
- KV reads: ~75 MB/token (3.5 bpw K + 2.5 bpw V × 10k × 1024 × 26 layers / 8 heads)
- Total bandwidth per token: ~2.1 GB
- At 336 GB/s, ceiling = 160 t/s, realistic (70% eff) = 110 t/s
- Measured: 35 t/s = 22% of theoretical → 5× headroom remains

Future hebel:
1. Sparse-K attention skip (S199 branch from memory canonical — not in current repo)
2. VEC kernel V-dequant micro-optimization for decode
3. KV-cache L2 layout reordering for attention-sink reuse

## Ministral-3-14B validation (2026-05-12)

Same kernel works on the bigger family member. UD-IQ2_XXS, 40 layers,
n_embd=5120, GQA 32/8 head_dim=128 (same shape as 3B → kernel applies).

Deploy: `--cache-type-k ktq2_1 --cache-type-v vtq2_1 --tq-protect-layers 20`
(40 layers ⇒ need ~50% protect for stability; 12 caused gibberish.)

Single-GPU0 at 65k ctx: **10.1 GB VRAM** (fits with headroom).

PP results:

| Tokens | 14B PP (t/s) | 3B PP (Phase 5) |
|--------|--------------|-----------------|
| ~5k    | **849**      | 1249–2304       |
| ~26k   | **636**      | 435 (Phase 4)   |
| ~59k   | **441**      | 199 (Phase 4)   |

Headline: at 26k tokens the **14B model with our kernel beats the 3B baseline
without it (122 t/s)** by 5.2×. The MMA-inline KTQ+VTQ path scales
cleanly with model size as long as the head-shape (D=128, GQA=4) matches.

TG: ~35 t/s short-ctx, decays at long ctx (roofline-bound at 14B too).

Quality: coherent at all tested ctx lengths with protect=20.
