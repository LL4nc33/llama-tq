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
