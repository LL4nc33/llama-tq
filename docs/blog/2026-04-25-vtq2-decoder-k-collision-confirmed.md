# VTQ_2 K-Collision Bug — confirmed reproduction after encoder fix

Stand: 2026-04-25 16:38 CEST. Live-tested auf test-box mit aktuellem turboquant HEAD `976082f4c` (DEBUG-commits reverted, encoder OOB-fix `b771f9267` in place).

## Setup

- Model: Gemma4-26B-A4B-bartowski-IQ2_XXS (IT-finetuned, raw-wikitext gives high absolute PPL)
- Hardware: test-box, 2× RTX 2060 (12 GB each), `-ngl 99 -ts 12,12 -fa 1`
- Wikitext-2-raw, `-c 512 -b 512 --chunks 4 --no-warmup`

## Reproduction

| K cache | V cache | PPL |
|---|---|---:|
| f16 | vtq2_2 | 12171.02 |
| f16 | vtq3_2 | **12171.02** (identical) |
| f16 | vtq4_2 | **12171.02** (identical) |
| ktq2_1 | vtq2_2 | 14581.40 |
| ktq2_1 | vtq3_2 | **14581.40** (identical) |
| ktq2_1 | vtq4_2 | **14581.40** (identical) |

K-cache type changes PPL. V-cache K-bit-depth does NOT. Reproducible across two K-types.

## What was already excluded

1. **Encoder OOB-write** — `VTQ_ENC_N=256` vs `QK_VTQ_TRELLIS=128`. Fixed in `b771f9267`. Caused PPL to shift from 13047→12171 — encoder *does* write something different now. But the residual K-collision persists.
2. **Encoder K-routing** — Agent A confirmed via DEBUG print `[VTQ_ENC] K=4` that the encoder template is correctly instantiated per V-type.
3. **Dispatch table** — `fattn-vec-dispatch-vtq2.cu:36-49` has separate `FATTN_VEC_CASES_ALL_D_WITH_512_RET` entries for VTQ2_2/3_2/4_2.

## What's left

The bug is in one of:

a) **Decoder template specialization** — `dequantize_V_vtq_2<block_t, K, T, ne>` may be receiving K but reading bit-extraction with hardcoded shift amount. Check `fattn-common.cuh:933-984`.

b) **Trellis decode** — `trellis_decode_block<K>` in `trellis.cuh:52-88` — verify the bit unpacking actually uses K not a constant.

c) **FA-vec-vtq2.cuh** — the high-level dequant call at `fattn-vec-vtq2.cuh:60` uses `constexpr int N = QK_VTQ_TRELLIS;` but maybe doesn't pass K correctly to the inner dequant. The 16-state Trellis Viterbi maps bits→codebook entries; if the bit-mask is hardcoded `(1u<<2)-1u` instead of `(1u<<K)-1u`, all 3 K-values would read only the lowest 2 bits → identical decoded output.

The third hypothesis (c) is most consistent with the symptom: **encoder writes correctly for all K, but decoder reads only the low 2 bits regardless** → K=3 and K=4 lose their extra information at decode.

Agent B (a3d4c28443be816eb) reviewed these files and reported "all correct" — but the symptom proves otherwise. Reviewer probably checked the template signatures but missed a bit-mask constant inside the body of `trellis_decode_block` or its inlined helpers.

## Next steps

1. Manual code-walk through trellis.cuh and fattn-common.cuh focused on `Kmask` / `(1<<K)` / `>> X` patterns that should depend on K but might not.
2. Targeted unit test: encode 128 samples with K=2 vs K=4, decode each, compare. CPU path (`test-vtq3-roundtrip` extended).
3. If CPU is correct but CUDA is broken → CUDA-only bug in dequantize_V_vtq_2.
4. If CPU is broken too → bug in shared `trellis_decode_block` template body.

## Impact

- Phase 3 VTQ_3 win claim is invalid (already noted in README caveat block, commit 834c41b09).
- VTQ_2 production deployments using vtq3_2 or vtq4_2 are silently equivalent to vtq2_2 — wasted bpw budget without quality gain.
- Existing 80B/122B prod with `ktq2_1/vtq2_2` is unaffected (K=2 is the de-facto used K).

## Files modified in this session

- `b771f9267` fix(critical): VTQ_ENC_N=256→QK_VTQ_TRELLIS — encoder OOB-write
- `c9ac63017` + `65054fabd` DEBUG instrumentation (reverted in 8fa76c2da, 976082f4c)
- `834c41b09` docs(README): flag Phase 3 unverified
- This doc + commit pending.
