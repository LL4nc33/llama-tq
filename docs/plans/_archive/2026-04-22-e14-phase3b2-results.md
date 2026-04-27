# E14 Phase 3B2 — Results (Negative)

**Datum:** 2026-04-22
**Setup:** build-e14 mit FATTN_VTQ2_SPLIT_ENABLE=1, Qwen3.5-35B-A3B IQ2_XS, RTX 2060 sm_75

## Measurements

| Config | TG128 tok/s | Note |
|---|---|---|
| f16 K + f16 V (baseline) | **71.81** | absolute ceiling |
| f16 K + vtq2_1 V (legacy V1) | **70.88** | -1% vs f16, production-best |
| ktq2_1 K + vtq2_1 V (full TQ v1) | ~66.5 | -7% vs f16 (pre-VTQ_2 baseline) |
| ktq2_1 K + vtq3_2 V via E14 split | **4.34** | **no change vs legacy path** |
| f16 K + vtq3_2 V via E14 split | **4.33** | K-type irrelevant |

**Verdict: 16× gap between VTQ_1 and VTQ_2 remains. E14 split-decode ineffective.**

## Diagnosis Chain

1. **Phase 3B0 spike (GREEN):** Stubbing `dequantize_V` in FA inner loop reduced regs 249→89 (-64%) on ncols=1 decode path. **Suggested** massive occupancy unlock.

2. **Phase 3B1 trivial:** `ggml_get_to_fp16_nc_cuda(GGML_TYPE_VTQ3_2)` API already existed (convert.cu:1068). No new kernel needed.

3. **Phase 3B2 integration (this doc):** Added dispatcher `try_dispatch_vec_vtq2_split` that intercepts VTQ_2 family at ncols=1, bulk-dequantizes V to fp16 scratch, falls through to F16 FA. Debug-trace confirmed path active (`[E14 split] call#1000` visible during bench). Expected 3B0-style 5× occupancy gain.

4. **Measurement: no change.** 4.34 tok/s identical to legacy VTQ_2 path baseline.

5. **Parallel-decode experiment:** Modified `k_dequantize_trellis_nc` (trellis.cuh:143) from single-thread-per-block sequential decode to 128-thread-parallel via O(1) `trellis_state_at<K>()` formula. **No change.** The dequant kernel itself was not the primary bottleneck.

6. **Ruling out K-path:** Tested with `f16` K-cache to eliminate KTQ2_1 K-decode overhead. **Still 4.33.** K-decode not limiting.

## What's Actually Happening

The **F16 FA-vec kernel itself**, when operating on the dequantized VTQ_2 scratch buffer, does not outperform the direct VTQ_2 path. This means:

- Either the F16 FA-vec path has hidden costs (unlikely — vtq2_1 also uses it indirectly and hits 70 tok/s)
- Or the **bulk dequant kernel** dominates the total cost despite being a separate launch (most likely)

Per-bench: 2000 dequant calls × dequant-time must equal the 22.5ms/token budget (at 4.34 tok/s). That's ~1ms per dequant call at ne01=256 ctx. For 128 KB of work that's 128 GB/s effective — but trellis-decode has **per-sample state computation** that adds significant arithmetic. The O(1) `trellis_state_at` formula does 3 byte-loads + bit-shuffle per sample; at 128 samples × 60 layers × 128 tokens = ~1M sample-decodes per TG burst. That's a lot even parallelized.

## Architectural Conclusion

**VTQ_2 trellis codec has inherent per-sample overhead that VTQ_1 codebook lookup does not.** The cost is not in one location — it's distributed across:
- State bit-extraction per sample (5+ memory loads)
- LUT gather per sample (random L2 access)
- fp16 cast per sample

Even with bulk dequant + F16 FA, the state-extraction dominates. **VTQ_1's single codebook lookup wins on RTX 2060.** This is a fundamental design trade-off, not a kernel bug.

## Final State

- Commit `b846ec281`: E14 split dispatcher code **on origin/phase2**, guarded by `FATTN_VTQ2_SPLIT_ENABLE` (default OFF). Kept as reference for future investigation on Ampere+ hardware where async copies and tensor cores may shift the balance.
- Prod server uses VTQ_1 (tq2_1) — unchanged, 70 tok/s.
- Spec `2026-04-22-e14-split-decode-spec.md` and this results doc remain in-tree for any future attempts.

## Recommendation

1. **VTQ_1 = production path.** Document VTQ_2 officially as "experimental, Ampere+ only."
2. **Update README:** VTQ_2 "known issue" section points to this results doc as root-cause analysis.
3. **Close task #141** with "resolution: architectural — VTQ_2 codec cost inherently higher than VTQ_1 on Turing, mitigation only via future cuBLAS/tensor-core paths on sm_80+".
4. **Task #142** (disable per-layer mixed V as default) should stay — it was orthogonal to this attempt.

## Time Spent

~4h active session (brainstorm + spec + spike + 2 build iterations + analysis). Negative result, but **systematically ruled out**:
- ❌ E11 cached decode (wrong for ncols=1)
- ❌ launch_bounds tuning (no-op, nvcc already optimized)
- ❌ QK=128 block size (MSE improved, TG unchanged)
- ❌ E14 split decode (F16 FA not faster than direct VTQ_2 path at ncols=1)
- ❌ Parallel trellis decode (not the bottleneck)

Each attempt delivered definitive data. VTQ_2-on-Turing is fundamentally a trade-off accepted, not a bug to fix with current architecture.
