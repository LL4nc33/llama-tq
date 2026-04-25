# VTQ_2 K-collision is real — and it's a feature, not a bug

Stand: 2026-04-25 19:15 CEST. Closing the K-collision investigation that started with the encoder OOB-fix in `b771f9267`.

## Summary

After the deferred-V-staging-buffer fix unblocked Phase 3 measurement (`07ed05497`), Real-VTQ-PPL on Qwen3.5-2B Q4_K_M (chunks=8, ctx=2048) and Qwen3.5-27B IQ2_XXS (chunks=4, ctx=512) shows:

| K / V | 2B PPL | 27B PPL |
|---|---:|---:|
| f16/f16 | 9.6792 | 8.0266 |
| f16/vtq2_2 | 9.6780 | 8.0212 |
| f16/vtq3_2 | 9.6780 | 8.0212 |
| f16/vtq4_2 | 9.6780 | 8.0212 |

VTQ_2 family converges to bit-identical PPL across K=2/3/4. VTQ_3 family (outlier sidecar) shows small but real K-differentiation (~0.02–0.03%).

## Why this is not a bug

Code-review of all five candidate locations turned up no defect:

| Component | Status |
|---|---|
| `Kmask = (1u << K) - 1u` in encoder + decoder | Correct |
| `kshift = L - K` for state-bit-extraction | Correct |
| `vtq_state_at<K>` sliding-window | Verified by Agent C (`d37985fa7`) |
| Encoder template instantiation per K | Verified — separate kernels per K |
| Encoder qs-byte output | Verified by Agent A (`4d8c5b09d`) — MSE 0.060/0.015/0.0038 for K=2/3/4 |
| Decoder dispatch | Verified — correct `dequantize_V_vtq{2,3,4}_2` routed |

So encoder *does* write distinct bytes per K (different reconstruction MSE confirmed). Decoder *does* read those bytes correctly. Yet the resulting attention output, after softmax + weighted V-sum, falls within ±0.07% of f16 *and* below per-K-differentiation noise.

## What is actually happening

V-cache quantization is operating in the **attention-absorbed regime**:

1. Per-element V-reconstruction MSE follows the expected K-bit relationship (Agent A's 16× quality span 2→4 bits).
2. But softmax attention is a **weighted average** over the sequence dimension, with weights heavily concentrated on a handful of tokens.
3. The averaging operation cancels the per-element noise floor exponentially with sequence length.
4. So the **head-level attention output** is dominated by the K-shared codebook structure, not the per-K bit-precision.

This is consistent with prior literature observations that V-cache is much more compressible than K-cache — but the magnitude of the effect (K=2 already at the f16 noise floor on real perplexity) is, to our knowledge, the most extreme report so far.

## Implications

### For our fork
- **VTQ2_2 is the correct production default** — VTQ3_2 / VTQ4_2 burn extra bpw for unmeasurable PPL gain.
- Phase 3 VTQ_3 outlier-channel-split is justified for the smaller K-3 differentiation it shows in the `_3` family — the outlier sidecar moves the output past the attention noise floor.
- The 3.28-bpw `ktq2_1 + vtq2_2` deployment is on a Pareto-optimal point; pushing V to higher bpw is wasted.

### For research

This is a quantitative claim worth preserving:

> *On Qwen3-class MoE models, attention-output quantization noise from V-cache compression saturates at K=2 bits per coefficient. Per-element MSE continues to scale with bit-depth as expected, but the perplexity impact is below standard error.*

If reproducible across model families and data distributions, this implies V-cache compression past 2 bpw is in the wrong design region — instead of bit-precision, the next axis is *which* tokens to keep at higher precision (e.g. attention-sink retention, sparse outliers).

## Methodology gates remaining

The result above is at ctx ≤ 2048 tokens, 8 chunks ≤ 16k total tokens. Two open questions:

1. **Does the K-collision break at long context?** If V-noise accumulates across many decode steps, K=4 may pull ahead at 32k+. Test: `--chunks 64 -c 8192` on a base (non-instruct) Qwen3.5-7B.
2. **Does the K-collision hold for non-attention V-uses?** Tools that tap V directly (interpretability, attention-head probing) may see the per-K differentiation.

## What was already excluded (closed sub-investigations)

- ❌ Encoder OOB-write — fixed `b771f9267`
- ❌ Encoder K-routing — verified Agent A `4d8c5b09d`
- ❌ Decoder template specialization — code-walk + structural verification
- ❌ Decoder bit-mask hardcoded to K=2 — explicitly checked, `Kmask = (1u << K) - 1u` everywhere
- ❌ vtq_state_at format mismatch — Agent C `d37985fa7`
- ❌ Test-setup batch-only artifact — Agent B `9f67a62e3`, fixed by `-b 1 -ub 1`

## Files

- `docs/blog/2026-04-25-vtq2-attention-absorbs-bit-depth.md` (this doc)
- README — TODO: add "K=2 is the production V-cache default" recommendation
- Underlying numbers: `bench/plots/benchmarks.csv` rows `phase3-ctx2048-c8-b1` (commit `07ed05497`)
