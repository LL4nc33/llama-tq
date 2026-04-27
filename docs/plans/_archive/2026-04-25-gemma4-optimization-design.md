# Gemma4-26B-A4B specific optimization opportunities

After resolving the (non-)gibberish bug on 2026-04-25, Gemma4 runs at 81-82 tok/s tg256 with f16/f16 KV. This doc inventories Gemma4-specific opportunities to push that further while keeping PPL within budget. Sweep results land in `bench/plots/benchmarks.csv` once `2026-04-25-gemma4-sweep.log` finishes.

## Architecture facts that matter for KV-cache

Gemma4-26B-A4B (30 layers, MoE A4B):

- **iSWA hybrid**: every 6th layer is sliding-window with `head_count_kv = 2` (GQA(2)). The other 5/6 are full-attention `head_count_kv = 8` (GQA(8)).
- **Per-layer head dim mismatch**: full-attention layers use `n_embd_head_k/v = 512`. SWA layers use `n_embd_head_k/v = 256`.
- **Shared-KV tail**: `n_layer_kv_from_start` cuts off the last few layers from having their own KV — they reuse earlier layers' cache.
- **V is RMS-normed before KV write** (`gemma4-iswa.cpp:92`). No learned weight on that norm. Quantized V therefore sees a different distribution than e.g. Qwen3.6 V (which has no V-norm).

## Lever 1: per-layer V type (already wired)

`cparams.tq_v_layers` already exists (Trick 2 PR2). On Gemma4 we can ship:

| Layer class       | Count (of 30) | head_dim | Best V type        |
|-------------------|---------------|----------|--------------------|
| Full-attention    | ~25           | 512      | `vtq2_2` or `vtq3_2` (large head_dim, more samples → trellis wins) |
| SWA               | ~5            | 256      | `f16` (small enough that quant overhead > savings) |
| Shared-tail (no KV) | ~3          | —        | n/a                |

Mixed config: `--cache-type-v vtq2_2 --tq-v-override "swa=f16"` (CLI flag would need wiring; per-layer vector already supported internally).

## Lever 2: SWA-specific aggressive K quant

SWA layers have GQA(2) — fewer heads, less attention work per layer. PPL impact of quantization on those 5 layers should be small. Try: `ktq1_1` on SWA, `ktq2_1` on full. Saves ~1 bit on 1/6 of layers (small, but free if PPL holds).

## Lever 3: deferred-K only on full-attention layers

Deferred-K quantization (auto-enabled for KTQ types) staging buffer is currently global. On Gemma4, SWA layers have ~4× smaller KV per layer — staging and bulk-converting them is overkill. Skip deferred path on SWA layers, do them inline with f16 staging.

Implementation: `is_swa(il)` check in `llama-kv-cache.cpp` deferred-K path. Estimated win: prefill 5-10% faster on Gemma4 (SWA layers dominated prefill cost in earlier profiling).

## Lever 4: V-RMS-norm-aware VTQ_2 calibration

VTQ_2 trellis uses Lloyd-Max over a calibration distribution. Default calib was done on Qwen V (no pre-quant norm). Gemma4 V is rms-normed → distribution is approximately unit-norm spheres, not the wider Qwen distribution. Re-running the trellis calibration on Gemma4 V samples might shrink PPL delta by 0.2-0.5%.

This is a Python-side prep step (no kernel change). Run `tools/calibrate_vtq2.py` (if it exists; otherwise add) on a few thousand V tensors from Gemma4.

## Lever 5: Skip rms_norm on already-normed V if quantized to TQ

When V will be VTQ-quantized anyway, the rms_norm is partially redundant — the trellis quantizes to fixed centroids regardless. Keeping the norm preserves precision for f16 path but adds a kernel pass.

Test: replace `Vcur = ggml_rms_norm(...)` with passthrough when `cparams.is_vtq_v(il)` is true. Measure PPL impact and TG win. Risk: if the norm changes V scale, downstream attention output scales also change → PPL drift possible. Conservative version: keep norm but route the un-normed V through to TQ encode path.

## Lever 6: Channel/thought reasoning truncation in inference loop

Gemma4 is a **reasoning model** that prepends `<|channel>thought\n...` before the actual answer. For chat workloads where users only want the final answer, this is wasted decode. Options:

- Stop sequence on `<|channel>final\n` → only print after that.
- Skip-and-emit sampling: tokens between `<|channel>thought` and `<|channel>final` go straight to KV without rendering. UI feature, not kernel.

Saves wall-clock latency to first useful token (~50-200 hidden reasoning tokens × decode cost).

## Tested 2026-04-25 (after sweep)

**Lever 1 status: infrastructure works, measurement blocked.**

`--tq-v-base vtq2_2 --tq-v-override 5:f16,11:f16,17:f16,23:f16,29:f16` runs successfully — log confirms `tq-v-mixed: n_layer=30 vtq2_2=25 other=5 avg_bpw=3.55`. SWA layers ([5,11,17,23,29]) get f16, full-attention get vtq2_2.

Issues blocking measurement:
1. **llama-bench doesn't accept `--tq-v-override`** — would need to extend bench arg parser.
2. **llama-perplexity is prefill-only** → no TG difference observable (uniform vs mixed gave identical 17.3s wall time on 5 chunks @ 2048 ctx).
3. **PPL on Gemma4 is broken metric** — wikitext PPL=14000-19000 baseline because the model expects `<|channel>thought` chat prefix; raw text is OOD.
4. **llama-cli with `-no-cnv` doesn't print `llama_perf_context_print` for Gemma4** — no eval time output at all.

Marginal expected gain: SWA is 5 of 30 layers → max ~15% of V-cache work touched. With f16 swap-in being slower than vtq2_2 per layer, net might even be negative on TG.

**Verdict: Lever 1 deprioritized.** The actual bottleneck from sweep data is VTQ_1 family on D=512 (PP −24.5% vs −6.5% on Qwen D=128). That's a kernel-vectorization issue, requires rewrite of `fattn-vec-vtq1.cuh` inner loop for D=512 stride. Major work, deferred.

VTQ_2 already at near-f16 performance on Gemma4 (`f16/vtq2_2` = −1.0% PP, −2.4% TG). Production recommendation for Gemma4: stick with `f16/vtq2_2` or `ktq2_1/vtq2_2`.

## Priority (revised)

1. **Lever 1** (per-layer V type) — code path exists, just CLI ergonomics + bench. ~2h work, expected 5-15% TG win plus PPL near baseline.
2. **Lever 3** (skip deferred-K on SWA) — simple guard. ~1h work, prefill speedup.
3. **Lever 4** (V-norm-aware VTQ_2 calib) — offline calib, no risk to other models. PPL improvement only, no TG change.
4. **Lever 6** (reasoning skip) — UX win, server-side feature, doesn't help raw bench numbers.
5. **Lever 2 / 5** — risky, defer until Levers 1+3 verified.

## How sweep results inform this

The 21-config sweep currently running on gpu00 will show:
- Whether VTQ_2 on Gemma4 has the same Pareto position as on Qwen3.6 (`f16/vtq2_2` near-free) → if not, Lever 4 (V-norm calib) is needed.
- Whether `ktq2_1/vtq2_2` keeps the +slight-PPL-improvement we see on Qwen3.6 (-0.04%) → if it goes positive, the V-norm interaction is the suspect.
- Absolute TG headroom vs f16/f16 — gives us the measurement floor to evaluate Lever 1.

Wait for sweep, then pick top 1-2 levers to implement.
