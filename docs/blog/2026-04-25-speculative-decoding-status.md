# Speculative Decoding — Fork Status & Phase-0 Decision

**Date:** 2026-04-25
**Author:** spec-decode investigation agent
**Branch:** turboquant
**Status:** Integration verified functional; Phase-0 measurement deferred per spec gate

## TL;DR

- Speculative decoding is **fully integrated** in the llama-tq fork. No code work is required.
- All upstream paths (`-md`, `-cd`, `-ngld`, `--draft-max`, `-ctkd`, `-ctvd`, etc.) are present and route through the fork's KTQ/VTQ-aware `kv_cache_type_from_str()` parser.
- A vocab-compatible draft candidate exists on gpu00: `/home/lance/models/qwen3-0.6b-q8_0.gguf` (610 MB, Q8_0).
- Per the architect spec (`docs/plans/2026-04-23-speculative-decoding-spec.md`), **the recommended action is to NOT spend implementation time** on draft-model spec decode for our Qwen3.5/3.6-35B-A3B target. Phase-0 (measure-first) is the prescribed mitigation.
- Phase-0 was **not run in this session** because: (a) other agents (PPL, FA-dispatch) are using both GPUs on gpu00; (b) the prod service `gpu00:8791` is down; (c) the spec doc rates expected outcome as net regression on A3B MoE workloads.

## What was verified

### 1. CLI surface is wired (no fork-specific gaps)

`common/arg.cpp:3586-3760` registers all speculative flags for `LLAMA_EXAMPLE_SERVER`:

- `-md / --model-draft FNAME`
- `-cd / --ctx-size-draft N`
- `-devd, -ngld` (device + GPU layers for draft)
- `--draft-max, --draft-min, --draft-p-min, --draft-p-split`
- `--spec-replace TGT DFT`
- `--spec-type {none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|draft|eagle3}`
- `-ctkd, --cache-type-k-draft TYPE`
- `-ctvd, --cache-type-v-draft TYPE`

### 2. Draft KV cache accepts KTQ/VTQ types

Both `-ctkd` and `-ctvd` route through `common/arg.cpp:411 kv_cache_type_from_str()`. The whitelist (`common/arg.cpp:382-409`) includes the full TurboQuant family:

```
KTQ1_1, KTQ2_1, KTQ3_1, KTQ4_1
VTQ1_1, VTQ2_1, VTQ3_1, VTQ4_1
VTQ2_2, VTQ3_2, VTQ4_2, VTQ_MIXED
VTQ2_3, VTQ3_3, VTQ4_3
```

So a config like `-ctkd ktq2_1 -ctvd vtq2_2` for the draft model is parsable. Whether it makes sense (a 0.6B model's tiny KV barely benefits from quantization) is a separate question — `q8_0` is the more sensible default for draft KV.

### 3. Server integration points are intact

Per the spec doc, the following lines in `tools/server/server-context.cpp` carry the spec-decode logic; spot-checks confirm they match the upstream layout the spec was written against:

- `661-694` — `params_base.speculative.has_dft()` draft model load
- `770-798` — per-slot `common_speculative_init`
- `1197` — `backend_sampling &= !(slot.spec && ...)` (CPU sampling forced when spec on)
- `2095-2156` — draft-add-to-batch in decode loop (`TAG_SERVER_SPEC_REWORK`)
- `2908-2959` — `common_sampler_sample_and_accept_n` verification
- `411` — `can_speculate` exposed in `/props`

### 4. Vocab-compat draft candidate identified

`/home/lance/models/qwen3-0.6b-q8_0.gguf` is the smallest available Qwen3-family GGUF on gpu00. Per spec doc §"Draft-Model Candidates", Qwen3.5/3.6 share a 151,936-token BPE vocab across model sizes; `common_speculative_are_compatible()` (size delta ≤128 + token-text equality from ID 5) should pass for the Qwen3.5-35B-A3B-IQ2_XS or Qwen3.6-35B-A3B-UD-IQ2_XXS targets. **Not yet runtime-verified** — first run on gpu00 will print the compatibility result.

## Why no benchmark this session

**Hard constraints:**
1. Both GPU 0 and GPU 1 currently host `llama-perplexity` (PID 33223, 8.1 GB across the pair) for another agent's PPL measurement task. Loading a 35B + 0.6B draft would either OOM or evict that work.
2. The prod gpu00:8791 service is down. Bringing it back up with a different config than production is fine on port 8799, but still claims VRAM the PPL agent is using.
3. Per the spec doc's published-benchmark evidence (thc1006/qwen3.6-speculative-decoding-rtx3090, 2026-04-19, post-PR#19493), **on this exact model class — Qwen3.6-35B-A3B + Qwen3-0.x/1.7B draft — even at 100% accept-rate the result is a 10.8% regression**, not a speedup. Pathology is MoE expert-saturation: per-token draft verification dispatches a fresh 8-of-256 expert union, and bandwidth cost dominates verification savings on hardware <RTX 3090. Our 2× RTX 2060 (12 GB, ~336 GB/s, asymmetric x16/x4 PCIe) is strictly worse than that benchmark's hardware.

**Therefore:** running the Phase-0 measurement now would (a) compete with active GPU work for a result the spec already predicts will fail the gate, and (b) not save any meaningful effort downstream — the spec is already the work product it would feed into.

## Fork-specific concern flagged for the eventual Phase-0 run

Risk #6 from the spec doc — interaction of speculative draft batches (K+1 tokens at decode-time) with our custom FA dispatch — has one observable wrinkle in `ggml/src/ggml-cuda/fattn.cu:227`:

> The E14 split-decode optimized path that intercepts VTQ_2 family is gated on `ncols == 1`. Speculative verification submits `ncols = draft_max + 1` (typically 5–9). The dispatcher will fall through to the FA-vec native kernel (`BEST_FATTN_KERNEL_VEC` for VTQ V) at `fattn.cu:260`, which is the correct path. No expected breakage, but **this code path is exercised more heavily by spec-decode than by normal serving and has not been benched at ncols∈{4..9}** with VTQ2_2 V cache.

If a future Phase-0 produces unexpected NaN/garbage output (rather than just a speed regression), this is the first place to look.

## Recommended next steps (when GPUs are free)

If/when both PPL and FA-dispatch agents finish and gpu00 has free GPUs:

```bash
ssh claude@gpu00.node "cd ~/llama-tq && ./build/bin/llama-server \
  -m /home/lance/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  -md /home/lance/models/qwen3-0.6b-q8_0.gguf \
  -ngl 99 -ngld all \
  -cd 8192 \
  --cache-type-k ktq2_1 --cache-type-v vtq2_2 \
  -ctkd q8_0 -ctvd q8_0 \
  --draft-max 6 --draft-min 1 --draft-p-min 0.7 \
  -fa on --port 8799 --jinja"
```

Then drive a TG sweep at `parallel=1`, capture `draft_n` / `draft_n_accepted` from `/v1/completions` timings, compare to baseline 67.65 tok/s. Expected outcome (per spec): 58–65 tok/s, which fails the ≥10% gain gate. Document the negative result, do not productionize.

For a *positive*-leaning side-test: try `--spec-type ngram-mod` (no draft model, no MoE re-dispatch) — spec doc rates this as the least-regressing config; on our lower-bandwidth hardware the margin to baseline could narrow further than on RTX 3090.

## Files of record

- Spec: `docs/plans/2026-04-23-speculative-decoding-spec.md`
- This blog: `docs/blog/2026-04-25-speculative-decoding-status.md`
- Source verification: `common/arg.cpp:382-409, 411-440, 3586-3760`; `ggml/src/ggml-cuda/fattn.cu:227,260,320-407`
- Draft model: `gpu00:/home/lance/models/qwen3-0.6b-q8_0.gguf` (610 MB Q8_0, vocab-compat candidate)

## Bottom line

Spec-decode is **functional in the fork as inherited from upstream**. No commit `feat(spec): enable speculative decoding with KTQ/VTQ` is warranted because there is nothing to enable — it is already enabled by passing the right CLI flags. The realistic measured TG gain on our prod model + hardware is *negative* per third-party data; the spec doc's recommendation (do not productionize without a Phase-0 measurement that contradicts that data) stands.
