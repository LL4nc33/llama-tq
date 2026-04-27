# Speculative Decoding Integration — Implementation Spec

**Date:** 2026-04-23
**Target:** llama-server (gpu00:8791) running Qwen3.5-35B-A3B-IQ2_XS with KTQ2_1 KV, f16 V, parallel=2
**Author:** architect (research phase)
**Status:** Proposed — proceed with caution (see Motivation)

---

## TL;DR

1. **llama.cpp already ships full speculative-decoding support in llama-server.** No new integration code is required. All plumbing exists: `--model-draft`, `--draft-max`, `--draft-min`, `--draft-p-min`, `--gpu-layers-draft`, `--device-draft`, per-slot `common_speculative` state, and modes `draft | eagle3 | ngram-*`. See "Existing Integration" below.
2. **Public benchmark data on the exact same class of workload (Qwen3.6-35B-A3B on a single RTX 3090, post-PR#19493, 2026-04-19) shows 10.8% REGRESSION, not speedup, even at 100% draft acceptance.** Root cause is an MoE expert-saturation pathology specific to A3B-class sparsity. [Source: thc1006/qwen3.6-speculative-decoding-rtx3090.](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
3. **Our hardware (2× RTX 2060 12GB, asymmetric x16/x4 PCIe) is strictly worse than the RTX 3090 single-GPU case in that benchmark.** We should expect equal or worse behaviour.
4. **Recommendation:** Do NOT invest implementation effort in draft-model speculative decoding on Qwen3.5-35B-A3B as a primary TG speedup path. Phase work should be **measurement-first** (Phase 0, ~2h) to confirm the regression locally, then either stop, switch to ngram-mod (no draft model, lower overhead), or target a dense (non-MoE) model where speedup is realistic.

The rest of this document describes what the work *would* look like if we proceed, and the measurement plan to decide whether we do.

---

## Motivation

Goal stated by user: 2–3× TG speedup on Qwen3.5-35B-A3B-IQ2_XS by using a small draft model (Qwen3.5-0.8B), baseline 67.65 tok/s.

Why the goal is likely unreachable on this model + hardware combination:

- **A3B MoE routes 8-of-256 experts per token (sparsity ≈ 0.031).** Per-token draft verification dispatches a new set of experts each step. The expert-loading cost is the decode bottleneck on bandwidth-limited hardware.
- **Expert-saturation threshold** (published figure ≈ 94 tokens) is the draft batch size at which unique-expert union stops growing. Below it, more draft tokens = more experts to load = more memory traffic, which cancels verification savings.
- **Our draft batch K is 4–8** (typical `--draft-max`). Well below saturation. Every extra accepted token costs more bandwidth than it saves.
- **100% acceptance does not rescue this.** The benchmark confirmed 100% accept-rate with Qwen3.5-0.8B draft and still observed regression.

This pathology is model-architecture specific. It does NOT apply to:
- Dense models (Llama 3.x 70B, Qwen3 dense 14B/32B)
- Larger MoE variants (A10B+)
- Compute-bound regimes (large batch, prefill)

It DOES apply to us.

## Existing Integration (verified — no work needed)

llama-server already implements full draft-model speculative decoding. Evidence:

- `common/speculative.h` — stable API (`common_speculative_init`, `_draft`, `_accept`, `_begin`, `_is_compat`)
- `common/speculative.cpp:50` — `common_speculative_are_compatible()` validates vocab type, add_bos/add_eos, bos/eos IDs, vocab size delta (≤128), and byte-for-byte match of token text from ID 5 upward. This is the "same tokenizer" guard.
- `common/arg.cpp:3586-3760` — all CLI flags registered for `LLAMA_EXAMPLE_SERVER`:
  - `-md, --model-draft FNAME`
  - `-cd, --ctx-size-draft N`
  - `-devd, --device-draft DEV`
  - `-ngld, --gpu-layers-draft N|auto|all`
  - `--draft-max, --draft-min, --draft-p-min, --draft-p-split`
  - `--spec-replace TGT DFT` (token-text translation when vocabs differ)
  - `--spec-type {none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|draft|eagle3}`
  - `-ctkd, --cache-type-k-draft`, `-ctvd, --cache-type-v-draft`
- `tools/server/server-context.cpp:661-694` — draft model load path (`params_base.speculative.has_dft()`)
- `tools/server/server-context.cpp:770-798` — per-slot `common_speculative_init` in slot setup
- `tools/server/server-context.cpp:2101-2156` — draft-add-to-batch in the decode loop
- `tools/server/server-context.cpp:2908-2959` — target-verify-and-accept path (`common_sampler_sample_and_accept_n`)
- `tools/server/server-task.h:275-277` — `draft_n`, `draft_n_accepted` exposed in timings response
- `tools/server/server-context.cpp:411` — `can_speculate` reported in `/props`

**Known gaps in upstream:**

1. `server-context.cpp:1197` — `backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0)` disables GPU-side batched sampling when spec is active; forces CPU sampling path.
2. `server-context.cpp:2102` (`TAG_SERVER_SPEC_REWORK`) — each slot has its own `common_speculative` (one draft context per slot). The multi-slot rework to share a single draft context across slots is an outstanding TODO. With `--parallel 2`, that means **2× draft model memory** and **serial (not batched) drafting** across slots.
3. Eagle3/ngram-map support was added in the same wave as PR#19493. The benchmark above tested the post-merge state.

## Draft-Model Candidates (Qwen family)

Vocab compatibility requirement per `speculative.cpp:50-107`: same vocab type, same special tokens, size delta ≤128, token text identical from ID 5. Qwen3.5 family (incl. 35B-A3B) uses a 151,936-token BPE vocab shared across sizes. The published benchmark and LM Studio confirm Qwen3.5/Qwen3.6 dense models at 0.6B, 0.8B, 1.7B are vocab-compatible drafts for the 35B-A3B target.

| Draft candidate      | Params | GGUF @ Q4_K_M | Est. draft step latency | Vocab-compat with tgt | Memory cost on 24GB | Notes |
|----------------------|--------|---------------|-------------------------|-----------------------|---------------------|-------|
| Qwen3.5-0.6B         | 0.6B   | ~0.4 GB       | ~1.5 ms                 | YES                   | ~0.5 GB + KV        | LM Studio reports ~60% lower accept-rate than 1.7B draft |
| **Qwen3.5-0.8B**     | 0.8B   | ~0.5 GB       | ~2 ms                   | YES (vocab 248320-byte compat confirmed by benchmark) | ~0.6 GB + KV | **Benchmark reference config. 100% accept-rate on A3B target.** Still regressed. |
| Qwen3.5-1.7B         | 1.7B   | ~1.1 GB       | ~3.5 ms                 | YES                   | ~1.3 GB + KV        | Higher accept-rate, higher per-draft latency. Likely worse for MoE saturation pathology (more draft time per token). |
| Qwen2.5-0.5B / 1.5B  | 0.5/1.5B | ~0.3/1.0 GB | similar                 | NO — different tokenizer generation | — | Rejected; requires `--spec-replace` tables with unknown coverage and would degrade accept-rate. |

**Recommendation (conditional):** If Phase 0 measurement unexpectedly shows a net gain, use **Qwen3.5-0.8B Q4_K_M** as draft. It is the smallest vocab-verified candidate. Rationale: on a decode-time-limited MoE target, we want the cheapest possible draft forward, because draft cost is effectively pure overhead unless the expert-loading pathology reverses.

## Integration Points (file:line)

All edits below are opt-in CLI-flag driven. No code change is required to enable speculative decoding — upstream already supports it. These are the spots to be aware of if we later do a custom rework.

| Purpose | File | Lines | Notes |
|---------|------|-------|-------|
| Draft model load | `tools/server/server-context.cpp` | 661–694 | `has_dft()` branch. Respects `-cd`, `-ngld`, `-devd`, cache-type-draft. |
| Per-slot spec init | `tools/server/server-context.cpp` | 770–798 | One `common_speculative` per slot. This is the 2× memory point. |
| Draft batching into decode | `tools/server/server-context.cpp` | 2095–2156 | `TAG_SERVER_SPEC_REWORK` TODO for single shared draft ctx. |
| Target sample-and-accept | `tools/server/server-context.cpp` | 2908–2959 | `common_sampler_sample_and_accept_n` — verification. |
| Backend sampling gate | `tools/server/server-context.cpp` | 1197 | Forces CPU sampling when spec is on. Small extra overhead. |
| Stats reporting | `tools/server/server-task.h` | 275–277 | `draft_n`, `draft_n_accepted`. Feeds `/v1/completions` timings. |
| Vocab compat check | `common/speculative.cpp` | 50–107 | Fails closed if tokenizers differ. |
| CLI flags | `common/arg.cpp` | 3586–3760 | All flags already registered for `LLAMA_EXAMPLE_SERVER`. |

## Memory Budget (2× RTX 2060 12GB, 24 GB total)

Current Qwen3.5-35B-A3B-IQ2_XS deployment (from memory + CLAUDE.md):
- Target weights IQ2_XS: ~10–11 GB
- KV cache KTQ2_1 K / f16 V @ 400k ctx × 2 slots: substantial — user reports 200K/slot working
- Headroom is already tight.

Adding Qwen3.5-0.8B Q4_K_M draft:
- Weights: ~0.5 GB
- Draft KV at matching 200k ctx: with `-ctkd q4_0 -ctvd q4_0` (let's approximate) — draft uses a full `n_ctx_seq` context; for 0.8B (~24 layers × 1024 hidden): ~0.5–1 GB / slot. With 2 slots: 1–2 GB.
- Per-slot duplication due to `TAG_SERVER_SPEC_REWORK`: multiplier applies.

**Verdict:** ~2–3 GB additional VRAM with `--parallel 2` and 200k draft ctx. Feasible but eats into our headroom. If it pushes us to OOM, reduce `-cd` to e.g. 32k (drafts only see recent context anyway).

## Performance Prediction (honest)

Baseline: 67.65 tok/s TG on target.

Per-token decode time: 1000/67.65 ≈ 14.8 ms.
Draft forward (0.8B on RTX 2060): ~2.5 ms estimated. Ratio 2.5/14.8 ≈ 0.17.

**Naïve speculative formula:** net_speedup ≈ (1 + N_accepted) / (1 + N_draft × r_draft)
With N_draft=5, r_draft=0.17, N_accepted=4 (80% accept): (1+4) / (1+5×0.17) ≈ 5 / 1.85 ≈ **2.7×**.

**This is the marketing number. Ignore it.** It assumes target forward time is independent of draft batch size. For A3B MoE it is not.

**Corrected prediction (informed by benchmark):**
- Target forward over a K-token draft batch on MoE costs ≈ base_decode_ms × (1 + α × K) where α captures extra-expert-load fraction.
- The 2026-04-19 benchmark measured α such that even at K=5–8 the gain from verification is fully cancelled for A3B on RTX 3090 (8–12% net regression at 100% accept).
- Our RTX 2060 has **lower memory bandwidth** (~336 GB/s per GPU, split across tensor-split), which makes expert loading MORE expensive relative to compute. The regression will be equal or deeper.

**Expected outcome:** TG drops from 67.65 to 58–65 tok/s. No user-visible win. Latency per first-token improves slightly because nothing in spec affects prefill.

**Where we could get a win:**
- **ngram-mod / ngram-map-k** — no draft model, so no MoE re-dispatch for draft; tokens come from n-gram tables built from prior context. Benchmark reports these are the least-regressing configs. Still not a net gain there, but the margin is much smaller, and on a 2060 (worse bandwidth) the gap could conceivably close.
- **Dense Qwen3 32B target** — speculative decoding works as advertised on dense models; 2× is realistic.

## Phases

### Phase 0 — Measure before we commit (MANDATORY, ~2 h)

No code. Purely operational.

1. Pull Qwen3.5-0.8B Q4_K_M (or whatever smallest vocab-compat GGUF is available).
2. Restart current gpu00:8791 service with extra flags:
   ```
   -md /models/qwen3.5-0.8b-q4_k_m.gguf \
   -ngld all \
   -cd 32768 \
   -ctkd q8_0 -ctvd q8_0 \
   --draft-max 6 --draft-min 1 --draft-p-min 0.7
   ```
3. Run `llama-bench`-style TG sweep at batch=1, parallel=1 and parallel=2.
4. Capture `/v1/completions` timings showing `draft_n`, `draft_n_accepted`, observed TG tok/s.
5. Repeat with `--spec-type ngram-mod` (no `-md`).
6. Compare to baseline 67.65 tok/s.

**Gate:** If Phase 0 does not show ≥10% net TG gain over baseline on representative prompts (coding, German chat, long-context), stop here. Document the negative result. Do not proceed.

Effort: 1–2 h wall time, mostly model download and service restart.

### Phase 1 — Productionize config (only if Phase 0 passes) — ~2 h

- Add `--model-draft` and friends to the on-llm.service systemd unit or deploy script.
- Update `FORK_CHANGES.md` and `CLAUDE.md` phase list.
- Verify Cortex `/v1/completions` clients still parse timings (new fields `draft_n`, `draft_n_accepted`).
- Add a WebUI tooltip in the status bar showing accept-rate.

### Phase 2 — Slot rework (only if Phase 1 shows gain AND --parallel 2 regresses vs --parallel 1 spec) — ~12–20 h

Address `TAG_SERVER_SPEC_REWORK` at `server-context.cpp:2102`: share a single draft `llama_context` across all slots, batching draft forwards over the union of slots.

- Refactor `common_speculative` to decouple state-per-slot from context-shared.
- Modify slot setup to pass a ref to a shared draft ctx rather than per-slot `common_speculative_init`.
- Handle token-sequence identity per slot in the batched draft forward.
- Tests: parallel=1 equivalent, parallel=2 reduces draft VRAM by ~50% and draft time by ≥30%.

High-risk. Touches upstream-community-owned code. Prefer upstreaming this change via PR rather than carrying it as a fork delta.

### Phase 3 — Streaming compatibility — already works

Streaming response path (`process_token` → SSE) is orthogonal to spec decode; accepted tokens are flushed per-verification in the existing code path. No change needed.

### Phase 4 — Skip (multi-slot not worth it)

Until Phase 0 validates the pathway, do not plan multi-slot work.

**Total effort ceiling if everything goes green:** ~16–24 h, dominated by Phase 2.
**Expected effort:** 2 h (Phase 0 only, negative result).

## Risks

1. **Primary risk: net regression** — covered above. Phase 0 is the mitigation.
2. **VRAM pressure** — draft model + draft KV × 2 slots may push us past 24 GB total. Mitigation: reduce `-cd` to 32k, use `-ctkd q8_0 -ctvd q8_0` for draft KV.
3. **Vocab compat silent failure** — `common_speculative_are_compatible` checks text equality from ID 5. Any custom-tokenizer variant of Qwen 0.8B (e.g. community fine-tunes with added special tokens) will fail the size-delta check and refuse to load. Mitigation: use stock `Qwen/Qwen3-0.8B` GGUF, not a derivative.
4. **Per-slot draft context duplication** — memory + draft-latency cost is linear in `--parallel`. Plan for single-slot testing first.
5. **Backend sampling disabled with spec** (`server-context.cpp:1197`) — minor TG penalty on top of the MoE regression. Won't show as a separate effect in Phase 0; included in the measured number.
6. **Interaction with our custom KV types (KTQ2_1, VTQ, etc.)** — `common_speculative_is_compat(ctx)` is called at `server-context.cpp:770`. It clears target KV memory. Safe but means Phase 0 must re-prefill. Verify our custom fattn dispatch paths in `fattn.cu` accept the draft batch sizes; there have been prior bugs with TQ types + unusual batch configurations (see `project_on_llama_tq_bugs.md`).
7. **Non-determinism from CPU sampling path** — acceptable, but flag it in QA.

## Open Questions

1. Is there a Qwen3.5-0.8B Q4_K_M GGUF with confirmed vocab size 151,936 (not 248,320 as the benchmark used — that appears to be a Qwen3.6 variant)? If not, what does the size-delta gate say for a mismatched Qwen3.5-0.8B against a Qwen3.5-35B-A3B target? Verify at download time before Phase 0.
2. Does `--spec-type ngram-mod` actually provide any benefit on our bandwidth profile? Benchmark had it near-tied with classic draft. Worth a 30-minute side-test in Phase 0.
3. If the result is negative, is there appetite for keeping a dense Qwen3-14B model deployed as a "speculative-happy" alternative for the use cases where TG matters most (code completion)? Not in scope of this spec, but raise with user after Phase 0.

## Decision Record (ADR-style)

**Status:** Proposed
**Date:** 2026-04-23
**Context:** User requested 2–3× TG speedup via draft-model speculative decoding integrated into llama-server for our Qwen3.5-35B-A3B IQ2_XS production deployment.
**Decision:** Do NOT write integration code. The feature is already in upstream. Run a 2-hour measurement (Phase 0) using existing CLI flags before any further work. Default assumption based on third-party benchmark data: this will not produce the requested speedup on an A3B MoE target.
**Consequences:**
- Positive: avoids 12–20 h of effort on a likely-dead path; forces empirical validation first.
- Negative: user expectation of 2–3× will not be met by this work package. Re-scope conversation required after Phase 0.
- Neutral: Phase 0 artifacts (benchmark script, systemd unit drop-in) are reusable for any future draft-model config test (e.g. against a dense target).

## References

- [llama.cpp speculative decoding discussion #10466](https://github.com/ggml-org/llama.cpp/discussions/10466)
- [thc1006/qwen3.6-speculative-decoding-rtx3090 — primary evidence](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) — 135.7 → 121.1 tok/s regression, 100% accept, Qwen3.5-0.8B draft, RTX 3090, post-PR#19493.
- [LM Studio speculative decoding guide](https://lmstudio.ai/docs/app/advanced/speculative-decoding) — Qwen3 0.6B vs 1.7B accept-rate trade-off.
- Local code: `common/speculative.{h,cpp}`, `common/arg.cpp:3586-3760`, `tools/server/server-context.cpp:661-2959`.
