# llama-tq Roadmap

Last updated: 2026-04-20
Maintainer: LL4nc33

## Status: Phase 1 complete ✅

The `trellis-v2-phase1` work is merged to `master` and validated on
the author's hardware. The VTQ_2 Trellis-coded V-cache pipeline is
functional and deployable.

### Reference numbers (Qwen3.5-35B-A3B IQ2_XS on 2× RTX 2060 12 GB)

| config | tg tok/s | speedup vs old |
|--------|----------|----------------|
| old (f16 V-cache) | ~8-10 | baseline |
| new (vtq3_2 + ktq2_1 + deferred + sink, parallel=2) | **~66.7** | **7×** |

Measurement: 250-token generation, German prompts, 2026-04-20.

### What Phase 1 shipped

**Types:** VTQ2_2 / VTQ3_2 / VTQ4_2 (2.06 / 3.06 / 4.06 bpw V-cache,
bit-exact Trellis-Coded Quantization with Viterbi encoder and
shift-register decoder)

**CUDA path:** dequant kernel, Viterbi encoder (~57μs/call),
Flash-Attention dispatch, `convert.cu` + `set-rows.cu` integration

**Runtime features:**
- `--tq-deferred-v` — f16 staging buffer, bulk Viterbi at the prefill→decode transition
- `--tq-protect-sinks N` — StreamingLLM-inspired protection of the first N tokens
- `--tq-protect-layers N` — boundary protection (first/last N layers in q8_0)
- CLI flags in `server`, `cli`, `perplexity`, `bench`, `mtmd`

**Validation:**
- Qwen3.5-0.8B tg512: 7 → 196 t/s (28×)
- Qwen3.5-27B dual-GPU tg1024: 14.62 vs f16 14.89 (-1.8%)
- Qwen3.5-35B-A3B live: byte-identical deterministic output, ~6× faster
- PPL: vtq3_2 +1.9%, vtq2_2 +8.0%, vtq4_2 +0.6% (wikitext-2)
- Zero crashes in 6h+ cumulative runs

**Reference recipe:**
```
--cache-type-k ktq2_1 --cache-type-v vtq3_2 \
--tq-deferred-k --tq-deferred-v --tq-protect-sinks 4
```

---

## Benchmark track (continuous, parallel to all phases)

Solid baselines are a prerequisite for every optimization. Running
measurement series in `docs/plans/benchmarks/` (one file per run,
`YYYY-MM-DD-topic.md`).

**Regular:**
- Master-vs-older-commit comparison (regression check)
- Context scaling: tg @ 4k / 16k / 64k / 200k ctx length
- Model scaling: 0.8B / 2B / 27B / 35B with fixed V recipe
- Competitor comparison: Q4_K_M / IQ4_XS / Q8_0 V vs VTQ_2

**Ad hoc per trick:**
- PPL wikitext-2 10/40 chunks
- tg/pp bench on the same hardware config
- Memory footprint

---

## Phase 2 — Polish the current version (immediate)

Goal: make the existing VTQ_2 implementation progressively more
robust and faster without expanding scope. Small wins, lots of
measurement.

**Open items from Phase 1:**
- ~~35B deploy~~ ✅ DONE 2026-04-20
  (ctx=200K instead of the planned 400K due to compute-buffer OOM at parallel=2)
- Make PPL-prefill measurable under `--tq-deferred-v` mode (state
  currently stays in STAGING during pure prefill)
- 27B pp1024 target −3% under f16 — optimize the bulk-Viterbi transition
- 400K ctx on 2× 12 GB GPU: either parallel=1 or smaller ubatch
- CUDA kernel review: `__syncthreads()` placement still optimal everywhere?
- Error paths: OOM, invalid `-ngl`, broken GGUFs — graceful failure
  messages instead of crashes

**Quality of life:**
- Clean up `--help` output for VTQ flags
- Review defaults (should the TQ flags have a default-on profile?)
- Error logging on failed dequant (silent f16 fallback at the moment)

**Timeframe:** 1-3 weeks. No research, only polish.

---

## Phase 3 — Trick-17 series (research lane)

Goal: improve the quality/bpw ratio through smarter quantization
algorithms. Each trick is independent and has its own measurement
gate.

See `tests/trellis-phase1/BACKLOG.md` for details. Hard limit: never
more than 17. New ideas replace old ones.

**Done:**
- Trick 1 — Attention-sink protection (layer-level)
- Trick 3 — Per-model RHT seed calibration

**Next:**
- Trick 2 — Per-head precision mixing (high variance → higher bpw)
- Trick 4 — Correction overlay buffer (lossless top-N patch)
- Trick 5 — Per-head learned lambda sharpening (needs training)

**Later (6-16):** see BACKLOG — FWHT per token, deferred K hybrid
precision, learned RHT matrix, block-variable bpw, adaptive Lloyd-Max.

**Trick 17** — "The big one". Reserved. If it lands, the paper is
written.

**Timeframe:** parallel to Phase 2, 1-2 weeks per trick.

---

## Phase 4 — TQW2 weight quantization (the big lever)

VTQ was KV-cache only. Model weights remain IQ2_XXS / Q8_0.
**Weights are the bulk of VRAM** — TQW2 would push weights themselves
into the 2-3 bpw range with Lloyd-Max-level quality.

**Status:**
- Python validation: RHT + Lloyd-Max vs IQ2_XXS MSE — DONE (Task #126)
- CUDA sprint: open (Task #127, in progress)

**Open questions:**
- Separate type enums `TQW{1,2,3,4}_1` or reuse KTQ*?
- Integration into llama.cpp's convert pipeline (gguf-py)
- Interaction with existing quant types

**Timeframe:** 1-3 months after Phase 2. Bigger project than the
trick series.

---

## Phase 5 — Community / paper / hardware

**Upstream PR to `ggml-org/llama.cpp`:**
split cleanly into digestible PRs (type enums → CPU path → CUDA
path), paper-like documentation. Optional — only if there is
meaningful community demand.

**Paper:**
once Trick 17 is named and validated, draft for ICLR 2027 or
similar. Competitor benchmarks: KVQuant, Aquila, QuaRot.

**Hardware support:**
- RTX 40-series tuning (Ada architecture)
- AMD ROCm path (contingent on community interest)
- Apple Silicon MPS (Metal-shader equivalents of the Viterbi encoder)

**Timeframe:** 3-12 months, depending on paper timing.

---

## Infrastructure / ops

### Repositories
- `LL4nc33/llama-tq` — actively maintained fork with VTQ_2
- `ggml-org/llama.cpp` — upstream (periodic rebase/merge)
- `LL4nc33/oidanice-llama` — whitelabel AI platform (uses llama-tq as backend)

### Deployed servers
- primary — Qwen3.5-35B-A3B, VTQ_2 (since 2026-04-20)
- secondary — FunctionGemma 270M (tool router)

### Testing
- Local CPU round-trip in `tests/trellis-phase1/`
- PPL sweep on the test hardware (wikitext-2)
- Stability runs: bench tg1024, long generation

### LEGION
Local-only shared message board with `oidanice-distillery` for
training/deployment coordination. Never pushed to remote.

---

## Decision log (loose)

**Why 17 tricks?** German idiom for "the clever-but-simple solution
that makes the problem go away". Hard limit against featurism.

**Why trellis instead of codebook?** Paper validation: trellis beats
Lloyd-Max codebooks at the same bpw by ~0.3-0.5% PPL. Cost: more
complex encoder (Viterbi DP) vs LUT.

**Why deferred_v?** Per-token Viterbi on short ubatches = 21.7 ms
GPU-call overhead dominates. Bulk-quantize at the prefill→decode
transition = one Viterbi invocation, then a read-optimized decoder.
26× tg speedup, no quality change.

**Why no k_cache protect-sinks?** K-cache is written symmetrically
per token and behaves differently under quantization. Early
measurements show no sink dominance → not pursued.

---

## References
- [docs/plans/2026-04-20-cuda-stability-validation.md](plans/2026-04-20-cuda-stability-validation.md)
- [docs/plans/2026-04-19-deferred-v-results.md](plans/2026-04-19-deferred-v-results.md)
- [docs/plans/2026-04-19-sink-protection-results.md](plans/2026-04-19-sink-protection-results.md)
- [docs/plans/2026-04-17-trellis-v2-design.md](plans/2026-04-17-trellis-v2-design.md)
- [tests/trellis-phase1/BACKLOG.md](../tests/trellis-phase1/BACKLOG.md)
- [docs/turboquant.md](turboquant.md) — TQ v5 technical details
