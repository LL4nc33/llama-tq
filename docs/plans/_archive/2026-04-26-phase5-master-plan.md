# Phase 5 Master Plan — Consolidated Roadmap

**Date:** 2026-04-26
**Status:** consolidated from 8 parallel research agents
**Source docs (all in this directory):**
- `2026-04-26-phase5-pie-port-spec.md` — Pie/Turnip dead-ends (PCIe-microbench)
- `2026-04-26-phase5-research-synthesis.md` — innovator + researcher synthesis
- `2026-04-26-phase5-review.md` — independent skeptical review
- `2026-04-26-kitty-sink-qbuffer-spec.md` — Sink+Q-Buffer KITTY pattern impl spec
- `2026-04-26-kitty-coderecon.md` — code-recon for KITTY patch points
- `2026-04-26-xquant-port-spec.md` — XQuant cross-layer KV reuse port spec
- `2026-04-26-kvlinc-audit.md` — KVLINC vs PolarQuant overlap audit
- `2026-04-26-expected-attention-spec.md` — Gaussian-prior eviction spec
- `2026-04-26-qtip-weight-spec.md` — QTIP weight-trellis port spec
- `2026-04-26-rocketkv-spec.md` — RocketKV 2-stage compression spec
- `2026-04-26-ruler-harness-spec.md` — RULER + LongBench eval harness
- `2026-04-26-competitive-landscape.md` — competitive position vs vLLM/SGLang/ik_llama

## Reality check (microbench)

Hardware: 2× RTX 2060 12GB asymmetric PCIe x16/x4, KVM-VM, 12 vCPUs.

| Microbench | Result | Implication |
|---|---|---|
| GPU0 PCIe-x16 H2D 256MB | **13.14 GB/s** in 20.4 ms | borderline for prefetch |
| GPU1 PCIe-x4 H2D 256MB | **1.44 GB/s** in 186.5 ms | 6.7× over 27ms decode budget |
| Decode budget @ 36 t/s on 80B | 27.7 ms/tok | hard ceiling |

**Result:** Pie-style PCIe prefetch infeasible. Must reduce KV reads, not increase bandwidth.

## Competitive position (verified)

**We are SOTA in deployed sub-3bpw KV on consumer GPU.** No other shipped engine has KV below ~3.5 bpw on real models on consumer hardware.

The 3 competing PRs/issues in flight:
- **vLLM PR #38280** — CLOSED (0.36× throughput, fundamental architecture issue)
- **SGLang PR #21617** — OPEN draft, no real-model bench, hobbyist author, unlikely to merge soon
- **ik_llama.cpp Issue #1509** — CLOSED, no code PR

**The "weeks not months" pressure was overstated.** We have months, not weeks. But upstream-llama.cpp PR Discussion #20969 is still the priority — AmesianX is being credited there as reference impl when it should be us.

## Phase-5 ranked roadmap

### Tier S: Must-do, low-risk, high-impact

**1. Upstream PR to ggml-org/llama.cpp (2-3 weeks, PRIORITY)**
- Submit KTQ2_1 + VTQ2_2 to upstream
- Target Discussion #20969
- Lead with deployment evidence: HellaSwag 83.5% on 35B-A3B, 75 t/s, 200K ctx, 24GB consumer hw
- Keep Sparse V Dequant + adaptive expert routing as proprietary moat
- **No competitor has these numbers verified.**

**2. RULER + LongBench harness (~2 dev-days)**
- See `ruler-harness-spec.md`
- Required gate for ALL Phase-5 spikes
- Without this, every patch ships on PPL + vibes
- 500 LOC wrapper, vendored RULER + LongBench, runs nightly on test-box
- 3.2h wall-clock for smoke profile

### Tier A: High-value, medium-effort

**3. XQuant cross-layer KV (~2 weeks, ~1165 LOC)**
- See `xquant-port-spec.md`
- Per reviewer: this is the actual top pick (not KITTY)
- Training-free, github code at brinenick511/XQuant
- **K-only v1: 13.6 GB → 9.16 GB KV at 200K (-32%)** → enables 290K ctx OR 3rd parallel slot
- Stack-friendly with KTQ2_1 (RHT commutes)
- VTQ trellis sharing = v2 separate spec

**4. Sink + FP16 recent-token window (~400 LOC, 2 days)**
- From KVLINC audit (#2 there)
- KITTY pattern but smaller scope
- Keep last 128 tokens FP16, rest quantized
- Solves real long-ctx pathology for free
- Naturally pairs with TQ-protect-sinks (already in code)

### Tier B: Research-grade, defer until Tier S+A done

**5. KITTY full Sink+Q-Buffer hybrid (~1000 LOC, 5 days)**
- See `kitty-sink-qbuffer-spec.md` + `kitty-coderecon.md`
- Full three-tier KV (sink/qbuffer/mid)
- Risk: ring-buffer + iSWA + deferred-quant interaction = 3 state machines on 1 buffer
- Lower priority than XQuant per reviewer

**6. Expected Attention eviction (~1225 LOC)**
- See `expected-attention-spec.md`
- Gaussian-prior eviction, training-free
- Stack interaction with VTQ2_2 needs deferred-V f16 staging hook
- Choose ONE of #5 or #6 — both touch slot allocation

**7. QTIP weight-trellis (~2345 LOC, ~6 weeks)**
- See `qtip-weight-spec.md`
- Biggest absolute payoff: 122B in 24GB @ Q3_K_M-equiv
- 70% reusable from existing trellis infra
- Risk 5/5: tail-biting Viterbi, MoE expert quality cliff, sm_75 register pressure
- **Phase 5 stretch goal**, not core

### Tier C: Skip / defer

- DynaExq per-expert bit allocation — wrong target per reviewer (PCIe class problem)
- Spec-decode on Qwen3.6-27B dense — bench was inconclusive, low priority unless 27B becomes primary deploy
- LinC adapter (KVLINC #1) — needs calibration, breaks zero-calib philosophy
- KVLINC #3-6 — overlap or contradict our design
- RocketKV — interesting but FA dispatch fix is prereq + GQA dependency + permanent eviction risk
- DASH-KV / MoA / GTA / SWAA — all from innovator report, deferred for clearer ROI evidence
- CUDA Graphs decode replay (#177 task) — research done, marginal +5-10% TG, defer

## Phase-5 Decision Tree

```
START
  │
  ▼
Build RULER harness ─── 2 days ───┐
                                  │
                                  ▼
              Capture KTQ2_1+VTQ2_2 baseline ── 1 day
                                  │
                                  ▼
            ┌─────────────────────┴─────────────────────┐
            │                                           │
            ▼                                           ▼
    Upstream PR (parallel)                  Implement XQuant
    2-3 weeks                               2 weeks
            │                                           │
            └────────────────┬──────────────────────────┘
                             │
                             ▼
                    XQuant bench-gate?
                    (PPL +0.3% / TG -5%)
                             │
                ┌────────────┴────────────┐
                │                         │
              PASS                       FAIL
                │                         │
                ▼                         ▼
        Implement FP16-recent       Tune η values, retry
        window (2 days)             OR shelf XQuant
                │
                ▼
        Bench-gate pass?
                │
        ┌───────┴────────┐
        │                │
       YES              NO
        │                │
        ▼                ▼
   Phase 5 done     Pick KITTY OR
                   Expected-Attention
                   (not both)
```

## Hard constraints (don't violate)

- **No parallel benches on test-box** (feedback rule)
- **Smoke first on Qwen3.5-0.8B** before 35B/80B/122B runs
- **Never touch live deploy on port 8791** — kill ssh-detached procs via `nvidia-smi --query-compute-apps`
- **Never commit "production"/"prod"** in docs/code
- **Sequential tasks only** for benches and quality gates
- **All specs gitignored** under docs/plans/* — local only

## Estimated total effort (Tier S + A)

- Upstream PR (parallel track): 2-3 weeks calendar (1-2 dev-weeks)
- RULER harness: 2 dev-days
- Baseline capture: 1 dev-day
- XQuant: 2 dev-weeks
- FP16-recent window: 2 dev-days

**Total active dev: ~3 weeks for the core Phase 5 outcomes (XQuant + recent-window + harness).**

If both ship clean: **1.27 bpw effective KV** (XQuant K + FP16 recent over current 2.78 baseline) → unambiguous SOTA for deployed open-source consumer-GPU inference.

## Next concrete action

Build RULER harness skeleton (Phase 1-3 of `ruler-harness-spec.md`) — this is the prereq for all Tier-S+A bench-gates. ~2 dev-days, then everything else gets a real quality measurement.

Or — if upstream PR is the higher priority — start writing the upstream PR description with deployment evidence in parallel (no implementation work, just writeup).
