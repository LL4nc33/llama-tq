# Upstream-PR Audit — TurboQuant Submission Readiness

**Datum:** 2026-04-22
**Scope:** Evaluate what's needed to submit TurboQuant as one or more PRs to `ggml-org/llama.cpp`
**TL;DR:** The phase2 branch is **not PR-ready**. Significant rebase + feature-isolation work needed first.

## Current Fork State

- Branch: `phase2`, 39 commits ahead of local `master`
- Local `master`: 196 commits ahead of `upstream/master`
- **Total: 235 commits** over upstream/master
- Diff vs upstream: **543 files changed, +32,426 / -16,052 lines**

**Far too large for a single PR.** Splitting is mandatory.

## What Lives in this Diff (Content Taxonomy)

| Category | Est. Size | PR-ready? |
|---|---|---|
| TurboQuant types (KTQ_1, VTQ_1, VTQ_2 families) | ~6,000 LOC | needs isolation |
| WebUI (OidaNice branding, chat features) | ~8,000 LOC | **out of scope** for upstream |
| Cortex-native C++ backend | ~7,000 LOC | **out of scope** for upstream |
| Python backend (oidanice/) | ~10,000 LOC | **out of scope** |
| RoPE docs/comment cleanup | ~50 LOC | orthogonal — should be own PR |
| Trellis v2 experiments (VTQ_2) | ~3,000 LOC | not PR-ready (15× TG regression) |
| ctx=400k OOM fixes | ~300 LOC | potentially upstream-suitable, orthogonal |
| Trick 1-6 experimental | ~1,500 LOC | **not PR-ready** (experimental, no gates passed) |

## PR-Candidate #1: VTQ2_1 + KTQ2_1 + KTQ_1 family (MINIMAL UPSTREAM-READY)

**Target:** Production-proven, 70 tok/s TG, -1% PPL vs f16 on Qwen3.5-35B-A3B.

**Files affected (est. 15-18 files):**
- `ggml/include/ggml.h` — 4 new enum entries (KTQ2_1, KTQ3_1, VTQ2_1, VTQ3_1)
- `ggml/src/ggml-common.h` — block structs (4 new types)
- `ggml/src/ggml.c` — type_traits entries
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU dequant wiring
- `ggml/src/ggml-cuda/turboquant.cuh` — CUDA kernel impls (NEW file)
- `ggml/src/ggml-cuda/fattn-common.cuh` — FA V-dequant helpers
- `ggml/src/ggml-cuda/fattn-vec-dispatch-ktq.cu` — FA K-dispatch (NEW)
- `ggml/src/ggml-cuda/fattn-vec-dispatch-vtq1.cu` — FA V-dispatch (NEW)
- `ggml/src/ggml-cuda/convert.cu` — dequant lookup tables
- `ggml/src/llama-kv-cache.cpp` — KV cache integration + rotation trigger
- `common/arg.cpp` — CLI parser entries
- `tests/test-backend-ops.cpp` — dequant correctness tests

**Estimated clean diff: ~3,000 LOC across 15 files.** Still large but reviewable.

**Blockers before submission:**
1. **Rebase + squash.** Current 196+39 commits contain many iterative fixes, unrelated changes, and experimental branches. Need clean linear history: 1 commit per logical unit (design doc, type enum, CPU impl, CUDA impl, FA integration, tests).
2. **Remove fork-local additions** not related to TQ: RoPE doc removal, ggml.h comment strips, any OidaNice-specific branding.
3. **Upstream-style tests.** Currently `test-backend-ops` has custom cases for KTQ/VTQ but scattered. Need clean test additions following upstream conventions.
4. **Documentation.** Currently in `docs/plans/2026-04-16-vtq-design.md` — need to polish into a proper upstream-ready description in the PR body.
5. **Per-model validation.** Upstream will ask: does it work on Llama-3.1-70B, Qwen3, Gemma, Mistral? Current validation is primarily Qwen3.5-35B-A3B.

**Estimated effort: 2-3 days** focused work after strategy nailed down.

## PR-Candidate #2: ctx=400k OOM fixes

**Scope:** Memory management fixes for long-context scenarios.
- `llama-kv-cache.cpp` size calculations
- `ggml-cuda.cu` allocator adjustments

**Est. size:** 300 LOC, 4 files
**Status:** probably ready after isolation. Orthogonal to TurboQuant. **Good warm-up PR** to establish upstream relationship.

## PR-Candidate #3: Trellis v2 (VTQ_2)

**Status: NOT READY.** Measured 15× TG regression vs VTQ_1 on Turing. Systematic elimination of mitigations complete (see `2026-04-22-e14-phase3b2-results.md`). Architectural limitation — per-sample state extraction in trellis decode vs codebook lookup.

**Would need:** Either Ampere-only guard (tensor cores help), or a codec redesign (pre-compute states in shmem). Neither is a weekend's work.

**Recommendation:** Do NOT submit VTQ_2 family upstream. Either drop from fork OR keep as experimental flag-guarded code.

## PR-Candidate #4: E11/E14 experimental kernels

**Status: NOT READY.** Negative results. Kept in-tree as reference only.

**Recommendation:** Remove from any upstream submission, keep in fork under `experimental/` or equivalent.

## Recommended Strategy

### Phase 1 (this week): Foundation
- Open a **draft issue** at ggml-org/llama.cpp proposing TurboQuant. Gauge maintainer interest BEFORE investing in PR prep.
- Polish the existing `docs/plans/2026-04-16-vtq-design.md` into a shareable design doc.
- Prepare a results-summary table (PPL + TG for Qwen/Llama/Mistral) — these are what maintainers will actually read.

### Phase 2 (following weeks): Warm-up PR
- Submit the ctx=400k OOM fix as a small independent PR. Builds reputation, tests upstream's review process.
- Duration: 1-2 weeks including review cycle.

### Phase 3 (month 2-3): TurboQuant PR #1
- Start a fresh branch `upstream-tq-core` off latest upstream/master.
- Cherry-pick / reimplement ONLY the 4 types (KTQ2_1, KTQ3_1, VTQ2_1, VTQ3_1) as atomic commits.
- Full upstream-style testing.
- Submit as single large PR OR split into: (a) types+CPU, (b) CUDA kernels, (c) FA integration, (d) KV-cache wiring, (e) CLI+tests.

### Phase 4 (month 3+): Follow-ups
- If PR #1 succeeds: consider KTQ4_1, KTQ1_1 as follow-up
- If VTQ_2 gets a redesign that works on Ampere: separate PR

## Honest Assessment

**Upstream PR is not a weekend task.** Minimum 1-2 months of prep + review cycles realistically. 

**But:** the VTQ_1 production path is already working in this fork. You can **use it locally / in OidaNice production NOW** and pursue upstreaming as a parallel, long-running effort.

**My recommendation for today:** Document this audit, then choose one of:

- **Path A (community):** Open draft issue at ggml-org/llama.cpp today to start dialogue. Low-effort, learns quickly if there's appetite.
- **Path B (internal):** Accept current fork state as production-ready for your own use, defer upstream work indefinitely, focus on VTQ_1 deployment + monitoring.
- **Path C (research):** Pivot to VTQ_2 codec redesign for Ampere — that's the real frontier, not upstream bureaucracy.

## What I Did Today

- Audited 235-commit diff vs upstream/master
- Identified 8 content categories, determined which are PR-suitable
- Defined 4 candidate PR scopes with size estimates + blockers
- Wrote this doc as planning reference

No code changes. This is the **planning artifact** for future upstream work.
