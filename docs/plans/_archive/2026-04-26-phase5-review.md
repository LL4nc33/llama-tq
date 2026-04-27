# Phase 5 Synthesis — Critical Review

**Date:** 2026-04-26
**Reviewer:** Claude (ask-mode, independent of innovator/researcher agents)
**Inputs reviewed:**
- `docs/plans/2026-04-26-phase5-research-synthesis.md`
- `docs/plans/2026-04-26-phase5-pie-port-spec.md` (microbench + decisions)
- `docs/bench/LIVE_NUMBERS.md` (Phase 4 deploy state)

---

## TL;DR Verdict

**Conditional agree, with reordering.** The reframe ("Phase 5 = quality+capacity, not raw t/s") is correct and well-supported by the Pie/Turnip microbench reality-check — the hardware genuinely forecloses the t/s axis. However, the synthesis ranks **Sink+Q-Buffer (#1)** above **XQuant (#3)** primarily on patch-size, not on user-visible payoff, and it under-weights the operational cost of stacking three independent KV-cache modifications on top of the already-novel KTQ/VTQ split. The top-3 picks are defensible individually but should not all ship — pick **one** capacity win (XQuant *or* Expected-Attention eviction) and **one** quality win (Sink+Q-Buffer), then stop. DynaExq is correctly placed in Tier-C but is over-sold; it solves a problem (MoE quality at fixed mem) we already solve adequately with KTQ2_1+VTQ2_2 at 83.5% HellaSwag on 35B-A3B. QTIP is the only idea here that would *change the product*; everything else is incremental.

---

## Challenges to the top-3 ranking

### Challenge 1: Sink+Q-Buffer is ranked #1 on the wrong axis

The synthesis ranks it #1 because it has the smallest patch surface (2/5 difficulty) and "fixes a known long-ctx pathology." But:

- **The pathology is not actually demonstrated on our deploy.** LIVE_NUMBERS shows Qwen3.6-35B-A3B at HellaSwag 83.5% with our current KTQ2_1+VTQ2_2. That's *above* Llama-3-8B-Q4 (80%) and Mistral-7B-Q4 (82%). We have no measured PPL drift at long-ctx for our specific KTQ — only generic 2-bit-KV-cache literature. **The bench-gate "RULER 64k ≥95% FP16" assumes a problem we haven't observed.**
- **Three regions × two bit-widths in one cache** is a real complexity tax. Slot-allocation, eviction interplay with quant-zone boundaries, and FA dispatch table proliferation. The "2/5 difficulty" is optimistic — the recent llama-tq history (SEGFAULT bug from missing FA-dispatch entries on TQ types per `project_on_llama_tq_bugs.md`) shows that "small patches in the cache layer" routinely surface dispatch-table bugs.
- **Sinks also fight token-eviction policies.** If we later add Expected-Attention eviction (#5), sink-protection logic must compose — un-evictable sink slots + evictable middle + protected Q-buffer = three policies that need joint testing.

Verdict: **Real win, but de-risk by measuring our own long-ctx PPL drift first.** A 1-day measurement spike (compare KTQ2_1 vs FP16 KV at 32k/64k/128k on RULER) determines whether Sink+Q-Buffer is solving a real problem or a paper problem.

### Challenge 2: XQuant cross-layer reuse is the actual top pick

If Phase 5's reframed metric is **effective ctx at fixed VRAM**, XQuant gives that directly: −30% KV mem on top of KTQ2_1, training-free, code on github, no quality regression at 0.5 PPL. That is a far more product-shaping win than fixing a long-ctx PPL drift that may not exist on our stack. The synthesis penalizes it for being 3/5 difficulty, but difficulty is a one-time cost; effective-ctx-at-fixed-VRAM is permanent product surface area.

XQuant also stacks cleanly with PolarQuant (no rotation conflict — XQuant operates on stored layer-pair tensors, PolarQuant operates within a single layer's RHT basis). **It should be #1.**

### Challenge 3: DynaExq is the wrong target for our deploy

The synthesis pitches +4.5% accuracy on Qwen3-80B and "direct fit for Qwen3.6-35B-A3B." But:

- **Our 35B is already at 83.5% HellaSwag.** Qwen3.5-7B-Q4 is at ~80%. We are not in the regime where +4.5% accuracy from per-expert bit promotion changes anyone's perception of the product.
- **80B is currently at +18.5% Phase 4 TG (tg128 ~36 t/s).** Spending 6-8 weeks to do hot/cold expert tracking, async promotion, quant state machine — and the win is "≥80% of uniform-KTQ2 decode t/s" (i.e. we *lose* up to 20% TG to *maybe* gain 0.3 PPL). This is a regression-shaped win.
- **Hot-expert tracking on PCIe-x4 GPU1** has the same cost-class problem that killed Pie's expert-prefetch (1.44 GB/s, 186ms per 256MB expert). Promoting a "hot" expert from KTQ2 → Q8 means re-quantizing on-device or reloading from disk; either path eats the win.

Verdict: **Downgrade DynaExq to "skip" or at most a research-track investigation.** The fork's MoE story is already differentiated by adaptive layer-split — that's the right level for our hardware.

---

## Missing risks the synthesis didn't flag

### Risk 1: Stacking complexity in `llama-kv-cache-unified.cpp`

If we ship Sink+Q-Buffer (#1) AND XQuant (#3) AND Expected-Attention (#5), we have **three independent modifications** to the unified KV cache, each with their own slot-policy invariants. The `feedback_clean_code.md` and `feedback_readable_code.md` guidance from project memory ("sauberer Code, keine Redundanz, Code muss für Laien intuitiv lesbar sein") argues against this. The synthesis treats these as orthogonal — they're not, they all touch the slot-allocation path.

**Mitigation:** Pick at most two KV-cache mods and gate them behind a single feature flag. Ship one, measure, ship the other.

### Risk 2: FA-dispatch table debt

The fork already carries 8 KTQ/VTQ types (`project_vtq_implementation.md`). Adding XQuant layer-pair logic, sink-zone logic, and eviction logic all interact with FlashAttention dispatch. We have prior art on this going wrong (SEGFAULT bug in `fattn.cu` from missing dispatch entries). Each new KV mod is a multiplier on the dispatch-table maintenance surface, not an additive cost.

### Risk 3: KVLINC overlap audit may invalidate prior work

The synthesis lists KVLINC (#7) as "2 hours of reading, may find free improvements." But the inverse risk is also real: KVLINC may demonstrate that our PolarQuant RHT is sub-optimal (e.g. fixed Hadamard vs. learned per-head rotation), and the response would be a non-trivial re-quant of all shipped KTQ checkpoints. **The audit can produce a "we should redo Phase 19.5" outcome, not just "free improvements."** Worth doing first precisely because the downside is informational not implementation.

### Risk 4: Quality benchmark coverage is thin

LIVE_NUMBERS has HellaSwag-200 for two models (27B, 35B). Phase 5's reframed metric is "quality+capacity," but we lack RULER, LongBench, or any long-ctx eval baseline. Every Phase 5 spike will need to **build its own bench harness** before it can prove a win. This is unbudgeted work in the current plan.

---

## Recommended ordering

**Pre-spike (1 day, must happen first):**
- **0a.** Measure RULER / long-ctx PPL drift on our actual KTQ2_1+VTQ2_2 stack at 8k/32k/64k/128k against FP16-KV reference. Determines whether Sink+Q-Buffer solves a real problem.
- **0b.** Read KVLINC paper (#7), compare to our PolarQuant impl. 2-hour spike. Outcome is informational.

**Phase 5a (1-2 weeks, ship if pre-spike justifies):**
- **1.** **XQuant cross-layer KV reuse** (was #3) — promoted to top because it directly delivers the reframed metric (capacity at fixed VRAM), is training-free, has reference code, and stacks cleanly. Bench-gate: KV-mem −30% beyond KTQ2_1 at PPL within 0.5 FP16.
- **2.** **Sink+Q-Buffer** (was #1) — only if 0a shows measurable long-ctx drift on our stack. Otherwise defer.

**Phase 5b (2-3 weeks, optional capacity win):**
- **3.** **Expected-Attention eviction** (#5) — the strongest capacity multiplier (3-4× usable ctx). Higher value than Sink+Q-Buffer if we have to choose. Conflict: needs joint design with Sink+Q-Buffer if both ship.

**Phase 5c (research-grade, ship-or-shelve):**
- **4.** **QTIP weight trellis** (#4) — *the* product-changing idea on this list. Reuses our existing trellis machinery from VTQ. Makes 122B single-GPU-feasible. 5/5 difficulty is honest, but the payoff is categorical, not incremental. Worth the 6-8 weeks **if** Phase 5a/b ships clean wins and the team has appetite.

**Demoted from synthesis:**
- **DynaExq** (was #2) — defer or skip. Wrong target for our quality regime; promotion-cost vs. PCIe-x4 likely cancels the win.
- **RocketKV** (was #6) — overlaps with Expected-Attention; pick one.

**Resurrect from skip-list:**
- Nothing. The skip-list rationale (QAT infra, sm_75 hostility, GGUF MoE layout) is correct.

**Downgrade from recommended:**
- See DynaExq above.

---

## Is Phase 5 worth doing at all?

**Honest assessment: yes, but only the narrow version.**

The fork is **not** feature-complete in a meaningful sense — XQuant alone would deliver a real capacity win that the current deploy lacks, and QTIP would unlock a 122B single-GPU path that currently doesn't exist. Both are quality/capacity moves, both are defensible, both are differentiated.

But the broad Phase 5 as synthesized — six ideas across three tiers, ~3 months of work — is over-scoped for the actual product gap. The reframe to quality+capacity is correct, but the *implication* of the reframe should be **fewer spikes, not more**. Phase 4 shipped the t/s wins (+18.5% on 80B) and that's the headline. Phase 5 should add **one capacity win (XQuant)** and **maybe one quality win (Sink+Q-Buffer, contingent on 0a measurement)**, then stop and let the fork stabilize.

If the team has appetite for one ambitious bet after that, **QTIP is the only Tier-C idea worth the risk**. DynaExq is not.

The "declare feature-complete" option is also legitimate. The current deploy (35B@83.5% HellaSwag, 36 t/s on 80B, 200K ctx, KTQ+VTQ+OMP_active+layer-split) is a coherent product. Shipping XQuant adds capacity; shipping nothing more leaves a clean fork. Both are honest endings — what is *not* honest is shipping all six ideas in the synthesis and calling it "Phase 5."

---

## Specific deliverables this review recommends

1. **Reorder** the synthesis's "Approach-of-attack ranking" so XQuant precedes Sink+Q-Buffer, and Sink+Q-Buffer is gated on a 1-day RULER measurement.
2. **Add** a pre-spike (0a/0b) measurement step before any Phase 5 implementation.
3. **Demote** DynaExq from #2 to "defer/skip" with rationale (quality regime + PCIe-x4 promotion cost).
4. **Cap** Phase 5 scope at "XQuant + at most one of {Sink+Q-Buffer, Expected-Attention}" before deciding on QTIP.
5. **Document** the FA-dispatch-table maintenance cost as a stacking risk.
