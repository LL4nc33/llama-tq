# Phase 5 Research Synthesis (Innovator + Researcher)

**Date:** 2026-04-26
**Sources:** 2 parallel agents (cwe:innovator + researcher), ~75 arxiv papers reviewed

## Convergent recommendations (both agents flagged)

These ideas appeared independently in both agent reports — strongest signal:

### #1 — Sink + Q-Buffer + Hybrid INT2/INT4 KV  (KITTY pattern)
- arxiv:2511.18643 (KITTY, Nov 2025) + arxiv:2510.13334 + arxiv:2502.13176
- **Pattern:** First 4 tokens FP16 (StreamingLLM sinks), last N=128 tokens
  high-bit (KTQ4_1), middle KTQ2_1/VTQ2_2.
- **Win:** Closes most quality gap to FP16 at 2-bit. Long-context PPL drift
  fix.
- **Difficulty:** 2/5 — mostly slot-write patch in `llama-kv-cache-unified.cpp`
- **Conflict:** None with KTQ. Stack-friendly.
- **Bench-gate:** RULER 64k ≥ 95% FP16, decode t/s within 1%.

### #2 — Per-Expert dynamic bit allocation (DynaExq / DyBit-MoE)
- arxiv:2511.15015 (DynaExq, Nov 2025) + arxiv:2604.06515
- **Pattern:** Router-trace tracks hot/cold experts. Hot → Q8/FP16,
  cold → KTQ2/IQ2. Async promotion/demotion.
- **Win:** +4.5% accuracy on Qwen3-80B at same memory; 2.7× throughput vs
  offload baselines. Direct fit for Qwen3.6-35B-A3B.
- **Difficulty:** 4/5 — new hot-tracker in MoE forward, quant state machine.
- **Conflict:** Orthogonal to KV quant.
- **Bench-gate:** PPL within 0.3 FP16, ≥80% uniform-KTQ2 decode t/s.

### #3 — Cross-Layer KV reuse  (XQuant)
- arxiv:2510.11236 (XQuant, EMNLP 2025)
- **Pattern:** Layer pairs share KV storage; upper layer dequants from lower.
  Effective sub-1.4 bpw on top of KTQ2_1.
- **Win:** ~30-40% additional VRAM beyond KTQ2_1 → effective KV ~2.5 bpw.
  Better than KIVI-2bit quality.
- **Difficulty:** 3/5 — orthogonal to PolarQuant. Needs layer-pairing logic
  in `llama_kv_cache_unified` + sibling-layer dequant fetch.
- **Conflict:** None — stacks cleanly with KTQ.
- **Bench-gate:** PPL within 0.5 FP16, KV-mem -30% beyond KTQ2_1.

## Single-source high-value ideas

### #4 — QTIP weight trellis  (Innovator agent)
- arxiv:2406.11235 (QTIP, ICLR'25)
- **Pattern:** Apply our existing trellis machinery (built for VTQ V-cache)
  to weight quantization. Bitshift trellis = compute-based codebook (no LUT).
  RHT shared with VTQ.
- **Win:** Q3_K-quality at Q2 size. Makes 122B single-GPU-feasible (24GB
  total fits).
- **Difficulty:** 5/5 — full new ggml type + CUDA kernel + quantizer.
- **Bench-gate:** PPL within 0.5 of Q3_K_M at 2 bpw, mmvq decode ≥ Q2_K.
- **Strategic:** Highest payoff. Natural extension of fork's trellis work.

### #5 — Expected Attention Eviction  (Both agents)
- arxiv:2510.00636 (Oct 2025)
- **Pattern:** Score KV by predicted *future* attention-mass (Gaussian Q
  prior). Evict bottom CDF when ctx > threshold. Training-free.
- **Win:** +3-4× usable context at fixed VRAM, ~0% PPL on long tasks.
- **Difficulty:** 3/5 — prefill-side scorer + slot-eviction policy.
- **Bench-gate:** RULER no regression at 4× claimed ctx.

### #6 — RocketKV 2-stage compression  (Researcher)
- arxiv:2502.14051 (Feb 2025)
- **Pattern:** Stage 1 = SnapKV++ eviction at prefill, Stage 2 = top-k
  sparse attention at decode. GQA-aware.
- **Win:** Long-context decode speedup, no quality loss. Direct fit for
  our 200K deploy on test-box:8791.
- **Difficulty:** 3/5 — new prefill scoring pass + decode-time top-k mask.
- **Bench-gate:** Decode t/s @100k ctx ≥ 1.3× current.

### #7 — KVLINC incoherence vs PolarQuant RHT  (Researcher)
- arxiv:2510.05373 (Oct 2025)
- **Pattern:** Incoherence processing for KV. **OVERLAPS heavily with our
  PolarQuant RHT.** Need to verify we're not double-doing it.
- **Action:** Read paper, compare to our RHT impl. If novel tricks
  (per-head incoherence, learned rotations) exist, port. Otherwise skip.
- **Difficulty:** 1/5 to read, 2/5 if novel parts exist.

## Skip / defer

- **ParetoQ, Tequila, Bitnet.cpp** — QAT-required, no training pipeline
- **PuzzleMoE** — bit-packed kernels hostile to sm_75 Turing
- **LogQuant** — overlaps too much with our Lloyd-Max+RHT
- **MergeMoE / PuzzleMoE expert merging** — pre-deploy step, breaks
  GGUF MoE layouts
- **GTA grouped-tied attention (#11 innovator)** — architectural change,
  needs PTQ validation we don't have infra for
- **SWAA** — needs LoRA finetune

## Approach-of-attack ranking

**Phase 5a (1-2 weeks, low risk):**
1. **Sink+Q-Buffer KV pattern** (#1) — fastest win, fixes a known long-ctx
   pathology
2. **KVLINC vs PolarQuant audit** (#7) — 2 hours of reading, may find free
   improvements

**Phase 5b (3-4 weeks, medium risk):**
3. **XQuant cross-layer KV reuse** (#3) — VRAM win, training-free, code
   on github
4. **Expected Attention eviction** (#5) — capacity-side win

**Phase 5c (research-grade, 6-8 weeks):**
5. **DynaExq per-expert bit allocation** (#2) — biggest MoE-specific win
6. **QTIP weight trellis** (#4) — biggest absolute win, highest risk

## What changes the answer to "is Phase 5 worth doing"

- All HW-level paths blocked (PCIe, spec-decode-MoE)
- All software-level paths viable. **Phase 5 = quality+capacity, not raw t/s**
- Reframe Phase 5 metric: **HellaSwag/RULER score** + **effective ctx
  at fixed VRAM**, NOT decode t/s
- Phase 4 already shipped the t/s wins. Phase 5 is "make 2-bit feel like
  4-bit" — ROI is in deploying to weaker hardware (laptop, single-GPU).

## Recommended next session

Pick **Sink+Q-Buffer (#1)** as Phase 5a Spike. Smallest patch surface,
biggest user-visible quality bump on long-ctx, no conflict with anything
shipped. If it works, follow up with XQuant.
