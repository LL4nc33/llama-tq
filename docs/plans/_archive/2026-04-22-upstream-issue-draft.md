# Draft: Upstream Issue for TurboQuant

**Platform:** https://github.com/ggml-org/llama.cpp/issues
**Type:** Feature Proposal / Discussion
**Labels suggested:** `enhancement`, `ggml`, `CUDA`, `performance`

---

## Title

Proposal: TurboQuant KV-cache quantization (2.5 bpw V with -1% PPL via PolarQuant)

## Body

Hi maintainers,

I've implemented a KV-cache quantization method called TurboQuant in a fork over the last ~6 weeks. Before investing in a proper PR split + rebase, I'd like to gauge upstream interest.

### What it is

TurboQuant applies [PolarQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) to the K/V cache:

- **RHT (Random Hadamard Transform)** rotates each block to make the per-sample distribution approximately Gaussian.
- **Lloyd-Max scalar quantizer** then uses an optimal non-uniform codebook (1-4 bit per sample).
- Separate types for K (inner FA-loop dequant) and V (graph-level rotation + codebook-only FA).

### Results (production-deployed)

Measured on Qwen3.5-35B-A3B (IQ2_XS base model), RTX 2060 sm_75, single GPU:

| K-Cache | V-Cache | bpw (K/V) | PP512 tok/s | TG128 tok/s | PPL vs f16 |
|---------|---------|-----------|:---:|:---:|:---:|
| f16 | f16 | 16/16 | 731 | 58.8 | baseline |
| q8_0 | q4_0 | 8.5/4.5 | 485 | 50.6 | +14% |
| q8_0 | **vtq2_1** | 8.5/**2.5** | 684 | **57.5** | **+1%** |
| ktq2_1 | **vtq2_1** | **3.5/2.5** | — | **66.5** | **-1%** (better!) |

Key numbers:
- **2.5 bpw V-cache** (vs q4_0's 4.5 bpw) with -1% PPL
- **3.5 bpw K-cache** with FA integration (no fp16 back-conversion in inner loop)
- Production: 400k-token context on 12 GiB VRAM, 70 tok/s TG sustained for months

### What makes it different from existing KV-quant (q4_0, q8_0)

- Uses distribution-aware quantizer (post-RHT Gaussian → Lloyd-Max) instead of min/max uniform
- V-side rotation is applied **once at graph build**, so FA inner-loop just does `codebook[idx] * scale` — a single LUT load
- FA V-dequant avoids inverse Hadamard per-sample (the expensive part)

### Status

Repo: https://github.com/LL4nc33/llama-tq (branch `master` for stable, `phase2` for experimental)

- 4 K-cache types (KTQ1/2/3/4_1) + 4 V-cache types (VTQ1/2/3/4_1) — production-validated
- Deployed continuously since 2026-03 on Qwen3.5-35B-A3B for a private service
- Design docs in `docs/plans/2026-04-16-vtq-design.md` and `docs/plans/2026-04-16-ktq-design.md`
- CUDA validated on Turing sm_75 only so far

### What's NOT ready for upstream

- Experimental Trellis v2 family (VTQ_2) — has 15× TG regression on Turing (architectural, not a bug). Wouldn't propose.
- Various unrelated fork-local changes (WebUI, Python backend, experimental tricks)

### What I'm asking

1. **Is there interest** in KV-cache quantization beyond the existing q4_0 / q8_0 path?
2. If yes: **would you prefer** one large PR (types + CPU + CUDA + FA + KV-cache) or multiple small ones (types → CPU only → CUDA → FA integration)?
3. **Any concerns** upfront — naming (`GGML_TYPE_KTQ2_1` vs alternatives)? Separate K/V enum types vs unified? RHT precomputation strategy (per-session seed vs global)?
4. **Hardware coverage** — I only have sm_75. Would you want sm_80+ / ROCm validation before a PR, or is "CUDA-only initial + follow-up PR for HIP" acceptable?

Happy to open a draft PR with just one type (e.g. VTQ2_1 standalone) as a starting point if that's easier to review.

References:
- PolarQuant paper: https://arxiv.org/abs/2504.19874
- My regression analysis docs if relevant: docs/plans/ in the fork

---

**Notes for submitter (not part of issue body):**

- Keep the tone neutral, not salesy. Upstream respects numbers over adjectives.
- The "what's NOT ready" section is CRUCIAL — shows you understand scope. Maintainers get frustrated when contributors overclaim.
- Don't mention OidaNice / commercial context — they care about the algorithm, not your business.
- If they respond with questions in the issue, answer promptly but DON'T jump into coding until there's explicit "yes, this sounds useful, please draft a PR" from a maintainer.
- If they respond "not interested" → save yourself 2 months. Cherish the clarity.
- If radio silence for 2+ weeks → ping once politely, then move on.

## Alternative shorter version (if you prefer)

If the full version feels too long, here's a 200-word variant:

---

**Title:** Feature request: TurboQuant — 2.5 bpw V-cache with ~0% PPL loss

Hi, I've implemented KV-cache quantization via [PolarQuant](https://arxiv.org/abs/2504.19874) (RHT + Lloyd-Max) in a fork. Production-deployed for ~2 months on Qwen3.5-35B-A3B.

Measured: **3.5 bpw K + 2.5 bpw V** at **-1% PPL vs f16** (better than f16, likely regularization) and **66 tok/s TG** (vs 59 baseline). 400k context in 12 GiB VRAM.

Code: https://github.com/LL4nc33/llama-tq/blob/master/docs/plans/2026-04-16-vtq-design.md

Before I invest in proper PR isolation:
1. Is this a feature upstream wants?
2. One big PR or split into stages?
3. CUDA-sm_75-only first version acceptable, or need sm_80/HIP too?

If yes I can draft a single-type PR (VTQ2_1 alone) as a starting point. If not, saves us both time 🙂

Thanks!

---

## How to submit

1. Log in to github.com as LL4nc33
2. Go to https://github.com/ggml-org/llama.cpp/issues/new
3. Choose "Feature request" or "Blank issue" template
4. Paste the short version (it's better)
5. Submit

Estimated effort: **10 minutes.** Response timeline: **1-14 days** typically.
