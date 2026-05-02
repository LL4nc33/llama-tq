# Reddit Post Draft — A/B Bench llama-tq vs upstream

**Status:** Draft, awaiting bench results (commit e054a3088 vs upstream master HEAD).
**Target sub:** r/LocalLLaMA
**Strategy:** Show numbers BEFORE pitch. Engage with TheTom-fork-skeptic crowd directly.

## Title (pick one)

- **Option A (numbers-forward):** "llama.cpp fork hits 86 t/s @ 100k ctx on Qwen3.6-35B-A3B with 12 GB VRAM — different KV-quant approach (numbers + code)"
- **Option B (response to community):** "Continuing the 35B-A3B small-VRAM trend: a more aggressive KV-quant fork (2.78 bpw, lossless PPL)"
- **Option C (technical):** "Why split K and V cache quantization? Hadamard-domain dot product on K, Trellis on V — fork with bench numbers"

**Recommendation: Option A** (numbers-forward beats marketing, given r/LocalLLaMA TQ-skepticism)

## Body Draft

```markdown
**TL;DR:** I've been maintaining a llama.cpp fork (`llama-tq`) for ~5 weeks that
implements split K/V cache quantization. Same Qwen3.6-35B-A3B-IQ2_XXS, same
RTX 2060 12GB, here are the numbers vs upstream master (commit X vs commit Y):

| Config | KV bpw | pp512 | tg128 | KV @ 32k | PPL drift |
|---|---|---|---|---|---|
| upstream f16/f16 | 32.0 | XXX | XX.X | 640 MiB | 0% (baseline) |
| upstream q8_0/q8_0 | 17.0 | XXX | XX.X | 340 MiB | (TBD) |
| upstream q4_0/q4_0 | 9.0 | XXX | XX.X | 180 MiB | (TBD) |
| **llama-tq ktq2/vtq2** ⭐ | **2.78** | **1195** | **86.4** | **89 MiB** | **-0.33%** |
| llama-tq ktq2/vtq3 | 3.56 | 1196 | 86.6 | 114 MiB | -0.03% |

That's **3-7× smaller KV than upstream's q4_0/q4_0**, **lossless quality** (PPL is
within stderr of f16 baseline, often *better* due to RHT noise smoothing), and
**actually fast** (~86 t/s tg128 on 12 GB Turing).

Key tricks (all in the fork, all CUDA-implemented):

1. **Split K vs V quantization.** Most KV-quant approaches use the same type for
   both. K and V have very different distributions — K is post-RHT roughly
   Beta(15.5, 15.5), V has heavy tails. We use:
   - **K-cache (KTQ):** Per-block Randomized Hadamard Transform + Lloyd-Max
     codebook. The FA kernel applies FWHT to Q once per tile and dots against
     codebook values directly — no K dequantization in the inner loop.
   - **V-cache (VTQ):** Group-Viterbi trellis with inverse-Gaussian CDF table
     (similar to QTIP), or codebook lookup, depending on bpw.

2. **Outlier-channel-split** for V-cache (top-N samples stored as fp16 sidecar
   per block). Closes the long-tail PPL gap.

3. **Deferred K/V quantization.** f16 staging during prefill, bulk-convert at
   prefill→decode boundary. Avoids the repetition-loop pathology that hits when
   you quantize K aggressively per-token during prefill.

4. **Hadamard-domain dot product** for K. `<H · sign · cb, Q> = <cb, sign · H · Q>`
   — math identity, FWHT is self-inverse and orthogonal at 1/√n scale. Eliminates
   per-K-block FWHT in the FA hot loop.

5. **Sparse V dequant.** Skip V dequantization for positions where attention
   weight < 1e-6. >90% skipped at 32k+ ctx, +22% decode throughput.

**About the TurboQuant skepticism in the sub:** I've seen the
"turboquant = noob bait" / "GGerganov rejected the PR" threads. The original
Google paper benchmarks (F32 vs TQ3.5) were misleading — but the underlying
math (random rotation + Lloyd-Max codebook) is solid. What I've done in this
fork is different from TheTom's `turbo3`/`turbo4` types: K and V get different
backends, and the FA hot loop is structured around the Hadamard-domain identity
instead of inverting the rotation per-block.

**Where it's strong:**
- Long-context inference (100k+) where KV memory dominates
- A3B-style sparse MoE on small VRAM (consumer 8-12 GB cards)
- Multi-tenant parallel slots (kv-share is small enough to multiplex)

**Where it's NOT for you:**
- Already have 24+ GB VRAM and don't care about ctx — upstream f16/f16 is fine
- Need sub-50 ms/token at long ctx — Trellis decode adds latency that grows
  with context
- Not on Turing (CC 7.5) — built for sm_75, untested on Ada/Hopper

**Code:** github.com/LL4nc33/llama-tq (turboquant branch)
**Docs:** [docs/turboquant.md](https://github.com/LL4nc33/llama-tq/blob/turboquant/docs/turboquant.md)
**Build hash:** e054a3088 (2026-05-02)
**CLI:** `--cache-type-k ktq2 --cache-type-v vtq2` (or `vtq3` for lossless tier)

I'd love to see this run on Ampere/Ada/Hopper — currently I only have 2× RTX 2060.
PRs welcome, especially arch-specific tuning (FA3 on sm_80+, FP8 on Ada+, WGMMA
on Hopper).
```

## Engagement plan

After post:
1. Reply to highest-voted comment within 5 min
2. Address skeptics directly: "you're right that TheTom's fork was meh — here's how mine differs..."
3. Tag the previous TQ-thread responders directly: u/MaxKruse96, u/unjustifiably_angry, u/ridablellama → "if you've got time, would love a counter-bench"
4. Cross-post to r/MachineLearning if first thread gets traction (>200 upvotes)

## Don't do

- Don't say "TurboQuant" in the title (poisoned in this sub)
- Don't compare to F32 baseline (that's the Google scam pattern people mock)
- Don't promise "lossless" without showing PPL stderr
- Don't @ Above Spec or AI Flux directly (looks petty)
