# Post-Entwurf für llama.cpp #20969

Dieser Entwurf ist für einen Comment-Post in dem TurboQuant Discussion Thread.
Ton: ehrlich, technisch präzise, keine Marketing-Sprache, klare Limits.

---

## Entwurf: Kurzversion (empfohlen)

**Title:** `llama-tq: asymmetric K/V split with register-light V-dequant (CUDA)`

Sharing another fork for the discussion: **[llama-tq](https://github.com/LL4nc33/llama-tq)** — started from TurboQuant but ended up with a few different design choices. Posting in case the approach is useful to others.

**Core idea:** split K-cache and V-cache into two different types with different dequant paths.

- **KTQ** (K-cache): RHT + Lloyd-Max, similar to TheTom's approach. FA uses the Hadamard-domain dot product trick (FWHT on Q, not inverse-FWHT on K).
- **VTQ** (V-cache): codebook-only quantization, no per-block FWHT, no sign bits. V is pre-rotated once at graph-level via `self_v_rot` with a fixed D·H·D rotation. The FA V-dequant reduces to `codebook[idx] * scale` — `__forceinline__`, ~8 registers. Inverse rotation is a single post-FA matmul.

**Why split K and V:** the 32-element FWHT butterfly that KTQ needs inside the FA V-accumulation path pushed the kernel over the 255-register/thread limit on CC 7.5 in my tests, producing LMEM spills and (in one case) FA accumulator corruption. Moving the V-rotation out of the hot loop sidesteps that.

**Measured on 2x RTX 2060 12GB (CC 7.5), Flash Attention, -ngl 99:**

| Config | PP512 | TG128 | PPL Δ (Qwen3.5-27B Q4_K_M) |
|--------|:---:|:---:|:---:|
| f16/f16 | 318 | 14.6 | baseline |
| q8_0/vtq3_1 (6.25 bpw) | 288 | 14.4 | +0.6% |
| q8_0/vtq2_1 (5.5 bpw)  | 297 | 14.5 | +5.1% |
| q8_0/q4_0 (6.25 bpw)   | 214 | 12.8 | +0.7% |

VTQ V-cache (2.5 bpw) consistently runs at ~-1 to -3% TG vs f16, while q4_0 V-cache is at -12 to -22%. PP512 is ~30-50% higher with VTQ than with q4_0 in the same role.

**Explicit trade-offs:**
- CUDA only (no Metal, no Vulkan). TheTom's fork is the right answer for those.
- PPL at 3-bit is worse than TheTom's turbo3 on MoE (+1.0-2.5% vs +1.06%) and clearly worse than spiritbuun's TCQ at 3-bit.
- KTQ+VTQ at low bits (e.g. ktq2_1+vtq2_1) shows super-additive PPL degradation in my tests — I recommend `q8_0` or `q4_0` for K when pairing with VTQ for V. Root cause not fully characterized yet.
- PPL benchmarks are at 3 wikitext-2 chunks (noisy); a proper 64+ chunk rerun is on my list.

**Deviations from the TurboQuant paper** (documented in README):
- FWHT + Philox signs instead of QR rotation (cheaper, approximate decorrelation)
- Fixed D·H·D rotation for V (own design, not from paper) to enable position-independent pre-rotation
- No QJL residual (dropped in v5, similar to TheTom/Aaryan-Kapoor)
- Lloyd-Max codebooks fit to a Laplace(0,1) prior for the 1-2 bit VTQ cases, since the fixed rotation gives slightly heavier tails than per-block RHT

**Where this might be useful:** Consumer NVIDIA hardware (RTX 2060-era and up) where aggressive model quants (IQ2_XS, Q4_K_M) are paired with long contexts and decode throughput matters more than squeezing the last percent of PPL. Dense models > MoE in my PPL measurements so far.

**Where it's not useful:** Apple Silicon (use TheTom), max-quality 3-bit (use spiritbuun's TCQ), production-stability (nothing here is merged upstream).

Happy to rerun specific configs, answer questions, or contribute measurements if helpful.

---

## Entwurf: Falls du noch kürzer willst (Minimal-Version)

**Title:** `Another fork: llama-tq (asymmetric K/V, CUDA)`

Sharing another fork in case it's useful: **[llama-tq](https://github.com/LL4nc33/llama-tq)**.

Inspired by TurboQuant but diverges significantly:
- **Asymmetric K/V types** (KTQ vs VTQ) — different dequant paths
- **KTQ** (K-cache): RHT + Lloyd-Max, FA uses FWHT-on-Q dot product trick
- **VTQ** (V-cache): codebook-only, no per-block FWHT, pre-rotated via `self_v_rot` at graph level. FA V-dequant is `codebook[idx] * scale`, `__forceinline__`

**Motivation for the split:** KTQ's 32-element FWHT inside the FA V-accumulation pushed my kernels over the register limit (CC 7.5, 255/thread). Moving the V-rotation out of the hot loop sidesteps LMEM spills.

**Trade-offs I hit:**
- CUDA only
- PPL at 3-bit worse than TheTom/spiritbuun on MoE (+1.0-2.5%)
- KTQ+VTQ combo at low bits shows super-additive PPL degradation; pair VTQ with q8_0/q4_0 for K instead
- 3-chunk PPL is noisy; 64+ chunk rerun pending

**What works well:** Decode overhead of -1 to -3% TG128 for 2.5 bpw V-cache on Consumer NVIDIA (2x RTX 2060), tested on Qwen3.5/3.6 35B-A3B and Qwen3.5-27B Dense.

Full benchmarks + explicit paper deviations in the README. Happy to contribute measurements.

---

## Notes zum Posting

1. **Nicht im Haupt-Thread-Eingang antworten** — das wirkt wie Plug. Stattdessen:
   - Als Reply auf einen spezifischen Kommentar wo K/V Asymmetrie oder Decode-Performance diskutiert wird
   - Oder als eigener Comment mit "Adding another data point:"

2. **Was tun nach dem Post:**
   - Auf Fragen reagieren (Community schätzt Responsiveness)
   - Wenn jemand nach bestimmten Benchmarks fragt: einfach laufen lassen und posten
   - Wenn jemand einen Bug findet: danken, fixen, reporten

3. **Was NICHT tun:**
   - Claims wie "best" / "fastest" / "breakthrough"
   - Deine Zahlen mit TheTom's direkt vergleichen ohne Kontext (unterschiedliche HW)
   - Auf Kritik defensiv reagieren
   - Promise für Metal/Vulkan wenn du keine Roadmap dafür hast

4. **Kontext-Anker:** Der Thread hat Teilnehmer die direkt mit den Autoren kommunizieren könnten. Ehrliche Abweichungen vom Paper sind besser als Paper-Name Dropping.
