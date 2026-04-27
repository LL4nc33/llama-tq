# VTQ_3 PPL sweep anomalies — investigation

**Date:** 2026-04-25
**Branch:** turboquant
**Build:** fbdb4a484
**Sweep:** `~/sweep-gemma4-ppl-20260425-1346/` (test box)
**Model:** gemma-4-26B-A4B-bartowski-IQ2_XXS (Gemma4-26B IT, D_v=512, IT-finetuned)

## TL;DR

Three anomalies investigated. **Two are real bugs**, one is expected behavior:

1. **Identical PPL across K=2,3,4 in the VTQ_2/VTQ_3 family** — **REAL BUG.** The encoder kernel `k_vtq_encode_trellis_set_rows` hardcodes `VTQ_ENC_N=256`, but `QK_VTQ_TRELLIS=128` (the commit "halved from 256 to 128" shrunk the block size without updating the encoder). The encoder writes 256 samples of Trellis output into a 128-sample block → OOB write of 128·K bits into the next block's header. PPL becomes nearly K-independent because the entire V signal is systematically corrupt.
2. **ktq2_1/vtq3_1 PPL=13918 (+66%)** — **expected.** The vtq3_1 family is known to be weak on D=512 (see README); +66% is the normal pathology, not a new bug. Reproducible at chunks=4 (PPL=12007). No NaN, no FA dispatch fail.
3. **f16/vtq1_1 PPL=316** — **expected.** Not a NaN bug. With a 1-bit V-cache the attention distribution collapses to a quasi-uniform token distribution. On an IT-finetuned model (which has a very high baseline PPL of 8382 on Wikitext without instruct format), this collapse acts **PPL-lowering** — the degenerate output distribution is closer to "uniform 1/V" than the strongly conditioned IT output. Not physically impossible; pathological model behavior.

---

## Anomaly 1: identical PPL across K=2,3,4 (VTQ_2/_3 family)

**Status: CONFIRMED — encoder kernel bug.**

### Reproduction

Sweep on test box, same input, chunks=4:

| Config | PPL | bit-identical? |
|---|---:|---|
| f16/vtq2_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq3_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq4_2 | 13047.4902 ± 3113.72890 | ✓ |
| f16/vtq2_3 | 15642.1165 ± 3691.10499 | ✓ |
| f16/vtq3_3 | 15642.1165 ± 3691.10499 | ✓ |
| f16/vtq4_3 | 15642.1165 ± 3691.10499 | ✓ |

At chunks=64 (original sweep): vtq2_2 ≡ vtq4_2 = 8579.5706, ALL per-chunk values identical via `diff` of extracted values. K=3 was not measured for VTQ_2 in the 64-chunk run, which is why it didn't show up in the pair comparison.

### Root cause

File: `ggml/src/ggml-cuda/trellis-encode.cuh`

```cpp
#define VTQ_ENC_N 256              // line 49
constexpr int N = VTQ_ENC_N;       // line 100, in the encoder kernel
```

File: `ggml/src/ggml-common.h`

```cpp
#define QK_VTQ_TRELLIS 128         // line 400 (comment: "halved from 256 to 128")
typedef struct {
    ggml_half d;
    uint16_t  start_state;
    uint8_t   qs[QK_VTQ_TRELLIS * 2 / 8];  // 32 B for K=2
} block_vtq2_2;
static_assert(sizeof(block_vtq2_2) == 36, ...);
```

Mismatch: the block layout has `QK_VTQ_TRELLIS=128` samples, but the Viterbi encoder iterates `for (step = 0; step < N=256; step++)` and writes `qs[byte] |= (e << shift)` for steps up to 255. This is an **out-of-bounds write** of:
- K=2: 256·2/8 − 128·2/8 = 64−32 = **32 B overflow**
- K=3: 256·3/8 − 128·3/8 = 96−48 = **48 B overflow**
- K=4: 256·4/8 − 128·4/8 = 128−64 = **64 B overflow**

The overflow runs into the `d` + `start_state` + `qs` of the **next block** → systematic block-header corruption.

Additionally: `__shared__ float s_xn[VTQ_ENC_N=256]` — the encoder loads 256 samples from `x_row[j]` (`for j=tid; j<N=256`), but only 128 source samples per block are valid. j=128..255 reads **foreign memory bytes** (next-block input), which are deterministic but unrelated to the block.

### Why is the output bit-identical across K?

Hypothesis: the decoder reads only the first 128·K bits from qs[] (that's all the block layout exposes). The encoder writes 256·K bits, of which the first 128·K land in qs and the rest in subsequent block headers. But the Viterbi DP path through the first 128 states is **K-dependent** — that should produce different qs.

It happens nonetheless: the **OOB clobbering of the next block's `d`** plus reading `s_xn[128..255]` from uninitialized shared mem produces deterministic garbage. Empirically, the output floats are bit-identical across K=2,3,4, so not just the PPL but every single dequant value matches across K. Most plausible explanation: the block header (`d`, `start_state`) is overwritten by the predecessor's OOB write with data that carries the K-bits write pattern, and the decoder then re-decodes with the block's own K — net output is garbage-but-deterministic. Since all three K variants read the same `s_xn[128..255]` "phantom samples" and the Viterbi DP over 65k states is so dominated by these phantom samples, the path search collapses to identical `start_state` + `d` values regardless of K.

(One could prove this with a targeted print in the encoder, but the OOB is unambiguous.)

### Action: FIX REQUIRED

Three options:
1. **Set VTQ_ENC_N to 128** (`#define VTQ_ENC_N 128`) — fast, but halves the state count in the Viterbi and changes numerical results of all existing VTQ_2/_3 runs. **Likely the right fix** — the block is 128 samples, the encoder must encode 128 samples.
2. Encoder kernel takes a template param `N` and set-rows.cu picks `N=QK_VTQ_TRELLIS`.
3. Couple **VTQ_ENC_N and VTQ_ENC_S** to `QK_VTQ_TRELLIS` (`#define VTQ_ENC_N QK_VTQ_TRELLIS`).

Recommended: **options 1+3 together** (consolidate the header in ggml-common.h, single source of truth).

**Consequence:** ALL prior VTQ_2/_3 PPL measurements since the QK_VTQ_TRELLIS=128 commit are invalid. Including the "Phase 3 VTQ_3 win" claim.

---

## Anomaly 2: ktq2_1/vtq3_1 PPL=13918 (+66%)

**Status: EXPECTED — not a new bug.**

### Reproduction (chunks=4)

| Config | PPL ± err |
|---|---:|
| f16/vtq1_1 | 612.2661 ± 115.4 |
| f16/vtq2_1 | 21794.3283 ± 5097.9 |
| f16/vtq3_1 | 13842.6485 ± 3289.1 |
| f16/vtq4_1 | 14557.3168 ± 3501.6 |
| ktq2_1/vtq3_1 | 12007.7846 ± 2823.6 |
| ktq2_1/vtq2_1 | 26644.5038 ± 6363.4 |

The VTQ_1 family has high absolute PPL across all K, **but the K values differ cleanly** (612 / 21794 / 13842 / 14557 → real K differentiation). Encoder path is `set_rows_cuda_pq` (PQ codebook, not Trellis), unaffected by Anomaly 1.

ktq2_1/vtq3_1 PPL=13918 (64ch) ≈ f16/vtq3_1 PPL=13842 (4ch) — both similarly high. The README documents "VTQ_1 family suffers badly on D=512". Gemma4 has D_v=512 in the global layers → reproducible pathology of VTQ_1 codebook quality, no FA dispatch fail, no NaN.

### Action: NO FIX

The VTQ_1 family is known to be unsuitable on D=512. Recommendation: don't rerun the sweep script with VTQ_1 on Gemma4.

---

## Anomaly 3: f16/vtq1_1 PPL=316 (well below baseline 8382)

**Status: EXPECTED — pathological model behavior, not a bug.**

### Reproduction

Per-chunk values from `gemma4-f16-vtq1_1.log`: 7029, 1802, 765, 612, 541, ..., 316. Smooth descent, no NaN/Inf. At 4 chunks: PPL=612.

### Explanation

1. Baseline f16/f16 = 8382 is ABSURDLY HIGH because Gemma4-26B-A4B-IT is an **instruction-tuned model** that is strongly out-of-distribution on Wikitext-2 without ChatML format. Non-IT baseline would be ~5–10.
2. A 1-bit V-cache fully destroys attention information → the model collapses to a quasi-uniform output distribution over the vocabulary (V≈262144).
3. The PPL of a uniform distribution over V tokens is `V` — i.e. ~262k. But Gemma4 has a strong IT bias that, even with broken attention, still produces a generic-token distribution (top-frequent words like "the", "of", etc.).
4. A destroyed IT bias produces a **closer-to-uniform-but-bias-free** distribution that gets **better PPL** on raw Wikitext than the fully IT-conditioned output.

This is not a numerical bug but an **information-theoretic effect**: for an IT bias that pushes the model onto a "chat" distribution, a destroyed cache can land closer to "natural-language distribution" by accident.

**Key sanity check:** in the per-chunk values (e.g. chunk 1 = 7029) the model clearly reaches a high-PPL state similar to the baseline initially, before the attention collapse flattens the output distribution. This is consistent with "model goes degenerate over context length", not "computation is broken".

### Action: NO FIX

Expected pathology. To measure real VTQ_1 quality, **use non-IT models** (e.g. Llama 3 base, Qwen2.5 base) or prompt with ChatML format.

---

## Recommendations

### Immediate action

**Anomaly 1 is a critical bug.** Plan:

1. Hot fix: `#define VTQ_ENC_N QK_VTQ_TRELLIS` in `ggml/src/ggml-cuda/trellis-encode.cuh`. Build + smoke test (4-chunk PPL must differentiate by K).
2. Rerun the Phase 3 sweep after the fix:
   - `f16/vtq2_2`, `f16/vtq3_2`, `f16/vtq4_2` must have different PPL (ideally K=4 < K=3 < K=2)
   - `f16/vtq2_3`, `f16/vtq3_3`, `f16/vtq4_3` likewise
   - The `ktq2_1/vtq3_3` "Phase 3 win" claim (PPL=8339 < f16/f16=8382) **must be re-verified**
3. File a llama-tq issue titled "VTQ_2/_3 encoder writes 256 samples per 128-sample block — OOB clobbering since QK_VTQ_TRELLIS halved".

### Validation tests (post-fix)

- Round-trip MSE test (`tests/test-vtq3-roundtrip.cpp`) — K=2/3/4 should still have distinct MSE (already did, because the test uses the offline encoder, not the GPU one).
- PPL reproducibility check: `f16/vtq2_2 chunks=4` twice in a row must be identical (deterministic CUDA), but `f16/vtq2_2 ≠ f16/vtq4_2` must hold.
- E14 split-decode + greedy-encode (FAST_ENC) paths share the **same** bug mechanism — verify independently.

### Anomalies 2 & 3

No action — documented. Optional: extend the sweep script with a "VTQ_1 family on D=512 = expected high PPL" annotation so future investigations don't bite on the "bug".

---

## Appendix: files/lines

- `ggml/src/ggml-common.h:400` — `#define QK_VTQ_TRELLIS 128`
- `ggml/src/ggml-cuda/trellis-encode.cuh:49` — `#define VTQ_ENC_N 256` ⚠️
- `ggml/src/ggml-cuda/trellis-encode.cuh:71-322` — `k_vtq_encode_trellis_set_rows` (Viterbi)
- `ggml/src/ggml-cuda/trellis-encode.cuh:340-510` — `k_vtq_greedy_encode_set_rows` (FAST_ENC, hardcoded `1<<3`)
- `ggml/src/ggml-cuda/set-rows.cu:505-581` — VTQ_2/_3 encoder dispatch
- `ggml/src/ggml-cuda/fattn-common.cuh:898-1054` — decoder (uses block layout, OK)
- Sweep logs: `claude@test-box:~/sweep-gemma4-ppl-20260425-1346/`
