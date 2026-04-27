# VTQ_1_3 — Research + Rejection

Date: 2026-04-25. Phase 3.5 of the post-Gemma4 roadmap. Analogous to the KTQ_3 investigation, the question here is whether a 1-bit counterpart to the VTQ_3 family (`vtq1_3`) can fill the Pareto gap between `vtq1_1` (1.06 bpw, PPL +16%) and `vtq2_3` (3.00 bpw, with tail protection).

**Result: rejected.** The outlier split does not work in the VTQ_1 family because the block size is too small and the relevant PQ code path has no Trellis header against which the outlier overhead could amortize.

---

## 1. Family architecture — what differs `_1` vs `_3`?

### VTQ_1 family (`vtq{1..4}_1`) — PolarQuant
- Block size `QK_VTQ = 32` (see `ggml/src/ggml-common.h:348`)
- Path: `set_rows_cuda_pq` in `ggml/src/ggml-cuda/set-rows.cu:466`
- Layout: `{ ggml_half d; uint8_t qs[QK_VTQ * b / 8]; }` — no Trellis header, just scale + codebook indices
- bpw = `b + 0.5` (1.5 / 2.5 / 3.5 / 4.5 bpw for b=1..4)

### VTQ_3 family (`vtq{2,3,4}_3`) — Trellis + Outlier-Channel-Split
- Block size `QK_VTQ_TRELLIS = 128`
- Path: `vtq_cuda_encode_set_rows` (Trellis encoder)
- Layout: `{ d, start_state, outlier_pos[4], outlier_val[4], qs[128 * b / 8] }`
  - Header `(d + start_state)` = 4 B, outlier block = 12 B, data block scales with `b`
- bpw = `(4 + 12 + 16·b) / 128 * 8 = (16 + 16b) / 128` ≈ `b + 1.0` for b=2,3,4 → 3.0 / 4.0 / 5.0 bpw

The outlier overhead in the VTQ_3 family is **a constant 12 B / 128 samples = 0.75 bpw**, independent of `b`. That's tolerable at b=2,3,4 because Trellis header (4 B) and data block (16·b B) amortize anyway.

---

## 2. What would `vtq1_3` cost?

We would need to integrate the outlier mechanism into the **PolarQuant path** (block_size 32) — the roadmap note was explicit that VTQ_1 is not Trellis-based and has no `start_state` field.

Possible layouts at `QK_VTQ = 32`, `b = 1`, and outlier count K:

| K outliers | Layout | Size | bpw | Comparison |
|---:|---|---:|---:|---|
| 0 | `{d, qs[4]}` | 6 B | **1.50** | = `vtq1_1` (already exists) |
| 1 | `{d, qs[4], pos[1], val[1]}` | 9 B | **2.25** | ≈ `vtq2_2` (3.0 bpw avg in code), but 1 outlier is sparse |
| 2 | `{d, qs[4], pos[2], val[2]}` | 12 B | **3.00** | = `vtq2_3`, dominated by existing type |
| 4 | `{d, qs[4], pos[4], val[4]}` | 18 B | **4.50** | > `vtq3_3` (4.0 bpw), worse |

The roadmap assumed "1.81 bpw" — that would have required outlier overhead to amortize over 128 samples (as it does for VTQ_3). At `QK_VTQ = 32` the overhead is 4× as dense. The central Pareto-gap rationale collapses.

---

## 3. Three alternatives — all rejected

### 3.1 With 1 outlier (~2.25 bpw)
Pareto-dominated by `vtq2_2` (Trellis, 2.25 bpw, ~6× better rel-MSE thanks to 2-bit + Trellis correlation). A single outlier slot per 32 samples cannot reliably catch the 1-bit-codebook tail-collapse — the probability that the "real" max-|x| sample is captured is roughly Top-1/32, but 1-bit codebooks typically fail on the top-3-to-4 samples.

### 3.2 With 4 outliers @ QK_VTQ=32 (4.50 bpw)
4 of 32 samples = 12.5% are no longer outliers, which is a complete codebook switch. At 4.50 bpw `vtq3_3` (4.00 bpw, Trellis + 4 outliers) is strictly superior.

### 3.3 New block type `QK_VTQ_PQ_LARGE = 128` for `vtq1_3`
- Layout: `{ d (2B), qs[16], pos[4], val[8] }` = 30 B / 128 = **1.875 bpw** ✓ (matches roadmap target)
- But: 1-bit PolarQuant over 128 samples with only **one** scale `d`. The standard PolarQuant assumption (RHT-rotated → near-Gaussian) holds over 32 lanes; at 128 lanes per-block variance becomes significant. The scale `d` would need to grow to avoid clipping the tail → more mid-range quantization error.
- Encoder + decoder + FA dispatch + KV-cache sizing + bench harness are **not** a drop-in extension of VTQ_1. It would be its own family, not a "_3" variant.
- Plus: without Trellis, 1-bit "_3" is conceptually weak — the biggest win of the `_3` family on VTQ_2/3/4 came from the combination of Trellis correlation + tail handling. 1-bit Trellis would essentially be greedy (too little info per step).

---

## 4. What would actually help

The real reason `vtq1_1` struggles on D=512 is not "missing outlier slots" but **single-scale per block** + **no Trellis context**. The right answer is one of the following, all outside Phase 3.5:

1. **Phase 7 (imatrix-aware calibration)** — model-specific Lloyd-Max codebook refit. Would directly improve `vtq1_1` at +0 bpw cost.
2. **Phase 5 (per-head adaptive bpw)** — high-variance heads get `vtq2_3`, low-variance heads `vtq1_1`. Achieves the same "fill the Pareto gap" effect at the system level.
3. **VTQ_1 D=512 dedicated kernel** (already in the "non-Pareto" tracker) — the problem is the kernel layout, not the block encoding.

---

## 5. Decision

**vtq1_3 will not be implemented.** Phase 3.5 in the roadmap is set to "rejected, see this research doc".

Rationale in short:
- PolarQuant family has `QK_VTQ = 32` — outlier overhead does not amortize (analogous to KTQ_3 at `QK_KTQ = 32`)
- With an honest 4 outliers the layout lands at 4.5 bpw (worse than `vtq3_3`)
- The larger-block alternative (1.875 bpw) leaves the VTQ_1 family and becomes 1-bit PolarQuant over 128 samples — its own research question, not a Phase 3.5 extension
- The 1.06 → 3.00 bpw Pareto gap is better addressed by Phase 5 (adaptive bpw) and Phase 7 (calibration)

Pattern matches the KTQ_3 rejection (`docs/plans/2026-04-25-ktq3-research.md`): small block sizes + outlier overhead does not work. The "_3" generation is restricted to Trellis block sizes.
