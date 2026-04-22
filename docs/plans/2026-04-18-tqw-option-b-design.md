# TQW (TurboQuant Weights) — Option B Design

**Date**: 2026-04-18
**Branch**: `tqw-option-b` (off trellis-v2-phase1 HEAD)
**Target**: RTX 2060 sm_75, CUDA
**Status**: Phase 1 (design + Python round-trip only, no C/C++/CUDA yet)

---

## 1. Context & why Option A failed

KTQ{2,3,4}_1 store weights in the **Hadamard domain** (post-RHT). The flash-attention kernel works because it applies FWHT to Q at dequant time — a one-sided rotation trick. `mul_mat` (the hot path for weight tensors) has no such trick: activations are in the physical domain and would need an inverse-RHT applied to the dequant output before the dot product.

**Option A** tried to alias KTQ3_1 as a weight type by reinterpreting blocks. It failed because mul_mat kernels (mmvq, mmq) would multiply Hadamard-domain weights against physical-domain activations, producing garbage.

**Option B** (this doc): introduce a dedicated **TQW** quant type whose dequant kernel bakes the inverse-RHT in. Output of dequant is physical-domain values, mul_mat works unchanged.

---

## 2. Block layout

### Key choice: regenerate signs vs. store them?

The 4-byte `sb[4]` per 32-element block in KTQ burns 1 bpw — unacceptable for weights (we want 3 bpw, not 4). At **dequant time during mul_mat**, signs must be applied to the inverse-FWHT output (per-element ±1 multiply). Two options:

- **(a) Store sb**: +1 bpw. Simpler kernel, no Philox at dequant.
- **(b) Regenerate from `kktq_derive_seed(block_index)` via `ktq_philox_6r`**: zero extra bytes, but each dequant call computes 32 Philox rounds per block.

On sm_75, 32 × 6-round Philox = ~192 imul/xor ops per block, and we can parallelize across 32 warp lanes (one lane = one element). This is cheap compared to the FWHT itself. **Choose (b): regenerate signs from seed.**

### Layout (QK=32, matches KTQ infrastructure):

```c
#define QK_TQW 32

typedef struct {
    ggml_half d;                  // 2 bytes: fp16 per-block norm scale
    uint8_t   qs[QK_TQW * 3 / 8]; // 12 bytes: 3-bit Lloyd-Max indices (8 centroids)
} block_tqw3_0;  // total: 14 bytes / 32 elems = 3.5 bpw
```

**Byte budget**:
- K=2 (2-bit, 4 centroids): `2 + 8 = 10 B / 32 = 2.5 bpw` → `block_tqw2_0`
- K=3 (3-bit, 8 centroids): `2 + 12 = 14 B / 32 = 3.5 bpw` → `block_tqw3_0`
- K=4 (4-bit, 16 centroids): `2 + 16 = 18 B / 32 = 4.5 bpw` → `block_tqw4_0`

TQW3_0 at 3.5 bpw slots between Q3_K_M (3.44) and Q3_K_L (3.75). The "3.0-3.3 bpw" target from the brief is not reachable at QK=32 without sacrificing the fp16 scale (would need QK=64 → `4 + 24 = 28 B / 64 = 3.5 bpw` still). True 3 bpw requires QK=128 or shared scales. **Proposed: target 3.5 bpw for initial sprint, consider QK=64 super-blocks in a follow-up.**

---

## 3. Quantize algorithm (CPU, ref only)

Per block i of 32 weights `x[0..31]`:

1. `norm = ||x||_2`; store `d = fp16(norm)`; `x_hat = x / norm`.
2. `seed = kktq_derive_seed(i)` (FNV-1a over block index).
3. `y = RHT_forward(x_hat, seed)` — apply signs then normalized FWHT (√(1/32)).
4. For each j in 0..31: `qs[j] = argmin_k |y[j] - PQ_CODEBOOK_3BIT[k]*cb_scale|` where `cb_scale = 1/√32`.
5. Pack 3-bit `qs[32]` into `qs[12]` bytes.
6. **Norm correction** (borrowed from KTQ2_1): recon → inverse-RHT → compute recon_norm; replace `d = fp16(norm / recon_norm)`. Recovers ~1.2% PPL per KTQ v5 data.

---

## 4. Dequant algorithm (CUDA, hot path)

Per block, one warp (32 threads, lane = element index j):

```
lane j:
  idx   = read 3-bit qs[j]            // shared-mem unpacked
  y_j   = PQ_CODEBOOK_3BIT[idx] * cb_scale
  // warp-parallel FWHT (5 butterfly stages via __shfl_xor_sync)
  for stage in 0..4:
      partner = j XOR (1 << stage)
      v = __shfl_xor_sync(0xFFFFFFFF, y_j, 1 << stage)
      y_j = (j < partner) ? (y_j + v) : (v - y_j)  // butterfly
  y_j *= 1/√32
  // apply sign — regenerated, not stored
  sign = (ktq_philox_6r(j, seed) & 1) ? +1.0f : -1.0f
  x_j  = y_j * sign * d
```

Matches `kktq_rht_inverse` in `ggml-quants.c:5566` but replaces byte-loop with `__shfl_xor_sync`.

---

## 5. GGUF metadata

- **No per-tensor seed needed**: `kktq_derive_seed(block_index)` is deterministic, matches across quantize and dequant. Same scheme KTQ already uses.
- New GGUFQuantizationType enum values: `GGML_TYPE_TQW2_0=42`, `TQW3_0=43`, `TQW4_0=44` (avoid conflict with KTQ range).
- `ggml_quantize_chunk` dispatch entries in `ggml-quants.c`.
- `ftype` values in `llama.h` for `--quantize` CLI: `LLAMA_FTYPE_MOSTLY_TQW3_0`, etc.

---

## 6. CUDA dispatch path

**Preferred**: register `vec_dot_tqw3_0_q8_1` in `ggml-cuda/mmvq.cu` for batch-1 decode (the common case). Dequant inside the warp, dot-product with Q8_1 activations in registers.

**Fallback for prefill / large batches**: `dequantize_row_tqw3_0` kernel → produces fp16 in scratch → calls existing fp16 cuBLAS mul_mat. Higher memory traffic but guaranteed correctness.

Plan: implement both. mmvq first (decode performance matters more for single-user chat).

**Files** (CUDA sprint, Phase 2):
- `ggml/src/ggml-cuda/vecdotq.cuh` — `vec_dot_tqw{2,3,4}_0_q8_1` templates
- `ggml/src/ggml-cuda/mmvq.cu` — dispatch entries
- `ggml/src/ggml-cuda/dequantize.cuh` — dequant kernels for fallback path
- `ggml/src/ggml-cuda/convert.cu` — fp16/fp32 conversion registration
- `ggml/src/ggml-common.h` — block structs (+ ~15 LOC)
- `ggml/src/ggml-quants.c` — CPU quant/dequant (+ ~200 LOC, mostly copy-adapt from ktq3_1)
- `ggml/src/ggml.c` — type traits table (+ ~10 LOC)
- `src/llama-quant.cpp` — ftype routing (+ ~15 LOC)
- `common/arg.cpp` — CLI exposure
- **Do NOT touch**: `ggml-cuda/trellis-encode.cuh`, `trellis.cuh`, `fattn-*.cuh`, `llama-kv-cache.*` (V-cache agents own these on trellis-v2-phase1).

**Estimated LOC**: ~600 lines (C + CUDA), dominated by CUDA kernel templates for 3 K-values.

---

## 7. Risks & open questions

1. **3.5 bpw, not 3.0**: target spec asked 3.0-3.3. True 3.0 needs super-blocks (QK=64 with shared scale) — additional complexity. Recommend accepting 3.5 for v1.
2. **Philox-at-dequant cost**: unprofiled. If it turns out to dominate, fall back to sb[4] storage (+ 1 bpw → 4.5 bpw, equals KTQ3_1).
3. **Lloyd codebook fit**: current KTQ codebooks are fit on unit-Gaussian assumption. Weights may differ. Initial pass uses shipped PQ_CODEBOOK_3BIT; follow-up may fit per-tensor-class codebook.
4. **Warp-shuffled FWHT correctness** at stage 4 (across lanes 0-15 ↔ 16-31): Must verify `__shfl_xor_sync` mask ordering matches butterfly.
5. **Row-size constraint**: tensor dims must be %32 == 0. All modern model dims are, but double-check embedding tables.

---

## 8. Go/no-go gate

**Phase 1 deliverable**: Python round-trip test (`tests/trellis-phase1/tqw2_validation.py` `--roundtrip` mode):
- quantize(weight) → bytes-equivalent representation → dequant → reconstruction
- MSE(recon - quant_dequant_path_reference) < 1e-6 (float precision).
- If pass → **GREEN**, proceed to CUDA sprint.
- If fail → block layout or algorithm flawed; reconsider.

**Note**: the 1e-6 target is for *round-trip self-consistency* (quantize-then-dequant reconstructs the stored codebook points + signs correctly), NOT for reconstruction error against the original weight (that's bounded by 3.5 bpw Lloyd-Max error, typically 1e-3 to 1e-4).
