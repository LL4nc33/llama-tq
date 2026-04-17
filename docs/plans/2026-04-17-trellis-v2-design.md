# KTQ/VTQ v2 — Trellis-Coded Quantization Hybrid Design

**Author:** LL4nc33
**Date:** 2026-04-17
**Status:** Design — not implemented
**Depends on:** Current KTQ/VTQ v1 (scalar Lloyd-Max), upstream `self_v_rot` infrastructure

---

## Executive Summary

Introduce a `_2` variant family for KTQ and VTQ that replaces the scalar
Lloyd-Max quantizer with **Trellis-Coded Quantization (TCQ)**. The goal is
to reach or exceed spiritbuun/buun-llama-cpp's PPL quality at matched bit
depth, while preserving the asymmetric K/V split that is this fork's
design identity.

The `_1` types stay as-is and continue to be supported as the stable
baseline. `_2` types are introduced incrementally: `vtq2_2` first
(proof-of-concept), then expanded to other bit depths and to KTQ if
successful.

### What Changes

| Type family | Quantizer | Rotation | FA dequant path |
|-------------|-----------|----------|-----------------|
| KTQ `_1` | Lloyd-Max scalar | RHT per block | Hadamard-domain dot product |
| KTQ `_2` | Bitshift trellis (Viterbi) | RHT per block | Hadamard-domain dot product + trellis decode |
| VTQ `_1` | Lloyd-Max scalar | Fixed D·H·D | `codebook[idx] · scale`, `__forceinline__` |
| VTQ `_2` | Bitshift trellis (Viterbi) | Fixed D·H·D | Trellis bitshift decode — inlinability TBD |

### Non-Goals

- Dropping `_1` types. They remain the stable deploy target.
- Full QTIP paper fidelity. Weight quantization is out of scope; this is
  KV-cache only.
- Beating buun on Q6_K weights. Target is parity at matched config.

---

## Motivation

Current `_1` types use scalar Lloyd-Max codebooks. At 2-bit and 3-bit,
this is fundamentally limited by the scalar quantization distortion
bound. spiritbuun's buun-llama-cpp uses Trellis-Coded Quantization and
achieves PPL delta `-0.05%` at 3-bit (actually below f16 baseline on
Qwen3.5-27B Q6_K). This is mathematically unreachable for any scalar
quantizer at the same bit depth.

Two open questions this design addresses:

1. Can a trellis decoder stay in the FA inner loop without killing
   VTQ's register-light advantage?
2. Is the combination "asymmetric K/V + trellis" better than either
   (asymmetric K/V + scalar) or (symmetric + trellis) alone?

Neither question is answered anywhere in the community literature. Both
TheTom and spiritbuun use symmetric K+V. This fork is the only one with
asymmetric K/V; combining it with trellis is novel ground.

---

## Background: Trellis-Coded Quantization

### Classical TCQ (source coding)

Instead of quantizing each sample independently with a codebook lookup,
TCQ quantizes a **sequence** of samples by finding the minimum-distortion
path through a trellis. Each state transition emits a code point drawn
from a larger effective codebook, but the per-step bit cost stays low
because state is context-dependent.

- Encoding: Viterbi algorithm over the trellis, `O(N · S)` per block of
  `N` samples and `S` states. Slow but runs once at cache write.
- Decoding: walk the trellis using the stored bit sequence. Traditionally
  sequential (each state depends on the previous), which is a problem on
  GPU.

### QTIP / spiritbuun adaptation

QTIP (Tseng et al., arXiv:2406.11235) introduces two key tricks that
make TCQ hardware-friendly:

1. **Bitshift trellis**: state transitions are a bit-shift-register
   update. Sliding-window decoder reads `N` bits at offset `i` to produce
   sample `i`. This is **parallel** — no sequential dependency.
2. **Compute-based codes**: code points are generated from state via a
   deterministic PRNG, not stored in a table. Codebook size is
   effectively unlimited.

spiritbuun ports this to KV-cache quantization with FWHT incoherence
processing + context-adaptive alpha scaling.

### What makes the FA inner loop hard

Our VTQ v1 dequant is:

```cuda
// __forceinline__, ~8 live registers
float val = codebook[idx] * scale;
```

This is 2 instructions: table load + multiply. No state, no branches.

A trellis decoder with bit-window needs:

```cuda
// TCQ decode at position i
uint32_t bits = read_bits(qs, i, window_bits);   // unpack packed bits
uint32_t state = seed ^ bits;                    // XOR with seed for diffusion
float val = compute_code(state) * scale;         // PRNG → float
```

The `read_bits` at arbitrary offset from a packed array, plus the
PRNG-based `compute_code`, together add roughly 10-20 instructions and
~15-25 live registers. Whether this still inlines into the FA P·V inner
loop without spilling is the open engineering question.

---

## Design

### Naming Convention

```
ktq1_1, ktq2_1, ktq3_1, ktq4_1   existing scalar Lloyd-Max (unchanged)
ktq1_2, ktq2_2, ktq3_2, ktq4_2   new: trellis variant
vtq1_1, vtq2_1, vtq3_1, vtq4_1   existing scalar Lloyd-Max (unchanged)
vtq1_2, vtq2_2, vtq3_2, vtq4_2   new: trellis variant
```

The first digit is bit-depth, the second digit is algorithm generation.
This matches llama.cpp's convention (`q4_0` vs `q4_1`, `q4_K_M` vs
`q4_K_S`).

### Block Layout (VTQ `_2` example, 2-bit)

```c
#define QK_VTQ_V2 32

// Trellis state: 8-bit shift register, 2 bits per step
typedef struct {
    ggml_half d;              // 2B: scale factor (same role as v1)
    uint8_t   qs[QK_VTQ_V2 / 4];  // 8B: 2-bit trellis steps, packed
} block_vtq2_2;
// bpw = (16 + 32·2) / 32 = 2.5 bpw  — same as v1

static_assert(sizeof(block_vtq2_2) == sizeof(ggml_half) + QK_VTQ_V2/4,
              "wrong vtq2_2 block size");
```

At 2-bit scalar vs 2-bit trellis, **bpw is identical**. The win is in
PPL quality, not bpw reduction. For `_2` to beat `_1` on bpw too, we'd
need to reduce the `d` field or amortize over longer blocks — see
"Future work" below.

### KTQ `_2` Layout (keeps sign bits)

KTQ's Hadamard-domain dot product trick requires per-block RHT sign
bits. These stay in `_2`:

```c
typedef struct {
    ggml_half d;
    uint8_t   qs[QK_KTQ / 4];    // 2-bit trellis steps
    uint8_t   sb[QK_KTQ / 8];    // RHT sign bits (unchanged from _1)
} block_ktq2_2;
```

No bpw change vs `_1`. Pure quality play for K.

### Trellis Construction

Use the bitshift trellis from QTIP (simplest for GPU decode):

- State: 8-bit shift register
- Emit: 2 or 3 bits per step (matches bit-depth)
- Code generation: `float code = g(state)` where `g` is a Gaussian PRNG
  keyed on state — eliminates codebook table

Exact `g` choice: start with QTIP's hash-based Gaussian code. Iterate
if PPL doesn't beat scalar Lloyd-Max by a meaningful margin.

### Encoder (Viterbi, cache-write path)

Standard Viterbi algorithm:

```
for each block of 32 elements (pre-rotated values):
    dp[0][state] = 0 for initial states, inf otherwise
    for i in 0..31:
        for next_state in all states:
            for prev_state in predecessors(next_state):
                cost = dp[i][prev_state] +
                       (x[i] - g(next_state))^2
                dp[i+1][next_state] = min(dp[i+1][next_state], cost)
                backtrack[i+1][next_state] = prev_state
    # backtrack to recover optimal state sequence
    # emit the 2/3 bits per step that drove each transition
```

Complexity: `O(N · S^2)` per block, where `S = 2^8 = 256` states for
the 8-bit shift register. For N=32, S=256: ~2M ops per block. Slow but
acceptable at cache-write rate (writes happen once per token per layer,
not per sample access).

CPU implementation first. GPU implementation later if needed.

### Decoder (parallel bitshift, FA-read path)

At FA time, to decode position `i`:

```cuda
// Read window of bits centered on position i
uint32_t bits = read_bits_window(qs, i, WINDOW_BITS);
// Derive state from bit window (bitshift trellis property:
// only the previous k bits matter for state i)
uint32_t state = state_from_window(bits);
// Compute code from state via PRNG (no table lookup)
float val = gaussian_hash(state) * scale;
```

**Critical question**: does `state_from_window` + `gaussian_hash`
inline? Target: stay under ~20 live registers so the FA P·V loop
doesn't spill.

If it does inline → VTQ `_2` preserves our decode-speed advantage.
If it doesn't → VTQ `_2` pays ~5-10% decode cost vs `_1`, still better
than symmetric turbo2 (-22%).

### Correctness Strategy

Build an **offline verifier** before touching CUDA:

1. CPU reference encoder (Viterbi)
2. CPU reference decoder (trellis walk)
3. Round-trip test: encode → decode, assert `MSE(x, decode(encode(x))) <
   known_scalar_MSE` on synthetic Gaussian data of the dimension we
   actually use (head_dim = 128)
4. Only after round-trip passes: port decoder to CUDA

This catches algorithm bugs before they become CUDA debugging problems.

---

## Implementation Plan

### Phase 1: CPU Reference (`vtq2_2` only) — ~1-2 weeks

- [ ] Trellis design: pick state bits, window size, code function
- [ ] CPU Viterbi encoder (`quantize_row_vtq2_2_ref`)
- [ ] CPU trellis decoder (`dequantize_row_vtq2_2`)
- [ ] Block struct + type registration in `ggml.h` / `ggml-common.h`
- [ ] CPU round-trip test: synthetic Gaussian data, compare MSE vs
      scalar Lloyd-Max baseline. **Gate**: trellis MSE ≤ 0.7 · Lloyd-Max
      MSE at matched bit depth. If not, iterate design before GPU work.

### Phase 2: CUDA Decoder (`vtq2_2`) — ~2-3 weeks

- [ ] CUDA kernel: `dequantize_V_vtq2_2` matching the VTQ v1 signature
- [ ] Register-count check via `ptxas --verbose`. Target: ≤ 24 live
      registers in FA inner loop (VTQ v1 is ~8, acceptable ceiling ~24)
- [ ] FA dispatch: register `GGML_TYPE_VTQ2_2` in `fattn.cu` for all
      K-types that work with VTQ V (f16, q4_0, q8_0)
- [ ] CUDA round-trip test (GPU dequant vs CPU dequant bit-identical)
- [ ] `llama-bench` run on all 5 reference models. **Gate**: TG128 delta
      vs `vtq2_1` is ≥ -5% (acceptable degradation), PPL delta is better
      than `vtq2_1` by at least 1 percentage point

### Phase 3: CUDA Encoder + PPL Validation — ~1 week

- [ ] CUDA Viterbi encoder (used at cache-write time) — can be a
      direct CPU port since it's not the hot path
- [ ] PPL benchmarks: `q8_0 + vtq2_2` on 5 reference models
- [ ] Compare against buun's reported numbers at matched bpw

### Phase 4: Expand — ~2-3 weeks

Contingent on Phase 2-3 success:

- [ ] `vtq3_2` (3-bit trellis V) — main competitive target vs buun
- [ ] `ktq2_2`, `ktq3_2` (trellis K with RHT sign bits preserved)
- [ ] Documentation updates, README benchmarks, charts

### Phase 5 (optional): bpw reduction — ~2 weeks

If `_2` works at matched bpw, consider variants that also reduce bpw:

- Shared `d` across multiple blocks (shared_d variant)
- Block size 128 instead of 32 (amortize any remaining overhead)
- Named `_2c` (compact) to distinguish from the quality-only `_2`

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Trellis decoder does not inline in FA | High | Phase 2 gate catches this; if it blows register budget, fall back to `__noinline__` with measured decode cost, still ship if PPL win is clear |
| PPL improvement < 1 percentage point over scalar | High | Phase 1 MSE gate is an early warning; redesign code function before GPU port |
| CUDA Viterbi too slow to be practical | Low | Encoder runs at cache-write rate, not access rate. Worst case 100us/block acceptable |
| Breaking changes to existing users | Low | `_1` types unchanged. `_2` is opt-in |
| Author burnout | Medium | 6-8 weeks is a lot. Phase 1 is 1-2 weeks, can stop there if blocker found |

---

## Success Criteria

The project succeeds if any of the following is true after Phase 2-3:

1. **Strong win**: `vtq2_2` at 5.5 bpw (q8_0 K + vtq2_2 V) achieves PPL
   delta ≤ +3% on Qwen3.5-27B Dense Q4_K_M (vs `vtq2_1`'s +5.1%)
2. **Clean win**: `vtq3_2` at 6.25 bpw matches or beats buun's
   turbo3_tcq PPL (-0.05% on their reference config)
3. **Architectural win**: any `_2` type is the first public
   implementation combining asymmetric K/V + trellis in llama.cpp,
   independent of PPL outcome

Fail criteria (stop and ship `_1` only):
- Phase 1 MSE gate fails after two design iterations
- Phase 2 register budget blown with no `__noinline__` fallback that
  keeps decode cost under -10% TG

---

## Out of Scope

- Weight quantization (QTIP's original domain). KV-cache only.
- Metal / Vulkan ports. CUDA-only like the rest of this fork.
- Replacing existing `_1` types. They stay.
- Multi-GPU optimization beyond what already works.

---

## References

- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) — Tseng et al., NeurIPS 2024. The bitshift trellis + compute-based codes design.
- [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) — first implementation of TCQ for KV-cache in llama.cpp. Symmetric K+V on RTX 3090, Q6_K weights.
- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh et al. Our v1 quantizer inspiration.
- This fork's existing design: `docs/plans/2026-04-16-vtq-design.md`
