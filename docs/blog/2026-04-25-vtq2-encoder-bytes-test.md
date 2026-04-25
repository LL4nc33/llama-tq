# VTQ_2 K-collision: Hypothesis A (encoder bytes) refuted

**Date:** 2026-04-25
**Branch:** `turboquant`
**Test:** `tests/test-vtq2-encoding-diff.cpp` (commit `8a4be1f5a`)
**Status:** Hypothesis A REFUTED — bug lives downstream of CPU encoder.

## Context

After the encoder OOB fix (`b771f9267`) and confirmation from Agent A that
the CUDA encoder dispatch correctly calls distinct templates for K=2/3/4
(via `[VTQ_ENC] K=…` debug prints), VTQ2_2 / VTQ3_2 / VTQ4_2 *still*
produce **bit-identical PPL**:

| K config (cache-type-v) | --cache-type-k f16 | --cache-type-k ktq2_1 |
|---|---|---|
| vtq2_2 | 12171.00 | 14581.00 |
| vtq3_2 | 12171.00 | 14581.00 |
| vtq4_2 | 12171.00 | 14581.00 |

Identical to the last decimal across K. Not noise — a structural bug.

## Hypothesis A

> Even if `[VTQ_ENC] K=4` is printed, the encoder may write qs[] bytes
> that are byte-identical (or merely zero-padded / low-K-bit-only)
> across K=2/3/4. The Viterbi step might be effectively ignoring the
> high bits and only storing 2 useful bits.

If true, decoder distinctness is irrelevant — same input → same output.

## Test design

`test-vtq2-encoding-diff.cpp` calls the **CPU reference encoder**
`ggml_trellis_encode_group()` from `ggml/src/ggml-trellis.h` with the
same deterministic 128-sample Gaussian input (mt19937 seed `0xC0FFEE`)
for K = 2, 3, 4 and:

1. Hex-dumps the resulting `qs[]` byte streams.
2. Compares K=4 first 32 bytes vs K=2 (would-be padding signal).
3. Extracts the **low-2 / low-3 bit slices** from K=4's packed stream
   and compares to K=2 / K=3 streams (would-be "high bits ignored"
   signal).
4. Verifies `start_state`, `d`, and decode MSE all differ.

## Result

```
=== K=2 ===  start_state=0x955c  d=11.338468  qs_bytes=32
  8f 8c 08 4f 73 be bc c1 5c c1 1c 97 a0 e5 74 f5
  2c 5f 8d 8b c5 0a 20 6c 90 e6 40 0b 13 8b 32 bc

=== K=3 ===  start_state=0x8bd8  d=10.874896  qs_bytes=48
  7c fd 39 e5 06 67 a6 2c 8b ad ce 03 e0 c9 f3 a8
  24 d3 4d fe f7 47 43 47 cc 9d 4d 6c 22 70 48 37
  58 83 ea db 3c 18 bd c1 ba d9 bc b8 74 3c 1d b7

=== K=4 ===  start_state=0xe290  d=11.004765  qs_bytes=64
  0d 2a e8 c8 d6 4e 37 86 f8 e9 42 43 e8 e6 99 39
  ea 64 ef d0 48 af 22 3c 9c 48 64 ea af 07 60 27
  3e bb e6 15 1f b0 69 4c 57 b3 6a c4 d0 bb 6c 83
  e8 36 fe 40 ee 40 66 2e 50 d6 fd 5a e4 b3 df 52

start_state  K=2:0x955c  K=3:0x8bd8  K=4:0xe290  [differ ✓]
d            K=2:11.338  K=3:10.875  K=4:11.005  [differ ✓]
K=4[0..31] == K=2:                    no ✓
K=4[32..63] all zero:                 no ✓
low-2 bits of K=4 == K=2 bytes:       no ✓
low-3 bits of K=4 == K=3 bytes:       no ✓
low-2 bits of K=3 == K=2 bytes:       no ✓
popcount  K=2:119/256  K=3:201/384  K=4:258/512   (~50% — healthy)
decode MSE  K=2:0.060227  K=3:0.015208  K=4:0.003848  [monotone ✓]
```

Verdict: **HYPOTHESIS A REFUTED**. Every test signal is healthy:

- Distinct `start_state` per K (Viterbi found different optimal paths).
- Distinct scale `d` per K (encoder normalises with K-aware reconstruction).
- Distinct `qs[]` byte content — neither padded nor low-K-bits-only.
- Bit-population near the 50% Bernoulli expectation across all K.
- Monotone-improving roundtrip MSE: K=4 is **15× more accurate** than
  K=2 on the same input. The encoder is doing real work for the extra
  bits.

## Where the bug actually lives

CPU encoder is fine. Possibilities for the K-collision (in priority order):

1. **CUDA encoder writes wrong K** — even though `[VTQ_ENC] K=…` prints
   the right value, the CUDA encoder might be calling a kernel that
   internally hard-codes K=2 (template instantiation issue, e.g. wrong
   `<int K>` parameter value at the kernel launch site, vs. the print
   site).
2. **Decoder dispatch in FA path ignores K** — the FlashAttention V-decode
   path may pick a dequant routine keyed by `type` but then call a shared
   helper that always treats the cache as K=2. Block-size constants
   (`block_vtq2_2` vs `block_vtq3_2`) might be correct on disk but read
   with the wrong stride.
3. **Cache layout / set_rows stride collision** — if all three types end
   up using the same row-stride (e.g. all rounded up to the K=4 size),
   then K=2/K=3 reads at K=4 stride sample-shift the data into nonsense
   that *happens* to be deterministic across K. PPL=12171 (vs ~9 for a
   working cache) is consistent with "cache contains garbage that's the
   same garbage every time".

## Recommended next test (Hypothesis B)

Dump the actual on-device cache bytes after a single token through the
FA pipeline for cache-type-v=vtq2_2 vs vtq3_2 vs vtq4_2 with identical
input. If the cache bytes are bit-identical, the encoder kernel is the
culprit (Hypothesis 1 above). If the cache bytes differ but PPL is
identical, the decoder/FA dispatch is the culprit (Hypothesis 2/3).

Concretely:

- Add a one-shot `printf("%02x", ...)` over the first 64 B of the V-cache
  page after the first decode step in `cudaSetTensor` / `set_rows` for
  the V tensor.
- Run `llama-perplexity` for 1 chunk with each of vtq2_2/vtq3_2/vtq4_2
  and capture the dump.
- Compare. Identical → encoder bug; differ → decoder bug.

## Repro

```bash
cd ~/llama-tq
git checkout turboquant
cmake --build build --target test-vtq2-encoding-diff -j 8
./build/bin/test-vtq2-encoding-diff
# Exit 0 = hypothesis refuted (expected)
# Exit 2 = hypothesis confirmed (encoder bug)
```
