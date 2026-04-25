# vtq_state_at audit — Hypothesis C falsified

Date: 2026-04-25
Branch: turboquant
Bug under investigation: vtq2_2 / vtq3_2 / vtq4_2 produce identical PPL.

## Hypothesis C

`vtq_state_at<K>` (fattn-common.cuh:897-931) and/or the encoder/decoder
bit-stream layout for K=2,3,4 contain a format-mismatch or out-of-bounds
read that collapses different K values to the same state sequence.

## Method

1. Read `vtq_state_at` (fattn-common.cuh:880-970) carefully.
2. Read encoder qs-write sequence in `trellis-encode.cuh:300-320` (Viterbi)
   and `:494-509` (beam) — both write `bits = (state(step+1) >> kshift) & Kmask`
   at qs bit-position `step * K`.
3. Verified block sizes (`ggml-common.h:402-421`):
   - block_vtq2_2.qs[32] (128*2/8)
   - block_vtq3_2.qs[48] (128*3/8)
   - block_vtq4_2.qs[64] (128*4/8)
4. Verified `vtq_state_at` byte indexes for the highest call (`i = 128` from
   `il+l+1` with il+l=127): for K=2 stream_bit=256, qs_bit=240, byte=30,
   reads qs[30..32] — the last byte (32) is one past the array end (qs[0..31]).
   **WAIT**: actually for K=2 N=128, the last *needed* state is for sample
   il+l=127, called with `i = 128`, stream_bit = 128*2 = 256, qs_bit = 240,
   byte = 30. Reads qs[30], qs[31], qs[32]. qs has size 32 → qs[32] is OOB.
   Same OOB-by-one for K=3 (qs_bit=368, byte=46, reads up to qs[48]) and
   K=4 (qs_bit=496, byte=62, reads up to qs[64]).

   **However:** the high byte b2 is shifted by `>> shift` with shift = qs_bit & 7.
   For K=2, qs_bit=240 → shift=0; we right-shift then mask with 0xFFFF, which
   discards bytes ≥ 2 entirely. So b2's contribution is masked out. The OOB
   read is benign (and inside the trellis pool padding). Not the bug.

5. Wrote a CPU mini-test (`/tmp/test_state_at.c`) that:
   - Encodes a synthetic edge sequence with the same `qs[byte] |= bits<<shift`
     write pattern as the GPU encoder (Viterbi & beam paths agree).
   - Walks the shift register `state = ((state>>K) | (e << kshift)) & 0xFFFF`
     for i=1..128 → reference state(i).
   - Calls a verbatim copy of `vtq_state_at` for i=1..128.
   - Compares for K=2, 3, 4.

## Result

```
K=2: 0 mismatches over 128 samples
K=3: 0 mismatches over 128 samples
K=4: 0 mismatches over 128 samples
```

For all three K, `vtq_state_at` reproduces the reference shift-register state
exactly. The qs streams are visibly different across K (different bytes).
The encoder write order matches the decoder bit-window read order.

## Conclusion

**Hypothesis C is falsified.** `vtq_state_at` and the qs bit-layout are
correct and K-distinguishing. The decoder is not the source of the
identical-PPL bug.

## Remaining live hypotheses (other agents)

- A: encoder dispatch — confirmed OK by Agent A.
- B: type plumbing in `ggml_internal_get_type_traits` /
  `ggml_cuda_op_set_rows` may misroute VTQ3_2/VTQ4_2 → VTQ2_2 at a higher
  layer (e.g. quantize-utility, llama-kv-cache.cpp cache-type parser, or
  GGUF write path). Worth checking by hashing the actual `dst->data` bytes
  immediately after `set_rows_cuda` returns for the three types.
- D: dequant LUT (`vtq_trellis_table_storage`) is shared across K and
  initialized via `cudaMemcpyToSymbol`. If RDC (separable compilation) is
  somehow producing per-TU duplicates, one TU could see all-zeros while
  another sees the populated table — but this would manifest as zero PPL,
  not equal-PPL.
- E: K is being stripped at higher level (e.g. cache-type CLI parser maps
  vtq3_2 / vtq4_2 → vtq2_2 silently). Cross-check with
  `tools/server/server.cpp` and `common/arg.cpp` cache-type parsing.
