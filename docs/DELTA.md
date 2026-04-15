# Delta from upstream llama.cpp

This fork adds TurboQuant v7 KV cache quantization. All changes are contained in these files:

## New Files
- `ggml/src/ggml-cuda/turboquant.cuh` -- CUDA kernels (Philox, FWHT, quantize, dequant, get-rows)
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*-tq*.cu` -- 72 FA template instances

## Modified Files

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | +3 type enums (TQ2_1=41, TQ3_1=42, TQ4_1=43) |
| `ggml/src/ggml-common.h` | +3 block structs (block_tq2_1, block_tq3_1, block_tq4_1) |
| `ggml/src/ggml.c` | +3 type metadata entries (name, size, block_size) |
| `ggml/src/ggml-quants.h` | +6 function declarations (quantize + dequantize per type) |
| `ggml/src/ggml-quants.c` | +359 lines: CPU quantize/dequantize/vec_dot for all 3 types |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +3 dequant dispatch entries |
| `ggml/src/ggml-cuda/convert.cu` | +161 lines: CUDA dequant kernels (contiguous + non-contiguous) |
| `ggml/src/ggml-cuda/getrows.cu` | +13 lines: get_rows dispatch for TQ types |
| `ggml/src/ggml-cuda/set-rows.cu` | +142 lines: k_set_rows_tq kernels |
| `ggml/src/ggml-cuda/CMakeLists.txt` | +9 lines: turboquant.cuh + template instances |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +453 lines: vec_dot_KQ + dequantize_V for TQ types |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | +47 lines: Q_tq constexpr, f32 Q registers, TQ vec_dot call |
| `ggml/src/ggml-cuda/fattn.cu` | +81 lines: TQ type dispatch in FA entry point |
| `common/arg.cpp` | +11 lines: CLI --cache-type-k/v tq2_1/tq3_1/tq4_1, --tq-deferred-k |
| `common/common.h` | +2 lines: tq_deferred_k param |
| `common/common.cpp` | +1 line: param forwarding |
| `tools/llama-bench/llama-bench.cpp` | +9 lines: TQ type parser for -ctk/-ctv flags |

**Total delta:** ~2,650 lines added, 3 lines modified across 66 files.
