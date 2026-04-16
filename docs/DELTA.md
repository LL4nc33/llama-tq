# Delta from upstream llama.cpp

This fork adds TurboQuant v7 KV cache quantization with KTQ/VTQ split (8 types total). All changes are contained in these files:

## New Files

| File | Description |
|------|-------------|
| `ggml/src/ggml-cuda/turboquant.cuh` | CUDA kernels: Philox PRNG, FWHT, KTQ quantize/dequant, VTQ quantize/dequant, get-rows, set-rows |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*-ktq*.cu` | 72 FA template instances for KTQ K-type x {all V-types} combinations |

## Modified Files

### Core Type System

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | +8 type enums: KTQ2_1=42, KTQ3_1=43, KTQ4_1=44, KTQ1_1=45, VTQ1_1=46, VTQ2_1=47, VTQ3_1=48, VTQ4_1=49 |
| `ggml/src/ggml-common.h` | +8 block structs: `block_ktq1_1..block_ktq4_1` (with `sb[4]` sign bits) + `block_vtq1_1..block_vtq4_1` (no sign bits) |
| `ggml/src/ggml.c` | +8 type metadata entries (name, size, block_size) |

### CPU Implementation

| File | Change |
|------|--------|
| `ggml/src/ggml-quants.h` | +16 function declarations (quantize + dequantize per KTQ/VTQ type) |
| `ggml/src/ggml-quants.c` | KTQ + VTQ CPU quantize/dequantize, shared `PQ_CODEBOOK_*BIT` constants |

### CUDA Implementation

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/turboquant.cuh` | Main CUDA file: Philox 6r PRNG, warp-cooperative FWHT, KTQ quant/dequant, VTQ quant/dequant, shared `PQ_CUDA_CB_*BIT` codebooks |
| `ggml/src/ggml-cuda/fattn-common.cuh` | FA functions: `vec_dot_KQ_ktq*` (Hadamard-domain dot product), `dequantize_V_ktq*` (`__noinline__`), `dequantize_V_vtq*` (`__forceinline__`, ~8 registers), sparse V dequant guard |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | FA vec kernel: `Q_tq` constexpr, f32 Q registers, KTQ/VTQ vec_dot + V-dequant dispatch |
| `ggml/src/ggml-cuda/fattn.cu` | FA entry point: KTQ + VTQ type dispatch (KTQ as K-type, VTQ as V-type) |
| `ggml/src/ggml-cuda/convert.cu` | CUDA dequant kernels (contiguous + non-contiguous) for all 8 types |
| `ggml/src/ggml-cuda/getrows.cu` | get_rows dispatch for KTQ + VTQ types |
| `ggml/src/ggml-cuda/set-rows.cu` | `k_set_rows_ktq*` + `k_set_rows_vtq*` kernels |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +8 dequant dispatch entries |
| `ggml/src/ggml-cuda/CMakeLists.txt` | turboquant.cuh + 72 template instances |

### Graph / KV Cache / Model

| File | Change |
|------|--------|
| `src/llama-kv-cache.cpp` | `ggml_gen_hadamard()` rotation matrix construction + KV cache rotation init for VTQ |
| `src/llama-kv-cache.h` | VTQ rotation state in KV cache struct |
| `src/llama-context.cpp` | KTQ/VTQ type support in context init |
| `src/llama-model.cpp` | `self_v_rot` pre/post FA graph building for VTQ |

### CLI / Common

| File | Change |
|------|--------|
| `common/arg.cpp` | CLI: `--cache-type-k ktq1_1..ktq4_1`, `--cache-type-v vtq1_1..vtq4_1` |
| `common/common.h` | KTQ/VTQ cache type params |
| `tools/llama-bench/llama-bench.cpp` | KTQ/VTQ type parser for `-ctk`/`-ctv` flags |

## Type Summary

| Type | Enum | Family | bpw | Block | Layout |
|------|:----:|--------|:---:|:-----:|--------|
| `ktq1_1` | 45 | KTQ | 2.5 | 10B | `[d:2B] [qs:4B] [sb:4B]` |
| `ktq2_1` | 42 | KTQ | 3.5 | 14B | `[d:2B] [qs:8B] [sb:4B]` |
| `ktq3_1` | 43 | KTQ | 4.5 | 18B | `[d:2B] [qs:12B] [sb:4B]` |
| `ktq4_1` | 44 | KTQ | 5.5 | 22B | `[d:2B] [qs:16B] [sb:4B]` |
| `vtq1_1` | 46 | VTQ | 1.5 | 6B | `[d:2B] [qs:4B]` |
| `vtq2_1` | 47 | VTQ | 2.5 | 10B | `[d:2B] [qs:8B]` |
| `vtq3_1` | 48 | VTQ | 4.0 | 16B | `[d:2B] [qs:14B]` |
| `vtq4_1` | 49 | VTQ | 4.5 | 18B | `[d:2B] [qs:16B]` |

**Total delta:** ~4,300 lines added across 74 code files (+ 72 template instance files).
