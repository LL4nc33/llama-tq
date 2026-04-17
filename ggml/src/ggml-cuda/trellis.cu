// Placeholder TU for trellis — actual kernel logic + LUT are in trellis.cuh
// as header-only definitions (static __device__). Each TU that includes
// the header gets its own TU-local LUT copy, initialized via
// GGML_CUDA_INIT_TRELLIS_TABLE_IMPL() (see trellis.cuh).
//
// This file exists so the build system still links a `trellis.cu.o`
// (referenced via the GLOB in ggml-cuda/CMakeLists.txt).
#include "trellis.cuh"
