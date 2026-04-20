// Trick 6: CUDA receiver-side Viterbi encoder — STANDALONE API.
//
// This header exposes a flat, set_rows-independent entry point for bulk
// trellis encoding on the GPU. The existing `trellis-encode.cuh` kernel
// is specialised for the set_rows indexing path; this API is the tensor-
// free sibling used by the kv-cache bulk-quantize hook (prefill -> decode
// transition) where the caller already holds contiguous float groups
// and just wants them encoded.
//
// Signature mirrors the CPU reference `ggml_trellis_encode_group` but
// operates on G groups in one launch:
//
//   void trellis_encode_group_cuda(
//       const float * x,          // G * QK_GROUP contiguous floats (device)
//       int           K,          // code bits (2, 3, or 4)
//       int           G,          // number of groups to encode
//       uint8_t     * qs,         // G * (QK_GROUP*K+7)/8 bytes (device)
//       uint16_t    * start_state,// G entries (device)
//       float       * d,          // G entries (device, fp32)
//       cudaStream_t  stream);
//
// Opt-in via env `GGML_TRELLIS_CUDA_ENCODE=1`. On failure, the caller
// should fall back to the CPU encoder (`ggml_trellis_encode_group`).
// Beam width is taken from env `GGML_TRELLIS_BEAM`:
//     0        -> full Viterbi (S=65536 states, highest quality)
//     256..    -> beam search (faster, small quality cost)
// Default beam here is 256 to keep shared-memory footprint tractable
// (B=256 * 3 * 4 = 3 KiB per block for cost/state/parent, plus edge
//  history N*B*1 = 64 KiB → we use global-memory scratch for edge log).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Query whether the opt-in env flag is enabled. Cached after first call.
int trellis_cuda_encode_enabled(void);

// Return currently configured beam width (0 = full Viterbi). Cached.
int trellis_cuda_encode_beam_width(void);

// Bulk-encode G groups. Each group is QK_GROUP=256 floats, produces one
// trellis block = start_state(16b) + d(fp32) + qs((256*K+7)/8 bytes).
//
// The caller is responsible for:
//   * Allocating all buffers on the device.
//   * Ensuring `x` is contiguous (stride = QK_GROUP per group).
//   * Zero-initialising or ignoring `qs` — the kernel fully overwrites it.
//
// Returns cudaSuccess on launch success. Returns an error if K not in {2,3,4}
// or G <= 0 (no-op).
cudaError_t trellis_encode_group_cuda(
    const float * x,
    int           K,
    int           G,
    uint8_t     * qs,
    uint16_t    * start_state,
    float       * d,
    cudaStream_t  stream);

// Release any scratch buffers allocated for the beam-search path.
// Safe to call from module teardown.
void trellis_encode_group_cuda_free(void);

#ifdef __cplusplus
}
#endif
