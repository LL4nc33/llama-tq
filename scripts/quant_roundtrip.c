// quant_roundtrip.c — expose the REAL llama.cpp quantization codebook (q2_K, IQ2_XXS, ...)
// as an f32 -> quant -> f32 roundtrip, callable from Python via ctypes.
//
// This is the heart of the C++ QAT path: gguf-py cannot quantize k-quants/i-quants
// (NotImplementedError), so a Python STE proxy overfits its own grid and the learned
// contraction vanishes under the real deploy codebook. ggml_quantize_chunk IS the function
// llama-quantize uses, so this gives the exact deploy codebook (incl. imatrix for i-quants).
//
// Build (on the box with libggml):
//   cc -O2 -shared -fPIC quant_roundtrip.c -o libquant_roundtrip.so \
//      -I<repo>/ggml/include -L<build>/ggml/src -lggml-base
//
// Python:
//   lib = ctypes.CDLL("./libquant_roundtrip.so")
//   lib.qrt_roundtrip(type_id, src_f32_ptr, dst_f32_ptr, nrows, n_per_row, imatrix_ptr_or_null)
#include "ggml.h"
#include <stdlib.h>
#include <string.h>

// f32 [nrows, n_per_row] -> quantize(type) -> dequantize -> f32 [nrows, n_per_row], in place to dst.
// imatrix may be NULL for types that don't require it (q2_K); required for i-quants (IQ2_XXS).
// Returns 0 on success, non-zero on failure.
int qrt_roundtrip(int type_id, const float * src, float * dst,
                  long nrows, long n_per_row, const float * imatrix) {
    enum ggml_type type = (enum ggml_type) type_id;
    if (!ggml_is_quantized(type)) return 1;
    if (ggml_quantize_requires_imatrix(type) && imatrix == NULL) return 2;

    const size_t row_size_q = ggml_row_size(type, n_per_row);
    void * q = malloc(row_size_q * (size_t) nrows);
    if (!q) return 3;

    // quantize all rows at once (start=0). imatrix is per-column (n_per_row), shared across rows.
    ggml_quantize_chunk(type, src, q, 0, nrows, n_per_row, imatrix);

    // dequantize back to f32 via the type's to_float trait
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) { free(q); return 4; }
    for (long r = 0; r < nrows; r++) {
        const void * qr = (const char *) q + (size_t) r * row_size_q;
        float * dr = dst + (size_t) r * n_per_row;
        tr->to_float(qr, dr, n_per_row);
    }
    free(q);
    return 0;
}

// resolve a ggml_type id from its name ("q2_K", "iq2_xxs", ...) so Python doesn't hardcode enum ints.
int qrt_type_by_name(const char * name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const struct ggml_type_traits * tr = ggml_get_type_traits((enum ggml_type) t);
        if (tr && tr->type_name && strcmp(tr->type_name, name) == 0) return t;
    }
    return -1;
}
