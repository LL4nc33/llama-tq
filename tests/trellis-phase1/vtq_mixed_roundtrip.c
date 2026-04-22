// Quick C roundtrip test for VTQ_MIXED on real V-samples.
// Compares MSE against the Python prediction (~9.38%) from the same bin.
//
// Usage: ./vtq_mixed_rt /tmp/vcache-qwen35-27b.bin
// Build: gcc vtq_mixed_roundtrip.c -o vtq_mixed_rt \
//          -I../../ggml/include -I../../ggml/src \
//          ../../build/ggml/src/libggml.a -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-common.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <samples.bin>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long n_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int64_t n_floats = n_bytes / sizeof(float);
    // round down to multiple of QK_VTQ
    n_floats -= n_floats % QK_VTQ;
    int64_t n_blocks = n_floats / QK_VTQ;

    float *samples = malloc(n_floats * sizeof(float));
    float *recon   = malloc(n_floats * sizeof(float));
    block_vtq_mixed *quant = malloc(n_blocks * sizeof(block_vtq_mixed));
    if (!samples || !recon || !quant) { perror("malloc"); return 1; }
    fread(samples, sizeof(float), n_floats, f);
    fclose(f);

    quantize_row_vtq_mixed_ref(samples, quant, n_floats);
    dequantize_row_vtq_mixed(quant, recon, n_floats);

    double sum_mse = 0.0, var = 0.0;
    double mean = 0.0;
    for (int64_t i = 0; i < n_floats; i++) mean += samples[i];
    mean /= n_floats;
    for (int64_t i = 0; i < n_floats; i++) {
        double d = samples[i] - recon[i];
        sum_mse += d * d;
        double c = samples[i] - mean;
        var += c * c;
    }
    double mse = sum_mse / n_floats;
    double rel_mse = sum_mse / var;

    printf("VTQ_MIXED roundtrip on %lld blocks (%lld samples):\n",
           (long long)n_blocks, (long long)n_floats);
    printf("  MSE      = %.5e\n", mse);
    printf("  rel MSE  = %.3f%%\n", rel_mse * 100);
    printf("  Python prediction was 9.384%% — ");
    printf(fabs(rel_mse * 100 - 9.384) < 0.5 ? "PASS\n" : "DRIFT\n");

    free(samples); free(recon); free(quant);
    return 0;
}
