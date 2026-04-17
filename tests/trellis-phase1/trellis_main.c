// Phase-1 sweep driver. Runs configs, writes CSV.
// Usage:
//   trellis_phase1 --mode gauss  --n 32768 --out results/sweep_gauss.csv
//   trellis_phase1 --mode real   --n 32768 --data vcache.bin --out results/sweep_real.csv

#include "trellis_phase1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trellis_config CONFIGS[] = {
    // label,          L,  K, QK,  beam, norm, code
    { 8,  2,  32, 0, 1, TRELLIS_CODE_3GAUSS, "L8_K2_Q32" },
    {16,  2,  32, 0, 1, TRELLIS_CODE_3GAUSS, "L16_K2_Q32" },
    {16,  2,  64, 0, 1, TRELLIS_CODE_3GAUSS, "L16_K2_Q64" },
    {16,  2, 128, 0, 1, TRELLIS_CODE_3GAUSS, "L16_K2_Q128" },
    { 8,  2, 128, 0, 1, TRELLIS_CODE_3GAUSS, "L8_K2_Q128" },
    {12,  2,  64, 0, 1, TRELLIS_CODE_3GAUSS, "L12_K2_Q64" },
    {16,  3,  32, 0, 1, TRELLIS_CODE_3GAUSS, "L16_K3_Q32" },
};

static const size_t N_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

// Lloyd-Max 2-bit Beta(15.5,15.5) MSE on N(0,1) — baseline for gate.
static const float LLOYD_MAX_MSE_2BIT = 0.1175f;

static void run_sweep(const float * data, size_t n, const char * out_path, const char * mode) {
    FILE * f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return; }
    fprintf(f, "mode,config,L,K,QK,beam,norm,code,n_blocks,mse,mse_ratio,encode_ms_total,encode_ms_per_block\n");

    for (size_t ci = 0; ci < N_CONFIGS; ci++) {
        const trellis_config * cfg = &CONFIGS[ci];
        size_t QK = (size_t)cfg->block_size;
        size_t nb = n / QK;
        if (nb == 0) continue;

        double t0 = trellis_now_ms();
        double sq_err = 0.0;
        double sq_ref = 0.0;

        trellis_block blk;
        float recon[128];

        for (size_t b = 0; b < nb; b++) {
            const float * xb = data + b * QK;
            (void)trellis_encode_block(cfg, xb, &blk);
            trellis_decode_block(cfg, &blk, recon);
            for (size_t j = 0; j < QK; j++) {
                float e = xb[j] - recon[j];
                sq_err += (double)e * e;
                sq_ref += (double)xb[j] * xb[j];
            }
        }
        double t1 = trellis_now_ms();
        double mse = sq_err / (double)(nb * QK);
        double mse_ratio = mse / LLOYD_MAX_MSE_2BIT;

        fprintf(f, "%s,%s,%d,%d,%d,%d,%d,%d,%zu,%.6f,%.4f,%.2f,%.4f\n",
                mode, cfg->label, cfg->state_bits, cfg->code_bits, cfg->block_size,
                cfg->beam_width, cfg->norm_correction, (int)cfg->code,
                nb, mse, mse_ratio, (t1 - t0), (t1 - t0) / (double)nb);
        fprintf(stderr, "[%s] %s: MSE=%.5f ratio=%.3f time=%.1fms/%zu blocks\n",
                mode, cfg->label, mse, mse_ratio, (t1 - t0), nb);
        fflush(f);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    const char * mode = "gauss";
    const char * data_path = NULL;
    const char * out_path = "trellis_sweep.csv";
    size_t n = 32768;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc)  mode = argv[++i];
        else if (!strcmp(argv[i], "--n")    && i + 1 < argc) n = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else if (!strcmp(argv[i], "--out")  && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) trellis_seed_rng((uint64_t)atoll(argv[++i]));
    }

    float * data = (float *)malloc(n * sizeof(float));
    if (!data) { fprintf(stderr, "alloc failed\n"); return 1; }

    if (!strcmp(mode, "gauss")) {
        trellis_gen_gaussian(data, n);
    } else if (!strcmp(mode, "real")) {
        if (!data_path) { fprintf(stderr, "--data required in real mode\n"); return 1; }
        if (trellis_load_binary(data_path, data, n) != 0) {
            fprintf(stderr, "failed to load %s\n", data_path); return 1;
        }
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode); return 1;
    }

    fprintf(stderr, "loaded %zu %s samples\n", n, mode);
    run_sweep(data, n, out_path, mode);
    free(data);
    fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
