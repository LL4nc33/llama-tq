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
    // Group-size sweep on L16_K2_Q32_TBL. G=1 matches Run 1b; larger G
    // shares one start_state across G blocks, reducing bpw overhead from
    // 0.5 (per-block) to 0.5/G. G=4 is a head_dim=128 row (4 blocks).
    // Format: L, K, QK, beam, norm, group, code, label
    {16, 2,  32, 0, 1, 1, TRELLIS_CODE_TABLE, "L16_K2_Q32_G1" },
    {16, 2,  32, 0, 1, 2, TRELLIS_CODE_TABLE, "L16_K2_Q32_G2" },
    {16, 2,  32, 0, 1, 4, TRELLIS_CODE_TABLE, "L16_K2_Q32_G4" },
    {16, 2,  32, 0, 1, 8, TRELLIS_CODE_TABLE, "L16_K2_Q32_G8" },
    // Same for Q=64
    {16, 2,  64, 0, 1, 1, TRELLIS_CODE_TABLE, "L16_K2_Q64_G1" },
    {16, 2,  64, 0, 1, 2, TRELLIS_CODE_TABLE, "L16_K2_Q64_G2" },
    {16, 2,  64, 0, 1, 4, TRELLIS_CODE_TABLE, "L16_K2_Q64_G4" },
    // Q=128 (row-size for head_dim=128)
    {16, 2, 128, 0, 1, 1, TRELLIS_CODE_TABLE, "L16_K2_Q128_G1" },
    {16, 2, 128, 0, 1, 2, TRELLIS_CODE_TABLE, "L16_K2_Q128_G2" },
    // And 3-bit
    {16, 3,  32, 0, 1, 1, TRELLIS_CODE_TABLE, "L16_K3_Q32_G1" },
    {16, 3,  32, 0, 1, 4, TRELLIS_CODE_TABLE, "L16_K3_Q32_G4" },
};

static const size_t N_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

// Lloyd-Max 2-bit Beta(15.5,15.5) MSE on N(0,1) — baseline for gate.
static const float LLOYD_MAX_MSE_2BIT = 0.1175f;

static void run_sweep(const float * data, size_t n, const char * out_path, const char * mode) {
    FILE * f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return; }
    fprintf(f, "mode,config,L,K,QK,beam,norm,group,code,n_blocks,bpw,mse,mse_ratio,encode_ms_total,encode_ms_per_block\n");

    for (size_t ci = 0; ci < N_CONFIGS; ci++) {
        const trellis_config * cfg = &CONFIGS[ci];
        size_t QK = (size_t)cfg->block_size;
        size_t nb = n / QK;
        if (nb == 0) continue;
        int G = (cfg->group_size > 0) ? cfg->group_size : 1;

        // bpw: 16 (d) + QK·K (qs) per block + L (start_state) per group
        double bpw = (16.0 + (double)QK * cfg->code_bits + (double)cfg->state_bits / G) / (double)QK;

        double t0 = trellis_now_ms();
        double sq_err = 0.0;

        // Allocate G blocks worth of state so we can chain within groups.
        trellis_block blks[16]; // G ≤ 16
        uint32_t ends[16];
        float recon[128];

        size_t ng = nb / (size_t)G;
        for (size_t g = 0; g < ng; g++) {
            // Encode G chained blocks.
            uint32_t prev_end = 0xFFFFFFFFu; // first block: open start
            for (int bi = 0; bi < G; bi++) {
                size_t b = g * (size_t)G + bi;
                const float * xb = data + b * QK;
                ends[bi] = trellis_encode_block(cfg, xb, prev_end, &blks[bi]);
                prev_end = ends[bi];
            }
            // Decode G chained blocks (re-derive start from blks[0].start_state).
            uint32_t dec_start = 0xFFFFFFFFu; // blk0: read from field
            for (int bi = 0; bi < G; bi++) {
                size_t b = g * (size_t)G + bi;
                const float * xb = data + b * QK;
                trellis_decode_block(cfg, &blks[bi], dec_start, recon);
                dec_start = ends[bi]; // chain end_state into next block's forced start
                for (size_t j = 0; j < QK; j++) {
                    float e = xb[j] - recon[j];
                    sq_err += (double)e * e;
                }
            }
        }
        size_t total_samples = ng * (size_t)G * QK;
        // reuse nb for reporting (blocks we actually processed)
        nb = ng * (size_t)G;
        double t1 = trellis_now_ms();
        double mse = sq_err / (double)total_samples;
        double mse_ratio = mse / LLOYD_MAX_MSE_2BIT;

        fprintf(f, "%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%.4f,%.6f,%.4f,%.2f,%.4f\n",
                mode, cfg->label, cfg->state_bits, cfg->code_bits, cfg->block_size,
                cfg->beam_width, cfg->norm_correction, G, (int)cfg->code,
                nb, bpw, mse, mse_ratio, (t1 - t0), (t1 - t0) / (double)nb);
        fprintf(stderr, "[%s] %-20s bpw=%.3f MSE=%.5f ratio=%.3f time=%.1fms/%zu\n",
                mode, cfg->label, bpw, mse, mse_ratio, (t1 - t0), nb);
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
