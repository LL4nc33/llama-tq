// Phase-1 sweep driver. Runs configs, writes CSV.
// Usage:
//   trellis_phase1 --mode gauss  --n 32768 --out results/sweep_gauss.csv
//   trellis_phase1 --mode real   --n 32768 --data vcache.bin --out results/sweep_real.csv

#include "trellis_phase1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Format: L, K, QK, beam, norm, group, shared_d, code, label
static const trellis_config CONFIGS[] = {
    // --- Best known baseline (post-RHT Gaussian, real-data-validated) ---
    {16, 2, 128, 0, 1, 2, 0, TRELLIS_CODE_TABLE, "baseline_Q128_G2"   },  // 2.1875 bpw
    // --- QK=256 sweep: amortize d and start_state further ---
    {16, 2, 256, 0, 1, 1, 0, TRELLIS_CODE_TABLE, "Q256_G1"            },  // 2.125 bpw
    {16, 2, 256, 0, 1, 2, 0, TRELLIS_CODE_TABLE, "Q256_G2"            },  // 2.09375 bpw
    {16, 2, 256, 0, 1, 4, 0, TRELLIS_CODE_TABLE, "Q256_G4"            },  // 2.078 bpw
    // --- shared_d variants (one fp16 d per group) ---
    {16, 2, 128, 0, 1, 2, 1, TRELLIS_CODE_TABLE, "Q128_G2_sharedD"    },  // 2.125 bpw
    {16, 2, 128, 0, 1, 4, 1, TRELLIS_CODE_TABLE, "Q128_G4_sharedD"    },  // 2.094 bpw
    {16, 2, 256, 0, 1, 2, 1, TRELLIS_CODE_TABLE, "Q256_G2_sharedD"    },  // 2.063 bpw
    {16, 2, 256, 0, 1, 4, 1, TRELLIS_CODE_TABLE, "Q256_G4_sharedD"    },  // 2.047 bpw
    // --- aggressive G on QK=128 ---
    {16, 2, 128, 0, 1, 4, 0, TRELLIS_CODE_TABLE, "Q128_G4"            },  // 2.156 bpw
    {16, 2, 128, 0, 1, 8, 0, TRELLIS_CODE_TABLE, "Q128_G8"            },  // 2.141 bpw
    // --- 3-bit with aggressive amortization ---
    {16, 3, 128, 0, 1, 4, 0, TRELLIS_CODE_TABLE, "K3_Q128_G4"         },  // 3.156 bpw
    {16, 3, 128, 0, 1, 4, 1, TRELLIS_CODE_TABLE, "K3_Q128_G4_sharedD" },  // 3.125 bpw
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

        // bpw: per block: QK·K (qs). Per block plus shared fields:
        //   if shared_d: 16 (d) per group + L (start) per group = 16+L per group
        //   else       : 16 (d) per block + L (start) per group
        double per_block_bits = (double)QK * cfg->code_bits;
        double per_group_bits = cfg->state_bits + (cfg->shared_d ? 16.0 : 0.0);
        double per_block_shared = cfg->shared_d ? 0.0 : 16.0;
        double bpw = (per_block_bits + per_block_shared
                      + per_group_bits / G) / (double)QK;

        double t0 = trellis_now_ms();
        double sq_err = 0.0;

        // Allocate G blocks worth of state so we can chain within groups.
        trellis_block blks[16]; // G ≤ 16
        uint32_t ends[16];
        float recon[256];      // enough for QK up to 256

        size_t ng = nb / (size_t)G;
        for (size_t g = 0; g < ng; g++) {
            // If shared_d: compute group L2 norm, pass as override to each block.
            float group_norm = -1.0f;
            if (cfg->shared_d) {
                double gn2 = 0.0;
                for (int bi = 0; bi < G; bi++) {
                    size_t b = g * (size_t)G + bi;
                    const float * xb = data + b * QK;
                    for (size_t j = 0; j < QK; j++) gn2 += (double)xb[j] * xb[j];
                }
                // per-block effective norm: the encoder scales by `norm`, and
                // the codebook has E[sum(code²)] ≈ QK; so each block needs
                // norm = sqrt(sum_per_block). We use sqrt(total / G) as a
                // uniform per-block scale so the decoder's per-block apply
                // of this same value reproduces the right magnitude.
                group_norm = (float)sqrt(gn2 / (double)G);
            }

            // Encode G chained blocks.
            uint32_t prev_end = 0xFFFFFFFFu;
            for (int bi = 0; bi < G; bi++) {
                size_t b = g * (size_t)G + bi;
                const float * xb = data + b * QK;
                ends[bi] = trellis_encode_block(cfg, xb, prev_end,
                                                group_norm, &blks[bi]);
                prev_end = ends[bi];
            }
            // Decode G chained blocks.
            uint32_t dec_start = 0xFFFFFFFFu;
            for (int bi = 0; bi < G; bi++) {
                size_t b = g * (size_t)G + bi;
                const float * xb = data + b * QK;
                trellis_decode_block(cfg, &blks[bi], dec_start,
                                     group_norm, recon);
                dec_start = ends[bi];
                for (size_t j = 0; j < QK; j++) {
                    float e = xb[j] - recon[j];
                    sq_err += (double)e * e;
                }
            }
        }
        size_t total_samples = ng * (size_t)G * QK;
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
    } else if (!strcmp(mode, "laplace")) {
        trellis_gen_laplace(data, n);
    } else if (!strcmp(mode, "student5")) {
        trellis_gen_student_t(data, n, 5.0f);
    } else if (!strcmp(mode, "bimodal")) {
        trellis_gen_bimodal(data, n);
    } else if (!strcmp(mode, "vcachelike")) {
        trellis_gen_vcache_like(data, n);
    } else if (!strcmp(mode, "vcache_real")) {
        trellis_gen_vcache_realistic(data, n);
    } else if (!strcmp(mode, "real")) {
        if (!data_path) { fprintf(stderr, "--data required in real mode\n"); return 1; }
        if (trellis_load_binary(data_path, data, n) != 0) {
            fprintf(stderr, "failed to load %s\n", data_path); return 1;
        }
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode); return 1;
    }

    // Empirical variance diagnostic
    double vsum = 0.0, vsum2 = 0.0;
    for (size_t i = 0; i < n; i++) { vsum += data[i]; vsum2 += (double)data[i]*data[i]; }
    double vmean = vsum/n, vvar = vsum2/n - vmean*vmean;
    fprintf(stderr, "data stats: mean=%+.4f var=%.4f std=%.4f\n", vmean, vvar, sqrt(vvar));

    fprintf(stderr, "loaded %zu %s samples\n", n, mode);
    run_sweep(data, n, out_path, mode);
    free(data);
    fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
