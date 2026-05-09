// gen_fixture.c — Generate KTQ2_1 golden-block fixture for the Vulkan POC.
//
// Self-contained: replicates the in-tree CPU encode (`quantize_row_ktq2_1_ref`
// in ggml/src/ggml-quants.c:5832) verbatim, then computes the reference
// dequant using **CUDA-style arithmetic** (codebook → FWHT → (1-2*sb)*norm),
// which is what the GLSL shader will implement. This sidesteps the documented
// sign-convention difference between the CPU `dequantize_row_ktq2_1` and the
// CUDA `dequantize_block_ktq2_1_v2` — for a POC harness we only need
// a self-consistent encode/dequant pair that exercises the same arithmetic
// the shader uses.
//
// Output: /tmp/poc-ktq2/fixture.bin
//   uint32  magic = 0x4B544932 ("KTI2")
//   uint32  num_blocks (= 100)
//   uint32  block_bytes (= 14)
//   uint32  elements_per_block (= 32)
//   block_ktq2_1 blocks[N]            // raw, 14B each, no padding
//   float        ref_dequant[N*32]    // CUDA-arithmetic reference
//
// Build (host or test-rig):
//   gcc -O2 -Wall gen_fixture.c -o /tmp/poc-ktq2/gen_fixture -lm

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define QK_KTQ 32
#define NUM_BLOCKS 100
#define MAGIC 0x4B544932u

#pragma pack(push, 1)
typedef struct {
    uint16_t d;          // ggml_half (raw fp16 bits)
    uint8_t  qs[QK_KTQ / 4];  // 8 bytes
    uint8_t  sb[QK_KTQ / 8];  // 4 bytes
} block_ktq2_1;
#pragma pack(pop)
_Static_assert(sizeof(block_ktq2_1) == 14, "block_ktq2_1 must be 14 bytes");

// ----------------------------------------------------------------------------
// fp32 <-> fp16 (IEEE-754 half-precision)
// Mirror of ggml_compute_fp32_to_fp16 / fp16_to_fp32 (no NaN/Inf handling
// needed here: norms are bounded). Matches GGML_FP16_TO_FP32 round-trip.
// ----------------------------------------------------------------------------
static uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t u = v.u;
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = 14 - exp;
        uint32_t round = (mant >> (shift - 1)) & 1u;
        uint16_t h = (uint16_t)(sign | (mant >> shift));
        return (uint16_t)(h + round);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    uint16_t h = (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
    if (mant & 0x1000u) h++;
    return h;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            // subnormal — rare for our norms; renormalize
            int e = -1;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            u = sign | ((uint32_t)(127 - 15 + e + 1) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { .u = u };
    return v.f;
}

// ----------------------------------------------------------------------------
// Verbatim copies of the in-tree TurboQuant CPU primitives (see citations
// in source headers). These are static-linked into the POC tool to keep it
// build-system-free.
// ----------------------------------------------------------------------------

// ggml-quants.c:5527 — Philox-6r counter PRNG
static inline uint32_t ktq_philox_6r(uint32_t counter, uint32_t key) {
    uint32_t lo = counter;
    uint32_t hi = key;
    for (int i = 0; i < 6; ++i) {
        const uint32_t lo_old = lo;
        lo = (uint32_t)(((uint64_t)lo_old * 0xD2511F53u) >> 32) ^ hi ^ (0x9E3779B9u * (uint32_t)(i + 1));
        hi = lo_old * 0xD2511F53u;
    }
    return lo;
}

// ggml-quants.c:5538 — sign vector from Philox bits
static void tq_random_signs(uint16_t seed, float * signs, int n) {
    for (int i = 0; i < n; i++) {
        signs[i] = (ktq_philox_6r((uint32_t)i, (uint32_t)seed) & 1) ? 1.0f : -1.0f;
    }
}

// ggml-quants.c:5544 — serial FWHT (matches warp shuffle butterfly bit-exactly)
static void kktq_fwht(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    const float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) data[i] *= scale;
}

// ggml-quants.c:5721
static void kktq_rht_forward(const float * x, float * y, int n, uint16_t seed) {
    float signs[QK_KTQ];
    tq_random_signs(seed, signs, n);
    for (int i = 0; i < n; i++) y[i] = x[i] * signs[i];
    kktq_fwht(y, n);
}

// ggml-quants.c:5728
static void kktq_rht_inverse(const float * y, float * x, int n, uint16_t seed) {
    float signs[QK_KTQ];
    tq_random_signs(seed, signs, n);
    for (int i = 0; i < n; i++) x[i] = y[i];
    kktq_fwht(x, n);
    for (int i = 0; i < n; i++) x[i] *= signs[i];
}

// ggml-quants.c:5755
static inline uint16_t kktq_derive_seed(int64_t block_index) {
    uint32_t h = 2166136261u;
    h ^= (uint32_t)(block_index & 0xFF);        h *= 16777619u;
    h ^= (uint32_t)((block_index >> 8) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 16) & 0xFF); h *= 16777619u;
    h ^= (uint32_t)((block_index >> 24) & 0xFF); h *= 16777619u;
    return (uint16_t)(h & 0xFFFF);
}

static const float PQ_CODEBOOK_2BIT[4] = {
    -1.489560f, -0.451428f, 0.451428f, 1.489560f
};

// Verbatim port of `quantize_row_ktq2_1_ref` (ggml-quants.c:5832).
// Greedy quantization (not stochastic — matches CPU ref, not CUDA encode).
static void quantize_row_ktq2_1_ref(const float * x, block_ktq2_1 * y, int64_t k) {
    assert(k % QK_KTQ == 0);
    const int nb = (int)(k / QK_KTQ);
    const float cb_scale = 1.0f / sqrtf((float)QK_KTQ);

    for (int i = 0; i < nb; i++) {
        const float * xi = x + i * QK_KTQ;
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_KTQ; j++) norm_sq += xi[j] * xi[j];
        float norm = sqrtf(norm_sq);
        y[i].d = fp32_to_fp16(norm);

        if (norm < 1e-30f) {
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].sb, 0, sizeof(y[i].sb));
            continue;
        }

        float x_hat[QK_KTQ];
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_KTQ; j++) x_hat[j] = xi[j] * inv_norm;

        uint16_t seed = kktq_derive_seed((int64_t)i);
        float rotated[QK_KTQ];
        kktq_rht_forward(x_hat, rotated, QK_KTQ, seed);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < QK_KTQ; j++) {
            float val = rotated[j];
            float best_dist = FLT_MAX;
            uint8_t best_idx = 0;
            for (int c = 0; c < 4; c++) {
                float centroid = PQ_CODEBOOK_2BIT[c] * cb_scale;
                float dist = (val - centroid) * (val - centroid);
                if (dist < best_dist) { best_dist = dist; best_idx = (uint8_t)c; }
            }
            y[i].qs[j / 4] |= (best_idx << (2 * (j % 4)));
        }

        // Norm-correction (v5): d = norm_input / norm_recon
        {
            float recon[QK_KTQ];
            for (int j = 0; j < QK_KTQ; j++) {
                int idx = (y[i].qs[j / 4] >> (2 * (j % 4))) & 0x3;
                recon[j] = PQ_CODEBOOK_2BIT[idx] * cb_scale;
            }
            float result[QK_KTQ];
            kktq_rht_inverse(recon, result, QK_KTQ, seed);
            float recon_sq = 0.0f;
            for (int j = 0; j < QK_KTQ; j++) recon_sq += result[j] * result[j];
            float recon_norm = sqrtf(recon_sq);
            y[i].d = fp32_to_fp16((recon_norm > 1e-30f) ? norm / recon_norm : norm);
        }

        // Precomputed RHT sign bits (v5 design)
        memset(y[i].sb, 0, sizeof(y[i].sb));
        for (int j = 0; j < QK_KTQ; j++) {
            uint8_t sign_bit = (uint8_t)(ktq_philox_6r((uint32_t)j, (uint32_t)seed) & 1u);
            y[i].sb[j / 8] |= (sign_bit << (j % 8));
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA-arithmetic reference dequant: identical to what the shader does.
// (Deliberately NOT calling `dequantize_row_ktq2_1` from the in-tree CPU side
//  because that function uses the *opposite* sign convention from CUDA — see
//  comment in dequant_ktq2_1.comp. The shader mirrors CUDA, so the reference
//  must too.)
// ----------------------------------------------------------------------------
static void dequant_cuda_arith(const block_ktq2_1 * x, float * y, int nb) {
    const float cb_scale = 1.0f / sqrtf((float)QK_KTQ);
    for (int ib = 0; ib < nb; ib++) {
        float norm = fp16_to_fp32(x[ib].d);
        float val[QK_KTQ];
        for (int j = 0; j < QK_KTQ; j++) {
            int idx = (x[ib].qs[j / 4] >> (2 * (j % 4))) & 0x3;
            val[j] = PQ_CODEBOOK_2BIT[idx] * cb_scale;
        }
        // FWHT (serial form — bit-exact to the warp shuffle butterfly per stage).
        for (int len = 1; len < QK_KTQ; len <<= 1) {
            for (int i = 0; i < QK_KTQ; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    float u = val[i + j];
                    float v = val[i + j + len];
                    val[i + j]       = u + v;
                    val[i + j + len] = u - v;
                }
            }
        }
        for (int j = 0; j < QK_KTQ; j++) val[j] *= cb_scale; // 1/sqrt(32) end of FWHT
        // CUDA convention: bit 0 → +1, bit 1 → -1.
        for (int j = 0; j < QK_KTQ; j++) {
            int sb = (x[ib].sb[j / 8] >> (j % 8)) & 1;
            y[ib * QK_KTQ + j] = val[j] * (1.0f - 2.0f * (float)sb) * norm;
        }
    }
}

// xorshift64* — deterministic, header-free PRNG for input generation.
static uint64_t xs_state = 0xC0FFEEDEADBEEFULL;
static uint32_t xs_next(void) {
    uint64_t x = xs_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    xs_state = x;
    return (uint32_t)(x * 0x2545F4914F6CDD1DULL >> 32);
}
static float xs_uniform(float lo, float hi) {
    float u = (float)xs_next() / (float)0xFFFFFFFFu;  // [0,1]
    return lo + u * (hi - lo);
}

int main(int argc, char ** argv) {
    const char * outpath = (argc > 1) ? argv[1] : "/tmp/poc-ktq2/fixture.bin";
    const int N = NUM_BLOCKS;

    float * input = (float *)calloc((size_t)N * QK_KTQ, sizeof(float));
    block_ktq2_1 * blocks = (block_ktq2_1 *)calloc((size_t)N, sizeof(block_ktq2_1));
    float * ref = (float *)calloc((size_t)N * QK_KTQ, sizeof(float));
    if (!input || !blocks || !ref) { perror("calloc"); return 1; }

    // Mix of distributions to stress the codebook + sign paths.
    // Block 0..49: uniform [-1, 1]
    // Block 50..89: gaussian-ish (sum of 8 uniforms, scaled)
    // Block 90..99: edge cases — small magnitudes, then large
    for (int b = 0; b < N; b++) {
        float * dst = input + b * QK_KTQ;
        if (b < 50) {
            for (int j = 0; j < QK_KTQ; j++) dst[j] = xs_uniform(-1.0f, 1.0f);
        } else if (b < 90) {
            for (int j = 0; j < QK_KTQ; j++) {
                float s = 0.0f;
                for (int k = 0; k < 8; k++) s += xs_uniform(-1.0f, 1.0f);
                dst[j] = s * 0.354f;  // ≈ unit variance
            }
        } else if (b < 95) {
            for (int j = 0; j < QK_KTQ; j++) dst[j] = xs_uniform(-1e-3f, 1e-3f);
        } else {
            for (int j = 0; j < QK_KTQ; j++) dst[j] = xs_uniform(-100.0f, 100.0f);
        }
    }

    // Encode.
    quantize_row_ktq2_1_ref(input, blocks, (int64_t)N * QK_KTQ);

    // Reference dequant (CUDA-arithmetic).
    dequant_cuda_arith(blocks, ref, N);

    // Write fixture.
    FILE * f = fopen(outpath, "wb");
    if (!f) { perror(outpath); return 1; }
    uint32_t hdr[4] = { MAGIC, (uint32_t)N, 14u, (uint32_t)QK_KTQ };
    fwrite(hdr, sizeof(uint32_t), 4, f);
    fwrite(blocks, sizeof(block_ktq2_1), (size_t)N, f);
    fwrite(ref, sizeof(float), (size_t)N * QK_KTQ, f);
    fclose(f);

    printf("Wrote %d blocks (%zu B blocks + %zu B ref) to %s\n",
        N, sizeof(block_ktq2_1) * N, sizeof(float) * N * QK_KTQ, outpath);

    // Sanity print: block 0 first 4 ref values.
    printf("ref[0,0..3] = %.6f %.6f %.6f %.6f\n", ref[0], ref[1], ref[2], ref[3]);
    printf("input[0,0..3] = %.6f %.6f %.6f %.6f\n", input[0], input[1], input[2], input[3]);
    printf("block[0]: d=%u qs[0..3]=%02x %02x %02x %02x sb=%02x %02x %02x %02x\n",
        blocks[0].d, blocks[0].qs[0], blocks[0].qs[1], blocks[0].qs[2], blocks[0].qs[3],
        blocks[0].sb[0], blocks[0].sb[1], blocks[0].sb[2], blocks[0].sb[3]);

    free(input); free(blocks); free(ref);
    return 0;
}
