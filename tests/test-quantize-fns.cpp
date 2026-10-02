// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

// Group-128 ternary types: data that is exactly ternary per block ({-d, 0, +d}
// with an fp16-exact d) must survive quantize -> dequantize bit-exactly, through
// both the reference row quantizer and ggml_quantize_chunk.
static int test_ternary_lossless(bool verbose) {
    int num_failed = 0;
    std::mt19937 rng(42);
    const int64_t n_per_row = 3 * 128;
    const int64_t nrows     = 4;
    const int64_t n         = n_per_row * nrows;

    for (ggml_type type : {GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0}) {
        const auto * traits = ggml_get_type_traits(type);
        for (int trial = 0; trial < 16; ++trial) {
            std::vector<float> x(n);
            for (int64_t b = 0; b < n / 128; ++b) {
                const float d = ldexpf(1.0f, (int) (rng() % 9) - 4); // 2^-4 .. 2^4
                for (int j = 0; j < 128; ++j) {
                    x[b*128 + j] = d * (float) ((int) (rng() % 3) - 1);
                }
                if (trial % 4 != 3) {
                    x[b*128 + rng() % 128] = (rng() & 1) ? d : -d; // make d the block max (trial%4==3 may leave all-zero blocks)
                }
            }

            const size_t row_size = ggml_row_size(type, n_per_row);
            std::vector<uint8_t> q_ref(row_size * nrows), q_chunk(row_size * nrows);
            traits->from_float_ref(x.data(), q_ref.data(), n);
            ggml_quantize_chunk(type, x.data(), q_chunk.data(), 0, nrows, n_per_row, nullptr);

            bool failed = memcmp(q_ref.data(), q_chunk.data(), q_ref.size()) != 0;
            failed = failed || !ggml_validate_row_data(type, q_ref.data(), q_ref.size());

            std::vector<float> y(n);
            traits->to_float(q_ref.data(), y.data(), n);
            for (int64_t i = 0; i < n && !failed; ++i) {
                failed = y[i] != x[i];
            }

            num_failed += failed;
            if (failed) {
                printf("%6s ternary lossless roundtrip trial %d: FAILED\n", ggml_type_name(type), trial);
            }
        }
    }
    if (num_failed || verbose) {
        printf("ternary lossless roundtrip: %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
    }
    return num_failed;
}

// Group-128 ternary types: vec_dot against Q8_0 must match the float dot of the
// dequantized operands exactly for arbitrary packed bit patterns (power-of-two
// scales keep both sides exact).
static int test_ternary_packed_dot(bool verbose) {
    int num_failed = 0;
    for (ggml_type type : {GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0}) {
        const auto * traits = ggml_get_type_traits(type);
        const auto * cpu    = ggml_get_type_traits_cpu(type);
        if (cpu->vec_dot_type != GGML_TYPE_Q8_0) {
            printf("%6s: unexpected vec_dot_type %s\n", ggml_type_name(type), ggml_type_name(cpu->vec_dot_type));
            num_failed++;
            continue;
        }
        for (int nb : {1, 3}) {
            const int n = nb * 128;
            std::vector<block_pq2_0>  pq(nb);
            std::vector<block_ptq1_0> ptq(nb);
            std::vector<block_q8_0>   q8(nb * 4);
            std::vector<float> x(n), y(n);
            const void * w = type == GGML_TYPE_PQ2_0 ? (const void *) pq.data() : (const void *) ptq.data();
            for (int pattern = 0; pattern < 256; ++pattern) {
                for (int i = 0; i < nb; ++i) {
                    pq[i].d = ptq[i].d = ggml_fp32_to_fp16(0.25f * (i + 1));
                    for (size_t j = 0; j < sizeof(pq[i].qs); ++j) {
                        pq[i].qs[j] = (uint8_t) (pattern + 17*j + i);
                    }
                    for (size_t j = 0; j < sizeof(ptq[i].qs); ++j) {
                        ptq[i].qs[j] = (uint8_t) (pattern + 17*j + i);
                    }
                    for (size_t j = 0; j < sizeof(ptq[i].qh); ++j) {
                        ptq[i].qh[j] = (uint8_t) (pattern + 37*j + i);
                    }
                }
                for (int i = 0; i < nb * 4; ++i) {
                    q8[i].d = ggml_fp32_to_fp16(0.125f * (i % 4 + 1));
                    for (int j = 0; j < QK8_0; ++j) {
                        q8[i].qs[j] = (int8_t) ((pattern + 13*j + i) % 256 - 128);
                    }
                }
                traits->to_float(w, x.data(), n);
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q8.data(), y.data(), n);
                const float ref = dot_product(x.data(), y.data(), n);
                float result = INFINITY;
                cpu->vec_dot(n, &result, 0, w, 0, q8.data(), 0, 1);
                const bool failed = result != ref;
                num_failed += failed;
                if (failed) {
                    printf("%6s packed dot nb=%d pattern=%d: FAILED (ref=%f got=%f)\n", ggml_type_name(type), nb, pattern, ref, result);
                }
            }
        }
    }
    if (num_failed || verbose) {
        printf("ternary packed dot products: %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
    }
    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();

    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_PQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_PTQ1_0  ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_Q1_0
                                          ? MAX_DOT_PRODUCT_ERROR_BINARY
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 ||
                                            type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : type == GGML_TYPE_NVFP4
                                          ? MAX_DOT_PRODUCT_ERROR_FP4
                                          : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    num_failed += test_ternary_lossless(verbose);
    num_failed += test_ternary_packed_dot(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
