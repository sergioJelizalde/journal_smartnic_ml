/*
 * mlp_correctness.c — verifies all kernels agree on the predicted class.
 *
 * Build on the BF3 (ARM, compares scalar vs NEON [vs XNNPACK]):
 *   gcc -O3 -march=armv8-a+simd -DHAVE_NEON -DMODEL_HEADER='"mlp_64_32.h"' \
 *       mlp_correctness.c -o correctness_bf3 -lm
 *
 * Build on the x86 host (compares scalar vs AVX2 [vs XNNPACK]):
 *   gcc -O3 -mavx2 -mfma -DHAVE_AVX -DMODEL_HEADER='"mlp_64_32.h"' \
 *       mlp_correctness.c -o correctness_x86 -lm
 *
 * Add -DHAVE_XNNPACK plus the XNNPACK include/lib flags (see mlp_bench.c)
 * on either platform to also check XNNPACK against the reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(HAVE_NEON)
#include <arm_neon.h>
#endif
#if defined(HAVE_AVX)
#include <immintrin.h>
#endif
#if defined(HAVE_XNNPACK)
#include <xnnpack.h>
#endif

#ifndef MODEL_HEADER
#error "Define MODEL_HEADER, e.g. -DMODEL_HEADER='\"mlp_64_32.h\"'"
#endif
#include "feature_stats.h"
#include MODEL_HEADER

#ifndef NUM_TESTS
#define NUM_TESTS 20000
#endif

static inline float randomf(void) { return (float)rand() / (float)RAND_MAX; }

/* ---- scalar reference ---- */
static int predict_scalar(const float *in_features, float *buf_a, float *buf_b) {
    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, in_features, LAYER_SIZES[0] * sizeof(float));
    for (int L = 0; L < NUM_LAYERS; L++) {
        int is_output = (L == NUM_LAYERS - 1);
        int size_in = LAYER_SIZES[L], size_out = LAYER_SIZES[L + 1];
        for (int j = 0; j < size_out; j++) {
            float acc = BIASES[L][j];
            for (int k = 0; k < size_in; k++)
                acc += WEIGHTS[L][k * size_out + j] * in_buf[k];
            out_buf[j] = is_output ? acc : (acc > 0.0f ? acc : 0.0f);
        }
        float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }
    int final_size = LAYER_SIZES[NUM_LAYERS], best = 0;
    float best_v = in_buf[0];
    for (int i = 1; i < final_size; i++)
        if (in_buf[i] > best_v) { best_v = in_buf[i]; best = i; }
    return best;
}

#if defined(HAVE_NEON)
static void layer_forward_neon(const float *W, const float *B, const float *in,
                                float *out, int size_in, int size_out, int is_output) {
    int j = 0;
    for (; j + 4 <= size_out; j += 4) {
        float32x4_t acc = vld1q_f32(&B[j]);
        for (int k = 0; k < size_in; k++)
            acc = vfmaq_f32(acc, vdupq_n_f32(in[k]), vld1q_f32(&W[k * size_out + j]));
        if (!is_output) acc = vmaxq_f32(acc, vdupq_n_f32(0.0f));
        vst1q_f32(&out[j], acc);
    }
    for (; j < size_out; j++) {
        float a = B[j];
        for (int k = 0; k < size_in; k++) a += W[k * size_out + j] * in[k];
        out[j] = is_output ? a : (a > 0.0f ? a : 0.0f);
    }
}
static int predict_neon(const float *in_features, float *buf_a, float *buf_b) {
    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, in_features, LAYER_SIZES[0] * sizeof(float));
    for (int L = 0; L < NUM_LAYERS; L++) {
        layer_forward_neon(WEIGHTS[L], BIASES[L], in_buf, out_buf,
                            LAYER_SIZES[L], LAYER_SIZES[L + 1], (L == NUM_LAYERS - 1));
        float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }
    int final_size = LAYER_SIZES[NUM_LAYERS], best = 0;
    float best_v = in_buf[0];
    for (int i = 1; i < final_size; i++)
        if (in_buf[i] > best_v) { best_v = in_buf[i]; best = i; }
    return best;
}
#endif

#if defined(HAVE_AVX)
static void layer_forward_avx2(const float *W, const float *B, const float *in,
                                float *out, int size_in, int size_out, int is_output) {
    int j = 0;
    for (; j + 8 <= size_out; j += 8) {
        __m256 acc = _mm256_loadu_ps(&B[j]);
        for (int k = 0; k < size_in; k++) {
            __m256 wv = _mm256_loadu_ps(&W[k * size_out + j]);
            __m256 ib = _mm256_set1_ps(in[k]);
            acc = _mm256_fmadd_ps(ib, wv, acc);
        }
        if (!is_output) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
        _mm256_storeu_ps(&out[j], acc);
    }
    for (; j + 4 <= size_out; j += 4) {
        __m128 acc = _mm_loadu_ps(&B[j]);
        for (int k = 0; k < size_in; k++) {
            __m128 wv = _mm_loadu_ps(&W[k * size_out + j]);
            __m128 ib = _mm_set1_ps(in[k]);
            acc = _mm_fmadd_ps(ib, wv, acc);
        }
        if (!is_output) acc = _mm_max_ps(acc, _mm_setzero_ps());
        _mm_storeu_ps(&out[j], acc);
    }
    for (; j < size_out; j++) {
        float a = B[j];
        for (int k = 0; k < size_in; k++) a += W[k * size_out + j] * in[k];
        out[j] = is_output ? a : (a > 0.0f ? a : 0.0f);
    }
}
static int predict_avx(const float *in_features, float *buf_a, float *buf_b) {
    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, in_features, LAYER_SIZES[0] * sizeof(float));
    for (int L = 0; L < NUM_LAYERS; L++) {
        layer_forward_avx2(WEIGHTS[L], BIASES[L], in_buf, out_buf,
                            LAYER_SIZES[L], LAYER_SIZES[L + 1], (L == NUM_LAYERS - 1));
        float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }
    int final_size = LAYER_SIZES[NUM_LAYERS], best = 0;
    float best_v = in_buf[0];
    for (int i = 1; i < final_size; i++)
        if (in_buf[i] > best_v) { best_v = in_buf[i]; best = i; }
    return best;
}
#endif

#if defined(HAVE_XNNPACK)
static xnn_subgraph_t g_subgraph;
static xnn_runtime_t  g_runtime;
static uint32_t g_input_id, g_output_id;
static float *g_input_buf, *g_output_buf;

static void build_xnnpack_runtime(void) {
    xnn_initialize(NULL);
    xnn_create_subgraph(2, 0, &g_subgraph);
    uint32_t cur_id, next_id;
    size_t dims_in[2] = {1, (size_t)LAYER_SIZES[0]};
    xnn_define_tensor_value(g_subgraph, xnn_datatype_fp32, 2, dims_in, NULL, 0,
                             XNN_VALUE_FLAG_EXTERNAL_INPUT, &g_input_id);
    cur_id = g_input_id;
    for (int L = 0; L < NUM_LAYERS; L++) {
        int size_in = LAYER_SIZES[L], size_out = LAYER_SIZES[L + 1];
        int is_output = (L == NUM_LAYERS - 1);
        size_t w_dims[2] = {(size_t)size_out, (size_t)size_in};
        uint32_t w_id;
        xnn_define_tensor_value(g_subgraph, xnn_datatype_fp32, 2, w_dims,
                                 WEIGHTS_XNN[L], XNN_INVALID_VALUE_ID, 0, &w_id);
        size_t b_dims[1] = {(size_t)size_out};
        uint32_t b_id;
        xnn_define_tensor_value(g_subgraph, xnn_datatype_fp32, 1, b_dims,
                                 BIASES_XNN[L], XNN_INVALID_VALUE_ID, 0, &b_id);
        size_t out_dims[2] = {1, (size_t)size_out};
        uint32_t flags = is_output ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
        uint32_t out_ext_id = is_output ? 1 : XNN_INVALID_VALUE_ID;
        xnn_define_tensor_value(g_subgraph, xnn_datatype_fp32, 2, out_dims, NULL,
                                 out_ext_id, flags, &next_id);
        float out_min = is_output ? -INFINITY : 0.0f;
        xnn_define_fully_connected(g_subgraph, out_min, INFINITY, cur_id, w_id, b_id, next_id, 0);
        if (is_output) g_output_id = next_id;
        cur_id = next_id;
    }
    pthreadpool_t tp = pthreadpool_create(1);
    xnn_create_runtime_v2(g_subgraph, tp, 0, &g_runtime);
    posix_memalign((void**)&g_input_buf, 16, LAYER_SIZES[0] * sizeof(float));
    posix_memalign((void**)&g_output_buf, 16, LAYER_SIZES[NUM_LAYERS] * sizeof(float));
}
static int predict_xnnpack(const float *in_features) {
    memcpy(g_input_buf, in_features, LAYER_SIZES[0] * sizeof(float));
    struct xnn_external_value ext[2] = {{0, g_input_buf}, {1, g_output_buf}};
    xnn_setup_runtime(g_runtime, 2, ext);
    xnn_invoke_runtime(g_runtime);
    int final_size = LAYER_SIZES[NUM_LAYERS], best = 0;
    float best_v = g_output_buf[0];
    for (int i = 1; i < final_size; i++)
        if (g_output_buf[i] > best_v) { best_v = g_output_buf[i]; best = i; }
    return best;
}
#endif

int main(void) {
    srand(777);
    int max_neurons = 0;
    for (int i = 0; i <= NUM_LAYERS; i++)
        if (LAYER_SIZES[i] > max_neurons) max_neurons = LAYER_SIZES[i];

    float *scratch_a, *scratch_b, *raw_input, *input;
    posix_memalign((void**)&scratch_a, 16, max_neurons * sizeof(float));
    posix_memalign((void**)&scratch_b, 16, max_neurons * sizeof(float));
    posix_memalign((void**)&raw_input, 16, LAYER_SIZES[0] * sizeof(float));
    posix_memalign((void**)&input,     16, LAYER_SIZES[0] * sizeof(float));

#if defined(HAVE_XNNPACK)
    build_xnnpack_runtime();
#endif

    int mismatches_simd = 0, mismatches_xnn = 0;
    for (int it = 0; it < NUM_TESTS; it++) {
        for (int k = 0; k < LAYER_SIZES[0]; k++) raw_input[k] = randomf();
        for (int k = 0; k < LAYER_SIZES[0]; k++)
            input[k] = (raw_input[k] - FEATURE_MEAN[k]) / FEATURE_STD[k];

        int ref = predict_scalar(input, scratch_a, scratch_b);

#if defined(HAVE_NEON)
        int simd = predict_neon(input, scratch_a, scratch_b);
        const char *simd_name = "NEON";
#elif defined(HAVE_AVX)
        int simd = predict_avx(input, scratch_a, scratch_b);
        const char *simd_name = "AVX2";
#else
        int simd = ref;
        const char *simd_name = "(none)";
#endif
        if (simd != ref) {
            mismatches_simd++;
            printf("[%s mismatch] test %d: scalar=%d %s=%d\n", simd_name, it, ref, simd_name, simd);
        }

#if defined(HAVE_XNNPACK)
        int xnn = predict_xnnpack(input);
        if (xnn != ref) {
            mismatches_xnn++;
            printf("[XNNPACK mismatch] test %d: scalar=%d xnnpack=%d\n", it, ref, xnn);
        }
#endif
    }

    printf("\n=== Correctness summary (%d tests) ===\n", NUM_TESTS);
#if defined(HAVE_NEON)
    printf("scalar vs NEON:    %d/%d match\n", NUM_TESTS - mismatches_simd, NUM_TESTS);
#elif defined(HAVE_AVX)
    printf("scalar vs AVX2:    %d/%d match\n", NUM_TESTS - mismatches_simd, NUM_TESTS);
#endif
#if defined(HAVE_XNNPACK)
    printf("scalar vs XNNPACK: %d/%d match\n", NUM_TESTS - mismatches_xnn, NUM_TESTS);
#endif
    return (mismatches_simd || mismatches_xnn) ? 1 : 0;
}