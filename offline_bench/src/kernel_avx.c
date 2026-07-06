/* kernel_avx.c
 * x86 AVX2+FMA (256-bit, 8x float32) kernel -- direct structural port of
 * kernel_neon.c, op-for-op, so the two are easy to diff/compare:
 *   vld1q_f32/vst1q_f32   -> _mm256_loadu_ps/_mm256_storeu_ps
 *   vdupq_n_f32           -> _mm256_set1_ps
 *   vfmaq_f32(acc,a,b)    -> _mm256_fmadd_ps(a,b,acc)   (FMA operand order differs)
 *   vmaxq_f32             -> _mm256_max_ps
 *   vaddq_f32/vsubq_f32/vdivq_f32 -> _mm256_add_ps/_mm256_sub_ps/_mm256_div_ps
 *
 * Because every hidden layer width here (16,32,64,128,256) is a multiple
 * of 8, this loop never needs a tail for hidden layers -- exactly the
 * "assume multiples of 8" contract requested. Only the output layer
 * (1..8 wide) falls through to the scalar tail, same as the NEON kernel.
 * Compile with -mavx2 -mfma.
 */
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "mlp_kernel.h"

#include MODEL_HEADER_FILE
#include FEATURE_HEADER_FILE

static inline float sigmoid_piece(float x) {
    if (x <= -4.0f) return 0.0f;
    else if (x <= -2.0f) return 0.0625f * x + 0.25f;
    else if (x <= 0.0f)  return 0.125f * x + 0.5f;
    else if (x <= 2.0f)  return -0.125f * x + 0.5f;
    else if (x <= 4.0f)  return -0.0625f * x + 0.75f;
    else return 1.0f;
}

void normalize_features(const float *in, float *out) {
    int i = 0;
    for (; i + 8 <= NUM_FEATURES; i += 8) {
        __m256 v = _mm256_loadu_ps(&in[i]);
        __m256 m = _mm256_loadu_ps(&FEATURE_MEAN[i]);
        __m256 s = _mm256_loadu_ps(&FEATURE_STD[i]);
        _mm256_storeu_ps(&out[i], _mm256_div_ps(_mm256_sub_ps(v, m), s));
    }
    for (; i < NUM_FEATURES; i++)
        out[i] = (in[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
}

static void layer_forward_avx(const float *W, const float *B,
                               const float *in, float *out,
                               int size_in, int size_out, int is_output) {
    int j = 0;
    for (; j + 8 <= size_out; j += 8) {
        __m256 acc = _mm256_loadu_ps(&B[j]);
        for (int k = 0; k < size_in; k++) {
            __m256 in_k = _mm256_set1_ps(in[k]);
            __m256 w = _mm256_loadu_ps(&W[k * size_out + j]);
            acc = _mm256_fmadd_ps(in_k, w, acc);
        }
        if (!is_output) acc = _mm256_max_ps(acc, _mm256_setzero_ps()); /* relu */
        _mm256_storeu_ps(&out[j], acc);
    }
    for (; j < size_out; j++) {
        float a = B[j];
        for (int k = 0; k < size_in; k++)
            a += W[k * size_out + j] * in[k];
        if (!is_output) a = (a > 0.0f) ? a : 0.0f;
        out[j] = a;
    }
}

int predict_mlp(const float *in_features, float *buf_a, float *buf_b) {
    float norm[NUM_FEATURES];
    normalize_features(in_features, norm);

    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, norm, LAYER_SIZES[0] * sizeof(float));

    for (int L = 0; L < NUM_LAYERS; L++) {
        int is_output_layer = (L == NUM_LAYERS - 1);
        layer_forward_avx(WEIGHTS[L], BIASES[L], in_buf, out_buf,
                           LAYER_SIZES[L], LAYER_SIZES[L + 1], is_output_layer);
        if (is_output_layer) {
#if IS_BINARY_CLASSIFICATION
            for (int i = 0; i < LAYER_SIZES[L + 1]; i++)
                out_buf[i] = sigmoid_piece(out_buf[i]);
#elif IS_MULTICLASS_CLASSIFICATION
            float max_val = out_buf[0];
            for (int i = 1; i < LAYER_SIZES[L + 1]; i++)
                if (out_buf[i] > max_val) max_val = out_buf[i];
            float sum = 0.0f;
            for (int i = 0; i < LAYER_SIZES[L + 1]; i++) {
                out_buf[i] = expf(out_buf[i] - max_val);
                sum += out_buf[i];
            }
            for (int i = 0; i < LAYER_SIZES[L + 1]; i++)
                out_buf[i] /= sum;
#endif
        }
        float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
    }

    int final_size = LAYER_SIZES[NUM_LAYERS];
#if IS_BINARY_CLASSIFICATION
    (void)final_size;
    return (in_buf[0] >= 0.5f) ? 1 : 0;
#elif IS_MULTICLASS_CLASSIFICATION
    int best_class = 0;
    float best_prob = in_buf[0];
    for (int i = 1; i < final_size; i++)
        if (in_buf[i] > best_prob) { best_prob = in_buf[i]; best_class = i; }
    return best_class;
#endif
}
