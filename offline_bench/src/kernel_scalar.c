/* kernel_scalar.c
 * Portable reference implementation -- no intrinsics, compiles anywhere.
 * This is the ground truth the NEON/AVX kernels are checked against.
 */
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
    for (int i = 0; i < NUM_FEATURES; i++)
        out[i] = (in[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
}

/* size_out is always a multiple of 8 for hidden layers (16,32,64,128,256);
 * the output layer (1..8) just runs the same scalar loop, no special-casing
 * needed since this kernel never vectorizes anyway. */
static void layer_forward_scalar(const float *W, const float *B,
                                  const float *in, float *out,
                                  int size_in, int size_out, int is_output) {
    for (int j = 0; j < size_out; j++) {
        float acc = B[j];
        for (int k = 0; k < size_in; k++)
            acc += W[k * size_out + j] * in[k];
        if (!is_output) acc = (acc > 0.0f) ? acc : 0.0f; /* relu */
        out[j] = acc;
    }
}

int predict_mlp(const float *in_features, float *buf_a, float *buf_b) {
    float norm[NUM_FEATURES];
    normalize_features(in_features, norm);

    float *in_buf = buf_a, *out_buf = buf_b;
    memcpy(in_buf, norm, LAYER_SIZES[0] * sizeof(float));

    for (int L = 0; L < NUM_LAYERS; L++) {
        int is_output_layer = (L == NUM_LAYERS - 1);
        layer_forward_scalar(WEIGHTS[L], BIASES[L], in_buf, out_buf,
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
