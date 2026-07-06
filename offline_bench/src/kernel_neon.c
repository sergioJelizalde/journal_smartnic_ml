/* kernel_neon.c
 * ARM NEON (128-bit, 4x float32) kernel for the BlueField-3 ARM A78 cores.
 * NOTE: this targets the ARM cores specifically -- NOT the DPA RISC-V
 * cores (those have no NEON and need your separate RC4ML/BenchBF3-style
 * scalar/vector path).
 *
 * Cleaned up from the uploaded draft:
 *  - dropped the unused sigmoid_neon() (final layer is always scalar --
 *    see kernel_scalar.c rationale, output width is 1..8, not worth
 *    vectorizing and it was dead code in the original file)
 *  - kept the 4-wide main loop + scalar tail; since every hidden layer
 *    width here is a multiple of 8 (hence also a multiple of 4), the tail
 *    loop never actually executes for hidden layers -- it only ever fires
 *    on the small output layer, where it's just the natural scalar path.
 */
#include <arm_neon.h>
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
    for (; i + 4 <= NUM_FEATURES; i += 4) {
        float32x4_t v = vld1q_f32(&in[i]);
        float32x4_t m = vld1q_f32(&FEATURE_MEAN[i]);
        float32x4_t s = vld1q_f32(&FEATURE_STD[i]);
        vst1q_f32(&out[i], vdivq_f32(vsubq_f32(v, m), s));
    }
    for (; i < NUM_FEATURES; i++)
        out[i] = (in[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
}

/* matrix-vector forward for a layer, NEON-optimized.
 * W is size_in x size_out row-major (k * size_out + j)
 */
static void layer_forward_neon(const float *W, const float *B,
                                const float *in, float *out,
                                int size_in, int size_out, int is_output) {
    int j = 0;
    for (; j + 4 <= size_out; j += 4) {
        float32x4_t acc = vld1q_f32(&B[j]);
        for (int k = 0; k < size_in; k++) {
            acc = vfmaq_f32(acc, vdupq_n_f32(in[k]), vld1q_f32(&W[k * size_out + j]));
        }
        if (!is_output) acc = vmaxq_f32(acc, vdupq_n_f32(0.0f)); /* relu */
        vst1q_f32(&out[j], acc);
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
        layer_forward_neon(WEIGHTS[L], BIASES[L], in_buf, out_buf,
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
