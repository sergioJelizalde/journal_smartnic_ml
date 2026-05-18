#include <string.h>
#include <math.h>
#include "inference.h"
#include "../models/model_mlp.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define APP_HAS_NEON 1
#else
#define APP_HAS_NEON 0
#endif

static inline float fast_sigmoid(float x)
{
    /* Monotonic, branch-light approximation. Exact sigmoid is not needed for argmax. */
    return 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;
}

static void mlp_layer_scalar(const float *W, const float *B,
                             const float *in, float *out,
                             int size_in, int size_out,
                             int apply_relu)
{
    for (int j = 0; j < size_out; j++) {
        float acc = B[j];
        for (int k = 0; k < size_in; k++) {
            acc += W[(size_t)k * size_out + j] * in[k];
        }
        out[j] = apply_relu && acc < 0.0f ? 0.0f : acc;
    }
}

#if APP_HAS_NEON
static void mlp_layer_neon(const float *W, const float *B,
                           const float *in, float *out,
                           int size_in, int size_out,
                           int apply_relu)
{
    int j = 0;
    for (; j + 4 <= size_out; j += 4) {
        float32x4_t acc = vld1q_f32(&B[j]);
        for (int k = 0; k < size_in; k++) {
            float32x4_t w = vld1q_f32(&W[(size_t)k * size_out + j]);
            acc = vfmaq_f32(acc, vdupq_n_f32(in[k]), w);
        }
        if (apply_relu) acc = vmaxq_f32(acc, vdupq_n_f32(0.0f));
        vst1q_f32(&out[j], acc);
    }
    /* Scalar tail: handles any layer size, not only multiples of 4. */
    for (; j < size_out; j++) {
        float acc = B[j];
        for (int k = 0; k < size_in; k++) {
            acc += W[(size_t)k * size_out + j] * in[k];
        }
        out[j] = apply_relu && acc < 0.0f ? 0.0f : acc;
    }
}
#endif

int mlp_predict_entry(const struct model_runtime *m, const float *x, void *scratch)
{
    (void)m;
    float *buf_a = (float *)scratch;
    float *buf_b = buf_a + MLP_LAYER_SIZES[0];

    /* The scratch split above is conservative only if max hidden <= input.
     * Use an offset equal to max neurons instead. */
    uint16_t maxn = 0;
    for (int i = 0; i <= MLP_NUM_LAYERS; i++) if (MLP_LAYER_SIZES[i] > maxn) maxn = MLP_LAYER_SIZES[i];
    buf_b = buf_a + maxn;

    memcpy(buf_a, x, (size_t)MLP_LAYER_SIZES[0] * sizeof(float));
    float *in = buf_a;
    float *out = buf_b;

    for (int l = 0; l < MLP_NUM_LAYERS; l++) {
        int is_output = (l == MLP_NUM_LAYERS - 1);
        int size_in = MLP_LAYER_SIZES[l];
        int size_out = MLP_LAYER_SIZES[l + 1];
#if APP_HAS_NEON
        if (m->kernel == APP_KERNEL_NEON) {
            mlp_layer_neon(MLP_WEIGHTS[l], MLP_BIASES[l], in, out,
                           size_in, size_out, !is_output);
        } else
#endif
        {
            mlp_layer_scalar(MLP_WEIGHTS[l], MLP_BIASES[l], in, out,
                             size_in, size_out, !is_output);
        }
        float *tmp = in;
        in = out;
        out = tmp;
    }

    int final = MLP_LAYER_SIZES[MLP_NUM_LAYERS];
    if (final == 1) {
        float prob = fast_sigmoid(in[0]);
        return prob >= 0.5f ? 1 : 0;
    }

    /* For multiclass, argmax of logits is equal to argmax of softmax. */
    int best = 0;
    float best_v = in[0];
    for (int i = 1; i < final; i++) {
        if (in[i] > best_v) {
            best_v = in[i];
            best = i;
        }
    }
    return best;
}
