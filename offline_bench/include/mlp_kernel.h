#pragma once
#include <stdint.h>

/* Every kernel (scalar.c / neon.c / avx.c) implements this exact signature,
 * and includes MODEL_HEADER_FILE + FEATURE_HEADER_FILE to get its
 * LAYER_SIZES / WEIGHTS / BIASES / FEATURE_MEAN / FEATURE_STD.
 *
 * in_features: raw (unscaled) feature vector, length NUM_FEATURES
 * buf_a/buf_b: scratch ping-pong buffers, each >= max(LAYER_SIZES)
 * returns: predicted class index (0/1 for binary, 0..C-1 for multiclass)
 */
int predict_mlp(const float *in_features, float *buf_a, float *buf_b);

/* z-score normalize in place: out[i] = (in[i] - mean[i]) / std[i] */
void normalize_features(const float *in, float *out);
