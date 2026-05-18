#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>
#include "app_config.h"
#include "common.h"

struct model_runtime;

typedef int (*model_predict_fn)(const struct model_runtime *m,
                                const float *x,
                                void *scratch);

struct model_runtime {
    enum app_model_type type;
    enum app_kernel_type kernel;
    uint16_t input_dim;
    uint16_t num_classes;
    const float *mean;
    const float *std;
    size_t scratch_bytes;
    model_predict_fn predict;
};

int model_runtime_init(struct model_runtime *m,
                       enum app_model_type type,
                       enum app_kernel_type kernel);
int model_prepare_input(const struct model_runtime *m,
                        const struct feature_vector *raw,
                        float *scaled_out,
                        bool allow_pad);
static inline int model_predict(const struct model_runtime *m,
                                const float *scaled,
                                void *scratch)
{
    return m->predict(m, scaled, scratch);
}

#endif
