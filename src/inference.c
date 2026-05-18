#include <string.h>
#include <stdio.h>
#include "inference.h"
#include "../models/model_mlp.h"
#include "../models/model_rf.h"
#include "../models/model_bnn.h"

int mlp_predict_entry(const struct model_runtime *m, const float *x, void *scratch);
int rf_predict_entry(const struct model_runtime *m, const float *x, void *scratch);
int bnn_predict_entry(const struct model_runtime *m, const float *x, void *scratch);
size_t mlp_scratch_bytes(void);
size_t bnn_scratch_bytes(void);

static uint16_t mlp_max_neurons(void)
{
    uint16_t maxn = 0;
    for (int i = 0; i <= MLP_NUM_LAYERS; i++) {
        if (MLP_LAYER_SIZES[i] > maxn) maxn = MLP_LAYER_SIZES[i];
    }
    return maxn;
}

int model_runtime_init(struct model_runtime *m,
                       enum app_model_type type,
                       enum app_kernel_type kernel)
{
    memset(m, 0, sizeof(*m));
    m->type = type;
    m->kernel = kernel;

    switch (type) {
    case APP_MODEL_MLP:
        m->input_dim = MLP_INPUT_DIM;
        m->num_classes = MLP_NUM_CLASSES;
        m->mean = MLP_FEATURE_MEAN;
        m->std = MLP_FEATURE_STD;
        m->scratch_bytes = (size_t)2 * mlp_max_neurons() * sizeof(float);
        m->predict = mlp_predict_entry;
        return 0;
    case APP_MODEL_RF:
        m->input_dim = RF_NUM_FEATURES;
        m->num_classes = RF_NUM_CLASSES;
        m->mean = RF_FEATURE_MEAN;
        m->std = RF_FEATURE_STD;
        m->scratch_bytes = 0;
        m->predict = rf_predict_entry;
        return 0;
    case APP_MODEL_BNN:
        m->input_dim = BNN_INPUT_DIM;
        m->num_classes = BNN_NUM_CLASSES;
        m->mean = BNN_FEATURE_MEAN;
        m->std = BNN_FEATURE_STD;
        m->scratch_bytes = bnn_scratch_bytes();
        m->predict = bnn_predict_entry;
        return 0;
    default:
        return -1;
    }
}

int model_prepare_input(const struct model_runtime *m,
                        const struct feature_vector *raw,
                        float *scaled_out,
                        bool allow_pad)
{
    if (!m || !raw || !scaled_out) return -1;
    if (raw->n != m->input_dim && (!allow_pad || raw->n > m->input_dim)) {
        return -1;
    }
    for (uint16_t i = 0; i < m->input_dim; i++) {
        float x = (i < raw->n) ? raw->v[i] : m->mean[i];
        float s = m->std[i];
        scaled_out[i] = (x - m->mean[i]) / ((s > 0.0f) ? s : 1.0f);
    }
    return 0;
}
