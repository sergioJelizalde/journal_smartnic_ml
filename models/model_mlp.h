#ifndef MODEL_MLP_H
#define MODEL_MLP_H

#include <stdint.h>

/* Placeholder model. Replace this file with tools/export_mlp.py output. */
#define MLP_INPUT_DIM 16
#define MLP_NUM_CLASSES 2
#define MLP_NUM_LAYERS 2

static const uint16_t MLP_LAYER_SIZES[MLP_NUM_LAYERS + 1] = {16, 16, 2};

static const float MLP_FEATURE_MEAN[MLP_INPUT_DIM] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
static const float MLP_FEATURE_STD[MLP_INPUT_DIM] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static const float MLP_W0[16 * 16] = {
    0.05f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0.05f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0.05f,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0.05f,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0.05f,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0.05f,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0.05f,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0.05f,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0.05f,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0.05f,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0.05f,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0.05f,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0.05f,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0.05f,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05f,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05f
};
static const float MLP_B0[16] = {0};
static const float MLP_W1[16 * 2] = {
    0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f,
    0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f,
    0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f,
    0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f, 0.01f,-0.01f
};
static const float MLP_B1[2] = {0, 0};

static const float * const MLP_WEIGHTS[MLP_NUM_LAYERS] = {MLP_W0, MLP_W1};
static const float * const MLP_BIASES[MLP_NUM_LAYERS] = {MLP_B0, MLP_B1};

#endif
