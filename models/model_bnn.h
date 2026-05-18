#ifndef MODEL_BNN_H
#define MODEL_BNN_H

#include <stdint.h>

/* Placeholder BNN. Replace with tools/export_bnn.py output. */
#define BNN_INPUT_DIM 16
#define BNN_NUM_CLASSES 2
#define BNN_NUM_LAYERS 2
#define BNN_MAX_WORDS_PER_OUT 1

static const uint16_t BNN_LAYER_SIZES[BNN_NUM_LAYERS + 1] = {16, 16, 2};
static const uint16_t BNN_WORDS_PER_OUT[BNN_NUM_LAYERS] = {1, 1};

static const float BNN_FEATURE_MEAN[BNN_INPUT_DIM] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
static const float BNN_FEATURE_STD[BNN_INPUT_DIM] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static const uint64_t BNN_W0[16 * 1] = {
    0xffffULL,0xffffULL,0xffffULL,0xffffULL,
    0xffffULL,0xffffULL,0xffffULL,0xffffULL,
    0xffffULL,0xffffULL,0xffffULL,0xffffULL,
    0xffffULL,0xffffULL,0xffffULL,0xffffULL
};
static const float BNN_B0[16] = {0};
static const uint64_t BNN_W1[2 * 1] = {0x0000ffffULL, 0xffff0000ULL};
static const float BNN_B1[2] = {0, 0};

static const uint64_t * const BNN_WEIGHTS[BNN_NUM_LAYERS] = {BNN_W0, BNN_W1};
static const float * const BNN_BIASES[BNN_NUM_LAYERS] = {BNN_B0, BNN_B1};

#endif
