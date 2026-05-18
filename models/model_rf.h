#ifndef MODEL_RF_H
#define MODEL_RF_H

#include <stdint.h>

/* Placeholder RF. Replace with tools/export_rf.py output. */
#define RF_NUM_FEATURES 16
#define RF_NUM_CLASSES 2
#define RF_NUM_TREES 1
#define RF_MAX_DEPTH 2
#define RF_NUM_NODES 3

static const float RF_FEATURE_MEAN[RF_NUM_FEATURES] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
static const float RF_FEATURE_STD[RF_NUM_FEATURES] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};
static const uint32_t RF_TREE_OFFSETS[RF_NUM_TREES + 1] = {0, 3};
static const int16_t RF_LEFT[RF_NUM_NODES] = {1, -1, -1};
static const int16_t RF_RIGHT[RF_NUM_NODES] = {2, -1, -1};
static const int16_t RF_FEATURE[RF_NUM_NODES] = {0, -1, -1};
static const float RF_THRESHOLD[RF_NUM_NODES] = {0.0f, 0.0f, 0.0f};
static const int16_t RF_VALUE[RF_NUM_NODES] = {-1, 0, 1};

#endif
