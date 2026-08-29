/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Random-forest inference for per-flow anomaly scoring, running entirely on the DPA.
 *
 * Same tree-ensemble shape as flexio/pcc's ml_cc model (see device/rp/ml_cc/algo/ml_cc_model.h):
 * a handful of integer compares per tree, no FPU, no libc, so it stays cheap per packet even
 * without any hardware CC assist. Every tracked packet produces a feature vector (see flow_track.h
 * for how it's built); each tree walks from its root comparing one raw feature against an integer
 * threshold until it reaches a leaf. Leaf values are an unsigned fxp16 anomaly score in [0, 1<<16]
 * (0.0 .. 1.0); the forest output is the average leaf value across all trees.
 *
 * The weights below are a small placeholder forest (flags a size outlier on a flow's very first
 * packet) so the app runs out of the box. Train a real model offline against the same feature
 * definitions (FLOW_FEAT_*) and export it with tools/export_forest.py to regenerate this table.
 */

#ifndef FLOW_ML_MODEL_H_
#define FLOW_ML_MODEL_H_

#include <stdint.h>

/* Feature indices of the vector passed to flow_ml_forest_infer() */
enum {
	FLOW_FEAT_PKT_SIZE = 0,	  /* current packet size, in bytes */
	FLOW_FEAT_PKT_COUNT = 1,  /* packets seen on this flow so far, saturating at 255 */
	FLOW_FEAT_EWMA_SIZE = 2,  /* EWMA-smoothed packet size for this flow, fxp16 */
	FLOW_FEAT_SIZE_DELTA = 3, /* |current size - previous size| on this flow, in bytes */
	FLOW_ML_NUM_FEATURES
};

#define FLOW_ML_LEAF_MARKER (0xFFU) /* feature_idx value marking a leaf node */
#define FLOW_ML_NUM_TREES (4)
#define FLOW_ML_NODES_PER_TREE (7) /* full binary tree, depth 2: 2^3 - 1 nodes */
#define FLOW_ML_ANOMALY_THRESHOLD_FXP16 (1 << 15) /* 0.5: forest output above this => anomalous */

typedef struct {
	uint8_t feature_idx; /* FLOW_ML_LEAF_MARKER => this is a leaf */
	uint8_t left_child;  /* internal node: index of the left child */
	uint8_t right_child; /* internal node: index of the right child */
	uint8_t reserved;
	int32_t value; /* internal node: split threshold (feature < value -> left)
			  leaf node: fxp16 anomaly score, 0..(1<<16) */
} flow_ml_tree_node_t;

/* clang-format off */
static const flow_ml_tree_node_t flow_ml_forest[FLOW_ML_NUM_TREES][FLOW_ML_NODES_PER_TREE] = {
	/* PLACEHOLDER, replace with trained weights: flags a large packet arriving as the very
	 * first packet on a brand-new flow (no history yet to justify the burst). */
	{
		{FLOW_FEAT_PKT_COUNT, 1, 2, 0, 1},
		{FLOW_FEAT_PKT_SIZE, 3, 4, 0, 1200},
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 3277},	/* 0.05: established flow, low weight */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 6554},	/* 0.10: first packet, small */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 60000}, /* ~0.92: first packet, large */
	},
	{
		{FLOW_FEAT_SIZE_DELTA, 1, 2, 0, 800},
		{FLOW_FEAT_EWMA_SIZE, 3, 4, 0, (200 << 16)},
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 6554},	/* 0.10: small delta */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 13107},	/* 0.20: large delta, big established mean */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 39321},	/* 0.60: large delta, small established mean */
	},
	{
		{FLOW_FEAT_PKT_SIZE, 1, 2, 0, 1500},
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 9830},	/* 0.15: normal-sized packet */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 32768},	/* 0.50: jumbo-sized packet */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 0}, {FLOW_ML_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{FLOW_FEAT_PKT_COUNT, 1, 2, 0, 3},
		{FLOW_FEAT_SIZE_DELTA, 3, 4, 0, 400},
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 13107},	/* 0.20: still ramping up, moderate delta */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 6554},	/* 0.10: established flow, small delta */
		{FLOW_ML_LEAF_MARKER, 0, 0, 0, 26214},	/* 0.40: established flow, large delta */
	},
};
/* clang-format on */

/*
 * Walk a single tree to its leaf and return the leaf value.
 * Bounded by FLOW_ML_NODES_PER_TREE so a malformed/corrupt table can never spin an EU forever.
 */
static inline int32_t flow_ml_tree_infer(const flow_ml_tree_node_t *tree, const int32_t *feat)
{
	uint8_t idx = 0;

	for (uint8_t step = 0; step < FLOW_ML_NODES_PER_TREE; step++) {
		const flow_ml_tree_node_t *node = &tree[idx];

		if (node->feature_idx == FLOW_ML_LEAF_MARKER)
			return node->value;

		idx = (feat[node->feature_idx] < node->value) ? node->left_child : node->right_child;
	}
	return 0; /* Defensive fallback: neutral (non-anomalous) vote */
}

/*
 * Run the full forest on one feature vector.
 * Returns the averaged leaf vote: an unsigned fxp16 anomaly score in [0, 1<<16].
 */
static inline int32_t flow_ml_forest_infer(const int32_t *feat)
{
	int32_t sum = 0;

	for (uint32_t t = 0; t < FLOW_ML_NUM_TREES; t++)
		sum += flow_ml_tree_infer(flow_ml_forest[t], feat);

	return sum / (int32_t)FLOW_ML_NUM_TREES;
}

#endif /* FLOW_ML_MODEL_H_ */
