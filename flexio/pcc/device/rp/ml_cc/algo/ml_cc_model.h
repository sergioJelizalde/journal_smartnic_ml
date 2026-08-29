/*
 * Copyright (c) 2026 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ML_CC_MODEL_H_
#define ML_CC_MODEL_H_

#include "utils.h"

/*
 * Random-forest inference for the congestion-control rate decision.
 *
 * A decision-tree ensemble is used instead of a neural net because DPA execution units run one
 * event at a time with a tight cycle budget and no FPU: tree traversal is a handful of integer
 * compares and branches per tree, with no multiply-accumulate work, so it stays cheap even as the
 * app is spread across many EUs (many cores) each driving its own hardware queue pair.
 *
 * Every RTT sample produces a feature vector; each tree walks from its root comparing one raw
 * (unscaled) feature against an integer threshold until it reaches a leaf. Leaf values are a
 * signed fxp16 rate-multiplier delta (0 == hold, positive == increase, negative == decrease).
 * The forest output is the average leaf value across all trees, applied multiplicatively to the
 * flow's current rate.
 *
 * The weights below are a small placeholder forest that mimics an AIMD-like policy so the app
 * runs out of the box. Train a real model offline (e.g. sklearn RandomForestRegressor, target =
 * a multiplicative rate-correction ratio, features in the same units as ML_FEAT_*) and export it
 * with tools/export_forest.py to regenerate this table.
 */

/* Feature indices of the vector passed to ml_cc_forest_infer() */
enum {
	ML_FEAT_RELATIVE_RTT = 0, /* rtt - min_rtt, in nanoseconds */
	ML_FEAT_EWMA_RTT = 1,	  /* EWMA-smoothed RTT, in nanoseconds */
	ML_FEAT_CUR_RATE = 2,	  /* current rate, fxp20 */
	ML_FEAT_WAS_CNP = 3,	  /* 0 or 1 */
	ML_FEAT_WAS_NACK = 4,	  /* 0 or 1 */
	ML_FEAT_CNP_STREAK = 5,	  /* saturating consecutive-CNP count, 0..255 */
	ML_CC_NUM_FEATURES
};

#define ML_CC_LEAF_MARKER (0xFFU) /* feature_idx value marking a leaf node */
#define ML_CC_NUM_TREES (8)
#define ML_CC_NODES_PER_TREE (15) /* full binary tree, depth 3: 2^4 - 1 nodes */

typedef struct {
	uint8_t feature_idx;  /* ML_CC_LEAF_MARKER => this is a leaf */
	uint8_t left_child;   /* internal node: index of the left child */
	uint8_t right_child;  /* internal node: index of the right child */
	uint8_t reserved;
	int32_t value;	      /* internal node: split threshold (feature < value -> left)
				 leaf node: fxp16 rate-multiplier delta */
} ml_tree_node_t;

/* clang-format off */
static const ml_tree_node_t ml_cc_forest[ML_CC_NUM_TREES][ML_CC_NODES_PER_TREE] = {
	/* Trees 0-3: primarily split on relative RTT against the base-delay style thresholds,
	 * secondary split on CNP/NACK flags -- PLACEHOLDER, replace with trained weights. */
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 13000},
		{ML_FEAT_WAS_NACK, 3, 4, 0, 1},
		{ML_FEAT_WAS_CNP, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 3277},   /* +0.05x: low delay, no NACK -> increase */
		{ML_CC_LEAF_MARKER, 0, 0, 0, -19661}, /* -0.30x: low delay but NACK'd */
		{ML_CC_LEAF_MARKER, 0, 0, 0, -13107}, /* -0.20x: high delay, CNP seen */
		{ML_CC_LEAF_MARKER, 0, 0, 0, -6554},  /* -0.10x: high delay, no CNP */
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_EWMA_RTT, 1, 2, 0, 20000},
		{ML_FEAT_CNP_STREAK, 3, 4, 0, 1},
		{ML_FEAT_WAS_NACK, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 3277},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -9830},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -6554},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -19661},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 8000},
		{ML_FEAT_CUR_RATE, 3, 4, 0, (1 << 17)},
		{ML_FEAT_WAS_CNP, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 6554},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 3277},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -13107},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -6554},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 30000},
		{ML_FEAT_WAS_NACK, 3, 4, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -26214},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 1638},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -19661},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	/* Trees 4-7: mirror trees 0-3 with slightly perturbed thresholds/leaves, standing in for
	 * the bagged-sample diversity a real training run would produce. */
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 12000},
		{ML_FEAT_WAS_NACK, 3, 4, 0, 1},
		{ML_FEAT_WAS_CNP, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 2949},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -18000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -12000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -5500},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_EWMA_RTT, 1, 2, 0, 18000},
		{ML_FEAT_CNP_STREAK, 3, 4, 0, 1},
		{ML_FEAT_WAS_NACK, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 3000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -9000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -6000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -18000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 9000},
		{ML_FEAT_CUR_RATE, 3, 4, 0, (1 << 17)},
		{ML_FEAT_WAS_CNP, 5, 6, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 6000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 3000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -12000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -6000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
	{
		{ML_FEAT_RELATIVE_RTT, 1, 2, 0, 28000},
		{ML_FEAT_WAS_NACK, 3, 4, 0, 1},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -24000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 1500},
		{ML_CC_LEAF_MARKER, 0, 0, 0, -18000},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
		{ML_CC_LEAF_MARKER, 0, 0, 0, 0}, {ML_CC_LEAF_MARKER, 0, 0, 0, 0},
	},
};
/* clang-format on */

/*
 * Walk a single tree to its leaf and return the leaf value.
 * Bounded by ML_CC_NODES_PER_TREE so a malformed/corrupt table can never spin an EU forever.
 */
FORCE_INLINE int32_t ml_cc_tree_infer(const ml_tree_node_t *tree, const int32_t *feat)
{
	uint8_t idx = 0;

	for (uint8_t step = 0; step < ML_CC_NODES_PER_TREE; step++) {
		const ml_tree_node_t *node = &tree[idx];

		if (node->feature_idx == ML_CC_LEAF_MARKER)
			return node->value;

		idx = (feat[node->feature_idx] < node->value) ? node->left_child : node->right_child;
	}
	return 0; /* Defensive fallback: neutral vote */
}

/*
 * Run the full forest on one feature vector.
 * Returns the averaged leaf vote: a signed fxp16 rate-multiplier delta.
 */
FORCE_INLINE int32_t ml_cc_forest_infer(const int32_t *feat)
{
	int32_t sum = 0;

	for (uint32_t t = 0; t < ML_CC_NUM_TREES; t++)
		sum += ml_cc_tree_infer(ml_cc_forest[t], feat);

	return sum / (int32_t)ML_CC_NUM_TREES;
}

#endif /* ML_CC_MODEL_H_ */
