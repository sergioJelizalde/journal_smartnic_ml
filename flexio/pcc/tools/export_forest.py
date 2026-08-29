#!/usr/bin/env python3
"""
Export a trained scikit-learn RandomForestRegressor into the ml_tree_node_t table consumed by
device/rp/ml_cc/algo/ml_cc_model.h.

Train the forest so that:
  - Features, in order, are:
      0: relative_rtt  (rtt - min_rtt, nanoseconds)
      1: ewma_rtt      (EWMA-smoothed RTT, nanoseconds)
      2: cur_rate      (current rate, fxp20 raw units)
      3: was_cnp       (0 or 1)
      4: was_nack      (0 or 1)
      5: cnp_streak    (0..255)
  - The regression target is a multiplicative rate-correction ratio minus 1.0, e.g. -0.10 means
    "cut the rate by 10%", +0.05 means "raise the rate by 5%", 0.0 means "hold". Clip your
    training targets to roughly [-1.0, 1.0] since the device clamps to that range anyway.
  - max_depth <= 3 (so each tree fits in ML_CC_NODES_PER_TREE = 15 nodes) and n_estimators
    matches ML_CC_NUM_TREES in ml_cc_model.h (8 by default). Pass --nodes-per-tree /
    --num-trees if you changed those constants on the device side.

Usage:
    python export_forest.py model.joblib -o ml_cc_model_generated.h

The output header only redefines `ml_cc_forest`; splice its body into ml_cc_model.h in place of
the placeholder table (the surrounding enum/struct/inference-function code stays as-is).
"""
import argparse
import pickle
import sys

LEAF_MARKER = "ML_CC_LEAF_MARKER"
FXP16_ONE = 1 << 16


def load_model(path):
    if path.endswith(".joblib"):
        import joblib

        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_tree(tree, nodes_per_tree):
    """Convert one sklearn DecisionTreeRegressor's internal tree_ into a flat, fixed-size,
    breadth-first-compatible node list matching ml_tree_node_t (feature_idx/left/right/value)."""
    t = tree.tree_
    n_nodes = t.node_count
    if n_nodes > nodes_per_tree:
        raise ValueError(
            f"tree has {n_nodes} nodes, exceeds --nodes-per-tree={nodes_per_tree}; "
            "retrain with a shallower max_depth or raise ML_CC_NODES_PER_TREE on the device"
        )

    nodes = []
    for i in range(n_nodes):
        left, right = t.children_left[i], t.children_right[i]
        if left == -1 and right == -1:  # leaf
            leaf_value = float(t.value[i].reshape(-1)[0])
            fxp_value = int(round(leaf_value * FXP16_ONE))
            nodes.append((LEAF_MARKER, 0, 0, fxp_value))
        else:
            feature = int(t.feature[i])
            threshold = int(round(t.threshold[i]))
            nodes.append((str(feature), int(left), int(right), threshold))

    # Pad to a fixed size with inert, unreachable leaves so the C array dimension is constant.
    while len(nodes) < nodes_per_tree:
        nodes.append((LEAF_MARKER, 0, 0, 0))
    return nodes


def render_header(forest_nodes, num_trees, nodes_per_tree):
    lines = []
    lines.append(f"static const ml_tree_node_t ml_cc_forest[{num_trees}][{nodes_per_tree}] = {{")
    for tree_idx, nodes in enumerate(forest_nodes):
        lines.append(f"\t{{ /* tree {tree_idx} */")
        for feature_idx, left, right, value in nodes:
            lines.append(f"\t\t{{{feature_idx}, {left}, {right}, 0, {value}}},")
        lines.append("\t},")
    lines.append("};")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Path to a pickled/joblib sklearn RandomForestRegressor")
    ap.add_argument("-o", "--output", default="-", help="Output header path (default: stdout)")
    ap.add_argument("--num-trees", type=int, default=8, help="Must match ML_CC_NUM_TREES on the device")
    ap.add_argument("--nodes-per-tree", type=int, default=15, help="Must match ML_CC_NODES_PER_TREE on the device")
    args = ap.parse_args()

    model = load_model(args.model)
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        sys.exit("Error: expected a fitted sklearn RandomForestRegressor (no estimators_ found)")
    if len(estimators) != args.num_trees:
        sys.exit(
            f"Error: model has {len(estimators)} trees, but --num-trees={args.num_trees} "
            "(must match ML_CC_NUM_TREES in ml_cc_model.h)"
        )

    forest_nodes = [flatten_tree(est, args.nodes_per_tree) for est in estimators]
    header = render_header(forest_nodes, args.num_trees, args.nodes_per_tree)

    if args.output == "-":
        sys.stdout.write(header)
    else:
        with open(args.output, "w") as f:
            f.write(header)


if __name__ == "__main__":
    main()
