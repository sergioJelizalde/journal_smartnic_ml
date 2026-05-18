#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report


def c_array_float(name, arr, per_line=8):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        vals = ", ".join(f"{x:.9g}f" for x in flat[i:i+per_line])
        out.append("    " + vals + ("," if i + per_line < len(flat) else ""))
    out.append("};")
    return "\n".join(out)


def c_array_int(name, arr, ctype="int16_t", per_line=16):
    flat = np.asarray(arr).reshape(-1)
    out = [f"static const {ctype} {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        vals = ", ".join(str(int(x)) for x in flat[i:i+per_line])
        out.append("    " + vals + ("," if i + per_line < len(flat) else ""))
    out.append("};")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--trees", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--out", default="models/model_rf.h")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    features = args.features.split(",") if args.features else [c for c in df.columns if c != args.label]
    X = df[features].astype(np.float32).values
    le = LabelEncoder()
    y = le.fit_transform(df[args.label].values)
    classes = len(le.classes_)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=args.test_size, random_state=9, stratify=y)
    clf = RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth, n_jobs=-1, random_state=9)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(classification_report(yte, pred, target_names=[str(c) for c in le.classes_]))

    offsets = [0]
    left = []
    right = []
    feat = []
    thr = []
    value = []
    max_depth_seen = 0
    for est in clf.estimators_:
        t = est.tree_
        n = t.node_count
        offsets.append(offsets[-1] + n)
        left.extend(t.children_left.astype(np.int16).tolist())
        right.extend(t.children_right.astype(np.int16).tolist())
        feat.extend(t.feature.astype(np.int16).tolist())
        thr.extend(t.threshold.astype(np.float32).tolist())
        # class at leaf, -1 at internal node
        for i in range(n):
            if t.children_left[i] < 0 and t.children_right[i] < 0:
                value.append(int(np.argmax(t.value[i][0])))
            else:
                value.append(-1)
        # compute depth
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            max_depth_seen = max(max_depth_seen, d)
            if t.children_left[node] >= 0:
                stack.append((int(t.children_left[node]), d + 1))
                stack.append((int(t.children_right[node]), d + 1))

    lines = ["#ifndef MODEL_RF_H", "#define MODEL_RF_H", "#include <stdint.h>", ""]
    lines.append(f"#define RF_NUM_FEATURES {Xs.shape[1]}")
    lines.append(f"#define RF_NUM_CLASSES {classes}")
    lines.append(f"#define RF_NUM_TREES {len(clf.estimators_)}")
    lines.append(f"#define RF_MAX_DEPTH {max_depth_seen}")
    lines.append(f"#define RF_NUM_NODES {len(left)}")
    lines.append(c_array_float("RF_FEATURE_MEAN", scaler.mean_))
    lines.append(c_array_float("RF_FEATURE_STD", scaler.scale_))
    lines.append(c_array_int("RF_TREE_OFFSETS", offsets, "uint32_t"))
    lines.append(c_array_int("RF_LEFT", left, "int16_t"))
    lines.append(c_array_int("RF_RIGHT", right, "int16_t"))
    lines.append(c_array_int("RF_FEATURE", feat, "int16_t"))
    lines.append(c_array_float("RF_THRESHOLD", thr))
    lines.append(c_array_int("RF_VALUE", value, "int16_t"))
    lines.append("#endif")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}; nodes={len(left)}")


if __name__ == "__main__":
    main()
