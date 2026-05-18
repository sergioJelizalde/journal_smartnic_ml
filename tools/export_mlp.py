#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--features", default=None, help="comma-separated feature names; default all columns except label")
    ap.add_argument("--hidden", default="32,16")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/model_mlp.h")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    features = args.features.split(",") if args.features else [c for c in df.columns if c != args.label]
    X = df[features].astype(np.float32).values
    le = LabelEncoder()
    y = le.fit_transform(df[args.label].values)
    classes = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=7, stratify=y)

    hidden = [int(x) for x in args.hidden.split(",") if x]
    dims = [X.shape[1]] + hidden + [1 if classes == 2 else classes]
    model = MLP(dims)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if classes == 2:
        loss_fn = nn.BCEWithLogitsLoss()
        ytr_t = torch.tensor(ytr.astype(np.float32)).view(-1, 1)
    else:
        loss_fn = nn.CrossEntropyLoss()
        ytr_t = torch.tensor(ytr.astype(np.int64))
    Xtr_t = torch.tensor(Xtr)

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte)).numpy()
    if classes == 2:
        pred = (1.0 / (1.0 + np.exp(-logits.reshape(-1))) >= 0.5).astype(int)
    else:
        pred = logits.argmax(axis=1)
    print(classification_report(yte, pred, target_names=[str(c) for c in le.classes_]))

    weights = []
    biases = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy().T.astype(np.float32)  # C expects [in][out]
            B = layer.bias.detach().numpy().astype(np.float32)
            weights.append(W)
            biases.append(B)

    lines = ["#ifndef MODEL_MLP_H", "#define MODEL_MLP_H", "#include <stdint.h>", ""]
    lines.append(f"#define MLP_INPUT_DIM {X.shape[1]}")
    lines.append(f"#define MLP_NUM_CLASSES {classes}")
    lines.append(f"#define MLP_NUM_LAYERS {len(weights)}")
    lines.append("static const uint16_t MLP_LAYER_SIZES[MLP_NUM_LAYERS + 1] = {" + ", ".join(map(str, dims)) + "};")
    lines.append(c_array_float("MLP_FEATURE_MEAN", scaler.mean_))
    lines.append(c_array_float("MLP_FEATURE_STD", scaler.scale_))
    for i, (W, B) in enumerate(zip(weights, biases)):
        lines.append(c_array_float(f"MLP_W{i}", W))
        lines.append(c_array_float(f"MLP_B{i}", B))
    lines.append("static const float * const MLP_WEIGHTS[MLP_NUM_LAYERS] = {" + ", ".join(f"MLP_W{i}" for i in range(len(weights))) + "};")
    lines.append("static const float * const MLP_BIASES[MLP_NUM_LAYERS] = {" + ", ".join(f"MLP_B{i}" for i in range(len(biases))) + "};")
    lines.append("#endif")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
