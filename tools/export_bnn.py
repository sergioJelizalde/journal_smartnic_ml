#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryLinear(nn.Linear):
    def forward(self, x):
        wb = SignSTE.apply(self.weight)
        xb = SignSTE.apply(x)
        return F.linear(xb, wb, self.bias)


class BNN(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(BinaryLinear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = SignSTE.apply(x)
        return x


def c_array_float(name, arr, per_line=8):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        vals = ", ".join(f"{x:.9g}f" for x in flat[i:i+per_line])
        out.append("    " + vals + ("," if i + per_line < len(flat) else ""))
    out.append("};")
    return "\n".join(out)


def c_array_u64(name, arr, per_line=4):
    flat = np.asarray(arr, dtype=np.uint64).reshape(-1)
    out = [f"static const uint64_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        vals = ", ".join(f"0x{int(x):016x}ULL" for x in flat[i:i+per_line])
        out.append("    " + vals + ("," if i + per_line < len(flat) else ""))
    out.append("};")
    return "\n".join(out)


def pack_weights(W):
    # W shape [in, out] in C convention. Pack each output neuron's input signs.
    Wsign = (W >= 0).astype(np.uint8)
    in_dim, out_dim = Wsign.shape
    words = (in_dim + 63) // 64
    packed = np.zeros((out_dim, words), dtype=np.uint64)
    for o in range(out_dim):
        for i in range(in_dim):
            if Wsign[i, o]:
                packed[o, i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return packed.reshape(-1), words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--hidden", default="64,32")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/model_bnn.h")
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
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=args.test_size, random_state=11, stratify=y)
    hidden = [int(x) for x in args.hidden.split(",") if x]
    dims = [Xs.shape[1]] + hidden + [1 if classes == 2 else classes]
    model = BNN(dims)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if classes == 2:
        loss_fn = nn.BCEWithLogitsLoss()
        ytr_t = torch.tensor(ytr.astype(np.float32)).view(-1, 1)
    else:
        loss_fn = nn.CrossEntropyLoss()
        ytr_t = torch.tensor(ytr.astype(np.int64))
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Xtr), ytr_t), batch_size=args.batch_size, shuffle=True)
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
        pred = (1/(1+np.exp(-logits.reshape(-1))) >= 0.5).astype(int)
    else:
        pred = logits.argmax(axis=1)
    print(classification_report(yte, pred, target_names=[str(c) for c in le.classes_]))

    packed = []
    biases = []
    words = []
    for layer in model.layers:
        W = layer.weight.detach().numpy().T.astype(np.float32)
        B = layer.bias.detach().numpy().astype(np.float32)
        P, w = pack_weights(W)
        packed.append(P)
        biases.append(B)
        words.append(w)

    lines = ["#ifndef MODEL_BNN_H", "#define MODEL_BNN_H", "#include <stdint.h>", ""]
    lines.append(f"#define BNN_INPUT_DIM {Xs.shape[1]}")
    lines.append(f"#define BNN_NUM_CLASSES {classes}")
    lines.append(f"#define BNN_NUM_LAYERS {len(packed)}")
    lines.append(f"#define BNN_MAX_WORDS_PER_OUT {max(words)}")
    lines.append("static const uint16_t BNN_LAYER_SIZES[BNN_NUM_LAYERS + 1] = {" + ", ".join(map(str, dims)) + "};")
    lines.append("static const uint16_t BNN_WORDS_PER_OUT[BNN_NUM_LAYERS] = {" + ", ".join(map(str, words)) + "};")
    lines.append(c_array_float("BNN_FEATURE_MEAN", scaler.mean_))
    lines.append(c_array_float("BNN_FEATURE_STD", scaler.scale_))
    for i, (P, B) in enumerate(zip(packed, biases)):
        lines.append(c_array_u64(f"BNN_W{i}", P))
        lines.append(c_array_float(f"BNN_B{i}", B))
    lines.append("static const uint64_t * const BNN_WEIGHTS[BNN_NUM_LAYERS] = {" + ", ".join(f"BNN_W{i}" for i in range(len(packed))) + "};")
    lines.append("static const float * const BNN_BIASES[BNN_NUM_LAYERS] = {" + ", ".join(f"BNN_B{i}" for i in range(len(biases))) + "};")
    lines.append("#endif")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
