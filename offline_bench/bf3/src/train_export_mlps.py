#!/usr/bin/env python3
"""
train_export_mlps.py

Trains the 5 MLP architectures on dummy data (16 inputs, 4 classes) and
exports each one as a C header usable by:
  - the scalar reference kernel
  - the NEON kernel (BlueField-3 / ARM A78)
  - the AVX2 kernel (x86 host)
  - the XNNPACK subgraph (needs weights in [out,in] layout, provided too)

Later, swap `make_dummy_data()` for your real dataset loader -- keep the
same (X, y) shapes: X is (N, 16) float32, y is (N,) int in [0,3].

Usage:
    pip install torch --break-system-packages   # if not already installed
    python3 train_export_mlps.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

NUM_FEATURES = 16
NUM_CLASSES  = 4
OUT_DIR      = Path("weights")
OUT_DIR.mkdir(exist_ok=True)

# name -> hidden layer sizes (matches the ids you gave)
ARCHS = {
    "16_8":        [16, 8],
    "32_16":       [32, 16],
    "64_32":       [64, 32],
    "128_64_16":   [128, 64, 16],
    "256_128_32":  [256, 128, 32],
}

SEED = 42
N_SAMPLES = 4000
EPOCHS = 30
BATCH = 128


def make_dummy_data(n=N_SAMPLES, seed=SEED):
    """Synthetic but learnable 16-feature / 4-class dataset.
    Replace this with your real feature extractor's output later."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, NUM_FEATURES)).astype(np.float32)
    # random linear projection -> logits -> labels, plus noise
    W_true = rng.normal(0, 1, size=(NUM_FEATURES, NUM_CLASSES)).astype(np.float32)
    logits = X @ W_true + rng.normal(0, 0.5, size=(n, NUM_CLASSES)).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


class MLP(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        dims = [NUM_FEATURES] + hidden_dims + [NUM_CLASSES]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.dims = dims

    def forward(self, x):
        return self.net(x)  # raw logits; softmax applied at inference in C


def train_one(hidden_dims, X, y, mean, std):
    Xn = (X - mean) / std
    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y)

    model = MLP(hidden_dims)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    n = Xt.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch == EPOCHS - 1:
            acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  final loss={total_loss/n:.4f} train_acc={acc:.3f}")
    return model


def fmt_array(arr):
    return ", ".join(f"{v:.6f}" for v in arr.flatten())


def export_header(name, model, out_path):
    """Writes mlp_<name>.h with:
       - LAYER_SIZES / NUM_LAYERS
       - W{L}/B{L} in [in,out] layout (k*size_out+j) for scalar & NEON
       - WX{L}/BX{L} in [out,in] layout (PyTorch native) for XNNPACK
    """
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    dims = model.dims
    num_layers = len(linears)

    lines = []
    lines.append(f"// Auto-generated MLP weights for '{name}' (16 in -> 4 out, multiclass)")
    lines.append("#pragma once")
    lines.append('#define ALIGN16 __attribute__((aligned(16)))')
    lines.append("")
    lines.append(f"#define NUM_LAYERS {num_layers}")
    lines.append(f"#define IS_MULTICLASS_CLASSIFICATION 1")
    lines.append(f"#define IS_BINARY_CLASSIFICATION 0")
    lines.append("")
    dims_str = ", ".join(str(d) for d in dims)
    lines.append(f"static const int LAYER_SIZES[NUM_LAYERS+1] = {{ {dims_str} }};")
    lines.append("")

    w_names, b_names = [], []
    wx_names, bx_names = [], []
    for L, lin in enumerate(linears):
        W = lin.weight.detach().numpy()   # shape [out, in]  (PyTorch native)
        B = lin.bias.detach().numpy()     # shape [out]
        size_in, size_out = W.shape[1], W.shape[0]

        # ---- scalar/NEON layout: [in, out] row-major -> W[k*size_out+j] ----
        W_io = W.T.copy()  # shape [in, out]
        lines.append(f"// Layer {L}: {size_in} -> {size_out}  (scalar/NEON layout: in x out)")
        lines.append(f"static const float W{L}[{size_in*size_out}] ALIGN16 = {{")
        lines.append(f"    {fmt_array(W_io)}")
        lines.append("};")
        lines.append(f"static const float B{L}[{size_out}] ALIGN16 = {{")
        lines.append(f"    {fmt_array(B)}")
        lines.append("};")
        lines.append("")
        w_names.append(f"W{L}")
        b_names.append(f"B{L}")

        # ---- XNNPACK layout: [out, in] row-major (native nn.Linear layout) ----
        lines.append(f"// Layer {L}: XNNPACK layout (out x in)")
        lines.append(f"static const float WX{L}[{size_in*size_out}] ALIGN16 = {{")
        lines.append(f"    {fmt_array(W)}")
        lines.append("};")
        lines.append(f"static const float BX{L}[{size_out}] ALIGN16 = {{")
        lines.append(f"    {fmt_array(B)}")
        lines.append("};")
        lines.append("")
        wx_names.append(f"WX{L}")
        bx_names.append(f"BX{L}")

    lines.append(f"static const float *WEIGHTS[NUM_LAYERS] = {{ {', '.join(w_names)} }};")
    lines.append(f"static const float *BIASES[NUM_LAYERS]  = {{ {', '.join(b_names)} }};")
    lines.append(f"static const float *WEIGHTS_XNN[NUM_LAYERS] = {{ {', '.join(wx_names)} }};")
    lines.append(f"static const float *BIASES_XNN[NUM_LAYERS]  = {{ {', '.join(bx_names)} }};")
    lines.append("")
    lines.append("#undef ALIGN16")
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def export_feature_stats(mean, std, out_path):
    lines = []
    lines.append("// Auto-generated feature normalization stats")
    lines.append("#pragma once")
    lines.append(f"#define NUM_FEATURES {NUM_FEATURES}")
    lines.append(f"static const float FEATURE_MEAN[NUM_FEATURES] = {{ {fmt_array(mean)} }};")
    lines.append(f"static const float FEATURE_STD[NUM_FEATURES]  = {{ {fmt_array(np.where(std==0,1,std))} }};")
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    torch.manual_seed(SEED)
    X, y = make_dummy_data()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    export_feature_stats(mean, std, OUT_DIR / "feature_stats.h")

    for name, hidden in ARCHS.items():
        print(f"Training MLP {name} (hidden={hidden}) ...")
        model = train_one(hidden, X, y, mean, std)
        export_header(name, model, OUT_DIR / f"mlp_{name}.h")

    print("\nDone. Headers are in ./weights/")
    print("Copy feature_stats.h and the mlp_<name>.h you want next to mlp_bench.c / mlp_correctness.c")


if __name__ == "__main__":
    main()
