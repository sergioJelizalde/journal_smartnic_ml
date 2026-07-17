#!/usr/bin/env python3
"""
train_export_hardcoded.py -- train an MLP on flow-feature CSV data (PyTorch)
and emit a WEIGHT-HARDCODED, shape-specialized NEON microkernel for the
BlueField-3 (Cortex-A78AE).

Differences vs gen_neon_mlp.py (which packs at runtime from WEIGHTS[]):

  * Weights/biases are baked into the header as `static const float` literals,
    ALREADY PACKED in [K][NR] tile order at generation time. No init call,
    no runtime packing, arrays live in .rodata (shared, demand-paged, and the
    compiler knows their alignment + address at link time).
  * Feature standardization (x - mean) / std is FOLDED into layer 0:
        W0f[k][n] = W0[k][n] / std[k]
        b0f[n]    = b0[n] - sum_k W0[k][n] * mean[k] / std[k]
    so the raw-feature entry point skips the normalization loop entirely.
  * Two entry points are emitted:
        int predict_mlp_hardcoded(const float *normalized_in);   // bench-compatible
        int predict_mlp_hardcoded_raw(const float *raw_in);      // folded norm
    plus *_logits variants for correctness checking.
  * All constants are emitted as C99 hex floats -> bit-exact round trip
    from the trained float32 model.

Outputs (default names, see -o/--prefix):
  mlp_hardcoded.h    the microkernel (self-contained, includes ref scalar path)
  mlp_flow_model.h   classic model header (WEIGHTS/BIASES/WEIGHTS_XNN/...) so the
                     SAME model runs through mlp_bench.c scalar/NEON/XNNPACK
  feature_stats.h    FEATURE_MEAN / FEATURE_STD for mlp_bench.c

Usage:
  # train from CSV (last column = label by default) and export everything
  python3 train_export_hardcoded.py --csv flows.csv --hidden 64 32 \
      --epochs 60 --backend torch

  # re-export from previously saved weights (skip training)
  python3 train_export_hardcoded.py --from-npz model.npz

CSV format: numeric feature columns + one label column (--label-col NAME or
last column). Labels may be strings; they are label-encoded and the mapping
is emitted as a comment + HC_CLASS_NAMES[].
"""
import argparse
import sys
import numpy as np

FULL_UNROLL_K = 128
K_LOOP_UNROLL = 8

# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def load_csv(path, label_col=None):
    import csv as csvmod
    with open(path, newline="") as f:
        reader = csvmod.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    if label_col is None:
        li = len(header) - 1
    else:
        li = header.index(label_col)
    feat_names = [h for i, h in enumerate(header) if i != li]
    X, y = [], []
    for r in rows:
        y.append(r[li])
        X.append([float(r[i]) for i in range(len(header)) if i != li])
    X = np.asarray(X, dtype=np.float64)
    classes = sorted(set(y))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cmap[v] for v in y], dtype=np.int64)
    return X, y, feat_names, classes


def train_torch(X, y, hidden, epochs, lr, batch, seed, val_frac=0.15):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X)
    idx = np.random.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, ti = idx[:nv], idx[nv:]
    mean = X[ti].mean(axis=0)
    std = X[ti].std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std

    sizes = [X.shape[1]] + list(hidden) + [int(y.max()) + 1]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)

    Xt = torch.tensor(Xn[ti], dtype=torch.float32)
    yt = torch.tensor(y[ti])
    Xv = torch.tensor(Xn[vi], dtype=torch.float32)
    yv = torch.tensor(y[vi])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()
            print(f"epoch {ep:3d}  val_acc {acc:.4f}")

    W, B = [], []
    for m in model:
        if hasattr(m, "weight"):
            # nn.Linear stores [out, in]; kernel wants [in][out] row-major
            W.append(m.weight.detach().numpy().T.astype(np.float32).copy())
            B.append(m.bias.detach().numpy().astype(np.float32).copy())
    return W, B, mean.astype(np.float32), std.astype(np.float32)


def train_numpy(X, y, hidden, epochs, lr, batch, seed, val_frac=0.15):
    """Dependency-free fallback (Adam, ReLU MLP) for smoke tests / CI."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    vi, ti = idx[:nv], idx[nv:]
    mean = X[ti].mean(axis=0)
    std = X[ti].std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    sizes = [X.shape[1]] + list(hidden) + [int(y.max()) + 1]
    W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
    B = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
    mB = [np.zeros_like(b) for b in B]; vB = [np.zeros_like(b) for b in B]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0

    def forward(x):
        acts = [x]
        for i in range(len(W)):
            z = acts[-1] @ W[i] + B[i]
            acts.append(np.maximum(z, 0) if i < len(W) - 1 else z)
        return acts

    for ep in range(epochs):
        order = rng.permutation(len(ti))
        for s in range(0, len(order), batch):
            bidx = ti[order[s:s + batch]]
            xb, yb = Xn[bidx], y[bidx]
            acts = forward(xb)
            z = acts[-1]
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            g = p; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            t += 1
            for i in reversed(range(len(W))):
                gW = acts[i].T @ g
                gB = g.sum(axis=0)
                if i > 0:
                    g = (g @ W[i].T) * (acts[i] > 0)
                for buf, gm, gv, gr in ((W, mW, vW, gW), (B, mB, vB, gB)):
                    gm[i] = b1 * gm[i] + (1 - b1) * gr
                    gv[i] = b2 * gv[i] + (1 - b2) * gr * gr
                    mh = gm[i] / (1 - b1 ** t); vh = gv[i] / (1 - b2 ** t)
                    buf[i] -= lr * mh / (np.sqrt(vh) + eps)
        if ep % 10 == 0 or ep == epochs - 1:
            pred = forward(Xn[vi])[-1].argmax(axis=1)
            print(f"epoch {ep:3d}  val_acc {(pred == y[vi]).mean():.4f}")
    return ([w.astype(np.float32) for w in W], [b.astype(np.float32) for b in B],
            mean.astype(np.float32), std.astype(np.float32))

# --------------------------------------------------------------------------
# codegen helpers
# --------------------------------------------------------------------------

def fhex(x):
    """Exact C99 hex-float literal for a float32 value."""
    f = float(np.float32(x))
    if f == 0.0:
        return "-0.0f" if np.signbit(np.float32(x)) else "0.0f"
    return f.hex() + "f"


def emit_const_array(name, vals, per_line=4):
    out = [f"static const float {name}[{len(vals)}] __attribute__((aligned(64))) = {{"]
    line = []
    for v in vals:
        line.append(fhex(v))
        if len(line) == per_line:
            out.append("    " + ", ".join(line) + ",")
            line = []
    if line:
        out.append("    " + ", ".join(line) + ",")
    out.append("};")
    return "\n".join(out)


def block_plan(N):
    plan, j = [], 0
    for nr in (16, 8, 4):
        while N - j >= nr:
            plan.append((j, nr))
            j += nr
    return plan


def pack_weights(W):
    """Pack [K][N] weight matrix into concatenated [K][NR] tiles (gen-time)."""
    K, N = W.shape
    tiles = []
    for (j0, nr) in block_plan(N):
        tiles.append(np.ascontiguousarray(W[:, j0:j0 + nr]).reshape(-1))
    return np.concatenate(tiles) if tiles else np.zeros(0, np.float32)


def emit_wide_block(K, nr, j0, off, relu, wname, bname, indent="    "):
    nregs = nr // 4
    a = lambda bank, r: f"acc{bank}_{r}"
    lines = []
    for r in range(nregs):
        lines.append(f"float32x4_t {a('e', r)} = vld1q_f32(&{bname}[{j0 + 4*r}]);")
    for r in range(nregs):
        lines.append(f"float32x4_t {a('o', r)} = vdupq_n_f32(0.0f);")
    lines.append(f"const float *w = {wname} + {off};")

    body = []
    K4 = K - (K % 4)
    if K <= FULL_UNROLL_K:
        for k in range(0, K4, 4):
            body.append(f"{{ float32x4_t xv = vld1q_f32(in + {k});")
            for lane in range(4):
                bank = 'e' if lane % 2 == 0 else 'o'
                for r in range(nregs):
                    body.append(f"{a(bank, r)} = vfmaq_laneq_f32({a(bank, r)}, "
                                f"vld1q_f32(w + {k + lane}*{nr} + {4*r}), xv, {lane});")
            body.append("}")
    else:
        body.append(f"for (int k = 0; k < {K4}; k += {K_LOOP_UNROLL}) {{")
        for u in range(0, K_LOOP_UNROLL, 4):
            body.append(f"  float32x4_t xv{u} = vld1q_f32(in + k + {u});")
            for lane in range(4):
                bank = 'e' if lane % 2 == 0 else 'o'
                for r in range(nregs):
                    body.append(f"  {a(bank, r)} = vfmaq_laneq_f32({a(bank, r)}, "
                                f"vld1q_f32(w + (k + {u + lane})*{nr} + {4*r}), xv{u}, {lane});")
        body.append("}")
    for k in range(K4, K):
        bank = 'e' if k % 2 == 0 else 'o'
        body.append(f"{{ float32x4_t xb = vdupq_n_f32(in[{k}]);")
        for r in range(nregs):
            body.append(f"  {a(bank, r)} = vfmaq_f32({a(bank, r)}, "
                        f"vld1q_f32(w + {k}*{nr} + {4*r}), xb);")
        body.append("}")
    lines += body
    for r in range(nregs):
        lines.append(f"{a('e', r)} = vaddq_f32({a('e', r)}, {a('o', r)});")
        if relu:
            lines.append(f"{a('e', r)} = vmaxq_f32({a('e', r)}, vdupq_n_f32(0.0f));")
        lines.append(f"vst1q_f32(out + {j0 + 4*r}, {a('e', r)});")
    return [indent + l for l in lines]


def emit_dot_outputs(K, N, j_start, relu, wref, bname, indent="    "):
    lines = []
    K4 = K - (K % 8) if K >= 8 else 0
    for j in range(j_start, N):
        lines.append(f"{{ /* output {j} */")
        lines.append("  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);")
        if K4:
            lines.append(f"  for (int k = 0; k < {K4}; k += 8) {{")
            lines.append("    float32x4_t x0 = vld1q_f32(in + k);")
            lines.append(f"    float32x4_t w0 = (float32x4_t){{{wref}[(k+0)*{N}+{j}], {wref}[(k+1)*{N}+{j}], {wref}[(k+2)*{N}+{j}], {wref}[(k+3)*{N}+{j}]}};")
            lines.append("    s0 = vfmaq_f32(s0, x0, w0);")
            lines.append("    float32x4_t x1 = vld1q_f32(in + k + 4);")
            lines.append(f"    float32x4_t w1 = (float32x4_t){{{wref}[(k+4)*{N}+{j}], {wref}[(k+5)*{N}+{j}], {wref}[(k+6)*{N}+{j}], {wref}[(k+7)*{N}+{j}]}};")
            lines.append("    s1 = vfmaq_f32(s1, x1, w1);")
            lines.append("  }")
        lines.append(f"  float acc = {bname}[{j}] + vaddvq_f32(vaddq_f32(s0, s1));")
        for k in range(K4, K):
            lines.append(f"  acc += {wref}[{k}*{N}+{j}] * in[{k}];")
        if relu:
            lines.append(f"  out[{j}] = acc > 0.0f ? acc : 0.0f;")
        else:
            lines.append(f"  out[{j}] = acc;")
        lines.append("}")
    return [indent + l for l in lines]


def emit_layer_fn(fname, K, N, relu, wpack, wref, bname):
    lines = [f"static inline void {fname}(const float *restrict in, float *restrict out) {{"]
    off = 0
    plan = block_plan(N)
    for (j0, nr) in plan:
        lines.append(f"    {{ /* block j={j0}..{j0+nr-1} (NR={nr}) */")
        lines += emit_wide_block(K, nr, j0, off, relu, wpack, bname)
        lines.append("    }")
        off += K * nr
    covered = sum(nr for _, nr in plan)
    if covered < N:
        lines += emit_dot_outputs(K, N, covered, relu, wref, bname)
    lines.append("}")
    return "\n".join(lines)

# --------------------------------------------------------------------------
# header generation
# --------------------------------------------------------------------------

def gen_hardcoded_header(W, B, mean, std, class_names=None):
    sizes = [W[0].shape[0]] + [w.shape[1] for w in W]
    nlayers = len(W)
    nclasses = sizes[-1]
    bufmax = max(sizes)

    # folded layer 0 (normalization baked into weights/bias)
    W0f = (W[0] / std[:, None]).astype(np.float32)
    B0f = (B[0] - (W[0] * (mean / std)[:, None]).sum(axis=0)).astype(np.float32)

    p = []
    p.append("/* AUTO-GENERATED by train_export_hardcoded.py -- do not edit.")
    p.append(f" * Model shape: {' -> '.join(map(str, sizes))}  (weights hardcoded, packed at gen time)")
    if class_names:
        p.append(f" * Classes: {', '.join(f'{i}={c}' for i, c in enumerate(class_names))}")
    p.append(" * Self-contained: no model header, no init call required. */")
    p.append("#ifndef MLP_HARDCODED_H")
    p.append("#define MLP_HARDCODED_H")
    p.append("")
    p.append(f"#define HC_NUM_LAYERS {nlayers}")
    p.append(f"#define HC_NUM_CLASSES {nclasses}")
    p.append(f"#define HC_INPUT_DIM {sizes[0]}")
    p.append("static const int HC_LAYER_SIZES[] = {" + ", ".join(map(str, sizes)) + "};")
    if class_names:
        p.append("static const char *const HC_CLASS_NAMES[] = {" +
                 ", ".join(f'"{c}"' for c in class_names) + "};")
    p.append("")
    p.append("/* feature standardization stats (for the normalized entry point / ref path) */")
    p.append(emit_const_array("HC_FEATURE_MEAN", mean))
    p.append(emit_const_array("HC_FEATURE_STD", std))
    p.append("")
    p.append("/* ---- reference (unpacked) weights: correctness path + remainder outputs ---- */")
    for L in range(nlayers):
        p.append(emit_const_array(f"HC_W{L}_REF", W[L].reshape(-1)))
        p.append(emit_const_array(f"HC_B{L}", B[L]))
    p.append(emit_const_array("HC_W0F_REF", W0f.reshape(-1)))
    p.append(emit_const_array("HC_B0F", B0f))
    p.append("")
    p.append("/* ---- reference scalar inference (any ISA), for correctness checks ---- */")
    p.append("static void hc_reference_logits(const float *raw_in, float *logits) {")
    p.append(f"    float a[{bufmax}], b[{bufmax}];")
    p.append(f"    for (int k = 0; k < {sizes[0]}; k++)")
    p.append("        a[k] = (raw_in[k] - HC_FEATURE_MEAN[k]) / HC_FEATURE_STD[k];")
    p.append("    float *ib = a, *ob = b;")
    p.append("    const float *const WREF[] = {" + ", ".join(f"HC_W{L}_REF" for L in range(nlayers)) + "};")
    p.append("    const float *const BREF[] = {" + ", ".join(f"HC_B{L}" for L in range(nlayers)) + "};")
    p.append(f"    for (int L = 0; L < {nlayers}; L++) {{")
    p.append(f"        int si = HC_LAYER_SIZES[L], so = HC_LAYER_SIZES[L + 1], last = (L == {nlayers - 1});")
    p.append("        for (int j = 0; j < so; j++) {")
    p.append("            float acc = BREF[L][j];")
    p.append("            for (int k = 0; k < si; k++) acc += WREF[L][k * so + j] * ib[k];")
    p.append("            ob[j] = last ? acc : (acc > 0.0f ? acc : 0.0f);")
    p.append("        }")
    p.append("        float *t = ib; ib = ob; ob = t;")
    p.append("    }")
    p.append(f"    for (int i = 0; i < {nclasses}; i++) logits[i] = ib[i];")
    p.append("}")
    p.append("static int hc_reference_predict(const float *raw_in) {")
    p.append(f"    float lg[{nclasses}]; hc_reference_logits(raw_in, lg);")
    p.append(f"    int best = 0; for (int i = 1; i < {nclasses}; i++) if (lg[i] > lg[best]) best = i;")
    p.append("    return best;")
    p.append("}")
    p.append("")
    p.append("#if defined(__aarch64__) || defined(__ARM_NEON)")
    p.append("#include <arm_neon.h>")
    p.append("")
    p.append("/* ---- gen-time packed weight tiles ([K][NR] order, .rodata) ---- */")
    for L in range(nlayers):
        pk = pack_weights(W[L])
        if len(pk):
            p.append(emit_const_array(f"HC_W{L}_PACK", pk))
    pk0f = pack_weights(W0f)
    if len(pk0f):
        p.append(emit_const_array("HC_W0F_PACK", pk0f))
    p.append("")
    for L in range(nlayers):
        K, N = W[L].shape
        relu = (L != nlayers - 1)
        p.append(emit_layer_fn(f"hc_layer{L}_fwd", K, N, relu,
                               f"HC_W{L}_PACK", f"HC_W{L}_REF", f"HC_B{L}"))
        p.append("")
    K0, N0 = W0f.shape
    p.append("/* layer 0 with (x-mean)/std folded in: takes RAW features */")
    p.append(emit_layer_fn("hc_layer0f_fwd", K0, N0, nlayers > 1,
                           "HC_W0F_PACK", "HC_W0F_REF", "HC_B0F"))
    p.append("")

    def emit_predict(name, first_layer):
        q = [f"static void {name}_logits(const float *restrict in_features, float *restrict logits) {{"]
        q.append(f"    float b0[{bufmax}] __attribute__((aligned(64)));")
        q.append(f"    float b1[{bufmax}] __attribute__((aligned(64)));")
        cur_in, cur_out = "in_features", "b0"
        q.append(f"    {first_layer}({cur_in}, {cur_out});")
        cur_in = cur_out; cur_out = "b1"
        for L in range(1, nlayers):
            q.append(f"    hc_layer{L}_fwd({cur_in}, {cur_out});")
            cur_in, cur_out = cur_out, ("b1" if cur_out == "b0" else "b0")
        q.append(f"    for (int i = 0; i < {nclasses}; i++) logits[i] = {cur_in}[i];")
        q.append("}")
        q.append(f"static int {name}(const float *restrict in_features) {{")
        q.append(f"    float lg[{nclasses}];")
        q.append(f"    {name}_logits(in_features, lg);")
        q.append(f"    int best = 0; float bv = lg[0];")
        q.append(f"    for (int i = 1; i < {nclasses}; i++) if (lg[i] > bv) {{ bv = lg[i]; best = i; }}")
        q.append("    return best;")
        q.append("}")
        return "\n".join(q)

    p.append(emit_predict("predict_mlp_hardcoded", "hc_layer0_fwd"))
    p.append("")
    p.append(emit_predict("predict_mlp_hardcoded_raw", "hc_layer0f_fwd"))
    p.append("")
    p.append("#endif /* aarch64 */")
    p.append("#endif /* MLP_HARDCODED_H */")
    return "\n".join(p) + "\n"


def gen_model_header(W, B):
    """Classic format consumed by mlp_bench.c (scalar/NEON/AVX/XNNPACK)."""
    sizes = [W[0].shape[0]] + [w.shape[1] for w in W]
    nlayers = len(W)
    p = []
    p.append("/* AUTO-GENERATED by train_export_hardcoded.py (classic bench format) */")
    p.append("#ifndef MLP_FLOW_MODEL_H")
    p.append("#define MLP_FLOW_MODEL_H")
    p.append(f"#define NUM_LAYERS {nlayers}")
    p.append("static const int LAYER_SIZES[] = {" + ", ".join(map(str, sizes)) + "};")
    for L in range(nlayers):
        p.append(emit_const_array(f"MODEL_W{L}", W[L].reshape(-1)))          # [in][out]
        p.append(emit_const_array(f"MODEL_W{L}_XNN", W[L].T.reshape(-1)))    # [out][in]
        p.append(emit_const_array(f"MODEL_B{L}", B[L]))
    p.append("static const float *const WEIGHTS[] = {" + ", ".join(f"MODEL_W{L}" for L in range(nlayers)) + "};")
    p.append("static const float *const WEIGHTS_XNN[] = {" + ", ".join(f"MODEL_W{L}_XNN" for L in range(nlayers)) + "};")
    p.append("static const float *const BIASES[] = {" + ", ".join(f"MODEL_B{L}" for L in range(nlayers)) + "};")
    p.append("static const float *const BIASES_XNN[] = {" + ", ".join(f"MODEL_B{L}" for L in range(nlayers)) + "};")
    p.append("#endif")
    return "\n".join(p) + "\n"


def gen_feature_stats(mean, std):
    p = ["/* AUTO-GENERATED feature standardization stats */",
         "#ifndef FEATURE_STATS_H", "#define FEATURE_STATS_H",
         emit_const_array("FEATURE_MEAN", mean),
         emit_const_array("FEATURE_STD", std),
         "#endif"]
    return "\n".join(p) + "\n"

# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="flow-feature CSV (numeric features + label column)")
    ap.add_argument("--label-col", default=None, help="label column name (default: last)")
    ap.add_argument("--hidden", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--from-npz", help="skip training; load W0,B0,...,mean,std from npz")
    ap.add_argument("--save-npz", help="save trained weights to npz")
    ap.add_argument("--prefix", default=".", help="output directory")
    args = ap.parse_args()

    class_names = None
    if args.from_npz:
        z = np.load(args.from_npz, allow_pickle=True)
        nl = sum(1 for k in z.files if k.startswith("W"))
        W = [z[f"W{i}"].astype(np.float32) for i in range(nl)]
        B = [z[f"B{i}"].astype(np.float32) for i in range(nl)]
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        if "classes" in z.files:
            class_names = list(z["classes"])
    else:
        if not args.csv:
            sys.exit("need --csv or --from-npz")
        X, y, feat_names, class_names = load_csv(args.csv, args.label_col)
        print(f"{len(X)} rows, {X.shape[1]} features, {len(class_names)} classes: {class_names}")
        trainer = train_torch if args.backend == "torch" else train_numpy
        W, B, mean, std = trainer(X, y, args.hidden, args.epochs, args.lr, args.batch, args.seed)

    if args.save_npz:
        np.savez(args.save_npz, mean=mean, std=std, classes=np.array(class_names or []),
                 **{f"W{i}": W[i] for i in range(len(W))},
                 **{f"B{i}": B[i] for i in range(len(B))})

    import os
    for fn, content in (("mlp_hardcoded.h", gen_hardcoded_header(W, B, mean, std, class_names)),
                        ("mlp_flow_model.h", gen_model_header(W, B)),
                        ("feature_stats.h", gen_feature_stats(mean, std))):
        path = os.path.join(args.prefix, fn)
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path} ({len(content)//1024} KiB)")
    sizes = [W[0].shape[0]] + [w.shape[1] for w in W]
    print(f"shape: {' -> '.join(map(str, sizes))}")


if __name__ == "__main__":
    main()
