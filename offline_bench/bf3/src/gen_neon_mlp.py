#!/usr/bin/env python3
"""
gen_neon_mlp.py -- generate a shape-specialized NEON (aarch64) MLP inference
kernel for data-plane deployment (BlueField-3 / Cortex-A78AE).

Design (XNNPACK f32-gemm style, but specialized at generation time):

  * Weights are pre-packed ONCE at init into contiguous [K][NR] tiles per
    output block (NR = 16/8/4), so the hot loop is pure sequential loads.
  * Inner loop loads 4 inputs with one vld1q_f32 and reuses them via
    vfmaq_laneq_f32 -- 1 load feeds 4 k-steps per accumulator register.
  * Accumulators are split into even/odd k banks ("2-way k-split"): with
    NR=16 that is 8 independent FMLA chains, enough to hide the ~4-cycle
    FMLA latency on the A78's two 128-bit FMA pipes.
  * ReLU is fused into the store; the output layer fuses argmax directly
    on the logits (softmax skipped -- argmax invariant).
  * Every bound is a literal constant: no graph, no dispatch, no runtime
    shape logic. Layers small in K are fully unrolled; large K uses a
    constant-trip-count loop unrolled by 8 (compiler still knows the count).

Usage:
    python3 gen_neon_mlp.py 64 32 16 2 -o mlp_generated.h
    # sizes = input_dim hidden1 [hidden2 ...] num_classes

The generated header expects the model header (WEIGHTS/BIASES/LAYER_SIZES,
same format as mlp_bench.c uses) to be included FIRST, and exposes:

    void mlp_generated_init(void);              /* pack weights, call once */
    int  predict_mlp_generated(const float *in);

Build (with the patched mlp_bench.c):
    gcc -O3 -march=armv8-a+simd -DUSE_GENERATED \
        -DMODEL_HEADER='"mlp_64_32.h"' mlp_bench.c -o bench_gen -lm
"""
import argparse
import sys

FULL_UNROLL_K = 128       # fully unroll k when K <= this
K_LOOP_UNROLL = 8         # otherwise unroll constant-bound loop by this


def emit_pack(layers):
    """Emit packed-weight buffers + init() that packs from WEIGHTS[L]."""
    out = []
    out.append("/* ---- packed weights (filled once by mlp_generated_init) ---- */")
    for L, (K, N) in enumerate(layers):
        sz = sum(K * nr for _, nr in block_plan(N))
        if sz:
            out.append(f"static float g_wpack{L}[{sz}] "
                       f"__attribute__((aligned(64)));")
    out.append("")
    out.append("static void mlp_generated_init(void) {")
    for L, (K, N) in enumerate(layers):
        plan = block_plan(N)
        if not plan:
            continue
        out.append(f"    {{ /* layer {L}: {K} -> {N} */")
        out.append(f"        const float *W = WEIGHTS[{L}];")
        out.append(f"        float *p = g_wpack{L};")
        off = 0
        for (j0, nr) in plan:
            out.append(f"        for (int k = 0; k < {K}; k++)")
            out.append(f"            for (int n = 0; n < {nr}; n++)")
            out.append(f"                p[{off} + k*{nr} + n] = W[k*{N} + {j0} + n];")
            off += K * nr
        out.append("    }")
    out.append("}")
    out.append("")
    return "\n".join(out)


def block_plan(N):
    """Split N outputs into blocks of 16, then 8, then 4. Remainder < 4 is
    handled by the 'dot' strategy in the caller (only for tiny layers)."""
    plan, j = [], 0
    for nr in (16, 8, 4):
        while N - j >= nr:
            plan.append((j, nr))
            j += nr
    return plan


def emit_wide_block(L, K, N, j0, nr, off, relu, indent="    "):
    """Emit one NR-wide output block with laneq FMA and 2-way k-split."""
    nregs = nr // 4
    lines = []
    a = lambda bank, r: f"acc{bank}_{r}"
    # init accumulators: even bank gets bias, odd bank zero
    for r in range(nregs):
        lines.append(f"float32x4_t {a('e', r)} = vld1q_f32(&BIASES[{L}][{j0 + 4*r}]);")
    for r in range(nregs):
        lines.append(f"float32x4_t {a('o', r)} = vdupq_n_f32(0.0f);")
    lines.append(f"const float *w = g_wpack{L} + {off};")

    def k_step(kexpr_vec, lane, bank, koff_rows):
        s = []
        for r in range(nregs):
            s.append(f"{a(bank, r)} = vfmaq_laneq_f32({a(bank, r)}, "
                     f"vld1q_f32(w + {koff_rows}*{nr} + {4*r}), {kexpr_vec}, {lane});")
        return s

    body = []
    K4 = K - (K % 4)
    if K <= FULL_UNROLL_K:
        for k in range(0, K4, 4):
            body.append(f"{{ float32x4_t xv = vld1q_f32(in + {k});")
            for lane in range(4):
                bank = 'e' if lane % 2 == 0 else 'o'
                body += k_step("xv", lane, bank, k + lane)
            body.append("}")
    else:
        body.append(f"for (int k = 0; k < {K4}; k += {K_LOOP_UNROLL}) {{")
        for u in range(0, K_LOOP_UNROLL, 4):
            body.append(f"  float32x4_t xv{u} = vld1q_f32(in + k + {u});")
            for lane in range(4):
                bank = 'e' if lane % 2 == 0 else 'o'
                for r in range(nregs):
                    body.append(
                        f"  {a(bank, r)} = vfmaq_laneq_f32({a(bank, r)}, "
                        f"vld1q_f32(w + (k + {u + lane})*{nr} + {4*r}), xv{u}, {lane});")
        body.append("}")
    # k remainder (K % 4)
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


def emit_dot_outputs(L, K, N, j_start, relu, indent="    "):
    """Per-output vectorized dot products for leftover outputs (N-j_start < 4)
    or tiny layers. Reads the ORIGINAL (unpacked) weights with stride N --
    fine, these are a handful of outputs."""
    lines = []
    K4 = K - (K % 4)
    for j in range(j_start, N):
        lines.append(f"{{ /* output {j} */")
        lines.append("  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);")
        lines.append(f"  const float *W = WEIGHTS[{L}];")
        lines.append(f"  for (int k = 0; k < {K4}; k += 8) {{")
        lines.append(f"    float32x4_t x0 = vld1q_f32(in + k);")
        lines.append(f"    float32x4_t w0 = (float32x4_t){{W[(k+0)*{N}+{j}], W[(k+1)*{N}+{j}], W[(k+2)*{N}+{j}], W[(k+3)*{N}+{j}]}};")
        lines.append("    s0 = vfmaq_f32(s0, x0, w0);")
        lines.append(f"    if (k + 8 <= {K4}) {{")
        lines.append(f"      float32x4_t x1 = vld1q_f32(in + k + 4);")
        lines.append(f"      float32x4_t w1 = (float32x4_t){{W[(k+4)*{N}+{j}], W[(k+5)*{N}+{j}], W[(k+6)*{N}+{j}], W[(k+7)*{N}+{j}]}};")
        lines.append("      s1 = vfmaq_f32(s1, x1, w1);")
        lines.append("    }")
        lines.append("  }")
        lines.append(f"  float acc = BIASES[{L}][{j}] + vaddvq_f32(vaddq_f32(s0, s1));")
        for k in range(K4, K):
            lines.append(f"  acc += WEIGHTS[{L}][{k}*{N}+{j}] * in[{k}];")
        if relu:
            lines.append(f"  out[{j}] = acc > 0.0f ? acc : 0.0f;")
        else:
            lines.append(f"  out[{j}] = acc;")
        lines.append("}")
    return [indent + l for l in lines]


def emit_layer_fn(L, K, N, relu):
    lines = [f"static inline void layer{L}_fwd(const float *restrict in, "
             f"float *restrict out) {{"]
    off = 0
    plan = block_plan(N)
    for (j0, nr) in plan:
        lines.append(f"    {{ /* block j={j0}..{j0+nr-1} (NR={nr}) */")
        lines += emit_wide_block(L, K, N, j0, nr, off, relu)
        lines.append("    }")
        off += K * nr
    covered = sum(nr for _, nr in plan)
    if covered < N:
        lines += emit_dot_outputs(L, K, N, covered, relu)
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def emit_predict(layers, n_classes):
    sizes = [layers[0][0]] + [n for _, n in layers]
    bufmax = max(sizes)
    lines = []
    lines.append(f"static int predict_mlp_generated(const float *restrict in_features) {{")
    lines.append(f"    float b0[{bufmax}] __attribute__((aligned(64)));")
    lines.append(f"    float b1[{bufmax}] __attribute__((aligned(64)));")
    cur_in, cur_out = "in_features", "b0"
    for L in range(len(layers)):
        lines.append(f"    layer{L}_fwd({cur_in}, {cur_out});")
        cur_in = cur_out
        cur_out = "b1" if cur_out == "b0" else "b0"
    # fused argmax on logits
    lines.append(f"    int best = 0; float bv = {cur_in}[0];")
    lines.append(f"    for (int i = 1; i < {n_classes}; i++)")
    lines.append(f"        if ({cur_in}[i] > bv) {{ bv = {cur_in}[i]; best = i; }}")
    lines.append("    return best;")
    lines.append("}")
    return "\n".join(lines)


def generate(sizes):
    layers = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    parts = []
    parts.append("/* AUTO-GENERATED by gen_neon_mlp.py -- do not edit.")
    parts.append(f" * Model shape: {' -> '.join(map(str, sizes))}")
    parts.append(" * Include AFTER the model header (needs WEIGHTS/BIASES). */")
    parts.append("#ifndef MLP_GENERATED_H")
    parts.append("#define MLP_GENERATED_H")
    parts.append("#include <arm_neon.h>")
    parts.append("")
    # sanity guard
    parts.append(f"_Static_assert(1, \"shape guard\");")
    guard = " && ".join(f"LAYER_SIZES[{i}] == {s}" for i, s in enumerate(sizes))
    parts.append(f"#define MLP_GEN_SHAPE_OK() ({guard})")
    parts.append("")
    parts.append(emit_pack(layers))
    for L, (K, N) in enumerate(layers):
        relu = (L != len(layers) - 1)
        parts.append(emit_layer_fn(L, K, N, relu))
    parts.append(emit_predict(layers, sizes[-1]))
    parts.append("#endif /* MLP_GENERATED_H */")
    return "\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate specialized NEON MLP kernel")
    ap.add_argument("sizes", nargs="+", type=int,
                    help="layer sizes: input hidden... classes (e.g. 64 32 16 2)")
    ap.add_argument("-o", "--output", default="mlp_generated.h")
    args = ap.parse_args()
    if len(args.sizes) < 2:
        sys.exit("need at least input and output size")
    with open(args.output, "w") as f:
        f.write(generate(args.sizes))
    print(f"wrote {args.output} for shape {' -> '.join(map(str, args.sizes))}")


if __name__ == "__main__":
    main()
