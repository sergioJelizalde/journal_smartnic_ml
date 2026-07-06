#!/usr/bin/env python3
"""
bench_onnx.py -- ONNX Runtime inference benchmark.

Same script, same file, runs unmodified on x86 CPU and on BF3 ARM64 --
that's the whole point of testing ONNX here: one artifact, two platforms,
CPUExecutionProvider picks whatever's underneath.

Mirrors bench_main.c's methodology so the numbers are comparable:
  - warm-up
  - single-sample throughput pass (batch size 1, matches the "one packet
    in, one verdict out" shape the C kernels test)
  - separate per-call latency pass for percentiles

Usage:
    python3 bench_onnx.py --model ../models/model_16_8.onnx \
        --platform cpu-x86 --json ../results/bench_results.json
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort

N_WARMUP = 2000
N_THROUGHPUT = 50000   # ORT python-call overhead dominates -- fewer iters than the C bench is fine
N_LATENCY = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--platform", required=True, help="e.g. cpu-x86 or bf3-arm")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    n_features = sess.get_inputs()[0].shape[1]
    if not isinstance(n_features, int):
        n_features = 16  # dynamic dim fallback -- matches our fixed 16-feature models

    rng = np.random.default_rng(args.seed)

    def rand_batch(n):
        return (rng.random((n, n_features), dtype=np.float32) - 0.5) * 10.0

    # ---- warm-up ----
    warm = rand_batch(N_WARMUP)
    for i in range(N_WARMUP):
        sess.run(None, {input_name: warm[i:i + 1]})

    # ---- throughput pass ----
    tp_data = rand_batch(N_THROUGHPUT)
    t0 = time.perf_counter()
    for i in range(N_THROUGHPUT):
        sess.run(None, {input_name: tp_data[i:i + 1]})
    t1 = time.perf_counter()
    total_s = t1 - t0
    throughput_ips = N_THROUGHPUT / total_s
    avg_latency_ns = total_s / N_THROUGHPUT * 1e9

    # ---- per-call latency pass ----
    lat_data = rand_batch(N_LATENCY)
    lat_ns = np.empty(N_LATENCY)
    for i in range(N_LATENCY):
        a = time.perf_counter()
        sess.run(None, {input_name: lat_data[i:i + 1]})
        b = time.perf_counter()
        lat_ns[i] = (b - a) * 1e9
    lat_ns.sort()
    p50 = lat_ns[int(N_LATENCY * 0.50)]
    p95 = lat_ns[int(N_LATENCY * 0.95)]
    p99 = lat_ns[int(N_LATENCY * 0.99)]

    model_tag = args.model.split("model_")[-1].replace(".onnx", "")
    print(f"platform={args.platform:<10} kernel=onnx   model={model_tag:<16} "
          f"n_features={n_features} avg_ns={avg_latency_ns:.1f} "
          f"p50_ns={p50:.0f} p95_ns={p95:.0f} p99_ns={p99:.0f} "
          f"throughput_ips={throughput_ips:.0f}")

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({
                "platform": args.platform, "kernel": "onnx", "model": model_tag,
                "num_features": int(n_features), "avg_latency_ns": avg_latency_ns,
                "p50_ns": float(p50), "p95_ns": float(p95), "p99_ns": float(p99),
                "throughput_ips": throughput_ips,
            }) + "\n")


if __name__ == "__main__":
    main()
