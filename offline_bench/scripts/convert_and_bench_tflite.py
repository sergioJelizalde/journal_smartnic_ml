#!/usr/bin/env python3
"""
convert_and_bench_tflite.py

Builds a Keras MLP with the SAME hidden_layer_sizes as your sklearn
models (weights don't matter for a latency benchmark -- only the
architecture/FLOP count does), converts it to .tflite, then benchmarks
it with tf.lite.Interpreter, which uses XNNPACK as its CPU backend by
default since TF 2.3+ (no extra flags needed for float32 models -- you'd
only need to disable it explicitly to get the older reference kernels).

Same warmup/throughput/percentile methodology and JSON schema as
bench_main.c / bench_onnx.py, kernel label "tflite-xnnpack", so it drops
straight into the same bench_results*.json / plotting pipeline.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_SIZES = {
    "16_8": (16, 8),
    "32_16": (32, 16),
    "64_32": (64, 32),
    "128_64_16": (128, 64, 16),
    "256_128_32": (256, 128, 32),
}
N_FEATURES = 16
N_CLASSES = 2  # binary classifier head, matches the sklearn models' shape

N_WARMUP = 2000
N_THROUGHPUT = 50000
N_LATENCY = 5000


def build_and_convert(hidden_sizes, out_path):
    """Same layer shape as the sklearn MLPClassifier: Dense+ReLU per
    hidden layer, single Dense head. Random weights -- fine for latency,
    since compute cost depends on shape, not values."""
    inputs = tf.keras.Input(shape=(N_FEATURES,), dtype=tf.float32)
    x = inputs
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    return out_path


def bench_tflite(model_path, platform_label, model_tag, json_out=None, num_threads=1):
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    rng = np.random.default_rng(42)

    def rand_input():
        return ((rng.random((1, N_FEATURES), dtype=np.float32) - 0.5) * 10.0).astype(np.float32)

    for _ in range(N_WARMUP):
        interpreter.set_tensor(input_detail["index"], rand_input())
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])

    t0 = time.perf_counter()
    for _ in range(N_THROUGHPUT):
        interpreter.set_tensor(input_detail["index"], rand_input())
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])
    t1 = time.perf_counter()
    total_s = t1 - t0
    throughput_ips = N_THROUGHPUT / total_s
    avg_latency_ns = total_s / N_THROUGHPUT * 1e9

    lat_ns = np.empty(N_LATENCY)
    for i in range(N_LATENCY):
        a = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], rand_input())
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])
        b = time.perf_counter()
        lat_ns[i] = (b - a) * 1e9
    lat_ns.sort()
    p50 = lat_ns[int(N_LATENCY * 0.50)]
    p95 = lat_ns[int(N_LATENCY * 0.95)]
    p99 = lat_ns[int(N_LATENCY * 0.99)]

    print(f"platform={platform_label:<10} kernel=tflite-xnnpack model={model_tag:<16} "
          f"avg_ns={avg_latency_ns:.1f} p50_ns={p50:.0f} p95_ns={p95:.0f} p99_ns={p99:.0f} "
          f"throughput_ips={throughput_ips:.0f}")

    if json_out:
        with open(json_out, "a") as f:
            f.write(json.dumps({
                "platform": platform_label, "kernel": "tflite-xnnpack", "model": model_tag,
                "avg_latency_ns": avg_latency_ns, "p50_ns": float(p50), "p95_ns": float(p95),
                "p99_ns": float(p99), "throughput_ips": throughput_ips,
            }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, help="e.g. cpu-x86 or bf3-arm")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out-dir", default="../models")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, sizes in MODEL_SIZES.items():
        tflite_path = out_dir / f"model_{tag}.tflite"
        build_and_convert(sizes, tflite_path)
        print(f"[OK] converted {tag} -> {tflite_path}")
        bench_tflite(tflite_path, args.platform, tag, args.json, num_threads=args.threads)


if __name__ == "__main__":
    main()
