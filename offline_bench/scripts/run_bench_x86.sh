#!/usr/bin/env bash
# run_bench_x86.sh -- runs the full x86 test matrix in easiest->hardest order:
#   1. onnx (python)   2. scalar C   3. avx C   4. onnx (C API)
set -e
cd "$(dirname "$0")/.."
MODELS="16_8 32_16 64_32 128_64_16 256_128_32"
JSON=results/bench_results.json
mkdir -p results
rm -f "$JSON"

echo "### 1/4  ONNX Runtime (python) -- easiest, zero build step ###"
for m in $MODELS; do
  python3 scripts/bench_onnx.py --model models/model_$m.onnx --platform cpu-x86 --json "$JSON"
done

echo "### 2/4  Scalar C kernel ###"
for m in $MODELS; do
  ./bin/x86_scalar_$m cpu-x86 scalar $m "$JSON"
done

echo "### 3/4  AVX2/FMA C kernel ###"
for m in $MODELS; do
  ./bin/x86_avx_$m cpu-x86 avx $m "$JSON"
done

echo "### 4/4  ONNX Runtime (C API, no python overhead) ###"
for m in $MODELS; do
  ./bin/bench_onnx_capi models/model_$m.onnx cpu-x86 $m "$JSON"
done

echo
echo "All results appended to $JSON"
