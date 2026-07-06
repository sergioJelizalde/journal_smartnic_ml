#!/usr/bin/env bash
# run_bench_bf3.sh -- run natively ON the BlueField-3 ARM cores (A78), not
# under qemu. The bin/arm_* binaries were cross-compiled and are already
# statically linked (scalar/neon), so just scp the whole project dir over
# and run this. The onnx-capi step needs BF3's own libonnxruntime.so
# (the one you already built the C API against) -- point ORT_LIB at it.
#
# Usage on BF3:
#   ORT_LIB=/path/to/libonnxruntime.so ./scripts/run_bench_bf3.sh
set -e
cd "$(dirname "$0")/.."
MODELS="16_8 32_16 64_32 128_64_16 256_128_32"
JSON=results/bench_results_bf3.json
mkdir -p results
rm -f "$JSON"

echo "### 1/4  ONNX Runtime (python) -- only if onnxruntime wheel installs on the DPU OS ###"
for m in $MODELS; do
  python3 scripts/bench_onnx.py --model models/model_$m.onnx --platform bf3-arm --json "$JSON" || \
    echo "[skip] python onnxruntime not available on this DPU image"
done

echo "### 2/4  Scalar C kernel (ARM A78) ###"
for m in $MODELS; do
  ./bin/arm_scalar_$m bf3-arm scalar $m "$JSON"
done

echo "### 3/4  NEON C kernel (ARM A78) ###"
for m in $MODELS; do
  ./bin/arm_neon_$m bf3-arm neon $m "$JSON"
done

echo "### 4/4  ONNX Runtime (C API) -- needs ORT_LIB pointed at your on-device build ###"
if [ -z "$ORT_LIB" ]; then
  echo "[skip] set ORT_LIB=/path/to/libonnxruntime.so and re-run this step manually:"
  echo "  aarch64-linux-gnu-gcc -O3 -Ithird_party/onnxruntime/include src/bench_onnx_capi.c \\"
  echo "    -L\$(dirname \$ORT_LIB) -lonnxruntime -Wl,-rpath,\$(dirname \$ORT_LIB) -o bin/bench_onnx_capi_arm"
else
  ORTDIR=$(dirname "$ORT_LIB")
  aarch64-linux-gnu-gcc -O3 -Ithird_party/onnxruntime/include src/bench_onnx_capi.c \
    -L"$ORTDIR" -lonnxruntime -Wl,-rpath,"$ORTDIR" -o bin/bench_onnx_capi_arm
  for m in $MODELS; do
    ./bin/bench_onnx_capi_arm models/model_$m.onnx bf3-arm $m "$JSON"
  done
fi

echo
echo "All results appended to $JSON"
