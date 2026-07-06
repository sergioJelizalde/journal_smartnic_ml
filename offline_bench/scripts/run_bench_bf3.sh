#!/usr/bin/env bash
# run_bench_bf3.sh -- run natively ON the BlueField-3 ARM cores (A78), not
# under qemu. Works whether you built with `make arm` (cross-compiled from
# an x86 dev box -> bin/arm_scalar_*, bin/arm_neon_*) or `make native` run
# directly on the BF3 (-> bin/native_arm_scalar_*, bin/native_arm_neon_*);
# this script auto-detects which naming exists.
#
# The onnx-capi step needs BF3's own libonnxruntime.so (the one you already
# built the C API against) -- point ORT_LIB at it.
#
# Usage on BF3:
#   ORT_LIB=/path/to/libonnxruntime.so ./scripts/run_bench_bf3.sh
set -e
cd "$(dirname "$0")/.."
MODELS="16_8 32_16 64_32 128_64_16 256_128_32"
JSON=results/bench_results_bf3.json
mkdir -p results
rm -f "$JSON"

# resolve_bin <scalar|neon> <model_tag> -> echoes the binary path that
# actually exists, preferring the cross-compiled name, falling back to
# the native-build name.
resolve_bin() {
  local kernel="$1" tag="$2"
  if [ -x "bin/arm_${kernel}_${tag}" ]; then
    echo "bin/arm_${kernel}_${tag}"
  elif [ -x "bin/native_arm_${kernel}_${tag}" ]; then
    echo "bin/native_arm_${kernel}_${tag}"
  else
    echo ""
  fi
}

echo "### 1/4  ONNX Runtime (python) -- only if numpy/onnxruntime install on the DPU OS ###"
for m in $MODELS; do
  python3 scripts/bench_onnx.py --model models/model_$m.onnx --platform bf3-arm --json "$JSON" 2>/dev/null || \
    echo "[skip] python numpy/onnxruntime not available on this DPU image (model $m) -- see note below"
done

echo "### 2/4  Scalar C kernel (ARM A78) ###"
for m in $MODELS; do
  bin=$(resolve_bin scalar "$m")
  if [ -z "$bin" ]; then
    echo "[MISSING] no bin/arm_scalar_$m or bin/native_arm_scalar_$m -- run 'make native' first"
    continue
  fi
  "$bin" bf3-arm scalar $m "$JSON"
done

echo "### 3/4  NEON C kernel (ARM A78) ###"
for m in $MODELS; do
  bin=$(resolve_bin neon "$m")
  if [ -z "$bin" ]; then
    echo "[MISSING] no bin/arm_neon_$m or bin/native_arm_neon_$m -- run 'make native' first"
    continue
  fi
  "$bin" bf3-arm neon $m "$JSON"
done

echo "### 4/4  ONNX Runtime (C API) -- needs ORT_LIB pointed at your on-device build ###"
if [ -z "$ORT_LIB" ]; then
  echo "[skip] set ORT_LIB=/path/to/libonnxruntime.so and re-run this step manually:"
  echo "  aarch64-linux-gnu-gcc -O3 -Ithird_party/onnxruntime/include src/bench_onnx_capi.c \\"
  echo "    -L\$(dirname \$ORT_LIB) -lonnxruntime -Wl,-rpath,\$(dirname \$ORT_LIB) -o bin/bench_onnx_capi_arm"
  echo "  (on the BF3 itself, drop the aarch64-linux-gnu- prefix and use plain gcc)"
else
  ORTDIR=$(dirname "$ORT_LIB")
  NATIVE_GCC=gcc
  if ! "$NATIVE_GCC" -dumpmachine 2>/dev/null | grep -q aarch64; then
    NATIVE_GCC=aarch64-linux-gnu-gcc  # cross-compiling from an x86 box
  fi
  "$NATIVE_GCC" -O3 -Ithird_party/onnxruntime/include src/bench_onnx_capi.c \
    -L"$ORTDIR" -lonnxruntime -Wl,-rpath,"$ORTDIR" -o bin/bench_onnx_capi_arm
  for m in $MODELS; do
    ./bin/bench_onnx_capi_arm models/model_$m.onnx bf3-arm $m "$JSON"
  done
fi

echo
echo "All results appended to $JSON"
echo
echo "Note: to get numpy/onnxruntime working for step 1 on an offline DPU image,"
echo "either 'pip install numpy onnxruntime' if the DPU has any pip access, or"
echo "skip python entirely -- step 4 (C API) already gives you the real ONNX number"
echo "without needing numpy at all."
