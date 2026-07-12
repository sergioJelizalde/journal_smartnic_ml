#!/bin/bash
set -e

echo "Running x86 benchmarks..."
echo ""

MODELS="16_8 32_16 64_32 128_64_16 256_128_32"

for model in $MODELS; do
    echo "=== Scalar ($model) ==="
    ./bin/bench_x86_scalar_$model || echo "  (skipped)"
    echo ""
    
    echo "=== AVX ($model) ==="
    ./bin/bench_x86_avx_$model || echo "  (skipped)"
    echo ""
    
    echo "=== XNNPACK ($model) ==="
    ./bin/bench_x86_xnnpack_$model || echo "  (skipped)"
    echo ""
done

echo "✓ All benchmarks complete. Check results/"
ls -lh ../results/x86/*.csv 2>/dev/null || echo "(No CSVs yet - may need to build first)"
