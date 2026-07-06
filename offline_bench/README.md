# MLP Multi-Platform Inference Benchmark (no DPDK yet)

Pure inference microbenchmark: one feature vector in, one class index out,
for 5 MLP sizes, across scalar / NEON / AVX C kernels and ONNX Runtime,
on both BF3 ARM (A78 cores) and x86 CPU. No packet I/O yet on purpose --
this isolates compute cost so the eventual DPDK integration only adds
what it actually needs to add (ring dequeue + feature extraction).

## Layout
```
scripts/train_multi_mlp.py   trains all 5 sizes, exports C headers + ONNX
models/                      generated: model_<sizes>.h, model_<sizes>.onnx,
                              feature_stats.h, manifest.json
include/mlp_kernel.h          shared predict_mlp() / normalize_features() API
src/kernel_scalar.c           portable reference kernel (no intrinsics)
src/kernel_neon.c             ARM NEON kernel (BF3 A78 cores only, not DPA)
src/kernel_avx.c              x86 AVX2+FMA kernel, structural port of NEON
src/bench_main.c              generic timing harness (same source, any kernel/model)
src/correctness_check.c       cheap fixed-vector output check (safe under qemu)
src/bench_onnx_capi.c         ONNX Runtime via C API (no python overhead)
scripts/bench_onnx.py         ONNX Runtime via python (same script, any platform)
scripts/run_bench_x86.sh      full x86 matrix, easiest->hardest
scripts/run_bench_bf3.sh      full BF3 matrix -- run natively ON the DPU
third_party/onnxruntime/      C API headers (v1.24.4, matches pip onnxruntime)
```

## Model sizes
All 5 share the same 16 input features (from your notebook's `top_features`),
so hidden widths are always multiples of 8 -- no tail loop needed on hidden
layers for either NEON (4-wide) or AVX (8-wide):

| tag             | hidden layers    |
|-----------------|------------------|
| 16_8            | (16, 8)          |
| 32_16           | (32, 16)         |
| 64_32           | (64, 32)         |
| 128_64_16       | (128, 64, 16)    |
| 256_128_32      | (256, 128, 32)   |

Only the output layer (1 for binary, C for multiclass) falls through to a
plain scalar loop in every kernel -- vectorizing a 1-8-wide output buys
nothing, so this isn't a compatibility compromise, it's the same choice
all three kernels make identically.

## Test order: easiest -> hardest

1. **x86 CPU, ONNX Runtime (python)** -- `pip install onnxruntime`, run
   `scripts/bench_onnx.py`. Zero build step, zero cross-compilation.
2. **x86 CPU, scalar C** -- `make x86`, trivial compile, this is your
   correctness baseline (`src/kernel_scalar.c`).
3. **x86 CPU, AVX2/FMA C** -- same `make x86` target, just needs
   `-mavx2 -mfma`; still the same machine, same OS.
4. **x86 CPU, ONNX Runtime (C API)** -- needs the ORT C headers +
   `libonnxruntime.so` (already vendored under `third_party/`), removes
   ~7-8us of python/pybind overhead per call vs step 1. See finding below.
5. **BF3 ARM, scalar C** -- cross-compiled here (`make arm`), but the
   *numbers only mean something run natively on the DPU* -- copy the
   project over and run `scripts/run_bench_bf3.sh`.
6. **BF3 ARM, NEON C** -- same deployment friction as step 5, plus this is
   the one place a genuine correctness bug could hide (lane layout,
   alignment). Verified bit-identical to scalar in this sandbox via
   `qemu-aarch64` (functional only, not timing -- qemu numbers are not
   real perf numbers, don't benchmark under it).
7. **BF3 ARM, ONNX Runtime (C API)** -- hardest: needs your existing
   from-source ORT C API build on the DPU (you already hit the
   LD_LIBRARY_PATH / Git LFS issues doing this). `bench_onnx_capi.c`
   cross-compiles cleanly against the vendored headers; link it on-device
   against your BF3 `libonnxruntime.so`.

Note: BF3's NEON path only applies to the ARM A78 cores. The DPA RISC-V
cores have no NEON and stay on your separate RC4ML/BenchBF3 scalar/BNN
path -- this whole NEON kernel is an ARM-core artifact, not a DPA one.

## Correctness (already verified in this environment)

All 4 kernel/platform combos produce byte-identical predicted classes on
identical input vectors, across all 5 model sizes (x86 scalar vs x86 AVX
vs qemu-emulated ARM scalar vs qemu-emulated ARM NEON): see
`src/correctness_check.c`. Run `make check` then diff outputs before
trusting any timing numbers on new hardware.

## x86 results so far (this sandbox, `make x86 && bash scripts/run_bench_x86.sh`)

Single-sample latency, `avg_ns` = mean over 500k calls (C) / 50-100k calls
(ONNX), `throughput_ips` = 1/avg_latency:

| model       | scalar (ns) | AVX (ns) | AVX speedup | ONNX-python (ns) | ONNX-C-API (ns) |
|-------------|------------:|---------:|------------:|------------------:|------------------:|
| 16_8        |       335   |      100 |       3.3x  |            17,600  |            9,800  |
| 32_16       |       800   |      156 |       5.1x  |            18,000  |           10,000  |
| 64_32       |     2,420   |      352 |       6.9x  |            17,900  |           10,100  |
| 128_64_16   |    10,110   |    1,170 |       8.6x  |            20,300  |           11,800  |
| 256_128_32  |    42,950   |    5,070 |       8.5x  |            19,500  |           13,000  |

**The finding that matters most for your architecture decision:** ONNX
Runtime's *per-call* overhead (session/tensor setup + graph dispatch) is
~10-18 microseconds regardless of model size, even through the C API.
Your hand-written scalar kernel beats ONNX outright below the 128-neuron
tier, and AVX beats it at every tier tested here, by 2-3 orders of
magnitude at the small end. ONNX only starts looking competitive once the
model itself needs >~10us of real compute (much bigger nets, or batching
many packets per call). This is consistent with what published in-network
inference work reports -- Nepco's own numbers show queueing/model
inference landing at millisecond scale specifically because they run full
transformer-class models; DGA/DoH-sized MLPs like these are exactly what
should stay on your hand-rolled NEON/AVX path, with ONNX reserved for the
heavier ARM-tier transformer models in your cascade, not these small MLPs.

## Running it

```bash
# 1. train + export all 5 sizes (edit --data-root to point at your real CSVs)
python3 scripts/train_multi_mlp.py --data-root /home/ubuntu/DoH_DGA_training --N 16
#   or, to just validate the pipeline without your datasets:
python3 scripts/train_multi_mlp.py --synthetic --n-samples 20000

# 2. build everything (x86 native + ARM cross-compiled)
make all

# 3. sanity-check correctness before trusting any timing
make check
for m in 16_8 32_16 64_32 128_64_16 256_128_32; do
  diff <(./bin/check_x86_scalar_$m) <(./bin/check_x86_avx_$m) && echo "$m OK"
done

# 4. run the x86 matrix
bash scripts/run_bench_x86.sh

# 5. copy the whole mlp_bench/ dir to the BF3, then on-device:
ORT_LIB=/path/to/your/libonnxruntime.so bash scripts/run_bench_bf3.sh
```

## Next step (not done here on purpose)

Wire `predict_mlp()` into a DPDK worker-core loop reading from an mbuf
ring instead of the synthetic PRNG feature vectors -- the kernels
themselves don't change at all, only `bench_main.c`'s input source does.
