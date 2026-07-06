/* bench_tflite_capi.c
 * TensorFlow Lite C API + XNNPACK delegate, structured identically to
 * bench_onnx_capi.c: input/output tensor pointers are fetched ONCE after
 * TfLiteInterpreterAllocateTensors(), then the timed loop only writes
 * into that pointer (via TfLiteTensorData(), no CopyFromBuffer call) and
 * calls TfLiteInterpreterInvoke() -- nothing allocated in the hot path,
 * same "pure engine execution" measurement as the ORT IoBinding version.
 *
 * Same source builds for BOTH x86 and BF3 ARM (A78) -- TFLite's C API is
 * portable C, XNNPACK auto-dispatches to AVX2/FMA or NEON internally
 * based on the CPU it's actually running on. You do NOT need separate
 * x86/ARM source files the way the hand-written kernel_avx.c/kernel_neon.c
 * needed intrinsics -- XNNPACK does that dispatch for you, which is
 * arguably the whole value proposition of using it over hand-rolled SIMD.
 *
 * Verified against the REAL TFLite v2.16.1 C API headers (fetched from
 * https://github.com/tensorflow/tensorflow, not written from memory) --
 * see third_party/tflite/include/tensorflow/lite/core/c/c_api.h and
 * .../delegates/xnnpack/xnnpack_delegate.h for the exact signatures used.
 *
 * NOT yet compiled/run end-to-end: building libtensorflowlite_c.so from
 * source requires CMake FetchContent to pull Eigen from gitlab.com, which
 * this sandbox's network policy blocks (only github.com/pypi.org/etc are
 * allowlisted). This is a sandbox restriction, not a code issue -- your
 * own x86 dev box and the BF3 almost certainly don't have this
 * restriction. See the build instructions below; please report back if
 * anything doesn't compile so I can fix it against your actual build.
 *
 * Build (on a machine WITHOUT this sandbox's network restriction):
 *   1. Get libtensorflowlite_c.so + headers, either:
 *      a) Build from source (CMake path, ~20-60 min depending on cores):
 *           git clone --branch v2.16.1 --depth 1 \
 *             https://github.com/tensorflow/tensorflow.git
 *           mkdir tflite_build && cd tflite_build
 *           cmake -DCMAKE_BUILD_TYPE=Release -DTFLITE_ENABLE_XNNPACK=ON \
 *             ../tensorflow/tensorflow/lite/c
 *           cmake --build . -j$(nproc)
 *         -> produces libtensorflowlite_c.so in tflite_build/
 *      b) Or check if your distro/environment already has a prebuilt one
 *         (some ML-focused DPU/Jetson images ship it under a package name
 *         like libtensorflow-lite-dev).
 *   2. Compile this file against it:
 *        gcc -O3 -I<tensorflow source root> \
 *          bench_tflite_capi.c \
 *          -L<dir with libtensorflowlite_c.so> -ltensorflowlite_c \
 *          -Wl,-rpath,<same dir> \
 *          -o bench_tflite_capi
 *      (on BF3: same command, no cross-prefix needed if building natively
 *      there, same as the ONNX-C-API pattern earlier in this project)
 *
 * Usage: ./bench_tflite_capi <model.tflite> <platform_label> <model_tag> [json_out] [num_threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#define N_FEATURES 16
#define N_WARMUP     20000
#define N_THROUGHPUT 100000
#define N_LATENCY    10000

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static inline float rand_feature(void) {
    return ((float)(xorshift64() % 100000) / 100000.0f - 0.5f) * 10.0f;
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.tflite> <platform_label> <model_tag> [json_out] [num_threads]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *platform_label = argv[2];
    const char *model_tag = argv[3];
    const char *json_out = argc > 4 ? argv[4] : NULL;
    int num_threads = argc > 5 ? atoi(argv[5]) : 1;

    TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
    if (!model) { fprintf(stderr, "failed to load model: %s\n", model_path); return 1; }

    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, num_threads);

    /* XNNPACK delegate: this is the actual "XNNPACK" part -- without
     * adding this delegate, TFLite falls back to its older reference
     * kernels. Options-default gives XNNPACK's standard single/multi
     * -thread CPU config; num_threads above governs both the interpreter
     * and (if built with threading enabled) XNNPACK's own thread pool. */
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_opts.num_threads = num_threads;
    TfLiteDelegate *xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    TfLiteInterpreterOptionsAddDelegate(options, xnnpack_delegate);

    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) { fprintf(stderr, "failed to create interpreter\n"); return 1; }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "failed to allocate tensors\n"); return 1;
    }

    /* fetch tensor pointers ONCE -- TfLiteTensorData() gives a raw void*
     * into the tensor's backing buffer; we write into it directly every
     * iteration with no TfLiteTensorCopyFromBuffer() call, matching the
     * "nothing allocated in the hot loop" contract from bench_onnx_capi.c */
    TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor *output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float *in_buf = (float *)TfLiteTensorData(input_tensor);
    /* output type depends on how the model was exported (softmax head
     * here means float32 probabilities, not an int64 label like the ONNX
     * sklearn export) -- adjust this cast if your .tflite's output differs */
    float *out_buf = (float *)TfLiteTensorData(output_tensor);

    volatile double sink = 0;

    /* ---- warm-up ---- */
    for (int i = 0; i < N_WARMUP; i++) {
        for (int f = 0; f < N_FEATURES; f++) in_buf[f] = rand_feature();
        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) { fprintf(stderr, "invoke failed\n"); return 1; }
        sink += out_buf[0];
    }

    /* ---- throughput pass ---- */
    uint64_t t0 = now_ns();
    for (int i = 0; i < N_THROUGHPUT; i++) {
        for (int f = 0; f < N_FEATURES; f++) in_buf[f] = rand_feature();
        TfLiteInterpreterInvoke(interpreter);
        sink += out_buf[0];
    }
    uint64_t t1 = now_ns();
    double total_s = (double)(t1 - t0) / 1e9;
    double throughput_ips = N_THROUGHPUT / total_s;
    double avg_latency_ns = (double)(t1 - t0) / N_THROUGHPUT;

    /* ---- per-call latency pass, for percentiles ---- */
    uint64_t *lat = malloc(sizeof(uint64_t) * N_LATENCY);
    for (int i = 0; i < N_LATENCY; i++) {
        for (int f = 0; f < N_FEATURES; f++) in_buf[f] = rand_feature();
        uint64_t a = now_ns();
        TfLiteInterpreterInvoke(interpreter);
        uint64_t b = now_ns();
        sink += out_buf[0];
        lat[i] = b - a;
    }
    qsort(lat, N_LATENCY, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = lat[(int)(N_LATENCY * 0.50)];
    uint64_t p95 = lat[(int)(N_LATENCY * 0.95)];
    uint64_t p99 = lat[(int)(N_LATENCY * 0.99)];

    printf("platform=%-10s kernel=tflite-xnnpack-c model=%-16s avg_ns=%.1f p50_ns=%llu "
           "p95_ns=%llu p99_ns=%llu throughput_ips=%.0f (sink=%f)\n",
           platform_label, model_tag, avg_latency_ns, (unsigned long long)p50,
           (unsigned long long)p95, (unsigned long long)p99, throughput_ips, sink);

    if (json_out) {
        FILE *f = fopen(json_out, "a");
        if (f) {
            fprintf(f, "{\"platform\":\"%s\",\"kernel\":\"tflite-xnnpack-c\",\"model\":\"%s\","
                       "\"avg_latency_ns\":%.1f,\"p50_ns\":%llu,\"p95_ns\":%llu,"
                       "\"p99_ns\":%llu,\"throughput_ips\":%.0f}\n",
                    platform_label, model_tag, avg_latency_ns, (unsigned long long)p50,
                    (unsigned long long)p95, (unsigned long long)p99, throughput_ips);
            fclose(f);
        }
    }

    free(lat);
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteXNNPackDelegateDelete(xnnpack_delegate);
    TfLiteModelDelete(model);
    return 0;
}
