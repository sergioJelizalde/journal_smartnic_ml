/* bench_onnx_capi_reuse.c
 * Exploratory tool (kept for reproducibility, not part of the main build):
 * this is what established that tensor-allocation churn is only ~9% of
 * ONNX Runtime's per-call cost, with the rest being graph-dispatch
 * overhead. That finding is now baked into the canonical
 * src/bench_onnx_capi.c (IoBinding-based), which is what run_bench_*.sh
 * actually calls -- this file is left as the A/B comparison that proved it.
 *
 * Isolates WHICH part of ONNX Runtime's per-call overhead is reducible:
 *   mode=baseline  -- session reused (already how bench_onnx_capi.c works),
 *                     but a fresh input/output OrtValue is created and
 *                     released every call (this is the "baseline" cost).
 *   mode=iobinding -- session AND input/output tensors created ONCE;
 *                     each iteration only memcpy's new features into the
 *                     already-bound input buffer, then calls Run() again.
 *                     No CreateTensor/ReleaseValue in the hot loop at all.
 *   mode=batch     -- like iobinding, but runs a batch of N samples per
 *                     Run() call, amortizing whatever fixed per-call cost
 *                     remains across N samples instead of 1.
 *
 * Build: gcc -O3 -I<ort include> bench_onnx_capi_reuse.c -L<ort lib> \
 *          -lonnxruntime -Wl,-rpath,<ort lib> -o bench_onnx_capi_reuse
 * Usage: ./bench_onnx_capi_reuse <model.onnx> <mode> <batch_size>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

static const OrtApi *g_ort = NULL;
#define N_FEATURES 16
#define N_WARMUP 2000
#define N_ITERS 50000

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
#define CHECK(expr) do { OrtStatus *st = (expr); if (st) { \
    fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(st)); \
    g_ort->ReleaseStatus(st); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.onnx> <baseline|iobinding|batch> <batch_size>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *mode = argv[2];
    int batch = atoi(argv[3]);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env; CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));
    OrtSessionOptions *opts; CHECK(g_ort->CreateSessionOptions(&opts));
    g_ort->SetIntraOpNumThreads(opts, 1);
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    OrtSession *session; CHECK(g_ort->CreateSession(env, model_path, opts, &session));
    OrtMemoryInfo *mem_info;
    CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
    OrtAllocator *allocator; CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    char *input_name, *output_name;
    CHECK(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    CHECK(g_ort->SessionGetOutputName(session, 0, allocator, &output_name));
    const char *input_names[] = {input_name};
    const char *output_names[] = {output_name};

    float *samples = malloc(sizeof(float) * batch * N_FEATURES);
    int64_t input_shape[] = {batch, N_FEATURES};
    volatile double sink = 0;

    if (strcmp(mode, "baseline") == 0) {
        /* fresh OrtValue every call -- this is what bench_onnx_capi.c does */
        for (int i = 0; i < N_WARMUP; i++) {
            for (int f = 0; f < N_FEATURES; f++) samples[f] = rand_feature();
            OrtValue *in_t, *out_t = NULL;
            CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem_info, samples, sizeof(float) * N_FEATURES,
                  input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_t));
            CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&in_t, 1,
                              output_names, 1, &out_t));
            float *out_data; g_ort->GetTensorMutableData(out_t, (void **)&out_data);
            sink += out_data[0];
            g_ort->ReleaseValue(out_t); g_ort->ReleaseValue(in_t);
        }
        uint64_t t0 = now_ns();
        for (int i = 0; i < N_ITERS; i++) {
            for (int f = 0; f < N_FEATURES; f++) samples[f] = rand_feature();
            OrtValue *in_t, *out_t = NULL;
            CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem_info, samples, sizeof(float) * N_FEATURES,
                  input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_t));
            CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&in_t, 1,
                              output_names, 1, &out_t));
            float *out_data; g_ort->GetTensorMutableData(out_t, (void **)&out_data);
            sink += out_data[0];
            g_ort->ReleaseValue(out_t); g_ort->ReleaseValue(in_t);
        }
        uint64_t t1 = now_ns();
        double per_sample_ns = (double)(t1 - t0) / N_ITERS / batch;
        printf("mode=baseline  batch=%-4d per_sample_ns=%.1f  (sink=%f)\n", batch, per_sample_ns, sink);

    } else if (strcmp(mode, "iobinding") == 0 || strcmp(mode, "batch") == 0) {
        /* create input/output tensors ONCE, mutate the buffer in place,
         * never Create/Release inside the loop */
        OrtValue *in_t;
        CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem_info, samples, sizeof(float) * batch * N_FEATURES,
              input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_t));

        for (int i = 0; i < N_WARMUP; i++) {
            for (int f = 0; f < batch * N_FEATURES; f++) samples[f] = rand_feature();
            OrtValue *out_t = NULL;
            CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&in_t, 1,
                              output_names, 1, &out_t));
            float *out_data; g_ort->GetTensorMutableData(out_t, (void **)&out_data);
            sink += out_data[0];
            g_ort->ReleaseValue(out_t);  /* output shape can vary call-to-call in general; still freed */
        }
        uint64_t t0 = now_ns();
        for (int i = 0; i < N_ITERS; i++) {
            for (int f = 0; f < batch * N_FEATURES; f++) samples[f] = rand_feature();
            OrtValue *out_t = NULL;
            CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&in_t, 1,
                              output_names, 1, &out_t));
            float *out_data; g_ort->GetTensorMutableData(out_t, (void **)&out_data);
            sink += out_data[0];
            g_ort->ReleaseValue(out_t);
        }
        uint64_t t1 = now_ns();
        double per_sample_ns = (double)(t1 - t0) / N_ITERS / batch;
        printf("mode=%-9s batch=%-4d per_sample_ns=%.1f  (sink=%f)\n", mode, batch, per_sample_ns, sink);
        g_ort->ReleaseValue(in_t);
    }

    free(samples);
    g_ort->ReleaseMemoryInfo(mem_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    return 0;
}
