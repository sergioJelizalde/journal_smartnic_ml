/* bench_onnx_capi.c
 * ONNX Runtime via the C API directly -- no Python. This is the "hardest
 * tier" ONNX test: it isolates true session.Run() cost from the ~15us of
 * Python/pybind overhead that bench_onnx.py's numbers include, and it's
 * structurally identical to how a real DPDK-worker-core / BF3-ARM-core
 * caller would invoke ORT (matches your existing ORT-C-API-on-BF3 setup).
 *
 * Build:
 *   gcc -O3 -Ithird_party/onnxruntime/include bench_onnx_capi.c \
 *       -L<path-to-libonnxruntime.so-dir> -lonnxruntime -Wl,-rpath,<same dir> \
 *       -o bench_onnx_capi
 *
 * Usage: ./bench_onnx_capi <model.onnx> <platform_label> <model_tag> [json_out]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

static const OrtApi *g_ort = NULL;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline float rand_feature(void) {
    return ((float)(xorshift64() % 100000) / 100000.0f - 0.5f) * 10.0f;
}

#define CHECK(expr) do { \
    OrtStatus *st = (expr); \
    if (st) { fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(st)); \
              g_ort->ReleaseStatus(st); exit(1); } \
} while (0)

#define N_FEATURES 16
#define N_WARMUP 2000
#define N_THROUGHPUT 100000
#define N_LATENCY 10000

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.onnx> <platform_label> <model_tag> [json_out]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *platform_label = argv[2];
    const char *model_tag = argv[3];
    const char *json_out = argc > 4 ? argv[4] : NULL;

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions *opts;
    CHECK(g_ort->CreateSessionOptions(&opts));
    g_ort->SetIntraOpNumThreads(opts, 1); /* single-threaded: matches one data-plane core doing inference */
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    OrtSession *session;
    CHECK(g_ort->CreateSession(env, model_path, opts, &session));

    OrtMemoryInfo *mem_info;
    CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtAllocator *allocator;
    CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    char *input_name, *output_name;
    CHECK(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    CHECK(g_ort->SessionGetOutputName(session, 0, allocator, &output_name));
    const char *input_names[] = {input_name};
    const char *output_names[] = {output_name};

    int64_t input_shape[] = {1, N_FEATURES};
    float sample[N_FEATURES];
    volatile int64_t sink = 0;

    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;

    /* ---- warm-up ---- */
    for (int i = 0; i < N_WARMUP; i++) {
        for (int f = 0; f < N_FEATURES; f++) sample[f] = rand_feature();
        CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, sample, sizeof(sample), input_shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        output_tensor = NULL;
        CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1,
                          output_names, 1, &output_tensor));
        int64_t *out_data;
        g_ort->GetTensorMutableData(output_tensor, (void **)&out_data);
        sink ^= out_data[0];
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
    }

    /* ---- throughput pass ---- */
    uint64_t t0 = now_ns();
    for (int i = 0; i < N_THROUGHPUT; i++) {
        for (int f = 0; f < N_FEATURES; f++) sample[f] = rand_feature();
        CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, sample, sizeof(sample), input_shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        output_tensor = NULL;
        CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1,
                          output_names, 1, &output_tensor));
        int64_t *out_data;
        g_ort->GetTensorMutableData(output_tensor, (void **)&out_data);
        sink ^= out_data[0];
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
    }
    uint64_t t1 = now_ns();
    double total_s = (double)(t1 - t0) / 1e9;
    double throughput_ips = N_THROUGHPUT / total_s;
    double avg_latency_ns = (double)(t1 - t0) / N_THROUGHPUT;

    /* ---- per-call latency pass ---- */
    uint64_t *lat = malloc(sizeof(uint64_t) * N_LATENCY);
    for (int i = 0; i < N_LATENCY; i++) {
        for (int f = 0; f < N_FEATURES; f++) sample[f] = rand_feature();
        uint64_t a = now_ns();
        CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, sample, sizeof(sample), input_shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        output_tensor = NULL;
        CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1,
                          output_names, 1, &output_tensor));
        uint64_t b = now_ns();
        int64_t *out_data;
        g_ort->GetTensorMutableData(output_tensor, (void **)&out_data);
        sink ^= out_data[0];
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        lat[i] = b - a;
    }
    qsort(lat, N_LATENCY, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = lat[(int)(N_LATENCY * 0.50)];
    uint64_t p95 = lat[(int)(N_LATENCY * 0.95)];
    uint64_t p99 = lat[(int)(N_LATENCY * 0.99)];

    printf("platform=%-10s kernel=onnx-c  model=%-16s avg_ns=%.1f p50_ns=%llu "
           "p95_ns=%llu p99_ns=%llu throughput_ips=%.0f (sink=%lld)\n",
           platform_label, model_tag, avg_latency_ns, (unsigned long long)p50,
           (unsigned long long)p95, (unsigned long long)p99, throughput_ips, (long long)sink);

    if (json_out) {
        FILE *f = fopen(json_out, "a");
        if (f) {
            fprintf(f, "{\"platform\":\"%s\",\"kernel\":\"onnx-c\",\"model\":\"%s\","
                       "\"avg_latency_ns\":%.1f,\"p50_ns\":%llu,\"p95_ns\":%llu,"
                       "\"p99_ns\":%llu,\"throughput_ips\":%.0f}\n",
                    platform_label, model_tag, avg_latency_ns, (unsigned long long)p50,
                    (unsigned long long)p95, (unsigned long long)p99, throughput_ips);
            fclose(f);
        }
    }

    free(lat);
    g_ort->ReleaseMemoryInfo(mem_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    return 0;
}
