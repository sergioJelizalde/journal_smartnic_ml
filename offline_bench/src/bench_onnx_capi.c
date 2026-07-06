/* bench_onnx_capi.c
 * ONNX Runtime via the C API, using IoBinding to measure pure engine
 * execution -- input/output tensors are bound to fixed pre-allocated
 * buffers ONCE outside the timed loop, so nothing is Create'd or
 * Release'd per call. This isolates ORT's actual per-call cost (graph
 * dispatch: per-node kernel lookup, shape checks, thread-pool sync) from
 * tensor-allocation churn, which turned out to be only ~9% of the total
 * (see the baseline-vs-iobinding comparison this file grew out of).
 *
 * This is the closest ORT's public API gets to what predict_mlp() does
 * in the scalar/NEON/AVX kernels -- fixed buffers in, one call, read the
 * result out -- though RunWithBinding() still walks the graph node-by-node
 * (that per-node dispatch has no equivalent in a hand-fused function like
 * predict_mlp(); it is NOT a further-reducible cost, it's the structural
 * price of a generic graph executor).
 *
 * Same CLI signature and JSON schema as the original Run()-based version,
 * so it's a drop-in replacement -- existing run_bench_*.sh scripts and
 * bench_results*.json files need no changes; re-running just supersedes
 * old "onnx-c" rows for the same (platform, kernel, model) key.
 *
 * Build:
 *   gcc -O3 -I<ort include> bench_onnx_capi.c -L<ort lib> \
 *     -lonnxruntime -Wl,-rpath,<ort lib> -o bin/bench_onnx_capi
 *
 * Usage: ./bench_onnx_capi <model.onnx> <platform_label> <model_tag> [json_out] [batch]
 *   batch defaults to 1 -- matches the single-sample convention used by
 *   the scalar/neon/avx kernels and the plain-Run() onnx benchmarks, so
 *   the numbers stay directly comparable. Pass a larger batch only if you
 *   specifically want to study the amortization effect (see the
 *   baseline/iobinding/batch exploration -- batching is the real lever,
 *   this binary just measures the honest batch=1 number by default).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

static const OrtApi *g_ort = NULL;
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
#define CHECK(expr) do { OrtStatus *st = (expr); if (st) { \
    fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(st)); \
    g_ort->ReleaseStatus(st); exit(1); } } while (0)

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.onnx> <platform_label> <model_tag> [json_out] [batch]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *platform_label = argv[2];
    const char *model_tag = argv[3];
    const char *json_out = argc > 4 ? argv[4] : NULL;
    int batch = argc > 5 ? atoi(argv[5]) : 1;

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env; CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));
    OrtSessionOptions *opts; CHECK(g_ort->CreateSessionOptions(&opts));
    g_ort->SetIntraOpNumThreads(opts, 1); /* single-threaded: matches one data-plane core doing inference */
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    OrtSession *session; CHECK(g_ort->CreateSession(env, model_path, opts, &session));
    OrtMemoryInfo *mem_info;
    CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
    OrtAllocator *allocator; CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    char *input_name, *output_name;
    CHECK(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    CHECK(g_ort->SessionGetOutputName(session, 0, allocator, &output_name));

    /* discover output #0's shape/type so we can bind a fixed buffer to it
     * (for these sklearn-exported classifiers, output #0 is the int64
     * predicted label) */
    OrtTypeInfo *out_type_info;
    CHECK(g_ort->SessionGetOutputTypeInfo(session, 0, &out_type_info));
    const OrtTensorTypeAndShapeInfo *out_tensor_info;
    CHECK(g_ort->CastTypeInfoToTensorInfo(out_type_info, &out_tensor_info));
    size_t out_dims_count;
    CHECK(g_ort->GetDimensionsCount(out_tensor_info, &out_dims_count));
    int64_t out_dims[8];
    CHECK(g_ort->GetDimensions(out_tensor_info, out_dims, out_dims_count));
    if (out_dims_count > 0) out_dims[0] = batch;
    int64_t out_elems = 1;
    for (size_t i = 0; i < out_dims_count; i++) out_elems *= out_dims[i];
    g_ort->ReleaseTypeInfo(out_type_info);

    float *in_buf = malloc(sizeof(float) * batch * N_FEATURES);
    int64_t *out_buf = malloc(sizeof(int64_t) * out_elems);
    int64_t input_shape[] = {batch, N_FEATURES};

    OrtValue *in_tensor;
    CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem_info, in_buf, sizeof(float) * batch * N_FEATURES,
          input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor));
    OrtValue *out_tensor;
    CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem_info, out_buf, sizeof(int64_t) * out_elems,
          out_dims, out_dims_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &out_tensor));

    OrtIoBinding *binding;
    CHECK(g_ort->CreateIoBinding(session, &binding));
    CHECK(g_ort->BindInput(binding, input_name, in_tensor));
    CHECK(g_ort->BindOutput(binding, output_name, out_tensor));

    volatile int64_t sink = 0;

    /* ---- warm-up ---- */
    for (int i = 0; i < N_WARMUP; i++) {
        for (int f = 0; f < batch * N_FEATURES; f++) in_buf[f] = rand_feature();
        CHECK(g_ort->RunWithBinding(session, NULL, binding));
        sink ^= out_buf[0];
    }

    /* ---- throughput pass ---- */
    uint64_t t0 = now_ns();
    for (int i = 0; i < N_THROUGHPUT; i++) {
        for (int f = 0; f < batch * N_FEATURES; f++) in_buf[f] = rand_feature();
        CHECK(g_ort->RunWithBinding(session, NULL, binding));
        sink ^= out_buf[0];
    }
    uint64_t t1 = now_ns();
    double total_s = (double)(t1 - t0) / 1e9;
    double throughput_ips = (double)N_THROUGHPUT * batch / total_s;
    double avg_latency_ns = (double)(t1 - t0) / N_THROUGHPUT / batch;

    /* ---- per-call latency pass, for percentiles (per-sample, i.e. divided by batch) ---- */
    uint64_t *lat = malloc(sizeof(uint64_t) * N_LATENCY);
    for (int i = 0; i < N_LATENCY; i++) {
        for (int f = 0; f < batch * N_FEATURES; f++) in_buf[f] = rand_feature();
        uint64_t a = now_ns();
        CHECK(g_ort->RunWithBinding(session, NULL, binding));
        uint64_t b = now_ns();
        sink ^= out_buf[0];
        lat[i] = (b - a) / (uint64_t)batch;
    }
    qsort(lat, N_LATENCY, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = lat[(int)(N_LATENCY * 0.50)];
    uint64_t p95 = lat[(int)(N_LATENCY * 0.95)];
    uint64_t p99 = lat[(int)(N_LATENCY * 0.99)];

    printf("platform=%-10s kernel=onnx-c  model=%-16s batch=%-4d avg_ns=%.1f p50_ns=%llu "
           "p95_ns=%llu p99_ns=%llu throughput_ips=%.0f (sink=%lld)\n",
           platform_label, model_tag, batch, avg_latency_ns, (unsigned long long)p50,
           (unsigned long long)p95, (unsigned long long)p99, throughput_ips, (long long)sink);

    if (json_out) {
        FILE *f = fopen(json_out, "a");
        if (f) {
            fprintf(f, "{\"platform\":\"%s\",\"kernel\":\"onnx-c\",\"model\":\"%s\",\"batch\":%d,"
                       "\"avg_latency_ns\":%.1f,\"p50_ns\":%llu,\"p95_ns\":%llu,"
                       "\"p99_ns\":%llu,\"throughput_ips\":%.0f}\n",
                    platform_label, model_tag, batch, avg_latency_ns, (unsigned long long)p50,
                    (unsigned long long)p95, (unsigned long long)p99, throughput_ips);
            fclose(f);
        }
    }

    free(lat); free(in_buf); free(out_buf);
    g_ort->ReleaseIoBinding(binding);
    g_ort->ReleaseValue(out_tensor);
    g_ort->ReleaseValue(in_tensor);
    g_ort->ReleaseMemoryInfo(mem_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    return 0;
}
