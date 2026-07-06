/* bench_main.c
 * Same driver source is compiled once per (kernel x model-size) combo --
 * the Makefile picks which kernel_*.c and which model_*.h go into each
 * binary. No DPDK here on purpose: pure inference-only microbenchmark,
 * one feature vector in, one class index out, exactly like the eventual
 * per-packet call inside the data plane will be, minus the packet I/O.
 *
 * Usage: ./bench <platform_label> <kernel_label> <model_tag> [json_out]
 *   e.g. ./bench cpu-x86 avx 256_128_32 results/bench_results.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mlp_kernel.h"
#include MODEL_HEADER_FILE
#include FEATURE_HEADER_FILE

#define N_WARMUP        20000
#define N_THROUGHPUT    500000
#define N_LATENCY       20000
#define MAX_BUF         1024   /* >= largest hidden layer (256) with headroom */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* small deterministic PRNG so every platform/kernel sees identical inputs */
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline float rand_feature(void) {
    /* roughly N(0, 5) range, plausible raw flow-feature magnitude */
    return ((float)(xorshift64() % 100000) / 100000.0f - 0.5f) * 10.0f;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *platform_label = argc > 1 ? argv[1] : "unknown-platform";
    const char *kernel_label   = argc > 2 ? argv[2] : "unknown-kernel";
    const char *model_tag      = argc > 3 ? argv[3] : "unknown-model";
    const char *json_out       = argc > 4 ? argv[4] : NULL;

    float buf_a[MAX_BUF], buf_b[MAX_BUF];
    float sample[NUM_FEATURES];
    volatile int sink = 0; /* prevents the compiler from eliding predict_mlp */

    /* ---- warm-up (branch predictor / caches / frequency scaling) ---- */
    for (int i = 0; i < N_WARMUP; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) sample[f] = rand_feature();
        sink ^= predict_mlp(sample, buf_a, buf_b);
    }

    /* ---- throughput pass: one big timed block ---- */
    uint64_t t0 = now_ns();
    for (int i = 0; i < N_THROUGHPUT; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) sample[f] = rand_feature();
        sink ^= predict_mlp(sample, buf_a, buf_b);
    }
    uint64_t t1 = now_ns();
    double total_s = (double)(t1 - t0) / 1e9;
    double throughput_ips = N_THROUGHPUT / total_s;
    double avg_latency_ns = (double)(t1 - t0) / N_THROUGHPUT;

    /* ---- per-call latency pass, for percentiles ---- */
    uint64_t *lat = malloc(sizeof(uint64_t) * N_LATENCY);
    for (int i = 0; i < N_LATENCY; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) sample[f] = rand_feature();
        uint64_t a = now_ns();
        sink ^= predict_mlp(sample, buf_a, buf_b);
        uint64_t b = now_ns();
        lat[i] = b - a;
    }
    qsort(lat, N_LATENCY, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = lat[(int)(N_LATENCY * 0.50)];
    uint64_t p95 = lat[(int)(N_LATENCY * 0.95)];
    uint64_t p99 = lat[(int)(N_LATENCY * 0.99)];

    printf("platform=%-10s kernel=%-7s model=%-16s layers_in=%d out=%d "
           "avg_ns=%.1f p50_ns=%llu p95_ns=%llu p99_ns=%llu throughput_ips=%.0f (sink=%d)\n",
           platform_label, kernel_label, model_tag, LAYER_SIZES[0], LAYER_SIZES[NUM_LAYERS],
           avg_latency_ns, (unsigned long long)p50, (unsigned long long)p95,
           (unsigned long long)p99, throughput_ips, sink);

    if (json_out) {
        FILE *f = fopen(json_out, "a");
        if (f) {
            fprintf(f,
                "{\"platform\":\"%s\",\"kernel\":\"%s\",\"model\":\"%s\","
                "\"num_features\":%d,\"num_layers\":%d,\"avg_latency_ns\":%.1f,"
                "\"p50_ns\":%llu,\"p95_ns\":%llu,\"p99_ns\":%llu,"
                "\"throughput_ips\":%.0f}\n",
                platform_label, kernel_label, model_tag, NUM_FEATURES, NUM_LAYERS,
                avg_latency_ns, (unsigned long long)p50, (unsigned long long)p95,
                (unsigned long long)p99, throughput_ips);
            fclose(f);
        }
    }

    free(lat);
    return 0;
}
