/*
 * bench_main.c - Microbenchmark harness with CSV output
 *
 * Runs a tight loop of inference calls, collects timing stats,
 * and logs results to CSV for aggregation and plotting.
 *
 * Compile with -DMODEL_SIZE=16_8 -DKERNEL_AVX -DDEVICE_X86
 * (or appropriate substitutions)
 */

#include "mlp_kernel.h"
#include "csv_logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* Default model size if not specified at compile time */
#ifndef MODEL_SIZE
#define MODEL_SIZE 16_8
#endif

/* Stringify macro for MODEL_SIZE */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define MODEL_SIZE_STR TOSTRING(MODEL_SIZE)

/* Fallback feature counts by model size */
static uint32_t get_num_features(const char *model_str) {
    /* All models use 16 input features in this benchmark */
    return 16;
}

static const char* get_hidden_layers(const char *model_str) {
    if (strcmp(model_str, "16_8") == 0) return "16-8";
    if (strcmp(model_str, "32_16") == 0) return "32-16";
    if (strcmp(model_str, "64_32") == 0) return "64-32";
    if (strcmp(model_str, "128_64_16") == 0) return "128-64-16";
    if (strcmp(model_str, "256_128_32") == 0) return "256-128-32";
    return "unknown";
}

/* Platform/device identification for output path */
static const char* get_platform_prefix(void) {
#ifdef DEVICE_X86
    return "x86";
#elif defined(DEVICE_BF3)
    return "bf3";
#else
    return "unknown";
#endif
}

static const char* get_kernel_name(void) {
#ifdef KERNEL_SCALAR
    return "scalar";
#elif defined(KERNEL_AVX)
    return "avx";
#elif defined(KERNEL_NEON)
    return "neon";
#elif defined(KERNEL_XNNPACK)
    return "xnnpack";
#else
    return "unknown";
#endif
}

/* Timing utilities (nanosecond precision) */
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Warmup: prime CPU caches and branch predictors */
static void warmup(uint32_t iterations) {
    float features[16] = {0};  /* Placeholder */
    for (uint32_t i = 0; i < iterations; i++) {
        /* Dummy call to prime cache, prevent optimizations */
        volatile uint32_t result = predict_mlp(features, 16);
        (void)result;
    }
}

/* Collect statistics on latency array */
static void compute_stats(
    const uint64_t *latencies,
    uint32_t count,
    uint64_t *avg_ns,
    uint64_t *std_dev_ns,
    uint64_t *min_ns,
    uint64_t *max_ns
) {
    if (!latencies || count == 0) {
        *avg_ns = *std_dev_ns = *min_ns = *max_ns = 0;
        return;
    }

    /* Compute mean */
    uint64_t sum = 0;
    uint64_t min_val = latencies[0];
    uint64_t max_val = latencies[0];
    for (uint32_t i = 0; i < count; i++) {
        sum += latencies[i];
        if (latencies[i] < min_val) min_val = latencies[i];
        if (latencies[i] > max_val) max_val = latencies[i];
    }
    *avg_ns = sum / count;

    /* Compute std dev */
    double variance = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        double delta = (double)latencies[i] - (double)*avg_ns;
        variance += delta * delta;
    }
    variance /= count;
    *std_dev_ns = (uint64_t)sqrt(variance);

    *min_ns = min_val;
    *max_ns = max_val;
}

int main(void) {
    const char *model_size_str = MODEL_SIZE_STR;
    const char *hidden_str = get_hidden_layers(model_size_str);
    uint32_t num_features = get_num_features(model_size_str);
    const char *platform = get_platform_prefix();
    const char *kernel = get_kernel_name();

    printf("=== MLP Microbenchmark ===\n");
    printf("Model:       %s (hidden: %s)\n", model_size_str, hidden_str);
    printf("Kernel:      %s\n", kernel);
    printf("Platform:    %s\n", platform);
    printf("Features:    %u\n\n", num_features);

    /* ====================================================================== */
    /* 1. Initialize CSV logger                                              */
    /* ====================================================================== */

    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path),
             "../results/%s/optimized_%s.csv",
             platform, kernel);

    csv_logger_t logger;
    if (csv_logger_init(&logger, csv_path, kernel) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize CSV logger at %s\n", csv_path);
        return 1;
    }
    printf("CSV output:  %s\n\n", csv_path);

    /* ====================================================================== */
    /* 2. Generate synthetic feature vectors (or load from dataset)           */
    /* ====================================================================== */

    uint32_t num_iterations = 500000;  /* Adjust for faster iteration in debug mode */
    float *features = malloc(num_iterations * num_features * sizeof(float));
    if (!features) {
        fprintf(stderr, "ERROR: malloc failed\n");
        return 1;
    }

    /* Fill with synthetic normalized [0, 1] values */
    srand48(12345);  /* Reproducible */
    for (uint32_t i = 0; i < num_iterations * num_features; i++) {
        features[i] = (float)drand48();
    }

    /* ====================================================================== */
    /* 3. Warmup phase                                                        */
    /* ====================================================================== */

    printf("Warmup (10k iterations)...\n");
    warmup(10000);
    printf("Done.\n\n");

    /* ====================================================================== */
    /* 4. Benchmark: tight timing loop                                        */
    /* ====================================================================== */

    printf("Benchmarking %u iterations...\n", num_iterations);
    fflush(stdout);

    uint64_t *latencies = malloc(num_iterations * sizeof(uint64_t));
    if (!latencies) {
        fprintf(stderr, "ERROR: malloc failed\n");
        free(features);
        return 1;
    }

    /* Tight loop: time each inference call */
    for (uint32_t i = 0; i < num_iterations; i++) {
        const float *f = &features[i * num_features];

        uint64_t start = get_time_ns();
        volatile uint32_t result = predict_mlp(f, num_features);
        uint64_t end = get_time_ns();

        latencies[i] = end - start;
        (void)result;  /* Prevent optimization */

        /* Progress indicator every 100k iterations */
        if ((i + 1) % 100000 == 0) {
            printf("  ... %u/%u\n", i + 1, num_iterations);
        }
    }
    printf("Done.\n\n");

    /* ====================================================================== */
    /* 5. Analyze results                                                     */
    /* ====================================================================== */

    uint64_t avg_ns, std_dev_ns, min_ns, max_ns;
    compute_stats(latencies, num_iterations, &avg_ns, &std_dev_ns, &min_ns, &max_ns);

    uint64_t throughput_kips = (1000000000ULL / avg_ns);  /* kInferences per second */

    printf("Results:\n");
    printf("  Average latency:  %12lu ns\n", avg_ns);
    printf("  Std dev:          %12lu ns\n", std_dev_ns);
    printf("  Min:              %12lu ns\n", min_ns);
    printf("  Max:              %12lu ns\n", max_ns);
    printf("  Throughput:       %12lu kips (inferences/sec * 1000)\n\n", throughput_kips);

    /* ====================================================================== */
    /* 6. Write CSV row                                                       */
    /* ====================================================================== */

    if (csv_logger_row(&logger,
                       model_size_str,
                       hidden_str,
                       avg_ns, std_dev_ns, min_ns, max_ns,
                       throughput_kips,
                       num_iterations) != 0) {
        fprintf(stderr, "ERROR: Failed to write CSV row\n");
        free(features);
        free(latencies);
        return 1;
    }

    csv_logger_close(&logger);
    printf("Appended to:  %s\n\n", csv_path);

    /* ====================================================================== */
    /* Cleanup                                                                */
    /* ====================================================================== */

    free(features);
    free(latencies);

    printf("✓ Benchmark complete\n");
    return 0;
}
