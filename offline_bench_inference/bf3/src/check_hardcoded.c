/*
 * check_hardcoded.c — correctness + latency for the weight-hardcoded kernel.
 *
 * Build on the BF3 (or cross-compile aarch64):
 *   gcc -O3 -march=armv8-a+simd check_hardcoded.c -o check_hardcoded -lm
 *
 * What it does:
 *   1. CORRECTNESS: N_CHECK random raw inputs (mean ± 3σ per feature).
 *      For each input, compares:
 *        - scalar reference (unpacked weights, explicit normalization)
 *        - predict_mlp_hardcoded      (packed NEON, normalized input)
 *        - predict_mlp_hardcoded_raw  (packed NEON, folded normalization)
 *      Reports argmax agreement and max |logit| deviation (FMA reassociation
 *      makes small ULP-level differences expected; predictions must match
 *      except for near-tie logits, which are counted separately).
 *   2. LATENCY: per-inference ns for both entry points, written to
 *      latencies_hardcoded.csv / latencies_hardcoded_raw.csv (same format as
 *      mlp_bench.c: iter,latency_ns). Note the _raw kernel absorbs the
 *      normalization cost that mlp_bench.c keeps OUTSIDE the timed region,
 *      so compare _raw against (normalize + kernel) for a fair end-to-end view.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "mlp_hardcoded.h"

#ifndef N_CHECK
#define N_CHECK 100000
#endif
#ifndef WARMUP_ITERS
#define WARMUP_ITERS 2000
#endif
#ifndef MEASURE_ITERS
#define MEASURE_ITERS 10000
#endif

static inline float randomf(void) { return (float)rand() / (float)RAND_MAX; }

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void make_raw(float *raw) {
    for (int k = 0; k < HC_INPUT_DIM; k++)
        raw[k] = HC_FEATURE_MEAN[k] + HC_FEATURE_STD[k] * (randomf() * 6.0f - 3.0f);
}

static void normalize(const float *raw, float *out) {
    for (int k = 0; k < HC_INPUT_DIM; k++)
        out[k] = (raw[k] - HC_FEATURE_MEAN[k]) / HC_FEATURE_STD[k];
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void bench(const char *csvname, const char *label, int use_raw) {
    FILE *out = fopen(csvname, "w");
    fprintf(out, "iter,latency_ns\n");
    float raw[HC_INPUT_DIM] __attribute__((aligned(64)));
    float nrm[HC_INPUT_DIM] __attribute__((aligned(64)));
    uint64_t *lat = malloc(MEASURE_ITERS * sizeof(uint64_t));
    for (int it = 0; it < WARMUP_ITERS + MEASURE_ITERS; it++) {
        make_raw(raw);
        if (!use_raw) normalize(raw, nrm);
        uint64_t t0 = now_ns();
        volatile int cls = use_raw ? predict_mlp_hardcoded_raw(raw)
                                    : predict_mlp_hardcoded(nrm);
        uint64_t t1 = now_ns();
        (void)cls;
        if (it >= WARMUP_ITERS) {
            lat[it - WARMUP_ITERS] = t1 - t0;
            fprintf(out, "%d,%llu\n", it - WARMUP_ITERS, (unsigned long long)(t1 - t0));
        }
    }
    fclose(out);
    qsort(lat, MEASURE_ITERS, sizeof(uint64_t), cmp_u64);
    printf("%-28s p50 %6llu ns   p99 %6llu ns   min %6llu ns   -> %s\n", label,
           (unsigned long long)lat[MEASURE_ITERS / 2],
           (unsigned long long)lat[(int)(MEASURE_ITERS * 0.99)],
           (unsigned long long)lat[0], csvname);
    free(lat);
}

int main(void) {
    srand(12345);

    /* ---------------- correctness ---------------- */
    float raw[HC_INPUT_DIM] __attribute__((aligned(64)));
    float nrm[HC_INPUT_DIM] __attribute__((aligned(64)));
    float lg_ref[HC_NUM_CLASSES], lg_hc[HC_NUM_CLASSES], lg_raw[HC_NUM_CLASSES];

    int mism_hc = 0, mism_raw = 0, near_tie = 0;
    double max_dev_hc = 0.0, max_dev_raw = 0.0;

    for (int it = 0; it < N_CHECK; it++) {
        make_raw(raw);
        normalize(raw, nrm);

        hc_reference_logits(raw, lg_ref);
        predict_mlp_hardcoded_logits(nrm, lg_hc);
        predict_mlp_hardcoded_raw_logits(raw, lg_raw);

        int p_ref = 0, p_hc = 0, p_raw = 0;
        for (int i = 1; i < HC_NUM_CLASSES; i++) {
            if (lg_ref[i] > lg_ref[p_ref]) p_ref = i;
            if (lg_hc[i]  > lg_hc[p_hc])   p_hc  = i;
            if (lg_raw[i] > lg_raw[p_raw]) p_raw = i;
        }
        for (int i = 0; i < HC_NUM_CLASSES; i++) {
            double dh = fabs((double)lg_hc[i]  - lg_ref[i]);
            double dr = fabs((double)lg_raw[i] - lg_ref[i]);
            if (dh > max_dev_hc)  max_dev_hc  = dh;
            if (dr > max_dev_raw) max_dev_raw = dr;
        }
        if (p_hc != p_ref || p_raw != p_ref) {
            /* distinguish real bugs from FP near-ties at the decision boundary */
            float top = lg_ref[p_ref], second = -INFINITY;
            for (int i = 0; i < HC_NUM_CLASSES; i++)
                if (i != p_ref && lg_ref[i] > second) second = lg_ref[i];
            if (top - second < 1e-4f) near_tie++;
            else { if (p_hc != p_ref) mism_hc++; if (p_raw != p_ref) mism_raw++; }
        }
    }
    printf("correctness over %d inputs:\n", N_CHECK);
    printf("  packed-NEON (normalized in): %d real mismatches, max |logit dev| %.3e\n",
           mism_hc, max_dev_hc);
    printf("  packed-NEON (raw, folded)  : %d real mismatches, max |logit dev| %.3e\n",
           mism_raw, max_dev_raw);
    printf("  near-tie boundary cases    : %d (expected FP noise, not errors)\n", near_tie);
    if (mism_hc || mism_raw) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n\n");

    /* ---------------- latency ---------------- */
    bench("latencies_hardcoded.csv",     "hardcoded (normalized in)", 0);
    bench("latencies_hardcoded_raw.csv", "hardcoded (raw, folded)  ", 1);
    return 0;
}
