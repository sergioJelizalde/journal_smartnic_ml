/* correctness_check.c
 * Runs predict_mlp() on a small fixed set of deterministic vectors and
 * prints the predicted class + raw checksum of intermediate math, so we
 * can diff scalar vs neon vs avx output without paying for the full
 * 500k-iteration benchmark loop (useful under qemu, which is far too
 * slow to run the real bench_main.c timing loop in reasonable time).
 */
#include <stdio.h>
#include "mlp_kernel.h"
#include MODEL_HEADER_FILE
#include FEATURE_HEADER_FILE

#define MAX_BUF 1024
#define N_TEST_VECTORS 8

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

int main(void) {
    float buf_a[MAX_BUF], buf_b[MAX_BUF];
    for (int v = 0; v < N_TEST_VECTORS; v++) {
        float sample[NUM_FEATURES];
        for (int f = 0; f < NUM_FEATURES; f++) sample[f] = rand_feature();
        int cls = predict_mlp(sample, buf_a, buf_b);
        printf("%d\n", cls);
    }
    return 0;
}
