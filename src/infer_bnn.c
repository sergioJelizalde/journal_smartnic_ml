#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "inference.h"
#include "../models/model_bnn.h"

static inline uint64_t tail_mask(uint16_t bits)
{
    uint16_t r = bits & 63u;
    if (r == 0) return UINT64_MAX;
    return (1ULL << r) - 1ULL;
}

static void pack_input_bits(const float *x, uint16_t n, uint64_t *bits)
{
    uint16_t words = (uint16_t)((n + 63u) / 64u);
    for (uint16_t w = 0; w < words; w++) bits[w] = 0;
    for (uint16_t i = 0; i < n; i++) {
        if (x[i] >= 0.0f) bits[i >> 6] |= (1ULL << (i & 63u));
    }
}

static int bnn_layer(const uint64_t *in_bits,
                     uint16_t in_bits_n,
                     uint16_t out_bits_n,
                     uint16_t words_per_out,
                     const uint64_t *weights,
                     const float *bias,
                     uint64_t *out_bits,
                     float *out_scores,
                     int final_layer)
{
    uint16_t out_words = (uint16_t)((out_bits_n + 63u) / 64u);
    for (uint16_t w = 0; w < out_words; w++) out_bits[w] = 0;

    uint64_t mask = tail_mask(in_bits_n);
    for (uint16_t o = 0; o < out_bits_n; o++) {
        const uint64_t *wrow = &weights[(size_t)o * words_per_out];
        uint32_t pc = 0;
        for (uint16_t wi = 0; wi < words_per_out; wi++) {
            uint64_t xnor = ~(in_bits[wi] ^ wrow[wi]);
            if (wi == words_per_out - 1) xnor &= mask;
            pc += (uint32_t)__builtin_popcountll(xnor);
        }
        float score = (2.0f * (float)pc) - (float)in_bits_n + bias[o];
        if (out_scores) out_scores[o] = score;
        if (!final_layer && score >= 0.0f) out_bits[o >> 6] |= (1ULL << (o & 63u));
    }
    return 0;
}

size_t bnn_scratch_bytes(void)
{
    uint16_t max_bits = 0;
    for (int i = 0; i <= BNN_NUM_LAYERS; i++) {
        if (BNN_LAYER_SIZES[i] > max_bits) max_bits = BNN_LAYER_SIZES[i];
    }
    uint16_t words = (uint16_t)((max_bits + 63u) / 64u);
    return (size_t)2 * words * sizeof(uint64_t) + (size_t)max_bits * sizeof(float);
}

int bnn_predict_entry(const struct model_runtime *m, const float *x, void *scratch)
{
    (void)m;
    uint16_t max_bits = 0;
    for (int i = 0; i <= BNN_NUM_LAYERS; i++) if (BNN_LAYER_SIZES[i] > max_bits) max_bits = BNN_LAYER_SIZES[i];
    uint16_t max_words = (uint16_t)((max_bits + 63u) / 64u);

    uint64_t *a = (uint64_t *)scratch;
    uint64_t *b = a + max_words;
    float *scores = (float *)(b + max_words);

    pack_input_bits(x, BNN_LAYER_SIZES[0], a);
    uint64_t *in = a;
    uint64_t *out = b;

    for (int l = 0; l < BNN_NUM_LAYERS; l++) {
        int final = (l == BNN_NUM_LAYERS - 1);
        bnn_layer(in, BNN_LAYER_SIZES[l], BNN_LAYER_SIZES[l + 1],
                  BNN_WORDS_PER_OUT[l], BNN_WEIGHTS[l], BNN_BIASES[l],
                  out, final ? scores : NULL, final);
        uint64_t *tmp = in;
        in = out;
        out = tmp;
    }

    int outputs = BNN_LAYER_SIZES[BNN_NUM_LAYERS];
    if (outputs == 1) return scores[0] >= 0.0f ? 1 : 0;
    int best = 0;
    float best_v = scores[0];
    for (int i = 1; i < outputs; i++) {
        if (scores[i] > best_v) {
            best_v = scores[i];
            best = i;
        }
    }
    return best;
}
