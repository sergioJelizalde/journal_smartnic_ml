#include <string.h>
#include "inference.h"
#include "../models/model_rf.h"

int rf_predict_entry(const struct model_runtime *m, const float *x, void *scratch)
{
    (void)m;
    (void)scratch;
    int votes[RF_NUM_CLASSES];
    memset(votes, 0, sizeof(votes));

    for (uint16_t t = 0; t < RF_NUM_TREES; t++) {
        uint32_t off = RF_TREE_OFFSETS[t];
        uint32_t end = RF_TREE_OFFSETS[t + 1];
        uint32_t node = 0;
        int pred = 0;

        for (uint16_t depth = 0; depth <= RF_MAX_DEPTH + 1; depth++) {
            uint32_t g = off + node;
            if (g >= end) break;
            int16_t value = RF_VALUE[g];
            if (value >= 0) {
                pred = value;
                break;
            }
            int16_t f = RF_FEATURE[g];
            if (f < 0 || f >= RF_NUM_FEATURES) break;
            int16_t next = (x[f] <= RF_THRESHOLD[g]) ? RF_LEFT[g] : RF_RIGHT[g];
            if (next < 0) break;
            node = (uint32_t)next;
        }
        if (pred >= 0 && pred < RF_NUM_CLASSES) votes[pred]++;
    }

    int best = 0;
    int best_votes = votes[0];
    for (int c = 1; c < RF_NUM_CLASSES; c++) {
        if (votes[c] > best_votes) {
            best_votes = votes[c];
            best = c;
        }
    }
    return best;
}
