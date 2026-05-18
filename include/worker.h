#ifndef WORKER_H
#define WORKER_H

#include <stdint.h>
#include <stdatomic.h>
#include <rte_mempool.h>
#include "common.h"
#include "profiles.h"
#include "inference.h"
#include "logger.h"
#include "app_config.h"

struct worker_ctx {
    const struct app_config *cfg;
    const struct profile_ops *profile;
    void *profile_state;
    const struct model_runtime *model;
    struct logger *logger;
    atomic_int *stop_flag;

    struct rte_mempool *mbuf_pool;
    uint16_t port_id;
    uint16_t queue_id;
    unsigned lcore_id;
    uint16_t worker_id;

    void *model_scratch;
    float scaled_features[APP_MAX_FEATURES];

    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t dropped_packets;
    uint64_t feature_events;
    uint64_t invalid_features;
    uint64_t predictions[32];

#if APP_ENABLE_TIMING
    uint64_t infer_cycles_sum;
    uint64_t infer_cycles_count;
    uint64_t infer_cycles_min;
    uint64_t infer_cycles_max;
#endif
};

int worker_loop(void *arg);

#endif
