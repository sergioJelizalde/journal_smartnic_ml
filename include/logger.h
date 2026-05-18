#ifndef LOGGER_H
#define LOGGER_H

#include <stdint.h>
#include <stdatomic.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include "common.h"
#include "app_config.h"

struct log_event {
    uint64_t tsc;
    uint16_t worker_id;
    uint16_t queue_id;
    uint8_t profile;
    uint8_t model;
    uint8_t pred;
    uint8_t action;
    struct feature_id id;
    struct feature_vector fv;
};

struct logger {
    enum app_log_mode mode;
    struct rte_ring *ring;
    struct rte_mempool *pool;
    char file_name[256];
    atomic_int *stop_flag;
    uint64_t submitted;
    uint64_t dropped;
};

#if APP_ENABLE_LOGS
int logger_init(struct logger *lg, const struct app_config *cfg,
                atomic_int *stop_flag, int socket_id);
void logger_free(struct logger *lg);
int logger_submit_prediction(struct logger *lg, const struct log_event *ev);
int logger_loop(void *arg);
#else
static inline int logger_init(struct logger *lg, const struct app_config *cfg,
                              atomic_int *stop_flag, int socket_id)
{
    (void)lg; (void)cfg; (void)stop_flag; (void)socket_id; return 0;
}
static inline void logger_free(struct logger *lg) { (void)lg; }
static inline int logger_submit_prediction(struct logger *lg,
                                           const struct log_event *ev)
{
    (void)lg; (void)ev; return 0;
}
static inline int logger_loop(void *arg) { (void)arg; return 0; }
#endif

#endif
