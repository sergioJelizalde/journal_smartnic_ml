#ifndef PROFILES_H
#define PROFILES_H

#include "common.h"
#include "app_config.h"

struct worker_ctx;

#define APP_FEATURE_QUEUE_SIZE 64

struct feature_queue_item {
    struct feature_vector fv;
    struct feature_id id;
};

struct feature_queue {
    struct feature_queue_item items[APP_FEATURE_QUEUE_SIZE];
    uint16_t head;
    uint16_t tail;
    uint16_t count;
    uint64_t dropped;
};

static inline void feature_queue_init(struct feature_queue *q)
{
    q->head = q->tail = q->count = 0;
    q->dropped = 0;
}

static inline int feature_queue_push(struct feature_queue *q,
                                     const struct feature_vector *fv,
                                     const struct feature_id *id)
{
    if (q->count == APP_FEATURE_QUEUE_SIZE) {
        q->dropped++;
        return -1;
    }
    q->items[q->tail].fv = *fv;
    q->items[q->tail].id = *id;
    q->tail = (uint16_t)((q->tail + 1) % APP_FEATURE_QUEUE_SIZE);
    q->count++;
    return 0;
}

static inline int feature_queue_pop(struct feature_queue *q,
                                    struct feature_vector *fv,
                                    struct feature_id *id)
{
    if (q->count == 0) return 0;
    *fv = q->items[q->head].fv;
    *id = q->items[q->head].id;
    q->head = (uint16_t)((q->head + 1) % APP_FEATURE_QUEUE_SIZE);
    q->count--;
    return 1;
}

struct profile_ops {
    const char *name;
    enum app_profile_type type;
    uint16_t feature_count;
    int (*init_worker)(struct worker_ctx *w);
    void (*free_worker)(struct worker_ctx *w);
    int (*process_packet)(struct worker_ctx *w, struct rte_mbuf *m,
                          uint64_t now_tsc);
    int (*next_feature)(struct worker_ctx *w, struct feature_vector *fv,
                        struct feature_id *id);
};

const struct profile_ops *profile_get(enum app_profile_type type);

extern const struct profile_ops iot_profile_ops;
extern const struct profile_ops doh_profile_ops;
extern const struct profile_ops ics_profile_ops;

#endif
