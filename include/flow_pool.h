#ifndef FLOW_POOL_H
#define FLOW_POOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <rte_hash.h>
#include <rte_malloc.h>
#include "common.h"

struct flow_pool {
    struct rte_hash *hash;
    void *entries;
    uint32_t capacity;
    uint32_t next_free;
    size_t entry_size;
    size_t key_size;
    char name[64];
};

int flow_pool_init(struct flow_pool *p,
                   const char *name,
                   size_t key_size,
                   size_t entry_size,
                   uint32_t capacity,
                   int socket_id);
void flow_pool_free(struct flow_pool *p);
void *flow_pool_lookup_or_alloc(struct flow_pool *p, const void *key,
                                uint32_t *index_out, bool *is_new_out);
void *flow_pool_lookup(struct flow_pool *p, const void *key, uint32_t *index_out);
int flow_pool_del(struct flow_pool *p, const void *key);
uint32_t flow_pool_count(const struct flow_pool *p);

#endif
