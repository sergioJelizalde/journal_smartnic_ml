#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <rte_hash_crc.h>
#include <rte_jhash.h>
#include <rte_lcore.h>
#include "flow_pool.h"

int flow_pool_init(struct flow_pool *p,
                   const char *name,
                   size_t key_size,
                   size_t entry_size,
                   uint32_t capacity,
                   int socket_id)
{
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "%s", name);
    p->capacity = capacity;
    p->entry_size = entry_size;
    p->key_size = key_size;
    p->next_free = 0;

    struct rte_hash_parameters hp = {
        .name = p->name,
        .entries = capacity,
        .key_len = (uint32_t)key_size,
        .hash_func = rte_hash_crc,
        .hash_func_init_val = 0,
        .socket_id = socket_id,
    };
    p->hash = rte_hash_create(&hp);
    if (!p->hash) return -1;

    p->entries = rte_zmalloc_socket(p->name, (size_t)capacity * entry_size,
                                    RTE_CACHE_LINE_SIZE, socket_id);
    if (!p->entries) {
        rte_hash_free(p->hash);
        p->hash = NULL;
        return -1;
    }
    return 0;
}

void flow_pool_free(struct flow_pool *p)
{
    if (!p) return;
    if (p->hash) rte_hash_free(p->hash);
    if (p->entries) rte_free(p->entries);
    memset(p, 0, sizeof(*p));
}

void *flow_pool_lookup(struct flow_pool *p, const void *key, uint32_t *index_out)
{
    void *data = NULL;
    int ret = rte_hash_lookup_data(p->hash, key, &data);
    if (ret < 0) return NULL;
    uint32_t idx = (uint32_t)(uintptr_t)data;
    if (index_out) *index_out = idx;
    return (uint8_t *)p->entries + ((size_t)idx * p->entry_size);
}

void *flow_pool_lookup_or_alloc(struct flow_pool *p, const void *key,
                                uint32_t *index_out, bool *is_new_out)
{
    void *data = NULL;
    int ret = rte_hash_lookup_data(p->hash, key, &data);
    if (ret >= 0) {
        uint32_t idx = (uint32_t)(uintptr_t)data;
        if (index_out) *index_out = idx;
        if (is_new_out) *is_new_out = false;
        return (uint8_t *)p->entries + ((size_t)idx * p->entry_size);
    }

    if (p->next_free >= p->capacity) return NULL;
    uint32_t idx = p->next_free++;
    ret = rte_hash_add_key_data(p->hash, key, (void *)(uintptr_t)idx);
    if (ret < 0) {
        p->next_free--;
        return NULL;
    }

    void *entry = (uint8_t *)p->entries + ((size_t)idx * p->entry_size);
    memset(entry, 0, p->entry_size);
    if (index_out) *index_out = idx;
    if (is_new_out) *is_new_out = true;
    return entry;
}

int flow_pool_del(struct flow_pool *p, const void *key)
{
    return rte_hash_del_key(p->hash, key);
}

uint32_t flow_pool_count(const struct flow_pool *p)
{
    return p && p->hash ? (uint32_t)rte_hash_count(p->hash) : 0;
}
