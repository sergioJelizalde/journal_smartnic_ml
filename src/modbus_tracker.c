#include <stdio.h>
#include <string.h>
#include <rte_hash_crc.h>
#include "modbus_tracker.h"

int modbus_tracker_init(struct modbus_tracker *t, const char *name,
                        uint32_t capacity, int socket_id)
{
    memset(t, 0, sizeof(*t));
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->capacity = capacity;
    t->free_top = capacity;

    struct rte_hash_parameters hp = {
        .name = t->name,
        .entries = capacity,
        .key_len = sizeof(struct modbus_req_key),
        .hash_func = rte_hash_crc,
        .hash_func_init_val = 0,
        .socket_id = socket_id,
    };
    t->hash = rte_hash_create(&hp);
    if (!t->hash) return -1;

    t->pool = rte_zmalloc_socket(name, (size_t)capacity * sizeof(*t->pool),
                                 RTE_CACHE_LINE_SIZE, socket_id);
    t->free_stack = rte_zmalloc_socket(name, (size_t)capacity * sizeof(*t->free_stack),
                                       RTE_CACHE_LINE_SIZE, socket_id);
    if (!t->pool || !t->free_stack) {
        modbus_tracker_free(t);
        return -1;
    }
    for (uint32_t i = 0; i < capacity; i++) t->free_stack[i] = capacity - 1 - i;
    return 0;
}

void modbus_tracker_free(struct modbus_tracker *t)
{
    if (!t) return;
    if (t->hash) rte_hash_free(t->hash);
    if (t->pool) rte_free(t->pool);
    if (t->free_stack) rte_free(t->free_stack);
    memset(t, 0, sizeof(*t));
}

int modbus_tracker_add(struct modbus_tracker *t, uint16_t trans_id,
                       uint8_t unit_id, uint16_t start_addr,
                       uint16_t quantity, uint64_t now_tsc)
{
    struct modbus_req_key key = {.trans_id = trans_id, .unit_id = unit_id};
    void *data = NULL;
    int ret = rte_hash_lookup_data(t->hash, &key, &data);
    uint32_t idx;
    if (ret >= 0) {
        idx = (uint32_t)(uintptr_t)data;
    } else {
        if (t->free_top == 0) return -1;
        idx = t->free_stack[--t->free_top];
        ret = rte_hash_add_key_data(t->hash, &key, (void *)(uintptr_t)idx);
        if (ret < 0) {
            t->free_stack[t->free_top++] = idx;
            return -1;
        }
    }
    t->pool[idx].trans_id = trans_id;
    t->pool[idx].unit_id = unit_id;
    t->pool[idx].start_addr = start_addr;
    t->pool[idx].quantity = quantity;
    t->pool[idx].tsc = now_tsc;
    return 0;
}

int modbus_tracker_take(struct modbus_tracker *t, uint16_t trans_id,
                        uint8_t unit_id,
                        struct modbus_pending_request *out)
{
    struct modbus_req_key key = {.trans_id = trans_id, .unit_id = unit_id};
    void *data = NULL;
    int ret = rte_hash_lookup_data(t->hash, &key, &data);
    if (ret < 0) return 0;
    uint32_t idx = (uint32_t)(uintptr_t)data;
    if (out) *out = t->pool[idx];
    rte_hash_del_key(t->hash, &key);
    if (t->free_top < t->capacity) t->free_stack[t->free_top++] = idx;
    return 1;
}
