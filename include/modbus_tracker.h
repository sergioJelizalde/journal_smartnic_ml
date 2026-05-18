#ifndef MODBUS_TRACKER_H
#define MODBUS_TRACKER_H

#include <stdint.h>
#include <stdbool.h>
#include <rte_hash.h>
#include <rte_malloc.h>

struct modbus_req_key {
    uint16_t trans_id;
    uint8_t unit_id;
} __attribute__((packed));

struct modbus_pending_request {
    uint16_t trans_id;
    uint8_t unit_id;
    uint16_t start_addr;
    uint16_t quantity;
    uint64_t tsc;
};

struct modbus_tracker {
    struct rte_hash *hash;
    struct modbus_pending_request *pool;
    uint32_t *free_stack;
    uint32_t capacity;
    uint32_t free_top;
    char name[64];
};

int modbus_tracker_init(struct modbus_tracker *t, const char *name,
                        uint32_t capacity, int socket_id);
void modbus_tracker_free(struct modbus_tracker *t);
int modbus_tracker_add(struct modbus_tracker *t, uint16_t trans_id,
                       uint8_t unit_id, uint16_t start_addr,
                       uint16_t quantity, uint64_t now_tsc);
int modbus_tracker_take(struct modbus_tracker *t, uint16_t trans_id,
                        uint8_t unit_id,
                        struct modbus_pending_request *out);

#endif
