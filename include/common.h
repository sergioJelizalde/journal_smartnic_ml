#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <inttypes.h>

#include <rte_common.h>
#include <rte_byteorder.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>
#include <rte_branch_prediction.h>

#include "app_config.h"

#define APP_CACHE_ALIGNED __rte_cache_aligned
#define APP_ALIGN16 __attribute__((aligned(16)))
#define APP_UNUSED(x) (void)(x)

struct flow_key5 {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;
} __attribute__((packed));

struct sensor_key {
    uint8_t unit_id;
    uint16_t reg_addr;
} __attribute__((packed));

struct packet_view {
    struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ip4;
    struct rte_tcp_hdr *tcp;
    uint8_t *payload;
    uint32_t packet_len;
    uint32_t l2_len;
    uint32_t ip_hlen;
    uint32_t tcp_hlen;
    uint32_t payload_len;
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;
    uint8_t tcp_flags;
};

struct feature_vector {
    float v[APP_MAX_FEATURES];
    uint16_t n;
};

enum feature_id_type {
    FEATURE_ID_NONE = 0,
    FEATURE_ID_FLOW5 = 1,
    FEATURE_ID_SENSOR = 2
};

struct feature_id {
    enum feature_id_type type;
    union {
        struct flow_key5 flow;
        struct sensor_key sensor;
    } u;
};

static inline void flow_key5_canonicalize(struct flow_key5 *k)
{
    if (k->src_ip > k->dst_ip ||
        (k->src_ip == k->dst_ip && k->src_port > k->dst_port)) {
        uint32_t ip = k->src_ip;
        uint16_t port = k->src_port;
        k->src_ip = k->dst_ip;
        k->src_port = k->dst_port;
        k->dst_ip = ip;
        k->dst_port = port;
    }
}

static inline void flow_key5_from_packet(const struct packet_view *p,
                                         struct flow_key5 *k)
{
    k->src_ip = p->src_ip;
    k->dst_ip = p->dst_ip;
    k->src_port = p->src_port;
    k->dst_port = p->dst_port;
    k->proto = p->proto;
}

static inline void packet_swap_l2(struct rte_mbuf *m)
{
    struct rte_ether_hdr *eth = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
    struct rte_ether_addr tmp;
    rte_ether_addr_copy(&eth->src_addr, &tmp);
    rte_ether_addr_copy(&eth->dst_addr, &eth->src_addr);
    rte_ether_addr_copy(&tmp, &eth->dst_addr);
}

static inline uint8_t app_popcount8(uint8_t x)
{
    return (uint8_t)__builtin_popcount((unsigned)x);
}

#endif
