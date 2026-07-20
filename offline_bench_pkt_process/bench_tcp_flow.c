/* SPDX-License-Identifier: BSD-3-Clause
 *
 * bench_tcp_flow.c — Case A: TCP-only per-flow statistics tracking
 *
 * Extracted from iot_mlp/iot_multicore.c with the inference path removed.
 *
 * Workload semantics preserved:
 *   - every IPv4/TCP packet is tracked (no port filter)
 *   - 5-tuple key, canonicalized so both directions map to one flow
 *   - per-flow: len min/max/sum, IAT min/max/sum, total bytes,
 *     popcount of TCP flags, over a window of N_PACKETS packets
 *   - on window completion: build the 8-feature vector (this cost stays,
 *     it is part of the data path) and fold into feature_sink
 *
 * Changes vs. the original (generalization):
 *   - windows roll over (reset + continue) instead of finalizing forever,
 *     so steady-state throughput includes the feature-build cost.
 *     Compile with -DTCPF_WINDOW_MODE=0 to restore finalize-once.
 *   - per-core hash + preallocated entry pool (as original), but sizes
 *     are compile-time tunable.
 */

#include "bench_common.h"

#ifndef TCPF_N_PACKETS
#define TCPF_N_PACKETS      6
#endif
#ifndef TCPF_MAX_FLOWS
#define TCPF_MAX_FLOWS      500000
#endif
#ifndef TCPF_WINDOW_MODE          /* 1 = rolling windows, 0 = finalize once */
#define TCPF_WINDOW_MODE    1
#endif

#define TCPF_NUM_FEATURES   8

struct flow_key {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t  protocol;
} __attribute__((packed));

struct flow_entry {
    uint64_t first_ts;
    uint64_t last_ts;
    uint16_t pkt_count;
    uint32_t len_min;
    uint32_t len_max;
    uint64_t len_sum;
    uint64_t iat_min;
    uint64_t iat_max;
    uint64_t iat_sum;
    uint64_t total_len;
    uint32_t flag_bits_sum;
    uint8_t  finalized;
} __rte_cache_aligned;

struct tcpf_state {
    struct rte_hash   *table;
    struct flow_entry *pool;
    uint32_t           next_free;
};

static inline void canonicalize_5tuple(struct flow_key *k)
{
    if (k->src_ip > k->dst_ip ||
       (k->src_ip == k->dst_ip && k->src_port > k->dst_port)) {
        uint32_t ip  = k->src_ip;   k->src_ip   = k->dst_ip;   k->dst_ip   = ip;
        uint16_t prt = k->src_port; k->src_port = k->dst_port; k->dst_port = prt;
    }
}

static inline void
update_flow(struct flow_entry *e, uint16_t len, uint64_t now, uint8_t fbits)
{
    uint64_t iat = (e->pkt_count > 0) ? (now - e->last_ts) : 0;

    if (e->pkt_count == 0) {
        e->len_min = len; e->len_max = len; e->len_sum = len;
        e->iat_min = UINT64_MAX; e->iat_max = 0; e->iat_sum = 0;
        e->first_ts = now;
        e->total_len = len;
        e->flag_bits_sum = fbits;
    } else {
        if (len < e->len_min) e->len_min = len;
        if (len > e->len_max) e->len_max = len;
        e->len_sum += len;
        if (iat < e->iat_min) e->iat_min = iat;
        if (iat > e->iat_max) e->iat_max = iat;
        e->iat_sum += iat;
        e->total_len += len;
        e->flag_bits_sum += fbits;
    }
    e->last_ts = now;
    e->pkt_count++;
}

static inline void
finalize_window(struct bench_worker *w, struct flow_entry *e)
{
    double hz = (double)rte_get_tsc_hz();
    float mean_len = (float)(e->len_sum / (double)e->pkt_count);
    float mean_iat = (float)(e->iat_sum / (double)(e->pkt_count - 1))
                     * 1e6f / (float)hz;

    float f[TCPF_NUM_FEATURES] = {
        (float)e->len_min,
        (float)e->len_max,
        mean_len,
        (float)(e->iat_min / hz * 1e6),
        (float)(e->iat_max / hz * 1e6),
        mean_iat,
        (float)e->total_len,
        (float)e->flag_bits_sum,
    };

    /* fold into sink so the compiler cannot drop feature construction */
    float acc = 0.f;
    for (int i = 0; i < TCPF_NUM_FEATURES; i++) acc += f[i];
    w->feature_sink += acc;
    w->windows_done++;

#if TCPF_WINDOW_MODE
    /* rolling: reset stats, keep flow entry alive */
    e->pkt_count = 0;
#else
    e->finalized = 1;
#endif
}

static inline void
handle_packet(struct bench_worker *w, struct tcpf_state *s,
              struct flow_key *key, uint16_t len, uint64_t now, uint8_t fbits)
{
    void *data = NULL;
    uint32_t idx;
    int ret = rte_hash_lookup_data(s->table, key, &data);

    if (ret < 0) {
        if (s->next_free >= TCPF_MAX_FLOWS) { w->insert_fail++; return; }
        idx = s->next_free++;
        memset(&s->pool[idx], 0, sizeof(s->pool[idx]));
        if (rte_hash_add_key_data(s->table, key,
                                  (void *)(uintptr_t)idx) < 0) {
            s->next_free--; w->insert_fail++; return;
        }
        w->table_inserts++;
    } else {
        idx = (uint32_t)(uintptr_t)data;
    }

    struct flow_entry *e = &s->pool[idx];
    if (e->finalized) return;

    update_flow(e, len, now, fbits);
    w->track_updates++;

    if (e->pkt_count == TCPF_N_PACKETS)
        finalize_window(w, e);
}

static void
tcpf_proc_burst(struct bench_worker *w, struct rte_mbuf **bufs,
                uint16_t nb, uint64_t now)
{
    struct tcpf_state *s = (struct tcpf_state *)w->app;

    for (uint16_t i = 0; i < nb; i++) {
        if (i + 2 < nb)
            rte_prefetch0(rte_pktmbuf_mtod(bufs[i + 2], void *));

        uint8_t *p = rte_pktmbuf_mtod(bufs[i], uint8_t *);
        uint32_t plen = rte_pktmbuf_pkt_len(bufs[i]);

        struct rte_ether_hdr *eth = (struct rte_ether_hdr *)p;
        if (rte_be_to_cpu_16(eth->ether_type) != RTE_ETHER_TYPE_IPV4)
            continue;

        uint32_t off = sizeof(struct rte_ether_hdr);
        if (plen < off + sizeof(struct rte_ipv4_hdr)) continue;
        struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(p + off);
        if (ip->next_proto_id != IPPROTO_TCP) continue;
        uint32_t ihl = (ip->version_ihl & RTE_IPV4_HDR_IHL_MASK)
                       * RTE_IPV4_IHL_MULTIPLIER;
        off += ihl;
        if (plen < off + sizeof(struct rte_tcp_hdr)) continue;
        struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)(p + off);

        w->hit_pkts++;

        struct flow_key key = {
            .src_ip   = rte_be_to_cpu_32(ip->src_addr),
            .dst_ip   = rte_be_to_cpu_32(ip->dst_addr),
            .src_port = rte_be_to_cpu_16(tcp->src_port),
            .dst_port = rte_be_to_cpu_16(tcp->dst_port),
            .protocol = IPPROTO_TCP,
        };
        canonicalize_5tuple(&key);

        uint16_t ip_len = rte_be_to_cpu_16(ip->total_length);
        uint8_t  fbits  = (uint8_t)__builtin_popcount(tcp->tcp_flags);

        if (bench_should_sample(w)) {
            uint64_t t0 = rte_rdtsc_precise();
            handle_packet(w, s, &key, ip_len, now, fbits);
            uint64_t t1 = rte_rdtsc_precise();
            bench_store_sample(w, t1 - t0);
        } else {
            handle_packet(w, s, &key, ip_len, now, fbits);
        }
    }
}

static int tcpf_app_init(struct bench_worker *w)
{
    struct tcpf_state *s = rte_zmalloc(NULL, sizeof(*s), 0);
    if (!s) return -1;

    char name[32];
    snprintf(name, sizeof(name), "tcpf_%u", w->core_id);
    struct rte_hash_parameters hp = {
        .name       = name,
        .entries    = TCPF_MAX_FLOWS,
        .key_len    = sizeof(struct flow_key),
        .hash_func  = rte_jhash,
        .socket_id  = (int)rte_socket_id(),
    };
    s->table = rte_hash_create(&hp);
    if (!s->table) return -1;

    s->pool = rte_zmalloc_socket(NULL,
                sizeof(struct flow_entry) * TCPF_MAX_FLOWS,
                RTE_CACHE_LINE_SIZE, rte_socket_id());
    if (!s->pool) return -1;
    s->next_free = 0;

    w->app = s;
    return 0;
}

static void tcpf_report(struct bench_worker *w, FILE *out)
{
    struct tcpf_state *s = (struct tcpf_state *)w->app;
    fprintf(out, "        core %u: active flows %u (pool used %u/%u)\n",
            w->core_id, rte_hash_count(s->table), s->next_free,
            (unsigned)TCPF_MAX_FLOWS);
}

static const struct bench_ops tcpf_ops = {
    .name       = "tcp_flow",
    .app_init   = tcpf_app_init,
    .proc_burst = tcpf_proc_burst,
    .app_report = tcpf_report,
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, &tcpf_ops);
}
