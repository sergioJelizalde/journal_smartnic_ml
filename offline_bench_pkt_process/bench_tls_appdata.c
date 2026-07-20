/* SPDX-License-Identifier: BSD-3-Clause
 *
 * bench_tls_appdata.c — Case B: TLS Application-Data parsing +
 *                       bidirectional per-flow feature tracking
 *
 * Extracted from doh_mlp/doh_multicore.c with the inference path removed.
 *
 * Workload semantics preserved:
 *   - filter: IPv4/TCP, port 443 either side, TLS record type 23
 *     (Application Data), record length parsed from the 5-byte TLS header
 *   - 5-tuple key (canonicalized); direction = client if src_port != 443
 *   - per-flow bidirectional stats: per-direction pkt counts, byte counts,
 *     record-length min/max/sum, plus direction-switch count
 *   - window of N_RECORDS TLS records -> build the 16-feature vector
 *     (kept, folded into feature_sink), then the window resets and the
 *     flow keeps accumulating (same as the original)
 *
 * Changes vs. the original (generalization):
 *   - free-list allocator kept, but expressed through the common counters
 *   - all magic numbers lifted to -D-overridable macros
 */

#include "bench_common.h"

#ifndef TLSB_N_RECORDS
#define TLSB_N_RECORDS     64
#endif
#ifndef TLSB_MAX_FLOWS
#define TLSB_MAX_FLOWS     65536
#endif
#ifndef TLSB_PORT
#define TLSB_PORT          443
#endif

#define TLSB_NUM_FEATURES  16
#define TLS_TYPE_APPDATA   23

struct flow_key {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t  protocol;
} __attribute__((packed));

struct flow_entry {
    uint32_t pkt_count_client;
    uint32_t pkt_count_server;
    uint64_t bytes_client;
    uint64_t bytes_server;

    uint32_t len_min_client;
    uint32_t len_max_client;
    uint32_t len_min_server;
    uint32_t len_max_server;
    uint64_t len_sum_client;
    uint64_t len_sum_server;

    uint8_t  last_direction;   /* 0 unknown, 1 client, 2 server */
    uint16_t dir_switches;
    uint64_t first_tsc;
    uint64_t last_tsc;

    uint32_t next_free;        /* free-list link */
} __rte_cache_aligned;

struct tlsb_state {
    struct rte_hash   *table;
    struct flow_entry *pool;
    uint32_t free_head;
    uint32_t free_count;
};

static inline void canonicalize_5tuple(struct flow_key *k)
{
    if (k->src_ip > k->dst_ip ||
       (k->src_ip == k->dst_ip && k->src_port > k->dst_port)) {
        uint32_t ip  = k->src_ip;   k->src_ip   = k->dst_ip;   k->dst_ip   = ip;
        uint16_t prt = k->src_port; k->src_port = k->dst_port; k->dst_port = prt;
    }
}

static inline uint32_t alloc_entry(struct tlsb_state *s)
{
    if (s->free_count == 0) return BENCH_INVALID_IDX;
    uint32_t idx = s->free_head;
    s->free_head = s->pool[idx].next_free;
    s->free_count--;
    struct flow_entry *e = &s->pool[idx];
    memset(e, 0, sizeof(*e));
    e->len_min_client = UINT32_MAX;
    e->len_min_server = UINT32_MAX;
    return idx;
}

static inline void free_entry(struct tlsb_state *s, uint32_t idx)
{
    s->pool[idx].next_free = s->free_head;
    s->free_head = idx;
    s->free_count++;
}

static inline void
update_flow(struct flow_entry *e, uint16_t rec_len, bool is_client,
            uint64_t now)
{
    uint32_t total = e->pkt_count_client + e->pkt_count_server;
    if (total == 0) e->first_tsc = now;

    if (is_client) {
        if (rec_len < e->len_min_client) e->len_min_client = rec_len;
        if (rec_len > e->len_max_client) e->len_max_client = rec_len;
        e->len_sum_client += rec_len;
        e->bytes_client   += rec_len;
        e->pkt_count_client++;
    } else {
        if (rec_len < e->len_min_server) e->len_min_server = rec_len;
        if (rec_len > e->len_max_server) e->len_max_server = rec_len;
        e->len_sum_server += rec_len;
        e->bytes_server   += rec_len;
        e->pkt_count_server++;
    }

    uint8_t dir = is_client ? 1 : 2;
    if (e->last_direction != 0 && e->last_direction != dir)
        e->dir_switches++;
    e->last_direction = dir;
    e->last_tsc = now;
}

static inline void
finalize_window(struct bench_worker *w, struct flow_entry *e)
{
    uint32_t nc = e->pkt_count_client, ns = e->pkt_count_server;
    uint32_t total = nc + ns;
    double cb = (double)e->bytes_client, sb = (double)e->bytes_server;
    double tb = cb + sb;

    uint32_t gmin = UINT32_MAX, gmax = 0;
    uint64_t gsum = 0;
    if (nc) {
        if (e->len_min_client < gmin) gmin = e->len_min_client;
        if (e->len_max_client > gmax) gmax = e->len_max_client;
        gsum += e->len_sum_client;
    }
    if (ns) {
        if (e->len_min_server < gmin) gmin = e->len_min_server;
        if (e->len_max_server > gmax) gmax = e->len_max_server;
        gsum += e->len_sum_server;
    }

    float f[TLSB_NUM_FEATURES];
    f[0]  = nc ? (float)e->len_max_client : 0.f;
    f[1]  = (float)nc;
    f[2]  = tb > 0.0 ? (float)(cb / tb) : 0.f;
    f[3]  = (float)ns;
    f[4]  = total ? (float)((double)nc / total) : 0.f;
    f[5]  = (float)cb;
    f[6]  = ns ? (float)e->len_max_server : 0.f;
    f[7]  = (gmin != UINT32_MAX) ? (float)gmin : 0.f;
    f[8]  = total ? (float)((double)gsum / total) : 0.f;
    f[9]  = ns ? (float)((double)e->len_sum_server / ns) : 0.f;
    f[10] = (float)e->dir_switches;
    f[11] = (float)sb;
    f[12] = (float)gmax;
    f[13] = (nc && e->len_min_client != UINT32_MAX)
              ? (float)e->len_min_client : 0.f;
    f[14] = (ns && e->len_min_server != UINT32_MAX)
              ? (float)e->len_min_server : 0.f;
    f[15] = nc ? (float)((double)e->len_sum_client / nc) : 0.f;

    float acc = 0.f;
    for (int i = 0; i < TLSB_NUM_FEATURES; i++) acc += f[i];
    w->feature_sink += acc;
    w->windows_done++;

    /* reset window, keep flow alive (original behavior) */
    uint64_t new_start = e->last_tsc;
    uint8_t keep_dir = 0; /* direction context resets like the original */
    memset(e, 0, offsetof(struct flow_entry, next_free));
    e->len_min_client = UINT32_MAX;
    e->len_min_server = UINT32_MAX;
    e->first_tsc = new_start;
    e->last_direction = keep_dir;
}

static inline void
handle_record(struct bench_worker *w, struct tlsb_state *s,
              struct flow_key *key, uint16_t rec_len, bool is_client)
{
    void *data = NULL;
    uint32_t idx;
    int ret = rte_hash_lookup_data(s->table, key, &data);

    if (ret < 0) {
        idx = alloc_entry(s);
        if (idx == BENCH_INVALID_IDX) { w->insert_fail++; return; }
        if (rte_hash_add_key_data(s->table, key,
                                  (void *)(uintptr_t)idx) < 0) {
            free_entry(s, idx); w->insert_fail++; return;
        }
        w->table_inserts++;
    } else {
        idx = (uint32_t)(uintptr_t)data;
    }

    struct flow_entry *e = &s->pool[idx];
    update_flow(e, rec_len, is_client, rte_rdtsc_precise());
    w->track_updates++;

    if (e->pkt_count_client + e->pkt_count_server >= TLSB_N_RECORDS)
        finalize_window(w, e);
}

static void
tlsb_proc_burst(struct bench_worker *w, struct rte_mbuf **bufs,
                uint16_t nb, uint64_t now)
{
    (void)now;
    struct tlsb_state *s = (struct tlsb_state *)w->app;

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
        off += (ip->version_ihl & RTE_IPV4_HDR_IHL_MASK)
               * RTE_IPV4_IHL_MULTIPLIER;
        if (plen < off + sizeof(struct rte_tcp_hdr)) continue;
        struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)(p + off);

        uint16_t sport = rte_be_to_cpu_16(tcp->src_port);
        uint16_t dport = rte_be_to_cpu_16(tcp->dst_port);
        if (sport != TLSB_PORT && dport != TLSB_PORT) continue;

        off += ((tcp->data_off >> 4) & 0x0F) * 4;
        if (plen < off + 5) continue;          /* need TLS record header */

        uint8_t *tls = p + off;
        if (tls[0] != TLS_TYPE_APPDATA) continue;

        uint16_t rec_len = (uint16_t)((tls[3] << 8) | tls[4]);
        if ((uint32_t)rec_len > plen - off - 5) continue;

        w->hit_pkts++;

        struct flow_key key = {
            .src_ip   = ip->src_addr,   /* network order, as original */
            .dst_ip   = ip->dst_addr,
            .src_port = sport,
            .dst_port = dport,
            .protocol = IPPROTO_TCP,
        };
        canonicalize_5tuple(&key);
        bool is_client = (sport != TLSB_PORT);

        if (bench_should_sample(w)) {
            uint64_t t0 = rte_rdtsc_precise();
            handle_record(w, s, &key, rec_len, is_client);
            uint64_t t1 = rte_rdtsc_precise();
            bench_store_sample(w, t1 - t0);
        } else {
            handle_record(w, s, &key, rec_len, is_client);
        }
    }
}

static int tlsb_app_init(struct bench_worker *w)
{
    struct tlsb_state *s = rte_zmalloc(NULL, sizeof(*s), 0);
    if (!s) return -1;

    char name[32];
    snprintf(name, sizeof(name), "tlsb_%u", w->core_id);
    struct rte_hash_parameters hp = {
        .name      = name,
        .entries   = TLSB_MAX_FLOWS,
        .key_len   = sizeof(struct flow_key),
        .hash_func = rte_jhash,
        .socket_id = (int)rte_socket_id(),
    };
    s->table = rte_hash_create(&hp);
    if (!s->table) return -1;

    s->pool = rte_zmalloc_socket(NULL,
                sizeof(struct flow_entry) * TLSB_MAX_FLOWS,
                RTE_CACHE_LINE_SIZE, rte_socket_id());
    if (!s->pool) return -1;

    /* build free list */
    for (uint32_t i = 0; i < TLSB_MAX_FLOWS - 1; i++)
        s->pool[i].next_free = i + 1;
    s->pool[TLSB_MAX_FLOWS - 1].next_free = BENCH_INVALID_IDX;
    s->free_head  = 0;
    s->free_count = TLSB_MAX_FLOWS;

    w->app = s;
    return 0;
}

static void tlsb_report(struct bench_worker *w, FILE *out)
{
    struct tlsb_state *s = (struct tlsb_state *)w->app;
    fprintf(out, "        core %u: active flows %u (free %u/%u)\n",
            w->core_id, rte_hash_count(s->table), s->free_count,
            (unsigned)TLSB_MAX_FLOWS);
}

static const struct bench_ops tlsb_ops = {
    .name       = "tls_appdata",
    .app_init   = tlsb_app_init,
    .proc_burst = tlsb_proc_burst,
    .app_report = tlsb_report,
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, &tlsb_ops);
}
