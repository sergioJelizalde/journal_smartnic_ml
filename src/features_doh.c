#include <stdio.h>
#include <string.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include "worker.h"
#include "profiles.h"
#include "packet_parser.h"
#include "flow_pool.h"

struct doh_flow_entry {
    uint32_t pkt_count_client;
    uint32_t pkt_count_server;
    uint64_t bytes_client;
    uint64_t bytes_server;
    uint32_t min_client;
    uint32_t max_client;
    uint32_t min_server;
    uint32_t max_server;
    uint64_t sum_client;
    uint64_t sum_server;
    uint8_t last_dir;
    uint16_t dir_switches;
};

struct doh_state {
    struct flow_pool flows;
    struct feature_queue q;
    uint16_t window;
};

static void doh_reset_entry(struct doh_flow_entry *e)
{
    memset(e, 0, sizeof(*e));
    e->min_client = UINT32_MAX;
    e->min_server = UINT32_MAX;
}

static int doh_init_worker(struct worker_ctx *w)
{
    struct doh_state *s = rte_zmalloc_socket("doh_state", sizeof(*s),
                                             RTE_CACHE_LINE_SIZE,
                                             rte_lcore_to_socket_id(w->lcore_id));
    if (!s) return -1;
    char name[64];
    snprintf(name, sizeof(name), "doh_flows_%u", w->worker_id);
    if (flow_pool_init(&s->flows, name, sizeof(struct flow_key5),
                       sizeof(struct doh_flow_entry),
                       w->cfg->max_flows_per_worker,
                       rte_lcore_to_socket_id(w->lcore_id)) != 0) {
        rte_free(s);
        return -1;
    }
    feature_queue_init(&s->q);
    s->window = w->cfg->window_packets ? w->cfg->window_packets : 16;
    w->profile_state = s;
    return 0;
}

static void doh_free_worker(struct worker_ctx *w)
{
    struct doh_state *s = (struct doh_state *)w->profile_state;
    if (!s) return;
    flow_pool_free(&s->flows);
    rte_free(s);
    w->profile_state = NULL;
}

static void doh_update(struct doh_flow_entry *e, uint16_t tls_len, int client_dir)
{
    if (client_dir) {
        if (e->pkt_count_client == 0) {
            e->min_client = tls_len;
            e->max_client = tls_len;
        } else {
            if (tls_len < e->min_client) e->min_client = tls_len;
            if (tls_len > e->max_client) e->max_client = tls_len;
        }
        e->sum_client += tls_len;
        e->bytes_client += tls_len;
        e->pkt_count_client++;
    } else {
        if (e->pkt_count_server == 0) {
            e->min_server = tls_len;
            e->max_server = tls_len;
        } else {
            if (tls_len < e->min_server) e->min_server = tls_len;
            if (tls_len > e->max_server) e->max_server = tls_len;
        }
        e->sum_server += tls_len;
        e->bytes_server += tls_len;
        e->pkt_count_server++;
    }
    uint8_t d = client_dir ? 1 : 2;
    if (e->last_dir != 0 && e->last_dir != d) e->dir_switches++;
    e->last_dir = d;
}

static void doh_emit(struct doh_state *s,
                     const struct flow_key5 *key,
                     const struct doh_flow_entry *e)
{
    uint32_t n_client = e->pkt_count_client;
    uint32_t n_server = e->pkt_count_server;
    uint32_t total_pkts = n_client + n_server;
    double client_bytes = (double)e->bytes_client;
    double server_bytes = (double)e->bytes_server;
    double total_bytes = client_bytes + server_bytes;

    uint32_t global_min = UINT32_MAX;
    uint32_t global_max = 0;
    uint64_t global_sum = 0;
    if (n_client) {
        if (e->min_client < global_min) global_min = e->min_client;
        if (e->max_client > global_max) global_max = e->max_client;
        global_sum += e->sum_client;
    }
    if (n_server) {
        if (e->min_server < global_min) global_min = e->min_server;
        if (e->max_server > global_max) global_max = e->max_server;
        global_sum += e->sum_server;
    }

    struct feature_vector fv;
    struct feature_id id;
    memset(&fv, 0, sizeof(fv));
    memset(&id, 0, sizeof(id));
    fv.n = 16;
    fv.v[0] = n_client ? (float)e->max_client : 0.0f;
    fv.v[1] = (float)n_client;
    fv.v[2] = total_bytes > 0.0 ? (float)(client_bytes / total_bytes) : 0.0f;
    fv.v[3] = (float)n_server;
    fv.v[4] = total_pkts ? (float)((double)n_client / (double)total_pkts) : 0.0f;
    fv.v[5] = (float)client_bytes;
    fv.v[6] = n_server ? (float)e->max_server : 0.0f;
    fv.v[7] = (global_min == UINT32_MAX) ? 0.0f : (float)global_min;
    fv.v[8] = total_pkts ? (float)((double)global_sum / (double)total_pkts) : 0.0f;
    fv.v[9] = n_server ? (float)((double)e->sum_server / (double)n_server) : 0.0f;
    fv.v[10] = (float)e->dir_switches;
    fv.v[11] = (float)server_bytes;
    fv.v[12] = (float)global_max;
    fv.v[13] = n_client ? (float)e->min_client : 0.0f;
    fv.v[14] = n_server ? (float)e->min_server : 0.0f;
    fv.v[15] = n_client ? (float)((double)e->sum_client / (double)n_client) : 0.0f;
    id.type = FEATURE_ID_FLOW5;
    id.u.flow = *key;
    feature_queue_push(&s->q, &fv, &id);
}

static int doh_process_packet(struct worker_ctx *w, struct rte_mbuf *m,
                              uint64_t now_tsc)
{
    (void)now_tsc;
    struct doh_state *s = (struct doh_state *)w->profile_state;
    struct packet_view p;
    if (!packet_parse_ipv4_tcp(m, &p)) return 0;
    if (p.src_port != 443 && p.dst_port != 443) return 0;
    if (p.payload_len < 5) return 0;

    uint8_t tls_type = p.payload[0];
    if (tls_type != 23) return 0; /* TLS Application Data */
    uint16_t tls_record_len = (uint16_t)(((uint16_t)p.payload[3] << 8) | p.payload[4]);
    if ((uint32_t)tls_record_len > p.payload_len - 5) return 0;

    struct flow_key5 key;
    flow_key5_from_packet(&p, &key);
    flow_key5_canonicalize(&key);

    bool is_new = false;
    struct doh_flow_entry *e = flow_pool_lookup_or_alloc(&s->flows, &key, NULL, &is_new);
    if (!e) return 0;
    if (is_new) doh_reset_entry(e);

    int client_dir = (p.src_port != 443);
    doh_update(e, tls_record_len, client_dir);

    if (e->pkt_count_client + e->pkt_count_server >= s->window) {
        doh_emit(s, &key, e);
        doh_reset_entry(e);
    }
    return 0;
}

static int doh_next_feature(struct worker_ctx *w,
                            struct feature_vector *fv,
                            struct feature_id *id)
{
    struct doh_state *s = (struct doh_state *)w->profile_state;
    return feature_queue_pop(&s->q, fv, id);
}

const struct profile_ops doh_profile_ops = {
    .name = "doh_bidirectional",
    .type = APP_PROFILE_DOH,
    .feature_count = 16,
    .init_worker = doh_init_worker,
    .free_worker = doh_free_worker,
    .process_packet = doh_process_packet,
    .next_feature = doh_next_feature,
};
