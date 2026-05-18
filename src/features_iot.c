#include <stdio.h>
#include <string.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include "worker.h"
#include "profiles.h"
#include "packet_parser.h"
#include "flow_pool.h"

struct iot_flow_entry {
    uint64_t first_tsc;
    uint64_t last_tsc;
    uint16_t pkt_count;
    uint32_t len_min;
    uint32_t len_max;
    uint64_t len_sum;
    uint64_t iat_min;
    uint64_t iat_max;
    uint64_t iat_sum;
    uint64_t total_bytes;
    uint32_t tcp_flag_bits_sum;
};

struct iot_state {
    struct flow_pool flows;
    struct feature_queue q;
    uint16_t window;
    double tsc_hz;
};

static void iot_reset_entry(struct iot_flow_entry *e)
{
    memset(e, 0, sizeof(*e));
    e->len_min = UINT32_MAX;
    e->iat_min = UINT64_MAX;
}

static int iot_init_worker(struct worker_ctx *w)
{
    struct iot_state *s = rte_zmalloc_socket("iot_state", sizeof(*s),
                                             RTE_CACHE_LINE_SIZE,
                                             rte_lcore_to_socket_id(w->lcore_id));
    if (!s) return -1;
    char name[64];
    snprintf(name, sizeof(name), "iot_flows_%u", w->worker_id);
    if (flow_pool_init(&s->flows, name, sizeof(struct flow_key5),
                       sizeof(struct iot_flow_entry),
                       w->cfg->max_flows_per_worker,
                       rte_lcore_to_socket_id(w->lcore_id)) != 0) {
        rte_free(s);
        return -1;
    }
    feature_queue_init(&s->q);
    s->window = w->cfg->window_packets ? w->cfg->window_packets : 8;
    s->tsc_hz = (double)rte_get_tsc_hz();
    w->profile_state = s;
    return 0;
}

static void iot_free_worker(struct worker_ctx *w)
{
    struct iot_state *s = (struct iot_state *)w->profile_state;
    if (!s) return;
    flow_pool_free(&s->flows);
    rte_free(s);
    w->profile_state = NULL;
}

static void iot_update(struct iot_flow_entry *e,
                       uint16_t pkt_len,
                       uint64_t now,
                       uint8_t flag_bits)
{
    uint64_t iat = e->pkt_count ? (now - e->last_tsc) : 0;
    if (e->pkt_count == 0) {
        e->first_tsc = now;
        e->len_min = pkt_len;
        e->len_max = pkt_len;
        e->len_sum = pkt_len;
        e->iat_min = UINT64_MAX;
        e->iat_max = 0;
        e->iat_sum = 0;
        e->total_bytes = pkt_len;
        e->tcp_flag_bits_sum = flag_bits;
    } else {
        if (pkt_len < e->len_min) e->len_min = pkt_len;
        if (pkt_len > e->len_max) e->len_max = pkt_len;
        e->len_sum += pkt_len;
        if (iat < e->iat_min) e->iat_min = iat;
        if (iat > e->iat_max) e->iat_max = iat;
        e->iat_sum += iat;
        e->total_bytes += pkt_len;
        e->tcp_flag_bits_sum += flag_bits;
    }
    e->last_tsc = now;
    e->pkt_count++;
}

static void iot_emit(struct iot_state *s,
                     const struct flow_key5 *key,
                     struct iot_flow_entry *e)
{
    struct feature_vector fv;
    struct feature_id id;
    memset(&fv, 0, sizeof(fv));
    memset(&id, 0, sizeof(id));
    fv.n = 8;
    float mean_len = (float)((double)e->len_sum / (double)e->pkt_count);
    float mean_iat = 0.0f;
    if (e->pkt_count > 1) {
        mean_iat = (float)(((double)e->iat_sum / (double)(e->pkt_count - 1)) * 1e6 / s->tsc_hz);
    }
    fv.v[0] = (float)e->len_min;
    fv.v[1] = (float)e->len_max;
    fv.v[2] = mean_len;
    fv.v[3] = (e->iat_min == UINT64_MAX) ? 0.0f : (float)((double)e->iat_min * 1e6 / s->tsc_hz);
    fv.v[4] = (float)((double)e->iat_max * 1e6 / s->tsc_hz);
    fv.v[5] = mean_iat;
    fv.v[6] = (float)e->total_bytes;
    fv.v[7] = (float)e->tcp_flag_bits_sum;
    id.type = FEATURE_ID_FLOW5;
    id.u.flow = *key;
    feature_queue_push(&s->q, &fv, &id);
}

static int iot_process_packet(struct worker_ctx *w, struct rte_mbuf *m,
                              uint64_t now_tsc)
{
    struct iot_state *s = (struct iot_state *)w->profile_state;
    struct packet_view p;
    if (!packet_parse_ipv4_tcp(m, &p)) return 0;

    struct flow_key5 key;
    flow_key5_from_packet(&p, &key);
    flow_key5_canonicalize(&key);

    bool is_new = false;
    struct iot_flow_entry *e = flow_pool_lookup_or_alloc(&s->flows, &key, NULL, &is_new);
    if (!e) return 0;
    if (is_new) iot_reset_entry(e);

    uint16_t ip_total = rte_be_to_cpu_16(p.ip4->total_length);
    uint16_t pkt_len = ip_total ? ip_total : (uint16_t)p.packet_len;
    iot_update(e, pkt_len, now_tsc, app_popcount8(p.tcp_flags));

    if (e->pkt_count >= s->window) {
        iot_emit(s, &key, e);
        iot_reset_entry(e);
    }
    return 0;
}

static int iot_next_feature(struct worker_ctx *w,
                            struct feature_vector *fv,
                            struct feature_id *id)
{
    struct iot_state *s = (struct iot_state *)w->profile_state;
    return feature_queue_pop(&s->q, fv, id);
}

const struct profile_ops iot_profile_ops = {
    .name = "iot_tcp",
    .type = APP_PROFILE_IOT,
    .feature_count = 8,
    .init_worker = iot_init_worker,
    .free_worker = iot_free_worker,
    .process_packet = iot_process_packet,
    .next_feature = iot_next_feature,
};
