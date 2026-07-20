/* SPDX-License-Identifier: BSD-3-Clause
 *
 * bench_modbus.c — Case C: Modbus-TCP transaction matching +
 *                  per-sensor windowed tracking with CUSUM/EWMA state
 *
 * Extracted from ics_modbus/modbus.c with the inference path removed.
 *
 * Workload semantics preserved:
 *   - filter: IPv4/TCP, port 502 either side, valid MBAP header
 *   - requests (fc 0x03/0x04) recorded in a pending-request tracker keyed
 *     by (trans_id, unit_id) with (start_addr, quantity)
 *   - responses matched against pending requests; each register value is
 *     decoded (16-bit, with the original 32-bit pairing heuristic) and fed
 *     into a per-(unit_id, reg_addr) sensor stream
 *   - sensor stream: circular window of WINDOW_SIZE readings+timestamps;
 *     on window completion the full 16-feature CUSUM/EWMA vector is built
 *     (kept — folded into feature_sink) and the persistent EWMA/CUSUM/
 *     historical-mean state is updated across windows
 *
 * Changes vs. the original (generalization / correctness):
 *   - the request tracker is PER-CORE and lock-free instead of a global
 *     spinlocked array.  The symmetric RSS key in bench_common.h
 *     guarantees request and response of the same TCP connection land on
 *     the same queue/core, so no cross-core sharing is needed.  This
 *     removes the spinlock from the hot path (the original serialized all
 *     cores on one lock per Modbus PDU).
 *   - tracker is a small ring with linear search (bounded, cache-hot);
 *     stale entries are overwritten instead of the "drop half" heuristic.
 */

#include "bench_common.h"
#include <math.h>

#ifndef MB_WINDOW_SIZE
#define MB_WINDOW_SIZE     32
#endif
#ifndef MB_MAX_SENSORS
#define MB_MAX_SENSORS     500000
#endif
#ifndef MB_PENDING_CAP           /* per-core pending-request ring */
#define MB_PENDING_CAP     1024
#endif
#ifndef MB_PORT
#define MB_PORT            502
#endif

#define MB_NUM_FEATURES    16
#define EWMA_ALPHA         0.3f
#define CUSUM_ALLOWANCE    0.5f
#define CUSUM_THRESHOLD    2.0f

/* --- protocol structs --- */
struct modbus_mbap {
    uint16_t trans_id;
    uint16_t protocol_id;
    uint16_t length;
    uint8_t  unit_id;
} __attribute__((packed));

struct modbus_request_pdu {
    uint8_t  function_code;
    uint16_t start_addr;
    uint16_t quantity;
} __attribute__((packed));

struct modbus_response_pdu {
    uint8_t function_code;
    uint8_t byte_count;
} __attribute__((packed));

/* --- tracking structs --- */
struct sensor_key {
    uint8_t  unit_id;
    uint16_t reg_addr;
} __attribute__((packed));

struct sensor_entry {
    uint64_t timestamps[MB_WINDOW_SIZE];
    int32_t  readings[MB_WINDOW_SIZE];
    uint16_t sample_count;
    uint16_t write_index;
    /* persistent CUSUM/EWMA state across windows */
    float    ewma_value;
    float    cusum_positive;
    float    cusum_negative;
    float    historical_mean;
    uint32_t total_samples;
} __rte_cache_aligned;

struct pending_request {
    uint16_t trans_id;
    uint8_t  unit_id;
    uint8_t  valid;
    uint16_t start_addr;
    uint16_t num_registers;
    uint64_t timestamp;
};

struct mb_state {
    struct rte_hash     *table;
    struct sensor_entry *pool;
    uint32_t             next_free;
    /* per-core lock-free ring of pending requests */
    struct pending_request pending[MB_PENDING_CAP];
    uint32_t pending_wr;
    uint64_t req_seen, resp_matched, resp_unmatched, readings;
};

/* --- pending request tracker (per-core, no lock) --- */
static inline void
add_pending(struct mb_state *s, uint16_t tid, uint8_t uid,
            uint16_t addr, uint16_t qty, uint64_t now)
{
    struct pending_request *r = &s->pending[s->pending_wr];
    s->pending_wr = (s->pending_wr + 1) & (MB_PENDING_CAP - 1);
    r->trans_id = tid; r->unit_id = uid; r->valid = 1;
    r->start_addr = addr; r->num_registers = qty; r->timestamp = now;
}

static inline int
find_pending(struct mb_state *s, uint16_t tid, uint8_t uid,
             struct pending_request *out)
{
    for (uint32_t i = 0; i < MB_PENDING_CAP; i++) {
        struct pending_request *r = &s->pending[i];
        if (r->valid && r->trans_id == tid && r->unit_id == uid) {
            *out = *r;
            r->valid = 0;
            return 1;
        }
    }
    return 0;
}

/* --- feature computation (identical math to the original) --- */
static double
slope_seconds(const uint64_t *ts, const int32_t *ys, int n, double hz)
{
    double sumt = 0, sumy = 0, sumty = 0, sumt2 = 0;
    for (int i = 0; i < n; i++) {
        double t = (double)(ts[i] - ts[0]) / hz;
        double y = (double)ys[i];
        sumt += t; sumy += y; sumty += t * y; sumt2 += t * t;
    }
    double denom = (double)n * sumt2 - sumt * sumt;
    if (denom == 0.0) return 0.0;
    return ((double)n * sumty - sumt * sumy) / denom;
}

static void
compute_features(struct sensor_entry *e, float f[MB_NUM_FEATURES], double hz)
{
    /* base 8 */
    int32_t minv = INT32_MAX, maxv = INT32_MIN;
    double sum = 0.0;
    for (int i = 0; i < MB_WINDOW_SIZE; i++) {
        int32_t v = e->readings[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
    }
    double mean = sum / MB_WINDOW_SIZE;

    uint64_t iat_min = UINT64_MAX, iat_max = 0;
    double iat_sum = 0.0;
    for (int i = 1; i < MB_WINDOW_SIZE; i++) {
        uint64_t dt = (e->timestamps[i] > e->timestamps[i - 1])
                      ? e->timestamps[i] - e->timestamps[i - 1] : 0;
        if (dt < iat_min) iat_min = dt;
        if (dt > iat_max) iat_max = dt;
        iat_sum += (double)dt;
    }

    f[0] = (float)minv;
    f[1] = (float)maxv;
    f[2] = (float)mean;
    f[3] = (float)((double)iat_min / hz * 1e6);
    f[4] = (float)((double)iat_max / hz * 1e6);
    f[5] = (float)((iat_sum / (MB_WINDOW_SIZE - 1)) / hz * 1e6);
    f[6] = (float)(maxv - minv);
    f[7] = (float)slope_seconds(e->timestamps, e->readings,
                                MB_WINDOW_SIZE, hz);

    /* EWMA */
    float ewma = (float)e->readings[0];
    for (int i = 1; i < MB_WINDOW_SIZE; i++)
        ewma = EWMA_ALPHA * (float)e->readings[i] + (1.f - EWMA_ALPHA) * ewma;
    float dev = ewma - e->ewma_value;
    f[8]  = ewma;
    f[9]  = dev;
    f[10] = fabsf(dev) / (fabsf(e->ewma_value) + 1e-8f);
    e->ewma_value = ewma;

    /* CUSUM against running historical mean */
    if (e->total_samples == 0)
        e->historical_mean = f[2];
    else
        e->historical_mean =
            (e->historical_mean * e->total_samples + f[2] * MB_WINDOW_SIZE)
            / (e->total_samples + MB_WINDOW_SIZE);
    e->total_samples += MB_WINDOW_SIZE;

    float cp = 0.f, cn = 0.f;
    for (int i = 0; i < MB_WINDOW_SIZE; i++) {
        float d = (float)e->readings[i] - e->historical_mean;
        cp = fmaxf(0.f, cp + d - CUSUM_ALLOWANCE);
        cn = fminf(0.f, cn + d + CUSUM_ALLOWANCE);
    }
    e->cusum_positive = cp;
    e->cusum_negative = cn;
    f[11] = cp;
    f[12] = fabsf(cn);
    f[13] = fmaxf(cp, fabsf(cn));
    f[14] = (cp > CUSUM_THRESHOLD || fabsf(cn) > CUSUM_THRESHOLD) ? 1.f : 0.f;

    int rising = 0, falling = 0;
    for (int i = 1; i < MB_WINDOW_SIZE; i++) {
        if (e->readings[i] > e->readings[i - 1]) rising++;
        else if (e->readings[i] < e->readings[i - 1]) falling++;
    }
    f[15] = fabsf((float)(rising - falling) / (MB_WINDOW_SIZE - 1));
}

/* --- sensor stream update --- */
static inline void
handle_reading(struct bench_worker *w, struct mb_state *s,
               const struct sensor_key *key, int32_t reading, uint64_t now)
{
    void *data = NULL;
    uint32_t idx;
    int ret = rte_hash_lookup_data(s->table, key, &data);

    if (ret < 0) {
        if (s->next_free >= MB_MAX_SENSORS) { w->insert_fail++; return; }
        idx = s->next_free++;
        struct sensor_entry *e = &s->pool[idx];
        memset(e, 0, sizeof(*e));
        e->ewma_value = (float)reading;
        e->historical_mean = (float)reading;
        if (rte_hash_add_key_data(s->table, key,
                                  (void *)(uintptr_t)idx) < 0) {
            s->next_free--; w->insert_fail++; return;
        }
        w->table_inserts++;
    } else {
        idx = (uint32_t)(uintptr_t)data;
    }

    struct sensor_entry *e = &s->pool[idx];
    e->readings[e->write_index]   = reading;
    e->timestamps[e->write_index] = now;
    e->sample_count++;
    e->write_index++;
    s->readings++;
    w->track_updates++;

    if (e->sample_count == MB_WINDOW_SIZE) {
        float f[MB_NUM_FEATURES];
        compute_features(e, f, (double)rte_get_tsc_hz());
        float acc = 0.f;
        for (int i = 0; i < MB_NUM_FEATURES; i++) acc += f[i];
        w->feature_sink += acc;
        w->windows_done++;
        e->sample_count = 0;
        e->write_index  = 0;
        /* CUSUM/EWMA state persists across windows (original behavior) */
    }
}

static void
mb_proc_burst(struct bench_worker *w, struct rte_mbuf **bufs,
              uint16_t nb, uint64_t now)
{
    struct mb_state *s = (struct mb_state *)w->app;

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
        if (sport != MB_PORT && dport != MB_PORT) continue;

        off += ((tcp->data_off >> 4) & 0x0F) * 4;
        if (plen <= off + sizeof(struct modbus_mbap)) continue;

        w->hit_pkts++;

        int sampled = bench_should_sample(w);
        uint64_t t0 = sampled ? rte_rdtsc_precise() : 0;

        struct modbus_mbap *mbap = (struct modbus_mbap *)(p + off);
        uint16_t tid = rte_be_to_cpu_16(mbap->trans_id);
        uint8_t  uid = mbap->unit_id;
        uint32_t pdu = off + sizeof(struct modbus_mbap);

        if (dport == MB_PORT && sport != MB_PORT) {
            /* request */
            if (plen >= pdu + sizeof(struct modbus_request_pdu)) {
                struct modbus_request_pdu *req =
                    (struct modbus_request_pdu *)(p + pdu);
                if (req->function_code == 0x03 ||
                    req->function_code == 0x04) {
                    add_pending(s, tid, uid,
                                rte_be_to_cpu_16(req->start_addr),
                                rte_be_to_cpu_16(req->quantity), now);
                    s->req_seen++;
                }
            }
        } else if (sport == MB_PORT && dport != MB_PORT) {
            /* response */
            if (plen >= pdu + sizeof(struct modbus_response_pdu)) {
                struct modbus_response_pdu *resp =
                    (struct modbus_response_pdu *)(p + pdu);
                if (resp->function_code == 0x03 ||
                    resp->function_code == 0x04) {
                    struct pending_request req;
                    if (find_pending(s, tid, uid, &req)) {
                        s->resp_matched++;
                        uint16_t nregs = resp->byte_count / 2;
                        for (uint16_t r = 0; r < nregs; r++) {
                            uint32_t doff = pdu + 2 + r * 2;
                            if (plen < doff + 2) break;
                            uint8_t *rd = p + doff;
                            uint16_t val = (uint16_t)((rd[0] << 8) | rd[1]);
                            int32_t sensor_val;
                            uint16_t addr = req.start_addr + r;

                            /* 32-bit pairing heuristic from the original */
                            if (nregs >= 2 && r < nregs - 1 &&
                                plen >= doff + 4) {
                                uint16_t nxt =
                                    (uint16_t)((rd[2] << 8) | rd[3]);
                                if (val == 0 && nxt != 0) {
                                    sensor_val = (int32_t)
                                        (((uint32_t)val << 16) | nxt);
                                    r++;
                                } else {
                                    sensor_val = (int16_t)val;
                                }
                            } else {
                                sensor_val = (int16_t)val;
                            }

                            struct sensor_key key = {
                                .unit_id = uid, .reg_addr = addr };
                            handle_reading(w, s, &key, sensor_val, now);
                        }
                    } else {
                        s->resp_unmatched++;
                    }
                }
            }
        }

        if (sampled) {
            uint64_t t1 = rte_rdtsc_precise();
            bench_store_sample(w, t1 - t0);
        }
    }
}

static int mb_app_init(struct bench_worker *w)
{
    struct mb_state *s = rte_zmalloc(NULL, sizeof(*s), 0);
    if (!s) return -1;

    char name[32];
    snprintf(name, sizeof(name), "mb_%u", w->core_id);
    struct rte_hash_parameters hp = {
        .name      = name,
        .entries   = MB_MAX_SENSORS,
        .key_len   = sizeof(struct sensor_key),
        .hash_func = rte_jhash,
        .socket_id = (int)rte_socket_id(),
    };
    s->table = rte_hash_create(&hp);
    if (!s->table) return -1;

    s->pool = rte_zmalloc_socket(NULL,
                sizeof(struct sensor_entry) * MB_MAX_SENSORS,
                RTE_CACHE_LINE_SIZE, rte_socket_id());
    if (!s->pool) return -1;

    w->app = s;
    return 0;
}

static void mb_report(struct bench_worker *w, FILE *out)
{
    struct mb_state *s = (struct mb_state *)w->app;
    fprintf(out, "        core %u: sensors %u | req %" PRIu64
                 " | matched %" PRIu64 " | unmatched %" PRIu64
                 " | readings %" PRIu64 "\n",
            w->core_id, rte_hash_count(s->table), s->req_seen,
            s->resp_matched, s->resp_unmatched, s->readings);
}

static const struct bench_ops mb_ops = {
    .name       = "modbus",
    .app_init   = mb_app_init,
    .proc_burst = mb_proc_burst,
    .app_report = mb_report,
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, &mb_ops);
}
