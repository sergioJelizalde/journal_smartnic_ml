#include <stdio.h>
#include <string.h>
#include <math.h>
#include <rte_byteorder.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include "worker.h"
#include "profiles.h"
#include "packet_parser.h"
#include "flow_pool.h"
#include "modbus_tracker.h"

#define MODBUS_TCP_PORT 502
#define MODBUS_FUNC_READ_HOLDING 0x03
#define MODBUS_FUNC_READ_INPUT 0x04
#define ICS_MAX_WINDOW 256
#define ICS_EWMA_ALPHA 0.30f
#define ICS_CUSUM_ALLOWANCE 0.50f
#define ICS_CUSUM_THRESHOLD 2.00f

struct modbus_mbap {
    uint16_t trans_id;
    uint16_t protocol_id;
    uint16_t length;
    uint8_t unit_id;
} __attribute__((packed));

struct modbus_request_pdu {
    uint8_t function_code;
    uint16_t start_addr;
    uint16_t quantity;
} __attribute__((packed));

struct modbus_response_pdu {
    uint8_t function_code;
    uint8_t byte_count;
} __attribute__((packed));

struct sensor_entry {
    uint64_t timestamps[ICS_MAX_WINDOW];
    int32_t readings[ICS_MAX_WINDOW];
    uint16_t sample_count;
    uint16_t write_index;
    float ewma_value;
    float cusum_positive;
    float cusum_negative;
    float historical_mean;
    uint32_t total_samples;
};

struct ics_state {
    struct flow_pool sensors;
    struct modbus_tracker tracker;
    struct feature_queue q;
    uint16_t window;
    double tsc_hz;
};

static double compute_slope_seconds(const uint64_t *ts,
                                    const int32_t *y,
                                    uint16_t n,
                                    double hz)
{
    double sumt = 0.0, sumy = 0.0, sumty = 0.0, sumt2 = 0.0;
    for (uint16_t i = 0; i < n; i++) {
        double t = (double)(ts[i] - ts[0]) / hz;
        double yi = (double)y[i];
        sumt += t;
        sumy += yi;
        sumty += t * yi;
        sumt2 += t * t;
    }
    double denom = (double)n * sumt2 - sumt * sumt;
    if (denom == 0.0) return 0.0;
    return ((double)n * sumty - sumt * sumy) / denom;
}

static void compute_sensor_features(struct ics_state *s,
                                    struct sensor_entry *e,
                                    struct feature_vector *fv)
{
    uint16_t n = s->window;
    int32_t minv = INT32_MAX;
    int32_t maxv = INT32_MIN;
    double sum = 0.0;
    for (uint16_t i = 0; i < n; i++) {
        int32_t v = e->readings[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += (double)v;
    }
    double mean = sum / (double)n;

    uint64_t iat_min = UINT64_MAX;
    uint64_t iat_max = 0;
    double iat_sum = 0.0;
    for (uint16_t i = 1; i < n; i++) {
        uint64_t dt = e->timestamps[i] >= e->timestamps[i - 1]
                    ? e->timestamps[i] - e->timestamps[i - 1]
                    : 0;
        if (dt < iat_min) iat_min = dt;
        if (dt > iat_max) iat_max = dt;
        iat_sum += (double)dt;
    }
    double mean_iat_us = (iat_sum / (double)(n - 1)) * 1e6 / s->tsc_hz;
    double iat_min_us = (double)iat_min * 1e6 / s->tsc_hz;
    double iat_max_us = (double)iat_max * 1e6 / s->tsc_hz;
    double slope = compute_slope_seconds(e->timestamps, e->readings, n, s->tsc_hz);

    float current_ewma = (float)e->readings[0];
    for (uint16_t i = 1; i < n; i++) {
        current_ewma = ICS_EWMA_ALPHA * (float)e->readings[i] +
                       (1.0f - ICS_EWMA_ALPHA) * current_ewma;
    }
    float ewma_deviation = current_ewma - e->ewma_value;
    e->ewma_value = current_ewma;

    if (e->total_samples == 0) {
        e->historical_mean = (float)mean;
    } else {
        e->historical_mean = (e->historical_mean * (float)e->total_samples +
                              (float)mean * (float)n) /
                             (float)(e->total_samples + n);
    }
    e->total_samples += n;

    float cusum_pos = e->cusum_positive;
    float cusum_neg = e->cusum_negative;
    for (uint16_t i = 0; i < n; i++) {
        float d = (float)e->readings[i] - e->historical_mean;
        cusum_pos = fmaxf(0.0f, cusum_pos + d - ICS_CUSUM_ALLOWANCE);
        cusum_neg = fminf(0.0f, cusum_neg + d + ICS_CUSUM_ALLOWANCE);
    }
    e->cusum_positive = cusum_pos;
    e->cusum_negative = cusum_neg;

    int rising = 0, falling = 0;
    for (uint16_t i = 1; i < n; i++) {
        if (e->readings[i] > e->readings[i - 1]) rising++;
        else if (e->readings[i] < e->readings[i - 1]) falling++;
    }
    float trend_consistency = n > 1 ? fabsf((float)(rising - falling) / (float)(n - 1)) : 0.0f;

    memset(fv, 0, sizeof(*fv));
    fv->n = 16;
    fv->v[0] = (float)minv;
    fv->v[1] = (float)maxv;
    fv->v[2] = (float)mean;
    fv->v[3] = (float)iat_min_us;
    fv->v[4] = (float)iat_max_us;
    fv->v[5] = (float)mean_iat_us;
    fv->v[6] = (float)(maxv - minv);
    fv->v[7] = (float)slope;
    fv->v[8] = current_ewma;
    fv->v[9] = ewma_deviation;
    fv->v[10] = fabsf(ewma_deviation) / (fabsf(e->ewma_value) + 1e-8f);
    fv->v[11] = cusum_pos;
    fv->v[12] = fabsf(cusum_neg);
    fv->v[13] = fmaxf(cusum_pos, fabsf(cusum_neg));
    fv->v[14] = (cusum_pos > ICS_CUSUM_THRESHOLD || fabsf(cusum_neg) > ICS_CUSUM_THRESHOLD) ? 1.0f : 0.0f;
    fv->v[15] = trend_consistency;
}

static int ics_init_worker(struct worker_ctx *w)
{
    struct ics_state *s = rte_zmalloc_socket("ics_state", sizeof(*s),
                                             RTE_CACHE_LINE_SIZE,
                                             rte_lcore_to_socket_id(w->lcore_id));
    if (!s) return -1;
    char name[64];
    snprintf(name, sizeof(name), "ics_sensors_%u", w->worker_id);
    if (flow_pool_init(&s->sensors, name, sizeof(struct sensor_key),
                       sizeof(struct sensor_entry),
                       w->cfg->max_sensors_per_worker,
                       rte_lcore_to_socket_id(w->lcore_id)) != 0) {
        rte_free(s);
        return -1;
    }
    snprintf(name, sizeof(name), "ics_req_%u", w->worker_id);
    if (modbus_tracker_init(&s->tracker, name, 8192,
                            rte_lcore_to_socket_id(w->lcore_id)) != 0) {
        flow_pool_free(&s->sensors);
        rte_free(s);
        return -1;
    }
    feature_queue_init(&s->q);
    s->window = w->cfg->ics_window_samples ? w->cfg->ics_window_samples : 32;
    if (s->window > ICS_MAX_WINDOW) s->window = ICS_MAX_WINDOW;
    if (s->window < 2) s->window = 2;
    s->tsc_hz = (double)rte_get_tsc_hz();
    w->profile_state = s;
    return 0;
}

static void ics_free_worker(struct worker_ctx *w)
{
    struct ics_state *s = (struct ics_state *)w->profile_state;
    if (!s) return;
    modbus_tracker_free(&s->tracker);
    flow_pool_free(&s->sensors);
    rte_free(s);
    w->profile_state = NULL;
}

static void sensor_add_reading(struct ics_state *s,
                               const struct sensor_key *key,
                               int32_t reading,
                               uint64_t now_tsc)
{
    bool is_new = false;
    struct sensor_entry *e = flow_pool_lookup_or_alloc(&s->sensors, key, NULL, &is_new);
    if (!e) return;
    if (is_new) {
        memset(e, 0, sizeof(*e));
        e->ewma_value = (float)reading;
        e->historical_mean = (float)reading;
    }

    e->readings[e->write_index] = reading;
    e->timestamps[e->write_index] = now_tsc;
    e->write_index++;
    e->sample_count++;

    if (e->sample_count >= s->window) {
        struct feature_vector fv;
        struct feature_id id;
        compute_sensor_features(s, e, &fv);
        memset(&id, 0, sizeof(id));
        id.type = FEATURE_ID_SENSOR;
        id.u.sensor = *key;
        feature_queue_push(&s->q, &fv, &id);
        e->sample_count = 0;
        e->write_index = 0;
    }
}

static int ics_process_packet(struct worker_ctx *w, struct rte_mbuf *m,
                              uint64_t now_tsc)
{
    struct ics_state *s = (struct ics_state *)w->profile_state;
    struct packet_view p;
    if (!packet_parse_ipv4_tcp(m, &p)) return 0;
    if (p.src_port != MODBUS_TCP_PORT && p.dst_port != MODBUS_TCP_PORT) return 0;
    if (p.payload_len < sizeof(struct modbus_mbap)) return 0;

    struct modbus_mbap *mbap = (struct modbus_mbap *)p.payload;
    uint16_t protocol_id = rte_be_to_cpu_16(mbap->protocol_id);
    if (protocol_id != 0) return 0;
    uint16_t trans_id = rte_be_to_cpu_16(mbap->trans_id);
    uint8_t unit_id = mbap->unit_id;

    uint8_t *pdu = p.payload + sizeof(struct modbus_mbap);
    uint32_t pdu_len = p.payload_len - sizeof(struct modbus_mbap);
    if (pdu_len < 1) return 0;

    if (p.src_port != MODBUS_TCP_PORT && p.dst_port == MODBUS_TCP_PORT) {
        if (pdu_len < sizeof(struct modbus_request_pdu)) return 0;
        struct modbus_request_pdu *req = (struct modbus_request_pdu *)pdu;
        if (req->function_code == MODBUS_FUNC_READ_HOLDING ||
            req->function_code == MODBUS_FUNC_READ_INPUT) {
            uint16_t start_addr = rte_be_to_cpu_16(req->start_addr);
            uint16_t quantity = rte_be_to_cpu_16(req->quantity);
            modbus_tracker_add(&s->tracker, trans_id, unit_id, start_addr, quantity, now_tsc);
        }
        return 0;
    }

    if (p.src_port == MODBUS_TCP_PORT && p.dst_port != MODBUS_TCP_PORT) {
        if (pdu_len < sizeof(struct modbus_response_pdu)) return 0;
        struct modbus_response_pdu *resp = (struct modbus_response_pdu *)pdu;
        if (resp->function_code != MODBUS_FUNC_READ_HOLDING &&
            resp->function_code != MODBUS_FUNC_READ_INPUT) return 0;

        struct modbus_pending_request req;
        if (!modbus_tracker_take(&s->tracker, trans_id, unit_id, &req)) return 0;

        uint8_t byte_count = resp->byte_count;
        if ((uint32_t)byte_count + 2u > pdu_len) return 0;
        uint16_t regs = (uint16_t)(byte_count / 2u);
        uint8_t *data = pdu + 2;

        for (uint16_t i = 0; i < regs; i++) {
            if ((uint32_t)(i * 2 + 1) >= byte_count) break;
            uint16_t regval = ((uint16_t)data[i * 2] << 8) | data[i * 2 + 1];
            int32_t sensor_value;
            uint16_t reg_addr = req.start_addr + i;

            /* Optional 32-bit pair heuristic preserved from your prototype. */
            if (i + 1 < regs && regval == 0) {
                uint16_t next = ((uint16_t)data[(i + 1) * 2] << 8) | data[(i + 1) * 2 + 1];
                if (next != 0) {
                    sensor_value = (int32_t)(((uint32_t)regval << 16) | next);
                    i++;
                } else {
                    sensor_value = (int16_t)regval;
                }
            } else {
                sensor_value = (int16_t)regval;
            }

            struct sensor_key key = {.unit_id = unit_id, .reg_addr = reg_addr};
            sensor_add_reading(s, &key, sensor_value, now_tsc);
        }
    }
    return 0;
}

static int ics_next_feature(struct worker_ctx *w,
                            struct feature_vector *fv,
                            struct feature_id *id)
{
    struct ics_state *s = (struct ics_state *)w->profile_state;
    return feature_queue_pop(&s->q, fv, id);
}

const struct profile_ops ics_profile_ops = {
    .name = "ics_modbus_sensor",
    .type = APP_PROFILE_ICS,
    .feature_count = 16,
    .init_worker = ics_init_worker,
    .free_worker = ics_free_worker,
    .process_packet = ics_process_packet,
    .next_feature = ics_next_feature,
};
