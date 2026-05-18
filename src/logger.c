#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include <rte_cycles.h>
#include "logger.h"

#if APP_ENABLE_LOGS

int logger_init(struct logger *lg, const struct app_config *cfg,
                atomic_int *stop_flag, int socket_id)
{
    memset(lg, 0, sizeof(*lg));
    lg->mode = cfg->log_mode;
    lg->stop_flag = stop_flag;
    snprintf(lg->file_name, sizeof(lg->file_name), "%s", cfg->log_file);
    if (cfg->log_mode == APP_LOG_NONE) return 0;

    lg->ring = rte_ring_create("log_ring", cfg->log_ring_size, socket_id,
                               RING_F_SC_DEQ);
    if (!lg->ring) return -1;

    lg->pool = rte_mempool_create("log_pool", cfg->log_ring_size,
                                  sizeof(struct log_event), 0, 0,
                                  NULL, NULL, NULL, NULL,
                                  socket_id, 0);
    if (!lg->pool) return -1;
    return 0;
}

void logger_free(struct logger *lg)
{
    if (!lg) return;
    if (lg->ring) rte_ring_free(lg->ring);
    if (lg->pool) rte_mempool_free(lg->pool);
    memset(lg, 0, sizeof(*lg));
}

int logger_submit_prediction(struct logger *lg, const struct log_event *ev)
{
    if (!lg || lg->mode == APP_LOG_NONE || !lg->ring || !lg->pool) return 0;
    struct log_event *slot = NULL;
    if (rte_mempool_get(lg->pool, (void **)&slot) != 0) {
        lg->dropped++;
        return -1;
    }
    *slot = *ev;
    if (rte_ring_enqueue(lg->ring, slot) != 0) {
        rte_mempool_put(lg->pool, slot);
        lg->dropped++;
        return -1;
    }
    lg->submitted++;
    return 0;
}

static void write_header(FILE *f)
{
    fprintf(f, "tsc,worker,queue,profile,model,pred,action,id_type,src_ip,src_port,dst_ip,dst_port,unit_id,reg_addr,n_features");
    for (int i = 0; i < APP_MAX_FEATURES; i++) fprintf(f, ",f%d", i);
    fprintf(f, "\n");
}

static void write_event(FILE *f, const struct log_event *ev)
{
    uint32_t src_ip = 0, dst_ip = 0;
    uint16_t src_port = 0, dst_port = 0;
    uint8_t unit_id = 0;
    uint16_t reg_addr = 0;
    if (ev->id.type == FEATURE_ID_FLOW5) {
        src_ip = ev->id.u.flow.src_ip;
        dst_ip = ev->id.u.flow.dst_ip;
        src_port = ev->id.u.flow.src_port;
        dst_port = ev->id.u.flow.dst_port;
    } else if (ev->id.type == FEATURE_ID_SENSOR) {
        unit_id = ev->id.u.sensor.unit_id;
        reg_addr = ev->id.u.sensor.reg_addr;
    }
    fprintf(f, "%" PRIu64 ",%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
            ev->tsc, ev->worker_id, ev->queue_id, ev->profile,
            ev->model, ev->pred, ev->action, ev->id.type,
            src_ip, src_port, dst_ip, dst_port, unit_id, reg_addr, ev->fv.n);
    for (int i = 0; i < APP_MAX_FEATURES; i++) {
        float v = (i < ev->fv.n) ? ev->fv.v[i] : 0.0f;
        fprintf(f, ",%.6f", v);
    }
    fprintf(f, "\n");
}

int logger_loop(void *arg)
{
    struct logger *lg = (struct logger *)arg;
    if (!lg || lg->mode == APP_LOG_NONE) return 0;
    FILE *f = fopen(lg->file_name, "w");
    if (!f) return -1;
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    write_header(f);

    printf("Logger on lcore %u writing %s\n", rte_lcore_id(), lg->file_name);
    struct log_event *events[64];
    while (!atomic_load_explicit(lg->stop_flag, memory_order_relaxed) ||
           rte_ring_count(lg->ring) > 0) {
        unsigned n = rte_ring_dequeue_burst(lg->ring, (void **)events, 64, NULL);
        if (n == 0) {
            rte_pause();
            continue;
        }
        for (unsigned i = 0; i < n; i++) {
            write_event(f, events[i]);
            rte_mempool_put(lg->pool, events[i]);
        }
    }
    fflush(f);
    fclose(f);
    printf("Logger done: submitted=%" PRIu64 " dropped=%" PRIu64 "\n",
           lg->submitted, lg->dropped);
    return 0;
}

#endif
