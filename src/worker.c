#include <stdio.h>
#include <string.h>
#include <rte_ethdev.h>
#include <rte_lcore.h>
#include <rte_prefetch.h>
#include <rte_malloc.h>
#include "worker.h"
#include "feature_contract.h"

static void update_timing(struct worker_ctx *w, uint64_t cycles)
{
#if APP_ENABLE_TIMING
    w->infer_cycles_sum += cycles;
    w->infer_cycles_count++;
    if (w->infer_cycles_count == 1 || cycles < w->infer_cycles_min) w->infer_cycles_min = cycles;
    if (cycles > w->infer_cycles_max) w->infer_cycles_max = cycles;
#else
    (void)w; (void)cycles;
#endif
}

static void submit_log(struct worker_ctx *w,
                       const struct feature_vector *fv,
                       const struct feature_id *id,
                       int pred,
                       uint8_t action)
{
#if APP_ENABLE_LOGS
    if (!w->logger || w->cfg->log_mode == APP_LOG_NONE) return;
    struct log_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.tsc = rte_rdtsc();
    ev.worker_id = w->worker_id;
    ev.queue_id = w->queue_id;
    ev.profile = (uint8_t)w->cfg->profile;
    ev.model = (uint8_t)w->cfg->model;
    ev.pred = (uint8_t)pred;
    ev.action = action;
    ev.fv = *fv;
    ev.id = *id;
    logger_submit_prediction(w->logger, &ev);
#else
    (void)w; (void)fv; (void)id; (void)pred; (void)action;
#endif
}

static uint8_t classify_feature(struct worker_ctx *w,
                                const struct feature_vector *fv,
                                const struct feature_id *id)
{
    uint16_t contract_window = (w->cfg->profile == APP_PROFILE_ICS)
                             ? w->cfg->ics_window_samples
                             : w->cfg->window_packets;
    if (!feature_contract_validate(w->cfg->profile, fv, contract_window)) {
        w->invalid_features++;
        return 0;
    }
    if (model_prepare_input(w->model, fv, w->scaled_features,
                            w->cfg->allow_feature_pad) != 0) {
        w->invalid_features++;
        return 0;
    }

    uint64_t t0 = rte_rdtsc_precise();
    int pred = model_predict(w->model, w->scaled_features, w->model_scratch);
    uint64_t t1 = rte_rdtsc_precise();
    update_timing(w, t1 - t0);

    if (pred >= 0 && pred < (int)(sizeof(w->predictions) / sizeof(w->predictions[0]))) {
        w->predictions[pred]++;
    }
    w->feature_events++;

    uint8_t drop = (w->cfg->action_mode == APP_ACTION_DROP_ANOMALY && pred != 0) ? 1 : 0;
    submit_log(w, fv, id, pred, drop);
    return drop;
}

static uint8_t drain_features(struct worker_ctx *w)
{
    uint8_t should_drop = 0;
    struct feature_vector fv;
    struct feature_id id;
    while (w->profile->next_feature(w, &fv, &id) == 1) {
        should_drop |= classify_feature(w, &fv, &id);
    }
    return should_drop;
}

int worker_loop(void *arg)
{
    struct worker_ctx *w = (struct worker_ctx *)arg;
    struct rte_mbuf *bufs[APP_DEFAULT_BURST];
    struct rte_mbuf *tx_bufs[APP_DEFAULT_BURST];
    const uint16_t burst = w->cfg->burst_size;

    printf("Worker %u on lcore %u port %u queue %u profile=%s model=%s\n",
           w->worker_id, rte_lcore_id(), w->port_id, w->queue_id,
           w->profile->name, app_model_name(w->cfg->model));

    while (!atomic_load_explicit(w->stop_flag, memory_order_relaxed)) {
        uint16_t nb_rx = rte_eth_rx_burst(w->port_id, w->queue_id, bufs, burst);
        if (unlikely(nb_rx == 0)) {
            rte_pause();
            continue;
        }
        w->rx_packets += nb_rx;

        uint16_t tx_count = 0;
        for (uint16_t i = 0; i < nb_rx; i++) {
            if (i + 2 < nb_rx) rte_prefetch0(rte_pktmbuf_mtod(bufs[i + 2], void *));
            uint64_t now = rte_rdtsc_precise();
            w->profile->process_packet(w, bufs[i], now);
            uint8_t drop = drain_features(w);

            if (drop) {
                rte_pktmbuf_free(bufs[i]);
                w->dropped_packets++;
                continue;
            }
            if (w->cfg->swap_mac) packet_swap_l2(bufs[i]);
            tx_bufs[tx_count++] = bufs[i];
        }

        if (tx_count > 0) {
            uint16_t nb_tx = rte_eth_tx_burst(w->port_id, w->queue_id,
                                              tx_bufs, tx_count);
            w->tx_packets += nb_tx;
            if (unlikely(nb_tx < tx_count)) {
                for (uint16_t i = nb_tx; i < tx_count; i++) rte_pktmbuf_free(tx_bufs[i]);
                w->dropped_packets += (uint64_t)(tx_count - nb_tx);
            }
        }
    }
    return 0;
}
