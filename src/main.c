#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <inttypes.h>

#include <rte_eal.h>
#include <rte_lcore.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_cycles.h>
#include <rte_version.h>

#include "app_config.h"
#include "dpdk_port.h"
#include "worker.h"
#include "profiles.h"
#include "inference.h"
#include "logger.h"

static atomic_int g_stop = 0;

static void signal_handler(int signo)
{
    (void)signo;
    atomic_store_explicit(&g_stop, 1, memory_order_relaxed);
}

static void print_worker_summary(struct worker_ctx *workers, uint16_t n,
                                 const struct model_runtime *model)
{
    uint64_t total_rx = 0, total_tx = 0, total_drop = 0, total_events = 0, invalid = 0;
    uint64_t pred[32] = {0};
    for (uint16_t i = 0; i < n; i++) {
        total_rx += workers[i].rx_packets;
        total_tx += workers[i].tx_packets;
        total_drop += workers[i].dropped_packets;
        total_events += workers[i].feature_events;
        invalid += workers[i].invalid_features;
        for (uint16_t c = 0; c < model->num_classes && c < 32; c++) pred[c] += workers[i].predictions[c];
#if APP_ENABLE_TIMING
        if (workers[i].infer_cycles_count) {
            double hz = (double)rte_get_tsc_hz();
            double avg_ns = ((double)workers[i].infer_cycles_sum / (double)workers[i].infer_cycles_count) * 1e9 / hz;
            double min_ns = (double)workers[i].infer_cycles_min * 1e9 / hz;
            double max_ns = (double)workers[i].infer_cycles_max * 1e9 / hz;
            printf("worker %u inference avg=%.2f ns min=%.2f ns max=%.2f ns samples=%" PRIu64 "\n",
                   i, avg_ns, min_ns, max_ns, workers[i].infer_cycles_count);
        }
#endif
    }
    printf("\n=== Summary ===\n");
    printf("rx=%" PRIu64 " tx=%" PRIu64 " dropped=%" PRIu64 " feature_events=%" PRIu64 " invalid=%" PRIu64 "\n",
           total_rx, total_tx, total_drop, total_events, invalid);
    for (uint16_t c = 0; c < model->num_classes && c < 32; c++) {
        printf("class_%u=%" PRIu64 "\n", c, pred[c]);
    }
}

int main(int argc, char **argv)
{
    struct app_config cfg;
    app_config_set_defaults(&cfg);

    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_panic("Cannot initialize EAL\n");
    argc -= ret;
    argv += ret;

    int pr = app_config_parse(argc, argv, &cfg);
    if (pr > 0) return 0;
    if (pr < 0) rte_exit(EXIT_FAILURE, "Invalid application arguments\n");

#if !APP_ENABLE_LOGS
    cfg.log_mode = APP_LOG_NONE;
#endif

    printf("DPDK version: %s\n", rte_version());
    printf("TSC Hz: %" PRIu64 "\n", rte_get_tsc_hz());
    app_config_print(&cfg);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    const struct profile_ops *profile = profile_get(cfg.profile);
    if (!profile) rte_exit(EXIT_FAILURE, "Unknown profile\n");

    struct model_runtime model;
    if (model_runtime_init(&model, cfg.model, cfg.kernel) != 0)
        rte_exit(EXIT_FAILURE, "Cannot initialize model runtime\n");

    if (profile->feature_count != model.input_dim && !cfg.allow_feature_pad) {
        rte_exit(EXIT_FAILURE,
                 "Feature/model mismatch: profile %s emits %u features, model expects %u. Use matching exported header or --allow-feature-pad 1.\n",
                 profile->name, profile->feature_count, model.input_dim);
    }

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0 || cfg.port_id >= nb_ports)
        rte_exit(EXIT_FAILURE, "No usable DPDK port %u\n", cfg.port_id);

    unsigned main_lcore = rte_get_main_lcore();
    unsigned log_lcore = RTE_MAX_LCORE;
    unsigned worker_lcores[APP_MAX_WORKERS];
    uint16_t worker_count = 0;

    unsigned lcore_iter;
    RTE_LCORE_FOREACH_WORKER(lcore_iter) {
        unsigned lc = lcore_iter;
        if (cfg.log_mode == APP_LOG_ASYNC && log_lcore == RTE_MAX_LCORE) {
            log_lcore = lc;
            continue;
        }
        if (worker_count < APP_MAX_WORKERS) worker_lcores[worker_count++] = lc;
    }
    if (cfg.use_main_as_worker && worker_count < APP_MAX_WORKERS) {
        worker_lcores[worker_count++] = main_lcore;
    }
    if (cfg.worker_count > 0 && cfg.worker_count < worker_count) worker_count = cfg.worker_count;
    if (worker_count == 0) rte_exit(EXIT_FAILURE, "No worker lcores available\n");

    struct logger logger;
    if (logger_init(&logger, &cfg, &g_stop, rte_socket_id()) != 0)
        rte_exit(EXIT_FAILURE, "Cannot initialize logger\n");
    if (cfg.log_mode == APP_LOG_ASYNC && log_lcore != RTE_MAX_LCORE) {
        rte_eal_remote_launch(logger_loop, &logger, log_lcore);
    } else if (cfg.log_mode == APP_LOG_ASYNC) {
        printf("No spare lcore for logger; disabling async logs.\n");
        cfg.log_mode = APP_LOG_NONE;
    }

    struct rte_mempool *pools[APP_MAX_WORKERS];
    memset(pools, 0, sizeof(pools));
    for (uint16_t i = 0; i < worker_count; i++) {
        char name[64];
        snprintf(name, sizeof(name), "mbuf_pool_%u", i);
        int socket = rte_lcore_to_socket_id(worker_lcores[i]);
        pools[i] = rte_pktmbuf_pool_create(name,
                                           cfg.num_mbufs,
                                           cfg.mbuf_cache_size,
                                           0,
                                           RTE_MBUF_DEFAULT_BUF_SIZE,
                                           socket);
        if (!pools[i]) rte_exit(EXIT_FAILURE, "Cannot create mbuf pool %s\n", name);
    }

    if (app_port_init(cfg.port_id, pools, worker_count, &cfg) != 0)
        rte_exit(EXIT_FAILURE, "Cannot initialize port %u\n", cfg.port_id);

    struct worker_ctx *workers = rte_zmalloc("workers",
                                             (size_t)worker_count * sizeof(*workers),
                                             RTE_CACHE_LINE_SIZE);
    if (!workers) rte_exit(EXIT_FAILURE, "Cannot allocate worker contexts\n");

    for (uint16_t i = 0; i < worker_count; i++) {
        struct worker_ctx *w = &workers[i];
        w->cfg = &cfg;
        w->profile = profile;
        w->model = &model;
        w->logger = &logger;
        w->stop_flag = &g_stop;
        w->mbuf_pool = pools[i];
        w->port_id = cfg.port_id;
        w->queue_id = i;
        w->lcore_id = worker_lcores[i];
        w->worker_id = i;
        if (model.scratch_bytes > 0) {
            if (posix_memalign(&w->model_scratch, 64, model.scratch_bytes) != 0)
                rte_exit(EXIT_FAILURE, "Cannot allocate model scratch\n");
            memset(w->model_scratch, 0, model.scratch_bytes);
        }
        if (profile->init_worker(w) != 0)
            rte_exit(EXIT_FAILURE, "Cannot initialize profile state for worker %u\n", i);
    }

    for (uint16_t i = 0; i < worker_count; i++) {
        if (workers[i].lcore_id == main_lcore) continue;
        rte_eal_remote_launch(worker_loop, &workers[i], workers[i].lcore_id);
    }

    bool main_is_worker = false;
    for (uint16_t i = 0; i < worker_count; i++) {
        if (workers[i].lcore_id == main_lcore) {
            main_is_worker = true;
            worker_loop(&workers[i]);
            break;
        }
    }

    if (!main_is_worker) {
        while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) sleep(1);
    }

    rte_eal_mp_wait_lcore();
    print_worker_summary(workers, worker_count, &model);

    for (uint16_t i = 0; i < worker_count; i++) {
        if (workers[i].profile && workers[i].profile->free_worker)
            workers[i].profile->free_worker(&workers[i]);
        free(workers[i].model_scratch);
    }
    rte_free(workers);

    app_ports_close();
    for (uint16_t i = 0; i < worker_count; i++) {
        if (pools[i]) rte_mempool_free(pools[i]);
    }
    logger_free(&logger);
    rte_eal_cleanup();
    return 0;
}
