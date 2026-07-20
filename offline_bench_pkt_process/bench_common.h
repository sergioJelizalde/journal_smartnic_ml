/* SPDX-License-Identifier: BSD-3-Clause
 *
 * bench_common.h — shared infrastructure for DPDK stateful-tracking benchmarks
 *
 * Provides everything that is identical across the three workloads:
 *   - EAL / port / queue init with symmetric RSS (both directions of a flow
 *     land on the same RX queue -> per-core lock-free state)
 *   - per-core worker context, counters, and cycle-sample reservoirs
 *   - a generic run loop (RX burst -> app callback -> optional MAC-swap fwd)
 *   - sampled per-event timing helpers (rdtsc every Kth event only, so the
 *     instrumentation itself does not distort throughput)
 *   - percentile computation + CSV/console reporting at shutdown
 *
 * The application (one .c per workload) supplies a struct bench_ops with:
 *   app_init(w)     — create per-core hash tables / pools
 *   proc_burst(...) — parse + update tracking state; NO mbuf free / tx here
 *   app_report(w)   — optional extra per-core numbers
 *
 * Inference is intentionally absent: feature vectors, when a window
 * completes, are folded into w->feature_sink (volatile) so the compiler
 * cannot elide feature construction, which is part of the tracking cost.
 */

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <signal.h>
#include <getopt.h>
#include <stdatomic.h>

#include <rte_eal.h>
#include <rte_common.h>
#include <rte_ethdev.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>
#include <rte_malloc.h>
#include <rte_hash.h>
#include <rte_jhash.h>
#include <rte_version.h>

/* ------------------------------------------------------------------ */
/* Tunables (override with -D at compile time)                         */
/* ------------------------------------------------------------------ */
#ifndef BENCH_BURST_SIZE
#define BENCH_BURST_SIZE   32
#endif
#ifndef BENCH_RX_RING
#define BENCH_RX_RING      1024
#endif
#ifndef BENCH_TX_RING
#define BENCH_TX_RING      1024
#endif
#ifndef BENCH_NUM_MBUFS
#define BENCH_NUM_MBUFS    (8191 * 2)
#endif
#ifndef BENCH_MBUF_CACHE
#define BENCH_MBUF_CACHE   512
#endif
#ifndef BENCH_MAX_SAMPLES          /* per-core reservoir for event cycles  */
#define BENCH_MAX_SAMPLES  (1u << 20)
#endif
#ifndef BENCH_MAX_BURSTS           /* per-core reservoir for burst cycles  */
#define BENCH_MAX_BURSTS   (1u << 20)
#endif

#define BENCH_MAX_CORES    RTE_MAX_LCORE
#define BENCH_INVALID_IDX  UINT32_MAX

/* Symmetric Toeplitz key (0x6D5A repeating): src/dst-swapped packets hash
 * to the same queue.  Required for bidirectional (TLS) and request/response
 * (Modbus) tracking with per-core lock-free tables. */
#define BENCH_RSS_KEY_LEN 40
static uint8_t bench_rss_key[BENCH_RSS_KEY_LEN] = {
    0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,0x6D,0x5A, 0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,
    0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,0x6D,0x5A, 0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,
    0x6D,0x5A,0x6D,0x5A,0x6D,0x5A,0x6D,0x5A
};

/* ------------------------------------------------------------------ */
/* Runtime options (parsed after the EAL '--')                         */
/* ------------------------------------------------------------------ */
struct bench_opts {
    uint32_t sample_every;   /* time 1 out of every N tracked events (0=off) */
    uint32_t duration_sec;   /* auto-stop after N seconds (0 = run forever)  */
    int      forward;        /* 1 = MAC-swap + TX (fwd), 0 = rx+drop         */
    const char *out_prefix;  /* prefix for CSV output files                  */
};

/* ------------------------------------------------------------------ */
/* Per-core worker context                                             */
/* ------------------------------------------------------------------ */
struct bench_worker {
    uint16_t port_id;
    uint16_t queue_id;
    unsigned core_id;

    /* throughput counters */
    uint64_t rx_pkts;
    uint64_t rx_bytes;
    uint64_t tx_pkts;
    uint64_t hit_pkts;        /* packets matching the workload protocol   */
    uint64_t track_updates;   /* state-table updates performed            */
    uint64_t table_inserts;   /* new keys inserted                        */
    uint64_t insert_fail;     /* pool/table exhaustion                    */
    uint64_t windows_done;    /* feature windows completed                */

    /* sampled event timing (parse + lookup + update [+ feature build]) */
    uint64_t *ev_cycles;
    uint32_t  ev_count;
    uint32_t  ev_stride_ctr;  /* countdown for sampling stride            */

    /* whole-burst timing (always on: 2 rdtsc per burst)                */
    uint64_t *burst_cycles;
    uint16_t *burst_pkts;
    uint32_t  burst_count;

    uint64_t t_start, t_stop; /* wall tsc of the run loop                 */

    /* keeps feature construction alive without inference */
    volatile float feature_sink;

    void *app;                /* app-specific per-core state              */
};

struct bench_ops {
    const char *name;
    int  (*app_init)(struct bench_worker *w);
    /* parse + track; must NOT free or transmit mbufs */
    void (*proc_burst)(struct bench_worker *w, struct rte_mbuf **bufs,
                       uint16_t nb, uint64_t now_tsc);
    void (*app_report)(struct bench_worker *w, FILE *out); /* nullable */
};

/* globals */
static struct bench_worker bench_workers[BENCH_MAX_CORES];
static struct bench_opts   bench_opts = {
    .sample_every = 64, .duration_sec = 0, .forward = 1, .out_prefix = "bench"
};
static atomic_int bench_force_quit = 0;
static unsigned   bench_n_lcores   = 0;

/* ------------------------------------------------------------------ */
/* Sampled timing helpers                                              */
/* ------------------------------------------------------------------ */
/* Returns nonzero if this event should be timed. Call once per event. */
static inline int bench_should_sample(struct bench_worker *w)
{
    if (bench_opts.sample_every == 0 || w->ev_count >= BENCH_MAX_SAMPLES)
        return 0;
    if (++w->ev_stride_ctr >= bench_opts.sample_every) {
        w->ev_stride_ctr = 0;
        return 1;
    }
    return 0;
}

static inline void bench_store_sample(struct bench_worker *w, uint64_t cycles)
{
    if (w->ev_count < BENCH_MAX_SAMPLES)
        w->ev_cycles[w->ev_count++] = cycles;
}

/* ------------------------------------------------------------------ */
/* Port init                                                           */
/* ------------------------------------------------------------------ */
static inline int
bench_port_init(uint16_t port, struct rte_mempool *pool, uint16_t n_queues)
{
    struct rte_eth_dev_info dev_info;
    struct rte_eth_rxconf rxconf;
    struct rte_eth_txconf txconf;
    int ret;
    uint16_t q;

    ret = rte_eth_dev_info_get(port, &dev_info);
    if (ret != 0) {
        printf("port %u: dev_info_get failed: %s\n", port, strerror(-ret));
        return ret;
    }

    if (n_queues > dev_info.max_rx_queues) n_queues = dev_info.max_rx_queues;
    if (n_queues > dev_info.max_tx_queues) n_queues = dev_info.max_tx_queues;

    struct rte_eth_conf conf = {
        .rxmode = { .mq_mode = RTE_ETH_MQ_RX_RSS },
        .rx_adv_conf = {
            .rss_conf = {
                .rss_key     = bench_rss_key,
                .rss_key_len = BENCH_RSS_KEY_LEN,
                .rss_hf      = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_TCP,
            },
        },
        .txmode = { .mq_mode = RTE_ETH_MQ_TX_NONE },
    };

    conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;
    if (conf.rx_adv_conf.rss_conf.rss_hf == 0) {
        printf("port %u: no RSS support, single queue only\n", port);
        conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    }

    ret = rte_eth_dev_configure(port, n_queues, n_queues, &conf);
    if (ret < 0) return ret;

    uint16_t nb_rxd = BENCH_RX_RING, nb_txd = BENCH_TX_RING;
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (ret < 0) return ret;

    rxconf = dev_info.default_rxconf;
    for (q = 0; q < n_queues; q++) {
        ret = rte_eth_rx_queue_setup(port, q, nb_rxd,
                                     rte_eth_dev_socket_id(port), &rxconf, pool);
        if (ret < 0) return ret;
    }
    txconf = dev_info.default_txconf;
    txconf.offloads = conf.txmode.offloads;
    for (q = 0; q < n_queues; q++) {
        ret = rte_eth_tx_queue_setup(port, q, nb_txd,
                                     rte_eth_dev_socket_id(port), &txconf);
        if (ret < 0) return ret;
    }

    ret = rte_eth_dev_start(port);
    if (ret < 0) return ret;
    rte_eth_promiscuous_enable(port);

    printf("port %u: up, %u RX/TX queue pairs, symmetric RSS\n", port, n_queues);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Run loop                                                            */
/* ------------------------------------------------------------------ */
static const struct bench_ops *bench_active_ops;

static int bench_lcore_main(void *arg)
{
    struct bench_worker *w = (struct bench_worker *)arg;
    const struct bench_ops *ops = bench_active_ops;
    const int fwd = bench_opts.forward;

    printf("core %u: polling port %u queue %u (%s mode)\n",
           w->core_id, w->port_id, w->queue_id, fwd ? "forward" : "drop");

    w->t_start = rte_rdtsc_precise();

    for (;;) {
        if (unlikely(atomic_load_explicit(&bench_force_quit,
                                          memory_order_relaxed)))
            break;

        struct rte_mbuf *bufs[BENCH_BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(w->port_id, w->queue_id,
                                          bufs, BENCH_BURST_SIZE);
        if (unlikely(nb_rx == 0))
            continue;

        w->rx_pkts += nb_rx;
        for (uint16_t i = 0; i < nb_rx; i++)
            w->rx_bytes += rte_pktmbuf_pkt_len(bufs[i]);

        uint64_t t0 = rte_rdtsc_precise();
        ops->proc_burst(w, bufs, nb_rx, t0);
        uint64_t t1 = rte_rdtsc_precise();

        if (w->burst_count < BENCH_MAX_BURSTS) {
            w->burst_cycles[w->burst_count] = t1 - t0;
            w->burst_pkts[w->burst_count]   = nb_rx;
            w->burst_count++;
        }

        if (fwd) {
            /* MAC swap then bounce back out the same queue */
            for (uint16_t i = 0; i < nb_rx; i++) {
                struct rte_ether_hdr *eth =
                    rte_pktmbuf_mtod(bufs[i], struct rte_ether_hdr *);
                struct rte_ether_addr tmp;
                rte_ether_addr_copy(&eth->src_addr, &tmp);
                rte_ether_addr_copy(&eth->dst_addr, &eth->src_addr);
                rte_ether_addr_copy(&tmp, &eth->dst_addr);
            }
            uint16_t nb_tx = rte_eth_tx_burst(w->port_id, w->queue_id,
                                              bufs, nb_rx);
            w->tx_pkts += nb_tx;
            for (uint16_t i = nb_tx; i < nb_rx; i++)
                rte_pktmbuf_free(bufs[i]);
        } else {
            for (uint16_t i = 0; i < nb_rx; i++)
                rte_pktmbuf_free(bufs[i]);
        }
    }

    w->t_stop = rte_rdtsc_precise();
    return 0;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */
static int bench_cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static uint64_t bench_pctl(uint64_t *sorted, uint32_t n, double p)
{
    if (n == 0) return 0;
    uint32_t idx = (uint32_t)(p * (n - 1) + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

static void bench_dump_results(const struct bench_ops *ops)
{
    double hz = (double)rte_get_tsc_hz();
    char path[256];

    /* raw event samples (one file, tagged by core) */
    snprintf(path, sizeof(path), "%s_%s_samples.csv",
             bench_opts.out_prefix, ops->name);
    FILE *fs = fopen(path, "w");
    if (fs) fprintf(fs, "core,event_cycles,event_ns\n");

    /* per-core summary */
    snprintf(path, sizeof(path), "%s_%s_summary.csv",
             bench_opts.out_prefix, ops->name);
    FILE *fc = fopen(path, "w");
    if (fc)
        fprintf(fc, "core,elapsed_s,rx_pkts,rx_mpps,rx_gbps,tx_pkts,"
                    "hit_pkts,track_updates,table_inserts,insert_fail,"
                    "windows_done,ev_samples,"
                    "ev_p50_ns,ev_p90_ns,ev_p99_ns,ev_p999_ns,"
                    "burst_p50_cyc,burst_p99_cyc,cyc_per_pkt_mean\n");

    uint64_t agg_rx = 0, agg_hit = 0, agg_upd = 0, agg_win = 0, agg_ins = 0;
    double   agg_mpps = 0, agg_gbps = 0;

    printf("\n================ %s: results ================\n", ops->name);
    for (unsigned c = 0; c < bench_n_lcores; c++) {
        struct bench_worker *w = &bench_workers[c];
        if (w->t_stop <= w->t_start) continue;

        double secs = (double)(w->t_stop - w->t_start) / hz;
        double mpps = w->rx_pkts / secs / 1e6;
        double gbps = w->rx_bytes * 8.0 / secs / 1e9;

        if (fs)
            for (uint32_t i = 0; i < w->ev_count; i++)
                fprintf(fs, "%u,%" PRIu64 ",%.1f\n", c, w->ev_cycles[i],
                        (double)w->ev_cycles[i] / hz * 1e9);

        qsort(w->ev_cycles, w->ev_count, sizeof(uint64_t), bench_cmp_u64);
        uint64_t p50  = bench_pctl(w->ev_cycles, w->ev_count, 0.50);
        uint64_t p90  = bench_pctl(w->ev_cycles, w->ev_count, 0.90);
        uint64_t p99  = bench_pctl(w->ev_cycles, w->ev_count, 0.99);
        uint64_t p999 = bench_pctl(w->ev_cycles, w->ev_count, 0.999);

        /* mean cycles per packet from burst measurements */
        uint64_t bc_sum = 0, bp_sum = 0;
        for (uint32_t i = 0; i < w->burst_count; i++) {
            bc_sum += w->burst_cycles[i];
            bp_sum += w->burst_pkts[i];
        }
        double cyc_pkt = bp_sum ? (double)bc_sum / (double)bp_sum : 0.0;

        qsort(w->burst_cycles, w->burst_count, sizeof(uint64_t), bench_cmp_u64);
        uint64_t bp50 = bench_pctl(w->burst_cycles, w->burst_count, 0.50);
        uint64_t bp99 = bench_pctl(w->burst_cycles, w->burst_count, 0.99);

        printf("core %2u | %.2fs | rx %.3f Mpps %.3f Gbps | hits %" PRIu64
               " | upd %" PRIu64 " | win %" PRIu64
               " | ev ns p50/p99 %.0f/%.0f | cyc/pkt %.1f\n",
               c, secs, mpps, gbps, w->hit_pkts, w->track_updates,
               w->windows_done,
               (double)p50 / hz * 1e9, (double)p99 / hz * 1e9, cyc_pkt);

        if (fc)
            fprintf(fc, "%u,%.3f,%" PRIu64 ",%.4f,%.4f,%" PRIu64 ",%" PRIu64
                        ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                        ",%u,%.1f,%.1f,%.1f,%.1f,%" PRIu64 ",%" PRIu64 ",%.2f\n",
                    c, secs, w->rx_pkts, mpps, gbps, w->tx_pkts,
                    w->hit_pkts, w->track_updates, w->table_inserts,
                    w->insert_fail, w->windows_done, w->ev_count,
                    (double)p50 / hz * 1e9, (double)p90 / hz * 1e9,
                    (double)p99 / hz * 1e9, (double)p999 / hz * 1e9,
                    bp50, bp99, cyc_pkt);

        if (ops->app_report) ops->app_report(w, stdout);

        agg_rx += w->rx_pkts; agg_hit += w->hit_pkts;
        agg_upd += w->track_updates; agg_win += w->windows_done;
        agg_ins += w->table_inserts;
        agg_mpps += mpps; agg_gbps += gbps;
    }
    printf("---------------------------------------------------\n");
    printf("TOTAL   | rx %.3f Mpps %.3f Gbps | pkts %" PRIu64
           " | hits %" PRIu64 " | updates %" PRIu64
           " | inserts %" PRIu64 " | windows %" PRIu64 "\n",
           agg_mpps, agg_gbps, agg_rx, agg_hit, agg_upd, agg_ins, agg_win);
    printf("sink=%f (ignore; anti-elision)\n",
           (double)bench_workers[0].feature_sink);

    if (fs) fclose(fs);
    if (fc) fclose(fc);
}

/* ------------------------------------------------------------------ */
/* Signal + CLI                                                        */
/* ------------------------------------------------------------------ */
static void bench_sig_handler(int sig)
{
    (void)sig;
    atomic_store(&bench_force_quit, 1);
}

static void bench_parse_args(int argc, char **argv)
{
    int opt;
    while ((opt = getopt(argc, argv, "s:d:f:o:")) != -1) {
        switch (opt) {
        case 's': bench_opts.sample_every = (uint32_t)atoi(optarg); break;
        case 'd': bench_opts.duration_sec = (uint32_t)atoi(optarg); break;
        case 'f': bench_opts.forward      = atoi(optarg);           break;
        case 'o': bench_opts.out_prefix   = optarg;                 break;
        default:
            printf("usage: ... -- [-s sample_every] [-d duration_s]"
                   " [-f 0|1 forward] [-o out_prefix]\n");
            exit(1);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Generic main                                                        */
/* ------------------------------------------------------------------ */
static int bench_main(int argc, char **argv, const struct bench_ops *ops)
{
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_panic("EAL init failed\n");
    argc -= ret; argv += ret;
    bench_parse_args(argc, argv);

    signal(SIGINT,  bench_sig_handler);
    signal(SIGTERM, bench_sig_handler);

    bench_active_ops = ops;
    bench_n_lcores = rte_lcore_count();

    printf("%s bench | DPDK %s | %u lcores | TSC %.2f GHz | "
           "sample_every=%u duration=%us forward=%d\n",
           ops->name, rte_version(), bench_n_lcores,
           rte_get_tsc_hz() / 1e9, bench_opts.sample_every,
           bench_opts.duration_sec, bench_opts.forward);

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) rte_exit(EXIT_FAILURE, "no ports available\n");

    struct rte_mempool *pool = rte_pktmbuf_pool_create("MBUF_POOL",
        BENCH_NUM_MBUFS * nb_ports, BENCH_MBUF_CACHE, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool) rte_exit(EXIT_FAILURE, "mbuf pool creation failed\n");

    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid)
        if (bench_port_init(portid, pool, bench_n_lcores) != 0)
            rte_exit(EXIT_FAILURE, "port %u init failed\n", portid);

    /* per-core setup */
    uint16_t q = 0;
    for (unsigned c = 0; c < bench_n_lcores; c++) {
        struct bench_worker *w = &bench_workers[c];
        memset(w, 0, sizeof(*w));
        w->core_id  = c;
        w->port_id  = 0;
        w->queue_id = q++;
        w->ev_cycles    = rte_zmalloc(NULL,
                            BENCH_MAX_SAMPLES * sizeof(uint64_t), 0);
        w->burst_cycles = rte_zmalloc(NULL,
                            BENCH_MAX_BURSTS  * sizeof(uint64_t), 0);
        w->burst_pkts   = rte_zmalloc(NULL,
                            BENCH_MAX_BURSTS  * sizeof(uint16_t), 0);
        if (!w->ev_cycles || !w->burst_cycles || !w->burst_pkts)
            rte_exit(EXIT_FAILURE, "sample buffer alloc failed core %u\n", c);
        if (ops->app_init(w) != 0)
            rte_exit(EXIT_FAILURE, "app init failed core %u\n", c);
    }

    /* launch workers */
    for (unsigned c = 0; c < bench_n_lcores; c++)
        if (c != rte_get_main_lcore())
            rte_eal_remote_launch(bench_lcore_main, &bench_workers[c], c);

    /* main lcore: either run a worker too, or supervise duration.
     * We run the worker on main and use an alarm-style duration check
     * inside a small wrapper. */
    if (bench_opts.duration_sec > 0) {
        /* run worker on main but stop everyone after duration */
        uint64_t deadline = rte_rdtsc() +
            (uint64_t)bench_opts.duration_sec * rte_get_tsc_hz();
        struct bench_worker *w = &bench_workers[rte_get_main_lcore()];
        const struct bench_ops *o = bench_active_ops;
        const int fwd = bench_opts.forward;
        w->t_start = rte_rdtsc_precise();
        for (;;) {
            if (rte_rdtsc() >= deadline) {
                atomic_store(&bench_force_quit, 1);
                break;
            }
            if (atomic_load_explicit(&bench_force_quit,
                                     memory_order_relaxed)) break;
            struct rte_mbuf *bufs[BENCH_BURST_SIZE];
            uint16_t nb_rx = rte_eth_rx_burst(w->port_id, w->queue_id,
                                              bufs, BENCH_BURST_SIZE);
            if (unlikely(nb_rx == 0)) continue;
            w->rx_pkts += nb_rx;
            for (uint16_t i = 0; i < nb_rx; i++)
                w->rx_bytes += rte_pktmbuf_pkt_len(bufs[i]);
            uint64_t t0 = rte_rdtsc_precise();
            o->proc_burst(w, bufs, nb_rx, t0);
            uint64_t t1 = rte_rdtsc_precise();
            if (w->burst_count < BENCH_MAX_BURSTS) {
                w->burst_cycles[w->burst_count] = t1 - t0;
                w->burst_pkts[w->burst_count]   = nb_rx;
                w->burst_count++;
            }
            if (fwd) {
                for (uint16_t i = 0; i < nb_rx; i++) {
                    struct rte_ether_hdr *eth =
                        rte_pktmbuf_mtod(bufs[i], struct rte_ether_hdr *);
                    struct rte_ether_addr tmp;
                    rte_ether_addr_copy(&eth->src_addr, &tmp);
                    rte_ether_addr_copy(&eth->dst_addr, &eth->src_addr);
                    rte_ether_addr_copy(&tmp, &eth->dst_addr);
                }
                uint16_t nb_tx = rte_eth_tx_burst(w->port_id, w->queue_id,
                                                  bufs, nb_rx);
                w->tx_pkts += nb_tx;
                for (uint16_t i = nb_tx; i < nb_rx; i++)
                    rte_pktmbuf_free(bufs[i]);
            } else {
                for (uint16_t i = 0; i < nb_rx; i++)
                    rte_pktmbuf_free(bufs[i]);
            }
        }
        w->t_stop = rte_rdtsc_precise();
    } else {
        bench_lcore_main(&bench_workers[rte_get_main_lcore()]);
    }

    rte_eal_mp_wait_lcore();
    bench_dump_results(ops);

    RTE_ETH_FOREACH_DEV(portid) {
        rte_eth_dev_stop(portid);
        rte_eth_dev_close(portid);
    }
    rte_eal_cleanup();
    return 0;
}

#endif /* BENCH_COMMON_H */
