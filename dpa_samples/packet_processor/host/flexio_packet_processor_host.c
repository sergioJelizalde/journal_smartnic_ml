/*
 * SCALED FLEXIO REFLECTOR - DOCA 3.4
 * 
 * KEY INSIGHT: Scale to 190+ threads by using a THREAD POOL instead of
 * per-EU event handlers. This is what BenchBF3 does.
 * 
 * Your current design (N handlers on N EUs):
 *   - Limited to ~16 EUs available
 *   - Each handler = 1 EU affinity
 *   - Scales to 16 workers max
 * 
 * Production design (thread pool):
 *   - Create N threads in DPA with FLEXIBLE affinity
 *   - All threads poll shared work queue / RQ CQ ring
 *   - Hardware can schedule across all available EUs
 *   - Scales to 190+ threads
 */

#include <unistd.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <infiniband/mlx5dv.h>
#include <libflexio/flexio_ver.h>

#define FLEXIO_VER_USED FLEXIO_VER(26, 4, 0)

#include <libflexio/flexio.h>
#include "rss_steering.h"
#include "../flexio_packet_processor_com.h"

extern flexio_func_t flexio_pp_dev;

/* ================================================================== */
/* SCALED DESIGN PARAMETERS                                           */
/* ================================================================== */

/* Create a single shared RQ pair + CQ pair, with many threads polling it.
 * BenchBF3 approach: all threads read from the same RQ.
 */
#define L2V(l) (1UL << (l))
#define LOG_Q_DEPTH 8                  /* Increase from 7 to 8 for more buffer */
#define Q_DEPTH L2V(LOG_Q_DEPTH)
#define LOG_Q_DATA_ENTRY_BSIZE 11
#define Q_DATA_ENTRY_BSIZE L2V(LOG_Q_DATA_ENTRY_BSIZE)
#define Q_DATA_BSIZE (Q_DEPTH * Q_DATA_ENTRY_BSIZE)

#define CQE_BSIZE 64
#define CQ_BSIZE (Q_DEPTH * CQE_BSIZE)

#define LOG_SQ_WQE_BSIZE 6
#define SQ_WQE_BSIZE L2V(LOG_SQ_WQE_BSIZE)
#define SQ_RING_BSIZE (Q_DEPTH * SQ_WQE_BSIZE)

#define LOG_RQ_WQE_BSIZE 6
#define RQ_WQE_BSIZE L2V(LOG_RQ_WQE_BSIZE)
#define RQ_RING_BSIZE (Q_DEPTH * RQ_WQE_BSIZE)

/* Number of DPA threads to create (scale from 16 to 190+) */
#define NUM_THREADS 64                 /* Can go up to 190 with proper partition */
#define MAX_THREADS 256

/* Base source MAC for RX steering */
#define SMAC_BASE 0x0208a4d8ff43

#define MSG_HOST_BUFF_BSIZE (512 * L2V(FLEXIO_MSG_DEV_LOG_DATA_CHUNK_BSIZE))

/* ================================================================== */
/* THREAD POOL CONTEXT (shared by all threads)                        */
/* ================================================================== */
struct shared_context {
	/* Single RQ + CQ pair (all threads poll same CQ) */
	struct flexio_cq *rq_cq;
	struct flexio_rq *rq;
	struct flexio_mkey *rqd_mkey;

	/* Single SQ + CQ pair (all threads write to same SQ) */
	struct flexio_cq *sq_cq;
	struct flexio_sq *sq;
	struct flexio_mkey *sqd_mkey;

	/* Transfer structs for DPA */
	struct app_transfer_cq sq_cq_transf;
	struct app_transfer_wq sq_transf;
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_wq rq_transf;

	/* DPA heap address of shared data */
	flexio_uintptr_t app_data_daddr;
};

/* Global application context */
struct app_context {
	struct flexio_process *flexio_process;
	struct flexio_app *flexio_app;
	struct flexio_msg_stream *stream;

	struct flexio_uar *process_uar;
	struct ibv_pd *process_pd;
	struct ibv_context *ibv_ctx;

	struct rss_steering_ctx *rss;
	struct flow_matcher *rx_matcher;
	struct flow_rule *rx_rule;
	struct flow_matcher *tx_matcher;
	struct flow_rule *tx_rule_table;
	struct flow_rule *tx_rule_vport;

	int num_threads;
	struct flexio_event_handler *threads[MAX_THREADS];  /* Thread pool */
	struct shared_context shared_ctx;
};

/* ================================================================== */
/* IBV device open (unchanged)                                        */
/* ================================================================== */
static int app_open_ibv_ctx(struct app_context *app_ctx, char *device)
{
	struct ibv_device **dev_list;
	int ret = 0, dev_i;

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		fprintf(stderr, "Failed to get IB devices list\n");
		return -1;
	}

	for (dev_i = 0; dev_list[dev_i]; dev_i++)
		if (!strcmp(ibv_get_device_name(dev_list[dev_i]), device))
			break;

	if (!dev_list[dev_i]) {
		fprintf(stderr, "No IBV device found for device name '%s'\n", device);
		ret = -1;
		goto cleanup;
	}

	{
		struct mlx5dv_context_attr dv_attr = {
			.flags = MLX5DV_CONTEXT_FLAGS_DEVX,
		};
		app_ctx->ibv_ctx = mlx5dv_open_device(dev_list[dev_i], &dv_attr);
	}
	if (!app_ctx->ibv_ctx) {
		fprintf(stderr, "Couldn't open an IBV context for device '%s'\n", device);
		ret = -1;
	}

cleanup:
	ibv_free_device_list(dev_list);
	return ret;
}

/* ================================================================== */
/* MKey creation                                                       */
/* ================================================================== */
static struct flexio_mkey *create_dpa_mkey(struct app_context *app_ctx,
					   flexio_uintptr_t daddr)
{
	struct flexio_mkey_attr mkey_attr = {0};
	struct flexio_mkey *mkey;

	mkey_attr.pd = app_ctx->process_pd;
	mkey_attr.daddr = daddr;
	mkey_attr.len = Q_DATA_BSIZE;
	mkey_attr.access = IBV_ACCESS_LOCAL_WRITE;
	if (flexio_device_mkey_create(app_ctx->flexio_process, &mkey_attr, &mkey)) {
		fprintf(stderr, "Failed to create Flex IO Mkey\n");
		return NULL;
	}

	return mkey;
}

/* ================================================================== */
/* CQ memory allocation                                               */
/* ================================================================== */
static int cq_mem_alloc(struct flexio_process *process, struct app_transfer_cq *cq_transf)
{
	struct mlx5_cqe64 *cq_ring_src;
	struct mlx5_cqe64 *cqe;
	__be32 dbr[2] = { 0, 0 };
	int ret = 0;
	uint32_t i;

	if (flexio_copy_from_host(process, dbr, sizeof(dbr), &cq_transf->cq_dbr_daddr)) {
		fprintf(stderr, "Failed to allocate CQ DBR memory on DPA heap.\n");
		return -1;
	}

	cq_ring_src = calloc(Q_DEPTH, CQE_BSIZE);
	if (!cq_ring_src) {
		fprintf(stderr, "Failed to allocate memory for cq_ring_src.\n");
		return -1;
	}

	for (i = 0, cqe = cq_ring_src; i < Q_DEPTH; i++)
		mlx5dv_set_cqe_owner(cqe++, 1);

	if (flexio_copy_from_host(process, cq_ring_src, CQ_BSIZE, &cq_transf->cq_ring_daddr)) {
		fprintf(stderr, "Failed to allocate CQ ring memory on DPA heap.\n");
		ret = -1;
	}

	free(cq_ring_src);
	return ret;
}

/* ================================================================== */
/* SQ memory allocation                                               */
/* ================================================================== */
static int sq_mem_alloc(struct flexio_process *process, struct app_transfer_wq *sq_transf)
{
	flexio_buf_dev_alloc(process, Q_DATA_BSIZE, &sq_transf->wqd_daddr);
	if (!sq_transf->wqd_daddr)
		return -1;

	flexio_buf_dev_alloc(process, SQ_RING_BSIZE, &sq_transf->wq_ring_daddr);
	if (!sq_transf->wq_ring_daddr)
		return -1;

	return 0;
}

/* ================================================================== */
/* Shared SQ + SQ-CQ creation                                          */
/* ================================================================== */
static int create_shared_sq(struct app_context *app_ctx, struct shared_context *sc)
{
	struct flexio_process *app_fp = app_ctx->flexio_process;
	struct flexio_cq_attr sqcq_attr = {0};
	struct flexio_wq_attr sq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	if (cq_mem_alloc(app_fp, &sc->sq_cq_transf)) {
		fprintf(stderr, "Failed to alloc memory for SQ's CQ.\n");
		return -1;
	}

	sqcq_attr.log_cq_depth = LOG_Q_DEPTH;
	/* SQ CQ is polled by DPA threads (non-handler CQ) */
	sqcq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ;
	sqcq_attr.uar_id = uar_id;
	sqcq_attr.cq_dbr_daddr = sc->sq_cq_transf.cq_dbr_daddr;
	sqcq_attr.cq_ring_qmem.daddr = sc->sq_cq_transf.cq_ring_daddr;
	if (flexio_cq_create(app_fp, NULL, &sqcq_attr, &sc->sq_cq)) {
		fprintf(stderr, "Failed to create Flex IO SQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(sc->sq_cq);
	sc->sq_cq_transf.cq_num = cq_num;
	sc->sq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	if (sq_mem_alloc(app_fp, &sc->sq_transf)) {
		fprintf(stderr, "Failed to allocate memory for SQ\n");
		return -1;
	}

	sq_attr.log_wq_depth = LOG_Q_DEPTH;
	sq_attr.uar_id = uar_id;
	sq_attr.wq_ring_qmem.daddr = sc->sq_transf.wq_ring_daddr;
	sq_attr.pd = app_ctx->process_pd;

	if (flexio_sq_create(app_fp, NULL, cq_num, &sq_attr, &sc->sq)) {
		fprintf(stderr, "Failed to create Flex IO SQ\n");
		return -1;
	}

	sc->sq_transf.wq_num = flexio_sq_get_wq_num(sc->sq);

	sc->sqd_mkey = create_dpa_mkey(app_ctx, sc->sq_transf.wqd_daddr);
	if (!sc->sqd_mkey) {
		fprintf(stderr, "Failed to create an MKey for SQ data buffer\n");
		return -1;
	}
	sc->sq_transf.wqd_mkey_id = flexio_mkey_get_id(sc->sqd_mkey);

	return 0;
}

/* ================================================================== */
/* RQ memory allocation                                               */
/* ================================================================== */
static int rq_mem_alloc(struct flexio_process *process, struct app_transfer_wq *rq_transf)
{
	__be32 dbr[2] = { 0, 0 };

	flexio_buf_dev_alloc(process, Q_DATA_BSIZE, &rq_transf->wqd_daddr);
	if (!rq_transf->wqd_daddr)
		return -1;

	flexio_buf_dev_alloc(process, RQ_RING_BSIZE, &rq_transf->wq_ring_daddr);
	if (!rq_transf->wq_ring_daddr)
		return -1;

	flexio_copy_from_host(process, dbr, sizeof(dbr), &rq_transf->wq_dbr_daddr);
	if (!rq_transf->wq_dbr_daddr)
		return -1;

	return 0;
}

static int init_dpa_rq_ring(struct app_context *app_ctx, struct shared_context *sc)
{
	flexio_uintptr_t wqe_data_daddr = sc->rq_transf.wqd_daddr;
	uint32_t mkey_id = sc->rq_transf.wqd_mkey_id;
	struct mlx5_wqe_data_seg *rx_wqes;
	struct mlx5_wqe_data_seg *dseg;
	int retval = 0;
	uint32_t i;

	rx_wqes = calloc(1, RQ_RING_BSIZE);
	if (!rx_wqes) {
		fprintf(stderr, "Failed to allocate memory for rx_wqes\n");
		return -1;
	}

	for (i = 0, dseg = rx_wqes; i < Q_DEPTH; i++, dseg++) {
		mlx5dv_set_data_seg(dseg, Q_DATA_ENTRY_BSIZE, mkey_id, wqe_data_daddr);
		wqe_data_daddr += Q_DATA_ENTRY_BSIZE;
	}

	if (flexio_host2dev_memcpy(app_ctx->flexio_process, rx_wqes, RQ_RING_BSIZE,
				   sc->rq_transf.wq_ring_daddr))
		retval = -1;

	free(rx_wqes);
	return retval;
}

static int init_rq_dbr(struct app_context *app_ctx, struct shared_context *sc)
{
	__be32 dbr[2];

	dbr[0] = htobe32(Q_DEPTH & 0xffff);
	dbr[1] = htobe32(0);
	if (flexio_host2dev_memcpy(app_ctx->flexio_process, dbr, sizeof(dbr),
				   sc->rq_transf.wq_dbr_daddr))
		return -1;

	return 0;
}

/* ================================================================== */
/* Shared RQ + RQ-CQ creation (single pair, no handler binding)        */
/* ================================================================== */
static int create_shared_rq(struct app_context *app_ctx, struct shared_context *sc)
{
	struct flexio_process *app_fp = app_ctx->flexio_process;
	struct flexio_cq_attr rqcq_attr = {0};
	struct flexio_wq_attr rq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	if (cq_mem_alloc(app_fp, &sc->rq_cq_transf)) {
		fprintf(stderr, "Failed to alloc memory for RQ's CQ.\n");
		return -1;
	}

	rqcq_attr.log_cq_depth = LOG_Q_DEPTH;
	/* KEY: RQ CQ is NOT bound to a specific handler.
	 * Multiple threads will poll it independently.
	 */
	rqcq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ;
	rqcq_attr.uar_id = uar_id;
	rqcq_attr.cq_dbr_daddr = sc->rq_cq_transf.cq_dbr_daddr;
	rqcq_attr.cq_ring_qmem.daddr = sc->rq_cq_transf.cq_ring_daddr;
	if (flexio_cq_create(app_fp, NULL, &rqcq_attr, &sc->rq_cq)) {
		fprintf(stderr, "Failed to create Flex IO RQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(sc->rq_cq);
	sc->rq_cq_transf.cq_num = cq_num;
	sc->rq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	if (rq_mem_alloc(app_fp, &sc->rq_transf)) {
		fprintf(stderr, "Failed to allocate memory for RQ.\n");
		return -1;
	}

	sc->rqd_mkey = create_dpa_mkey(app_ctx, sc->rq_transf.wqd_daddr);
	if (!sc->rqd_mkey) {
		fprintf(stderr, "Failed to create an MKey for RQ data buffer.\n");
		return -1;
	}
	sc->rq_transf.wqd_mkey_id = flexio_mkey_get_id(sc->rqd_mkey);

	if (init_dpa_rq_ring(app_ctx, sc)) {
		fprintf(stderr, "Failed to init RQ ring.\n");
		return -1;
	}

	rq_attr.log_wq_depth = LOG_Q_DEPTH;
	rq_attr.pd = app_ctx->process_pd;
	rq_attr.wq_dbr_qmem.memtype = FLEXIO_MEMTYPE_DPA;
	rq_attr.wq_dbr_qmem.daddr = sc->rq_transf.wq_dbr_daddr;
	rq_attr.wq_ring_qmem.daddr = sc->rq_transf.wq_ring_daddr;
	if (flexio_rq_create(app_fp, NULL, cq_num, &rq_attr, &sc->rq)) {
		fprintf(stderr, "Failed to create Flex IO RQ.\n");
		return -1;
	}

	sc->rq_transf.wq_num = flexio_rq_get_wq_num(sc->rq);
	if (init_rq_dbr(app_ctx, sc)) {
		fprintf(stderr, "Failed to init RQ DBR.\n");
		return -1;
	}

	return 0;
}

/* ================================================================== */
/* Steering (same as before, but single RQ now)                       */
/* ================================================================== */
static int create_steering_rules(struct app_context *app_ctx, int nic_mode)
{
	uint32_t rqn = app_ctx->shared_ctx.rq_transf.wq_num;

	/* Single RQT with just this one RQ */
	app_ctx->rss = rss_steering_create(app_ctx->ibv_ctx, &rqn, 1, SMAC_BASE);
	if (!app_ctx->rss) {
		fprintf(stderr, "Failed to create RSS steering chain\n");
		return -1;
	}

	app_ctx->rx_matcher = create_matcher_rx(app_ctx->ibv_ctx);
	if (!app_ctx->rx_matcher) {
		fprintf(stderr, "Failed to create RX matcher\n");
		return -1;
	}

	app_ctx->rx_rule = create_rule_rx_mac_match(app_ctx->rx_matcher,
			rss_steering_get_tir(app_ctx->rss, 1),
			SMAC_BASE);
	if (!app_ctx->rx_rule) {
		fprintf(stderr, "Failed to create RX->RSS-TIR rule\n");
		return -1;
	}

	if (!nic_mode) {
		app_ctx->tx_matcher = create_matcher_tx(app_ctx->ibv_ctx);
		if (!app_ctx->tx_matcher) {
			fprintf(stderr, "Failed to create TX matcher\n");
			return -1;
		}

		app_ctx->tx_rule_table =
			create_rule_tx_fwd_to_sws_table(app_ctx->tx_matcher, SMAC_BASE);
		if (!app_ctx->tx_rule_table) {
			fprintf(stderr, "Failed to create TX table steering rule\n");
			return -1;
		}

		app_ctx->tx_rule_vport =
			create_rule_tx_fwd_to_vport(app_ctx->tx_matcher, SMAC_BASE);
		if (!app_ctx->tx_rule_vport) {
			fprintf(stderr, "Failed to create TX vport steering rule\n");
			return -1;
		}
	}

	return 0;
}

/* ================================================================== */
/* Copy shared data to DPA                                            */
/* ================================================================== */
static int copy_shared_data_to_dpa(struct app_context *app_ctx)
{
	uint64_t struct_bsize = sizeof(struct host2dev_packet_processor_data);
	struct host2dev_packet_processor_data *h2d_data;
	int ret = 0;

	h2d_data = calloc(1, struct_bsize);
	if (!h2d_data) {
		fprintf(stderr, "Failed to allocate memory for h2d_data\n");
		return -1;
	}

	h2d_data->sq_cq_transf = app_ctx->shared_ctx.sq_cq_transf;
	h2d_data->sq_transf = app_ctx->shared_ctx.sq_transf;
	h2d_data->rq_cq_transf = app_ctx->shared_ctx.rq_cq_transf;
	h2d_data->rq_transf = app_ctx->shared_ctx.rq_transf;
	h2d_data->not_first_run = 0;

	if (flexio_copy_from_host(app_ctx->flexio_process, h2d_data, struct_bsize,
				  &app_ctx->shared_ctx.app_data_daddr)) {
		fprintf(stderr, "Failed to copy shared data to DPA.\n");
		ret = -1;
	}

	free(h2d_data);
	return ret;
}

/* ================================================================== */
/* Create thread pool (NUM_THREADS threads with flexible affinity)    */
/* ================================================================== */
static int create_thread_pool(struct app_context *app_ctx)
{
	struct flexio_event_handler_attr eh_attr = {0};
	char eh_name[32];
	int i;

	eh_attr.host_stub_func = flexio_pp_dev;
	/* KEY: No STRICT affinity - DPA can schedule threads flexibly */
	eh_attr.affinity.type = FLEXIO_AFFINITY_FLEXIBLE;

	for (i = 0; i < app_ctx->num_threads; i++) {
		snprintf(eh_name, sizeof(eh_name), "pp_thread_%d", i);

		if (flexio_event_handler_create(app_ctx->flexio_process, &eh_attr, 
					       &app_ctx->threads[i])) {
			fprintf(stderr, "Failed to create thread %d\n", i);
			return -1;
		}
	}

	printf("Created %d thread pool with flexible affinity\n", app_ctx->num_threads);
	return 0;
}

/* ================================================================== */
/* Launch all threads (all get same shared data daddr)                */
/* ================================================================== */
static int launch_thread_pool(struct app_context *app_ctx)
{
	int i;

	for (i = 0; i < app_ctx->num_threads; i++) {
		if (flexio_event_handler_run(app_ctx->threads[i], 
					     app_ctx->shared_ctx.app_data_daddr)) {
			fprintf(stderr, "Failed to run thread %d.\n", i);
			return -1;
		}
	}

	return 0;
}

/* ================================================================== */
/* Cleanup                                                            */
/* ================================================================== */
static int cleanup_app(struct app_context *app_ctx)
{
	struct flexio_process *fp = app_ctx->flexio_process;
	int err = 0, i;

	/* Destroy threads */
	for (i = 0; i < app_ctx->num_threads; i++) {
		if (app_ctx->threads[i] && flexio_event_handler_destroy(app_ctx->threads[i]))
			err = -1;
	}

	/* Destroy shared resources */
	if (app_ctx->shared_ctx.app_data_daddr && 
	    flexio_buf_dev_free(fp, app_ctx->shared_ctx.app_data_daddr))
		err = -1;

	if (app_ctx->shared_ctx.sq && flexio_sq_destroy(app_ctx->shared_ctx.sq))
		err = -1;
	if (app_ctx->shared_ctx.sqd_mkey && flexio_device_mkey_destroy(app_ctx->shared_ctx.sqd_mkey))
		err = -1;
	if (app_ctx->shared_ctx.sq_transf.wq_ring_daddr && 
	    flexio_buf_dev_free(fp, app_ctx->shared_ctx.sq_transf.wq_ring_daddr))
		err = -1;
	if (app_ctx->shared_ctx.sq_transf.wqd_daddr && 
	    flexio_buf_dev_free(fp, app_ctx->shared_ctx.sq_transf.wqd_daddr))
		err = -1;
	if (app_ctx->shared_ctx.sq_cq && flexio_cq_destroy(app_ctx->shared_ctx.sq_cq))
		err = -1;

	if (app_ctx->shared_ctx.rq && flexio_rq_destroy(app_ctx->shared_ctx.rq))
		err = -1;
	if (app_ctx->shared_ctx.rqd_mkey && flexio_device_mkey_destroy(app_ctx->shared_ctx.rqd_mkey))
		err = -1;
	if (app_ctx->shared_ctx.rq_transf.wq_ring_daddr && 
	    flexio_buf_dev_free(fp, app_ctx->shared_ctx.rq_transf.wq_ring_daddr))
		err = -1;
	if (app_ctx->shared_ctx.rq_transf.wqd_daddr && 
	    flexio_buf_dev_free(fp, app_ctx->shared_ctx.rq_transf.wqd_daddr))
		err = -1;
	if (app_ctx->shared_ctx.rq_cq && flexio_cq_destroy(app_ctx->shared_ctx.rq_cq))
		err = -1;

	if (app_ctx->rx_rule && destroy_rule(app_ctx->rx_rule))
		err = -1;
	if (app_ctx->rx_matcher && destroy_matcher(app_ctx->rx_matcher))
		err = -1;

	if (app_ctx->rss)
		rss_steering_destroy(app_ctx->rss);

	if (app_ctx->tx_rule_vport && destroy_rule(app_ctx->tx_rule_vport))
		err = -1;
	if (app_ctx->tx_rule_table && destroy_rule(app_ctx->tx_rule_table))
		err = -1;
	if (app_ctx->tx_matcher && destroy_matcher(app_ctx->tx_matcher))
		err = -1;

	if (app_ctx->flexio_process && app_ctx->stream && 
	    flexio_msg_stream_destroy(app_ctx->stream))
		err = -1;

	if (app_ctx->flexio_process && flexio_process_destroy(app_ctx->flexio_process))
		err = -1;

	if (app_ctx->ibv_ctx && ibv_close_device(app_ctx->ibv_ctx))
		err = -1;

	return err;
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */
int main(int argc, char **argv)
{
	struct flexio_app_select_attr flexio_app_sel_attr = {0};
	flexio_msg_stream_attr_t stream_fattr = {0};
	struct flexio_process *app_fp = NULL;
	struct app_context app_ctx = {0};
	struct ibv_port_attr port_attr;
	uint64_t udbg_token;
	int nic_mode = 0;
	char buf[2];
	int err = 0;
	int i;

	app_ctx.num_threads = NUM_THREADS;

	printf("FlexIO Reflector - SCALED (Thread Pool)\n");
	printf("Creating %d threads with FLEXIBLE affinity\n", app_ctx.num_threads);

	if (argc < 2) {
		fprintf(stderr,
			"Usage: %s <mlx5 device> [--nic-mode] [--threads N]\n",
			argv[0]);
		return -1;
	}

	for (i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "--nic-mode")) {
			nic_mode = 1;
		} else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
			app_ctx.num_threads = atoi(argv[++i]);
			if (app_ctx.num_threads < 1 ||
			    app_ctx.num_threads > MAX_THREADS) {
				fprintf(stderr, "--threads must be 1-%d\n", MAX_THREADS);
				return -1;
			}
		} else {
			fprintf(stderr, "Invalid parameter %s\n", argv[i]);
			return -1;
		}
	}

	if (geteuid()) {
		fprintf(stderr, "Failed - the application must run with root privileges\n");
		return -1;
	}

	err = app_open_ibv_ctx(&app_ctx, argv[1]);
	if (err)
		return -1;

	if (ibv_query_port(app_ctx.ibv_ctx, 1, &port_attr)) {
		fprintf(stderr, "Failed to query IBV port attributes\n");
		err = -1;
		goto cleanup;
	}

	if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
		fprintf(stderr, "IBV port is not Ethernet, state: %d\n", port_attr.link_layer);
		err = -1;
		goto cleanup;
	}

	if (flexio_version_set(FLEXIO_VER_USED)) {
		fprintf(stderr, "Failed to set version in FlexIO API.\n");
		err = -1;
		goto cleanup;
	}

	flexio_app_sel_attr.app_name = DEV_APP_NAME_XSTR(DEV_APP_NAME);
	flexio_app_sel_attr.hw_model_id = FLEXIO_HW_MODEL_DEF;
	flexio_app_sel_attr.ibv_ctx = app_ctx.ibv_ctx;

	err = flexio_app_get(&flexio_app_sel_attr, &app_ctx.flexio_app);
	if (err) {
		fprintf(stderr, "Failed to get Flex IO app\n");
		goto cleanup;
	}

	if (flexio_process_create(app_ctx.ibv_ctx, app_ctx.flexio_app, NULL, &app_fp)) {
		fprintf(stderr, "Failed to create Flex IO process.\n");
		err = -1;
		goto cleanup;
	}
	app_ctx.flexio_process = app_fp;

	udbg_token = flexio_process_udbg_token_get(app_fp);
	if (udbg_token)
		printf("Debug token: %#lx\n", udbg_token);

	stream_fattr.data_bsize = MSG_HOST_BUFF_BSIZE;
	stream_fattr.sync_mode = FLEXIO_MSG_DEV_SYNC_MODE_SYNC;
	stream_fattr.level = FLEXIO_MSG_DEV_INFO;
	stream_fattr.transport_mode = FLEXIO_MSG_TRANSPORT_QP_RC;

	if (flexio_msg_stream_create(app_fp, &stream_fattr, stdout, NULL, &app_ctx.stream)) {
		fprintf(stderr, "Failed to init device messaging environment\n");
		err = -1;
		goto cleanup;
	}

	app_ctx.process_pd = flexio_process_get_pd(app_fp);
	app_ctx.process_uar = flexio_process_get_uar(app_fp);

	/* Create shared resources */
	if (create_shared_sq(&app_ctx, &app_ctx.shared_ctx)) {
		fprintf(stderr, "Failed to create shared SQ.\n");
		err = -1;
		goto cleanup;
	}

	if (create_shared_rq(&app_ctx, &app_ctx.shared_ctx)) {
		fprintf(stderr, "Failed to create shared RQ.\n");
		err = -1;
		goto cleanup;
	}

	if (copy_shared_data_to_dpa(&app_ctx)) {
		fprintf(stderr, "Failed to copy shared data to DPA.\n");
		err = -1;
		goto cleanup;
	}

	if (create_steering_rules(&app_ctx, nic_mode)) {
		fprintf(stderr, "Failed to create steering rules.\n");
		err = -1;
		goto cleanup;
	}

	/* Create and launch thread pool */
	if (create_thread_pool(&app_ctx)) {
		fprintf(stderr, "Failed to create thread pool.\n");
		err = -1;
		goto cleanup;
	}

	if (launch_thread_pool(&app_ctx)) {
		fprintf(stderr, "Failed to launch thread pool.\n");
		err = -1;
		goto cleanup;
	}

	printf("Ready: %d threads running. Press Enter to exit.\n", app_ctx.num_threads);

	if (!fread(buf, 1, 1, stdin))
		fprintf(stderr, "Failed in fread\n");

cleanup:
	cleanup_app(&app_ctx);
	return err;
}