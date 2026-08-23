/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * flexio_packet_processor_parallel.c
 *
 * PARALLEL (multi-worker) version of the Flex IO packet processing sample.
 * Target: DOCA 3.4 / FlexIO API 26.4 / BlueField-3.
 *
 * Design (per DOCA 3.4 DPA Development guide):
 *   - N workers, each = { event handler (STRICT EU affinity) + RQ + RQ-CQ
 *                         + SQ + SQ-CQ + private data buffers + MKeys }.
 *   - Each RQ's CQ has element_type = DPA_THREAD bound to its own handler,
 *     so N packets on N queues trigger N handlers on N EUs concurrently.
 *   - Each handler receives its OWN host2dev data daddr as its thread arg,
 *     so the UNMODIFIED device code (flexio_pp_dev) runs per worker with
 *     zero sharing and zero locking between workers.
 *   - Traffic is distributed with one RX steering rule per worker
 *     (SMAC_BASE + i). For true 5-tuple RSS, replace this with a TIR over
 *     an RQT spanning all worker RQs (see NOTE-RSS below).
 *
 * Usage:
 *   ./flexio_pp_parallel <mlx5 device> [--nic-mode] [--workers N] [--eu-base E]
 *
 * Prerequisites (DOCA 3.4):
 *   - EU partition with >= workers EUs available for this device
 *     (default partition exists on the ECPF; verify with:
 *        dpaeumgmt partition -d <dev> query
 *      For non-ECPF functions you MUST create a partition first.)
 */

#include <unistd.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

#include <infiniband/mlx5dv.h>

#include <libflexio/flexio_ver.h>

#define FLEXIO_VER_USED FLEXIO_VER(26, 4, 0)

#include <libflexio/flexio.h>

#include "flow_steering_utils.h"
#include "rss_steering.h"
#include "../flexio_packet_processor_com.h"

/* Device-side function stub (unchanged device binary). */
extern flexio_func_t flexio_pp_dev;

/* ------------------------------------------------------------------ */
/* Sizing (unchanged per-queue geometry; total scales with workers).   */
/* ------------------------------------------------------------------ */
#define L2V(l) (1UL << (l))
#define LOG_Q_DEPTH 7
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

/* Base source MAC. Worker i matches SMAC_BASE + i.
 * NOTE-RSS: for hash-based spreading of a single MAC across all workers,
 * build an RQT containing every worker RQ's WQ number and create an RSS
 * TIR over it (mlx5dv_devx TIR with rx_hash_fn=TOEPLITZ), then install a
 * single RX rule pointing at that TIR. The per-worker DPA side needs no
 * change. This sample keeps the simple per-MAC rules so it works with the
 * unmodified flow_steering_utils helpers.
 */
#define SMAC_BASE 0x0208a4d8ff43

#define MAX_WORKERS 16
#define DEFAULT_WORKERS 8

#define MSG_HOST_BUFF_BSIZE (512 * L2V(FLEXIO_MSG_DEV_LOG_DATA_CHUNK_BSIZE))

#define DEV_APP_NAME_STR(_n) #_n
#define DEV_APP_NAME_XSTR(_n) DEV_APP_NAME_STR(_n)

/* ------------------------------------------------------------------ */
/* Per-worker context: fully private resource set.                     */
/* ------------------------------------------------------------------ */
struct worker_ctx {
	/* Event handler pinned to one EU (STRICT affinity). */
	struct flexio_event_handler *pp_eh;

	/* SQ side. */
	struct flexio_cq *sq_cq;
	struct flexio_sq *sq;
	struct flexio_mkey *sqd_mkey;

	/* RQ side. */
	struct flexio_cq *rq_cq;
	struct flexio_rq *rq;
	struct flexio_mkey *rqd_mkey;

	/* Transfer structs copied to DPA heap for this worker. */
	struct app_transfer_cq sq_cq_transf;
	struct app_transfer_wq sq_transf;
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_wq rq_transf;

	/* DPA heap address of this worker's host2dev data struct.
	 * Passed as the thread argument on flexio_event_handler_run(),
	 * so each DPA thread parses its OWN private queue set.
	 */
	flexio_uintptr_t app_data_daddr;
};

/* Global application context. */
/* Global application context. */
struct app_context {
	struct flexio_process *flexio_process;
	struct flexio_app *flexio_app;
	struct flexio_msg_stream *stream;

	struct flexio_uar *process_uar;
	struct ibv_pd *process_pd;
	struct ibv_context *ibv_ctx;

	/* 5-tuple RSS chain (TD + RQT + TIRs). */
	struct rss_steering_ctx *rss;

	/* RX rule via flow_steering_utils -> RSS TIR. */   /* <-- ADD */
	struct flow_matcher *rx_matcher;                    /* <-- ADD */
	struct flow_rule *rx_rule;                          /* <-- ADD */

	struct flow_matcher *tx_matcher;
	struct flow_rule *tx_rule_table;
	struct flow_rule *tx_rule_vport;

	int num_workers;
	int eu_base;
	struct worker_ctx wk[MAX_WORKERS];
};

/* ------------------------------------------------------------------ */
/* IBV device open (unchanged).                                        */
/* ------------------------------------------------------------------ */
static int app_open_ibv_ctx(struct app_context *app_ctx, char *device)
{
	struct ibv_device **dev_list;
	int ret = 0;
	int dev_i;

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
		/* DEVX access is required for the RSS PRM commands
		 * (ALLOC_TD / CREATE_RQT / CREATE_TIR). FlexIO runs fine
		 * on a devx-enabled context.
		 */
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

/* ------------------------------------------------------------------ */
/* MKey with DPA write access (unchanged logic).                       */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* CQ memory allocation on DPA heap (unchanged logic).                 */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* SQ memory allocation (unchanged logic).                             */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* Per-worker SQ + SQ-CQ creation.                                     */
/* ------------------------------------------------------------------ */
static int create_worker_sq(struct app_context *app_ctx, struct worker_ctx *wk)
{
	struct flexio_process *app_fp = app_ctx->flexio_process;
	struct flexio_cq_attr sqcq_attr = {0};
	struct flexio_wq_attr sq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	if (cq_mem_alloc(app_fp, &wk->sq_cq_transf)) {
		fprintf(stderr, "Failed to alloc memory for SQ's CQ.\n");
		return -1;
	}

	sqcq_attr.log_cq_depth = LOG_Q_DEPTH;
	/* SQ CQ is polled by the DPA thread itself, not bound to a handler. */
	sqcq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ;
	sqcq_attr.uar_id = uar_id;
	sqcq_attr.cq_dbr_daddr = wk->sq_cq_transf.cq_dbr_daddr;
	sqcq_attr.cq_ring_qmem.daddr = wk->sq_cq_transf.cq_ring_daddr;
	if (flexio_cq_create(app_fp, NULL, &sqcq_attr, &wk->sq_cq)) {
		fprintf(stderr, "Failed to create Flex IO SQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(wk->sq_cq);
	wk->sq_cq_transf.cq_num = cq_num;
	wk->sq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	if (sq_mem_alloc(app_fp, &wk->sq_transf)) {
		fprintf(stderr, "Failed to allocate memory for SQ\n");
		return -1;
	}

	sq_attr.log_wq_depth = LOG_Q_DEPTH;
	sq_attr.uar_id = uar_id;
	sq_attr.wq_ring_qmem.daddr = wk->sq_transf.wq_ring_daddr;
	sq_attr.pd = app_ctx->process_pd;

	if (flexio_sq_create(app_fp, NULL, cq_num, &sq_attr, &wk->sq)) {
		fprintf(stderr, "Failed to create Flex IO SQ\n");
		return -1;
	}

	wk->sq_transf.wq_num = flexio_sq_get_wq_num(wk->sq);

	wk->sqd_mkey = create_dpa_mkey(app_ctx, wk->sq_transf.wqd_daddr);
	if (!wk->sqd_mkey) {
		fprintf(stderr, "Failed to create an MKey for SQ data buffer\n");
		return -1;
	}
	wk->sq_transf.wqd_mkey_id = flexio_mkey_get_id(wk->sqd_mkey);

	return 0;
}

/* ------------------------------------------------------------------ */
/* RQ memory allocation + ring/DBR init (unchanged logic, per worker). */
/* ------------------------------------------------------------------ */
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

static int init_dpa_rq_ring(struct app_context *app_ctx, struct worker_ctx *wk)
{
	flexio_uintptr_t wqe_data_daddr = wk->rq_transf.wqd_daddr;
	uint32_t mkey_id = wk->rq_transf.wqd_mkey_id;
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
				   wk->rq_transf.wq_ring_daddr))
		retval = -1;

	free(rx_wqes);
	return retval;
}

static int init_rq_dbr(struct app_context *app_ctx, struct worker_ctx *wk)
{
	__be32 dbr[2];

	dbr[0] = htobe32(Q_DEPTH & 0xffff);
	dbr[1] = htobe32(0);
	if (flexio_host2dev_memcpy(app_ctx->flexio_process, dbr, sizeof(dbr),
				   wk->rq_transf.wq_dbr_daddr))
		return -1;

	return 0;
}

/* ------------------------------------------------------------------ */
/* Per-worker event handler with STRICT EU affinity.                   */
/*                                                                     */
/* DOCA 3.4: affinity is passed in eh_attr.affinity {type, id}.        */
/*   FLEXIO_AFFINITY_STRICT + id  -> run ONLY on EU <id>.              */
/* This is what guarantees the N handlers land on N distinct EUs and   */
/* actually execute in parallel instead of contending for one EU.      */
/* Requires the EUs to exist in the device's EU partition              */
/* (dpaeumgmt partition -d <dev> query).                               */
/* ------------------------------------------------------------------ */
static int create_worker_event_handler(struct app_context *app_ctx,
				       struct worker_ctx *wk, int worker_id)
{
	struct flexio_event_handler_attr eh_attr = {0};
	char eh_name[32];

	snprintf(eh_name, sizeof(eh_name), "pp_eh_w%d", worker_id);

	eh_attr.host_stub_func = flexio_pp_dev;
	eh_attr.affinity.type = FLEXIO_AFFINITY_STRICT;
	eh_attr.affinity.id = app_ctx->eu_base + worker_id;

	if (flexio_event_handler_create(app_ctx->flexio_process, &eh_attr, &wk->pp_eh)) {
		fprintf(stderr,
			"Failed to create event handler for worker %d (EU %d).\n"
			"Hint: verify the EU partition has enough EUs:\n"
			"  dpaeumgmt partition -d <dev> query\n",
			worker_id, app_ctx->eu_base + worker_id);
		return -1;
	}

	return 0;
}

/* ------------------------------------------------------------------ */
/* Per-worker RQ + RQ-CQ (CQ bound to THIS worker's handler thread).   */
/* ------------------------------------------------------------------ */
static int create_worker_rq(struct app_context *app_ctx, struct worker_ctx *wk)
{
	struct flexio_process *app_fp = app_ctx->flexio_process;
	struct flexio_cq_attr rqcq_attr = {0};
	struct flexio_wq_attr rq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	if (cq_mem_alloc(app_fp, &wk->rq_cq_transf)) {
		fprintf(stderr, "Failed to alloc memory for RQ's CQ.\n");
		return -1;
	}

	rqcq_attr.log_cq_depth = LOG_Q_DEPTH;
	/* CQE on this CQ triggers THIS worker's DPA thread only. */
	rqcq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_DPA_THREAD;
	rqcq_attr.thread = flexio_event_handler_get_thread(wk->pp_eh);
	rqcq_attr.uar_id = uar_id;
	rqcq_attr.cq_dbr_daddr = wk->rq_cq_transf.cq_dbr_daddr;
	rqcq_attr.cq_ring_qmem.daddr = wk->rq_cq_transf.cq_ring_daddr;
	if (flexio_cq_create(app_fp, NULL, &rqcq_attr, &wk->rq_cq)) {
		fprintf(stderr, "Failed to create Flex IO RQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(wk->rq_cq);
	wk->rq_cq_transf.cq_num = cq_num;
	wk->rq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	if (rq_mem_alloc(app_fp, &wk->rq_transf)) {
		fprintf(stderr, "Failed to allocate memory for RQ.\n");
		return -1;
	}

	wk->rqd_mkey = create_dpa_mkey(app_ctx, wk->rq_transf.wqd_daddr);
	if (!wk->rqd_mkey) {
		fprintf(stderr, "Failed to create an MKey for RQ data buffer.\n");
		return -1;
	}
	wk->rq_transf.wqd_mkey_id = flexio_mkey_get_id(wk->rqd_mkey);

	if (init_dpa_rq_ring(app_ctx, wk)) {
		fprintf(stderr, "Failed to init RQ ring.\n");
		return -1;
	}

	rq_attr.log_wq_depth = LOG_Q_DEPTH;
	rq_attr.pd = app_ctx->process_pd;
	rq_attr.wq_dbr_qmem.memtype = FLEXIO_MEMTYPE_DPA;
	rq_attr.wq_dbr_qmem.daddr = wk->rq_transf.wq_dbr_daddr;
	rq_attr.wq_ring_qmem.daddr = wk->rq_transf.wq_ring_daddr;
	if (flexio_rq_create(app_fp, NULL, cq_num, &rq_attr, &wk->rq)) {
		fprintf(stderr, "Failed to create Flex IO RQ.\n");
		return -1;
	}

	wk->rq_transf.wq_num = flexio_rq_get_wq_num(wk->rq);
	if (init_rq_dbr(app_ctx, wk)) {
		fprintf(stderr, "Failed to init RQ DBR.\n");
		return -1;
	}

	return 0;
}

/* ------------------------------------------------------------------ */
/* Steering: single-SMAC 5-tuple RSS across all worker RQs.            */
/* RX: {SMAC, ip_proto} -> Toeplitz TIR -> RQT -> RQ[0..N-1].          */
/* TX: original single pair (all traffic carries SMAC_BASE).           */
/* ------------------------------------------------------------------ */
static int create_steering_rules(struct app_context *app_ctx, int nic_mode)
{
	uint32_t rqns[MAX_WORKERS];
	int w;

	/* Collect the FlexIO RQ numbers of all workers for the RQT. */
	for (w = 0; w < app_ctx->num_workers; w++)
		rqns[w] = app_ctx->wk[w].rq_transf.wq_num;

	/* One SMAC, hashed by 5-tuple across all worker RQs:
	 *   RX rule (SMAC_BASE, ip_proto) -> RSS TIR -> RQT -> RQ[0..N-1].
	 */
	app_ctx->rss = rss_steering_create(app_ctx->ibv_ctx, rqns,
					   app_ctx->num_workers, SMAC_BASE);
	if (!app_ctx->rss) {
		fprintf(stderr, "Failed to create RSS steering chain\n");
		return -1;
	}

	/* add to struct app_context: */
	struct flow_matcher *rx_matcher;
	struct flow_rule *rx_rule;

	/* in create_steering_rules(), after rss_steering_create(): */
	app_ctx->rx_matcher = create_matcher_rx(app_ctx->ibv_ctx);
	if (!app_ctx->rx_matcher) {
		fprintf(stderr, "Failed to create RX matcher\n");
		return -1;
	}

	app_ctx->rx_rule = create_rule_rx_mac_match(app_ctx->rx_matcher,
			rss_steering_get_tir(app_ctx->rss, 1 /* UDP hash */),
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

		/* All traffic carries the single base SMAC again, so the
		 * original one-pair TX path is sufficient.
		 */
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

/* ------------------------------------------------------------------ */
/* Copy each worker's queue info to its OWN DPA heap struct.           */
/* The device code is unchanged: every thread parses the struct at the */
/* daddr it received as its thread argument.                           */
/* ------------------------------------------------------------------ */
static int copy_worker_data_to_dpa(struct app_context *app_ctx, struct worker_ctx *wk)
{
	uint64_t struct_bsize = sizeof(struct host2dev_packet_processor_data);
	struct host2dev_packet_processor_data *h2d_data;
	int ret = 0;

	h2d_data = calloc(1, struct_bsize);
	if (!h2d_data) {
		fprintf(stderr, "Failed to allocate memory for h2d_data\n");
		return -1;
	}

	h2d_data->sq_cq_transf = wk->sq_cq_transf;
	h2d_data->sq_transf = wk->sq_transf;
	h2d_data->rq_cq_transf = wk->rq_cq_transf;
	h2d_data->rq_transf = wk->rq_transf;
	h2d_data->not_first_run = 0;

	if (flexio_copy_from_host(app_ctx->flexio_process, h2d_data, struct_bsize,
				  &wk->app_data_daddr)) {
		fprintf(stderr, "Failed to copy worker data to DPA.\n");
		ret = -1;
	}

	free(h2d_data);
	return ret;
}

/* ------------------------------------------------------------------ */
/* Per-worker cleanup (reverse order of creation).                     */
/* ------------------------------------------------------------------ */
static int clean_up_worker(struct app_context *app_ctx, struct worker_ctx *wk)
{
	struct flexio_process *fp = app_ctx->flexio_process;
	int err = 0;

	if (wk->app_data_daddr && flexio_buf_dev_free(fp, wk->app_data_daddr))
		err = -1;

	/* SQ side. */
	if (wk->sq && flexio_sq_destroy(wk->sq))
		err = -1;
	if (wk->sqd_mkey && flexio_device_mkey_destroy(wk->sqd_mkey))
		err = -1;
	if (wk->sq_transf.wq_ring_daddr && flexio_buf_dev_free(fp, wk->sq_transf.wq_ring_daddr))
		err = -1;
	if (wk->sq_transf.wqd_daddr && flexio_buf_dev_free(fp, wk->sq_transf.wqd_daddr))
		err = -1;
	if (wk->sq_cq && flexio_cq_destroy(wk->sq_cq))
		err = -1;
	if (wk->sq_cq_transf.cq_ring_daddr &&
	    flexio_buf_dev_free(fp, wk->sq_cq_transf.cq_ring_daddr))
		err = -1;
	if (wk->sq_cq_transf.cq_dbr_daddr &&
	    flexio_buf_dev_free(fp, wk->sq_cq_transf.cq_dbr_daddr))
		err = -1;

	/* RQ side. */
	if (wk->rq && flexio_rq_destroy(wk->rq))
		err = -1;
	if (wk->rqd_mkey && flexio_device_mkey_destroy(wk->rqd_mkey))
		err = -1;
	if (wk->rq_transf.wq_dbr_daddr && flexio_buf_dev_free(fp, wk->rq_transf.wq_dbr_daddr))
		err = -1;
	if (wk->rq_transf.wq_ring_daddr && flexio_buf_dev_free(fp, wk->rq_transf.wq_ring_daddr))
		err = -1;
	if (wk->rq_transf.wqd_daddr && flexio_buf_dev_free(fp, wk->rq_transf.wqd_daddr))
		err = -1;
	if (wk->rq_cq && flexio_cq_destroy(wk->rq_cq))
		err = -1;
	if (wk->rq_cq_transf.cq_ring_daddr &&
	    flexio_buf_dev_free(fp, wk->rq_cq_transf.cq_ring_daddr))
		err = -1;
	if (wk->rq_cq_transf.cq_dbr_daddr &&
	    flexio_buf_dev_free(fp, wk->rq_cq_transf.cq_dbr_daddr))
		err = -1;

	/* Handler last. */
	if (wk->pp_eh && flexio_event_handler_destroy(wk->pp_eh))
		err = -1;

	return err;
}

/* ------------------------------------------------------------------ */
/* Main.                                                               */
/* ------------------------------------------------------------------ */
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
	int w, i;

	app_ctx.num_workers = DEFAULT_WORKERS;
	app_ctx.eu_base = 0;

	printf("Flex IO packet processing sample - PARALLEL mode.\n");

	if (argc < 2) {
		fprintf(stderr,
			"Usage: %s <mlx5 device> [--nic-mode] [--workers N] [--eu-base E]\n",
			argv[0]);
		return -1;
	}

	for (i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "--nic-mode")) {
			nic_mode = 1;
		} else if (!strcmp(argv[i], "--workers") && i + 1 < argc) {
			app_ctx.num_workers = atoi(argv[++i]);
			if (app_ctx.num_workers < 1 ||
			    app_ctx.num_workers > MAX_WORKERS) {
				fprintf(stderr, "--workers must be 1-%d\n", MAX_WORKERS);
				return -1;
			}
		} else if (!strcmp(argv[i], "--eu-base") && i + 1 < argc) {
			app_ctx.eu_base = atoi(argv[++i]);
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
		printf("Use the token >>> %#lx <<< for debugging\n", udbg_token);

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

	/* ------------------------------------------------------------ *
	 * Build all workers: handler -> SQ -> RQ (order matters: the RQ *
	 * CQ needs the handler's thread at creation).                   *
	 * ------------------------------------------------------------ */
	for (w = 0; w < app_ctx.num_workers; w++) {
		struct worker_ctx *wk = &app_ctx.wk[w];

		if (create_worker_event_handler(&app_ctx, wk, w)) {
			err = -1;
			goto cleanup;
		}
		if (create_worker_sq(&app_ctx, wk)) {
			fprintf(stderr, "Failed to create SQ for worker %d.\n", w);
			err = -1;
			goto cleanup;
		}
		if (create_worker_rq(&app_ctx, wk)) {
			fprintf(stderr, "Failed to create RQ for worker %d.\n", w);
			err = -1;
			goto cleanup;
		}
		if (copy_worker_data_to_dpa(&app_ctx, wk)) {
			fprintf(stderr, "Failed to copy data for worker %d.\n", w);
			err = -1;
			goto cleanup;
		}
	}

	/* Steering after all RQs exist (rules point at per-worker TIRs). */
	if (create_steering_rules(&app_ctx, nic_mode)) {
		fprintf(stderr, "Failed to create Flex IO steering rules.\n");
		err = -1;
		goto cleanup;
	}

	/* ------------------------------------------------------------ *
	 * Launch: move every handler to running state. Each gets ITS    *
	 * OWN data daddr as the thread argument -> no shared state.     *
	 * ------------------------------------------------------------ */
	for (w = 0; w < app_ctx.num_workers; w++) {
		struct worker_ctx *wk = &app_ctx.wk[w];

		if (flexio_event_handler_run(wk->pp_eh, wk->app_data_daddr)) {
			fprintf(stderr, "Failed to run event handler %d.\n", w);
			err = -1;
			goto cleanup;
		}
	}

	printf("Ready: %d workers running on EUs %d-%d. Press Enter to exit.\n",
	       app_ctx.num_workers, app_ctx.eu_base,
	       app_ctx.eu_base + app_ctx.num_workers - 1);

	if (!fread(buf, 1, 1, stdin))
		fprintf(stderr, "Failed in fread\n");

cleanup:
	/* RX rule references the RSS TIR -> destroy it first. */
	if (app_ctx.rx_rule && destroy_rule(app_ctx.rx_rule))
		err = -1;
	if (app_ctx.rx_matcher && destroy_matcher(app_ctx.rx_matcher))
		err = -1;

	/* RSS chain next: TIRs / RQT hold references to the RQs. */
	if (app_ctx.rss)
		rss_steering_destroy(app_ctx.rss);

	/* Per-worker resources (reverse creation order inside). */
	for (w = app_ctx.num_workers - 1; w >= 0; w--) {
		if (clean_up_worker(&app_ctx, &app_ctx.wk[w]))
			err = -1;
	}

	/* Shared matchers / TX rules. */
	if (app_ctx.tx_rule_vport && destroy_rule(app_ctx.tx_rule_vport))
		err = -1;
	if (app_ctx.tx_rule_table && destroy_rule(app_ctx.tx_rule_table))
		err = -1;
	if (app_ctx.tx_matcher && destroy_matcher(app_ctx.tx_matcher))
		err = -1;

	if (app_fp && app_ctx.stream && flexio_msg_stream_destroy(app_ctx.stream))
		err = -1;

	if (app_fp && flexio_process_destroy(app_fp))
		err = -1;

	if (app_ctx.ibv_ctx && ibv_close_device(app_ctx.ibv_ctx))
		err = -1;

	return err;
}
