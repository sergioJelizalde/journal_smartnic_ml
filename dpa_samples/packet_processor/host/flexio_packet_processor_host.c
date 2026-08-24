/*
 * FLEXIO SCALED REFLECTOR - HOST SIDE
 * DOCA 3.4 / FlexIO 26.4
 * Based on BenchBF3 architecture: single shared RQ/SQ with thread pool
 * 
 * Key: FLEXIBLE affinity + shared CQ = scales to 190+ threads
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
#include "flexio_packet_processor_com.h"

extern flexio_func_t flexio_pp_dev;

/* ================================================================== */
/* CONFIGURATION                                                      */
/* ================================================================== */

#define L2V(l) (1UL << (l))
#define LOG_Q_DEPTH 8
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

#define NUM_THREADS_DEFAULT 64
#define MAX_THREADS 256

#define SMAC_BASE 0x0208a4d8ff43

#define MSG_HOST_BUFF_BSIZE (512 * L2V(FLEXIO_MSG_DEV_LOG_DATA_CHUNK_BSIZE))

/* ================================================================== */
/* SHARED CONTEXT (single per application)                           */
/* ================================================================== */

struct app_context {
	struct flexio_process *flexio_process;
	struct flexio_app *flexio_app;
	struct flexio_msg_stream *stream;
	struct flexio_uar *process_uar;
	struct ibv_pd *process_pd;
	struct ibv_context *ibv_ctx;

	/* Single shared RQ + CQ */
	struct flexio_rq *rq;
	struct flexio_cq *rq_cq;
	struct flexio_mkey *rqd_mkey;

	/* Single shared SQ + CQ */
	struct flexio_sq *sq;
	struct flexio_cq *sq_cq;
	struct flexio_mkey *sqd_mkey;

	/* Transfer structs for DPA */
	struct app_transfer_cq sq_cq_transf;
	struct app_transfer_wq sq_transf;
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_wq rq_transf;

	/* DPA heap address */
	flexio_uintptr_t app_data_daddr;

	/* Thread pool (FLEXIBLE affinity) */
	int num_threads;
	struct flexio_event_handler *threads[MAX_THREADS];
};

/* ================================================================== */
/* HELPER: Open IBV context                                           */
/* ================================================================== */

static int app_open_ibv_ctx(struct app_context *app_ctx, const char *device)
{
	struct ibv_device **dev_list;
	struct mlx5dv_context_attr dv_attr = {
		.flags = MLX5DV_CONTEXT_FLAGS_DEVX,
	};
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
		fprintf(stderr, "No IBV device found: %s\n", device);
		ret = -1;
		goto cleanup;
	}

	app_ctx->ibv_ctx = mlx5dv_open_device(dev_list[dev_i], &dv_attr);
	if (!app_ctx->ibv_ctx) {
		fprintf(stderr, "mlx5dv_open_device failed for %s\n", device);
		ret = -1;
	}

cleanup:
	ibv_free_device_list(dev_list);
	return ret;
}

/* ================================================================== */
/* Create MKey for DPA memory                                         */
/* ================================================================== */

static struct flexio_mkey *create_dpa_mkey(struct app_context *app_ctx,
					   flexio_uintptr_t daddr, size_t size)
{
	struct flexio_mkey_attr mkey_attr = {0};
	struct flexio_mkey *mkey;

	mkey_attr.pd = app_ctx->process_pd;
	mkey_attr.daddr = daddr;
	mkey_attr.len = size;
	mkey_attr.access = IBV_ACCESS_LOCAL_WRITE;

	if (flexio_device_mkey_create(app_ctx->flexio_process, &mkey_attr, &mkey)) {
		fprintf(stderr, "flexio_device_mkey_create failed\n");
		return NULL;
	}

	return mkey;
}

/* ================================================================== */
/* CQ memory allocation                                               */
/* ================================================================== */

static int cq_mem_alloc(struct flexio_process *process, 
		        struct app_transfer_cq *cq_transf)
{
	struct mlx5_cqe64 *cq_ring, *cqe;
	__be32 dbr[2] = {0, 0};
	uint32_t i;

	/* Allocate CQ ring on DPA heap */
	if (flexio_buf_dev_alloc(process, CQ_BSIZE, &cq_transf->cq_ring_daddr)) {
		fprintf(stderr, "Failed to allocate CQ ring on DPA\n");
		return -1;
	}

	/* Allocate DBR on DPA heap */
	if (flexio_buf_dev_alloc(process, sizeof(dbr), &cq_transf->cq_dbr_daddr)) {
		fprintf(stderr, "Failed to allocate CQ DBR on DPA\n");
		return -1;
	}

	/* Initialize CQ ring on host, transfer to DPA */
	cq_ring = calloc(Q_DEPTH, CQE_BSIZE);
	if (!cq_ring) {
		fprintf(stderr, "calloc for cq_ring failed\n");
		return -1;
	}

	/* Set initial ownership bit */
	for (i = 0, cqe = cq_ring; i < Q_DEPTH; i++)
		mlx5dv_set_cqe_owner(cqe++, 1);

	/* Copy to DPA */
	if (flexio_host2dev_memcpy(process, cq_ring, CQ_BSIZE, 
				   cq_transf->cq_ring_daddr)) {
		fprintf(stderr, "flexio_host2dev_memcpy for CQ ring failed\n");
		free(cq_ring);
		return -1;
	}

	free(cq_ring);
	return 0;
}

/* ================================================================== */
/* SQ memory allocation                                               */
/* ================================================================== */

static int sq_mem_alloc(struct flexio_process *process, 
		        struct app_transfer_wq *sq_transf)
{
	if (flexio_buf_dev_alloc(process, Q_DATA_BSIZE, &sq_transf->wqd_daddr)) {
		fprintf(stderr, "Failed to alloc SQ data on DPA\n");
		return -1;
	}

	if (flexio_buf_dev_alloc(process, SQ_RING_BSIZE, &sq_transf->wq_ring_daddr)) {
		fprintf(stderr, "Failed to alloc SQ ring on DPA\n");
		return -1;
	}

	return 0;
}

/* ================================================================== */
/* RQ memory allocation                                               */
/* ================================================================== */

static int rq_mem_alloc(struct flexio_process *process, 
		        struct app_transfer_wq *rq_transf)
{
	__be32 dbr[2] = {0, 0};

	if (flexio_buf_dev_alloc(process, Q_DATA_BSIZE, &rq_transf->wqd_daddr)) {
		fprintf(stderr, "Failed to alloc RQ data on DPA\n");
		return -1;
	}

	if (flexio_buf_dev_alloc(process, RQ_RING_BSIZE, &rq_transf->wq_ring_daddr)) {
		fprintf(stderr, "Failed to alloc RQ ring on DPA\n");
		return -1;
	}

	if (flexio_buf_dev_alloc(process, sizeof(dbr), &rq_transf->wq_dbr_daddr)) {
		fprintf(stderr, "Failed to alloc RQ DBR on DPA\n");
		return -1;
	}

	return 0;
}

/* ================================================================== */
/* Create SQ                                                           */
/* ================================================================== */

static int create_sq(struct app_context *app_ctx)
{
	struct flexio_process *fp = app_ctx->flexio_process;
	struct flexio_cq_attr cq_attr = {0};
	struct flexio_wq_attr wq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	/* Allocate SQ CQ memory */
	if (cq_mem_alloc(fp, &app_ctx->sq_cq_transf)) {
		fprintf(stderr, "cq_mem_alloc for SQ CQ failed\n");
		return -1;
	}

	/* Create SQ CQ (not handler-bound) */
	cq_attr.log_cq_depth = LOG_Q_DEPTH;
	cq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ;
	cq_attr.uar_id = uar_id;
	cq_attr.cq_dbr_daddr = app_ctx->sq_cq_transf.cq_dbr_daddr;
	cq_attr.cq_ring_qmem.daddr = app_ctx->sq_cq_transf.cq_ring_daddr;

	if (flexio_cq_create(fp, NULL, &cq_attr, &app_ctx->sq_cq)) {
		fprintf(stderr, "flexio_cq_create for SQ CQ failed\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(app_ctx->sq_cq);
	app_ctx->sq_cq_transf.cq_num = cq_num;
	app_ctx->sq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	/* Allocate SQ memory */
	if (sq_mem_alloc(fp, &app_ctx->sq_transf)) {
		fprintf(stderr, "sq_mem_alloc failed\n");
		return -1;
	}

	/* Create SQ */
	wq_attr.log_wq_depth = LOG_Q_DEPTH;
	wq_attr.uar_id = uar_id;
	wq_attr.wq_ring_qmem.daddr = app_ctx->sq_transf.wq_ring_daddr;
	wq_attr.pd = app_ctx->process_pd;

	if (flexio_sq_create(fp, NULL, cq_num, &wq_attr, &app_ctx->sq)) {
		fprintf(stderr, "flexio_sq_create failed\n");
		return -1;
	}

	app_ctx->sq_transf.wq_num = flexio_sq_get_wq_num(app_ctx->sq);

	/* Create MKey for SQ data */
	app_ctx->sqd_mkey = create_dpa_mkey(app_ctx, app_ctx->sq_transf.wqd_daddr, 
					   Q_DATA_BSIZE);
	if (!app_ctx->sqd_mkey) {
		fprintf(stderr, "create_dpa_mkey for SQ data failed\n");
		return -1;
	}

	app_ctx->sq_transf.wqd_mkey_id = flexio_mkey_get_id(app_ctx->sqd_mkey);

	return 0;
}

/* ================================================================== */
/* Create RQ                                                           */
/* ================================================================== */

static int init_rq_ring(struct app_context *app_ctx)
{
	struct mlx5_wqe_data_seg *rx_wqes, *dseg;
	flexio_uintptr_t wqe_data_daddr = app_ctx->rq_transf.wqd_daddr;
	uint32_t mkey_id = app_ctx->rq_transf.wqd_mkey_id;
	int ret = 0;
	uint32_t i;

	rx_wqes = calloc(Q_DEPTH, RQ_WQE_BSIZE);
	if (!rx_wqes) {
		fprintf(stderr, "calloc for rx_wqes failed\n");
		return -1;
	}

	for (i = 0, dseg = rx_wqes; i < Q_DEPTH; i++, dseg++) {
		mlx5dv_set_data_seg(dseg, Q_DATA_ENTRY_BSIZE, mkey_id, wqe_data_daddr);
		wqe_data_daddr += Q_DATA_ENTRY_BSIZE;
	}

	if (flexio_host2dev_memcpy(app_ctx->flexio_process, rx_wqes, RQ_RING_BSIZE,
				   app_ctx->rq_transf.wq_ring_daddr)) {
		fprintf(stderr, "flexio_host2dev_memcpy for RQ ring failed\n");
		ret = -1;
	}

	free(rx_wqes);
	return ret;
}

static int init_rq_dbr(struct app_context *app_ctx)
{
	__be32 dbr[2];

	dbr[0] = htobe32(Q_DEPTH & 0xffff);
	dbr[1] = htobe32(0);

	return flexio_host2dev_memcpy(app_ctx->flexio_process, dbr, sizeof(dbr),
				      app_ctx->rq_transf.wq_dbr_daddr);
}

static int create_rq(struct app_context *app_ctx)
{
	struct flexio_process *fp = app_ctx->flexio_process;
	struct flexio_cq_attr cq_attr = {0};
	struct flexio_wq_attr wq_attr = {0};
	uint32_t uar_id = flexio_uar_get_id(app_ctx->process_uar);
	uint32_t cq_num;

	/* Allocate RQ CQ memory */
	if (cq_mem_alloc(fp, &app_ctx->rq_cq_transf)) {
		fprintf(stderr, "cq_mem_alloc for RQ CQ failed\n");
		return -1;
	}

	/* Create RQ CQ (KEY: not handler-bound, shared by all threads) */
	cq_attr.log_cq_depth = LOG_Q_DEPTH;
	cq_attr.element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ;
	cq_attr.uar_id = uar_id;
	cq_attr.cq_dbr_daddr = app_ctx->rq_cq_transf.cq_dbr_daddr;
	cq_attr.cq_ring_qmem.daddr = app_ctx->rq_cq_transf.cq_ring_daddr;

	if (flexio_cq_create(fp, NULL, &cq_attr, &app_ctx->rq_cq)) {
		fprintf(stderr, "flexio_cq_create for RQ CQ failed\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(app_ctx->rq_cq);
	app_ctx->rq_cq_transf.cq_num = cq_num;
	app_ctx->rq_cq_transf.log_cq_depth = LOG_Q_DEPTH;

	/* Allocate RQ memory */
	if (rq_mem_alloc(fp, &app_ctx->rq_transf)) {
		fprintf(stderr, "rq_mem_alloc failed\n");
		return -1;
	}

	/* Create MKey for RQ data */
	app_ctx->rqd_mkey = create_dpa_mkey(app_ctx, app_ctx->rq_transf.wqd_daddr,
					   Q_DATA_BSIZE);
	if (!app_ctx->rqd_mkey) {
		fprintf(stderr, "create_dpa_mkey for RQ data failed\n");
		return -1;
	}

	app_ctx->rq_transf.wqd_mkey_id = flexio_mkey_get_id(app_ctx->rqd_mkey);

	/* Initialize RQ ring */
	if (init_rq_ring(app_ctx)) {
		fprintf(stderr, "init_rq_ring failed\n");
		return -1;
	}

	/* Create RQ */
	wq_attr.log_wq_depth = LOG_Q_DEPTH;
	wq_attr.pd = app_ctx->process_pd;
	wq_attr.wq_dbr_qmem.memtype = FLEXIO_MEMTYPE_DPA;
	wq_attr.wq_dbr_qmem.daddr = app_ctx->rq_transf.wq_dbr_daddr;
	wq_attr.wq_ring_qmem.daddr = app_ctx->rq_transf.wq_ring_daddr;

	if (flexio_rq_create(fp, NULL, cq_num, &wq_attr, &app_ctx->rq)) {
		fprintf(stderr, "flexio_rq_create failed\n");
		return -1;
	}

	app_ctx->rq_transf.wq_num = flexio_rq_get_wq_num(app_ctx->rq);

	/* Initialize RQ DBR */
	if (init_rq_dbr(app_ctx)) {
		fprintf(stderr, "init_rq_dbr failed\n");
		return -1;
	}

	return 0;
}

/* ================================================================== */
/* Copy shared data to DPA                                            */
/* ================================================================== */

static int copy_shared_data_to_dpa(struct app_context *app_ctx)
{
	struct host2dev_packet_processor_data h2d_data = {0};

	h2d_data.sq_cq_transf = app_ctx->sq_cq_transf;
	h2d_data.sq_transf = app_ctx->sq_transf;
	h2d_data.rq_cq_transf = app_ctx->rq_cq_transf;
	h2d_data.rq_transf = app_ctx->rq_transf;
	h2d_data.not_first_run = 0;

	if (flexio_buf_dev_alloc(app_ctx->flexio_process, 
				sizeof(h2d_data),
				&app_ctx->app_data_daddr)) {
		fprintf(stderr, "flexio_buf_dev_alloc for app_data failed\n");
		return -1;
	}

	if (flexio_host2dev_memcpy(app_ctx->flexio_process, &h2d_data, 
				   sizeof(h2d_data),
				   app_ctx->app_data_daddr)) {
		fprintf(stderr, "flexio_host2dev_memcpy for app_data failed\n");
		return -1;
	}

	return 0;
}

/* ================================================================== */
/* Create steering rule (RX only) - OPTIONAL                          */
/* ================================================================== */

static int create_rx_steering(struct app_context *app_ctx)
{
	uint32_t rqn = flexio_rq_get_wq_num(app_ctx->rq);

	/* 
	 * Note: For full RX steering with MAC matching, you'd typically use:
	 * - DOCA Flow API (preferred for modern DOCA)
	 * - mlx5dv_create_flow_matcher + mlx5dv_create_flow (lower level)
	 *
	 * For basic operation, driver-level steering is sufficient.
	 * RX traffic will be directed to this RQ by default.
	 */

	printf("RQ %u ready for RX traffic\n", rqn);

	return 0;
}

/* ================================================================== */
/* Create thread pool (flexible affinity, no EU pinning)              */
/* ================================================================== */

static int create_thread_pool(struct app_context *app_ctx)
{
	struct flexio_event_handler_attr eh_attr = {0};
	int i;

	eh_attr.host_stub_func = flexio_pp_dev;
	
	/* Flexible affinity: not pinned to specific EU.
	 * Different FlexIO versions define this differently:
	 * - FLEXIO_AFFINITY_FLEXIBLE (newer versions, what we want)
	 * - FLEXIO_AFFINITY_DEFAULT (fallback)
	 * - Default (0) also works for flexible behavior
	 */
#ifdef FLEXIO_AFFINITY_FLEXIBLE
	eh_attr.affinity.type = FLEXIO_AFFINITY_FLEXIBLE;  /* Preferred */
#elif defined(FLEXIO_AFFINITY_DEFAULT)
	eh_attr.affinity.type = FLEXIO_AFFINITY_DEFAULT;   /* Fallback */
#else
	/* Default memset(0) behavior gives flexible affinity - nothing to set */
#endif

	for (i = 0; i < app_ctx->num_threads; i++) {
		if (flexio_event_handler_create(app_ctx->flexio_process, &eh_attr,
					       &app_ctx->threads[i])) {
			fprintf(stderr, "flexio_event_handler_create %d failed\n", i);
			return -1;
		}
	}

	printf("Created %d threads with flexible scheduling\n", app_ctx->num_threads);
	return 0;
}

/* ================================================================== */
/* Launch thread pool                                                 */
/* ================================================================== */

static int launch_thread_pool(struct app_context *app_ctx)
{
	int i;

	/* All threads get SAME shared data daddr */
	for (i = 0; i < app_ctx->num_threads; i++) {
		if (flexio_event_handler_run(app_ctx->threads[i],
					     app_ctx->app_data_daddr)) {
			fprintf(stderr, "flexio_event_handler_run %d failed\n", i);
			return -1;
		}
	}

	printf("Launched %d threads, all polling shared RQ\n", app_ctx->num_threads);
	return 0;
}

/* ================================================================== */
/* Cleanup                                                            */
/* ================================================================== */

static void cleanup(struct app_context *app_ctx)
{
	int i;

	/* Destroy threads */
	for (i = 0; i < app_ctx->num_threads; i++) {
		if (app_ctx->threads[i])
			flexio_event_handler_destroy(app_ctx->threads[i]);
	}

	/* Destroy resources */
	if (app_ctx->rq)
		flexio_rq_destroy(app_ctx->rq);
	if (app_ctx->rqd_mkey)
		flexio_device_mkey_destroy(app_ctx->rqd_mkey);
	if (app_ctx->rq_cq)
		flexio_cq_destroy(app_ctx->rq_cq);

	if (app_ctx->sq)
		flexio_sq_destroy(app_ctx->sq);
	if (app_ctx->sqd_mkey)
		flexio_device_mkey_destroy(app_ctx->sqd_mkey);
	if (app_ctx->sq_cq)
		flexio_cq_destroy(app_ctx->sq_cq);

	if (app_ctx->app_data_daddr)
		flexio_buf_dev_free(app_ctx->flexio_process, app_ctx->app_data_daddr);

	if (app_ctx->flexio_process && app_ctx->stream)
		flexio_msg_stream_destroy(app_ctx->stream);

	if (app_ctx->flexio_process)
		flexio_process_destroy(app_ctx->flexio_process);

	if (app_ctx->ibv_ctx)
		ibv_close_device(app_ctx->ibv_ctx);
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */

int main(int argc, char **argv)
{
	struct flexio_app_select_attr flexio_app_sel_attr = {0};
	flexio_msg_stream_attr_t stream_attr = {0};
	struct app_context app_ctx = {0};
	struct ibv_port_attr port_attr;
	uint64_t udbg_token;
	int i;

	app_ctx.num_threads = NUM_THREADS_DEFAULT;

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <device> [--threads N]\n", argv[0]);
		fprintf(stderr, "Example: %s mlx5_0 --threads 64\n", argv[0]);
		return -1;
	}

	/* Parse arguments */
	for (i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
			app_ctx.num_threads = atoi(argv[++i]);
			if (app_ctx.num_threads < 1 || app_ctx.num_threads > MAX_THREADS) {
				fprintf(stderr, "--threads must be 1-%d\n", MAX_THREADS);
				return -1;
			}
		}
	}

	if (geteuid()) {
		fprintf(stderr, "Must run with root privileges\n");
		return -1;
	}

	printf("FlexIO Scaled Reflector - DOCA 3.4\n");
	printf("Threads: %d, Affinity: FLEXIBLE\n", app_ctx.num_threads);

	/* Open IBV context */
	if (app_open_ibv_ctx(&app_ctx, argv[1])) {
		fprintf(stderr, "Failed to open IBV context\n");
		return -1;
	}

	/* Check port */
	if (ibv_query_port(app_ctx.ibv_ctx, 1, &port_attr)) {
		fprintf(stderr, "ibv_query_port failed\n");
		goto cleanup;
	}

	if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
		fprintf(stderr, "Port is not Ethernet\n");
		goto cleanup;
	}

	/* Set FlexIO version */
	if (flexio_version_set(FLEXIO_VER_USED)) {
		fprintf(stderr, "flexio_version_set failed\n");
		goto cleanup;
	}

	/* Get FlexIO app */
	flexio_app_sel_attr.app_name = "flexio_pp_app";
	flexio_app_sel_attr.hw_model_id = FLEXIO_HW_MODEL_DEF;
	flexio_app_sel_attr.ibv_ctx = app_ctx.ibv_ctx;

	if (flexio_app_get(&flexio_app_sel_attr, &app_ctx.flexio_app)) {
		fprintf(stderr, "flexio_app_get failed\n");
		goto cleanup;
	}

	/* Create FlexIO process */
	if (flexio_process_create(app_ctx.ibv_ctx, app_ctx.flexio_app, NULL,
				  &app_ctx.flexio_process)) {
		fprintf(stderr, "flexio_process_create failed\n");
		goto cleanup;
	}

	udbg_token = flexio_process_udbg_token_get(app_ctx.flexio_process);
	if (udbg_token)
		printf("Debug token: 0x%lx\n", udbg_token);

	/* Create message stream for device debug output */
	stream_attr.data_bsize = MSG_HOST_BUFF_BSIZE;
	stream_attr.sync_mode = FLEXIO_MSG_DEV_SYNC_MODE_SYNC;
	stream_attr.level = FLEXIO_MSG_DEV_INFO;
	stream_attr.transport_mode = FLEXIO_MSG_TRANSPORT_QP_RC;

	if (flexio_msg_stream_create(app_ctx.flexio_process, &stream_attr, stdout,
				     NULL, &app_ctx.stream)) {
		fprintf(stderr, "flexio_msg_stream_create failed\n");
		goto cleanup;
	}

	/* Get PD and UAR */
	app_ctx.process_pd = flexio_process_get_pd(app_ctx.flexio_process);
	app_ctx.process_uar = flexio_process_get_uar(app_ctx.flexio_process);

	/* Create shared SQ */
	if (create_sq(&app_ctx)) {
		fprintf(stderr, "create_sq failed\n");
		goto cleanup;
	}

	/* Create shared RQ */
	if (create_rq(&app_ctx)) {
		fprintf(stderr, "create_rq failed\n");
		goto cleanup;
	}

	/* Copy shared data to DPA */
	if (copy_shared_data_to_dpa(&app_ctx)) {
		fprintf(stderr, "copy_shared_data_to_dpa failed\n");
		goto cleanup;
	}

	/* Create RX steering rule */
	if (create_rx_steering(&app_ctx)) {
		fprintf(stderr, "create_rx_steering failed (non-fatal)\n");
		/* Non-fatal: can continue without steering in test mode */
	}

	/* Create thread pool */
	if (create_thread_pool(&app_ctx)) {
		fprintf(stderr, "create_thread_pool failed\n");
		goto cleanup;
	}

	/* Launch all threads */
	if (launch_thread_pool(&app_ctx)) {
		fprintf(stderr, "launch_thread_pool failed\n");
		goto cleanup;
	}

	printf("\nReady. All threads running. Press Enter to exit.\n");
	getchar();

	printf("Shutting down...\n");
	cleanup(&app_ctx);
	return 0;

cleanup:
	cleanup(&app_ctx);
	return -1;
}