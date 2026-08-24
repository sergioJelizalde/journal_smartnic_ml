/*
 * SCALED FLEXIO REFLECTOR - DEVICE SIDE (DPA)
 * DOCA 3.4 / BlexIO 26.4
 * 
 * KEY: All threads poll the SAME shared RQ CQ (not handler-bound).
 * This allows 190+ threads to work concurrently.
 * 
 * MAC SWAP LOGIC:
 * 1. Read packet from RQ
 * 2. Swap src/dst MACs (critical: avoids loop if packet re-matches RX rule)
 * 3. Send packet via SQ
 */

#include "com_dev.h"
#include <libflexio-dev/flexio_dev_debug.h>
#include <libflexio-dev/flexio_dev_err.h>
#include <libflexio-dev/flexio_dev_queue_access.h>
#include <libflexio-libc/string.h>
#include <stddef.h>
#include <dpaintrin.h>
#include "../flexio_packet_processor_com.h"

#define CQ_IDX_MASK ((1 << 8) - 1)      /* LOG_CQ_DEPTH = 8 */
#define RQ_IDX_MASK ((1 << 8) - 1)
#define SQ_IDX_MASK ((1 << (8 + 2)) - 1)  /* 3 segments per WQE */
#define DATA_IDX_MASK ((1 << 8) - 1)

/* Shared context (all threads use same instance) */
static struct {
	uint64_t packets_count;
	uint32_t lkey;

	cq_ctx_t rq_cq_ctx;      /* Shared RQ CQ */
	rq_ctx_t rq_ctx;         /* Shared RQ */
	sq_ctx_t sq_ctx;         /* Shared SQ */
	cq_ctx_t sq_cq_ctx;      /* Shared SQ CQ */
	dt_ctx_t dt_ctx;         /* SQ Data ring */

	/* Thread-local counters (per-thread, use TLS) */
	uint64_t thread_packets[256];
} app_ctx;

/* Initialize shared context from host data */
static void app_ctx_init(struct host2dev_packet_processor_data *data_from_host)
{
	app_ctx.packets_count = 0;
	app_ctx.lkey = data_from_host->sq_transf.wqd_mkey_id;

	com_cq_ctx_init(&app_ctx.rq_cq_ctx,
			data_from_host->rq_cq_transf.cq_num,
			data_from_host->rq_cq_transf.log_cq_depth,
			data_from_host->rq_cq_transf.cq_ring_daddr,
			data_from_host->rq_cq_transf.cq_dbr_daddr);

	com_rq_ctx_init(&app_ctx.rq_ctx,
			data_from_host->rq_transf.wq_num,
			data_from_host->rq_transf.wq_ring_daddr,
			data_from_host->rq_transf.wq_dbr_daddr);

	com_sq_ctx_init(&app_ctx.sq_ctx,
			data_from_host->sq_transf.wq_num,
			data_from_host->sq_transf.wq_ring_daddr);

	com_cq_ctx_init(&app_ctx.sq_cq_ctx,
			data_from_host->sq_cq_transf.cq_num,
			data_from_host->sq_cq_transf.log_cq_depth,
			data_from_host->sq_cq_transf.cq_ring_daddr,
			data_from_host->sq_cq_transf.cq_dbr_daddr);

	com_dt_ctx_init(&app_ctx.dt_ctx, data_from_host->sq_transf.wqd_daddr);
}

/* 
 * MAC SWAP HELPER
 * Ethernet header: 6B dst MAC + 6B src MAC + 2B EtherType
 */
static inline void swap_macs_inplace(void *packet_data)
{
	uint8_t *p = (uint8_t *)packet_data;
	uint8_t tmp[6];

	/* dst MAC at offset 0, src MAC at offset 6 */
	memcpy(tmp, p, 6);           /* tmp = dst */
	memcpy(p, p + 6, 6);         /* dst = src */
	memcpy(p + 6, tmp, 6);       /* src = tmp (old dst) */
}

/*
 * PROCESS PACKET
 * All threads execute this. CQ is not handler-bound, so multiple threads
 * will contend on polling. Hardware CQ ownership bits handle thread safety.
 */
static void process_packet(void)
{
	struct flexio_dev_wqe_rcv_data_seg *rwqe;
	uint32_t rq_wqe_idx;
	char *rq_data;

	union flexio_dev_sqe_seg *swqe;
	char *sq_data;
	uint32_t data_sz;

	/* Extract CQE info */
	rq_wqe_idx = flexio_dev_cqe_get_wqe_counter(app_ctx.rq_cq_ctx.cqe);
	data_sz = flexio_dev_cqe_get_byte_cnt(app_ctx.rq_cq_ctx.cqe);

	/* Get RQ WQE and packet data */
	rwqe = &app_ctx.rq_ctx.rq_ring[rq_wqe_idx & RQ_IDX_MASK];
	rq_data = flexio_dev_rwqe_get_addr(rwqe);

	/* Get SQ data buffer */
	sq_data = get_next_dte(&app_ctx.dt_ctx, DATA_IDX_MASK, 11);

	/* Copy packet */
	memcpy(sq_data, rq_data, data_sz);

	/* CRITICAL: Swap MACs to avoid loop.
	 * If you send the packet back without swapping and it matches the RX rule,
	 * the same packet re-enters the RQ, creating an infinite loop.
	 */
	swap_macs_inplace(sq_data);

	/* Build SQ WQE (4 segments: CTRL, ETH, DATA, padding) */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_ctrl_set(swqe, app_ctx.sq_ctx.sq_pi, app_ctx.sq_ctx.sq_number,
				     FLEXIO_CTRL_SEG_CE_CQE_ON_CQE_ERROR,
				     FLEXIO_CTRL_SEG_TYPE_SEND_EN);

	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_eth_set(swqe, 0, 0, 0, NULL);

	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, data_sz, app_ctx.lkey, (uint64_t)sq_data);

	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);

	/* Ring doorbell */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_qp_sq_ring_db(++app_ctx.sq_ctx.sq_pi, app_ctx.sq_ctx.sq_number);
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);

	/* Increment RQ PI */
	flexio_dev_dbr_rq_inc_pi(app_ctx.rq_ctx.rq_dbr);

	app_ctx.packets_count++;
}

/*
 * ENTRY POINT
 * All threads call this with the same thread_arg (shared data daddr).
 * 
 * KEY DIFFERENCE from event-handler version:
 * - The CQ is NOT handler-bound (element_type = NON_DPA_CQ)
 * - Multiple threads poll the same CQ independently
 * - Hardware CQ ownership bits (owned by HW vs SW) prevent double-processing
 */
flexio_dev_event_handler_t flexio_pp_dev;
__dpa_global__ void flexio_pp_dev(uint64_t thread_arg)
{
	struct host2dev_packet_processor_data *data_from_host = (void *)thread_arg;
	static int init_done = 0;        /* Simple init guard (static in TLS) */
	uint32_t idle_spins = 0;

	/* Initialize once (all threads see this, but atomic semantics via static) */
	if (!__atomic_test_and_set(&data_from_host->not_first_run, __ATOMIC_SEQ_CST) == 0) {
		app_ctx_init(data_from_host);
	}

	/*
	 * Poll loop: check CQ ownership bit.
	 * If owned by HW, a new CQE arrived. Process it and step the CQ.
	 * If owned by SW, CQ is empty. Exit loop, arm, reschedule.
	 */
	while (flexio_dev_cqe_get_owner(app_ctx.rq_cq_ctx.cqe) !=
	       app_ctx.rq_cq_ctx.cq_hw_owner_bit) {
		/* CQE is owned by SW (we own it). Process packet. */
		__dpa_thread_fence(__DPA_MEMORY, __DPA_R, __DPA_R);
		process_packet();
		com_step_cq(&app_ctx.rq_cq_ctx);
		idle_spins = 0;
	}

	/* CQ is empty. Bail out and let this thread be rescheduled. */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_cq_arm(app_ctx.rq_cq_ctx.cq_idx, app_ctx.rq_cq_ctx.cq_number);

	/* Reschedule: this thread yields control back to the DPA scheduler.
	 * The NIC will re-fire this handler thread when a new packet arrives.
	 */
	flexio_dev_thread_reschedule();
}