/*
 * FLEXIO SCALED REFLECTOR - DEVICE SIDE (DPA)
 * DOCA 3.4 / FlexIO 26.4
 * Based on BenchBF3: shared RQ/SQ with thread pool polling same CQ
 * 
 * Key: CQE ownership bits handle thread safety for shared CQ polling
 * MAC swap prevents packet loop on reflector
 */

#include <libflexio/flexio_ver.h>
#define FLEXIO_DEV_VER_USED FLEXIO_VER(26, 4, 0)

#include <libflexio-dev/flexio_dev_debug.h>
#include <libflexio-dev/flexio_dev_err.h>
#include <libflexio-dev/flexio_dev_queue_access.h>
#include <libflexio-libc/string.h>
#include <stddef.h>
#include <dpaintrin.h>
#include "flexio_packet_processor_com.h"

/* ================================================================== */
/* CONFIGURATION                                                      */
/* ================================================================== */

#define CQ_IDX_MASK ((1 << 8) - 1)        /* LOG_Q_DEPTH=8 */
#define RQ_IDX_MASK ((1 << 8) - 1)
#define SQ_IDX_MASK ((1 << (8 + 2)) - 1)  /* 3 WQE segments per queue depth */
#define DATA_IDX_MASK ((1 << 8) - 1)

/* ================================================================== */
/* SHARED CONTEXT (all threads use this)                             */
/* ================================================================== */

static struct {
	/* Counters */
	uint64_t packets_count;
	uint32_t lkey;

	/* Queue contexts (SHARED by all threads) */
	cq_ctx_t rq_cq_ctx;
	rq_ctx_t rq_ctx;
	sq_ctx_t sq_ctx;
	cq_ctx_t sq_cq_ctx;
	dt_ctx_t dt_ctx;

} app_ctx;

/* ================================================================== */
/* HELPER: MAC SWAP (critical for reflector to avoid loop)            */
/* ================================================================== */

static inline void swap_macs(void *packet_data)
{
	uint8_t *p = (uint8_t *)packet_data;
	uint8_t tmp[6];

	/* Ethernet header: [dst MAC:6B][src MAC:6B][EtherType:2B][...] */
	memcpy(tmp, p, 6);           /* tmp = original dst MAC */
	memcpy(p, p + 6, 6);         /* dst MAC = src MAC */
	memcpy(p + 6, tmp, 6);       /* src MAC = original dst MAC */
}

/* ================================================================== */
/* INITIALIZE SHARED CONTEXT (called once by first thread to run)    */
/* ================================================================== */

static void app_ctx_init(struct host2dev_packet_processor_data *data_from_host)
{
	app_ctx.packets_count = 0;
	app_ctx.lkey = data_from_host->sq_transf.wqd_mkey_id;

	/* Initialize RQ CQ context */
	com_cq_ctx_init(&app_ctx.rq_cq_ctx,
			data_from_host->rq_cq_transf.cq_num,
			data_from_host->rq_cq_transf.log_cq_depth,
			data_from_host->rq_cq_transf.cq_ring_daddr,
			data_from_host->rq_cq_transf.cq_dbr_daddr);

	/* Initialize RQ context */
	com_rq_ctx_init(&app_ctx.rq_ctx,
			data_from_host->rq_transf.wq_num,
			data_from_host->rq_transf.wq_ring_daddr,
			data_from_host->rq_transf.wq_dbr_daddr);

	/* Initialize SQ context */
	com_sq_ctx_init(&app_ctx.sq_ctx,
			data_from_host->sq_transf.wq_num,
			data_from_host->sq_transf.wq_ring_daddr);

	/* Initialize SQ CQ context */
	com_cq_ctx_init(&app_ctx.sq_cq_ctx,
			data_from_host->sq_cq_transf.cq_num,
			data_from_host->sq_cq_transf.log_cq_depth,
			data_from_host->sq_cq_transf.cq_ring_daddr,
			data_from_host->sq_cq_transf.cq_dbr_daddr);

	/* Initialize data transfer context */
	com_dt_ctx_init(&app_ctx.dt_ctx, data_from_host->sq_transf.wqd_daddr);

	flexio_dev_print("DPA: app_ctx initialized. RQ=%u, SQ=%u\n",
			 data_from_host->rq_transf.wq_num,
			 data_from_host->sq_transf.wq_num);
}

/* ================================================================== */
/* PROCESS ONE PACKET                                                 */
/* ================================================================== */

static void process_packet(void)
{
	struct flexio_dev_wqe_rcv_data_seg *rwqe;
	uint32_t rq_wqe_idx;
	char *rq_data;
	uint32_t data_sz;

	union flexio_dev_sqe_seg *swqe;
	char *sq_data;

	/* Extract WQE index and packet size from completed CQE */
	rq_wqe_idx = flexio_dev_cqe_get_wqe_counter(app_ctx.rq_cq_ctx.cqe);
	data_sz = flexio_dev_cqe_get_byte_cnt(app_ctx.rq_cq_ctx.cqe);

	if (data_sz == 0 || data_sz > 2048) {
		flexio_dev_print("DPA: Invalid packet size %u\n", data_sz);
		return;
	}

	/* Get RQ WQE and packet buffer */
	rwqe = &app_ctx.rq_ctx.rq_ring[rq_wqe_idx & RQ_IDX_MASK];
	rq_data = flexio_dev_rwqe_get_addr(rwqe);

	/* Get SQ data buffer for this packet */
	sq_data = get_next_dte(&app_ctx.dt_ctx, DATA_IDX_MASK, 11);

	/* Copy packet data */
	memcpy(sq_data, rq_data, data_sz);

	/* ============================================================== */
	/* CRITICAL: SWAP MACs to avoid packet loop                       */
	/* ============================================================== */
	/* If packet is received with:
	 *   src_mac = host_mac (e.g., AA:BB:CC:DD:EE:FF)
	 *   dst_mac = nic_mac  (e.g., 02:08:A4:D8:FF:43)
	 *
	 * After MAC swap:
	 *   src_mac = nic_mac  (02:08:A4:D8:FF:43)
	 *   dst_mac = host_mac (AA:BB:CC:DD:EE:FF)
	 *
	 * When packet exits NIC and returns, RX rule matches on src_mac.
	 * Since src_mac is now nic_mac (not host_mac), it won't re-enter.
	 */
	swap_macs(sq_data);

	/* ============================================================== */
	/* BUILD SQ WQE (4 segments: CTRL, ETH, DATA, padding)            */
	/* ============================================================== */

	/* CTRL segment */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_ctrl_set(swqe, app_ctx.sq_ctx.sq_pi,
				     app_ctx.sq_ctx.sq_number,
				     FLEXIO_CTRL_SEG_CE_CQE_ON_CQE_ERROR,
				     FLEXIO_CTRL_SEG_TYPE_SEND_EN);

	/* ETH segment */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_eth_set(swqe, 0, 0, 0, NULL);

	/* DATA segment */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, data_sz, app_ctx.lkey,
					     (uint64_t)sq_data);

	/* Padding segment */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);

	/* Ring SQ doorbell to post WQE */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_qp_sq_ring_db(++app_ctx.sq_ctx.sq_pi, app_ctx.sq_ctx.sq_number);
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);

	/* Increment RQ PI to return WQE to hardware */
	flexio_dev_dbr_rq_inc_pi(app_ctx.rq_ctx.rq_dbr);

	/* Update packet counter */
	app_ctx.packets_count++;
}

/* ================================================================== */
/* ENTRY POINT: DPA thread handler                                   */
/* ================================================================== */

/*
 * ARCHITECTURE:
 * - Multiple threads call this with SAME thread_arg (shared data daddr)
 * - Each thread independently polls the RQ CQ
 * - CQE ownership bits (managed by hardware) prevent double-processing
 *
 * THREAD SAFETY:
 * - HW sets owner bit when NIC completes packet → CQE ownership flips
 * - Thread reads CQE owner bit; if SW owns it, processes packet
 * - Thread steps CQ: com_step_cq() advances to next CQE, ownership flips
 * - Next thread sees HW ownership on next CQE, skips to following one
 * - Result: no locking needed; HW ownership serializes access
 *
 * INITIALIZATION:
 * - not_first_run flag prevents re-init of static app_ctx
 * - Multiple threads may see not_first_run==0 initially, but init is
 *   idempotent (just copies from host data), so no corruption
 */

extern flexio_func_t flexio_pp_dev;
__dpa_global__ void flexio_pp_dev(uint64_t thread_arg)
{
	struct host2dev_packet_processor_data *data_from_host =
		(struct host2dev_packet_processor_data *)thread_arg;

	/* Initialize app_ctx once (from shared host data) */
	if (!data_from_host->not_first_run) {
		app_ctx_init(data_from_host);
		data_from_host->not_first_run = 1;
	}

	/*
	 * Poll loop: check CQE ownership bit
	 *
	 * Hardware sets owner bit when packet arrives. When SW (this thread)
	 * sees SW ownership (cq_hw_owner_bit != actual owner), it means WE own
	 * this CQE and can process it.
	 */
	while (flexio_dev_cqe_get_owner(app_ctx.rq_cq_ctx.cqe) !=
	       app_ctx.rq_cq_ctx.cq_hw_owner_bit) {
		/* We own this CQE. Process the packet. */
		__dpa_thread_fence(__DPA_MEMORY, __DPA_R, __DPA_R);
		process_packet();
		com_step_cq(&app_ctx.rq_cq_ctx);
		__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	}

	/* CQ is empty (HW owns all remaining CQEs).
	 * Arm CQ to get interrupt when next packet arrives.
	 */
	flexio_dev_cq_arm(app_ctx.rq_cq_ctx.cq_idx,
			  app_ctx.rq_cq_ctx.cq_number);

	/* Reschedule this thread: yield control to DPA scheduler.
	 * Thread will be re-woken by CQ interrupt on next packet.
	 */
	flexio_dev_thread_reschedule();
}