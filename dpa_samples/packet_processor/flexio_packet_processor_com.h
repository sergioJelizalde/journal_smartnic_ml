/*
 * FLEXIO PACKET PROCESSOR - COMMON HEADER
 * DOCA 3.4 / FlexIO 26.4
 * 
 * Shared data structures and utilities for host <-> device communication
 * Used by both host and device-side code
 */

#ifndef FLEXIO_PACKET_PROCESSOR_COM_H
#define FLEXIO_PACKET_PROCESSOR_COM_H

#include <stdint.h>

/* Device-side includes differ from host */
#ifdef __DPA__
#include <libflexio-libc/string.h>
#else
#include <string.h>
#endif

/* ================================================================== */
/* QUEUE CONTEXT STRUCTURES (from/to DPA)                            */
/* ================================================================== */

typedef struct {
	uint32_t cq_number;
	uint32_t cq_idx;
	uint32_t log_cq_depth;
	uint32_t cq_hw_owner_bit;
	uint64_t cq_ring_daddr;
	uint64_t cq_dbr_daddr;
	void *cqe;
} cq_ctx_t;

typedef struct {
	uint32_t wq_number;
	uint32_t wq_idx;
	uint64_t wq_ring_daddr;
	uint64_t wq_dbr_daddr;
} rq_ctx_t;

typedef struct {
	uint32_t wq_number;
	uint32_t sq_pi;
	uint64_t wq_ring_daddr;
} sq_ctx_t;

typedef struct {
	uint64_t wqd_daddr;
} dt_ctx_t;

/* ================================================================== */
/* TRANSFER STRUCTURES (host sends to device)                        */
/* ================================================================== */

struct app_transfer_cq {
	uint32_t cq_num;
	uint32_t log_cq_depth;
	uint64_t cq_ring_daddr;
	uint64_t cq_dbr_daddr;
};

struct app_transfer_wq {
	uint32_t wq_num;
	uint32_t wqd_mkey_id;
	uint64_t wqd_daddr;
	uint64_t wq_ring_daddr;
	uint64_t wq_dbr_daddr;
};

struct host2dev_packet_processor_data {
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_wq rq_transf;
	struct app_transfer_cq sq_cq_transf;
	struct app_transfer_wq sq_transf;
	uint32_t not_first_run;
};

/* ================================================================== */
/* DEVICE-SIDE HELPERS (when included in device code)               */
/* ================================================================== */

#ifdef __DPA__

/* CQ context initialization */
static inline void com_cq_ctx_init(cq_ctx_t *ctx, uint32_t cq_num,
				    uint32_t log_depth, uint64_t ring_daddr,
				    uint64_t dbr_daddr)
{
	ctx->cq_number = cq_num;
	ctx->log_cq_depth = log_depth;
	ctx->cq_idx = 0;
	ctx->cq_hw_owner_bit = 1;
	ctx->cq_ring_daddr = ring_daddr;
	ctx->cq_dbr_daddr = dbr_daddr;
	ctx->cqe = (void *)ring_daddr;
}

/* RQ context initialization */
static inline void com_rq_ctx_init(rq_ctx_t *ctx, uint32_t wq_num,
				    uint64_t ring_daddr, uint64_t dbr_daddr)
{
	ctx->wq_number = wq_num;
	ctx->wq_idx = 0;
	ctx->wq_ring_daddr = ring_daddr;
	ctx->wq_dbr_daddr = dbr_daddr;
}

/* SQ context initialization */
static inline void com_sq_ctx_init(sq_ctx_t *ctx, uint32_t wq_num,
				    uint64_t ring_daddr)
{
	ctx->wq_number = wq_num;
	ctx->sq_pi = 0;
	ctx->wq_ring_daddr = ring_daddr;
}

/* Data transfer context initialization */
static inline void com_dt_ctx_init(dt_ctx_t *ctx, uint64_t wqd_daddr)
{
	ctx->wqd_daddr = wqd_daddr;
}

/* Step CQ to next entry (advance index and flip ownership bit) */
static inline void com_step_cq(cq_ctx_t *ctx)
{
	uint32_t cq_depth = (1U << ctx->log_cq_depth);

	ctx->cq_idx = (ctx->cq_idx + 1) & (cq_depth - 1);

	/* Flip ownership bit every wrap-around */
	if (ctx->cq_idx == 0)
		ctx->cq_hw_owner_bit ^= 1;

	/* Advance CQE pointer */
	ctx->cqe = (void *)(ctx->cq_ring_daddr + (ctx->cq_idx * 64));
}

/* Get next SQ WQE segment (4 segments per packet) */
static inline void *get_next_sqe(sq_ctx_t *ctx, uint32_t idx_mask)
{
	uint32_t wqe_idx = ctx->sq_pi & idx_mask;
	uint64_t wqe_addr = ctx->wq_ring_daddr + (wqe_idx * 64);
	return (void *)wqe_addr;
}

/* Get next data buffer for packet (RQ data) */
static inline void *get_next_dte(dt_ctx_t *ctx, uint32_t idx_mask,
				  uint32_t log_entry_size)
{
	static uint32_t dt_idx = 0;
	uint32_t entry_idx = dt_idx & idx_mask;
	uint32_t entry_size = (1U << log_entry_size);
	uint64_t addr = ctx->wqd_daddr + (entry_idx * entry_size);

	dt_idx++;
	return (void *)addr;
}

#endif /* __DPA__ */

#endif /* FLEXIO_PACKET_PROCESSOR_COM_H */