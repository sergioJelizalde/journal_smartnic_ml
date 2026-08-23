/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * rss_steering.h - 5-tuple RSS across N FlexIO worker RQs (DOCA 3.4 / BF3).
 *
 * Builds via devx + mlx5dv_dr:
 *   ALLOC_TD -> CREATE_RQT(worker rqns) -> CREATE_TIR(TCP) + CREATE_TIR(UDP)
 *   -> DR rules {SMAC, ip_proto} -> TIRs.
 *
 * REQUIREMENT: ibv context opened with MLX5DV_CONTEXT_FLAGS_DEVX
 * (mlx5dv_open_device), otherwise all devx commands fail with EPERM.
 */

#ifndef RSS_STEERING_H_
#define RSS_STEERING_H_

#include <stdint.h>
#include <infiniband/verbs.h>

struct rss_steering_ctx;

/* worker_rqns: array of FlexIO RQ numbers (flexio_rq_get_wq_num()).
 * smac: the single source MAC all generated traffic carries.
 * Returns NULL on failure (syndrome printed to stderr).
 */
struct rss_steering_ctx *rss_steering_create(struct ibv_context *ibv_ctx,
					     const uint32_t *worker_rqns,
					     int num_workers, uint64_t smac);

/* Destroys rules, TIRs, RQT, TD. Call BEFORE destroying the worker RQs. */
void rss_steering_destroy(struct rss_steering_ctx *rss);

struct mlx5dv_devx_obj *rss_steering_get_tir(struct rss_steering_ctx *rss, int udp);

#endif /* RSS_STEERING_H_ */
