/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * rss_steering.c / rss_steering.h (single-file module; see header at bottom)
 *
 * 5-tuple RSS across N FlexIO worker RQs for DOCA 3.4 / BlueField-3.
 *
 * FlexIO has no RQT/RSS-TIR API, so this module issues the PRM commands
 * directly via devx and installs the RX rules via mlx5dv_dr:
 *
 *   ALLOC_TRANSPORT_DOMAIN                     (devx)
 *   CREATE_RQT  { rqn[0..N-1] }                (devx)  <- FlexIO RQ wq_nums
 *   CREATE_TIR  { INDIRECT, TOEPLITZ, TCP4 }   (devx)
 *   CREATE_TIR  { INDIRECT, TOEPLITZ, UDP4 }   (devx)
 *   DR rule: SMAC + ip_proto==TCP -> TCP TIR
 *   DR rule: SMAC + ip_proto==UDP -> UDP TIR
 *
 * Two TIRs because a TIR's hash-field selector applies to ONE l4_prot_type;
 * splitting by ip_protocol at the rule level gives true 5-tuple behavior
 * for both TCP and UDP (this mirrors what mlx5 DPDK PMD does internally).
 *
 * PRM layouts follow mlx5_ifc.h (kernel / DPDK mlx5_prm.h). If a command
 * fails, the syndrome is printed - cross-check offsets against the
 * mlx5_ifc.h shipped with your MLNX_OFED if that happens.
 *
 * REQUIREMENT: the ibv context MUST be opened with DEVX enabled:
 *     struct mlx5dv_context_attr da = { .flags = MLX5DV_CONTEXT_FLAGS_DEVX };
 *     ibv_ctx = mlx5dv_open_device(dev, &da);
 * (plain ibv_open_device() will make every devx command fail with EPERM.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

/* ------------------------------------------------------------------ */
/* Minimal PRM definitions (subset of mlx5_ifc.h needed here).         */
/* ------------------------------------------------------------------ */
#define MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN 0x816
#define MLX5_CMD_OP_CREATE_RQT             0x916
#define MLX5_CMD_OP_CREATE_TIR             0x900

#define MLX5_TIRC_DISP_TYPE_INDIRECT 0x1
#define MLX5_RX_HASH_FN_TOEPLITZ     0x2

/* rx_hash_field_select.selected_fields bits */
#define MLX5_HASH_FIELD_SEL_SRC_IP   (1 << 0)
#define MLX5_HASH_FIELD_SEL_DST_IP   (1 << 1)
#define MLX5_HASH_FIELD_SEL_L4_SPORT (1 << 2)
#define MLX5_HASH_FIELD_SEL_L4_DPORT (1 << 3)

/* rx_hash_field_select prot type bits (dword bit positions) */
#define HASH_SEL_L3_IPV6 (1u << 31) /* 0 = IPv4 */
#define HASH_SEL_L4_UDP  (1u << 30) /* 0 = TCP  */

/* fte_match_param: 512B; outer_headers at offset 0. */
#define FTE_MATCH_PARAM_BSIZE 512
/* outer_headers (fte_match_set_lyr_2_4) byte offsets: */
#define OUT_SMAC_47_16 0x00 /* be32 */
#define OUT_SMAC_15_0  0x04 /* be16 in [31:16] of the be32 at 0x04 */
#define OUT_IPPROTO_DW 0x10 /* ip_protocol = bits [31:24] of be32 at 0x10 */

/* Toeplitz key (standard Microsoft RSS key, 40B). */
static const uint8_t rss_key[40] = {
	0x6d, 0x5a, 0x56, 0xda, 0x25, 0x5b, 0x0e, 0xc2,
	0x41, 0x67, 0x25, 0x3d, 0x43, 0xa3, 0x8f, 0xb0,
	0xd0, 0xca, 0x2b, 0xcb, 0xae, 0x7b, 0x30, 0xb4,
	0x77, 0xcb, 0x2d, 0xa3, 0x80, 0x30, 0xf2, 0x0c,
	0x6a, 0x42, 0xb7, 0x3b, 0xbe, 0xac, 0x01, 0xfa,
};

/* be32 helpers over a byte buffer. */
static inline void buf_set32(void *buf, size_t byte_off, uint32_t host_val)
{
	uint32_t be = htonl(host_val);
	memcpy((uint8_t *)buf + byte_off, &be, 4);
}

static inline uint32_t buf_get32(const void *buf, size_t byte_off)
{
	uint32_t be;
	memcpy(&be, (const uint8_t *)buf + byte_off, 4);
	return ntohl(be);
}

static void print_syndrome(const char *what, const void *out)
{
	fprintf(stderr, "%s failed: status=0x%x syndrome=0x%x\n",
		what, buf_get32(out, 0) >> 24, buf_get32(out, 4));
}

void rss_steering_destroy(struct rss_steering_ctx *rss);

/* ------------------------------------------------------------------ */
/* Context handed back to the application.                             */
/* ------------------------------------------------------------------ */
struct rss_steering_ctx {
	struct mlx5dv_devx_obj *td_obj;
	struct mlx5dv_devx_obj *rqt_obj;
	struct mlx5dv_devx_obj *tir_tcp_obj;
	struct mlx5dv_devx_obj *tir_udp_obj;
	uint32_t tdn;
	uint32_t rqtn;
	uint32_t tirn_tcp;
	uint32_t tirn_udp;

	struct mlx5dv_dr_domain *dr_domain;
	struct mlx5dv_dr_table *dr_table;
	struct mlx5dv_dr_matcher *dr_matcher;
	struct mlx5dv_dr_action *act_tir_tcp;
	struct mlx5dv_dr_action *act_tir_udp;
	struct mlx5dv_dr_rule *rule_tcp;
	struct mlx5dv_dr_rule *rule_udp;
};

/* ------------------------------------------------------------------ */
/* devx: ALLOC_TRANSPORT_DOMAIN                                        */
/* ------------------------------------------------------------------ */
static int alloc_td(struct ibv_context *ctx, struct rss_steering_ctx *rss)
{
	uint8_t in[16] = {0};
	uint8_t out[16] = {0};

	buf_set32(in, 0x00, MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN << 16);

	rss->td_obj = mlx5dv_devx_obj_create(ctx, in, sizeof(in), out, sizeof(out));
	if (!rss->td_obj) {
		print_syndrome("ALLOC_TRANSPORT_DOMAIN", out);
		return -1;
	}
	rss->tdn = buf_get32(out, 0x08) & 0xffffff;
	return 0;
}

/* ------------------------------------------------------------------ */
/* devx: CREATE_RQT over the FlexIO worker RQ numbers.                 */
/* rqtc at in+0x20; rqt_max|actual at rqtc+0x14; rq list at rqtc+0xEC. */
/* ------------------------------------------------------------------ */
static int create_rqt(struct ibv_context *ctx, struct rss_steering_ctx *rss,
		      const uint32_t *rqn, int num_rqs)
{
	const size_t rqtc_off = 0x20;
	const size_t list_off = rqtc_off + 0xEC;
	size_t in_bsize = list_off + 4 * (size_t)num_rqs;
	uint8_t out[16] = {0};
	uint8_t *in;
	int i;

	in = calloc(1, in_bsize);
	if (!in)
		return -1;

	buf_set32(in, 0x00, MLX5_CMD_OP_CREATE_RQT << 16);
	/* rqt_max_size[31:16] | rqt_actual_size[15:0] */
	buf_set32(in, rqtc_off + 0x14,
		  ((uint32_t)num_rqs << 16) | (uint32_t)num_rqs);
	for (i = 0; i < num_rqs; i++)
		buf_set32(in, list_off + 4 * (size_t)i, rqn[i] & 0xffffff);

	rss->rqt_obj = mlx5dv_devx_obj_create(ctx, in, in_bsize, out, sizeof(out));
	free(in);
	if (!rss->rqt_obj) {
		print_syndrome("CREATE_RQT", out);
		return -1;
	}
	rss->rqtn = buf_get32(out, 0x08) & 0xffffff;
	return 0;
}

/* ------------------------------------------------------------------ */
/* devx: CREATE_TIR (indirect, Toeplitz over IPv4 5-tuple fields).     */
/* l4_udp selects UDP vs TCP for the hash field selector.              */
/* ------------------------------------------------------------------ */
static struct mlx5dv_devx_obj *create_rss_tir(struct ibv_context *ctx,
					      struct rss_steering_ctx *rss,
					      int l4_udp, uint32_t *tirn_out)
{
	const size_t tirc = 0x20;
	uint8_t in[0x20 + 0xF0];
	uint8_t out[16] = {0};
	struct mlx5dv_devx_obj *obj;
	uint32_t fsel;
	int i;

	memset(in, 0, sizeof(in));
	buf_set32(in, 0x00, MLX5_CMD_OP_CREATE_TIR << 16);

	/* disp_type[31:28] = INDIRECT */
	buf_set32(in, tirc + 0x04, (uint32_t)MLX5_TIRC_DISP_TYPE_INDIRECT << 28);
	/* indirect_table[23:0] = rqtn */
	buf_set32(in, tirc + 0x20, rss->rqtn & 0xffffff);
	/* rx_hash_fn[31:28]=TOEPLITZ | transport_domain[23:0] */
	buf_set32(in, tirc + 0x24,
		  ((uint32_t)MLX5_RX_HASH_FN_TOEPLITZ << 28) | (rss->tdn & 0xffffff));
	/* rx_hash_toeplitz_key: 10 dwords at tirc+0x28 */
	for (i = 0; i < 10; i++) {
		uint32_t k = ((uint32_t)rss_key[4 * i] << 24) |
			     ((uint32_t)rss_key[4 * i + 1] << 16) |
			     ((uint32_t)rss_key[4 * i + 2] << 8) |
			     ((uint32_t)rss_key[4 * i + 3]);
		buf_set32(in, tirc + 0x28 + 4 * (size_t)i, k);
	}
	/* outer rx_hash_field_select: IPv4 + (TCP|UDP) + 4 fields */
	fsel = MLX5_HASH_FIELD_SEL_SRC_IP | MLX5_HASH_FIELD_SEL_DST_IP |
	       MLX5_HASH_FIELD_SEL_L4_SPORT | MLX5_HASH_FIELD_SEL_L4_DPORT;
	if (l4_udp)
		fsel |= HASH_SEL_L4_UDP;
	buf_set32(in, tirc + 0x50, fsel);

	obj = mlx5dv_devx_obj_create(ctx, in, sizeof(in), out, sizeof(out));
	if (!obj) {
		print_syndrome(l4_udp ? "CREATE_TIR(UDP)" : "CREATE_TIR(TCP)", out);
		return NULL;
	}
	*tirn_out = buf_get32(out, 0x08) & 0xffffff;
	return obj;
}

/* ------------------------------------------------------------------ */
/* DR: matcher on {SMAC, ip_protocol}; two rules -> two TIRs.          */
/* ------------------------------------------------------------------ */
struct match_buf {
	uint32_t match_sz;
	uint8_t buf[FTE_MATCH_PARAM_BSIZE];
};

static void match_set_smac(uint8_t *p, uint64_t smac)
{
	buf_set32(p, OUT_SMAC_47_16, (uint32_t)(smac >> 16));
	buf_set32(p, OUT_SMAC_15_0, ((uint32_t)(smac & 0xffff)) << 16);
}

static void match_set_ipproto(uint8_t *p, uint8_t proto)
{
	buf_set32(p, OUT_IPPROTO_DW, (uint32_t)proto << 24);
}

static int create_dr_rules(struct ibv_context *ctx, struct rss_steering_ctx *rss,
			   uint64_t smac)
{
	struct match_buf mask = { .match_sz = FTE_MATCH_PARAM_BSIZE };
	struct match_buf val = { .match_sz = FTE_MATCH_PARAM_BSIZE };
	struct mlx5dv_dr_action *actions[1];

	rss->dr_domain = mlx5dv_dr_domain_create(ctx, MLX5DV_DR_DOMAIN_TYPE_NIC_RX);
	if (!rss->dr_domain) {
		fprintf(stderr, "dr_domain_create failed (%d)\n", errno);
		return -1;
	}

	rss->dr_table = mlx5dv_dr_table_create(rss->dr_domain, 0);
	if (!rss->dr_table) {
		fprintf(stderr, "dr_table_create failed (%d)\n", errno);
		return -1;
	}

	/* Mask: full SMAC + full ip_protocol. criteria_enable bit0 = outer. */
	memset(mask.buf, 0, sizeof(mask.buf));
	match_set_smac(mask.buf, 0xffffffffffffULL);
	match_set_ipproto(mask.buf, 0xff);

	rss->dr_matcher = mlx5dv_dr_matcher_create(rss->dr_table, 0, 1 /* outer */,
			(struct mlx5dv_flow_match_parameters *)&mask);
	if (!rss->dr_matcher) {
		fprintf(stderr, "dr_matcher_create failed (%d)\n", errno);
		return -1;
	}

	rss->act_tir_tcp = mlx5dv_dr_action_create_dest_devx_tir(rss->tir_tcp_obj);
	rss->act_tir_udp = mlx5dv_dr_action_create_dest_devx_tir(rss->tir_udp_obj);
	if (!rss->act_tir_tcp || !rss->act_tir_udp) {
		fprintf(stderr, "dr_action_create_dest_devx_tir failed (%d)\n", errno);
		return -1;
	}

	/* Rule 1: SMAC + TCP(6) -> TCP TIR */
	memset(val.buf, 0, sizeof(val.buf));
	match_set_smac(val.buf, smac);
	match_set_ipproto(val.buf, 6);
	actions[0] = rss->act_tir_tcp;
	rss->rule_tcp = mlx5dv_dr_rule_create(rss->dr_matcher,
			(struct mlx5dv_flow_match_parameters *)&val, 1, actions);
	if (!rss->rule_tcp) {
		fprintf(stderr, "dr_rule_create(TCP) failed (%d)\n", errno);
		return -1;
	}

	/* Rule 2: SMAC + UDP(17) -> UDP TIR */
	memset(val.buf, 0, sizeof(val.buf));
	match_set_smac(val.buf, smac);
	match_set_ipproto(val.buf, 17);
	actions[0] = rss->act_tir_udp;
	rss->rule_udp = mlx5dv_dr_rule_create(rss->dr_matcher,
			(struct mlx5dv_flow_match_parameters *)&val, 1, actions);
	if (!rss->rule_udp) {
		fprintf(stderr, "dr_rule_create(UDP) failed (%d)\n", errno);
		return -1;
	}

	return 0;
}

/* ------------------------------------------------------------------ */
/* Public API.                                                         */
/* ------------------------------------------------------------------ */
struct rss_steering_ctx *rss_steering_create(struct ibv_context *ibv_ctx,
					     const uint32_t *worker_rqns,
					     int num_workers, uint64_t smac)
{
	struct rss_steering_ctx *rss;

	rss = calloc(1, sizeof(*rss));
	if (!rss)
		return NULL;

	if (alloc_td(ibv_ctx, rss))
		goto err;
	if (create_rqt(ibv_ctx, rss, worker_rqns, num_workers))
		goto err;

	rss->tir_tcp_obj = create_rss_tir(ibv_ctx, rss, 0, &rss->tirn_tcp);
	if (!rss->tir_tcp_obj)
		goto err;
	rss->tir_udp_obj = create_rss_tir(ibv_ctx, rss, 1, &rss->tirn_udp);
	if (!rss->tir_udp_obj)
		goto err;

	if (create_dr_rules(ibv_ctx, rss, smac))
		goto err;

	printf("RSS: TD %#x, RQT %#x (%d RQs), TIR tcp=%#x udp=%#x, "
	       "SMAC %012lx 5-tuple spread active\n",
	       rss->tdn, rss->rqtn, num_workers, rss->tirn_tcp, rss->tirn_udp, smac);
	return rss;

err:
	fprintf(stderr, "rss_steering_create failed\n");
	rss_steering_destroy(rss); /* frees whatever was created */
	return NULL;
}

void rss_steering_destroy(struct rss_steering_ctx *rss)
{
	if (!rss)
		return;

	if (rss->rule_udp)
		mlx5dv_dr_rule_destroy(rss->rule_udp);
	if (rss->rule_tcp)
		mlx5dv_dr_rule_destroy(rss->rule_tcp);
	if (rss->act_tir_udp)
		mlx5dv_dr_action_destroy(rss->act_tir_udp);
	if (rss->act_tir_tcp)
		mlx5dv_dr_action_destroy(rss->act_tir_tcp);
	if (rss->dr_matcher)
		mlx5dv_dr_matcher_destroy(rss->dr_matcher);
	if (rss->dr_table)
		mlx5dv_dr_table_destroy(rss->dr_table);
	if (rss->dr_domain)
		mlx5dv_dr_domain_destroy(rss->dr_domain);

	if (rss->tir_udp_obj)
		mlx5dv_devx_obj_destroy(rss->tir_udp_obj);
	if (rss->tir_tcp_obj)
		mlx5dv_devx_obj_destroy(rss->tir_tcp_obj);
	if (rss->rqt_obj)
		mlx5dv_devx_obj_destroy(rss->rqt_obj);
	if (rss->td_obj)
		mlx5dv_devx_obj_destroy(rss->td_obj);

	free(rss);
}

/* ==================================================================
 * rss_steering.h - copy into a separate header:
 * ==================================================================
 * #ifndef RSS_STEERING_H_
 * #define RSS_STEERING_H_
 * #include <stdint.h>
 * #include <infiniband/verbs.h>
 *
 * struct rss_steering_ctx;
 *
 * struct rss_steering_ctx *rss_steering_create(struct ibv_context *ibv_ctx,
 *                                              const uint32_t *worker_rqns,
 *                                              int num_workers, uint64_t smac);
 * void rss_steering_destroy(struct rss_steering_ctx *rss);
 *
 * #endif
 * ==================================================================
 */
