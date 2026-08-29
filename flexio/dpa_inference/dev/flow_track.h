/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Minimal in-DPA flow table: enough per-5-tuple state to build a feature vector for
 * flow_ml_forest_infer(), with no dynamic allocation and no libc hash map (none is available in
 * this freestanding DPA environment). Direct-mapped by hash, single slot per bucket: a colliding
 * flow simply evicts and restarts tracking on the old one. That's an acceptable trade-off for a
 * lightweight per-packet anomaly signal; raise FLOW_TABLE_SIZE if collisions become a problem for
 * your traffic mix.
 *
 * No timestamps are used (there is no verified wall-clock/cycle-counter API wired into this
 * freestanding build yet), so there is no inter-arrival-time feature -- only packet size and
 * per-flow packet-size history. See flow_ml_model.h's FLOW_FEAT_* for the exact feature set.
 */

#ifndef FLOW_TRACK_H_
#define FLOW_TRACK_H_

#include <stdint.h>
#include "flow_ml_model.h"

#define FLOW_TABLE_SIZE (256) /* Must be a power of two */
#define FLOW_TABLE_MASK (FLOW_TABLE_SIZE - 1)

#define FLOW_EWMA_ALPHA_NUM (1) /* alpha = 1/4 in fxp16: new_ewma = (size<<16)/4 + 3*old_ewma/4 */
#define FLOW_EWMA_ALPHA_DEN (4)

/* One tracked flow's running state. ip/port pairs are stored in canonical (unordered-by-direction)
 * form so both directions of a flow land on the same entry. */
typedef struct {
	uint32_t ip_lo, ip_hi;	  /* canonicalized (lower, higher) endpoint IPs, network byte order */
	uint16_t port_lo, port_hi; /* canonicalized (lower, higher) endpoint ports, host byte order */
	uint8_t proto;		  /* IP protocol number (6 = TCP, 17 = UDP) */
	uint8_t valid;		  /* 0 = empty/unused slot */
	uint16_t pkt_count;	  /* packets seen so far, saturating at UINT16_MAX */
	uint16_t last_size;	  /* previous packet's size, in bytes */
	int32_t ewma_size;	  /* EWMA-smoothed packet size, fxp16 */
} flow_entry_t;

/*
 * Parse the Ethernet/IPv4/L4 headers of a raw wire packet.
 *
 * @pkt [in]: pointer to the start of the Ethernet header
 * @len [in]: total captured length of the packet, in bytes
 * @src_ip, @dst_ip [out]: IPv4 addresses, network byte order
 * @src_port, @dst_port [out]: L4 ports, host byte order (0 if proto has none)
 * @proto [out]: IP protocol number
 * @return: 1 if this is an IPv4 TCP/UDP packet with enough captured bytes to read the L4 ports,
 *          0 otherwise (non-IPv4, non-TCP/UDP, or truncated) -- caller should skip tracking it.
 */
static inline int flow_track_parse_5tuple(const uint8_t *pkt,
					   uint32_t len,
					   uint32_t *src_ip,
					   uint32_t *dst_ip,
					   uint16_t *src_port,
					   uint16_t *dst_port,
					   uint8_t *proto)
{
	uint32_t ip_off, ihl, l4_off;
	uint16_t ethertype;

	if (len < 14 + 20)
		return 0;

	ethertype = ((uint16_t)pkt[12] << 8) | pkt[13];
	if (ethertype != 0x0800) /* Only IPv4 is tracked */
		return 0;

	ip_off = 14;
	if ((pkt[ip_off] >> 4) != 4) /* IP version */
		return 0;

	ihl = (uint32_t)(pkt[ip_off] & 0x0F) * 4;
	if (ihl < 20 || len < ip_off + ihl + 4)
		return 0;

	*proto = pkt[ip_off + 9];
	if (*proto != 6 /* TCP */ && *proto != 17 /* UDP */)
		return 0;

	*src_ip = ((uint32_t)pkt[ip_off + 12] << 24) | ((uint32_t)pkt[ip_off + 13] << 16) |
		  ((uint32_t)pkt[ip_off + 14] << 8) | (uint32_t)pkt[ip_off + 15];
	*dst_ip = ((uint32_t)pkt[ip_off + 16] << 24) | ((uint32_t)pkt[ip_off + 17] << 16) |
		  ((uint32_t)pkt[ip_off + 18] << 8) | (uint32_t)pkt[ip_off + 19];

	l4_off = ip_off + ihl;
	*src_port = ((uint16_t)pkt[l4_off] << 8) | pkt[l4_off + 1];
	*dst_port = ((uint16_t)pkt[l4_off + 2] << 8) | pkt[l4_off + 3];

	return 1;
}

/* Small integer hash over the canonicalized 5-tuple, folded into FLOW_TABLE_MASK bits. */
static inline uint32_t flow_track_hash(uint32_t ip_lo, uint32_t ip_hi, uint16_t port_lo, uint16_t port_hi, uint8_t proto)
{
	uint32_t h = ip_lo * 2654435761u;

	h ^= ip_hi * 40503u;
	h ^= ((uint32_t)port_lo << 16) | port_hi;
	h ^= (uint32_t)proto * 97u;
	h ^= h >> 15;
	return h & FLOW_TABLE_MASK;
}

/*
 * Look up (or start tracking) the flow this packet belongs to, update its running state, and fill
 * in the feature vector for flow_ml_forest_infer().
 *
 * @table [in/out]: the flow table (FLOW_TABLE_SIZE entries)
 * @src_ip, @dst_ip, @src_port, @dst_port, @proto [in]: this packet's 5-tuple, as from
 *                                                       flow_track_parse_5tuple()
 * @pkt_size [in]: this packet's size, in bytes
 * @feat_out [out]: FLOW_ML_NUM_FEATURES-sized array to fill in
 */
static inline void flow_track_update(flow_entry_t *table,
				      uint32_t src_ip,
				      uint32_t dst_ip,
				      uint16_t src_port,
				      uint16_t dst_port,
				      uint8_t proto,
				      uint16_t pkt_size,
				      int32_t *feat_out)
{
	uint32_t ip_lo = src_ip, ip_hi = dst_ip;
	uint16_t port_lo = src_port, port_hi = dst_port;
	uint32_t idx;
	flow_entry_t *e;

	/* Canonicalize direction so both directions of a flow hash to the same entry. */
	if (ip_lo > ip_hi || (ip_lo == ip_hi && port_lo > port_hi)) {
		uint32_t tmp_ip = ip_lo;
		uint16_t tmp_port = port_lo;

		ip_lo = ip_hi;
		ip_hi = tmp_ip;
		port_lo = port_hi;
		port_hi = tmp_port;
	}

	idx = flow_track_hash(ip_lo, ip_hi, port_lo, port_hi, proto);
	e = &table[idx];

	if (!e->valid || e->ip_lo != ip_lo || e->ip_hi != ip_hi || e->port_lo != port_lo ||
	    e->port_hi != port_hi || e->proto != proto) {
		/* New flow, or a collision evicting whatever was here: (re)start tracking. */
		e->ip_lo = ip_lo;
		e->ip_hi = ip_hi;
		e->port_lo = port_lo;
		e->port_hi = port_hi;
		e->proto = proto;
		e->valid = 1;
		e->pkt_count = 0;
		e->last_size = pkt_size;
		e->ewma_size = (int32_t)pkt_size << 16;
	}

	feat_out[FLOW_FEAT_PKT_SIZE] = pkt_size;
	feat_out[FLOW_FEAT_PKT_COUNT] = (e->pkt_count > 255) ? 255 : (int32_t)e->pkt_count; /* saturate to match FLOW_FEAT_PKT_COUNT's documented 0..255 range */
	feat_out[FLOW_FEAT_EWMA_SIZE] = e->ewma_size;
	feat_out[FLOW_FEAT_SIZE_DELTA] = (pkt_size > e->last_size) ? (pkt_size - e->last_size) : (e->last_size - pkt_size);

	/* Update state for the next packet on this flow. */
	e->ewma_size = (int32_t)(((int64_t)pkt_size << 16) * FLOW_EWMA_ALPHA_NUM +
				  (int64_t)e->ewma_size * (FLOW_EWMA_ALPHA_DEN - FLOW_EWMA_ALPHA_NUM)) /
		       FLOW_EWMA_ALPHA_DEN;
	e->last_size = pkt_size;
	if (e->pkt_count < UINT16_MAX)
		e->pkt_count++;
}

#endif /* FLOW_TRACK_H_ */
