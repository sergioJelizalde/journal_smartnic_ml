/*
 * Copyright (c) 2026 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ML_CC_CTXT_H_
#define ML_CC_CTXT_H_

/* Per-flow event flags observed since the last model inference */
typedef struct {
	uint8_t was_nack : 1; /* A NACK was seen on this flow */
	uint8_t was_cnp : 1;  /* A CNP was seen on this flow */
	uint8_t reserved : 6; /* Reserved bits */
} ml_cc_flags_t;

/*
 * Per-flow context. This is the hardware-tracked state libpcc keeps per flow/QP and hands back
 * to the algorithm on every event. It is intentionally kept to the same 48-byte footprint the
 * reference PCC apps use (see rtt_template) since that is the fixed budget a flow context slot
 * is given; only the small set of fields the model actually consumes as features are kept here,
 * plus the minimal RTT-request/response bookkeeping needed to produce a fresh RTT sample.
 */
typedef struct {
	uint32_t cur_rate;	     /* Current rate applied to the flow (fxp20) */
	uint32_t start_delay;	     /* Timestamp at which the RTT-request packet was sent */
	uint32_t rtt;		     /* Last measured round-trip-time sample (ns) */
	uint32_t min_rtt;	     /* Running minimum RTT observed on this flow (ns) */
	uint32_t ewma_rtt;	     /* EWMA-smoothed RTT (ns) -- model input feature */
	ml_cc_flags_t flags;	     /* Flags accumulated since the last inference */
	uint8_t abort_cnt;	     /* Count of aborted RTT requests (protocol bookkeeping) */
	uint8_t rtt_meas_psn;	     /* RTT request sequence number (protocol bookkeeping) */
	uint8_t rtt_req_to_rtt_sent; /* Protocol bookkeeping between RTT request and RTT-sent event */
	uint8_t cnp_streak;	     /* Saturating count of consecutive CNPs -- model input feature */
	uint32_t reserved[5];	     /* Keep total context size at 48 bytes, matching the reference apps */
} ml_cc_ctxt_t;

#endif /* ML_CC_CTXT_H_ */
