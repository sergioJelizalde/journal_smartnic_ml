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

#include <doca_pcc_dev.h>
#include <doca_pcc_dev_event.h>
#include <doca_pcc_dev_algo_access.h>

#include "utils.h"
#include "ml_cc_ctxt.h"
#include "ml_cc_algo_params.h"
#include "ml_cc_model.h"
#include "ml_cc.h"

#pragma clang diagnostic ignored "-Wunused-parameter"

#define ABORT_TIME (300000) /* Time to abort a pending RTT request, in nanosec */

typedef enum {
	ML_CC_EWMA_ALPHA = 0,	  /* configurable RTT EWMA smoothing factor */
	ML_CC_NEW_FLOW_RATE = 1, /* configurable initial rate for a new flow */
	ML_CC_MIN_RATE = 2,	  /* configurable rate floor */
	ML_CC_MAX_DELAY = 3,	  /* configurable absolute-RTT safety-net threshold */
	ML_CC_PARAM_NUM		  /* Maximal number of configurable parameters */
} ml_cc_params_t;

enum {
	ML_CC_COUNTER_TX_EVENT = 0,   /* tx events handled */
	ML_CC_COUNTER_INFERENCE = 1, /* number of times the forest model was invoked */
	ML_CC_COUNTER_NUM	      /* Maximal number of counters */
} ml_cc_counter_t;

const volatile char ml_cc_desc[] = "ML congestion control (random-forest inference) v0.1";
static const volatile char ml_cc_param_ewma_alpha_desc[] = "EWMA_ALPHA, RTT smoothing factor";
static const volatile char ml_cc_param_new_flow_rate_desc[] = "NEW_FLOW_RATE, new flow rate";
static const volatile char ml_cc_param_min_rate_desc[] = "MIN_RATE, min rate";
static const volatile char ml_cc_param_max_delay_desc[] = "MAX_DELAY, max delay safety net";
static const volatile char ml_cc_counter_tx_desc[] = "COUNTER_TX_EVENT, number of tx events handled";
static const volatile char ml_cc_counter_inference_desc[] = "COUNTER_INFERENCE, number of model inferences run";

void ml_cc_init(uint32_t algo_idx)
{
	struct doca_pcc_dev_algo_meta_data algo_def = {0};

	algo_def.algo_id = 0xBFFE;
	algo_def.algo_major_version = 0x00;
	algo_def.algo_minor_version = 0x01;
	algo_def.algo_desc_size = sizeof(ml_cc_desc);
	algo_def.algo_desc_addr = (uint64_t)ml_cc_desc;

	uint32_t total_param_num = ML_CC_PARAM_NUM;
	uint32_t total_counter_num = ML_CC_COUNTER_NUM;
	uint32_t param_num = 0;
	uint32_t counter_num = 0;

	doca_pcc_dev_algo_init_metadata(algo_idx, &algo_def, total_param_num, total_counter_num);

	doca_pcc_dev_algo_init_param(algo_idx,
				     param_num++,
				     EWMA_ALPHA,
				     EWMA_ALPHA_MAX,
				     1,
				     1,
				     sizeof(ml_cc_param_ewma_alpha_desc),
				     (uint64_t)ml_cc_param_ewma_alpha_desc);
	doca_pcc_dev_algo_init_param(algo_idx,
				     param_num++,
				     NEW_FLOW_RATE,
				     RATE_MAX,
				     1,
				     1,
				     sizeof(ml_cc_param_new_flow_rate_desc),
				     (uint64_t)ml_cc_param_new_flow_rate_desc);
	doca_pcc_dev_algo_init_param(algo_idx,
				     param_num++,
				     MIN_RATE,
				     RATE_MAX,
				     1,
				     1,
				     sizeof(ml_cc_param_min_rate_desc),
				     (uint64_t)ml_cc_param_min_rate_desc);
	doca_pcc_dev_algo_init_param(algo_idx,
				     param_num++,
				     MAX_DELAY,
				     UINT32_MAX,
				     1,
				     1,
				     sizeof(ml_cc_param_max_delay_desc),
				     (uint64_t)ml_cc_param_max_delay_desc);

	doca_pcc_dev_algo_init_counter(algo_idx,
				       counter_num++,
				       UINT32_MAX,
				       2,
				       sizeof(ml_cc_counter_tx_desc),
				       (uint64_t)ml_cc_counter_tx_desc);
	doca_pcc_dev_algo_init_counter(algo_idx,
				       counter_num++,
				       UINT32_MAX,
				       2,
				       sizeof(ml_cc_counter_inference_desc),
				       (uint64_t)ml_cc_counter_inference_desc);
}

/*
 * Handle a ROCE TX event: tracks the RTT-request/response handshake so a fresh RTT sample keeps
 * arriving for the model to consume. This is protocol bookkeeping, not a CC decision -- the rate
 * itself is left untouched here.
 */
static inline void ml_cc_handle_roce_tx(doca_pcc_dev_event_t *event, ml_cc_ctxt_t *ctxt, doca_pcc_dev_results_t *results)
{
	uint8_t rtt_req = 0;
	uint32_t rtt_meas_psn = ctxt->rtt_meas_psn;
	uint32_t timestamp = doca_pcc_dev_get_timestamp(event);
	doca_pcc_dev_event_general_attr_t ev_attr = doca_pcc_dev_get_ev_attr(event);

	if (unlikely((ev_attr.flags & DOCA_PCC_DEV_TX_FLAG_RTT_REQ_SENT) && (rtt_meas_psn == 0))) {
		ctxt->rtt_meas_psn = 1;
		ctxt->rtt_req_to_rtt_sent = 0;
		ctxt->start_delay = timestamp;
	} else {
		uint32_t rtt_till_now = (timestamp - ctxt->start_delay);

		if (unlikely(ctxt->start_delay > timestamp))
			rtt_till_now += UINT32_MAX;
		if (rtt_meas_psn == 0) {
			rtt_till_now = 0;
			ctxt->rtt_req_to_rtt_sent += 1;
		}
		if (unlikely((rtt_till_now > ((uint32_t)ABORT_TIME << ctxt->abort_cnt)) ||
			     (ctxt->rtt_req_to_rtt_sent > 2))) {
			rtt_req = 1;
			if (rtt_till_now > ((uint32_t)ABORT_TIME << ctxt->abort_cnt))
				ctxt->abort_cnt += 1;
			ctxt->rtt_req_to_rtt_sent = 1;
		}
	}

	results->rate = ctxt->cur_rate;
	results->rtt_req = rtt_req;
}

/*
 * Handle a fresh RTT sample: update the tracked flow features and run the random-forest model to
 * decide the new rate. This is the only place in the algorithm that changes the rate from
 * scratch based on a learned decision -- CNP/NACK events only update flags consumed here.
 */
static inline void ml_cc_handle_roce_rtt(doca_pcc_dev_event_t *event,
					 uint32_t *param,
					 ml_cc_ctxt_t *ctxt,
					 uint32_t *counter,
					 doca_pcc_dev_results_t *results)
{
	uint32_t rtt_meas_psn = ctxt->rtt_meas_psn;

	if (unlikely(((rtt_meas_psn == 0) && (ctxt->rtt_req_to_rtt_sent == 0)))) {
		results->rate = ctxt->cur_rate;
		results->rtt_req = 0;
		return;
	}

	ctxt->rtt_meas_psn = 0;
	ctxt->abort_cnt = 0;

	uint32_t start_rtt = doca_pcc_dev_get_rtt_req_send_timestamp(event);
	uint32_t end_rtt = doca_pcc_dev_get_timestamp(event);
	int32_t rtt = end_rtt - start_rtt;

	if (unlikely(end_rtt < start_rtt))
		rtt += UINT32_MAX;
	ctxt->rtt = rtt;

	if ((end_rtt & 0xFF) == 0) {
		/* Periodically reset the running minimum so it tracks a changing environment */
		ctxt->min_rtt = INITIAL_MIN_RTT;
	}

	uint32_t relative_delay = rtt;

	if ((uint32_t)rtt >= ctxt->min_rtt)
		relative_delay = rtt - ctxt->min_rtt;
	else
		ctxt->min_rtt = rtt;

	/* EWMA update: ewma = alpha * rtt + (1 - alpha) * ewma, in fxp16 */
	uint32_t alpha = param[ML_CC_EWMA_ALPHA];

	ctxt->ewma_rtt = doca_pcc_dev_fxp_mult(alpha, (uint32_t)rtt) +
			 doca_pcc_dev_fxp_mult((1 << 16) - alpha, ctxt->ewma_rtt);

	/* Build the feature vector and run the on-device model */
	int32_t feat[ML_CC_NUM_FEATURES];

	feat[ML_FEAT_RELATIVE_RTT] = (int32_t)relative_delay;
	feat[ML_FEAT_EWMA_RTT] = (int32_t)ctxt->ewma_rtt;
	feat[ML_FEAT_CUR_RATE] = (int32_t)ctxt->cur_rate;
	feat[ML_FEAT_WAS_CNP] = ctxt->flags.was_cnp;
	feat[ML_FEAT_WAS_NACK] = ctxt->flags.was_nack;
	feat[ML_FEAT_CNP_STREAK] = ctxt->cnp_streak;

	int32_t rate_delta_fxp16 = ml_cc_forest_infer(feat);

	/* Clamp the model output to +/-100% so a malformed or badly trained forest can never
	 * underflow the unsigned rate subtraction below into a wrapped-around huge rate */
	if (rate_delta_fxp16 > (1 << 16))
		rate_delta_fxp16 = (1 << 16);
	else if (rate_delta_fxp16 < -(1 << 16))
		rate_delta_fxp16 = -(1 << 16);

	if (counter != NULL)
		counter[ML_CC_COUNTER_INFERENCE]++;

	uint32_t cur_rate = ctxt->cur_rate;
	uint32_t new_rate;

	if (rate_delta_fxp16 >= 0)
		new_rate = cur_rate + doca_pcc_dev_fxp_mult(cur_rate, (uint32_t)rate_delta_fxp16);
	else
		new_rate = cur_rate - doca_pcc_dev_fxp_mult(cur_rate, (uint32_t)(-rate_delta_fxp16));

	/* Safety net independent of the model: never let the rate stay high through an
	 * extreme, sustained delay */
	if (rtt >= (int32_t)param[ML_CC_MAX_DELAY])
		new_rate = doca_pcc_dev_fxp_mult(new_rate, (1 << 16) / 2);

	if (new_rate > DOCA_PCC_DEV_MAX_RATE)
		new_rate = DOCA_PCC_DEV_MAX_RATE;
	if (new_rate < param[ML_CC_MIN_RATE])
		new_rate = param[ML_CC_MIN_RATE];

	ctxt->flags.was_cnp = 0;
	ctxt->flags.was_nack = 0;
	ctxt->cur_rate = new_rate;
	ctxt->rtt_req_to_rtt_sent = 1;

	results->rate = new_rate;
	results->rtt_req = 1;
}

static inline void ml_cc_handle_new_flow(doca_pcc_dev_event_t *event,
					 uint32_t *param,
					 ml_cc_ctxt_t *ctxt,
					 doca_pcc_dev_results_t *results)
{
	ctxt->cur_rate = param[ML_CC_NEW_FLOW_RATE];
	ctxt->start_delay = doca_pcc_dev_get_timestamp(event);
	ctxt->rtt_meas_psn = 0;
	ctxt->rtt_req_to_rtt_sent = 1;
	ctxt->abort_cnt = 0;
	ctxt->min_rtt = INITIAL_MIN_RTT;
	ctxt->ewma_rtt = INITIAL_MIN_RTT; /* Same raw-ns scale as rtt, see ml_cc_handle_roce_rtt() */
	ctxt->cnp_streak = 0;
	ctxt->flags.was_cnp = 0;
	ctxt->flags.was_nack = 0;

	results->rate = param[ML_CC_NEW_FLOW_RATE];
	results->rtt_req = 1;
}

void ml_cc_algo(doca_pcc_dev_event_t *event,
		uint32_t *param,
		uint32_t *counter,
		doca_pcc_dev_algo_ctxt_t *algo_ctxt,
		doca_pcc_dev_results_t *results)
{
	ml_cc_ctxt_t *ctxt = (ml_cc_ctxt_t *)algo_ctxt;
	doca_pcc_dev_event_general_attr_t ev_attr = doca_pcc_dev_get_ev_attr(event);
	uint32_t ev_type = ev_attr.ev_type;

	if (unlikely(ctxt->cur_rate == 0)) {
		ml_cc_handle_new_flow(event, param, ctxt, results);
	} else if (ev_type == DOCA_PCC_DEV_EVNT_ROCE_TX) {
		ml_cc_handle_roce_tx(event, ctxt, results);
		if (counter != NULL)
			counter[ML_CC_COUNTER_TX_EVENT]++;
	} else if (ev_type == DOCA_PCC_DEV_EVNT_RTT) {
		ml_cc_handle_roce_rtt(event, param, ctxt, counter, results);
	} else if (ev_type == DOCA_PCC_DEV_EVNT_ROCE_CNP) {
		ctxt->flags.was_cnp = 1;
		if (ctxt->cnp_streak < UINT8_MAX)
			ctxt->cnp_streak++;
		results->rate = ctxt->cur_rate;
		results->rtt_req = 0;
	} else if (ev_type == DOCA_PCC_DEV_EVNT_ROCE_NACK) {
		ctxt->flags.was_nack = 1;
		ctxt->cnp_streak = 0;
		results->rate = ctxt->cur_rate;
		results->rtt_req = 0;
	} else {
		results->rate = ctxt->cur_rate;
		results->rtt_req = 0;
	}
}

doca_pcc_dev_error_t ml_cc_set_algo_params(uint32_t param_id_base,
					   uint32_t param_num,
					   const uint32_t *new_param_values,
					   uint32_t *params)
{
	if ((param_num > ML_CC_PARAM_NUM) || (param_id_base >= ML_CC_PARAM_NUM))
		return DOCA_PCC_DEV_STATUS_FAIL;

	if ((new_param_values == NULL) || (params == NULL))
		return DOCA_PCC_DEV_STATUS_FAIL;

	return DOCA_PCC_DEV_STATUS_OK;
}
