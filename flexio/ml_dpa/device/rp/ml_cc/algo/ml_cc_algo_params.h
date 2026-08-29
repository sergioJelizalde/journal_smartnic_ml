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

#ifndef ML_CC_ALGO_PARAMS_H_
#define ML_CC_ALGO_PARAMS_H_

/* Configurable algorithm parameters, tunable at runtime through the PPCC access register */
#define EWMA_ALPHA (((1 << 16) * 25) / 100) /* 0.25 in fxp16 - RTT EWMA smoothing factor */
#define NEW_FLOW_RATE (1 << 20)		    /* Rate assigned to a flow on its first packet (fxp20) */
#define MIN_RATE (1 << (20 - 14))	    /* Rate floor regardless of model output (fxp20) */
#define MAX_DELAY (150000)		    /* Absolute RTT (ns) safety net: forces a strong rate cut
					       independently of the model if ever exceeded */
#define INITIAL_MIN_RTT (200000)	    /* Initial value for the per-flow running minimum RTT (ns) */

#define EWMA_ALPHA_MAX (1 << 16)  /* Maximum value of EWMA_ALPHA */
#define RATE_MAX (1 << 20)	  /* Maximum value of rate */

#endif /* ML_CC_ALGO_PARAMS_H_ */
