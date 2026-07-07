/*
 * netfound_tokenizer.h
 *
 * Online (packet-at-a-time) re-implementation of the netFound preprocessing
 * pipeline (pre_process_src/3_field_extraction.cpp + Tokenize.py), for
 * inline use in a DPDK RX path feeding ONNX Runtime.
 *
 * Layout verified against:
 *   netFound/configs/DefaultConfigNoTCPOptions.json
 *   netFound/src/modules/netFoundConfigBase.py  (netFoundNoPayloadConfig)
 *   netFound/src/modules/netFoundTokenizer.py
 *
 * IMPORTANT ASSUMPTIONS (read before trusting this against the reference):
 *   1. Targets the *no-payload* configs (small/base/large all use
 *      netFoundNoPayloadConfig) -> payload tokens are NOT emitted.
 *      If you fine-tune a payload-enabled variant, you need to add the
 *      12-byte payload capture back in (see NETFOUND_PAYLOAD_BYTES below,
 *      currently unused).
 *   2. TCP options are NOT supported (DefaultConfigNoTCPOptions.json path).
 *   3. Burst boundary = >10ms gap between consecutive packets in the SAME
 *      direction (matches BURST_SPLIT_BORDER = 10,000,000 ns in Tokenize.py).
 *   4. Relative TCP seq/ack normalization replicates 3_field_extraction.cpp's
 *      online logic (anchor = first packet's seq; ack anchor = first
 *      nonzero ack, or the 2nd packet's seq if the 1st packet's ack was 0 -
 *      i.e. capture starts at the SYN). This has NOT been byte-validated
 *      against the reference offline pipeline on live traffic yet -- do
 *      that before trusting model outputs (see accompanying README).
 *   5. This module produces tokens; it does NOT run ONNX Runtime. It hands
 *      you a finished int64 tensor per flow window that you feed to your
 *      existing ORT session (same pattern as your DomURL/DistilBERT path).
 */

#ifndef NETFOUND_TOKENIZER_H
#define NETFOUND_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_icmp.h>
#include <rte_mbuf.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Geometry, from netFoundConfigBase.py / netFoundNoPayloadConfig ---- */
#define NF_MAX_BURSTS           12      /* max_bursts */
#define NF_MAX_PKTS_PER_BURST    6      /* BURST_SIZE_LIMIT in Tokenize.py */
#define NF_MAX_BURST_LEN        73      /* 6*12 + 1 CLS  (max_burst_length) */

/* Per-packet token counts (no payload), from PROTOCOLS_LENGTH_WITHOUT_PAYLOAD
 * in netFoundTokenizer.py, cross-checked against DefaultConfigNoTCPOptions.json */
#define NF_TOKENS_PER_PKT_TCP   12      /* 5 IP fields + 7 TCP fields */
#define NF_TOKENS_PER_PKT_UDP    6      /* 5 IP fields + 1 UDP field  */
#define NF_TOKENS_PER_PKT_ICMP   7      /* 5 IP fields + 2 ICMP fields */

#define NF_BURST_GAP_NS   10000000ULL  /* 10ms, BURST_SPLIT_BORDER */

/* Special token ids, from netFoundTokenizer.py */
#define NF_CLS_TOKEN   65537
#define NF_PAD_TOKEN   0
#define NF_VOCAB_MAX   65536            /* raw token value range is 0..65535,
                                            +1 is added at export time so PAD=0
                                            never collides with a real token */

typedef enum {
    NF_PROTO_TCP  = 6,
    NF_PROTO_UDP  = 17,
    NF_PROTO_ICMP = 1,
} nf_proto_t;

/* One finalized burst's worth of state, pre-flattening into tokens */
typedef struct {
    uint16_t tokens[NF_MAX_BURST_LEN];   /* raw field tokens, NOT yet +1/CLS-prepended */
    uint8_t  n_tokens;                   /* tokens actually written (<= 72, i.e. 6*12) */
    uint8_t  n_pkts;                     /* packets folded into this burst (<=6) */
    uint32_t byte_sum;                   /* sum of IP total length over burst pkts */
    uint64_t first_pkt_ts_ns;            /* ts of burst's first packet (abs, ns) */
    int64_t  iat_ms;                     /* gap since previous burst START, ms (0 for burst 0) */
    bool     direction_fwd;              /* true = same direction as flow's first packet */
} nf_burst_t;

/* Per-flow rolling state, keyed by 5-tuple in your existing flow table */
typedef struct {
    /* direction anchor */
    bool     have_anchor_ip;
    uint32_t anchor_src_ip;              /* src IP of the very first packet seen */

    /* TCP relative seq/ack normalization anchors (see assumption #4 above) */
    bool     tcp_seq_anchor_set;
    uint32_t tcp_seq_anchor;             /* forward-direction base SEQ */
    bool     tcp_ack_anchor_set;
    uint32_t tcp_ack_anchor;             /* reverse-direction base SEQ (via ACK) */

    /* per-direction last-packet time, for burst-gap detection */
    uint64_t last_ts_fwd_ns;
    bool     have_last_fwd;
    uint64_t last_ts_bwd_ns;
    bool     have_last_bwd;

    /* burst currently being accumulated */
    nf_burst_t cur_burst;
    bool       cur_burst_active;
    uint64_t   last_burst_start_ts_ns;
    bool       have_last_burst_start;

    /* finished bursts, ready to export */
    nf_burst_t bursts[NF_MAX_BURSTS];
    uint8_t    n_bursts_done;

    uint64_t flow_start_ns;
    nf_proto_t protocol;
    bool       protocol_set;
    bool       ready_for_inference;      /* true once n_bursts_done == NF_MAX_BURSTS,
                                             or you decide to flush early on FIN/timeout */
} nf_flow_state_t;

/* Flattened tensors ready to hand to your ORT session, matching the shapes
 * netFoundTokenizer.tokenize() produces for a single flow (batch size 1).
 * All arrays are length NF_MAX_BURSTS * NF_MAX_BURST_LEN = 876 for input_ids/
 * attention_mask, and length NF_MAX_BURSTS for the per-burst aux tensors. */
typedef struct {
    int64_t input_ids[NF_MAX_BURSTS * NF_MAX_BURST_LEN];
    int64_t attention_mask[NF_MAX_BURSTS * NF_MAX_BURST_LEN];
    int64_t direction[NF_MAX_BURSTS];
    int64_t bytes[NF_MAX_BURSTS];
    int64_t iats[NF_MAX_BURSTS];
    int64_t pkt_count[NF_MAX_BURSTS];
    int64_t total_bursts;
    int64_t protocol;
    int64_t flow_duration;   /* ms. NOTE: verify in netFoundFinetuning.py whether
                                the finetuning head actually consumes this input
                                before wiring it into your ORT feed -- some heads
                                only use input_ids/attention_mask/metadata. */
} nf_model_input_t;

/* --- API --- */

/* Zero-initialize a fresh per-flow state. Call when you create a new flow
 * table entry (e.g. on SYN, or first packet of a UDP/ICMP flow). */
void nf_flow_state_init(nf_flow_state_t *fs);

/* Feed one packet (already identified as belonging to this flow) into the
 * tokenizer state machine. `is_forward` = whether this packet's src IP
 * matches the flow's anchor IP (you likely already compute this for your
 * existing 5-tuple hash direction logic).
 * Returns true if this call completed a burst (i.e. cur_burst was finalized
 * into fs->bursts[]) -- useful if you want to trigger sliding-window
 * inference rather than waiting for all 12 bursts. */
bool nf_process_packet(nf_flow_state_t *fs, struct rte_mbuf *m,
                        uint64_t ts_ns, bool is_forward);

/* Force-finalize whatever burst is in progress (call on flow teardown/FIN,
 * or on an inactivity timeout, so short flows still produce output). */
void nf_flush_current_burst(nf_flow_state_t *fs);

/* Flatten fs->bursts[] into the exact tensors netFoundTokenizer.tokenize()
 * would produce (CLS-prepend, +1 shift, right-padding with PAD=0).
 * Safe to call once ready_for_inference is true, or after a manual flush. */
void nf_export_model_input(const nf_flow_state_t *fs, nf_model_input_t *out);

#ifdef __cplusplus
}
#endif

#endif /* NETFOUND_TOKENIZER_H */
