/*
 * netfound_tokenizer.c
 *
 * See netfound_tokenizer.h for the assumptions/caveats -- read them first,
 * especially the seq/ack anchor logic (#4) and the payload-stripped scope (#1).
 */

#include "netfound_tokenizer.h"
#include <string.h>
#include <rte_byteorder.h>

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static inline void nf_push_token(nf_burst_t *b, uint16_t tok)
{
    if (b->n_tokens < NF_MAX_BURST_LEN - 1) { /* -1 leaves room for CLS at export */
        b->tokens[b->n_tokens++] = tok;
    }
    /* else: burst already has 6 packets worth of tokens (72); extra
     * packets in an already-full burst are dropped, matching
     * `bursts.groupby("burstID").head(BURST_SIZE_LIMIT)` in Tokenize.py */
}

static inline uint8_t nf_tcp_flags_byte(const struct rte_tcp_hdr *tcp)
{
    /* Matches getTcpFields() in 3_field_extraction.cpp bit-for-bit:
     * CWR ECE URG ACK PSH RST SYN FIN, MSB first. */
    uint8_t f = 0;
    uint8_t flags = tcp->tcp_flags;
    if (flags & RTE_TCP_CWR_FLAG) f |= 1 << 7;
    if (flags & RTE_TCP_ECE_FLAG) f |= 1 << 6;
    if (flags & RTE_TCP_URG_FLAG) f |= 1 << 5;
    if (flags & RTE_TCP_ACK_FLAG) f |= 1 << 4;
    if (flags & RTE_TCP_PSH_FLAG) f |= 1 << 3;
    if (flags & RTE_TCP_RST_FLAG) f |= 1 << 2;
    if (flags & RTE_TCP_SYN_FLAG) f |= 1 << 1;
    if (flags & RTE_TCP_FIN_FLAG) f |= 1;
    return f;
}

/* ------------------------------------------------------------------ */
/* Init                                                                 */
/* ------------------------------------------------------------------ */

void nf_flow_state_init(nf_flow_state_t *fs)
{
    memset(fs, 0, sizeof(*fs));
}

/* ------------------------------------------------------------------ */
/* Burst finalize                                                       */
/* ------------------------------------------------------------------ */

void nf_flush_current_burst(nf_flow_state_t *fs)
{
    if (!fs->cur_burst_active || fs->cur_burst.n_pkts == 0)
        return;
    if (fs->n_bursts_done >= NF_MAX_BURSTS) {
        /* Already have 12 bursts (FLOW_SIZE_LIMIT) -- drop, matching
         * `bursts[bursts["burstID"] < FLOW_SIZE_LIMIT]` in Tokenize.py */
        fs->cur_burst_active = false;
        memset(&fs->cur_burst, 0, sizeof(fs->cur_burst));
        return;
    }

    if (!fs->have_last_burst_start) {
        fs->cur_burst.iat_ms = 0; /* first burst has IAT 0, per split_based_on_iat */
    } else {
        uint64_t delta_ns = fs->cur_burst.first_pkt_ts_ns - fs->last_burst_start_ts_ns;
        fs->cur_burst.iat_ms = (int64_t)(delta_ns / 1000000ULL); /* ns -> ms,
            matching the combined /1000 (Tokenize.py) then *1e-3 (tokenizer.py) */
    }
    fs->last_burst_start_ts_ns = fs->cur_burst.first_pkt_ts_ns;
    fs->have_last_burst_start = true;

    fs->bursts[fs->n_bursts_done] = fs->cur_burst;
    fs->n_bursts_done++;

    fs->cur_burst_active = false;
    memset(&fs->cur_burst, 0, sizeof(fs->cur_burst));

    if (fs->n_bursts_done >= NF_MAX_BURSTS)
        fs->ready_for_inference = true;
}

/* ------------------------------------------------------------------ */
/* Per-packet ingestion                                                 */
/* ------------------------------------------------------------------ */

bool nf_process_packet(nf_flow_state_t *fs, struct rte_mbuf *m,
                        uint64_t ts_ns, bool is_forward)
{
    bool completed_a_burst = false;

    struct rte_ipv4_hdr *ip = rte_pktmbuf_mtod_offset(m, struct rte_ipv4_hdr *,
                                                       sizeof(struct rte_ether_hdr));
    /* NOTE: IPv6 not handled in this skeleton -- 3_field_extraction.cpp
     * supports it (IP_ttl<-hopLimit, IP_tos<-trafficClass, etc, with a
     * fixed 40-byte header length). Add an rte_ipv6_hdr branch here
     * mirroring that mapping if you need v6 flows. */

    uint8_t proto = ip->next_proto_id;
    if (!fs->protocol_set) {
        fs->protocol = (nf_proto_t)proto;
        fs->protocol_set = true;
        fs->flow_start_ns = ts_ns;
    } else if ((uint8_t)fs->protocol != proto) {
        /* Tokenize.py's C++ stage hard-errors on mixed protocol within one
         * flow file; here we just refuse to ingest the mismatched packet. */
        return false;
    }

    if (!fs->have_anchor_ip) {
        fs->anchor_src_ip = rte_be_to_cpu_32(ip->src_addr);
        fs->have_anchor_ip = true;
    }

    /* ---- burst-boundary detection (10ms gap, per direction) ---- */
    uint64_t *last_ts = is_forward ? &fs->last_ts_fwd_ns : &fs->last_ts_bwd_ns;
    bool     *have_last = is_forward ? &fs->have_last_fwd : &fs->have_last_bwd;

    bool gap_exceeded = false;
    if (*have_last && ts_ns > *last_ts &&
        (ts_ns - *last_ts) > NF_BURST_GAP_NS) {
        gap_exceeded = true;
    }
    *last_ts = ts_ns;
    *have_last = true;

    bool burst_full = fs->cur_burst_active &&
                       fs->cur_burst.n_pkts >= NF_MAX_PKTS_PER_BURST;

    if (fs->cur_burst_active && (gap_exceeded || burst_full)) {
        nf_flush_current_burst(fs);
        completed_a_burst = true;
    }

    if (!fs->cur_burst_active) {
        memset(&fs->cur_burst, 0, sizeof(fs->cur_burst));
        fs->cur_burst_active = true;
        fs->cur_burst.first_pkt_ts_ns = ts_ns;
        fs->cur_burst.direction_fwd = is_forward;
    }

    if (fs->n_bursts_done >= NF_MAX_BURSTS) {
        /* Already collected the max flow-level bursts; nothing more to do. */
        return completed_a_burst;
    }

    /* ---- shared IP-header fields (5 tokens, per IPFields config) ---- */
    uint8_t  ip_hl  = (ip->version_ihl & 0x0F);              /* IP_hl, 4 bits */
    uint8_t  ip_tos = ip->type_of_service;                    /* IP_tos, 8 bits */
    uint16_t ip_tl  = rte_be_to_cpu_16(ip->total_length);      /* IP_tl, 16 bits */
    uint8_t  ip_flags = (rte_be_to_cpu_16(ip->fragment_offset) >> 13) & 0x7; /* IP_Flags, 3 bits */
    uint8_t  ip_ttl = ip->time_to_live;                        /* IP_ttl, 8 bits */

    nf_push_token(&fs->cur_burst, ip_hl);
    nf_push_token(&fs->cur_burst, ip_tos);
    nf_push_token(&fs->cur_burst, ip_tl);
    nf_push_token(&fs->cur_burst, ip_flags);
    nf_push_token(&fs->cur_burst, ip_ttl);

    fs->cur_burst.byte_sum += ip_tl;

    void *l4 = (uint8_t *)ip + ((ip->version_ihl & 0x0F) * 4);

    if (proto == NF_PROTO_TCP) {
        struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)l4;

        uint8_t  tcp_flags = nf_tcp_flags_byte(tcp);
        uint16_t win       = rte_be_to_cpu_16(tcp->rx_win);
        uint32_t seq       = rte_be_to_cpu_32(tcp->sent_seq);
        uint32_t ack       = rte_be_to_cpu_32(tcp->recv_ack);
        uint16_t urp       = rte_be_to_cpu_16(tcp->tcp_urp);

        /* ---- relative seq/ack normalization, mirroring
         * 3_field_extraction.cpp's per-flow anchoring exactly ---- */
        if (!fs->tcp_seq_anchor_set) {
            fs->tcp_seq_anchor = seq;
            fs->tcp_seq_anchor_set = true;
            /* Mirrors: if first packet's ack != 0 (capture started mid-flow),
             * anchor the reverse direction off it immediately. */
            if (ack != 0) {
                fs->tcp_ack_anchor = ack;
                fs->tcp_ack_anchor_set = true;
            }
        } else if (!fs->tcp_ack_anchor_set) {
            /* 2nd packet of the flow (typically the SYN-ACK): its SEQ is the
             * responder's ISN, used as the reverse-direction anchor. */
            fs->tcp_ack_anchor = seq;
            fs->tcp_ack_anchor_set = true;
        }

        uint32_t rel_seq, rel_ack;
        if (is_forward) {
            rel_seq = seq - fs->tcp_seq_anchor;
            rel_ack = fs->tcp_ack_anchor_set ? (ack - fs->tcp_ack_anchor) : 0;
        } else {
            rel_seq = seq - (fs->tcp_ack_anchor_set ? fs->tcp_ack_anchor : 0);
            rel_ack = ack - fs->tcp_seq_anchor;
        }

        nf_push_token(&fs->cur_burst, tcp_flags);
        nf_push_token(&fs->cur_burst, win);
        nf_push_token(&fs->cur_burst, (uint16_t)(rel_seq >> 16));
        nf_push_token(&fs->cur_burst, (uint16_t)(rel_seq & 0xFFFF));
        nf_push_token(&fs->cur_burst, (uint16_t)(rel_ack >> 16));
        nf_push_token(&fs->cur_burst, (uint16_t)(rel_ack & 0xFFFF));
        nf_push_token(&fs->cur_burst, urp);

    } else if (proto == NF_PROTO_UDP) {
        struct rte_udp_hdr *udp = (struct rte_udp_hdr *)l4;
        uint16_t udp_len = rte_be_to_cpu_16(udp->dgram_len);
        nf_push_token(&fs->cur_burst, udp_len);

    } else if (proto == NF_PROTO_ICMP) {
        struct rte_icmp_hdr *icmp = (struct rte_icmp_hdr *)l4;
        nf_push_token(&fs->cur_burst, icmp->icmp_type);
        nf_push_token(&fs->cur_burst, icmp->icmp_code);
    } else {
        /* Unsupported protocol for this model -- drop, mirroring the
         * hard error in 3_field_extraction.cpp for unknown L4 protocols. */
        return completed_a_burst;
    }

    /* NOTE: no payload tokens -- this targets netFoundNoPayloadConfig
     * (small/base/large all use it). Add 6 more tokens here, from the
     * first 12 raw payload bytes as 6 big-endian uint16s, if you fine-tune
     * a payload-enabled variant instead. */

    fs->cur_burst.n_pkts++;

    return completed_a_burst;
}

/* ------------------------------------------------------------------ */
/* Export to model input tensors                                       */
/* ------------------------------------------------------------------ */

void nf_export_model_input(const nf_flow_state_t *fs, nf_model_input_t *out)
{
    memset(out, 0, sizeof(*out));

    uint8_t n = fs->n_bursts_done;
    out->total_bursts = n;
    out->protocol = (int64_t)fs->protocol;

    for (uint8_t b = 0; b < NF_MAX_BURSTS; b++) {
        int64_t *ids_row  = &out->input_ids[b * NF_MAX_BURST_LEN];
        int64_t *mask_row = &out->attention_mask[b * NF_MAX_BURST_LEN];

        if (b >= n) {
            /* padded burst: all PAD (0) tokens, all-zero attention mask --
             * matches truncate_flow() padding behavior for short flows. */
            continue;
        }

        const nf_burst_t *src = &fs->bursts[b];

        /* CLS token first, matches prepend_to_list(..., bos_token_id) */
        ids_row[0]  = NF_CLS_TOKEN;
        mask_row[0] = 1;

        /* +1 shift on every real token, matches convert_to_tokens(add_one=True)
         * -- this keeps PAD_TOKEN (0) unambiguous. */
        int pos = 1;
        for (int t = 0; t < src->n_tokens && pos < NF_MAX_BURST_LEN; t++, pos++) {
            ids_row[pos]  = (int64_t)src->tokens[t] + 1;
            mask_row[pos] = 1;
        }
        /* remaining positions in this burst's row stay PAD=0 / mask=0 */

        out->direction[b]  = src->direction_fwd ? 1 : -1;
        out->bytes[b]      = src->byte_sum;
        out->iats[b]       = src->iat_ms;
        out->pkt_count[b]  = src->n_pkts;
    }

    if (n > 0) {
        out->flow_duration =
            (int64_t)((fs->bursts[n - 1].first_pkt_ts_ns - fs->flow_start_ns) / 1000000ULL);
    }
}
