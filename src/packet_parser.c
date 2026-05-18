#include <string.h>
#include <netinet/in.h>
#include <rte_byteorder.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include "packet_parser.h"

int packet_parse_ipv4_tcp(struct rte_mbuf *m, struct packet_view *out)
{
    memset(out, 0, sizeof(*out));
    uint32_t pkt_len = rte_pktmbuf_pkt_len(m);
    if (unlikely(pkt_len < sizeof(struct rte_ether_hdr))) return 0;

    uint8_t *base = rte_pktmbuf_mtod(m, uint8_t *);
    struct rte_ether_hdr *eth = (struct rte_ether_hdr *)base;
    uint16_t ether_type = rte_be_to_cpu_16(eth->ether_type);
    uint32_t l2_len = sizeof(struct rte_ether_hdr);

    /* Support one or two VLAN tags. */
    for (int i = 0; i < 2; i++) {
        if (ether_type == RTE_ETHER_TYPE_VLAN || ether_type == RTE_ETHER_TYPE_QINQ) {
            if (pkt_len < l2_len + sizeof(struct rte_vlan_hdr)) return 0;
            struct rte_vlan_hdr *vh = (struct rte_vlan_hdr *)(base + l2_len);
            ether_type = rte_be_to_cpu_16(vh->eth_proto);
            l2_len += sizeof(struct rte_vlan_hdr);
        }
    }

    if (ether_type != RTE_ETHER_TYPE_IPV4) return 0;
    if (pkt_len < l2_len + sizeof(struct rte_ipv4_hdr)) return 0;

    struct rte_ipv4_hdr *ip4 = (struct rte_ipv4_hdr *)(base + l2_len);
    uint8_t ihl = ip4->version_ihl & RTE_IPV4_HDR_IHL_MASK;
    uint32_t ip_hlen = (uint32_t)ihl * RTE_IPV4_IHL_MULTIPLIER;
    if (ip_hlen < sizeof(struct rte_ipv4_hdr)) return 0;
    if (pkt_len < l2_len + ip_hlen) return 0;
    if (ip4->next_proto_id != IPPROTO_TCP) return 0;

    uint32_t tcp_off = l2_len + ip_hlen;
    if (pkt_len < tcp_off + sizeof(struct rte_tcp_hdr)) return 0;

    struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)(base + tcp_off);
    uint32_t tcp_hlen = (uint32_t)((tcp->data_off >> 4) & 0x0f) * 4;
    if (tcp_hlen < sizeof(struct rte_tcp_hdr)) return 0;
    if (pkt_len < tcp_off + tcp_hlen) return 0;

    uint32_t payload_off = tcp_off + tcp_hlen;
    out->eth = eth;
    out->ip4 = ip4;
    out->tcp = tcp;
    out->payload = base + payload_off;
    out->packet_len = pkt_len;
    out->l2_len = l2_len;
    out->ip_hlen = ip_hlen;
    out->tcp_hlen = tcp_hlen;
    out->payload_len = (pkt_len > payload_off) ? (pkt_len - payload_off) : 0;
    out->src_ip = rte_be_to_cpu_32(ip4->src_addr);
    out->dst_ip = rte_be_to_cpu_32(ip4->dst_addr);
    out->src_port = rte_be_to_cpu_16(tcp->src_port);
    out->dst_port = rte_be_to_cpu_16(tcp->dst_port);
    out->proto = ip4->next_proto_id;
    out->tcp_flags = tcp->tcp_flags;
    return 1;
}
