#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <rte_ethdev.h>
#include "dpdk_port.h"

static uint8_t rss_key[40] = {
    0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,
    0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,
    0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,
    0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,
    0x6d,0x5a,0x6d,0x5a,0x6d,0x5a,0x6d,0x5a
};

int app_port_init(uint16_t port_id,
                  struct rte_mempool **rx_pools,
                  uint16_t queues,
                  const struct app_config *cfg)
{
    struct rte_eth_dev_info dev_info;
    int ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        printf("rte_eth_dev_info_get(%u) failed: %s\n", port_id, strerror(-ret));
        return ret;
    }

    if (queues > dev_info.max_rx_queues) queues = dev_info.max_rx_queues;
    if (queues > dev_info.max_tx_queues) queues = dev_info.max_tx_queues;
    if (queues == 0) return -EINVAL;

    struct rte_eth_conf port_conf;
    memset(&port_conf, 0, sizeof(port_conf));
    port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
    port_conf.rx_adv_conf.rss_conf.rss_key = rss_key;
    port_conf.rx_adv_conf.rss_conf.rss_key_len = sizeof(rss_key);
    port_conf.rx_adv_conf.rss_conf.rss_hf = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_TCP;
    port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;
    if (port_conf.rx_adv_conf.rss_conf.rss_hf == 0) {
        printf("Warning: requested RSS types not supported; disabling RSS.\n");
        port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    }

    ret = rte_eth_dev_configure(port_id, queues, queues, &port_conf);
    if (ret < 0) return ret;

    uint16_t nb_rxd = cfg->rx_ring_size;
    uint16_t nb_txd = cfg->tx_ring_size;
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rxd, &nb_txd);
    if (ret < 0) return ret;

    struct rte_eth_rxconf rxconf = dev_info.default_rxconf;
    struct rte_eth_txconf txconf = dev_info.default_txconf;

    for (uint16_t q = 0; q < queues; q++) {
        ret = rte_eth_rx_queue_setup(port_id, q, nb_rxd,
                                     rte_eth_dev_socket_id(port_id),
                                     &rxconf, rx_pools[q]);
        if (ret < 0) return ret;
    }
    for (uint16_t q = 0; q < queues; q++) {
        ret = rte_eth_tx_queue_setup(port_id, q, nb_txd,
                                     rte_eth_dev_socket_id(port_id),
                                     &txconf);
        if (ret < 0) return ret;
    }

    ret = rte_eth_dev_start(port_id);
    if (ret < 0) return ret;
    if (cfg->promiscuous) rte_eth_promiscuous_enable(port_id);

    printf("Port %u initialized with %u RX/TX queues. RSS mask=0x%" PRIx64 "\n",
           port_id, queues, port_conf.rx_adv_conf.rss_conf.rss_hf);
    return 0;
}

void app_ports_close(void)
{
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        printf("Closing port %u...\n", portid);
        int ret = rte_eth_dev_stop(portid);
        if (ret != 0) printf("rte_eth_dev_stop(%u): %s\n", portid, strerror(-ret));
        rte_eth_dev_close(portid);
    }
}
