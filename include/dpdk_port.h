#ifndef DPDK_PORT_H
#define DPDK_PORT_H

#include <stdint.h>
#include <rte_mempool.h>
#include "app_config.h"

int app_port_init(uint16_t port_id,
                  struct rte_mempool **rx_pools,
                  uint16_t queues,
                  const struct app_config *cfg);
void app_ports_close(void);

#endif
