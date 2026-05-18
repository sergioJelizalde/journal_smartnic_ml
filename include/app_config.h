#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#define APP_MAX_FEATURES 64
#define APP_MAX_WORKERS 64
#define APP_DEFAULT_BURST 32
#define APP_DEFAULT_MBUFS 8191
#define APP_DEFAULT_MBUF_CACHE 512
#define APP_DEFAULT_RX_RING 1024
#define APP_DEFAULT_TX_RING 1024
#define APP_DEFAULT_MAX_FLOWS 65536
#define APP_DEFAULT_MAX_SENSORS 65536
#define APP_DEFAULT_LOG_RING 8192

#ifndef APP_ENABLE_LOGS
#define APP_ENABLE_LOGS 1
#endif

#ifndef APP_ENABLE_TIMING
#define APP_ENABLE_TIMING 1
#endif

#ifndef APP_ENABLE_CONTRACTS
#define APP_ENABLE_CONTRACTS 1
#endif

enum app_profile_type {
    APP_PROFILE_IOT = 0,
    APP_PROFILE_DOH = 1,
    APP_PROFILE_ICS = 2
};

enum app_model_type {
    APP_MODEL_MLP = 0,
    APP_MODEL_RF = 1,
    APP_MODEL_BNN = 2
};

enum app_kernel_type {
    APP_KERNEL_SCALAR = 0,
    APP_KERNEL_NEON = 1
};

enum app_log_mode {
    APP_LOG_NONE = 0,
    APP_LOG_ASYNC = 1
};

enum app_action_mode {
    APP_ACTION_FORWARD = 0,
    APP_ACTION_DROP_ANOMALY = 1
};

struct app_config {
    enum app_profile_type profile;
    enum app_model_type model;
    enum app_kernel_type kernel;
    enum app_log_mode log_mode;
    enum app_action_mode action_mode;

    uint16_t port_id;
    uint16_t burst_size;
    uint16_t rx_ring_size;
    uint16_t tx_ring_size;
    uint32_t num_mbufs;
    uint32_t mbuf_cache_size;

    uint32_t max_flows_per_worker;
    uint32_t max_sensors_per_worker;
    uint32_t log_ring_size;

    uint16_t worker_count;      /* 0 means auto */
    uint16_t window_packets;    /* IoT and DoH window */
    uint16_t ics_window_samples;

    bool use_main_as_worker;
    bool swap_mac;
    bool allow_feature_pad;
    bool promiscuous;
    bool print_stats;

    char log_file[256];
};

void app_config_set_defaults(struct app_config *cfg);
int app_config_parse(int argc, char **argv, struct app_config *cfg);
const char *app_profile_name(enum app_profile_type p);
const char *app_model_name(enum app_model_type m);
const char *app_kernel_name(enum app_kernel_type k);
void app_config_print(const struct app_config *cfg);

#endif
