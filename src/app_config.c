#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include "app_config.h"

void app_config_set_defaults(struct app_config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->profile = APP_PROFILE_IOT;
    cfg->model = APP_MODEL_MLP;
    cfg->kernel = APP_KERNEL_NEON;
    cfg->log_mode = APP_LOG_NONE;
    cfg->action_mode = APP_ACTION_FORWARD;
    cfg->port_id = 0;
    cfg->burst_size = APP_DEFAULT_BURST;
    cfg->rx_ring_size = APP_DEFAULT_RX_RING;
    cfg->tx_ring_size = APP_DEFAULT_TX_RING;
    cfg->num_mbufs = APP_DEFAULT_MBUFS;
    cfg->mbuf_cache_size = APP_DEFAULT_MBUF_CACHE;
    cfg->max_flows_per_worker = APP_DEFAULT_MAX_FLOWS;
    cfg->max_sensors_per_worker = APP_DEFAULT_MAX_SENSORS;
    cfg->log_ring_size = APP_DEFAULT_LOG_RING;
    cfg->worker_count = 0;
    cfg->window_packets = 16;
    cfg->ics_window_samples = 32;
    cfg->use_main_as_worker = false;
    cfg->swap_mac = false;
    cfg->allow_feature_pad = false;
    cfg->promiscuous = true;
    cfg->print_stats = false;
    snprintf(cfg->log_file, sizeof(cfg->log_file), "smartnic_predictions.csv");
}

static int parse_bool(const char *s, bool *out)
{
    if (!s || !out) return -1;
    if (!strcmp(s, "1") || !strcmp(s, "true") || !strcmp(s, "yes") || !strcmp(s, "on")) {
        *out = true;
        return 0;
    }
    if (!strcmp(s, "0") || !strcmp(s, "false") || !strcmp(s, "no") || !strcmp(s, "off")) {
        *out = false;
        return 0;
    }
    return -1;
}

static int parse_u16(const char *s, uint16_t *out)
{
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!s || *s == 0 || (end && *end != 0) || v > 65535UL) return -1;
    *out = (uint16_t)v;
    return 0;
}

static int parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!s || *s == 0 || (end && *end != 0) || v > 0xffffffffUL) return -1;
    *out = (uint32_t)v;
    return 0;
}

const char *app_profile_name(enum app_profile_type p)
{
    switch (p) {
    case APP_PROFILE_IOT: return "iot";
    case APP_PROFILE_DOH: return "doh";
    case APP_PROFILE_ICS: return "ics";
    default: return "unknown";
    }
}

const char *app_model_name(enum app_model_type m)
{
    switch (m) {
    case APP_MODEL_MLP: return "mlp";
    case APP_MODEL_RF: return "rf";
    case APP_MODEL_BNN: return "bnn";
    default: return "unknown";
    }
}

const char *app_kernel_name(enum app_kernel_type k)
{
    switch (k) {
    case APP_KERNEL_SCALAR: return "scalar";
    case APP_KERNEL_NEON: return "neon";
    default: return "unknown";
    }
}

static int parse_profile(const char *s, enum app_profile_type *p)
{
    if (!strcmp(s, "iot") || !strcmp(s, "tcp")) { *p = APP_PROFILE_IOT; return 0; }
    if (!strcmp(s, "doh") || !strcmp(s, "dns")) { *p = APP_PROFILE_DOH; return 0; }
    if (!strcmp(s, "ics") || !strcmp(s, "modbus")) { *p = APP_PROFILE_ICS; return 0; }
    return -1;
}

static int parse_model(const char *s, enum app_model_type *m)
{
    if (!strcmp(s, "mlp")) { *m = APP_MODEL_MLP; return 0; }
    if (!strcmp(s, "rf") || !strcmp(s, "randomforest")) { *m = APP_MODEL_RF; return 0; }
    if (!strcmp(s, "bnn")) { *m = APP_MODEL_BNN; return 0; }
    return -1;
}

static int parse_kernel(const char *s, enum app_kernel_type *k)
{
    if (!strcmp(s, "scalar")) { *k = APP_KERNEL_SCALAR; return 0; }
    if (!strcmp(s, "neon")) { *k = APP_KERNEL_NEON; return 0; }
    return -1;
}

static int parse_log_mode(const char *s, enum app_log_mode *m)
{
    if (!strcmp(s, "none") || !strcmp(s, "off")) { *m = APP_LOG_NONE; return 0; }
    if (!strcmp(s, "async") || !strcmp(s, "ring")) { *m = APP_LOG_ASYNC; return 0; }
    return -1;
}

static void usage(const char *prog)
{
    printf("Usage: %s [EAL args] -- [APP args]\n", prog);
    printf("APP args:\n");
    printf("  --profile iot|doh|ics          traffic profile/parser\n");
    printf("  --model mlp|rf|bnn             inference model\n");
    printf("  --kernel scalar|neon           MLP kernel, ignored by RF/BNN except fallback\n");
    printf("  --port N                       DPDK port id\n");
    printf("  --workers N                    worker count; 0=auto\n");
    printf("  --window N                     IoT/DoH packet window\n");
    printf("  --ics-window N                 ICS sensor sample window\n");
    printf("  --log-mode none|async          async logger uses a reserved lcore\n");
    printf("  --log-file FILE                CSV file for async logs\n");
    printf("  --drop-anomaly                 drop packets after anomalous prediction\n");
    printf("  --swap-mac 0|1                 swap L2 addresses before TX\n");
    printf("  --main-worker 0|1              allow master lcore to run hot path\n");
    printf("  --allow-feature-pad 0|1        pad feature vectors if model input is larger\n");
    printf("  --promisc 0|1                  promiscuous mode\n");
    printf("  --help                         show this message\n");
}

int app_config_parse(int argc, char **argv, struct app_config *cfg)
{
    static const struct option opts[] = {
        {"profile", required_argument, NULL, 1000},
        {"model", required_argument, NULL, 1001},
        {"kernel", required_argument, NULL, 1002},
        {"port", required_argument, NULL, 1003},
        {"workers", required_argument, NULL, 1004},
        {"window", required_argument, NULL, 1005},
        {"ics-window", required_argument, NULL, 1006},
        {"log-mode", required_argument, NULL, 1007},
        {"log-file", required_argument, NULL, 1008},
        {"drop-anomaly", no_argument, NULL, 1009},
        {"swap-mac", required_argument, NULL, 1010},
        {"main-worker", required_argument, NULL, 1011},
        {"allow-feature-pad", required_argument, NULL, 1012},
        {"promisc", required_argument, NULL, 1013},
        {"max-flows", required_argument, NULL, 1014},
        {"max-sensors", required_argument, NULL, 1015},
        {"log-ring", required_argument, NULL, 1016},
        {"print-stats", no_argument, NULL, 1017},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    int idx;
    optind = 1;
    while ((opt = getopt_long(argc, argv, "h", opts, &idx)) != -1) {
        switch (opt) {
        case 1000:
            if (parse_profile(optarg, &cfg->profile) != 0) return -1;
            break;
        case 1001:
            if (parse_model(optarg, &cfg->model) != 0) return -1;
            break;
        case 1002:
            if (parse_kernel(optarg, &cfg->kernel) != 0) return -1;
            break;
        case 1003:
            if (parse_u16(optarg, &cfg->port_id) != 0) return -1;
            break;
        case 1004:
            if (parse_u16(optarg, &cfg->worker_count) != 0) return -1;
            break;
        case 1005:
            if (parse_u16(optarg, &cfg->window_packets) != 0) return -1;
            break;
        case 1006:
            if (parse_u16(optarg, &cfg->ics_window_samples) != 0) return -1;
            break;
        case 1007:
            if (parse_log_mode(optarg, &cfg->log_mode) != 0) return -1;
            break;
        case 1008:
            snprintf(cfg->log_file, sizeof(cfg->log_file), "%s", optarg);
            break;
        case 1009:
            cfg->action_mode = APP_ACTION_DROP_ANOMALY;
            break;
        case 1010:
            if (parse_bool(optarg, &cfg->swap_mac) != 0) return -1;
            break;
        case 1011:
            if (parse_bool(optarg, &cfg->use_main_as_worker) != 0) return -1;
            break;
        case 1012:
            if (parse_bool(optarg, &cfg->allow_feature_pad) != 0) return -1;
            break;
        case 1013:
            if (parse_bool(optarg, &cfg->promiscuous) != 0) return -1;
            break;
        case 1014:
            if (parse_u32(optarg, &cfg->max_flows_per_worker) != 0) return -1;
            break;
        case 1015:
            if (parse_u32(optarg, &cfg->max_sensors_per_worker) != 0) return -1;
            break;
        case 1016:
            if (parse_u32(optarg, &cfg->log_ring_size) != 0) return -1;
            break;
        case 1017:
            cfg->print_stats = true;
            break;
        case 'h':
            usage(argv[0]);
            return 1;
        default:
            usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

void app_config_print(const struct app_config *cfg)
{
    printf("Application configuration:\n");
    printf("  profile=%s model=%s kernel=%s\n",
           app_profile_name(cfg->profile), app_model_name(cfg->model),
           app_kernel_name(cfg->kernel));
    printf("  port=%u workers=%u burst=%u\n",
           cfg->port_id, cfg->worker_count, cfg->burst_size);
    printf("  window=%u ics_window=%u\n",
           cfg->window_packets, cfg->ics_window_samples);
    printf("  log_mode=%s log_file=%s logs_compile=%d timing_compile=%d contracts_compile=%d\n",
           cfg->log_mode == APP_LOG_ASYNC ? "async" : "none",
           cfg->log_file, APP_ENABLE_LOGS, APP_ENABLE_TIMING, APP_ENABLE_CONTRACTS);
    printf("  action=%s swap_mac=%d main_worker=%d allow_feature_pad=%d\n",
           cfg->action_mode == APP_ACTION_DROP_ANOMALY ? "drop_anomaly" : "forward",
           cfg->swap_mac ? 1 : 0,
           cfg->use_main_as_worker ? 1 : 0,
           cfg->allow_feature_pad ? 1 : 0);
}
