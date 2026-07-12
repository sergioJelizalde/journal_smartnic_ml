/*
 * csv_logger.h - Structured CSV output for microbenchmarks
 *
 * Writes timing results in CSV format for easy aggregation and plotting.
 * Each benchmark opens a single CSV file and appends one row per configuration.
 */

#ifndef CSV_LOGGER_H
#define CSV_LOGGER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *filename;  // Path to output CSV
    int header_written;    // Whether header row already exists
} csv_logger_t;

/*
 * csv_logger_init: Initialize logger and write header if file doesn't exist
 *
 * @log: logger instance
 * @filename: path to CSV file (e.g., "../results/x86/optimized_avx.csv")
 * @kernel_name: descriptive name (e.g., "avx", "neon", "xnnpack")
 *
 * Returns 0 on success, -1 on error.
 */
int csv_logger_init(csv_logger_t *log, const char *filename, const char *kernel_name);

/*
 * csv_logger_row: Append a single benchmark result row
 *
 * @log: logger instance
 * @model_size: config string (e.g., "16_8", "128_64_16")
 * @hidden_layers: human-readable architecture (e.g., "16-8", "128-64-16")
 * @avg_latency_ns: mean latency in nanoseconds
 * @std_dev_ns: standard deviation in nanoseconds
 * @min_ns: minimum observed latency
 * @max_ns: maximum observed latency
 * @throughput_kips: 1e3 / avg_latency_ns (inferences per second, thousands)
 * @iterations: total number of samples collected
 *
 * Returns 0 on success, -1 on error.
 */
int csv_logger_row(
    csv_logger_t *log,
    const char *model_size,
    const char *hidden_layers,
    uint64_t avg_latency_ns,
    uint64_t std_dev_ns,
    uint64_t min_ns,
    uint64_t max_ns,
    uint64_t throughput_kips,
    uint64_t iterations
);

/*
 * csv_logger_close: Flush and close the logger
 *
 * Returns 0 on success, -1 on error.
 */
int csv_logger_close(csv_logger_t *log);

#endif // CSV_LOGGER_H
