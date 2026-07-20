/*
 * csv_logger.c - Structured CSV output for microbenchmarks
 */

#include "csv_logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int csv_logger_init(csv_logger_t *log, const char *filename, const char *kernel_name) {
    if (!log || !filename) {
        return -1;
    }

    log->filename = filename;
    log->header_written = 0;

    // Check if file exists (try to open in read mode)
    FILE *fp = fopen(filename, "r");
    if (fp) {
        // File exists, header already written
        fclose(fp);
        log->header_written = 1;
    } else {
        // File doesn't exist, will write header
        fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "ERROR: Failed to open %s for writing\n", filename);
            return -1;
        }

        // Write CSV header
        fprintf(fp, "model_size,hidden_layers,avg_latency_ns,std_dev_ns,min_ns,max_ns,throughput_kips,iterations,kernel,timestamp\n");
        fclose(fp);
        log->header_written = 1;
    }

    return 0;
}

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
) {
    if (!log || !log->filename || !model_size || !hidden_layers) {
        return -1;
    }

    FILE *fp = fopen(log->filename, "a");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open %s for appending\n", log->filename);
        return -1;
    }

    // Get current timestamp (ISO 8601 format)
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);

    // Extract kernel name from #define or hardcode
    const char *kernel = "unknown";
    #ifdef KERNEL_SCALAR
        kernel = "scalar";
    #elif defined(KERNEL_AVX)
        kernel = "avx";
    #elif defined(KERNEL_NEON)
        kernel = "neon";
    #elif defined(KERNEL_XNNPACK)
        kernel = "xnnpack";
    #endif

    // Append row: one value per column
    fprintf(fp, "%s,%s,%lu,%lu,%lu,%lu,%lu,%lu,%s,%s\n",
            model_size,
            hidden_layers,
            avg_latency_ns,
            std_dev_ns,
            min_ns,
            max_ns,
            throughput_kips,
            iterations,
            kernel,
            timestamp
    );

    fclose(fp);
    return 0;
}

int csv_logger_close(csv_logger_t *log) {
    // Nothing to do in this simple implementation
    // (already flushed on each row)
    if (log) {
        log->filename = NULL;
    }
    return 0;
}
