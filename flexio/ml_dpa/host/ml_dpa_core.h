/**
 * Copyright (c) 2022-2026 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#ifndef ML_DPA_CORE_H_
#define ML_DPA_CORE_H_

#include <doca_pcc.h>
#include <doca_dev.h>
#include <doca_error.h>

#define ML_DPA_RP_THREADS_NUM_DEFAULT_VALUE \
	(48 + 1) /* Default Number of PCC RP threads, the extra one is used for communication */
#define WAIT_TIME_DEFAULT_VALUE (-1) /* Wait time - default value (infinity) */
#define ML_DPA_COREDUMP_FILE_DEFAULT_PATH \
	("/tmp/doca_ml_dpa_coredump.txt") /* Default pathname for device coredump file */
#define ML_DPA_PRINT_BUFFER_SIZE_DEFAULT_VALUE (512 * 2048) /* Device print buffer size - default value */
#define MAX_USER_ARG_SIZE (1024)	     /* Maximum size of user input argument */
#define MAX_ARG_SIZE (MAX_USER_ARG_SIZE + 1) /* Maximum size of input argument */

#define LOG_LEVEL_CRIT (20)    /* Critical log level */
#define LOG_LEVEL_ERROR (30)   /* Error log level */
#define LOG_LEVEL_WARNING (40) /* Warning log level */
#define LOG_LEVEL_INFO (50)    /* Info log level */
#define LOG_LEVEL_DEBUG (60)   /* Debug log level */

/*
 * Default PCC RP threads. This is a known-good subset of DPA execution units spanning several
 * clusters. To spread the ML inference workload across every EU the platform exposes (maximum
 * cores/queues), use --dpa-resources together with --dpa-app-key instead of relying on this
 * default list -- see register_ml_dpa_params().
 */
extern const uint32_t default_ml_dpa_rp_threads_list[ML_DPA_RP_THREADS_NUM_DEFAULT_VALUE];

/* Log level */
extern int log_level;

#define PRINT_CRIT(...) \
	do { \
		if (log_level >= LOG_LEVEL_CRIT) \
			printf(__VA_ARGS__); \
	} while (0)

#define PRINT_ERROR(...) \
	do { \
		if (log_level >= LOG_LEVEL_ERROR) \
			printf(__VA_ARGS__); \
	} while (0)

#define PRINT_WARNING(...) \
	do { \
		if (log_level >= LOG_LEVEL_WARNING) \
			printf(__VA_ARGS__); \
	} while (0)

#define PRINT_INFO(...) \
	do { \
		if (log_level >= LOG_LEVEL_INFO) \
			printf(__VA_ARGS__); \
	} while (0)

#define PRINT_DEBUG(...) \
	do { \
		if (log_level >= LOG_LEVEL_DEBUG) \
			printf(__VA_ARGS__); \
	} while (0)

/*
 * DOCA PCC Reaction Point ML Congestion Control DPA program
 */
extern struct doca_pcc_app *ml_dpa_rp_ml_cc_app;

struct ml_dpa_config {
	char device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /* DOCA device name */
	struct doca_pcc_app *app;			 /* Device program */
	uint32_t threads_num;				 /* Number of PCC threads */
	uint32_t threads_list[MAX_ARG_SIZE];		 /* Threads numbers */
	int wait_time;					 /* Wait duration */
	bool remote_sw_handler;				 /* CCMAD probe type remote SW handler flag */
	char coredump_file[MAX_ARG_SIZE];		 /* Coredump file pathname */
	char dpa_resources_file[MAX_ARG_SIZE];		 /* DPA resources yaml file path */
	char dpa_application_key[MAX_ARG_SIZE];	 /* DPA application file name */
};

struct ml_dpa_resources {
	struct doca_dev *doca_device; /* DOCA device */
	struct doca_pcc *doca_pcc;    /* DOCA PCC context */
};

/*
 * Initialize the ML DPA application resources
 *
 * @cfg [in]: ML DPA application user configurations
 * @resources [in/out]: ML DPA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ml_dpa_init(struct ml_dpa_config *cfg, struct ml_dpa_resources *resources);

/*
 * Destroy the ML DPA application resources
 *
 * @resources [in]: ML DPA resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t ml_dpa_destroy(struct ml_dpa_resources *resources);

/*
 * Register the command line parameters for the ML DPA application.
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t register_ml_dpa_params(void);

#endif /* ML_DPA_CORE_H_ */
