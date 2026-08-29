/*
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

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <ctype.h>

#include <doca_argp.h>

#include "ml_dpa_core.h"

/*
 * Formats of the trace message to be printed from the device
 */
static char *trace_message_formats[] = {
	"format 0 - user init: port num = %#lx, algo index = %#lx, algo slot = %#lx, algo enable = %#lx, disable event bitmask = %#lx\n",
	"format 1 - user algo: algo slot = %#lx, result rate = %#lx, result rtt req = %#lx, port num = %#lx, timestamp = %#lx\n",
	NULL};

/* Default PCC RP threads */
const uint32_t default_ml_dpa_rp_threads_list[ML_DPA_RP_THREADS_NUM_DEFAULT_VALUE] = {
	176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 192, 193, 194, 195, 196,
	197, 198, 199, 200, 201, 202, 203, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
	218, 219, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 240};

/*
 * Declare threads list flag
 */
static bool use_threads_list = false;

/*
 * Declare DPA resources flag
 */
static bool use_dpa_resources = false;

/*
 * Declare application key flag
 */
static bool use_dpa_application_key = false;

/**
 * @brief Get the size of a file
 *
 * @param[in] path - Path to the file
 * @param[out] file_size - Size of the file in bytes
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t get_file_size(const char *path, size_t *file_size)
{
	FILE *file;
	long nb_file_bytes;

	file = fopen(path, "rb");
	if (file == NULL)
		return DOCA_ERROR_NOT_FOUND;

	if (fseek(file, 0, SEEK_END) != 0) {
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	nb_file_bytes = ftell(file);
	fclose(file);

	if (nb_file_bytes == -1)
		return DOCA_ERROR_IO_FAILED;

	if (nb_file_bytes == 0)
		return DOCA_ERROR_INVALID_VALUE;

	*file_size = (size_t)nb_file_bytes;
	return DOCA_SUCCESS;
}

/**
 * @brief Read file content into a pre-allocated buffer
 *
 * @param[in] path - Path to the file
 * @param[out] buffer - Pre-allocated buffer to store file content
 * @param[out] bytes_read - Number of bytes read from the file
 * @return DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t read_file_into_buffer(const char *path, char *buffer, size_t *bytes_read)
{
	FILE *file;
	size_t read_byte_count;

	file = fopen(path, "rb");
	if (file == NULL)
		return DOCA_ERROR_NOT_FOUND;

	read_byte_count = fread(buffer, 1, *bytes_read, file);
	fclose(file);

	if (read_byte_count != *bytes_read)
		return DOCA_ERROR_IO_FAILED;

	*bytes_read = read_byte_count;
	return DOCA_SUCCESS;
}

/*
 * Check if the provided device name is a name of a valid IB device
 *
 * @device_name [in]: The wanted IB device name
 * @return: True if device_name is an IB device, false otherwise.
 */
static bool pcc_device_exists_check(const char *device_name)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs = 0;
	doca_error_t result;
	bool exists = false;
	char ibdev_name[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
	uint32_t i = 0;

	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to load DOCA devices list: %s\n", doca_error_get_descr(result));
		return false;
	}

	/* Search device with same device name */
	for (i = 0; i < nb_devs; i++) {
		result = doca_devinfo_get_ibdev_name(dev_list[i], ibdev_name, sizeof(ibdev_name));
		if (result != DOCA_SUCCESS)
			continue;

		/* Check if we found the device with the wanted name */
		if (strncmp(device_name, ibdev_name, DOCA_DEVINFO_IBDEV_NAME_SIZE) == 0) {
			exists = true;
			break;
		}
	}

	doca_devinfo_destroy_list(dev_list);

	return exists;
}

/*
 * Open DOCA device that supports PCC Reaction Point role
 *
 * @device_name [in]: Requested IB device name
 * @doca_device [out]: An allocated DOCA device that supports PCC on success and NULL otherwise
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t open_pcc_device(const char *device_name, struct doca_dev **doca_device)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs = 0;
	doca_error_t result;
	char ibdev_name[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
	uint32_t i = 0;

	result = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to load DOCA devices list: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Search device with same device name and PCC capabilities supported */
	for (i = 0; i < nb_devs; i++) {
		result = doca_devinfo_get_ibdev_name(dev_list[i], ibdev_name, sizeof(ibdev_name));
		if (result != DOCA_SUCCESS) {
			PRINT_ERROR("Error: could not get DOCA device name\n");
			continue;
		}

		/* Check if the device has the requested device name */
		if (strncmp(device_name, ibdev_name, DOCA_DEVINFO_IBDEV_NAME_SIZE) != 0)
			continue;

		result = doca_devinfo_get_is_pcc_supported(dev_list[i]);
		if (result != DOCA_SUCCESS) {
			doca_devinfo_destroy_list(dev_list);
			PRINT_ERROR("Error: DOCA device %s does not support PCC RP role\n", device_name);
			return result;
		}

		result = doca_dev_open(dev_list[i], doca_device);
		if (result != DOCA_SUCCESS) {
			doca_devinfo_destroy_list(dev_list);
			PRINT_ERROR("Error: Failed to open DOCA device: %s\n", doca_error_get_descr(result));
			return result;
		}
		break;
	}

	doca_devinfo_destroy_list(dev_list);

	if (*doca_device == NULL) {
		PRINT_ERROR("Error: Couldn't get DOCA device %s\n", device_name);
		return DOCA_ERROR_NOT_FOUND;
	}

	return result;
}

/*
 * Build the PCC thread list from every execution unit listed in a DPA resources file. This is
 * the supported way to spread the ML inference workload across all DPA cores (and their
 * per-thread hardware queues) the platform exposes, rather than a curated subset.
 */
static doca_error_t create_dpa_resources(struct ml_dpa_config *cfg)
{
	char *file_buffer;
	size_t bytes_read;
	struct doca_pcc_resources *doca_pcc_resources;
	doca_error_t status;
	const char *app_key = cfg->dpa_application_key;
	uint32_t num_eus;

	/* Get the file size first */
	status = get_file_size(cfg->dpa_resources_file, &bytes_read);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to get DPA resources file size: %s\n", doca_error_get_descr(status));
		return status;
	}

	/* Allocate buffer based on file size */
	file_buffer = (char *)malloc(bytes_read);
	if (file_buffer == NULL) {
		PRINT_ERROR("Error: Failed to allocate memory for DPA resources file\n");
		return DOCA_ERROR_NO_MEMORY;
	}

	/* Read the DPA resources file */
	status = read_file_into_buffer(cfg->dpa_resources_file, file_buffer, &bytes_read);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to open DPA resources file: %s\n", doca_error_get_descr(status));
		free(file_buffer);
		return status;
	}

	status = doca_pcc_resources_create(app_key, file_buffer, bytes_read, &doca_pcc_resources);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to create DPA resources: %s", doca_error_get_descr(status));
		free(file_buffer);
		return status;
	}

	status = doca_pcc_resources_get_num_eus(doca_pcc_resources, &num_eus);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to get number of execution units: %s", doca_error_get_descr(status));
		doca_pcc_resources_destroy(doca_pcc_resources);
		free(file_buffer);
		return status;
	}

	uint32_t eus[num_eus];
	status = doca_pcc_resources_get_eus(doca_pcc_resources, num_eus, eus);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to get execution units: %s", doca_error_get_descr(status));
		doca_pcc_resources_destroy(doca_pcc_resources);
		free(file_buffer);
		return status;
	}

	/* Print information about the execution units */
	PRINT_DEBUG("Debug: Found %d execution units in DPA resources file\n", num_eus);

	for (uint32_t i = 0; i < num_eus; i++) {
		cfg->threads_list[i] = eus[i];
	}
	cfg->threads_num = num_eus;

	status = doca_pcc_resources_destroy(doca_pcc_resources);
	if (status != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to destroy DPA resources: %s", doca_error_get_descr(status));
		free(file_buffer);
		return status;
	}
	free(file_buffer);

	return DOCA_SUCCESS;
}

doca_error_t ml_dpa_init(struct ml_dpa_config *cfg, struct ml_dpa_resources *resources)
{
	doca_error_t result, tmp_result;
	uint32_t min_num_threads, max_num_threads;

	/* Check if both threads list and DPA resources are specified */
	if (use_dpa_resources && use_threads_list) {
		PRINT_ERROR(
			"Error: Cannot specify both threads list and DPA resources. Use either threads list or DPA resources (with application key).\n");
		return DOCA_ERROR_BAD_CONFIG;
	}

	/* If DPA resources are specified, read the DPA resources file */
	if (use_dpa_resources) {
		if (!use_dpa_application_key) {
			PRINT_ERROR("Error: when using DPA resources file, DPA application key must be provided\n");
			return DOCA_ERROR_BAD_CONFIG;
		}
		result = create_dpa_resources(cfg);
		if (result != DOCA_SUCCESS) {
			PRINT_ERROR("Failed to create DPA resources: %s\n", doca_error_get_descr(result));
			return result;
		}
	}

	/* Open DOCA device that supports PCC RP role */
	result = open_pcc_device(cfg->device_name, &(resources->doca_device));
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to open DOCA device that supports PCC\n");
		return result;
	}

	/* Create DOCA PCC context */
	bool use_default_threads = !use_threads_list && !use_dpa_resources;
	result = doca_pcc_create(resources->doca_device, &(resources->doca_pcc));
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create DOCA PCC context\n");
		goto close_doca_dev;
	}

	/* Fall back to the known-good default thread subset if neither an explicit list nor a
	 * DPA resources file was given */
	if (use_default_threads) {
		memcpy(cfg->threads_list, default_ml_dpa_rp_threads_list, sizeof(default_ml_dpa_rp_threads_list));
		cfg->threads_num = ML_DPA_RP_THREADS_NUM_DEFAULT_VALUE;
	}

	result = doca_pcc_get_min_num_threads(resources->doca_pcc, &min_num_threads);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to get minimum DOCA PCC number of threads\n");
		goto destroy_pcc;
	}

	result = doca_pcc_get_max_num_threads(resources->doca_pcc, &max_num_threads);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Failed to get maximum DOCA PCC number of threads\n");
		goto destroy_pcc;
	}

	if (cfg->threads_num < min_num_threads || cfg->threads_num > max_num_threads) {
		PRINT_ERROR(
			"Invalid number of PCC threads: %u. The Minimum number of PCC threads is %d and the maximum number of PCC threads is %d\n",
			cfg->threads_num,
			min_num_threads,
			max_num_threads);
		result = DOCA_ERROR_INVALID_VALUE;
		goto destroy_pcc;
	}

	result = doca_pcc_set_app(resources->doca_pcc, cfg->app);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set DOCA PCC app\n");
		goto destroy_pcc;
	}

	/* Set DOCA PCC thread affinity -- this is what fans the ML inference workload out across
	 * many DPA cores/queues; see create_dpa_resources() for how to use every available EU. */
	result = doca_pcc_set_thread_affinity(resources->doca_pcc, cfg->threads_num, cfg->threads_list);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set thread affinity for DOCA PCC\n");
		goto destroy_pcc;
	}

	/* CCMAD is the only probe packet format this app uses */
	result = doca_pcc_set_ccmad_probe_packet_format(resources->doca_pcc, 0);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set CCMAD probe packet format for DOCA PCC\n");
		goto destroy_pcc;
	}
	result = doca_pcc_rp_set_ccmad_remote_sw_handler(resources->doca_pcc, 0, cfg->remote_sw_handler);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set CCMAD remote SW handler for DOCA PCC\n");
		goto destroy_pcc;
	}

	/* Set DOCA PCC print buffer size */
	result = doca_pcc_set_print_buffer_size(resources->doca_pcc, ML_DPA_PRINT_BUFFER_SIZE_DEFAULT_VALUE);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set print buffer size for DOCA PCC\n");
		goto destroy_pcc;
	}

	/* Set DOCA PCC trace message formats */
	result = doca_pcc_set_trace_message(resources->doca_pcc, trace_message_formats);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set trace message for DOCA PCC\n");
		goto destroy_pcc;
	}

	/* Set DOCA PCC coredump file pathname */
	result = doca_pcc_set_dev_coredump_file(resources->doca_pcc, cfg->coredump_file);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to set coredump file for DOCA PCC\n");
		goto destroy_pcc;
	}

	return result;

destroy_pcc:
	tmp_result = doca_pcc_destroy(resources->doca_pcc);
	if (tmp_result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to destroy DOCA PCC context: %s\n", doca_error_get_descr(result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
close_doca_dev:
	tmp_result = doca_dev_close(resources->doca_device);
	if (tmp_result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to close DOCA device: %s\n", doca_error_get_descr(result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

	return result;
}

doca_error_t ml_dpa_destroy(struct ml_dpa_resources *resources)
{
	doca_error_t result, tmp_result;

	result = doca_pcc_destroy(resources->doca_pcc);
	if (result != DOCA_SUCCESS)
		PRINT_ERROR("Error: Failed to destroy DOCA PCC context: %s\n", doca_error_get_descr(result));

	tmp_result = doca_dev_close(resources->doca_device);
	if (tmp_result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to close DOCA device: %s\n", doca_error_get_descr(result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}

	return result;
}

/*
 * ARGP Callback - Handle IB device name parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t device_name_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	char *device_name = (char *)param;
	int len;

	len = strnlen(device_name, DOCA_DEVINFO_IBDEV_NAME_SIZE);
	if (len == DOCA_DEVINFO_IBDEV_NAME_SIZE) {
		PRINT_ERROR("Error: Entered IB device name exceeding the maximum size of %d\n",
			    DOCA_DEVINFO_IBDEV_NAME_SIZE - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}
	strncpy(ml_dpa_cfg->device_name, device_name, len + 1);

	if (!pcc_device_exists_check(ml_dpa_cfg->device_name)) {
		PRINT_ERROR("Error: Entered IB device name: %s doesn't exist\n", ml_dpa_cfg->device_name);
		return DOCA_ERROR_INVALID_VALUE;
	}

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle PCC threads list parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t threads_list_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	char *threads_list_string = (char *)param;
	static const char delim[2] = " ";
	char *curr_pcc_string;
	int curr_pcc_check, i, len;
	uint32_t curr_pcc;

	len = strnlen(threads_list_string, MAX_ARG_SIZE);
	if (len == MAX_ARG_SIZE) {
		PRINT_ERROR("Error: Entered PCC threads list exceeded buffer size: %d\n", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	use_threads_list = true;
	ml_dpa_cfg->threads_num = 0;

	/* Check and fill out the PCC threads list */
	/* Get the first PCC thread number */
	curr_pcc_string = strtok(threads_list_string, delim);
	if (curr_pcc_string == NULL) {
		PRINT_ERROR("Error: Invalid PCC threads list: %s\n", threads_list_string);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Walk through rest of the PCC threads numbers */
	while (curr_pcc_string != NULL) {
		/* Check if it's a number by checking its digits */
		len = strnlen(threads_list_string, MAX_ARG_SIZE);
		for (i = 0; i < len; i++) {
			if (!isdigit(curr_pcc_string[i])) {
				PRINT_ERROR("Error: Invalid PCC thread number: %s\n", curr_pcc_string);
				return DOCA_ERROR_INVALID_VALUE;
			}
		}

		/* Convert to integer to check if it is non-negative */
		curr_pcc_check = (int)atoi(curr_pcc_string);
		if (curr_pcc_check < 0) {
			PRINT_ERROR("Error: Invalid PCC thread number %d. PCC threads numbers must be non-negative\n",
				    curr_pcc_check);
			return DOCA_ERROR_INVALID_VALUE;
		}

		curr_pcc = (uint32_t)atoi(curr_pcc_string);
		ml_dpa_cfg->threads_list[ml_dpa_cfg->threads_num++] = curr_pcc;
		curr_pcc_string = strtok(NULL, delim);
	}

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle PCC wait time parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t wait_time_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	int wait_time = *((int *)param);

	/* Wait time must be either positive or infinity (meaning -1 )*/
	if (wait_time == 0) {
		PRINT_ERROR(
			"Error: Entered wait time can't be zero. Must be either positive or infinity (meaning negative value)\n");
		return DOCA_ERROR_INVALID_VALUE;
	}

	ml_dpa_cfg->wait_time = wait_time;

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle PCC remote SW handler parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t ccmad_remote_sw_handler_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;

	ml_dpa_cfg->remote_sw_handler = *((bool *)param);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handle PCC device coredump file parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t coredump_file_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	const char *path = (char *)param;

	int path_len = strnlen(path, MAX_ARG_SIZE);
	if (path_len == MAX_ARG_SIZE) {
		PRINT_ERROR("Entered path exceeded buffer size: %d\n", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* The string will be '\0' terminated due to the strnlen check above */
	strncpy(ml_dpa_cfg->coredump_file, path, path_len + 1);

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handles DPA resources file path parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t dpa_resources_file_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	const char *path = (char *)param;

	int path_len = strnlen(path, MAX_ARG_SIZE);
	if (path_len == MAX_ARG_SIZE) {
		PRINT_ERROR("Error: Entered path exceeded buffer size: %d\n", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	strncpy(ml_dpa_cfg->dpa_resources_file, path, path_len + 1);

	/* Check if the DPA resources file exists */
	if (path_len > 0) {
		FILE *file = fopen(path, "r");
		if (file == NULL) {
			PRINT_ERROR("Error: DPA resources file '%s' does not exist or cannot be accessed\n", path);
			return DOCA_ERROR_NOT_FOUND;
		}
		fclose(file);
		use_dpa_resources = true;
	}

	return DOCA_SUCCESS;
}

/*
 * ARGP Callback - Handles DPA application key parameter
 *
 * @param [in]: Input parameter
 * @config [in/out]: Program configuration context
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t dpa_application_key_callback(void *param, void *config)
{
	struct ml_dpa_config *ml_dpa_cfg = (struct ml_dpa_config *)config;
	const char *app_key = (char *)param;

	int dpa_app_key_len = strnlen(app_key, MAX_ARG_SIZE);
	if (dpa_app_key_len == MAX_ARG_SIZE) {
		PRINT_ERROR("Entered application key exceeded buffer size: %d\n", MAX_USER_ARG_SIZE);
		return DOCA_ERROR_INVALID_VALUE;
	}

	strncpy(ml_dpa_cfg->dpa_application_key, app_key, dpa_app_key_len + 1);
	use_dpa_application_key = true;

	return DOCA_SUCCESS;
}

doca_error_t register_ml_dpa_params(void)
{
	struct doca_argp_param *device_param;
	struct doca_argp_param *threads_list_param;
	struct doca_argp_param *wait_time_param;
	struct doca_argp_param *remote_sw_handler_param;
	struct doca_argp_param *coredump_file_param;
	struct doca_argp_param *dpa_resources_file;
	struct doca_argp_param *dpa_application_key;

	/* Create and register DOCA device name parameter */
	doca_error_t result = doca_argp_param_create(&device_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(device_param, "d");
	doca_argp_param_set_long_name(device_param, "device");
	doca_argp_param_set_arguments(device_param, "<RDMA device names>");
	doca_argp_param_set_description(device_param, "RDMA device name that supports PCC (mandatory).");
	doca_argp_param_set_callback(device_param, device_name_callback);
	doca_argp_param_set_type(device_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(device_param);
	result = doca_argp_register_param(device_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register PCC threads list parameter */
	result = doca_argp_param_create(&threads_list_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(threads_list_param, "t");
	doca_argp_param_set_long_name(threads_list_param, "threads");
	doca_argp_param_set_arguments(threads_list_param, "<PCC threads list>");
	doca_argp_param_set_description(
		threads_list_param,
		"A list of the PCC threads numbers to be chosen for the DOCA PCC context to run on (optional). Must be provided as a string, such that the number are separated by a space.");
	doca_argp_param_set_callback(threads_list_param, threads_list_callback);
	doca_argp_param_set_type(threads_list_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(threads_list_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register PCC wait time parameter */
	result = doca_argp_param_create(&wait_time_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(wait_time_param, "w");
	doca_argp_param_set_long_name(wait_time_param, "wait-time");
	doca_argp_param_set_arguments(wait_time_param, "<PCC wait time>");
	doca_argp_param_set_description(
		wait_time_param,
		"The duration of the DOCA PCC wait (optional), can provide negative values which means infinity. If not provided then -1 will be chosen.");
	doca_argp_param_set_callback(wait_time_param, wait_time_callback);
	doca_argp_param_set_type(wait_time_param, DOCA_ARGP_TYPE_INT);
	result = doca_argp_register_param(wait_time_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register PCC remote SW handler */
	result = doca_argp_param_create(&remote_sw_handler_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(remote_sw_handler_param, "r-handler");
	doca_argp_param_set_long_name(remote_sw_handler_param, "remote-sw-handler");
	doca_argp_param_set_arguments(remote_sw_handler_param, "<CCMAD remote SW handler>");
	doca_argp_param_set_description(
		remote_sw_handler_param,
		"CCMAD remote SW handler flag (optional). If not provided then false will be chosen.");
	doca_argp_param_set_callback(remote_sw_handler_param, ccmad_remote_sw_handler_callback);
	doca_argp_param_set_type(remote_sw_handler_param, DOCA_ARGP_TYPE_BOOLEAN);
	result = doca_argp_register_param(remote_sw_handler_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register PCC device coredump file parameter */
	result = doca_argp_param_create(&coredump_file_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_short_name(coredump_file_param, "f");
	doca_argp_param_set_long_name(coredump_file_param, "coredump-file");
	doca_argp_param_set_arguments(coredump_file_param, "<PCC coredump file>");
	doca_argp_param_set_description(
		coredump_file_param,
		"A pathname to the file to write coredump data in case of unrecoverable error on the device (optional). Must be provided as a string.");
	doca_argp_param_set_callback(coredump_file_param, coredump_file_callback);
	doca_argp_param_set_type(coredump_file_param, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(coredump_file_param);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register DPA resources file parameter */
	result = doca_argp_param_create(&dpa_resources_file);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_long_name(dpa_resources_file, "dpa-resources");
	doca_argp_param_set_arguments(dpa_resources_file, "<DPA resources file>");
	doca_argp_param_set_description(
		dpa_resources_file,
		"Path to a DPA resources .yaml file (optional). Enumerates every DPA execution unit available "
		"to the app so the ML inference workload can be spread across all of them (many cores, many "
		"queues) instead of the built-in default subset. Must be provided together with DPA application key.");
	doca_argp_param_set_callback(dpa_resources_file, dpa_resources_file_callback);
	doca_argp_param_set_type(dpa_resources_file, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(dpa_resources_file);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	/* Create and register DPA application name parameter */
	result = doca_argp_param_create(&dpa_application_key);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: Failed to create ARGP param: %s\n", doca_error_get_descr(result));
		return result;
	}
	doca_argp_param_set_long_name(dpa_application_key, "dpa-app-key");
	doca_argp_param_set_arguments(dpa_application_key, "<DPA application key>");
	doca_argp_param_set_description(
		dpa_application_key,
		"Application key in specified DPA resources .yaml file (optional). Use 'ml_dpa_rp_ml_cc_app'. Must be "
		"provided together with DPA resources file.");
	doca_argp_param_set_callback(dpa_application_key, dpa_application_key_callback);
	doca_argp_param_set_type(dpa_application_key, DOCA_ARGP_TYPE_STRING);
	result = doca_argp_register_param(dpa_application_key);
	if (result != DOCA_SUCCESS) {
		PRINT_ERROR("Error: failed to register program param: %s\n", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}
