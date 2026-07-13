#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>

// Defines for the MLP architecture
#define INPUT_SIZE  16
#define OUTPUT_SIZE 4
#define NUM_INPUTS  16 // Number of inputs to test

// Simple function to get the current time in milliseconds
double get_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

int main() {
    printf("Starting XNNPACK MLP Benchmark on BlueField-3...\n");

    // 1. Initialize XNNPACK
    enum xnn_status status = xnn_initialize(NULL);
    if (status != xnn_status_success) {
        fprintf(stderr, "Error: Failed to initialize XNNPACK\n");
        return 1;
    }

    // 2. Allocate memory for input, weights, and output
    // We're creating a simple MLP with one hidden layer for demonstration.
    // For a generic MLP, you would need to define the number and size of layers.
    // For simplicity, let's make it a single fully-connected (dense) layer.
    // We'll create a matrix-vector multiplication operator (xnn_create_fully_connected_nc).

    // 3. Define operator parameters
    size_t batch_size = 1; // We will run inference for one input at a time
    size_t channels_in = INPUT_SIZE;
    size_t channels_out = OUTPUT_SIZE;
    size_t input_stride = INPUT_SIZE;
    size_t output_stride = OUTPUT_SIZE;
    size_t input_pixel_stride = INPUT_SIZE;
    size_t output_pixel_stride = OUTPUT_SIZE;

    // Allocate and initialize random weights and bias
    // In a real scenario, these would be pre-trained values.
    float* kernel = (float*)malloc(channels_in * channels_out * sizeof(float));
    float* bias = (float*)malloc(channels_out * sizeof(float));
    for (size_t i = 0; i < channels_in * channels_out; ++i) {
        kernel[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random [-1, 1]
    }
    for (size_t i = 0; i < channels_out; ++i) {
        bias[i] = ((float)rand() / RAND_MAX) * 0.1f;
    }

    // Create a fully connected operator
    struct xnn_operator* fc_op = NULL;
    status = xnn_create_fully_connected_nc_f32(
        channels_in, channels_out,
        input_stride, output_stride,
        kernel, bias,
        0, 0, // flags
        &fc_op);

    if (status != xnn_status_success) {
        fprintf(stderr, "Error: Failed to create fully connected operator\n");
        free(kernel);
        free(bias);
        return 1;
    }

    // 4. Prepare the input and output buffers
    float input_data[NUM_INPUTS][INPUT_SIZE];
    float output_data[NUM_INPUTS][OUTPUT_SIZE];

    // Initialize random input data
    for (int i = 0; i < NUM_INPUTS; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            input_data[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    // 5. Set up the operator for inference
    // This 'setup' step must be called before `xnn_run_operator`.
    // It uses the input and output buffers, shapes, and the operator to prepare for execution.
    status = xnn_setup_fully_connected_nc_f32(
        fc_op,
        batch_size,
        input_data[0], // pointer to the first input
        output_data[0], // pointer to the first output
        NULL, // threadpool (NULL = use internal default)
    );

    if (status != xnn_status_success) {
        fprintf(stderr, "Error: Failed to setup fully connected operator\n");
        xnn_delete_operator(fc_op);
        free(kernel);
        free(bias);
        return 1;
    }

    // 6. Warm-up run
    printf("Warming up XNNPACK...\n");
    status = xnn_run_operator(fc_op, NULL);
    if (status != xnn_status_success) {
        fprintf(stderr, "Error: Warm-up inference failed\n");
        xnn_delete_operator(fc_op);
        free(kernel);
        free(bias);
        return 1;
    }

    // 7. Benchmarking and Logging to CSV
    printf("Starting benchmark for %d inputs...\n", NUM_INPUTS);
    printf("Input size: %d, Output size: %d\n", INPUT_SIZE, OUTPUT_SIZE);

    // Open CSV file for writing
    FILE* csv_file = fopen("inference_latency_log.csv", "w");
    if (!csv_file) {
        fprintf(stderr, "Error: Could not create CSV file\n");
        xnn_delete_operator(fc_op);
        free(kernel);
        free(bias);
        return 1;
    }

    // Write CSV header
    fprintf(csv_file, "InputIndex,Latency_ms,OutputClass\n");

    // Loop through all inputs and measure inference time
    for (int i = 0; i < NUM_INPUTS; ++i) {
        // Set the input data for the i-th sample
        // We can either re-setup the operator or simply copy the input.
        // For a proper test, the setup step can change the input pointer.
        // A simpler approach is to copy the input data directly into the buffer used during setup.
        // However, let's demonstrate using `xnn_setup_fully_connected_nc_f32` again.
        // More efficient method: use a single buffer and modify its contents.
        // For clarity, we re-setup with the new input.

        status = xnn_setup_fully_connected_nc_f32(
            fc_op,
            batch_size,
            input_data[i], // pointer to the i-th input
            output_data[i], // pointer to the i-th output
            NULL,
        );
        if (status != xnn_status_success) {
            fprintf(stderr, "Error: Setup for input %d failed\n", i);
            continue;
        }

        // Measure inference time
        double start_time = get_time_in_ms();
        status = xnn_run_operator(fc_op, NULL);
        double end_time = get_time_in_ms();

        if (status != xnn_status_success) {
            fprintf(stderr, "Error: Inference for input %d failed\n", i);
            continue;
        }

        double latency_ms = end_time - start_time;

        // Determine the output class (the index of the maximum output value)
        int class_idx = 0;
        float max_val = output_data[i][0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) {
            if (output_data[i][j] > max_val) {
                max_val = output_data[i][j];
                class_idx = j;
            }
        }

        // Log to CSV file
        fprintf(csv_file, "%d,%f,%d\n", i, latency_ms, class_idx);
        printf("Input %d: Latency = %f ms, Output Class = %d\n", i, latency_ms, class_idx);
    }

    fclose(csv_file);

    // 8. Cleanup
    xnn_delete_operator(fc_op);
    free(kernel);
    free(bias);
    xnn_deinitialize();

    printf("Benchmark complete! Results logged to inference_latency_log.csv\n");
    return 0;
}