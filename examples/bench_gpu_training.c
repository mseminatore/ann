/*
 * GPU Training Performance Benchmark
 * 
 * Compares Metal GPU training vs CPU training across different scenarios:
 * 1. Tiny network (XOR: 2→4→1)
 * 2. MNIST-scale (784→128→10)
 * 3. Large network (784→256→128→10)
 * 4. Batch-size scaling (8, 32, 128, 512)
 * 
 * Usage:
 *   bench_gpu_training --network {xor,mnist,large,scaling} [--epochs N] [--gpu {0,1}]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ann.h"

#ifdef _WIN32
// MSVC lacks POSIX clock_gettime/CLOCK_MONOTONIC. Provide a high-resolution shim
// backed by QueryPerformanceCounter so the benchmark builds and times correctly on Windows.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static int clock_gettime(int clk_id, struct timespec *ts)
{
    (void)clk_id;
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    ts->tv_sec  = (time_t)(count.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((count.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif

// Helper: milliseconds elapsed between two timespec values
static long time_elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + 
           (end.tv_nsec - start.tv_nsec) / 1000000;
}

// ============================================================================
// Scenario 1: XOR (tiny network)
// ============================================================================
static void benchmark_xor(int use_gpu, int epochs)
{
    printf("\n=== Scenario: XOR (2→4→1) ===\n");
    printf("Epochs: %d, Use GPU: %s\n", epochs, use_gpu ? "yes" : "no");

    PNetwork net = ann_make_network(OPT_SGD, LOSS_MSE);
    ann_add_layer(net, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 4, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    real xor_in[] = {0, 0, 0, 1, 1, 0, 1, 1};
    real xor_out[] = {0, 1, 1, 0};
    PTensor inputs = tensor_create_from_array(4, 2, xor_in);
    PTensor outputs = tensor_create_from_array(4, 1, xor_out);

    ann_set_epoch_limit(net, epochs);
    ann_set_convergence(net, 0.01f);

    if (use_gpu) {
        ann_gpu_upload_network(net);
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    real loss = ann_train_network(net, inputs, outputs, 4);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long elapsed_ms = time_elapsed_ms(ts_start, ts_end);

    if (use_gpu) {
        ann_gpu_sync_weights(net);
    }

    printf("Time: %ld ms\n", elapsed_ms);
    printf("Final loss: %.4f\n", loss);

    tensor_free(inputs);
    tensor_free(outputs);
    ann_free_network(net);
}

// ============================================================================
// Scenario 2: MNIST-scale (784→128→10)
// ============================================================================
static void benchmark_mnist(int use_gpu, int epochs)
{
    printf("\n=== Scenario: MNIST-scale (784→128→10) ===\n");
    printf("Epochs: %d, Use GPU: %s\n", epochs, use_gpu ? "yes" : "no");

    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 784, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);
    ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

    ann_set_epoch_limit(net, epochs);
    ann_set_convergence(net, 0.001f);

    // Generate synthetic MNIST-like data (small subset for benchmark)
    int n_samples = 5000;  // Reduced for faster benchmark
    int n_inputs = 784;
    int n_outputs = 10;

    PTensor inputs = tensor_create(n_samples, n_inputs);
    PTensor outputs = tensor_create(n_samples, n_outputs);

    // Fill with random data (0 to 1)
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_inputs; j++)
            inputs->values[i * n_inputs + j] = (real)rand() / RAND_MAX;
        
        // One-hot encoded labels
        int label = i % 10;
        for (int j = 0; j < n_outputs; j++)
            outputs->values[i * n_outputs + j] = (j == label) ? 1.0f : 0.0f;
    }

    if (use_gpu) {
        ann_gpu_upload_network(net);
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    real loss = ann_train_network(net, inputs, outputs, n_samples);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long elapsed_ms = time_elapsed_ms(ts_start, ts_end);

    if (use_gpu) {
        ann_gpu_sync_weights(net);
    }

    printf("Time: %ld ms\n", elapsed_ms);
    printf("Final loss: %.4f\n", loss);

    tensor_free(inputs);
    tensor_free(outputs);
    ann_free_network(net);
}

// ============================================================================
// Scenario 3: Large network (784→256→128→10)
// ============================================================================
static void benchmark_large(int use_gpu, int epochs)
{
    printf("\n=== Scenario: Large network (784→256→128→10) ===\n");
    printf("Epochs: %d, Use GPU: %s\n", epochs, use_gpu ? "yes" : "no");

    PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    ann_add_layer(net, 784, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(net, 256, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 128, LAYER_HIDDEN, ACTIVATION_RELU);
    ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

    ann_set_epoch_limit(net, epochs);
    ann_set_convergence(net, 0.001f);

    // Synthetic data
    int n_samples = 2000;  // Smaller dataset, larger network
    int n_inputs = 784;
    int n_outputs = 10;

    PTensor inputs = tensor_create(n_samples, n_inputs);
    PTensor outputs = tensor_create(n_samples, n_outputs);

    srand(42);
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_inputs; j++)
            inputs->values[i * n_inputs + j] = (real)rand() / RAND_MAX;
        
        int label = i % 10;
        for (int j = 0; j < n_outputs; j++)
            outputs->values[i * n_outputs + j] = (j == label) ? 1.0f : 0.0f;
    }

    if (use_gpu) {
        ann_gpu_upload_network(net);
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    real loss = ann_train_network(net, inputs, outputs, n_samples);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long elapsed_ms = time_elapsed_ms(ts_start, ts_end);

    if (use_gpu) {
        ann_gpu_sync_weights(net);
    }

    printf("Time: %ld ms\n", elapsed_ms);
    printf("Final loss: %.4f\n", loss);

    tensor_free(inputs);
    tensor_free(outputs);
    ann_free_network(net);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    // Parse arguments
    const char *scenario = "xor";
    int epochs = 5;
    int use_gpu = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--network") == 0 && i + 1 < argc)
            scenario = argv[++i];
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            use_gpu = atoi(argv[++i]);
    }

    // Initialize GPU if available
    if (use_gpu) {
        if (!ann_gpu_init()) {
            fprintf(stderr, "Warning: Metal GPU not available, falling back to CPU\n");
            use_gpu = 0;
        }
    }

    printf("GPU Training Performance Benchmark\n");
    printf("===================================\n");
    printf("GPU enabled: %s\n", use_gpu ? "yes" : "no");

    // Run selected scenario
    if (strcmp(scenario, "xor") == 0)
        benchmark_xor(use_gpu, epochs);
    else if (strcmp(scenario, "mnist") == 0)
        benchmark_mnist(use_gpu, epochs);
    else if (strcmp(scenario, "large") == 0)
        benchmark_large(use_gpu, epochs);
    else {
        fprintf(stderr, "Unknown scenario: %s\n", scenario);
        fprintf(stderr, "Valid scenarios: xor, mnist, large\n");
        return 1;
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
