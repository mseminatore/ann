/**********************************************************************************/
/* Copyright (c) 2023 Mark Seminatore                                             */
/* All rights reserved.                                                           */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

/**
 * @file mnist_hypertune.c
 * @brief Hyperparameter tuning example using Bayesian optimization on Fashion-MNIST
 *
 * This example demonstrates how to use the ann_hypertune module to find optimal
 * hyperparameters for a neural network. It uses Bayesian optimization to search
 * over learning rate, hidden layer count, and nodes per layer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ann.h"
#include "ann_hypertune.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

// Configuration
#define TRAINING_SUBSET_SIZE  5000   // Use subset for faster trials
#define VALIDATION_RATIO      0.2f   // 20% for validation
#define NUM_CLASSES           10
#define INPUT_SIZE            784    // 28x28 images

//------------------------------
// Progress callback - shows trial results
//------------------------------
static void progress_callback(
    int current_trial,
    int total_trials,
    const HypertuneResult *best_so_far,
    const HypertuneResult *current_result,
    void *user_data)
{
    (void)user_data;
    
    printf("\n=== Trial %d/%d ===\n", current_trial, total_trials);
    printf("  LR: %.6f, Batch: %u, Hidden layers: %d\n",
           current_result->learning_rate,
           current_result->batch_size,
           current_result->hidden_layer_count);
    
    printf("  Layer sizes: ");
    for (int i = 0; i < current_result->hidden_layer_count; i++)
        printf("%d ", current_result->hidden_layer_sizes[i]);
    printf("\n");
    
    printf("  Score: %.4f (%.1f ms, %d epochs)\n",
           current_result->score,
           current_result->training_time_ms,
           current_result->epochs_used);
    
    printf("  Best so far: %.4f (Trial %d)\n",
           best_so_far->score,
           best_so_far->trial_id);
}

//------------------------------
// Silent print function - suppresses training output
//------------------------------
static void silent_print(const char *msg)
{
    (void)msg;
}

//------------------------------
// Main program
//------------------------------
int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    
    printf("=================================================\n");
    printf("  Fashion-MNIST Hyperparameter Tuning\n");
    printf("  Using Bayesian Optimization\n");
    printf("=================================================\n\n");

#if defined(USE_CBLAS)
    cblas_init(CBLAS_DEFAULT_THREADS);
    printf("CBLAS: %s\n", cblas_get_config());
    printf("Cores/Threads: %d/%d\n\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

    // Load Fashion-MNIST training data
    printf("Loading Fashion-MNIST data...\n");
    
    real *data = NULL;
    int rows, stride;
    
    if (ERR_OK != ann_load_csv("fashion-mnist_train.csv", CSV_HAS_HEADER, &data, &rows, &stride))
    {
        printf("Error: Unable to load fashion-mnist_train.csv\n");
        printf("Make sure the file is in the current directory.\n");
        return ERR_FAIL;
    }
    
    printf("Loaded %d samples with %d features\n", rows, stride);
    
    // Use subset for faster hypertuning
    int subset_size = (rows < TRAINING_SUBSET_SIZE) ? rows : TRAINING_SUBSET_SIZE;
    printf("Using subset of %d samples for hypertuning\n", subset_size);
    
    // Create tensors from loaded data
    PTensor full_data = tensor_create_from_array(subset_size, stride, data);
    free(data);
    
    // Split into features and labels
    PTensor x_data = tensor_slice_cols(full_data, 1);  // Remove first column (label)
    PTensor y_labels = tensor_onehot(full_data, NUM_CLASSES);  // One-hot encode labels
    
    // Normalize inputs to [0, 1]
    tensor_mul_scalar(x_data, (real)(1.0 / 255.0));
    
    // Split into training and validation sets
    DataSplit split;
    if (ERR_OK != hypertune_split_data(x_data, y_labels, 1.0f - VALIDATION_RATIO, 1, 42, &split))
    {
        printf("Error: Failed to split data\n");
        tensor_free(full_data);
        tensor_free(x_data);
        tensor_free(y_labels);
        return ERR_FAIL;
    }
    
    printf("Training samples: %d, Validation samples: %d\n\n", split.train_rows, split.val_rows);
    
    // Configure hyperparameter search space
    HyperparamSpace space;
    hypertune_space_init(&space);
    
    // Learning rate range (log scale)
    space.learning_rate_min = 0.0001f;
    space.learning_rate_max = 0.1f;
    space.learning_rate_log_scale = 1;
    
    // Batch sizes to try
    space.batch_sizes[0] = 16;
    space.batch_sizes[1] = 32;
    space.batch_sizes[2] = 64;
    space.batch_size_count = 3;
    
    // Hidden layer counts: 1, 2, or 3 hidden layers
    space.hidden_layer_counts[0] = 1;
    space.hidden_layer_counts[1] = 2;
    space.hidden_layer_counts[2] = 3;
    space.hidden_layer_count_options = 3;
    
    // Hidden layer sizes to try
    space.hidden_layer_sizes[0] = 32;
    space.hidden_layer_sizes[1] = 64;
    space.hidden_layer_sizes[2] = 128;
    space.hidden_layer_sizes[3] = 256;
    space.hidden_layer_size_count = 4;
    
    // Topology patterns
    space.topology_patterns[0] = TOPOLOGY_CONSTANT;
    space.topology_patterns[1] = TOPOLOGY_PYRAMID;
    space.topology_pattern_count = 2;
    
    // Training settings per trial
    space.epoch_limit = 5;  // Quick trials
    space.convergence_epsilon = 0.001f;
    
    // Configure Bayesian optimization
    BayesianOptions bayes_opts;
    bayesian_options_init(&bayes_opts);
    bayes_opts.n_initial = 5;      // Initial random samples
    bayes_opts.n_iterations = 10;  // Bayesian iterations after initial
    
    // Configure hypertuning options
    HypertuneOptions tune_opts;
    hypertune_options_init(&tune_opts);
    tune_opts.score_func = hypertune_score_accuracy;  // Use accuracy as score
    tune_opts.progress_func = progress_callback;
    tune_opts.verbosity = 1;
    tune_opts.seed = 42;  // For reproducibility
    
    // Allocate results array
    int max_results = bayes_opts.n_initial + bayes_opts.n_iterations;
    HypertuneResult *results = malloc(max_results * sizeof(HypertuneResult));
    HypertuneResult best_result;
    hypertune_result_init(&best_result);
    
    printf("Starting Bayesian optimization...\n");
    printf("  Initial random trials: %d\n", bayes_opts.n_initial);
    printf("  Bayesian iterations: %d\n", bayes_opts.n_iterations);
    printf("  Total trials: %d\n\n", max_results);
    
    clock_t start = clock();
    
    // Run Bayesian search
    int trials_completed = hypertune_bayesian_search(
        &space,
        INPUT_SIZE,
        NUM_CLASSES,
        ACTIVATION_SOFTMAX,
        LOSS_CATEGORICAL_CROSS_ENTROPY,
        &split,
        &tune_opts,
        &bayes_opts,
        results,
        max_results,
        &best_result
    );
    
    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (trials_completed < 0)
    {
        printf("Error: Hypertuning failed with code %d\n", trials_completed);
        free(results);
        hypertune_free_split(&split);
        tensor_free(full_data);
        tensor_free(x_data);
        tensor_free(y_labels);
        return ERR_FAIL;
    }
    
    // Print results
    printf("\n=================================================\n");
    printf("  Hypertuning Complete!\n");
    printf("=================================================\n\n");
    printf("Total trials: %d\n", trials_completed);
    printf("Total time: %.1f seconds\n\n", total_time);
    
    printf("=== Best Configuration ===\n");
    hypertune_print_result(&best_result);
    
    printf("\n=== Top 5 Results ===\n");
    hypertune_print_summary(results, trials_completed, 5);
    
    // Create and test a network with the best configuration
    printf("\n=== Verifying Best Configuration ===\n");
    
    PNetwork best_net = hypertune_create_network(
        &best_result,
        INPUT_SIZE,
        NUM_CLASSES,
        ACTIVATION_SOFTMAX,
        LOSS_CATEGORICAL_CROSS_ENTROPY
    );
    
    if (best_net)
    {
        best_net->epochLimit = 10;  // Train longer for verification
        best_net->batchSize = best_result.batch_size;
        
        printf("Training with best configuration for %d epochs...\n", best_net->epochLimit);
        ann_train_network(best_net, split.train_inputs, split.train_outputs, split.train_rows);
        
        // Evaluate on validation set
        real final_accuracy = hypertune_score_accuracy(
            best_net, split.val_inputs, split.val_outputs, NULL);
        
        printf("\nFinal validation accuracy: %.2f%%\n", final_accuracy * 100.0f);
        
        ann_free_network(best_net);
    }
    
    // Cleanup
    free(results);
    hypertune_free_split(&split);
    tensor_free(full_data);
    tensor_free(x_data);
    tensor_free(y_labels);
    
    printf("\nDone!\n");
    return ERR_OK;
}
