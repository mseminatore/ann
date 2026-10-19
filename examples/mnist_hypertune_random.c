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
 * @file mnist_hypertune_random.c
 * @brief Random search hyperparameter tuning example on Fashion-MNIST
 *
 * Random search samples each trial independently from the search space. It is a
 * good baseline, and it covers every dimension, including per-layer activations,
 * with a fixed trial budget.
 * See mnist_hypertune.c for the TPE version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ann.h"
#include "ann_hypertune.h"

#define TRAINING_SUBSET_SIZE  3000   // Use subset for faster trials
#define VALIDATION_RATIO      0.2f
#define NUM_CLASSES           10
#define INPUT_SIZE            784
#define MAX_TRIALS            40     // random search trial budget

static void progress_callback(
    int current_trial,
    int total_trials,
    const HypertuneResult *best_so_far,
    const HypertuneResult *current_result,
    void *user_data)
{
    (void)user_data;
    printf("[%3d/%3d] score=%.4f (best %.4f) lr=%.5f batch=%u layers=%d %s\n",
           current_trial, total_trials,
           current_result->score, best_so_far->score,
           current_result->learning_rate, current_result->batch_size,
           current_result->hidden_layer_count,
           hypertune_topology_name(current_result->topology_pattern));
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    printf("Random search on Fashion-MNIST\n\n");

    real *data = NULL;
    int rows, stride;
    if (ERR_OK != ann_load_csv("fashion-mnist_train.csv", CSV_HAS_HEADER, &data, &rows, &stride))
    {
        printf("Error: Unable to load fashion-mnist_train.csv\n");
        return ERR_FAIL;
    }

    int subset_size = (rows < TRAINING_SUBSET_SIZE) ? rows : TRAINING_SUBSET_SIZE;
    PTensor full_data = tensor_create_from_array(subset_size, stride, data);
    free(data);

    PTensor x_data = tensor_slice_cols(full_data, 1);
    PTensor y_labels = tensor_onehot(full_data, NUM_CLASSES);
    tensor_mul_scalar(x_data, (real)(1.0 / 255.0));

    DataSplit split;
    if (ERR_OK != hypertune_split_data(x_data, y_labels, 1.0f - VALIDATION_RATIO, 1, 42, &split))
    {
        printf("Error: Failed to split data\n");
        tensor_free(full_data);
        tensor_free(x_data);
        tensor_free(y_labels);
        return ERR_FAIL;
    }

    HyperparamSpace space;
    hypertune_space_init(&space);

    space.learning_rate_min = 0.0001f;
    space.learning_rate_max = 0.1f;
    space.learning_rate_log_scale = 1;

    space.batch_sizes[0] = 16;
    space.batch_sizes[1] = 32;
    space.batch_sizes[2] = 64;
    space.batch_sizes[3] = 128;
    space.batch_size_count = 4;

    space.optimizers[0] = OPT_ADAM;
    space.optimizers[1] = OPT_SGD;
    space.optimizers[2] = OPT_MOMENTUM;
    space.optimizers[3] = OPT_RMSPROP;
    space.optimizer_count = 4;

    space.hidden_layer_counts[0] = 1;
    space.hidden_layer_counts[1] = 2;
    space.hidden_layer_counts[2] = 3;
    space.hidden_layer_count_options = 3;

    space.hidden_layer_sizes[0] = 64;
    space.hidden_layer_sizes[1] = 128;
    space.hidden_layer_sizes[2] = 256;
    space.hidden_layer_size_count = 3;

    space.topology_patterns[0] = TOPOLOGY_CONSTANT;
    space.topology_patterns[1] = TOPOLOGY_PYRAMID;
    space.topology_patterns[2] = TOPOLOGY_FUNNEL;
    space.topology_pattern_count = 3;

    space.hidden_activations[0] = ACTIVATION_RELU;
    space.hidden_activations[1] = ACTIVATION_SIGMOID;
    space.hidden_activations[2] = ACTIVATION_TANH;
    space.hidden_activations[3] = ACTIVATION_LEAKY_RELU;
    space.hidden_activation_count = 4;

    // Choose a separate activation for each hidden layer
    space.search_per_layer_activation = 1;

    space.epoch_limit = 3;  // Quick trials
    space.convergence_epsilon = 0.001f;

    HypertuneOptions tune_opts;
    hypertune_options_init(&tune_opts);
    tune_opts.score_func = hypertune_score_accuracy;
    tune_opts.progress_func = progress_callback;

    HypertuneResult *results = malloc(MAX_TRIALS * sizeof(HypertuneResult));
    HypertuneResult best_result;
    hypertune_result_init(&best_result);

    clock_t start = clock();
    int trials_completed = hypertune_random_search(
        &space,
        MAX_TRIALS,
        INPUT_SIZE,
        NUM_CLASSES,
        ACTIVATION_SOFTMAX,
        LOSS_CATEGORICAL_CROSS_ENTROPY,
        &split,
        &tune_opts,
        results,
        MAX_TRIALS,
        &best_result
    );
    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;

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

    printf("\nTotal trials: %d in %.1f seconds\n\n", trials_completed, total_time);
    printf("=== Best Configuration ===\n");
    hypertune_print_result(&best_result);
    hypertune_print_summary(results, trials_completed, 5);

    free(results);
    hypertune_free_split(&split);
    tensor_free(full_data);
    tensor_free(x_data);
    tensor_free(y_labels);
    return ERR_OK;
}
