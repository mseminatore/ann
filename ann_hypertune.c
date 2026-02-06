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

#ifdef _WIN32
#   define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ann_hypertune.h"

//------------------------------
// Platform-specific timing
//------------------------------
#ifdef _WIN32
#   include <windows.h>
    static double get_time_ms(void) {
        LARGE_INTEGER freq, counter;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);
        return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
    }
#else
#   include <sys/time.h>
    static double get_time_ms(void) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }
#endif

//------------------------------
// Optimizer names for printing
//------------------------------
static const char *optimizer_names[] = {
    "SGD", "SGD_DECAY", "ADAPT", "MOMENTUM", "RMSPROP", "ADAGRAD", "ADAM"
};

static const char *activation_names[] = {
    "NULL", "SIGMOID", "RELU", "LEAKY_RELU", "TANH", "SOFTSIGN", "SOFTMAX"
};

static const char *topology_names[] = {
    "CONSTANT", "PYRAMID", "FUNNEL", "INVERSE", "CUSTOM"
};

// ============================================================================
// INITIALIZATION
// ============================================================================

void hypertune_space_init(HyperparamSpace *space)
{
    if (!space) return;
    
    memset(space, 0, sizeof(HyperparamSpace));
    
    // Default learning rate range
    space->learning_rate_min = 0.001f;
    space->learning_rate_max = 0.1f;
    space->learning_rate_steps = 3;
    space->learning_rate_log_scale = 1;
    
    // Default batch sizes
    space->batch_sizes[0] = 32;
    space->batch_sizes[1] = 64;
    space->batch_size_count = 2;
    
    // Default optimizers
    space->optimizers[0] = OPT_ADAM;
    space->optimizers[1] = OPT_SGD;
    space->optimizer_count = 2;
    
    // Default hidden layer counts
    space->hidden_layer_counts[0] = 1;
    space->hidden_layer_counts[1] = 2;
    space->hidden_layer_count_options = 2;
    
    // Default hidden layer sizes
    space->hidden_layer_sizes[0] = 32;
    space->hidden_layer_sizes[1] = 64;
    space->hidden_layer_sizes[2] = 128;
    space->hidden_layer_size_count = 3;
    
    // Default topology patterns
    space->topology_patterns[0] = TOPOLOGY_CONSTANT;
    space->topology_pattern_count = 1;
    
    // Default activations
    space->hidden_activations[0] = ACTIVATION_RELU;
    space->hidden_activations[1] = ACTIVATION_SIGMOID;
    space->hidden_activation_count = 2;
    
    // Per-layer activation search disabled by default
    space->search_per_layer_activation = 0;
    
    // Training settings
    space->epoch_limit = 1000;
    space->convergence_epsilon = 0.01f;
    space->early_stopping_patience = 10;
}

void hypertune_options_init(HypertuneOptions *options)
{
    if (!options) return;
    
    memset(options, 0, sizeof(HypertuneOptions));
    options->score_func = hypertune_score_accuracy;
    options->progress_func = NULL;
    options->user_data = NULL;
    options->seed = 0;
    options->verbosity = 1;
}

void hypertune_result_init(HypertuneResult *result)
{
    if (!result) return;
    
    memset(result, 0, sizeof(HypertuneResult));
    result->score = -1.0f;  // invalid score marker
}

// ============================================================================
// DATA SPLITTING
// ============================================================================

// Fisher-Yates shuffle
static void shuffle_indices(int *indices, int n, unsigned seed)
{
    if (seed == 0) seed = (unsigned)time(NULL);
    srand(seed);
    
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

int hypertune_split_data(
    PTensor inputs,
    PTensor outputs,
    real train_ratio,
    int shuffle,
    unsigned seed,
    DataSplit *split)
{
    if (!inputs || !outputs || !split)
        return ERR_NULL_PTR;
    
    if (inputs->rows != outputs->rows)
        return ERR_INVALID;
    
    if (train_ratio <= 0.0f || train_ratio >= 1.0f)
        return ERR_INVALID;
    
    int total_rows = inputs->rows;
    int train_rows = (int)(total_rows * train_ratio);
    int val_rows = total_rows - train_rows;
    
    if (train_rows < 1 || val_rows < 1)
        return ERR_INVALID;
    
    // Create index array
    int *indices = malloc(total_rows * sizeof(int));
    if (!indices)
        return ERR_ALLOC;
    
    for (int i = 0; i < total_rows; i++)
        indices[i] = i;
    
    if (shuffle)
        shuffle_indices(indices, total_rows, seed);
    
    // Allocate split tensors
    split->train_inputs = tensor_create(train_rows, inputs->cols);
    split->train_outputs = tensor_create(train_rows, outputs->cols);
    split->val_inputs = tensor_create(val_rows, inputs->cols);
    split->val_outputs = tensor_create(val_rows, outputs->cols);
    
    if (!split->train_inputs || !split->train_outputs ||
        !split->val_inputs || !split->val_outputs) {
        hypertune_free_split(split);
        free(indices);
        return ERR_ALLOC;
    }
    
    // Copy training data
    for (int i = 0; i < train_rows; i++) {
        int src_row = indices[i];
        for (int j = 0; j < inputs->cols; j++)
            split->train_inputs->values[i * inputs->cols + j] = 
                inputs->values[src_row * inputs->cols + j];
        for (int j = 0; j < outputs->cols; j++)
            split->train_outputs->values[i * outputs->cols + j] = 
                outputs->values[src_row * outputs->cols + j];
    }
    
    // Copy validation data
    for (int i = 0; i < val_rows; i++) {
        int src_row = indices[train_rows + i];
        for (int j = 0; j < inputs->cols; j++)
            split->val_inputs->values[i * inputs->cols + j] = 
                inputs->values[src_row * inputs->cols + j];
        for (int j = 0; j < outputs->cols; j++)
            split->val_outputs->values[i * outputs->cols + j] = 
                outputs->values[src_row * outputs->cols + j];
    }
    
    split->train_rows = train_rows;
    split->val_rows = val_rows;
    
    free(indices);
    return ERR_OK;
}

void hypertune_free_split(DataSplit *split)
{
    if (!split) return;
    
    if (split->train_inputs) tensor_free(split->train_inputs);
    if (split->train_outputs) tensor_free(split->train_outputs);
    if (split->val_inputs) tensor_free(split->val_inputs);
    if (split->val_outputs) tensor_free(split->val_outputs);
    
    memset(split, 0, sizeof(DataSplit));
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void hypertune_generate_topology(
    TopologyPattern pattern,
    int base_size,
    int layer_count,
    int *output_sizes)
{
    if (!output_sizes || layer_count <= 0) return;
    
    switch (pattern) {
        case TOPOLOGY_PYRAMID:
            // Decreasing sizes: base, base/2, base/4, ...
            for (int i = 0; i < layer_count; i++) {
                int divisor = 1 << i;  // 1, 2, 4, 8, ...
                output_sizes[i] = base_size / divisor;
                if (output_sizes[i] < 8) output_sizes[i] = 8;  // minimum size
            }
            break;
            
        case TOPOLOGY_INVERSE:
            // Increasing sizes: base/2^(n-1), ..., base/2, base
            for (int i = 0; i < layer_count; i++) {
                int divisor = 1 << (layer_count - 1 - i);
                output_sizes[i] = base_size / divisor;
                if (output_sizes[i] < 8) output_sizes[i] = 8;
            }
            break;
            
        case TOPOLOGY_FUNNEL:
            // Expand then contract: base, base*2, base (for 3 layers)
            for (int i = 0; i < layer_count; i++) {
                int mid = layer_count / 2;
                if (i <= mid) {
                    output_sizes[i] = base_size * (1 << i);
                    if (output_sizes[i] > base_size * 4) 
                        output_sizes[i] = base_size * 4;  // cap expansion
                } else {
                    int dist_from_end = layer_count - 1 - i;
                    output_sizes[i] = base_size * (1 << dist_from_end);
                    if (output_sizes[i] < 8) output_sizes[i] = 8;
                }
            }
            break;
            
        case TOPOLOGY_CONSTANT:
        case TOPOLOGY_CUSTOM:
        default:
            // All layers same size
            for (int i = 0; i < layer_count; i++) {
                output_sizes[i] = base_size;
            }
            break;
    }
}

const char* hypertune_topology_name(TopologyPattern pattern)
{
    if (pattern >= 0 && pattern <= TOPOLOGY_CUSTOM)
        return topology_names[pattern];
    return "UNKNOWN";
}

PNetwork hypertune_create_network(
    const HypertuneResult *result,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type)
{
    if (!result)
        return NULL;
    
    PNetwork net = ann_make_network(result->optimizer, loss_type);
    if (!net)
        return NULL;
    
    // Set training parameters
    ann_set_learning_rate(net, result->learning_rate);
    ann_set_batch_size(net, result->batch_size);
    
    // Add input layer
    if (ann_add_layer(net, input_size, LAYER_INPUT, ACTIVATION_NULL) != ERR_OK) {
        ann_free_network(net);
        return NULL;
    }
    
    // Add hidden layers with per-layer activations
    for (int i = 0; i < result->hidden_layer_count; i++) {
        int nodes = result->hidden_layer_sizes[i];
        Activation_type act = result->hidden_activations[i];
        if (ann_add_layer(net, nodes, LAYER_HIDDEN, act) != ERR_OK) {
            ann_free_network(net);
            return NULL;
        }
    }
    
    // Add output layer
    if (ann_add_layer(net, output_size, LAYER_OUTPUT, output_activation) != ERR_OK) {
        ann_free_network(net);
        return NULL;
    }
    
    return net;
}

real hypertune_score_accuracy(
    PNetwork trained_net,
    PTensor val_inputs,
    PTensor val_outputs,
    void *user_data)
{
    (void)user_data;
    return ann_evaluate_accuracy(trained_net, val_inputs, val_outputs);
}

int hypertune_count_grid_trials(const HyperparamSpace *space)
{
    if (!space) return 0;
    
    int lr_steps = space->learning_rate_steps > 0 ? space->learning_rate_steps : 1;
    int batch_count = space->batch_size_count > 0 ? space->batch_size_count : 1;
    int opt_count = space->optimizer_count > 0 ? space->optimizer_count : 1;
    int layer_count_opts = space->hidden_layer_count_options > 0 ? space->hidden_layer_count_options : 1;
    int layer_size_count = space->hidden_layer_size_count > 0 ? space->hidden_layer_size_count : 1;
    int act_count = space->hidden_activation_count > 0 ? space->hidden_activation_count : 1;
    int topo_count = space->topology_pattern_count > 0 ? space->topology_pattern_count : 1;
    
    return lr_steps * batch_count * opt_count * layer_count_opts * layer_size_count * act_count * topo_count;
}

// ============================================================================
// RESULT PRINTING
// ============================================================================

void hypertune_print_result(const HypertuneResult *result)
{
    if (!result) return;
    
    printf("Trial %d: score=%.4f, lr=%.6f, batch=%u, opt=%s, ",
           result->trial_id,
           result->score,
           result->learning_rate,
           result->batch_size,
           optimizer_names[result->optimizer]);
    
    printf("topo=%s, layers=%d [", 
           topology_names[result->topology_pattern],
           result->hidden_layer_count);
    for (int i = 0; i < result->hidden_layer_count; i++) {
        if (i > 0) printf(",");
        printf("%d", result->hidden_layer_sizes[i]);
    }
    printf("], acts=[");
    for (int i = 0; i < result->hidden_layer_count; i++) {
        if (i > 0) printf(",");
        printf("%s", activation_names[result->hidden_activations[i]]);
    }
    printf("], epochs=%d, time=%.1fms\n",
           result->epochs_used,
           result->training_time_ms);
}

void hypertune_print_summary(const HypertuneResult *results, int count, int top_n)
{
    if (!results || count <= 0) return;
    
    // Find indices of top N results
    int *sorted_indices = malloc(count * sizeof(int));
    if (!sorted_indices) return;
    
    for (int i = 0; i < count; i++)
        sorted_indices[i] = i;
    
    // Simple bubble sort by score (descending)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (results[sorted_indices[j]].score < results[sorted_indices[j + 1]].score) {
                int temp = sorted_indices[j];
                sorted_indices[j] = sorted_indices[j + 1];
                sorted_indices[j + 1] = temp;
            }
        }
    }
    
    printf("\n=== Top %d Results ===\n", top_n < count ? top_n : count);
    int show_count = top_n < count ? top_n : count;
    for (int i = 0; i < show_count; i++) {
        printf("%d. ", i + 1);
        hypertune_print_result(&results[sorted_indices[i]]);
    }
    
    free(sorted_indices);
}

// ============================================================================
// GRID SEARCH
// ============================================================================

// Helper: get learning rate value for step
static real get_learning_rate(const HyperparamSpace *space, int step)
{
    if (space->learning_rate_steps <= 1)
        return space->learning_rate_min;
    
    real t = (real)step / (real)(space->learning_rate_steps - 1);
    
    if (space->learning_rate_log_scale) {
        real log_min = (real)log(space->learning_rate_min);
        real log_max = (real)log(space->learning_rate_max);
        return (real)exp(log_min + t * (log_max - log_min));
    } else {
        return space->learning_rate_min + t * (space->learning_rate_max - space->learning_rate_min);
    }
}

int hypertune_grid_search(
    const HyperparamSpace *space,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type,
    const DataSplit *split,
    const HypertuneOptions *options,
    HypertuneResult *results,
    int max_results,
    HypertuneResult *best_result)
{
    if (!space || !split || !options || !options->score_func)
        return ERR_NULL_PTR;
    
    int total_trials = hypertune_count_grid_trials(space);
    if (total_trials <= 0)
        return ERR_INVALID;
    
    if (options->verbosity >= 1) {
        printf("Starting grid search: %d total trials\n", total_trials);
    }
    
    int trial = 0;
    HypertuneResult best;
    hypertune_result_init(&best);
    best.score = -999999.0f;
    
    int lr_steps = space->learning_rate_steps > 0 ? space->learning_rate_steps : 1;
    
    // Nested loops over all hyperparameters
    for (int lr_i = 0; lr_i < lr_steps; lr_i++) {
        real lr = get_learning_rate(space, lr_i);
        
        for (int batch_i = 0; batch_i < space->batch_size_count; batch_i++) {
            unsigned batch = space->batch_sizes[batch_i];
            
            for (int opt_i = 0; opt_i < space->optimizer_count; opt_i++) {
                Optimizer_type opt = space->optimizers[opt_i];
                
                for (int lc_i = 0; lc_i < space->hidden_layer_count_options; lc_i++) {
                    int layer_count = space->hidden_layer_counts[lc_i];
                    
                    for (int ls_i = 0; ls_i < space->hidden_layer_size_count; ls_i++) {
                        int layer_size = space->hidden_layer_sizes[ls_i];
                        
                        for (int act_i = 0; act_i < space->hidden_activation_count; act_i++) {
                            Activation_type act = space->hidden_activations[act_i];
                            
                            int topo_count = space->topology_pattern_count > 0 ? space->topology_pattern_count : 1;
                            for (int topo_i = 0; topo_i < topo_count; topo_i++) {
                                TopologyPattern topo = (space->topology_pattern_count > 0) 
                                    ? space->topology_patterns[topo_i] 
                                    : TOPOLOGY_CONSTANT;
                                
                                // Build result config
                                HypertuneResult current;
                                hypertune_result_init(&current);
                                current.trial_id = trial + 1;
                                current.learning_rate = lr;
                                current.batch_size = batch;
                                current.optimizer = opt;
                                current.hidden_layer_count = layer_count;
                                current.topology_pattern = topo;
                                
                                // Generate topology-based layer sizes
                                hypertune_generate_topology(topo, layer_size, layer_count, 
                                                           current.hidden_layer_sizes);
                                
                                // Set per-layer activations
                                for (int l = 0; l < layer_count; l++) {
                                    if (space->search_per_layer_activation && 
                                        space->hidden_activation_count > 1) {
                                        // Vary activation per layer based on topo_i
                                        current.hidden_activations[l] = 
                                            space->hidden_activations[(act_i + l) % space->hidden_activation_count];
                                    } else {
                                        current.hidden_activations[l] = act;
                                    }
                                }
                                
                                // Create and train network
                                PNetwork net = hypertune_create_network(
                                    &current, input_size, output_size,
                                    output_activation, loss_type);
                                
                                if (!net) {
                                    current.score = -1.0f;
                                } else {
                                    ann_set_epoch_limit(net, space->epoch_limit);
                                    ann_set_convergence(net, space->convergence_epsilon);
                                    
                                    double start_time = get_time_ms();
                                    ann_train_network(net, 
                                        split->train_inputs, 
                                        split->train_outputs,
                                        split->train_rows);
                                    double end_time = get_time_ms();
                                    
                                    current.training_time_ms = (real)(end_time - start_time);
                                    current.score = options->score_func(
                                        net, split->val_inputs, split->val_outputs,
                                        options->user_data);
                                    
                                    ann_free_network(net);
                                }
                                
                                // Store result
                                if (results && trial < max_results)
                                    results[trial] = current;
                                
                                // Update best
                                if (current.score > best.score)
                                    best = current;
                                
                                // Progress callback
                                if (options->progress_func) {
                                    options->progress_func(trial + 1, total_trials,
                                        &best, &current, options->user_data);
                                } else if (options->verbosity >= 1) {
                                    printf("[%d/%d] ", trial + 1, total_trials);
                                    hypertune_print_result(&current);
                                }
                                
                                trial++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (best_result)
        *best_result = best;
    
    if (options->verbosity >= 1) {
        printf("\nBest result: ");
        hypertune_print_result(&best);
    }
    
    return trial;
}

// ============================================================================
// RANDOM SEARCH
// ============================================================================

int hypertune_random_search(
    const HyperparamSpace *space,
    int num_trials,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type,
    const DataSplit *split,
    const HypertuneOptions *options,
    HypertuneResult *results,
    int max_results,
    HypertuneResult *best_result)
{
    if (!space || !split || !options || !options->score_func)
        return ERR_NULL_PTR;
    
    if (num_trials <= 0)
        return ERR_INVALID;
    
    // Initialize random seed
    unsigned seed = options->seed ? options->seed : (unsigned)time(NULL);
    srand(seed);
    
    if (options->verbosity >= 1) {
        printf("Starting random search: %d trials (seed=%u)\n", num_trials, seed);
    }
    
    HypertuneResult best;
    hypertune_result_init(&best);
    best.score = -999999.0f;
    
    for (int trial = 0; trial < num_trials; trial++) {
        // Randomly sample hyperparameters
        HypertuneResult current;
        hypertune_result_init(&current);
        current.trial_id = trial + 1;
        
        // Random learning rate (log-uniform)
        if (space->learning_rate_log_scale) {
            real log_min = (real)log(space->learning_rate_min);
            real log_max = (real)log(space->learning_rate_max);
            real t = (real)rand() / (real)RAND_MAX;
            current.learning_rate = (real)exp(log_min + t * (log_max - log_min));
        } else {
            real t = (real)rand() / (real)RAND_MAX;
            current.learning_rate = space->learning_rate_min + 
                t * (space->learning_rate_max - space->learning_rate_min);
        }
        
        // Random batch size
        if (space->batch_size_count > 0)
            current.batch_size = space->batch_sizes[rand() % space->batch_size_count];
        else
            current.batch_size = 32;
        
        // Random optimizer
        if (space->optimizer_count > 0)
            current.optimizer = space->optimizers[rand() % space->optimizer_count];
        else
            current.optimizer = OPT_ADAM;
        
        // Random layer count
        if (space->hidden_layer_count_options > 0)
            current.hidden_layer_count = space->hidden_layer_counts[
                rand() % space->hidden_layer_count_options];
        else
            current.hidden_layer_count = 1;
        
        // Random layer size (base for topology generation)
        int layer_size = 64;
        if (space->hidden_layer_size_count > 0)
            layer_size = space->hidden_layer_sizes[rand() % space->hidden_layer_size_count];
        
        // Random topology pattern
        if (space->topology_pattern_count > 0)
            current.topology_pattern = space->topology_patterns[
                rand() % space->topology_pattern_count];
        else
            current.topology_pattern = TOPOLOGY_CONSTANT;
        
        // Generate layer sizes based on topology
        hypertune_generate_topology(current.topology_pattern, layer_size, 
                                   current.hidden_layer_count, current.hidden_layer_sizes);
        
        // Random activation(s)
        Activation_type base_act = ACTIVATION_RELU;
        if (space->hidden_activation_count > 0)
            base_act = space->hidden_activations[rand() % space->hidden_activation_count];
        
        // Set per-layer activations
        for (int l = 0; l < current.hidden_layer_count; l++) {
            if (space->search_per_layer_activation && space->hidden_activation_count > 1) {
                current.hidden_activations[l] = space->hidden_activations[
                    rand() % space->hidden_activation_count];
            } else {
                current.hidden_activations[l] = base_act;
            }
        }
        
        // Create and train network
        PNetwork net = hypertune_create_network(
            &current, input_size, output_size,
            output_activation, loss_type);
        
        if (!net) {
            current.score = -1.0f;
        } else {
            ann_set_epoch_limit(net, space->epoch_limit);
            ann_set_convergence(net, space->convergence_epsilon);
            
            double start_time = get_time_ms();
            ann_train_network(net, 
                split->train_inputs, 
                split->train_outputs,
                split->train_rows);
            double end_time = get_time_ms();
            
            current.training_time_ms = (real)(end_time - start_time);
            current.score = options->score_func(
                net, split->val_inputs, split->val_outputs,
                options->user_data);
            
            ann_free_network(net);
        }
        
        // Store result
        if (results && trial < max_results)
            results[trial] = current;
        
        // Update best
        if (current.score > best.score)
            best = current;
        
        // Progress callback
        if (options->progress_func) {
            options->progress_func(trial + 1, num_trials,
                &best, &current, options->user_data);
        } else if (options->verbosity >= 1) {
            printf("[%d/%d] ", trial + 1, num_trials);
            hypertune_print_result(&current);
        }
    }
    
    if (best_result)
        *best_result = best;
    
    if (options->verbosity >= 1) {
        printf("\nBest result: ");
        hypertune_print_result(&best);
    }
    
    return num_trials;
}
