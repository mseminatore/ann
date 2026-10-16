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
    "SGD", "Momentum", "RMSProp", "Adagrad", "Adam"
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
// SHARED TRIAL HELPERS
// ============================================================================
//
// Every search strategy describes a trial as a TrialConfig: an index into each
// categorical list of the HyperparamSpace, plus the learning rate in "search
// units" (log(lr) when learning_rate_log_scale is set, otherwise lr). A list
// with zero entries is treated as a single default choice, so all strategies
// see the same dimensions with the same fallbacks.

enum {
    DIM_BATCH,
    DIM_OPTIMIZER,
    DIM_LAYER_COUNT,
    DIM_LAYER_SIZE,
    DIM_TOPOLOGY,
    DIM_ACTIVATION,     // one per hidden layer: DIM_ACTIVATION + layer
    DIM_COUNT = DIM_ACTIVATION + HYPERTUNE_MAX_HIDDEN_LAYERS
};

typedef struct {
    real lr;            // learning rate in search units
    int cat[DIM_COUNT]; // index into each categorical list
} TrialConfig;

static real rand_uniform(void)
{
    return (real)rand() / (real)RAND_MAX;
}

// Standard normal sample (Box-Muller transform)
static real rand_normal(void)
{
    real u1 = ((real)rand() + 1.0f) / ((real)RAND_MAX + 2.0f);
    real u2 = (real)rand() / (real)RAND_MAX;
    return (real)sqrt(-2.0f * log(u1)) * (real)cos(2.0f * 3.14159265358979f * u2);
}

// Number of choices for a categorical dimension (always >= 1)
static int dim_choices(const HyperparamSpace *space, int dim)
{
    int count;

    switch (dim) {
        case DIM_BATCH:       count = space->batch_size_count; break;
        case DIM_OPTIMIZER:   count = space->optimizer_count; break;
        case DIM_LAYER_COUNT: count = space->hidden_layer_count_options; break;
        case DIM_LAYER_SIZE:  count = space->hidden_layer_size_count; break;
        case DIM_TOPOLOGY:    count = space->topology_pattern_count; break;
        default:              count = space->hidden_activation_count; break;
    }
    return count > 0 ? count : 1;
}

// Learning rate bounds in search units
static void lr_search_bounds(const HyperparamSpace *space, real *lo, real *hi)
{
    if (space->learning_rate_log_scale) {
        *lo = (real)log(space->learning_rate_min);
        *hi = (real)log(space->learning_rate_max);
    } else {
        *lo = space->learning_rate_min;
        *hi = space->learning_rate_max;
    }
}

// Map u in [0,1] to a learning rate in search units
static real lr_from_unit(const HyperparamSpace *space, real u)
{
    real lo, hi;
    lr_search_bounds(space, &lo, &hi);
    return lo + u * (hi - lo);
}

static int cfg_layer_count(const HyperparamSpace *space, const TrialConfig *cfg)
{
    int count = (space->hidden_layer_count_options > 0)
        ? space->hidden_layer_counts[cfg->cat[DIM_LAYER_COUNT]] : 1;

    if (count > HYPERTUNE_MAX_HIDDEN_LAYERS)
        count = HYPERTUNE_MAX_HIDDEN_LAYERS;
    return count;
}

// Whether a dimension affects the network built from cfg. Activations past the
// first layer only matter with per-layer search on and when that layer exists.
static int dim_active(const HyperparamSpace *space, const TrialConfig *cfg, int dim)
{
    int layer = dim - DIM_ACTIVATION;

    if (layer <= 0)
        return 1;
    return space->search_per_layer_activation && layer < cfg_layer_count(space, cfg);
}

static void sample_random_config(const HyperparamSpace *space, TrialConfig *cfg)
{
    cfg->lr = lr_from_unit(space, rand_uniform());

    for (int d = 0; d <= DIM_ACTIVATION; d++)
        cfg->cat[d] = rand() % dim_choices(space, d);

    for (int d = DIM_ACTIVATION + 1; d < DIM_COUNT; d++)
        cfg->cat[d] = space->search_per_layer_activation
            ? rand() % dim_choices(space, d)
            : cfg->cat[DIM_ACTIVATION];
}

// Fill the configuration fields of a result from a TrialConfig
static void decode_config(const HyperparamSpace *space, const TrialConfig *cfg, HypertuneResult *out)
{
    out->learning_rate = space->learning_rate_log_scale ? (real)exp(cfg->lr) : cfg->lr;

    out->batch_size = (space->batch_size_count > 0)
        ? space->batch_sizes[cfg->cat[DIM_BATCH]] : 32;
    out->optimizer = (space->optimizer_count > 0)
        ? space->optimizers[cfg->cat[DIM_OPTIMIZER]] : OPT_ADAM;
    out->hidden_layer_count = cfg_layer_count(space, cfg);
    out->topology_pattern = (space->topology_pattern_count > 0)
        ? space->topology_patterns[cfg->cat[DIM_TOPOLOGY]] : TOPOLOGY_CONSTANT;

    int base_size = (space->hidden_layer_size_count > 0)
        ? space->hidden_layer_sizes[cfg->cat[DIM_LAYER_SIZE]] : 64;
    hypertune_generate_topology(out->topology_pattern, base_size,
                               out->hidden_layer_count, out->hidden_layer_sizes);

    for (int l = 0; l < out->hidden_layer_count; l++) {
        int idx = space->search_per_layer_activation
            ? cfg->cat[DIM_ACTIVATION + l] : cfg->cat[DIM_ACTIVATION];
        out->hidden_activations[l] = (space->hidden_activation_count > 0)
            ? space->hidden_activations[idx] : ACTIVATION_RELU;
    }
}

// Build, train and score the network described by trial's configuration
static void evaluate_trial(
    const HyperparamSpace *space,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type,
    const DataSplit *split,
    const HypertuneOptions *options,
    HypertuneResult *trial)
{
    PNetwork net = hypertune_create_network(trial, input_size, output_size,
                                            output_activation, loss_type);
    if (!net) {
        trial->score = -1.0f;
        return;
    }

    ann_set_epoch_limit(net, space->epoch_limit);
    ann_set_convergence(net, space->convergence_epsilon);

    double start_time = get_time_ms();
    ann_train_network(net, split->train_inputs, split->train_outputs, split->train_rows);
    double end_time = get_time_ms();

    trial->training_time_ms = (real)(end_time - start_time);
    trial->score = options->score_func(net, split->val_inputs, split->val_outputs,
                                       options->user_data);
    ann_free_network(net);
}

static void init_best(HypertuneResult *best)
{
    hypertune_result_init(best);
    best->score = -999999.0f;
}

// Store a finished trial, update the running best and report progress.
// phase is an optional label printed after the trial counter.
static void record_trial(
    const HypertuneOptions *options,
    const HypertuneResult *current,
    int trial,
    int total_trials,
    const char *phase,
    HypertuneResult *results,
    int max_results,
    HypertuneResult *best)
{
    if (results && trial < max_results)
        results[trial] = *current;

    if (current->score > best->score)
        *best = *current;

    if (options->progress_func) {
        options->progress_func(trial + 1, total_trials, best, current, options->user_data);
    } else if (options->verbosity >= 1) {
        printf("[%d/%d] %s", trial + 1, total_trials, phase ? phase : "");
        hypertune_print_result(current);
    }
}

static void finish_search(const HypertuneOptions *options, const HypertuneResult *best,
                          HypertuneResult *best_result)
{
    if (best_result)
        *best_result = *best;

    if (options->verbosity >= 1) {
        printf("\nBest result: ");
        hypertune_print_result(best);
    }
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
    init_best(&best);

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

                                evaluate_trial(space, input_size, output_size,
                                               output_activation, loss_type,
                                               split, options, &current);
                                record_trial(options, &current, trial, total_trials, NULL,
                                             results, max_results, &best);
                                trial++;
                            }
                        }
                    }
                }
            }
        }
    }

    finish_search(options, &best, best_result);
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
    init_best(&best);

    for (int trial = 0; trial < num_trials; trial++) {
        TrialConfig cfg;
        sample_random_config(space, &cfg);

        HypertuneResult current;
        hypertune_result_init(&current);
        current.trial_id = trial + 1;
        decode_config(space, &cfg, &current);

        evaluate_trial(space, input_size, output_size, output_activation, loss_type,
                       split, options, &current);
        record_trial(options, &current, trial, num_trials, NULL,
                     results, max_results, &best);
    }

    finish_search(options, &best, best_result);
    return num_trials;
}

// ============================================================================
// BAYESIAN OPTIMIZATION - GAUSSIAN PROCESS
// ============================================================================

// Standard normal CDF: 0.5 * (1 + erf(x / sqrt(2))), with erf from
// Abramowitz and Stegun 7.1.26
static real standard_normal_cdf(real x)
{
    const real a1 =  0.254829592f;
    const real a2 = -0.284496736f;
    const real a3 =  1.421413741f;
    const real a4 = -1.453152027f;
    const real a5 =  1.061405429f;
    const real p  =  0.3275911f;

    int sign = (x < 0) ? -1 : 1;
    real z = (real)fabs(x) * 0.7071067812f;  // x / sqrt(2)

    real t = 1.0f / (1.0f + p * z);
    real y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (real)exp(-z * z);

    return 0.5f * (1.0f + sign * y);
}

// Standard normal PDF
static real standard_normal_pdf(real x)
{
    const real inv_sqrt_2pi = 0.3989422804f;
    return inv_sqrt_2pi * (real)exp(-0.5f * x * x);
}

// Squared-exponential kernel over per-dimension distances. Continuous
// dimensions use a - b; categorical dimensions use 0 when equal, else 1.
static real gp_kernel(const GaussianProcess *gp, const real *a, const real *b)
{
    real sum = 0.0f;

    for (int d = 0; d < gp->n_dims; d++) {
        real diff;
        if (gp->dim_types[d] == GP_DIM_CATEGORICAL)
            diff = (a[d] == b[d]) ? 0.0f : 1.0f;
        else
            diff = a[d] - b[d];

        real ls = gp->length_scale * gp->ard_scales[d];
        sum += (diff * diff) / (ls * ls);
    }
    return gp->signal_variance * (real)exp(-0.5f * sum);
}

void gp_init(GaussianProcess *gp, int n_dims)
{
    if (!gp) return;

    memset(gp, 0, sizeof(GaussianProcess));
    if (n_dims < 1) n_dims = 1;
    if (n_dims > GP_MAX_DIMS) n_dims = GP_MAX_DIMS;

    gp->n_dims = n_dims;
    gp->length_scale = 0.3f;      // Default length scale
    gp->noise_variance = 0.01f;   // Observation noise
    gp->signal_variance = 1.0f;   // Signal amplitude
    gp->normalize_y = 1;          // Standardize scores before fitting

    for (int d = 0; d < GP_MAX_DIMS; d++) {
        gp->ard_scales[d] = 1.0f;
        gp->dim_types[d] = GP_DIM_CONTINUOUS;
    }
}

int gp_add_observation(GaussianProcess *gp, const real *x, real y)
{
    if (!gp || !x)
        return ERR_NULL_PTR;

    if (gp->n_observations >= GP_MAX_OBSERVATIONS)
        return ERR_INVALID;

    int idx = gp->n_observations;
    for (int i = 0; i < gp->n_dims; i++) {
        gp->X[idx][i] = x[i];
    }
    gp->y[idx] = y;
    gp->n_observations++;

    return ERR_OK;
}

// Cholesky factor of the kernel matrix and the solved weights. Computed once
// and reused for every prediction until the observations or hyperparameters change.
typedef struct {
    int n;
    real y_mean;                                        // score normalization
    real y_scale;
    real L[GP_MAX_OBSERVATIONS * GP_MAX_OBSERVATIONS];  // lower-triangular factor, stride n
    real alpha[GP_MAX_OBSERVATIONS];                    // K^-1 * y
    real log_det_half;                                  // sum(log(L_ii))
    real y_dot_alpha;                                   // y^T * K^-1 * y
} GpFit;

static void gp_fit(const GaussianProcess *gp, GpFit *fit)
{
    int n = gp->n_observations;
    real *L = fit->L;
    real yn[GP_MAX_OBSERVATIONS];

    fit->n = n;
    fit->y_mean = 0.0f;
    fit->y_scale = 1.0f;

    if (gp->normalize_y && n > 0) {
        real mean = 0.0f, var = 0.0f;
        for (int i = 0; i < n; i++) mean += gp->y[i];
        mean /= (real)n;
        for (int i = 0; i < n; i++) var += (gp->y[i] - mean) * (gp->y[i] - mean);
        var /= (real)n;

        fit->y_mean = mean;
        if (var > 1e-12f)
            fit->y_scale = (real)sqrt(var);
    }
    for (int i = 0; i < n; i++)
        yn[i] = (gp->y[i] - fit->y_mean) / fit->y_scale;

    // Lower triangle of K with noise on the diagonal
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            L[i * n + j] = gp_kernel(gp, gp->X[i], gp->X[j]);
            if (i == j)
                L[i * n + j] += gp->noise_variance;
        }
    }

    // In-place Cholesky decomposition: K = L * L^T
    fit->log_det_half = 0.0f;
    for (int j = 0; j < n; j++) {
        real sum = L[j * n + j];
        for (int k = 0; k < j; k++)
            sum -= L[j * n + k] * L[j * n + k];
        L[j * n + j] = (real)sqrt(sum > 1e-10f ? sum : 1e-10f);
        fit->log_det_half += (real)log(L[j * n + j]);

        for (int i = j + 1; i < n; i++) {
            real s = L[i * n + j];
            for (int k = 0; k < j; k++)
                s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / L[j * n + j];
        }
    }

    // Solve L * z = y (forward), then L^T * alpha = z (backward)
    real z[GP_MAX_OBSERVATIONS];
    for (int i = 0; i < n; i++) {
        real sum = yn[i];
        for (int j = 0; j < i; j++)
            sum -= L[i * n + j] * z[j];
        z[i] = sum / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        real sum = z[i];
        for (int j = i + 1; j < n; j++)
            sum -= L[j * n + i] * fit->alpha[j];
        fit->alpha[i] = sum / L[i * n + i];
    }

    fit->y_dot_alpha = 0.0f;
    for (int i = 0; i < n; i++)
        fit->y_dot_alpha += yn[i] * fit->alpha[i];
}

static void gp_predict_fit(const GaussianProcess *gp, const GpFit *fit, const real *x,
                           real *mean, real *variance)
{
    int n = fit->n;
    const real *L = fit->L;

    if (n == 0) {
        *mean = 0.0f;
        *variance = gp->signal_variance;
        return;
    }

    // Predicted mean: k_star^T * alpha
    real k_star[GP_MAX_OBSERVATIONS];
    real m = 0.0f;
    for (int i = 0; i < n; i++) {
        k_star[i] = gp_kernel(gp, x, gp->X[i]);
        m += k_star[i] * fit->alpha[i];
    }

    // Predicted variance: k(x,x) - v^T * v, where L * v = k_star
    real v[GP_MAX_OBSERVATIONS];
    real v_dot_v = 0.0f;
    for (int i = 0; i < n; i++) {
        real sum = k_star[i];
        for (int j = 0; j < i; j++)
            sum -= L[i * n + j] * v[j];
        v[i] = sum / L[i * n + i];
        v_dot_v += v[i] * v[i];
    }

    real var = gp_kernel(gp, x, x) - v_dot_v;
    if (var < 1e-10f) var = 1e-10f;

    *mean = fit->y_mean + fit->y_scale * m;
    *variance = var * fit->y_scale * fit->y_scale;
}

// Log marginal likelihood of the (normalized) observations under a fit
static real gp_log_marginal(const GpFit *fit)
{
    return -0.5f * fit->y_dot_alpha - fit->log_det_half - 0.9189385332f * (real)fit->n;
}

void gp_predict(const GaussianProcess *gp, const real *x, real *mean, real *variance)
{
    if (!gp || !x || !mean || !variance) return;

    GpFit fit;
    gp_fit(gp, &fit);
    gp_predict_fit(gp, &fit, x, mean, variance);
}

// Coordinate search over per-dimension length scales and noise, keeping
// whichever value most improves the log marginal likelihood
static void gp_tune(GaussianProcess *gp, GpFit *scratch)
{
    static const real scales[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
    static const real noises[] = {1e-4f, 1e-3f, 1e-2f, 1e-1f};
    const int n_scales = (int)(sizeof(scales) / sizeof(scales[0]));
    const int n_noises = (int)(sizeof(noises) / sizeof(noises[0]));

    gp_fit(gp, scratch);
    real best = gp_log_marginal(scratch);

    for (int pass = 0; pass < 2; pass++) {
        for (int d = 0; d < gp->n_dims; d++) {
            real keep = gp->ard_scales[d];
            for (int s = 0; s < n_scales; s++) {
                if (scales[s] == keep) continue;
                gp->ard_scales[d] = scales[s];
                gp_fit(gp, scratch);
                real lml = gp_log_marginal(scratch);
                if (lml > best) {
                    best = lml;
                    keep = scales[s];
                }
            }
            gp->ard_scales[d] = keep;
        }

        real keep = gp->noise_variance;
        for (int s = 0; s < n_noises; s++) {
            if (noises[s] == keep) continue;
            gp->noise_variance = noises[s];
            gp_fit(gp, scratch);
            real lml = gp_log_marginal(scratch);
            if (lml > best) {
                best = lml;
                keep = noises[s];
            }
        }
        gp->noise_variance = keep;
    }
}

int gp_fit_hyperparams(GaussianProcess *gp)
{
    if (!gp)
        return ERR_NULL_PTR;

    // Too few points to say anything about relevance
    if (gp->n_observations < 3)
        return ERR_OK;

    GpFit *scratch = (GpFit *)malloc(sizeof(GpFit));
    if (!scratch)
        return ERR_ALLOC;

    gp_tune(gp, scratch);
    free(scratch);
    return ERR_OK;
}

real gp_expected_improvement(real mean, real variance, real best_y, real xi)
{
    if (variance <= 0.0f)
        return 0.0f;

    real std = (real)sqrt(variance);
    real z = (mean - best_y - xi) / std;

    return (mean - best_y - xi) * standard_normal_cdf(z) + std * standard_normal_pdf(z);
}

void bayesian_options_init(BayesianOptions *opts)
{
    if (!opts) return;

    opts->n_initial = 10;
    opts->n_iterations = 20;
    opts->n_candidates = 100;
    opts->exploration_weight = 0.01f;
    opts->fit_hyperparams = 1;
}

// ============================================================================
// BAYESIAN OPTIMIZATION SEARCH
// ============================================================================

// GP coordinates for a trial: learning rate, then one per TrialConfig dimension
#define BO_DIMS (1 + DIM_COUNT)

// Dimensions whose choices are ordered (modeled as continuous positions)
static int dim_is_ordinal(int dim)
{
    return dim == DIM_BATCH || dim == DIM_LAYER_COUNT || dim == DIM_LAYER_SIZE;
}

static void bo_setup_dims(GaussianProcess *gp)
{
    gp->dim_types[0] = GP_DIM_CONTINUOUS;
    for (int d = 0; d < DIM_COUNT; d++)
        gp->dim_types[1 + d] = dim_is_ordinal(d) ? GP_DIM_CONTINUOUS : GP_DIM_CATEGORICAL;
}

// Encode a trial as GP coordinates. The learning rate and ordinal dimensions map
// to [0,1]; categorical dimensions keep their index. Dimensions that do not
// apply to the trial (activations of absent layers) are encoded as -1.
static void bo_encode(const HyperparamSpace *space, const TrialConfig *cfg, real *x)
{
    real lo, hi;
    lr_search_bounds(space, &lo, &hi);
    x[0] = (hi > lo) ? (cfg->lr - lo) / (hi - lo) : 0.0f;

    for (int d = 0; d < DIM_COUNT; d++) {
        int n_choices = dim_choices(space, d);
        if (!dim_active(space, cfg, d))
            x[1 + d] = -1.0f;
        else if (dim_is_ordinal(d))
            x[1 + d] = (n_choices > 1) ? (real)cfg->cat[d] / (real)(n_choices - 1) : 0.0f;
        else
            x[1 + d] = (real)cfg->cat[d];
    }
}

// Local candidate: jitter the learning rate and resample one categorical dimension
static void bo_mutate(const HyperparamSpace *space, TrialConfig *cfg)
{
    real lo, hi;
    lr_search_bounds(space, &lo, &hi);
    cfg->lr += 0.1f * (hi - lo) * rand_normal();
    if (cfg->lr < lo) cfg->lr = lo;
    if (cfg->lr > hi) cfg->lr = hi;

    int eligible[DIM_COUNT];
    int n_eligible = 0;
    for (int d = 0; d < DIM_COUNT; d++) {
        if (dim_choices(space, d) > 1 && dim_active(space, cfg, d))
            eligible[n_eligible++] = d;
    }
    if (n_eligible > 0) {
        int d = eligible[rand() % n_eligible];
        cfg->cat[d] = rand() % dim_choices(space, d);
    }
}

int hypertune_bayesian_search(
    const HyperparamSpace *space,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type,
    const DataSplit *split,
    const HypertuneOptions *tune_options,
    const BayesianOptions *bayes_options,
    HypertuneResult *results,
    int max_results,
    HypertuneResult *best_result)
{
    if (!space || !split || !tune_options || !tune_options->score_func || !bayes_options)
        return ERR_NULL_PTR;

    int total_trials = bayes_options->n_initial + bayes_options->n_iterations;

    if (tune_options->verbosity >= 1) {
        printf("Starting Bayesian optimization: %d initial + %d BO iterations\n",
               bayes_options->n_initial, bayes_options->n_iterations);
    }

    GaussianProcess gp;
    gp_init(&gp, BO_DIMS);
    bo_setup_dims(&gp);

    GpFit *fit = (GpFit *)malloc(sizeof(GpFit));
    if (!fit)
        return ERR_ALLOC;

    HypertuneResult best;
    init_best(&best);
    TrialConfig best_cfg;
    memset(&best_cfg, 0, sizeof(best_cfg));

    unsigned seed = tune_options->seed ? tune_options->seed : (unsigned)time(NULL);
    srand(seed);

    for (int trial = 0; trial < total_trials; trial++) {
        int initial = (trial < bayes_options->n_initial || gp.n_observations == 0);
        real best_ei = -999999.0f;
        TrialConfig cfg;

        // Phase 1: initial random sampling (also the fallback with no candidates)
        sample_random_config(space, &cfg);

        if (!initial) {
            // Phase 2: fit the GP once, then pick the candidate maximizing
            // Expected Improvement. Half the candidates are random, half are
            // local perturbations of the best trial so far.
            if (bayes_options->fit_hyperparams && gp.n_observations >= 3)
                gp_tune(&gp, fit);
            gp_fit(&gp, fit);

            for (int c = 0; c < bayes_options->n_candidates; c++) {
                TrialConfig cand;
                if (c % 2) {
                    cand = best_cfg;
                    bo_mutate(space, &cand);
                } else {
                    sample_random_config(space, &cand);
                }

                real x[BO_DIMS];
                bo_encode(space, &cand, x);

                real mean, variance;
                gp_predict_fit(&gp, fit, x, &mean, &variance);

                real ei = gp_expected_improvement(mean, variance, best.score,
                                                  bayes_options->exploration_weight);
                if (ei > best_ei) {
                    best_ei = ei;
                    cfg = cand;
                }
            }
        }

        HypertuneResult current;
        hypertune_result_init(&current);
        current.trial_id = trial + 1;
        decode_config(space, &cfg, &current);

        evaluate_trial(space, input_size, output_size, output_activation, loss_type,
                       split, tune_options, &current);

        real x[BO_DIMS];
        bo_encode(space, &cfg, x);
        gp_add_observation(&gp, x, current.score);

        if (trial == 0 || current.score > best.score)
            best_cfg = cfg;

        char phase[48];
        if (initial)
            strcpy(phase, "(init) ");
        else
            sprintf(phase, "(BO, EI=%.4f) ", best_ei);
        record_trial(tune_options, &current, trial, total_trials, phase,
                     results, max_results, &best);
    }

    free(fit);
    finish_search(tune_options, &best, best_result);
    return total_trials;
}

// ============================================================================
// TPE (Tree-structured Parzen Estimator) Implementation
// ============================================================================

void tpe_options_init(TPEOptions *opts)
{
    if (!opts) return;
    
    opts->n_startup = 10;           // Random trials before TPE
    opts->gamma = 0.25f;            // Top 25% are "good"
    opts->n_candidates = 24;        // Samples from l(x) per iteration
    opts->n_iterations = 40;        // TPE iterations after startup
    opts->bandwidth_factor = 1.0f;  // KDE bandwidth multiplier
}

// Gaussian kernel for KDE
static real kde_gaussian_kernel(real x, real xi, real bandwidth)
{
    real z = (x - xi) / bandwidth;
    return (real)exp(-0.5f * z * z) / (bandwidth * 2.5066282746f);  // sqrt(2*pi)
}

// Evaluate 1D Gaussian KDE at point x
static real kde_evaluate_1d(real x, const real *samples, int n_samples, real bandwidth)
{
    if (n_samples == 0) return 1e-10f;
    
    real sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        sum += kde_gaussian_kernel(x, samples[i], bandwidth);
    }
    return sum / (real)n_samples + 1e-10f;  // Add small value to avoid division by zero
}

// Sample from 1D Gaussian KDE (sample from a random component then add noise)
static real kde_sample_1d(const real *samples, int n_samples, real bandwidth)
{
    if (n_samples == 0) return (real)rand() / (real)RAND_MAX;
    
    // Pick a random sample as the center
    int idx = rand() % n_samples;
    real center = samples[idx];
    
    // Add Gaussian noise with the bandwidth as std dev
    return center + rand_normal() * bandwidth;
}

// Evaluate categorical probability (count-based with Laplace smoothing)
static real categorical_prob(int value, const int *samples, int n_samples, int n_categories)
{
    if (n_samples == 0) return 1.0f / (real)n_categories;
    
    int count = 0;
    for (int i = 0; i < n_samples; i++) {
        if (samples[i] == value) count++;
    }
    
    // Laplace smoothing
    return ((real)count + 1.0f) / ((real)n_samples + (real)n_categories);
}

// Sample from categorical distribution based on observed samples
static int categorical_sample(const int *samples, int n_samples, int n_categories)
{
    if (n_samples == 0) return rand() % n_categories;
    
    // Compute probabilities with Laplace smoothing
    real probs[16];  // Max categories
    real sum = 0.0f;
    
    for (int c = 0; c < n_categories && c < 16; c++) {
        probs[c] = categorical_prob(c, samples, n_samples, n_categories);
        sum += probs[c];
    }
    
    // Sample
    real r = ((real)rand() / (real)RAND_MAX) * sum;
    real cumsum = 0.0f;
    for (int c = 0; c < n_categories && c < 16; c++) {
        cumsum += probs[c];
        if (r <= cumsum) return c;
    }
    return n_categories - 1;
}

// Scott's rule for KDE bandwidth selection
static real scott_bandwidth(const real *samples, int n_samples)
{
    if (n_samples < 2) return 0.1f;
    
    // Compute std dev
    real mean = 0.0f;
    for (int i = 0; i < n_samples; i++) mean += samples[i];
    mean /= (real)n_samples;
    
    real var = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        real diff = samples[i] - mean;
        var += diff * diff;
    }
    var /= (real)(n_samples - 1);
    real std = (real)sqrt(var);
    if (std < 0.01f) std = 0.01f;
    
    // Scott's rule: h = n^(-1/5) * std * 1.06
    return 1.06f * std * (real)pow((double)n_samples, -0.2);
}

int hypertune_tpe_search(
    const HyperparamSpace *space,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type,
    const DataSplit *split,
    const HypertuneOptions *tune_options,
    const TPEOptions *tpe_options,
    HypertuneResult *results,
    int max_results,
    HypertuneResult *best_result)
{
    if (!space || !split || !tune_options || !tune_options->score_func || !tpe_options)
        return ERR_NULL_PTR;

    int total_trials = tpe_options->n_startup + tpe_options->n_iterations;
    if (total_trials > TPE_MAX_OBSERVATIONS) total_trials = TPE_MAX_OBSERVATIONS;

    // Observation history
    TrialConfig obs[TPE_MAX_OBSERVATIONS];
    real obs_scores[TPE_MAX_OBSERVATIONS];
    int n_obs = 0;

    HypertuneResult best;
    init_best(&best);

    unsigned seed = tune_options->seed ? tune_options->seed : (unsigned)time(NULL);
    srand(seed);

    real lr_lo, lr_hi;
    lr_search_bounds(space, &lr_lo, &lr_hi);

    if (tune_options->verbosity >= 1) {
        printf("\n=== TPE Search ===\n");
        printf("Startup trials: %d, TPE iterations: %d\n",
               tpe_options->n_startup, tpe_options->n_iterations);
        printf("Gamma: %.2f (top %.0f%% are 'good')\n\n",
               tpe_options->gamma, tpe_options->gamma * 100.0f);
    }

    for (int trial = 0; trial < total_trials; trial++) {
        TrialConfig cfg;
        int tpe_phase = (trial >= tpe_options->n_startup && n_obs >= 2);

        memset(&cfg, 0, sizeof(cfg));

        if (!tpe_phase || tpe_options->n_candidates <= 0) {
            // Random sampling during startup
            sample_random_config(space, &cfg);
        } else {
            // Sort observations by score (descending) to find good/bad split
            int sorted_indices[TPE_MAX_OBSERVATIONS];
            for (int i = 0; i < n_obs; i++) sorted_indices[i] = i;

            // Simple selection sort for indices (n_obs is small)
            for (int i = 0; i < n_obs - 1; i++) {
                int max_idx = i;
                for (int j = i + 1; j < n_obs; j++) {
                    if (obs_scores[sorted_indices[j]] > obs_scores[sorted_indices[max_idx]])
                        max_idx = j;
                }
                int tmp = sorted_indices[i];
                sorted_indices[i] = sorted_indices[max_idx];
                sorted_indices[max_idx] = tmp;
            }

            int n_good = (int)(tpe_options->gamma * (real)n_obs);
            if (n_good < 1) n_good = 1;
            if (n_good > n_obs) n_good = n_obs;
            int n_bad = n_obs - n_good;

            // Learning rate samples for the good (l) and bad (g) densities
            real good_lr[TPE_MAX_OBSERVATIONS], bad_lr[TPE_MAX_OBSERVATIONS];
            for (int i = 0; i < n_good; i++) good_lr[i] = obs[sorted_indices[i]].lr;
            for (int i = 0; i < n_bad; i++)  bad_lr[i] = obs[sorted_indices[n_good + i]].lr;

            real bw_good = scott_bandwidth(good_lr, n_good) * tpe_options->bandwidth_factor;
            real bw_bad = scott_bandwidth(bad_lr, n_bad) * tpe_options->bandwidth_factor;

            // Categorical samples per dimension. An observation only contributes
            // to dimensions that were active for it (e.g. layer 3's activation
            // only counts for networks with at least 3 hidden layers).
            int good_cat[DIM_COUNT][TPE_MAX_OBSERVATIONS], bad_cat[DIM_COUNT][TPE_MAX_OBSERVATIONS];
            int n_good_cat[DIM_COUNT], n_bad_cat[DIM_COUNT];

            for (int d = 0; d < DIM_COUNT; d++) {
                n_good_cat[d] = n_bad_cat[d] = 0;
                for (int i = 0; i < n_obs; i++) {
                    const TrialConfig *o = &obs[sorted_indices[i]];
                    if (!dim_active(space, o, d))
                        continue;
                    if (i < n_good)
                        good_cat[d][n_good_cat[d]++] = o->cat[d];
                    else
                        bad_cat[d][n_bad_cat[d]++] = o->cat[d];
                }
            }

            // TPE: sample candidates from l(x), keep the one maximizing l(x)/g(x)
            real best_ratio = -1e30f;

            for (int c = 0; c < tpe_options->n_candidates; c++) {
                TrialConfig cand;

                cand.lr = kde_sample_1d(good_lr, n_good, bw_good);
                if (cand.lr < lr_lo) cand.lr = lr_lo;
                if (cand.lr > lr_hi) cand.lr = lr_hi;

                for (int d = 0; d < DIM_COUNT; d++)
                    cand.cat[d] = categorical_sample(good_cat[d], n_good_cat[d],
                                                     dim_choices(space, d));

                // EI ratio (log space for numerical stability)
                real ratio = (real)log(kde_evaluate_1d(cand.lr, good_lr, n_good, bw_good) /
                                       kde_evaluate_1d(cand.lr, bad_lr, n_bad, bw_bad));

                for (int d = 0; d < DIM_COUNT; d++) {
                    if (!dim_active(space, &cand, d))
                        continue;
                    int n_choices = dim_choices(space, d);
                    real l = categorical_prob(cand.cat[d], good_cat[d], n_good_cat[d], n_choices);
                    real g = categorical_prob(cand.cat[d], bad_cat[d], n_bad_cat[d], n_choices);
                    ratio += (real)log(l / g);
                }

                if (c == 0 || ratio > best_ratio) {
                    best_ratio = ratio;
                    cfg = cand;
                }
            }
        }

        HypertuneResult current;
        hypertune_result_init(&current);
        current.trial_id = trial + 1;
        decode_config(space, &cfg, &current);

        evaluate_trial(space, input_size, output_size, output_activation, loss_type,
                       split, tune_options, &current);

        // Store observation
        if (n_obs < TPE_MAX_OBSERVATIONS) {
            obs[n_obs] = cfg;
            obs_scores[n_obs] = current.score;
            n_obs++;
        }

        record_trial(tune_options, &current, trial, total_trials,
                     tpe_phase ? "(TPE) " : "(random) ", results, max_results, &best);
    }

    finish_search(tune_options, &best, best_result);
    return total_trials;
}
