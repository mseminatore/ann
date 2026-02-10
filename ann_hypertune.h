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

#pragma once

#ifndef __ANN_HYPERTUNE_H
#define __ANN_HYPERTUNE_H

#include "ann.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------
// Configuration limits
//------------------------------
#define HYPERTUNE_MAX_BATCH_SIZES     8
#define HYPERTUNE_MAX_OPTIMIZERS      8
#define HYPERTUNE_MAX_ACTIVATIONS     8
#define HYPERTUNE_MAX_HIDDEN_LAYERS   5
#define HYPERTUNE_MAX_LAYER_SIZES     16

//------------------------------
// Topology patterns
//------------------------------
typedef enum {
    TOPOLOGY_CONSTANT,    // All hidden layers have same size
    TOPOLOGY_PYRAMID,     // Layers decrease in size toward output (e.g., 128-64-32)
    TOPOLOGY_FUNNEL,      // Layers increase then decrease (e.g., 64-128-64)
    TOPOLOGY_INVERSE,     // Layers increase in size toward output (e.g., 32-64-128)
    TOPOLOGY_CUSTOM       // Use explicit sizes from hidden_layer_sizes array
} TopologyPattern;

//------------------------------
// Hyperparameter search space
//------------------------------
typedef struct {
    // Learning rate range (log scale recommended)
    real learning_rate_min;
    real learning_rate_max;
    int learning_rate_steps;        // number of values to try (0 = use min only)
    int learning_rate_log_scale;    // 1 = logarithmic spacing, 0 = linear

    // Batch sizes to try
    unsigned batch_sizes[HYPERTUNE_MAX_BATCH_SIZES];
    int batch_size_count;

    // Optimizers to try
    Optimizer_type optimizers[HYPERTUNE_MAX_OPTIMIZERS];
    int optimizer_count;

    // Hidden layer configurations
    int hidden_layer_counts[HYPERTUNE_MAX_HIDDEN_LAYERS];  // e.g., [1, 2, 3]
    int hidden_layer_count_options;

    // Nodes per hidden layer (base size for patterns, or explicit sizes)
    int hidden_layer_sizes[HYPERTUNE_MAX_LAYER_SIZES];     // e.g., [32, 64, 128]
    int hidden_layer_size_count;

    // Topology patterns to try
    TopologyPattern topology_patterns[4];  // e.g., [CONSTANT, PYRAMID]
    int topology_pattern_count;

    // Activations for hidden layers
    Activation_type hidden_activations[HYPERTUNE_MAX_ACTIVATIONS];
    int hidden_activation_count;

    // Per-layer activation search (if enabled, tries different activations per layer)
    int search_per_layer_activation;  // 0 = same activation for all, 1 = vary per layer

    // Epoch limit for each trial
    unsigned epoch_limit;

    // Convergence threshold
    real convergence_epsilon;

    // Early stopping patience (0 = disabled)
    int early_stopping_patience;
} HyperparamSpace;

//------------------------------
// Network configuration (result of a trial)
//------------------------------
typedef struct {
    // Training parameters
    real learning_rate;
    unsigned batch_size;
    Optimizer_type optimizer;
    
    // Topology
    int hidden_layer_count;
    int hidden_layer_sizes[HYPERTUNE_MAX_HIDDEN_LAYERS];
    Activation_type hidden_activations[HYPERTUNE_MAX_HIDDEN_LAYERS];  // per-layer activations
    TopologyPattern topology_pattern;
    
    // Results
    real score;             // user-defined score (higher = better)
    real training_time_ms;  // time to train in milliseconds
    int epochs_used;        // actual epochs before stopping
    int trial_id;           // sequential trial number
} HypertuneResult;

//------------------------------
// Callback function types
//------------------------------

/**
 * User-defined scoring function.
 * Returns a score where HIGHER is BETTER (optimizer maximizes this).
 * Examples: accuracy, F1-score, 1.0/loss, custom metric
 *
 * @param trained_net The trained network to evaluate
 * @param val_inputs Validation input data
 * @param val_outputs Validation expected outputs
 * @param user_data Optional user context
 * @return Score value (higher = better)
 */
typedef real (*HypertuneScoreFunc)(
    PNetwork trained_net,
    PTensor val_inputs,
    PTensor val_outputs,
    void *user_data
);

/**
 * Progress callback (optional).
 * Called after each trial completes.
 *
 * @param current_trial Current trial number (1-based)
 * @param total_trials Total number of trials
 * @param best_so_far Best result found so far
 * @param current_result Result of the current trial
 * @param user_data Optional user context
 */
typedef void (*HypertuneProgressFunc)(
    int current_trial,
    int total_trials,
    const HypertuneResult *best_so_far,
    const HypertuneResult *current_result,
    void *user_data
);

//------------------------------
// Hypertuning options
//------------------------------
typedef struct {
    // Required: scoring function
    HypertuneScoreFunc score_func;
    
    // Optional: progress callback
    HypertuneProgressFunc progress_func;
    
    // Optional: user data passed to callbacks
    void *user_data;
    
    // Random seed for reproducibility (0 = use time)
    unsigned seed;
    
    // Verbosity: 0=quiet, 1=normal, 2=verbose
    int verbosity;
} HypertuneOptions;

//------------------------------
// Data split structure
//------------------------------
typedef struct {
    PTensor train_inputs;
    PTensor train_outputs;
    PTensor val_inputs;
    PTensor val_outputs;
    int train_rows;
    int val_rows;
} DataSplit;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize a hyperparameter space with default values.
 * Sets reasonable defaults that can be customized.
 *
 * @param space Space to initialize
 */
void hypertune_space_init(HyperparamSpace *space);

/**
 * Initialize hypertuning options with defaults.
 *
 * @param options Options to initialize
 */
void hypertune_options_init(HypertuneOptions *options);

/**
 * Initialize a result structure.
 *
 * @param result Result to initialize
 */
void hypertune_result_init(HypertuneResult *result);

// ============================================================================
// DATA SPLITTING
// ============================================================================

/**
 * Split data into training and validation sets.
 * Uses holdout method with specified ratio.
 *
 * @param inputs Full input dataset
 * @param outputs Full output dataset
 * @param train_ratio Fraction for training (e.g., 0.8 for 80%)
 * @param shuffle 1 = shuffle before splitting, 0 = sequential split
 * @param seed Random seed for shuffling (0 = use time)
 * @param split Output: populated split structure
 * @return ERR_OK on success, error code on failure
 *
 * Note: Caller must free split tensors with hypertune_free_split()
 */
int hypertune_split_data(
    PTensor inputs,
    PTensor outputs,
    real train_ratio,
    int shuffle,
    unsigned seed,
    DataSplit *split
);

/**
 * Free tensors allocated by hypertune_split_data.
 *
 * @param split Split to free
 */
void hypertune_free_split(DataSplit *split);

// ============================================================================
// SEARCH ALGORITHMS
// ============================================================================

/**
 * Perform grid search over hyperparameter space.
 * Exhaustively tries all combinations of parameters.
 *
 * @param space Hyperparameter search space
 * @param input_size Number of input features
 * @param output_size Number of output classes/values
 * @param output_activation Activation for output layer
 * @param loss_type Loss function to use
 * @param split Training/validation data split
 * @param options Search options (score function, callbacks)
 * @param results Array to store results (caller allocates)
 * @param max_results Maximum results to store
 * @param best_result Output: best result found
 * @return Number of trials completed, or negative error code
 */
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
    HypertuneResult *best_result
);

/**
 * Perform random search over hyperparameter space.
 * Randomly samples parameter combinations.
 *
 * @param space Hyperparameter search space
 * @param num_trials Number of random trials to run
 * @param input_size Number of input features
 * @param output_size Number of output classes/values
 * @param output_activation Activation for output layer
 * @param loss_type Loss function to use
 * @param split Training/validation data split
 * @param options Search options
 * @param results Array to store results
 * @param max_results Maximum results to store
 * @param best_result Output: best result found
 * @return Number of trials completed, or negative error code
 */
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
    HypertuneResult *best_result
);

// ============================================================================
// RESULT ANALYSIS
// ============================================================================

/**
 * Print a hypertuning result summary.
 *
 * @param result Result to print
 */
void hypertune_print_result(const HypertuneResult *result);

/**
 * Print summary of top N results.
 *
 * @param results Array of results
 * @param count Number of results
 * @param top_n Number of top results to show
 */
void hypertune_print_summary(const HypertuneResult *results, int count, int top_n);

/**
 * Calculate total number of trials for grid search.
 *
 * @param space Hyperparameter space
 * @return Total number of combinations
 */
int hypertune_count_grid_trials(const HyperparamSpace *space);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create a network from a hypertuning result configuration.
 *
 * @param result Configuration to use
 * @param input_size Number of input features
 * @param output_size Number of output classes/values
 * @param output_activation Activation for output layer
 * @param loss_type Loss function to use
 * @return Newly created network, or NULL on error
 */
PNetwork hypertune_create_network(
    const HypertuneResult *result,
    int input_size,
    int output_size,
    Activation_type output_activation,
    Loss_type loss_type
);

/**
 * Default scoring function using accuracy.
 * Can be used directly or as a template for custom scorers.
 */
real hypertune_score_accuracy(
    PNetwork trained_net,
    PTensor val_inputs,
    PTensor val_outputs,
    void *user_data
);

/**
 * Generate layer sizes based on topology pattern.
 *
 * @param pattern Topology pattern to use
 * @param base_size Base size for the pattern
 * @param layer_count Number of hidden layers
 * @param output_sizes Array to fill with layer sizes (must be >= layer_count)
 */
void hypertune_generate_topology(
    TopologyPattern pattern,
    int base_size,
    int layer_count,
    int *output_sizes
);

/**
 * Get string name for topology pattern.
 *
 * @param pattern Topology pattern
 * @return Static string name
 */
const char* hypertune_topology_name(TopologyPattern pattern);

// ============================================================================
// BAYESIAN OPTIMIZATION
// ============================================================================

/**
 * Maximum observations for Gaussian Process.
 * Limits memory usage and keeps GP computations tractable.
 */
#define GP_MAX_OBSERVATIONS 100

/**
 * Gaussian Process state for Bayesian optimization.
 * Maintains observed points and their scores for the surrogate model.
 */
typedef struct {
    int n_observations;                      // Number of observations so far
    int n_dims;                              // Number of dimensions (2 for lr+batch)
    real X[GP_MAX_OBSERVATIONS][2];          // Observed points (normalized to [0,1])
    real y[GP_MAX_OBSERVATIONS];             // Observed scores
    real length_scale;                       // RBF kernel length scale
    real noise_variance;                     // Observation noise variance
    real signal_variance;                    // Signal variance (amplitude)
} GaussianProcess;

/**
 * Bayesian optimization options.
 */
typedef struct {
    int n_initial;                           // Initial random samples (default: 10)
    int n_iterations;                        // BO iterations after initial (default: 20)
    int n_candidates;                        // Candidates to evaluate for EI (default: 100)
    real exploration_weight;                 // Exploration vs exploitation (default: 0.01)
} BayesianOptions;

/**
 * Initialize Gaussian Process state.
 *
 * @param gp GP state to initialize
 * @param n_dims Number of dimensions
 */
void gp_init(GaussianProcess *gp, int n_dims);

/**
 * Add an observation to the GP.
 *
 * @param gp GP state
 * @param x Normalized point (array of n_dims values in [0,1])
 * @param y Observed score
 * @return ERR_OK on success, error code otherwise
 */
int gp_add_observation(GaussianProcess *gp, const real *x, real y);

/**
 * Predict mean and variance at a point using the GP.
 *
 * @param gp GP state
 * @param x Normalized point to predict
 * @param mean Output: predicted mean
 * @param variance Output: predicted variance
 */
void gp_predict(const GaussianProcess *gp, const real *x, real *mean, real *variance);

/**
 * Compute Expected Improvement at a point.
 *
 * @param mean Predicted mean at point
 * @param variance Predicted variance at point
 * @param best_y Best observed score so far
 * @param xi Exploration parameter (typically 0.01)
 * @return Expected improvement value
 */
real gp_expected_improvement(real mean, real variance, real best_y, real xi);

/**
 * Initialize Bayesian optimization options with defaults.
 *
 * @param opts Options to initialize
 */
void bayesian_options_init(BayesianOptions *opts);

/**
 * Perform Bayesian optimization over hyperparameter space.
 * Optimizes learning rate and batch size using Gaussian Process surrogate.
 *
 * @param space Hyperparameter search space
 * @param input_size Number of input features
 * @param output_size Number of output classes/values
 * @param output_activation Activation for output layer
 * @param loss_type Loss function to use
 * @param split Training/validation data split
 * @param tune_options Hypertuning options (score function, callbacks)
 * @param bayes_options Bayesian optimization options
 * @param results Array to store results (caller allocates)
 * @param max_results Maximum results to store
 * @param best_result Output: best result found
 * @return Number of trials completed, or negative error code
 */
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
    HypertuneResult *best_result
);

// ============================================================================
// TPE (Tree-structured Parzen Estimator) OPTIMIZATION
// ============================================================================

#define TPE_MAX_OBSERVATIONS 200    // Maximum observations for TPE

/**
 * TPE optimization options.
 */
typedef struct {
    int n_startup;           // Random trials before TPE kicks in (default: 10)
    real gamma;              // Quantile for good/bad split (default: 0.25 = top 25%)
    int n_candidates;        // Samples from l(x) per iteration (default: 24)
    int n_iterations;        // TPE iterations after startup (default: 40)
    real bandwidth_factor;   // KDE bandwidth multiplier (default: 1.0)
} TPEOptions;

/**
 * Initialize TPE options with sensible defaults.
 *
 * @param opts Options to initialize
 */
void tpe_options_init(TPEOptions *opts);

/**
 * Perform TPE-based hyperparameter search.
 * Better than GP-BO for mixed categorical + continuous parameters.
 * Optimizes learning rate, batch size, optimizer, hidden layer count, and activations.
 *
 * @param space Hyperparameter search space
 * @param input_size Number of input features
 * @param output_size Number of output classes/values
 * @param output_activation Activation for output layer
 * @param loss_type Loss function to use
 * @param split Training/validation data split
 * @param tune_options Hypertuning options (score function, callbacks)
 * @param tpe_options TPE optimization options
 * @param results Array to store results (caller allocates)
 * @param max_results Maximum results to store
 * @param best_result Output: best result found
 * @return Number of trials completed, or negative error code
 */
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
    HypertuneResult *best_result
);

#ifdef __cplusplus
}
#endif

#endif // __ANN_HYPERTUNE_H
