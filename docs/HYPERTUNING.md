# Hyperparameter Tuning

The `ann_hypertune` module provides automated hyperparameter search to find
optimal network configurations. It supports **grid search** (exhaustive), 
**random search** (sampling-based), and **Bayesian optimization** (intelligent) strategies.

## Features

- **Grid Search** - exhaustively tries all combinations of hyperparameters
- **Random Search** - randomly samples from the hyperparameter space
- **Bayesian Optimization** - intelligent search using Gaussian Process surrogate
- **Topology Patterns** - automatic layer size generation (pyramid, funnel, etc.)
- **Per-Layer Activations** - different activation function for each layer
- **Data Splitting** - automatic train/validation holdout with optional shuffling
- **Custom Scoring** - user-defined callback for optimization metric
- **Progress Reporting** - callback for monitoring search progress
- **Reproducibility** - seed support for reproducible random searches

## Tunable Hyperparameters

| Parameter | Description |
|-----------|-------------|
| Learning rate | Continuous range with linear or log-scale spacing |
| Batch size | Discrete set of values to try |
| Optimizer | SGD, Momentum, Adam, RMSProp, AdaGrad |
| Hidden layers | Number of hidden layers (1-5) |
| Layer size | Base size for topology generation |
| Topology pattern | CONSTANT, PYRAMID, FUNNEL, INVERSE |
| Activation | Sigmoid, ReLU, LeakyReLU, Tanh (per layer optional) |

## Functions

| Function | Description |
|----------|-------------|
| hypertune_space_init | initialize search space with defaults |
| hypertune_options_init | initialize search options |
| hypertune_result_init | initialize a result structure |
| hypertune_split_data | split data into train/validation sets |
| hypertune_free_split | free split tensors |
| hypertune_grid_search | perform exhaustive grid search |
| hypertune_random_search | perform random search |
| hypertune_bayesian_search | perform Bayesian optimization search |
| hypertune_create_network | create network from result config |
| hypertune_count_grid_trials | calculate total grid combinations |
| hypertune_print_result | print a single result |
| hypertune_print_summary | print top N results |
| hypertune_score_accuracy | default scoring function (accuracy) |
| hypertune_generate_topology | generate layer sizes from pattern |
| hypertune_topology_name | get string name for topology pattern |
| gp_init | initialize Gaussian Process state |
| gp_add_observation | add observation to GP |
| gp_predict | predict mean and variance at a point |
| gp_expected_improvement | compute expected improvement |
| bayesian_options_init | initialize Bayesian optimization options |

## Basic Example

```c
#include "ann_hypertune.h"

// Load your data
PTensor inputs = /* your input data */;
PTensor outputs = /* your output data */;

// Split into train/validation (80/20)
DataSplit split;
hypertune_split_data(inputs, outputs, 0.8f, 1, 0, &split);

// Configure search space
HyperparamSpace space;
hypertune_space_init(&space);

// Customize the search space
space.learning_rate_min = 0.001f;
space.learning_rate_max = 0.1f;
space.learning_rate_steps = 3;
space.learning_rate_log_scale = 1;  // log-uniform sampling

space.batch_sizes[0] = 32;
space.batch_sizes[1] = 64;
space.batch_size_count = 2;

space.optimizers[0] = OPT_ADAM;
space.optimizers[1] = OPT_SGD;
space.optimizer_count = 2;

space.hidden_layer_counts[0] = 1;
space.hidden_layer_counts[1] = 2;
space.hidden_layer_count_options = 2;

space.hidden_layer_sizes[0] = 64;
space.hidden_layer_sizes[1] = 128;
space.hidden_layer_size_count = 2;

space.hidden_activations[0] = ACTIVATION_RELU;
space.hidden_activation_count = 1;

space.epoch_limit = 500;

// Configure options
HypertuneOptions options;
hypertune_options_init(&options);
options.verbosity = 1;  // show progress

// Run grid search
HypertuneResult results[100];
HypertuneResult best;
int trials = hypertune_grid_search(
    &space,
    input_size,           // number of input features
    output_size,          // number of output classes
    ACTIVATION_SOFTMAX,   // output activation
    LOSS_CATEGORICAL_CROSS_ENTROPY,
    &split,
    &options,
    results, 100,
    &best
);

printf("Completed %d trials\n", trials);
hypertune_print_result(&best);

// Create final network with best configuration
PNetwork net = hypertune_create_network(
    &best,
    input_size,
    output_size,
    ACTIVATION_SOFTMAX,
    LOSS_CATEGORICAL_CROSS_ENTROPY
);

// Train on full dataset, evaluate, etc.

// Cleanup
hypertune_free_split(&split);
ann_free_network(net);
```

## Random Search

For larger search spaces, random search is often more efficient than grid search:

```c
// Run random search with 50 trials
int trials = hypertune_random_search(
    &space,
    50,                   // number of random trials
    input_size,
    output_size,
    ACTIVATION_SOFTMAX,
    LOSS_CATEGORICAL_CROSS_ENTROPY,
    &split,
    &options,
    results, 100,
    &best
);

// Print top 5 results
hypertune_print_summary(results, trials, 5);
```

## Bayesian Optimization

For more efficient hyperparameter search, Bayesian optimization uses a Gaussian 
Process surrogate model to intelligently explore the search space:

```c
#include "ann_hypertune.h"

// Configure search space
HyperparamSpace space;
hypertune_space_init(&space);
space.learning_rate_min = 0.001f;
space.learning_rate_max = 0.1f;
space.batch_sizes[0] = 16;
space.batch_sizes[1] = 32;
space.batch_sizes[2] = 64;
space.batch_size_count = 3;
// ... other fixed hyperparameters

// Configure Bayesian optimization
BayesianOptions bo_opts;
bayesian_options_init(&bo_opts);
bo_opts.n_initial = 10;      // Random samples to initialize GP
bo_opts.n_iterations = 20;   // BO iterations after initialization
bo_opts.n_candidates = 100;  // Candidates to evaluate per iteration

// Run Bayesian optimization
HypertuneResult results[50], best;
int trials = hypertune_bayesian_search(
    &space, input_size, output_size,
    ACTIVATION_SOFTMAX, LOSS_CROSS_ENTROPY,
    &split, &tune_opts, &bo_opts,
    results, 50, &best
);
```

**How it works:**
1. **Initial phase**: Randomly samples `n_initial` configurations
2. **BO phase**: Uses Gaussian Process to predict performance, selects points 
   with highest Expected Improvement (EI)
3. **Optimizes**: Learning rate (log-scale) and batch size

**When to use Bayesian optimization:**
- Expensive evaluations (long training times)
- Smooth objective function
- 2-5 hyperparameters to tune

## Custom Scoring Function

By default, hypertuning optimizes for accuracy. You can provide a custom 
scoring function:

```c
// Custom scorer: optimize for F1 score, or minimize loss, etc.
real my_custom_scorer(PNetwork net, PTensor val_in, PTensor val_out, void *data) {
    // Your scoring logic here
    // Return higher values for better configurations
    real accuracy = ann_evaluate_accuracy(net, val_in, val_out);
    return accuracy;  // or any custom metric
}

// Use custom scorer
options.score_func = my_custom_scorer;
options.user_data = NULL;  // optional context data
```

## Topology Patterns

The hypertuning module supports automatic generation of layer sizes based on 
topology patterns. This helps explore different network architectures:

| Pattern | Description | Example (3 layers, base=64) |
|---------|-------------|----------------------------|
| CONSTANT | All layers same size | 64 → 64 → 64 |
| PYRAMID | Decreasing sizes toward output | 64 → 32 → 16 |
| INVERSE | Increasing sizes toward output | 16 → 32 → 64 |
| FUNNEL | Expand then contract | 32 → 64 → 32 |
| CUSTOM | Use explicit sizes | user-defined |

```c
// Configure multiple topology patterns
space.topology_patterns[0] = TOPOLOGY_CONSTANT;
space.topology_patterns[1] = TOPOLOGY_PYRAMID;
space.topology_patterns[2] = TOPOLOGY_INVERSE;
space.topology_pattern_count = 3;

// Generate sizes programmatically
int sizes[3];
hypertune_generate_topology(TOPOLOGY_PYRAMID, 64, 3, sizes);
// sizes = {64, 32, 16}
```

## Per-Layer Activations

Enable searching different activations for each hidden layer:

```c
space.hidden_activations[0] = ACTIVATION_RELU;
space.hidden_activations[1] = ACTIVATION_SIGMOID;
space.hidden_activations[2] = ACTIVATION_TANH;
space.hidden_activation_count = 3;
space.search_per_layer_activation = 1;  // enable per-layer search
```

When `search_per_layer_activation` is enabled, random search will assign 
different activations to each layer independently.

## Search Strategy Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Grid** | Small search spaces | Exhaustive, reproducible | Exponential cost |
| **Random** | Large spaces, many params | Efficient, parallelizable | May miss optimum |
| **Bayesian** | Expensive evaluations | Sample-efficient | Overhead, sequential |

**Rules of thumb:**
- Grid search: ≤100 total combinations
- Random search: 10-100 trials typically sufficient
- Bayesian: When each trial takes minutes/hours
