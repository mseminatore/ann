# libann

[![CMake](https://github.com/mseminatore/ann/actions/workflows/cmake.yml/badge.svg)](https://github.com/mseminatore/ann/actions/workflows/cmake.yml)
![GitHub License](https://img.shields.io/github/license/mseminatore/ann)

`libann` is a library for Artificial Neural Networks (ANN).

The goal of this library project is to provide a compact, light-weight, 
portable library of primitives that can be used for training and evaluating
Neural Networks. The code is written in ANSI C for portability. It is compiled
and tested regularly for the following configurations:

* Windows (x86 and x64) using Visual Studio and clang
* Mac OSX (Intel and M1) using clang
* Ubuntu Linux using gcc

There are two main components to the library. The first is a lightweight Tensor
library, `tensor.c`. The second is a minimal training and inference runtime, 
`ann.c`. Integrating `libann` into another application requires adding only 
these two files to the project, or linking to the `libann` library.

> The tensor library is not meant to be a comprehensive tensor library. It 
> provides only the minimal set of functions needed to support the current 
> training and inference runtime.

# Tensor library

The `tensor` module provides the fundamental mathematical operations over 
vectors and matrices required by neural networks.

> The module does not currently support 3D, or rank 3, tensors. Support for 
> rank 3 tensors may be added in the future.

Key functions include both scalar and vectorized versions controlled by the
`USE_BLAS` compiler define. The default build uses the non-optimized scalar
versions. To use the vector optimized versions build the code using 
`-DUSE_BLAS=1` or edit **ann_config.h** and uncomment the `USE_BLAS` define.

## Functions

The following tensor library functions are provided.

Function | Description
---------|------------
tensor_create | create a new empty tensor
tensor_create_from_array | create a tensor from an array of data
tensor_set_from_array | initialize a tensor from an array of data
tensor_free | free tensor memory
tensor_ones | create a new tensor of 1's
tensor_zeros | create a new tensor of 0's
tensor_create_random_uniform | create a tensor with random values
tensor_onehot | initialize a tensor as a onehot vector
tensor_copy | copy a tensor
tensor_add_scalar | add a scalar value to the tensor
tensor_add | add two tensors
tensor_sub | subtract two tensors
tensor_mul_scalar | multiply a tensor by a scalar
tensor_mul | multiply two tensors
tensor_div | divide one tensor by another
tensor_matvec | multiply a vector by a matrix
tensor_square | square a tensor
tensor_exp | form exponential of each tensor element 
tensor_max | return the max value in a tensor
tensor_sum | compute the horizontal sum of a tensor
tensor_axpy | multiply tensor by scalar and add to a tensor
tensor_axpby | multiply tensors by scalar values and then add them
tensor_outer | form the outer product of two tensors
tensor_get_element | get an elmeent from a tensor
tensor_set_element | set an element of a tensor
tensor_save_to_file | save a tensor to a file
tensor_slice_rows | slice out rows to a new tensor
tensor_slice_cols | slide out cols to a new tensor
tensor_fill | fill a tensor with a value
tensor_random_uniform | fill a tensor with random values
tensor_print | print out a tensor
tensor_argmax | find index of maximum value in each column
tensor_gemm | general matrix multiplication
tensor_heaviside | Heaviside step function (ReLU derivative)

# ANN training and inference library

The `ann` module provides functions for training and testing a neural network
using several types of gradient descent and backpropagation methods. The 
`tensor` module provides the underlying math operations required.

The module supports both **Mean-Squared Error** and 
**Categorical Cross Entropy** for loss functions. This option is set when a 
network is created via `ann_make_network()`.

For training, the module provides **Stochastic Gradient Descent**,
**Momentum**, **AdaGrad**, **RMSProp**, and **Adam** optimizers. This option 
is set when a network is created via `ann_make_network()`. **Adam is 
recommended as the default** for most use cases due to its adaptive per-parameter 
learning rates and stable convergence.

The layer activation types currently supported are **None**, **Sigmoid**, 
**Softmax**, **ReLU**, **LeakyReLU**, **Tanh**, and **Softsign**.

For performance, mini-batch support is provided. Batch size can be configured, 
along with other hyper-parameters, either directly via the network object or 
through various set_xx functions.

## Functions

The following training and inference runtime functions are provided.

Function | Description
-------- | -----------
ann_make_network | create a new neural network
ann_add_layer | add a layer to the network
ann_free_network | free a network
ann_load_csv | load data from a csv file to a tensor
ann_load_network | load a previously saved network (text)
ann_save_network | save a trained network (text)
ann_load_network_binary | load a previously saved network (binary)
ann_save_network_binary | save a trained network (binary)
ann_train_network | train a network
ann_predict | predict an output using a previously trained network
ann_set_convergence | set the convergence threshold (optional)
ann_evaluate_accuracy | evaluate accuracy of trained network using test data
ann_set_learning_rate | override the default learning rate
ann_set_loss_function | set the loss function
ann_set_batch_size | set the mini-batch size
ann_set_epoch_limit | set the maximum number of epochs
ann_set_lr_scheduler | set learning rate scheduler callback
ann_set_gradient_clip | set gradient clipping threshold
ann_get_layer_count | get the number of layers in the network
ann_get_layer_nodes | get the number of nodes in a layer
ann_get_layer_activation | get the activation type of a layer
ann_export_onnx | export trained network to ONNX JSON format
ann_class_prediction | determine predicted class from output activations
ann_print_props | print network properties and configuration
print_outputs | print output layer activations (debug)
ann_strerror | convert error code to human-readable message
ann_set_error_log_callback | install error logging callback
ann_get_error_log_callback | get current error callback
ann_clear_error_log_callback | disable error logging callback

## Learning Rate Schedulers

Built-in schedulers adjust the learning rate during training:

| Scheduler | Function | Description |
|-----------|----------|-------------|
| Step decay | `lr_scheduler_step` | Multiply LR by gamma every N epochs |
| Exponential | `lr_scheduler_exponential` | Multiply LR by gamma each epoch |
| Cosine | `lr_scheduler_cosine` | Smooth decay from base LR to min LR |

```c
// Step decay: halve LR every 10 epochs
LRStepParams step_params = { .step_size = 10, .gamma = 0.5f };
ann_set_lr_scheduler(net, lr_scheduler_step, &step_params);

// Exponential decay: 5% reduction per epoch
LRExponentialParams exp_params = { .gamma = 0.95f };
ann_set_lr_scheduler(net, lr_scheduler_exponential, &exp_params);

// Cosine annealing: decay to 0.0001 over 100 epochs
LRCosineParams cos_params = { .T_max = 100, .min_lr = 0.0001f };
ann_set_lr_scheduler(net, lr_scheduler_cosine, &cos_params);

// Custom scheduler
real my_scheduler(unsigned epoch, real base_lr, void *data) {
    return base_lr / (1.0f + 0.01f * epoch);  // 1/t decay
}
ann_set_lr_scheduler(net, my_scheduler, NULL);
```

# Hyperparameter Tuning

The `ann_hypertune` module provides automated hyperparameter search to find 
optimal network configurations. It supports both **grid search** (exhaustive) 
and **random search** (sampling-based) strategies.

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

The following hyperparameters can be tuned:

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

Function | Description
-------- | -----------
hypertune_space_init | initialize search space with defaults
hypertune_options_init | initialize search options
hypertune_result_init | initialize a result structure
hypertune_split_data | split data into train/validation sets
hypertune_free_split | free split tensors
hypertune_grid_search | perform exhaustive grid search
hypertune_random_search | perform random search
hypertune_bayesian_search | perform Bayesian optimization search
hypertune_create_network | create network from result config
hypertune_count_grid_trials | calculate total grid combinations
hypertune_print_result | print a single result
hypertune_print_summary | print top N results
hypertune_score_accuracy | default scoring function (accuracy)
hypertune_generate_topology | generate layer sizes from pattern
hypertune_topology_name | get string name for topology pattern
gp_init | initialize Gaussian Process state
gp_add_observation | add observation to GP
gp_predict | predict mean and variance at a point
gp_expected_improvement | compute expected improvement
bayesian_options_init | initialize Bayesian optimization options

## Example Usage

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

## Random Search Example

For larger search spaces, random search is often more efficient:

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

# Accelerating training with BLAS libraries

The `tensor` functions used for training and inference can be accelerated
using [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) 
libraries, often providing significant training speed increases. Many BLAS 
libraries use multi-threading and SIMD instructions with cache-aware 
partitioning algorithms to accelerate the various vector and matrix operations 
used with ML training and inference.

> Note: For all but the largest networks, inference using a pre-trained model 
> is very fast, even without the use of BLAS libraries. The BLAS libraries are
> most helpful for processing the huge amounts of data involved in training 
> networks.

The code is regularly tested against [OpenBLAS](https://openblas.net) on
multiple platforms. Though not yet tested, the 
[Intel MKL library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) 
should also work with appropriate build setup.

The `USE_BLAS` define controls whether the provided scalar tensor code
path is used or the BLAS code path.

# Error Handling

All public API functions return error codes to indicate success or failure. The library provides a callback mechanism for custom error logging and monitoring.

## Error Codes

| Code | Constant | Description |
|------|----------|-------------|
| 0 | `ERR_OK` | Success, no error |
| 1 | `ERR_NULL_PTR` | NULL pointer passed to function |
| 2 | `ERR_ALLOC` | Memory allocation failed |
| 3 | `ERR_INVALID` | Invalid parameter or state |
| 4 | `ERR_IO` | File I/O error |
| 5 | `ERR_FAIL` | General failure |

Use `ann_strerror(code)` to convert an error code to a human-readable message.

## Error Callback

Install a custom callback to receive error notifications:

```c
// Define your error handler
void my_error_handler(int code, const char *msg, const char *func) {
    fprintf(stderr, "[%s] Error %d: %s\n", func, code, msg);
    // Optionally log to monitoring system, trigger alerts, etc.
}

// Install the callback
ann_set_error_log_callback(my_error_handler);

// Use the library - errors will invoke your callback
PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
ann_add_layer(net, 0, LAYER_INPUT, ACTIVATION_NULL);  // Error: 0 nodes

// Get current callback (for chaining)
ErrorLogCallback prev = ann_get_error_log_callback();

// Disable callback
ann_clear_error_log_callback();
```

The callback signature is:
```c
typedef void (*ErrorLogCallback)(int error_code, const char *error_message, const char *function_name);
```

# Terminal Colors

Training output uses ANSI color codes for improved readability:
- **Headers** - Cyan/bold (Training ANN)
- **Labels** - Dim (Network shape:, Optimizer:, etc.)
- **Network shape** - Cyan
- **Optimizer name** - Magenta
- **Loss function name** - Yellow
- **Batch size** - Blue
- **Progress bar** - Green
- **Loss value** - Yellow
- **Learning rate** - Blue  
- **Epoch counter** - White/bold
- **Convergence message** - Green/bold
- **Error messages** - Red/bold (to stderr)

Colors are enabled by default on modern terminals (Windows 10+, Linux, macOS).

To disable colors, set the `ANN_NO_COLOR` environment variable:

```bash
# Linux/macOS
export ANN_NO_COLOR=1

# Windows PowerShell
$env:ANN_NO_COLOR = "1"

# Windows Command Prompt
set ANN_NO_COLOR=1
```

# Known Issues

No known issues at this time.

# Example usage

Basic usage of the library for training and prediction involves just a few function
calls.

```c
#include <stdio.h>
#include <stdlib.h>
#include "ann.h"

//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
    real *data;
    int rows, stride;

    // load the data
    if (argc > 1)
    	ann_load_csv(argv[1], CSV_NO_HEADER, &data, &rows, &stride);
    else
    	ann_load_csv("and.csv", CSV_NO_HEADER, &data, &rows, &stride);

    // create a new empty network
    PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_MSE);
    ann_set_learning_rate(pnet, 0.001);  // Adam works best with lower learning rates

    // setup training tensors
    PTensor x_train = tensor_create_from_array(rows, stride, data);
    PTensor y_train = tensor_slice_cols(x_train, 2);

    // define our network structure
    ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL);
    ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID);

    // train the network
    ann_train_network(pnet, x_train, y_train, x_train->rows);
	
    // make a prediction using the trained network
    real outputs[1];
    ann_predict(pnet, &data[0], outputs);

    print_outputs(pnet);

    // free resources
    ann_free_network(pnet);
    free(data);

    return 0;
}
```

# Building the code

You can build the code either from the provided `Makefile` or using `CMake`.

If you plan to use the vectorized version, you must first download and install,
or build, the `OpenBLAS` library. Ensure that **ann_config.h** defines `USE_BLAS`.

> The default build files assume that the OpenBLAS is installed at /opt. If
> another location is used, update `Makefile` or `CMakeLists.txt` as needed.

To build using `Make`:

```
% git clone https://github.com/mseminatore/ann
% cd ann
% make
```

To build using `CMake`:

```
% git clone https://github.com/mseminatore/ann
% cd ann
% mkdir build
% cd build
% cmake ..
% cmake --build . --config Release
```

# Examples

There are a few example projects included to help you familiarize yourself
with the library and its usage. These are:

* **logic** - a simple linear regression model for AND, OR, NOR logic
* **digit5x7** - multi-class image classification model for learning 5x7 character digits
* **save_test** - demonstrates loading and testing a pre-trained text model file
* **save_test_binary** - demonstrates loading and testing a pre-trained binary model file
* **mnist** - model for the MNIST datasets (digit or fashion)

> Note that **logic** is not able to learn XOR. That is because XOR is not 
> a linearly separable function and therefore it cannot be learned using 
> a linear regression model. For XOR, a network with a hidden layer is required.

The **logic** sample is a simple linear regression model with no hidden layer.
It optionally takes the name of dataset on the command line. 
Data files are provided for `AND`, `OR`, and `NAND`. For example:

```
% ./logic or.csv
```

The **save_test** sample reads in a previously trained network for the MNIST 
fashion dataset. It will load and create the network from the file 
`mnist-fashion.nna`. It then expects to find the test file 
`mnist-fashion_test.csv` in the current directory.

```
% ./save_test
Loading network /dev/ann/mnist-fashion.nna...done.
Loading /dev/ann/fashion-mnist_test.csv...done.

Test accuracy: 86.98%
```

The **digit5x7** sample is a multi-layer NN for image classification that 
learns a 1-bit color representation of 5x7 font digits. It has 35 input nodes,
one per pixel, 48 hidden layer nodes and 10 output nodes. Once trained, noise
is added to the training data and used as test data.

The **mnist** sample is a multi-layer NN for image classification. It trains on
all 60,000 images in the MNIST training database. Each image is 28 x 28 pixels 
in 8-bit grayscale. Once trained it tests against all 10,000 images in the MNIST
testing dataset.

> Note that due to large file size, the MNIST training data is not provided. A
> link to download the CSV files is provided below.

The output of the **mnist** sample looks like the following:

```
OpenBLAS 0.3.23.dev DYNAMIC_ARCH NO_AFFINITY Sandybridge MAX_THREADS=64
      CPU uArch: Sandybridge
  Cores/Threads: 8/8

Training ANN
------------
  Network shape: 784-128-10
      Optimizer: Adaptive Stochastic Gradient Descent
  Loss function: Categorical Cross-Entropy
Mini-batch size: 8
  Training size: 60000 rows

Epoch 1/5
[====================] - loss: 0.4 - LR: 0.05
Epoch 2/5
[====================] - loss: 0.63 - LR: 0.047
Epoch 3/5
[====================] - loss: 0.15 - LR: 0.011
Epoch 4/5
[====================] - loss: 0.24 - LR: 0.016
Epoch 5/5
[====================] - loss: 0.44 - LR: 0.02

Training time: 20.000000 seconds, 0.066667 ms/step

Test accuracy: 86.98%
```

# Machine learning Datasets

If you are not familiar with NN libraries and usage you may find it helpful
to work with existing datasets which have well explored solution spaces.

To help with this, there are a number of freely available datasets for 
learning and practicing machine learning.

* [10 Standard Datasets for Practicing Applied Machine Learning](https://machinelearningmastery.com/standard-machine-learning-datasets/)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview)
* [Kaggle Datasets](https://www.kaggle.com/datasets?fileType=csv)
* [MNIST Digits](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
* [MNIST Fashion](https://www.kaggle.com/datasets/tk230147/fashion-mnist)
