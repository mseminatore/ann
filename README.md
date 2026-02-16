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
ann_train_begin | begin an online/incremental training session
ann_train_step | train one mini-batch step (online training)
ann_train_end | end an online/incremental training session
ann_predict | predict an output using a previously trained network
ann_set_convergence | set the convergence threshold (optional)
ann_evaluate_accuracy | evaluate accuracy of trained network using test data
ann_set_learning_rate | override the default learning rate
ann_set_loss_function | set the loss function
ann_set_batch_size | set the mini-batch size
ann_set_epoch_limit | set the maximum number of epochs
ann_set_lr_scheduler | set learning rate scheduler callback
ann_set_gradient_clip | set gradient clipping threshold
ann_set_weight_decay | set L2 regularization (weight decay) coefficient
ann_set_l1_regularization | set L1 regularization (LASSO) coefficient
ann_set_dropout | set default dropout rate for hidden layers
ann_set_layer_dropout | set dropout rate for a specific layer
ann_get_layer_count | get the number of layers in the network
ann_get_layer_nodes | get the number of nodes in a layer
ann_get_layer_activation | get the activation type of a layer
ann_export_onnx | export trained network to ONNX JSON format
ann_import_onnx | import network from ONNX JSON file
ann_export_pikchr | export network architecture as PIKCHR diagram
ann_export_learning_curve | export training history as CSV
ann_clear_history | clear training history
ann_confusion_matrix | compute binary confusion matrix and MCC
ann_print_confusion_matrix | print formatted confusion matrix
ann_class_prediction | determine predicted class from output activations
ann_print_props | print network properties and configuration
ann_print_outputs | print output layer activations (debug)
ann_strerror | convert error code to human-readable message
ann_set_error_log_callback | install error logging callback
ann_get_error_log_callback | get current error callback
ann_clear_error_log_callback | disable error logging callback

## Learning Rate Schedulers

Built-in schedulers adjust the learning rate during training:

| Scheduler | Function | Description |
|-----------|----------|-------------|
| Step decay | `ann_lr_scheduler_step` | Multiply LR by gamma every N epochs |
| Exponential | `ann_lr_scheduler_exponential` | Multiply LR by gamma each epoch |
| Cosine | `ann_lr_scheduler_cosine` | Smooth decay from base LR to min LR |

```c
// Step decay: halve LR every 10 epochs
LRStepParams step_params = { .step_size = 10, .gamma = 0.5f };
ann_set_lr_scheduler(net, ann_lr_scheduler_step, &step_params);

// Exponential decay: 5% reduction per epoch
LRExponentialParams exp_params = { .gamma = 0.95f };
ann_set_lr_scheduler(net, ann_lr_scheduler_exponential, &exp_params);

// Cosine annealing: decay to 0.0001 over 100 epochs
LRCosineParams cos_params = { .T_max = 100, .min_lr = 0.0001f };
ann_set_lr_scheduler(net, ann_lr_scheduler_cosine, &cos_params);

// Custom scheduler
real my_scheduler(unsigned epoch, real base_lr, void *data) {
    return base_lr / (1.0f + 0.01f * epoch);  // 1/t decay
}
ann_set_lr_scheduler(net, my_scheduler, NULL);
```

## Dropout Regularization

Dropout randomly zeros neurons during training to reduce overfitting. Uses 
**inverted dropout**: values are scaled by `1/(1-rate)` during training so no 
adjustment is needed at inference time.

```c
// Set default dropout rate for all hidden layers (0.5 = 50% dropout)
ann_set_dropout(net, 0.5f);

// Override for a specific layer (layer index, rate)
ann_set_layer_dropout(net, 2, 0.3f);  // Layer 2: 30% dropout

// Manually control training mode (automatic during ann_train_network)
ann_set_training_mode(net, 1);  // Enable dropout
ann_set_training_mode(net, 0);  // Disable dropout (inference)
```

**Notes:**
- Dropout is only applied to **hidden layers** (not input or output)
- Dropout is **automatically enabled** during `ann_train_network()` and disabled when training completes
- Recommended rates: 0.2-0.5 for hidden layers, lower for layers with fewer neurons
- Use dropout with larger networks to prevent overfitting

## Confusion Matrix & MCC

For binary classification problems, compute a confusion matrix and Matthews 
Correlation Coefficient (MCC):

```c
int tp, fp, tn, fn;
real mcc = ann_confusion_matrix(net, inputs, outputs, &tp, &fp, &tn, &fn);

// Or print formatted output
ann_print_confusion_matrix(net, inputs, outputs);
```

**Output:**
```
Confusion Matrix
                Predicted
              Pos     Neg
Actual Pos     42       3
       Neg      5      50

MCC: 0.8732
```

**Notes:**
- Binary classification only (2 output classes)
- Class 0 = negative, Class 1 = positive
- MCC range: -1 (worst) to +1 (perfect)
- MCC handles imbalanced datasets better than accuracy

## Network Visualization (PIKCHR)

Export network architecture as a PIKCHR diagram that renders to SVG:

```c
ann_export_pikchr(net, "network.pikchr");
```

Then render with: `pikchr network.pikchr > network.svg`

**Two modes:**
- **Simple mode** (networks with >10 nodes per layer): Box diagram with layer info
- **Detailed mode** (≤10 nodes per layer): Individual node circles with connections

## Learning Curve Export

Export training history as CSV for visualization:

```c
ann_train_network(net, inputs, outputs, rows);
ann_export_learning_curve(net, "learning_curve.csv");

// Clear history before retraining (optional)
ann_clear_history(net);
```

**CSV format:**
```
epoch,loss,learning_rate
1,0.2534,0.001
2,0.1823,0.001
...
```

Plot with gnuplot, Python matplotlib, or Excel to diagnose training issues.

## Online / Incremental Training

For scenarios where data arrives incrementally (streaming, fine-tuning a loaded model, or user feedback), use the step-based training API:

```c
PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
ann_add_layer(net, 784, LAYER_INPUT, ACTIVATION_NULL);
ann_add_layer(net, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);
ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

ann_train_begin(net);

// Feed mini-batches one at a time
for (int i = 0; i < num_batches; i++)
{
    real loss = ann_train_step(net, batch_inputs[i], batch_targets[i], batch_size);
    printf("Step %d loss: %f\n", i, loss);

    // Safe to predict mid-training (dropout is auto-disabled)
    ann_predict(net, test_input, prediction);
}

ann_train_end(net);
```

Key differences from `ann_train_network()`:
- **Does not reset optimizer state** — Adam momentum/variance are preserved across calls
- **Does not reinitialize weights** — safe for fine-tuning loaded/pre-trained models
- **Single sample training** — pass `batch_size=1` to train on individual examples
- **`ann_predict()` is safe mid-training** — dropout is automatically disabled during inference

# Hyperparameter Tuning

The `ann_hypertune` module provides automated hyperparameter search to find
optimal network configurations.

**Search strategies:**
- **Grid Search** - exhaustive, tries all combinations
- **Random Search** - samples from hyperparameter space
- **Bayesian Optimization** - intelligent search using Gaussian Process
- **TPE** - Tree-structured Parzen Estimator

**Tunable parameters:** learning rate, batch size, optimizer, hidden layer count, 
layer sizes, topology patterns, activations (per-layer optional).

```c
#include "ann_hypertune.h"

// Split data 80/20
DataSplit split;
hypertune_split_data(inputs, outputs, 0.8f, 1, 0, &split);

// Configure search space
HyperparamSpace space;
hypertune_space_init(&space);
space.learning_rate_min = 0.001f;
space.learning_rate_max = 0.1f;
space.batch_sizes[0] = 32;
space.batch_sizes[1] = 64;
space.batch_size_count = 2;

// Run random search
HypertuneOptions options;
hypertune_options_init(&options);

HypertuneResult results[100], best;
int trials = hypertune_random_search(&space, 50, input_size, output_size,
    ACTIVATION_SOFTMAX, LOSS_CATEGORICAL_CROSS_ENTROPY,
    &split, &options, results, 100, &best);

// Create network with best config
PNetwork net = hypertune_create_network(&best, input_size, output_size,
    ACTIVATION_SOFTMAX, LOSS_CATEGORICAL_CROSS_ENTROPY);
```

See [docs/HYPERTUNING.md](docs/HYPERTUNING.md) for full documentation including 
topology patterns, Bayesian optimization, custom scoring, and API reference.

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

## Supported BLAS Libraries

| Library | CMake Flag | Platforms | Notes |
|---------|------------|-----------|-------|
| [CBLAS](https://github.com/xianyi/CBLAS) | `-DUSE_CBLAS=1` | Windows, macOS, Linux (x64, ARM64) | Recommended, cross-platform |
| [OpenBLAS](https://openblas.net) | `-DUSE_BLAS=1` | macOS, Linux | Well-established, good performance |

The [Intel MKL library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) 
should also work with appropriate build setup.

## Building with BLAS Support

To enable BLAS acceleration, pass the appropriate CMake flag:

```bash
# CBLAS (recommended for cross-platform builds)
cmake -DUSE_CBLAS=1 ..
cmake --build . --config Release

# OpenBLAS (Linux/macOS)
cmake -DUSE_BLAS=1 ..
cmake --build . --config Release
```

### Library Installation Paths

The build expects libraries at these default locations:

| Platform | CBLAS | OpenBLAS |
|----------|-------|----------|
| Windows | `C:/opt/cblas` | N/A |
| macOS/Linux | `/opt/cblas` | `/opt/OpenBLAS` |

If your library is installed elsewhere, update the paths in `CMakeLists.txt`.

### Using BLAS in Your Code

When using CBLAS, call `cblas_init()` before any network operations:

```c
#ifdef USE_CBLAS
#include <cblas.h>
#endif

int main() {
#ifdef USE_CBLAS
    cblas_init(CBLAS_DEFAULT_THREADS);
#endif
    
    PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_MSE);
    // ... rest of your code
}
```

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

    ann_print_outputs(pnet);

    // free resources
    ann_free_network(pnet);
    free(data);

    return 0;
}
```

# Building the code

You can build the code either from the provided `Makefile` or using `CMake`.

To build using `Make`:

```bash
git clone https://github.com/mseminatore/ann
cd ann
make
```

To build using `CMake`:

```bash
git clone https://github.com/mseminatore/ann
cd ann
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

To build with BLAS acceleration (see [Accelerating training with BLAS libraries](#accelerating-training-with-blas-libraries)):

```bash
# CBLAS (Windows, macOS, Linux)
cmake -DUSE_CBLAS=1 ..
cmake --build . --config Release

# OpenBLAS (macOS, Linux)
cmake -DUSE_BLAS=1 ..
cmake --build . --config Release
```

To build a shared library (DLL/so/dylib) for use with Python or other language bindings:

```bash
# CMake shared library build
cmake -DBUILD_SHARED=1 ..
cmake --build . --config Release

# Make shared library build (Linux/macOS)
make shared
```

This produces `libann.dll` (Windows), `libann.so` (Linux), or `libann.dylib` (macOS).

When linking against the DLL from external code, define `ANN_USING_DLL` to enable proper import declarations:
```c
#define ANN_USING_DLL
#include "ann.h"
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
