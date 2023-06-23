# libann

[![CMake](https://github.com/mseminatore/ann/actions/workflows/cmake.yml/badge.svg)](https://github.com/mseminatore/ann/actions/workflows/cmake.yml)

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
`ann.c`. Integrating the code into another application requires adding only these
two files to the project, or linking to the `libann` library.

> The tensor library is not meant to be a comprehensive tensor library. It provides 
> only the minimal set of functions needed to support the current inference runtime.

# Tensor library

The `tensor` module provides the fundamental mathematical operations over 
vectors and matrices required by neural networks.

> The module does not currently support 3D, or rank 3, tensors. Support for 
> rank 3 tensors may be added in the future.

Key functions include both scalar and vectorized versions controlled by the
`USE_BLAS` compiler define. The default build uses the non-optimized scalar
versions. To use the vector optimized versions build the code using `-DUSE_BLAS=1` or edit **ann_config.h** and uncomment the
`USE_BLAS` define.

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

# ANN training and inference library

The `ann` module provides functions for training and testing a neural network
using several types of gradient descent and backpropagation methods. The 
`tensor` module provides the underlying math operations required.

The module supports both **Mean-Squared Error** and **Categorical Cross Entropy** for 
loss functions. This option is set when a network is created via `ann_make_network()`.

For training, the module provides **Stochastic Gradient Descent**, and 
**Momentum** optimizers. Support for **RMSProp**, **AdaGrad** and **Adam** is
in progress. This option is also set when a network is created via 
`ann_make_network()`.

The layer activation types currently supported are **None**, **Sigmoid** and **Softmax**.
Support for **RELU** is currently in progress.

For performance, mini-batch support is provided. Batch size can be configured, along
with other hyper-parameters, either directly via the network object or through 
various set_xx functions.

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

# Accelerating training with BLAS libraries

The `tensor` functions used for training and inference can be accelerated
using [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) libraries, often providing significant training speed increases. 
Many BLAS libraries use multi-threading and SIMD instructions with cache
aware partitioning algorithms to accelerate the various vector and matrix
operations used with ML training and inference.

> Note: For all but the largest networks, inference using a pre-trained model is very fast 
> even without the use of BLAS libraries. The BLAS libraries are most helpful for processing
> the huge amounts of data involved in training a network.

The code is regularly tested against [OpenBLAS](https://openblas.net) on
multiple platforms. Though not yet tested, the 
[Intel MKL library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) should also work with appropriate build setup.

The `USE_BLAS` define controls whether the provided scalar tensor code
path is used or the BLAS code path.

# Known Issues

There are a few portions of the library that are currently under construction.
These are:

- RELU, leaky RELU, Tanh and softsign activation not yet functional
- Adagrad optimizer not yet vectorized
- RMSProp optimizer not yet vectorized
- Adam optimizer not yet vectorized

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
    PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);

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
> link to download the files is provided below.

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
