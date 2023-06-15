# libann

The goal of this library project is to provide a compact, light-weight, 
portable library of primitives that can be used for training and evaluating
Neural Networks. The code is written in ANSI C for portability. It is compiled
and tested regularly on Windows (x86 and x64) using Visual Studio and clang, 
and Mac OSX (Intel and Mx) using clang. It should also build fine on Linux 
using gcc.

There are two main components to the library. The first is a lightweight Tensor
library, `tensor.c`. The second is a minimal training and inference runtime, 
`ann.c`. Integrating the code into another application requires adding these
two files to the project or linking to the libann library.

> The tensor library is not meant to be a comprehensive tensor library. For 
> example, rank 3 tensors are not supported. It provides only the minimal set of 
> functions needed to support the current inference runtime.

# Tensor library

The `tensor` module provides the fundamental mathematical operations over 
vectors and matrices required by neural networks. The module does not currently
support 3D, or rank 3, tensors. 

> Support for rank 3 tensors may be added in the future.

Key functions include both scalar and vectorized versions controlled by the
`USE_BLAS` compiler define.

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
loss functions. This is set when a network is created via `ann_make_network()`.

For training, the module supports **Stochastic Gradient Descent**, and 
**Momentum** optimizers. Support for **RMSProp**, **AdaGrad** and **Adam** is
in progress.

Layer activation types supported are **None**, **Sigmoid** and **Softmax**.
Support for **RELU** is in progress.

For performance, mini-batching is supported and can be configured in several 
ways.

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
//
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

# Building

You can build the code either from the provided `Makefile` or using `CMake`.

If you plan to use the vectorized version, you must first download and install
or build the OpenBLAS library. Ensure that `ann_config.h` defines `USE_BLAS`.

> The default build files assume that the OpenBLAS is installed at /opt. If
> another location is used, update `Makefile` or `CMakeLists.txt` as needed.

To build using `Make`:

```
% git clone https://github.com/mseminatore/ann
% make
```

To build using `CMake`:

```
% git clone https://github.com/mseminatore/ann
% md build
% cd build
% cmake ..
% cmake --build .
```

# Examples

There are a few example projects included to help you familiarize yourself
with the library and its usage. These are:

* logic - a simple linear regression model for AND, OR, NOR logic
* digit5x7 - multi-class image classification model for learning 5x7 character digits
* save_test - demonstrates loading and testing a pre-trained model file
* mnist - model for the MNIST datasets (digit or fashion)

> Note that digit5x7 is not able to learn XOR. That is because XOR is not 
> a linearly separable function and therefore it cannot be learned using 
> a linear regression model. For XOR, a hidden layer is required.

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
