# ann

The goal of this library is to provide a compact, light-weight, portable library
of primitives that can be used for training and evaluating Neural Networks.

There are two main components to the library. The first is a lightweight Tensor library, `tensor.c`. The second is a minimal training and inference runtime, 
`ann.c`.

>The tensor library is not meant to be complete, it provides only the 
>functions needed to support the inference runtime.

# Tensor library

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

# ANN library



