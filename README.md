# libann

[![CMake](https://github.com/mseminatore/ann/actions/workflows/cmake.yml/badge.svg)](https://github.com/mseminatore/ann/actions/workflows/cmake.yml)
![GitHub License](https://img.shields.io/github/license/mseminatore/ann)

`libann` is a library for Artificial Neural Networks (ANN).

> [!TIP]
> **Try it live:** the [libann Workbench](https://mseminatore.github.io/ann/) web demo runs in your browser, with nothing to install.

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
network is created via `ann_make_network()`. The output activation picks the
task:

Task | Loss | Output activation
---- | ---- | -----------------
Multi-class classification (one class per sample) | `LOSS_CATEGORICAL_CROSS_ENTROPY` | `ACTIVATION_SOFTMAX`
Binary or multi-label classification | `LOSS_BINARY_CROSS_ENTROPY` | `ACTIVATION_SIGMOID`
Regression (unbounded targets) | `LOSS_MSE` | `ACTIVATION_NULL` (linear)

`LOSS_BINARY_CROSS_ENTROPY` is an alias of `LOSS_CATEGORICAL_CROSS_ENTROPY`:
with a sigmoid output the library computes binary cross-entropy per output.

For training, the module provides **Stochastic Gradient Descent**,
**Momentum**, **AdaGrad**, **RMSProp**, and **Adam** optimizers. This option 
is set when a network is created via `ann_make_network()`. **Adam is 
recommended as the default** for most use cases due to its adaptive per-parameter 
learning rates and stable convergence.

The layer activation types currently supported are **None**, **Sigmoid**, 
**Softmax**, **ReLU**, **LeakyReLU**, **Tanh**, **Softsign**, and **GELU**.
GELU uses the tanh approximation on every path (CPU, Metal, CUDA) and exports
to ONNX as `Gelu` with `approximate="tanh"`, which requires opset 20; models
without GELU still export with opset 17.

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
ann_evaluate_loss | evaluate the network's loss on a dataset
ann_set_learning_rate | override the default learning rate
ann_set_loss_function | set the loss function
ann_set_batch_size | set the mini-batch size
ann_set_epoch_limit | set the maximum number of epochs
ann_set_validation_data | set a validation set evaluated after each epoch
ann_set_early_stopping | stop when the monitored loss stops improving
ann_get_best_epoch | get the best epoch of the last training run
ann_get_epochs_trained | get the number of epochs the last training run trained
ann_set_lr_scheduler | set learning rate scheduler callback
ann_set_gradient_clip | set gradient clipping threshold
ann_set_weight_decay | set L2 regularization (weight decay) coefficient
ann_set_l1_regularization | set L1 regularization (LASSO) coefficient
ann_set_dropout | set default dropout rate for hidden layers
ann_set_layer_dropout | set dropout rate for a specific layer
ann_set_input_noise | corrupt training inputs (Gaussian, masking or salt-and-pepper), e.g. for denoising autoencoders
ann_get_input_noise | get the input noise settings
ann_apply_input_noise | apply the input noise to your own data (e.g. noisy test inputs)
ann_get_layer_count | get the number of layers in the network
ann_get_layer_nodes | get the number of nodes in a layer
ann_get_layer_activation | get the activation type of a layer
ann_get_layer_kind | get what a layer computes (`LAYER_KIND_DENSE`, `_CONV2D`, `_MAXPOOL2D`, `_FLATTEN`, `_UPSAMPLE2D`, `_RESHAPE`)
ann_get_layer_shape | get a layer's output shape (channels, height, width)
ann_get_layer_settings | get a conv/pool layer's kernel or pool size, stride and padding
ann_get_layer_param_count | get how many parameter tensors a layer has (weights, bias, ...)
ann_get_layer_param_shape | get a layer parameter's shape as [rows x cols]
ann_get_layer_param | copy a layer parameter's values out
ann_set_layer_param | set a layer parameter's values (pre-trained weights, e.g. GloVe vectors)
ann_set_layer_trainable | freeze a layer (training leaves it alone) or unfreeze it
ann_get_layer_trainable | check whether a layer is frozen
ann_set_num_threads | set how many CPU threads libann uses (0 = default)
ann_get_num_threads | get the CPU thread count
ann_get_backend_info | report the BLAS library, GPU backend, thread count and number size of this build
ann_export_onnx | export trained network to ONNX JSON format
ann_quantize | quantize a trained network to int8 (calibrated on sample inputs)
ann_quantized_predict | run the int8 model (also `_predict_int8`, `_evaluate_accuracy`)
ann_export_c_header | export an int8 or float32 model as a standalone C header
ann_import_onnx | import network from ONNX JSON file
ann_export_pikchr | export network architecture as PIKCHR diagram
ann_export_learning_curve | export training history as CSV
ann_clear_history | clear training history
ann_get_history_count | number of epochs in the training history
ann_get_history | copy per-epoch loss, learning rate and validation loss
ann_confusion_matrix | compute binary confusion matrix and MCC
ann_print_confusion_matrix | print formatted confusion matrix
ann_class_prediction | determine predicted class from output activations
ann_print_props | print network properties and configuration
ann_print_outputs | print output layer activations (debug)
ann_set_print_function | send a network's progress output to a function, or silence it with NULL
ann_get_print_function | get a network's progress output function
ann_strerror | convert error code to human-readable message
ann_set_error_log_callback | install error logging callback
ann_get_error_log_callback | get current error callback
ann_clear_error_log_callback | remove the callback, restoring the default stderr output

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

## Validation and Early Stopping

Give `ann_train_network()` a validation set and it reports `val_loss` after
every epoch. Early stopping then ends training once the validation loss has
not improved for `patience` epochs, and can restore the best epoch's weights:

```c
ann_set_validation_data(net, val_inputs, val_outputs);  // borrowed, not copied
ann_set_early_stopping(net, 10, 1e-4f, 1);  // patience, min_delta, restore_best
ann_set_epoch_limit(net, 500);              // an upper bound, rarely reached

ann_train_network(net, inputs, outputs, rows);
printf("best epoch: %u\n", ann_get_best_epoch(net));
```

Without a validation set, early stopping monitors the training loss.
`ann_evaluate_loss()` computes the same loss on any dataset.

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

## Input Noise

`ann_set_input_noise` corrupts the inputs of every training batch while the
targets stay as given, which is how a **denoising autoencoder** learns to
remove noise: train it with the clean images as targets.

```c
// Gaussian noise with standard deviation 0.3, clamped to the pixel range [0, 1]
ann_set_input_noise(net, NOISE_GAUSSIAN, 0.3f, 0.0f, 1.0f);
ann_train_network(net, x_train, x_train, x_train->rows);

// noisy test inputs made the same way, to measure how well it denoises
ann_apply_input_noise(net, noisy_test, rows * cols);
```

Type | amount | Effect
---- | ------ | ------
`NOISE_GAUSSIAN` | standard deviation | adds normal noise
`NOISE_MASKING` | probability | sets inputs to 0
`NOISE_SALT_PEPPER` | probability | sets inputs to `low` or `high`

When `low < high`, noisy values are clamped to `[low, high]` (pass `0, 0` for
no clamp). Noise is drawn fresh for every batch, on the CPU with `rand()`, so
GPU training and any thread count give the same results; predictions and
evaluation use the inputs as given. The setting is not saved with the model.
The `fashion_dae` example trains a convolutional denoising autoencoder on
Fashion-MNIST: 5 epochs (about 80 seconds on a 4-core Intel Mac CPU) raise the
test images' PSNR from 12.8 dB with noise to 19.2 dB, and it writes the clean,
noisy and denoised images as `fashion_dae.pgm` and `fashion_dae.png`.

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

## Convolutional Networks

Image inputs, convolution, max pooling and flatten layers build small CNNs.
Each sample is stored as `C*H*W` values in channel, row, column order (as in
ONNX and PyTorch), so image data loaded with `ann_load_csv()` can be used
directly.

```c
PNetwork net = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
ann_add_input(net, 1, 28, 28);                      // channels, height, width
ann_add_conv2d(net, 8, 3, 1, 0, ACTIVATION_RELU);   // filters, kernel, stride, padding -> 8x26x26
ann_add_maxpool2d(net, 2, 2);                       // pool size, stride               -> 8x13x13
ann_add_flatten(net);                               // optional before a dense layer  -> 1352
ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
```

Function | Description
-------- | -----------
ann_add_input | input layer with an image shape (channels, height, width)
ann_add_conv2d | 2-D convolution with square kernels, stride and zero padding
ann_add_conv2d_grouped | the same, with the channels and filters split into groups
ann_add_depthwise_conv2d | depthwise convolution: each channel convolved with its own filters
ann_add_maxpool2d | 2-D max pooling per channel
ann_add_conv2d_ex, ann_add_maxpool2d_ex | the same with separate height and width settings (non-square windows)
ann_add_conv1d, ann_add_maxpool1d | 1-D convolution and pooling over sequences shaped (C, 1, L)
ann_add_flatten | reshape an image to a vector (optional: dense layers flatten their input)
ann_add_upsample2d | nearest-neighbour upsampling by an integer scale, per channel
ann_add_reshape | give the previous layer's values a (channels, height, width) shape
ann_add_global_avgpool2d | each channel's mean: (C, H, W) to (C, 1, 1)
ann_add_batchnorm | batch normalization per channel (per node after a dense layer), then an optional activation
ann_add_embedding | token ids to learned vectors: L ids become the sequence (dim, 1, L)
ann_add_embedding_masked | an embedding whose padding steps are left out of the layers above (see below)
ann_add_global_maxpool2d | each channel's largest value: (C, H, W) to (C, 1, 1)

Output sizes are `(H + 2*padding - kernel) / stride + 1` for convolutions,
`(H - pool) / stride + 1` for pooling and `H * scale` for upsampling, and
likewise for the width with its own settings in the `_ex` functions. Settings
that don't fit the input, a reshape to a different size, or an activation
other than none on pool/flatten/upsample/reshape layers, return `ERR_INVALID`.

Sequences such as time series are images one row high: L steps of C features
are the input shape `(C, 1, L)`, and 1-D convolutions and pooling slide along
the steps:

```c
// x: rows of 3 features x 128 steps, stored feature by feature
ann_add_input(net, 3, 1, 128);
ann_add_conv1d(net, 16, 5, 1, 2, ACTIVATION_NULL);   // 16x1x128: windows of 5 steps
ann_add_batchnorm(net, ACTIVATION_RELU);
ann_add_maxpool1d(net, 2, 2);                        // 16x1x64
ann_add_conv1d(net, 32, 3, 1, 1, ACTIVATION_RELU);   // 32x1x64
ann_add_global_avgpool2d(net);                       // 32
ann_add_layer(net, 4, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
```

Text works the same way after an embedding, which turns each sample's token
ids (integers stored as reals, e.g. padded to a fixed length with id 0) into
a sequence of learned vectors. Global max pooling then keeps each feature's
strongest response wherever it occurs in the text:

```c
ann_add_layer(net, 200, LAYER_INPUT, ACTIVATION_NULL);   // 200 token ids per sample
ann_add_embedding(net, 10000, 64, 0);                    // 64x1x200; id 0 pads (always zeros)
ann_add_conv1d(net, 128, 5, 1, 2, ACTIVATION_RELU);      // 128x1x200: 5-token phrases
ann_add_global_maxpool2d(net);                           // 128
ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
```

Ids outside the vocabulary also give zeros. Metal and Vulkan train and run
these networks when the embedding reads the inputs, as here (CUDA keeps them
on the CPU). They cannot be quantized or exported as C headers yet.

The padding vectors are zero, but the layers above still see the padding
steps: a convolution's bias makes them nonzero, and they count in pooling and
batch norm statistics, so a text's answer depends on how much padding it has.
`ann_add_embedding_masked(net, 10000, 64, 0)` leaves them out. The mask
follows the sequence through 1-D convolutions, max pooling and batch norm,
whose outputs at masked steps are zero (the zero padding a shorter text would
have); batch norm's statistics and global average and max pooling skip them,
and a text that is all padding pools to zeros. A text padded at the end then
gives what the same text unpadded would (padding at the start also does when
every stride is 1). `ann_get_layer_masked` tells which layers carry the mask.
Saves keep it, and ONNX exports it as the ops that compute it, which libann
imports back.

Global average pooling replaces flatten + a large dense layer at the end of a
CNN: each channel becomes its mean, so the classifier sees one value per
filter. This removes most of the parameters, which keeps quantized and
exported models small:

```c
ann_add_conv2d(net, 32, 3, 1, 1, ACTIVATION_RELU);  // 32x14x14
ann_add_global_avgpool2d(net);                      // 32 (no flatten needed)
ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
```

Batch normalization goes between a layer and its activation: the conv or
dense layer has no activation, and the batch norm layer applies it. It lets
deeper networks train faster and with higher learning rates:

```c
ann_add_conv2d(net, 32, 3, 1, 1, ACTIVATION_NULL);  // no activation here...
ann_add_batchnorm(net, ACTIVATION_RELU);            // ...it comes after normalizing
```

Training normalizes with each batch's mean and variance (so use batches of
more than one sample) and keeps running averages of them (momentum 0.1, as
PyTorch); `ann_predict` and evaluation use the running averages. They are
saved with the model. A layer's bias before a batch norm is redundant, since
the batch norm subtracts the mean. Quantization and C export fold each batch
norm into the conv or dense layer before it, which must have no activation.

Depthwise separable convolutions (as in MobileNet) split a convolution into
a depthwise one, which filters each channel on its own, and a 1x1 one, which
mixes the channels: several times fewer weights and multiply-adds for about
the same accuracy, which suits int8 models on small devices.
`ann_add_conv2d_grouped` is the general form (`groups` = C is depthwise):

```c
ann_add_conv2d(net, 32, 3, 1, 1, ACTIVATION_NULL);             // 32x28x28
ann_add_batchnorm(net, ACTIVATION_RELU);
ann_add_depthwise_conv2d(net, 1, 3, 2, 1, ACTIVATION_NULL);    // 32x14x14: 9 weights per channel
ann_add_batchnorm(net, ACTIVATION_RELU);
ann_add_conv2d(net, 64, 1, 1, 0, ACTIVATION_NULL);             // 64x14x14: mixes the channels
ann_add_batchnorm(net, ACTIVATION_RELU);
ann_add_global_avgpool2d(net);
ann_add_layer(net, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
```

Upsampling and reshape layers build decoders, e.g. a convolutional
autoencoder trained to reproduce (or denoise) its input:

```c
PNetwork net = ann_make_network(OPT_ADAM, LOSS_MSE);
ann_add_input(net, 1, 28, 28);
ann_add_conv2d(net, 16, 3, 1, 1, ACTIVATION_RELU);    // 16x28x28
ann_add_maxpool2d(net, 2, 2);                         // 16x14x14
ann_add_conv2d(net, 8, 3, 1, 1, ACTIVATION_RELU);     // 8x14x14
ann_add_maxpool2d(net, 2, 2);                         // 8x7x7
ann_add_upsample2d(net, 2);                           // 8x14x14
ann_add_conv2d(net, 16, 3, 1, 1, ACTIVATION_RELU);    // 16x14x14
ann_add_upsample2d(net, 2);                           // 16x28x28
ann_add_conv2d(net, 1, 3, 1, 1, ACTIVATION_SIGMOID);  // 1x28x28
ann_train_network(net, x_train, x_train, x_train->rows);  // targets are the inputs
```

A dense bottleneck turns back into an image with `ann_add_reshape`, e.g.
`... -> ann_add_layer(net, 392, ...) -> ann_add_reshape(net, 8, 7, 7) -> ann_add_upsample2d(net, 2) -> ...`.

Convolutional networks, including ones with upsample, reshape, global
average and max pooling and batch norm layers, train and run on the GPU with the Metal and Vulkan backends (the CUDA backend returns
`ERR_INVALID` from `ann_gpu_upload_network()` for them, so they stay on the
CPU there), save in model format v2, and export to and import
from ONNX as `Conv`, `MaxPool`, `GlobalAveragePool`, `GlobalMaxPool`, `BatchNormalization`, `Flatten`, `Resize` (nearest) and `Reshape`
(embeddings as `Cast`, `Gather`, `Transpose` and `Unsqueeze`)
with a `[batch, C, H, W]` input.
The importer also reads simple CNNs and MLPs exported by other tools, such as
PyTorch (see [ONNX Export & Import](docs/ONNX_EXPORT.md)). The `fashion_cnn`
example trains a two-convolution network on Fashion-MNIST to about 90% test
accuracy.

## int8 Quantization and C Export

`ann_quantize()` converts a trained network to int8 (per-channel weights,
calibrated activation ranges), and `ann_export_c_header()` writes it as a C
header that runs with only `ann_rt.h`: integer-only math, no `malloc`, builds
freestanding for microcontrollers. A float32 export is also available. On
Fashion-MNIST, int8 accuracy stays within 0.2 points of float32 with 4x less
parameter memory. See [int8 Quantization & C Export](docs/C_EXPORT.md) and
`examples/export_model.c`.

```c
PQuantizedNetwork q = ann_quantize(pnet, calibration_inputs);
printf("int8 accuracy %g\n", ann_quantized_evaluate_accuracy(q, x_test, y_test));
ann_export_c_header(NULL, q, "model.h", "model");    // then: model_predict(in, out, workspace)
ann_free_quantized(q);
```

## Saved Model Formats

`ann_save_network()` (text, `.nna`) and `ann_save_network_binary()` (`.nnb`)
write format version 1 for networks made only of dense layers, so files stay
readable by older libann releases. Networks with other layer kinds or an image-shaped
input are saved as version 2, which also records each layer's kind and shape. The loaders read
both versions and return `NULL` for unknown versions or truncated files.

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

When a validation set was used, a `val_loss` column is added.

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

## Layer Parameters and Frozen Layers

Each layer's parameters can be read and set as row-major matrices, e.g. to
start an embedding from pre-trained word vectors (GloVe, word2vec), or to copy
layers between networks. `ann_get_layer_param_count` gives how many a layer
has, in this order: dense and Conv2D the weights then the bias, batch norm
gamma, beta and the running mean and variance, an embedding its vectors.
`ann_get_layer_param_shape` gives each one's shape: dense weights
`[nodes x nodes below]`, Conv2D weights `[filters x channels/groups * kh * kw]`
(the ONNX layout `[F, C/G, kh, kw]` flattened), an embedding `[vocab x dim]`
(row `i` is id `i`'s vector), biases and batch norm's tensors `[1 x channels]`.

```c
// ids 0..vocab-1 from your tokenizer; vectors[i * 100 + k] is word i's GloVe vector
ann_add_layer(net, 200, LAYER_INPUT, ACTIVATION_NULL);
ann_add_embedding_masked(net, vocab, 100, 0);
ann_add_conv1d(net, 64, 5, 1, 2, ACTIVATION_RELU);
ann_add_global_maxpool2d(net);
ann_add_layer(net, 2, LAYER_OUTPUT, ACTIVATION_SOFTMAX);

ann_set_layer_param(net, 1, 0, vectors, vocab * 100);
ann_set_layer_trainable(net, 1, 0);         // keep the vectors fixed
ann_train_network(net, inputs, targets, rows);
```

A layer given values this way counts as initialized: training starts from them
instead of randomizing them, and still randomizes the other layers. Reading a
layer that nothing has initialized yet initializes it first, so it reports the
values training starts from. Values are
checked (the count must match, and NaN or infinite values are rejected), and a
network uploaded to a GPU gets them too. A frozen layer's parameters stay as
they are through training on the CPU or a GPU; gradients still pass through it
to the layers below, and a frozen batch norm still updates its running
statistics. Freezing is not saved with the network. To fine-tune the vectors
afterwards, unfreeze the layer (`ann_set_layer_trainable(net, 1, 1)`) and train
a few more epochs at a low learning rate (training again keeps the weights).

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
See also the `mnist_hypertune*` samples in [Examples](#examples).

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
| [CBLAS](https://github.com/mseminatore/cblas) | `-DUSE_CBLAS=1` | Windows, macOS, Linux (x64, ARM64) | Recommended, cross-platform |
| [OpenBLAS](https://openblas.net) | `-DUSE_BLAS=1` | macOS, Linux | Well-established, good performance |
| [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | `-DUSE_MKL=1` | macOS, Linux (x64) | Highest performance on Intel CPUs |
| [Apple Accelerate](https://developer.apple.com/documentation/accelerate) | `-DUSE_ACCELERATE=1` | macOS | Ships with macOS, nothing to install |

## Building with BLAS Support

To enable BLAS acceleration, pass the appropriate CMake flag:

```bash
# CBLAS (recommended for cross-platform builds)
cmake -DUSE_CBLAS=1 ..
cmake --build . --config Release

# OpenBLAS (Linux/macOS)
cmake -DUSE_BLAS=1 ..
cmake --build . --config Release

# Intel MKL (requires oneAPI toolkit; uses ONEAPI_ROOT env var)
cmake -DUSE_MKL=1 ..
cmake --build . --config Release

# Apple Accelerate (macOS; uses the SDK's copy, found with xcrun)
cmake -DUSE_ACCELERATE=1 ..
cmake --build . --config Release
```

### Library Installation Paths

The build expects libraries at these default locations:

| Platform | CBLAS | OpenBLAS | Intel MKL |
|----------|-------|----------|-----------|
| Windows | `C:/opt/cblas` | N/A | N/A |
| macOS/Linux | `/opt/cblas` | `/opt/OpenBLAS` | `$ONEAPI_ROOT/mkl/latest` |

For CBLAS and OpenBLAS, update paths in `CMakeLists.txt` if your install differs. MKL uses the `ONEAPI_ROOT` environment variable set by the Intel oneAPI installer.

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

# Multithreading

Training and batched inference use several CPU threads. libann splits its own
loops across them: convolution and pooling by sample and by filter, the
element-wise work (activations, their derivatives, optimizer updates) by
element, and, in builds without BLAS, its matrix multiply by rows.

```c
ann_set_num_threads(4);   // 0 = default, 1 = single-threaded
int n = ann_get_num_threads();
```

The default is the `ANN_NUM_THREADS` environment variable if set, otherwise
the number of processors.

Training prints what it runs on, e.g. `Backend: CPU, 8 threads, no BLAS`,
`Backend: CPU, 8 threads, Accelerate BLAS` or `Backend: Metal GPU`, and
`ann_get_backend_info()` reports the same for your own code. A library built
without optimization (e.g. a CMake `Debug` build, which a reused build
directory can keep from an earlier configure) adds `unoptimized build` to the
line, since it trains several times slower:

```c
Backend_info info;
ann_get_backend_info(&info);
printf("BLAS %s, GPU %s (%s), %d threads, %d-byte reals\n", info.blas, info.gpu,
       info.gpu_active ? "started" : "not started", info.threads, info.real_bytes);
```

- **BLAS libraries keep their own threads.** BLAS calls (OpenBLAS, Accelerate,
  MKL, CBLAS) always run on the calling thread and libann's workers never call
  BLAS, so the two never run at the same time or oversubscribe the CPU.
- **Results don't depend on the thread count.** Every value is computed by
  one thread in the same order as a serial run, so 1, 4 or 8 threads train
  bit-identical networks (with a non-BLAS build; threaded BLAS libraries can
  differ slightly from run to run by themselves).
- **Small work stays serial**, so tiny networks see no overhead.
- Build with `-DANN_NO_THREADS` to compile the thread pool out.

Measured on a 4-core Intel Mac (training time per sample or epoch):

Workload | 1 thread | 4 threads
-------- | -------- | ---------
CNN (2 conv layers), scalar build | 173 µs | 64 µs (2.7x)
CNN, Accelerate | 100 µs | 69 µs (1.45x)
Dense 784-128-10, batch 128, scalar build | 488 ms | 169 ms (2.9x)
Dense 784-128-10, batch 128, Accelerate | 52 ms | 43 ms (1.2x)

Dense networks gain little with a BLAS library, which already spreads their
matrix multiplies over the cores.

# GPU training and inference

libann can run training and inference on the GPU. Three backends are supported:

| Backend | CMake Flag | Platforms | Requirements |
|---------|------------|-----------|--------------|
| Apple Metal | `-DUSE_METAL=1` | macOS | Metal-capable GPU, Xcode Command Line Tools |
| NVIDIA CUDA | `-DUSE_CUDA=1` | Windows, Linux | NVIDIA GPU, CUDA Toolkit (`cudart`, `cublas`) |
| Vulkan | `-DUSE_VULKAN=1` | Windows, Linux | Any Vulkan 1.1 GPU (AMD, Intel, NVIDIA); Vulkan SDK or `libvulkan-dev` + `glslc` to build |

Metal and Vulkan run every layer kind; CUDA runs dense networks, and keeps CNNs
on the CPU. A build may include several backends: `ann_gpu_init()` starts the
first that works (Metal, then Vulkan, then CUDA), and the `ANN_GPU` environment
variable (`metal`, `vulkan`, `cuda` or `none`) picks one. See
[docs/GPU_VULKAN.md](docs/GPU_VULKAN.md) for the Vulkan backend.

When a GPU backend is active, `ann_train_network()` runs each mini-batch
entirely on the GPU — forward pass, backpropagation, and the optimizer update
(all five optimizers, with gradient clipping and L1/L2 regularization).
CPU and GPU training converge to the same loss within float precision.

## Building with GPU Support

```bash
# Apple Metal (macOS)
cmake -DUSE_METAL=1 ..
cmake --build . --config Release

# NVIDIA CUDA
cmake -DUSE_CUDA=1 ..
cmake --build . --config Release

# Vulkan (AMD, Intel or NVIDIA)
cmake -DUSE_VULKAN=1 ..
cmake --build . --config Release
```

The CUDA build targets Turing and Ampere GPUs (`75;86`) by default. Override with
`-DCMAKE_CUDA_ARCHITECTURES`, e.g. `61` for Pascal or `89` for Ada. Pascal and
older GPUs require a CUDA 12.x toolkit.

## Using the GPU in Your Code

Initialize the GPU once, upload the network weights, train, then sync the
trained weights back to the CPU before saving:

```c
#ifdef USE_GPU
    if (!ann_gpu_init())
        fprintf(stderr, "No GPU available, using CPU\n");
#endif

    PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
    // ... add layers ...

#ifdef USE_GPU
    ann_gpu_upload_network(pnet);   // training now runs on the GPU
#endif

    ann_train_network(pnet, x_train, y_train, x_train->rows);

#ifdef USE_GPU
    ann_gpu_sync_weights(pnet);     // optional: training already copied weights back
#endif

    ann_save_network(pnet, "model.nna");

#ifdef USE_GPU
    ann_gpu_free_network(pnet);     // release GPU buffers
#endif
    ann_free_network(pnet);
```

`USE_GPU` is defined automatically when any backend is enabled. If the
network was never uploaded, every call falls back to the CPU path.

Once uploaded, `ann_predict()` and `ann_predict_batch()` also run on the GPU.
Use `ann_predict_batch()` to get the full benefit, since it spreads transfer
overhead across many samples.

> Note: GPU training pays off for networks with 128+ node hidden layers and
> datasets of 1,000+ samples, where it is roughly 5–7× faster than the CPU on
> Apple Silicon. For tiny networks the GPU setup cost dominates and the CPU is
> faster. The online training API (`ann_train_step()`) always runs on the CPU.
> The GPU backends support only the default `real = float` build. Metal also
> runs Conv2D, MaxPool2D and Flatten layers; on an M-series Mac a CNN with
> 32/64 filters trains about 2x faster than the CPU and one with 64/128 about
> 5x. Very small CNNs (8/16 filters, like `fashion_cnn`) train at about CPU
> speed with batches of 128 and slower with batches of 32. For prediction
> with `ann_predict_batch()`, the GPU is about 1.6x faster than the CPU at
> 32/64 filters and 3x at 64/128; at 8/16 the CPU is faster. CUDA runs dense
> networks only.

See [docs/GPU_INFERENCE.md](docs/GPU_INFERENCE.md) (Metal),
[docs/GPU_CUDA.md](docs/GPU_CUDA.md) (CUDA), and
[docs/GPU_TRAINING_PERFORMANCE.md](docs/GPU_TRAINING_PERFORMANCE.md) (benchmarks)
for details. `examples/bench_gpu_training.c` compares CPU and GPU training speed.

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

By default, library errors are printed to stderr as `[function] Error code: message`. Install a callback to send them somewhere else, or a callback that does nothing to silence them. Either way, functions still return their error code (or `NULL`), so check results as usual.

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

// Remove the callback (errors go to stderr again)
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

To build with GPU acceleration (see [GPU training and inference](#gpu-training-and-inference)):

```bash
# Apple Metal (macOS)
cmake -DUSE_METAL=1 ..
cmake --build . --config Release

# NVIDIA CUDA (Windows, Linux)
cmake -DUSE_CUDA=1 ..
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

Only functions marked `ANN_API` are exported from the shared library on every
platform (`__declspec(dllexport)` on Windows, default visibility with
`-fvisibility=hidden` on gcc/clang), so the exported symbols are exactly the
public API.

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
* **fashion_cnn** - convolutional network (conv, max pooling, flatten) for Fashion-MNIST
* **fashion_dae** - convolutional denoising autoencoder (pooling, upsampling, input noise) for Fashion-MNIST; writes before/after images as PGM and PNG
* **mnist_hypertune_tpe** - TPE hyperparameter search on Fashion-MNIST, including per-layer activations
* **mnist_hypertune_random** - random search over the full search space on Fashion-MNIST
* **mnist_hypertune_bayes** - Bayesian (Gaussian Process) optimization on Fashion-MNIST
* **mnist_hypertune_grid** - grid search over a small search space on Fashion-MNIST

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
