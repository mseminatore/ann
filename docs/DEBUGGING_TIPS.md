# Debugging Tips & Common Pitfalls

## Network Architecture

- **XOR requires hidden layers** - Linear models (input→output only) cannot learn non-linearly-separable functions
- **Input layer must use `ACTIVATION_NULL`** - Activations apply to layer outputs, not inputs
- **Match output activation to loss**: Use `ACTIVATION_SOFTMAX` + `LOSS_CATEGORICAL_CROSS_ENTROPY` for classification

## Data Preparation

- **Normalize inputs** to [0,1] range (e.g., `tensor_mul_scalar(x_train, 1.0f/255.0f)` for image data)
- **CSV loading allocates memory** - caller must `free(data)` after creating tensors
- **`tensor_slice_cols()` modifies the source tensor** - it removes columns from original and returns them in new tensor

## Training Issues

- **Loss not decreasing?** Try lower learning rate (0.01-0.001) or different optimizer
- **NaN in outputs?** Check for uninitialized weights or exploding gradients - reduce learning rate
- **Accuracy plateau?** Increase hidden layer size or add more layers
- **Slow training?** Enable BLAS (`-DUSE_BLAS=1`) for large networks

## Memory Debugging

- **Always check return values** - `tensor_create()` and `ann_make_network()` return NULL on allocation failure
- **Cleanup order matters** - free tensors before freeing data they were created from
- **Use error callbacks** during development:
  ```c
  void debug_errors(int code, const char *msg, const char *func) {
      fprintf(stderr, "[%s] %s\n", func, msg);
  }
  ann_set_error_log_callback(debug_errors);
  ```

## Platform-Specific

- **Windows x86**: Uses `_aligned_malloc`/`_aligned_free` for tensor memory
- **Linux builds**: Link math library (`-lm`) - handled automatically by CMake
