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

- **Loss not decreasing?** Try lower learning rate (0.01-0.001) or switch to `OPT_ADAM` with lr=0.001
- **Loss oscillating/overshooting?** Use `OPT_ADAM` instead of `OPT_ADAPT` - Adam handles adaptive learning rates more stably
- **NaN in outputs?** Check for uninitialized weights or exploding gradients - reduce learning rate
- **Accuracy plateau?** Increase hidden layer size or add more layers
- **Slow training?** Enable BLAS (`-DUSE_BLAS=1`) for large networks

## Optimizer Selection

| Optimizer | Best For | Learning Rate |
|-----------|----------|---------------|
| `OPT_ADAM` | Most tasks (recommended default) | 0.001 |
| `OPT_SGD` | Simple problems, fine-tuning | 0.01-0.1 |
| `OPT_MOMENTUM` | When SGD oscillates | 0.01 |
| `OPT_RMSPROP` | Non-stationary problems | 0.001 |
| `OPT_ADAGRAD` | Sparse gradients | 0.01 |
| `OPT_ADAPT` | Legacy - may overshoot | 0.05 (auto-adjusts) |

## Convergence Settings

The default convergence threshold is **0.01** (loss ≤ 1%). This works well for:
- Simple classification (logic gates, small digit recognition)
- Normalized inputs with MSE or cross-entropy loss

For complex problems (MNIST, noisy data), 0.01 may be unreachable. Options:
1. **Increase the threshold**: `ann_set_convergence(pnet, 0.1)` for looser target
2. **Rely on epoch limit**: Training stops at `epochLimit` (default 10,000) and monitor final loss
3. **Early stopping**: Watch for loss plateau and stop manually

```c
// For complex problems, relax convergence or focus on epochs
ann_set_convergence(pnet, 0.1);  // Stop at 10% loss
// Or just let it run to epoch limit and check final accuracy
```

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
