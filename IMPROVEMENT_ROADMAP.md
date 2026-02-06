# libann Improvement Roadmap

A prioritized list of improvements and enhancements for the library.

## High Priority

- [x] **Complete Activation Function Backpropagation**
  - ~~README notes ReLU/Tanh/LeakyReLU/Softsign are "not yet functional"~~
  - ~~`back_propagate_relu` exists but has uncertain implementation~~
  - All backprop functions implemented:
    - [x] `back_propagate_relu` (fixed and documented)
    - [x] `back_propagate_tanh`
    - [x] `back_propagate_leaky_relu`
    - [x] `back_propagate_softsign`

- [x] **Convert Remaining Asserts to Error Callbacks**
  - ~~31 asserts in tensor.c (null checks, shape validation, bounds checking)~~
  - ~~7 asserts in ann.c~~
  - All converted to proper error handling with return codes

- [x] **Add Bias Updates to AdaGrad/RMSProp/Adam**
  - ~~Current optimizer implementations only update weights, not biases~~
  - ~~Follow pattern from other optimizers that update both~~
  - All optimizers now properly update biases using their respective algorithms

- [ ] **True Batched Training with GEMM** *(~35-50 hours)*
  - **Highest performance opportunity** - potential 10-100x speedup for typical batch sizes
  - Currently: samples processed one-by-one with gemv, gradients accumulated
  - Goal: process entire batch in parallel using gemm
  - Required changes:
    - [ ] Restructure layer storage - `t_values` from vector to matrix `(batch_size × nodes)` *(~8-12 hours)*
    - [ ] Rewrite forward pass - use gemm: `Output = Input × Weights^T + Bias` *(~4-6 hours)*
    - [ ] Rewrite backward pass - batch gradient computation with gemm *(~8-12 hours)*
    - [ ] Update activation functions - operate on matrices row-wise *(~2-4 hours)*
    - [ ] Update softmax - batch-aware per-row softmax *(~2-3 hours)*
    - [ ] Update optimizers - verify compatibility with batched gradients *(~1-2 hours)*
    - [ ] Testing & debugging *(~8-12 hours)*
  - Note: rank-3 tensors not required if batch activations stored as 2D matrices

## Medium Priority

- [ ] **Add Unit Tests for New Optimizers** *(~2-4 hours)*
  - AdaGrad, RMSProp, and Adam have no dedicated tests
  - Add to test_network.c or create test_optimizers.c

- [ ] **Gradient Clipping** *(~2-3 hours)*
  - Prevents exploding gradients, especially with ReLU activation
  - Clip gradient magnitudes during backprop: `grad = clip(grad, -max_val, max_val)`
  - Add configurable `max_gradient` parameter to network

- [ ] **Activation-Aware Weight Initialization** *(~3-5 hours)*
  - Current: uniform distribution with fixed `weight_limit`
  - Add He initialization for ReLU: `std = sqrt(2/fan_in)` *(~2-3 hours)*
  - Add Xavier/Glorot for sigmoid/tanh: `std = sqrt(2/(fan_in + fan_out))` *(~1-2 hours)*
  - Requires `tensor_random_normal()` function in tensor library

- [ ] **Batch Normalization Support** *(~8-16 hours)*
  - Would significantly improve training stability for deeper networks
  - Requires forward pass normalization, learnable gamma/beta, and backward pass gradients

- [ ] **Dropout Regularization** *(~4-8 hours)*
  - Randomly zero out neurons during training to prevent overfitting
  - Scale activations at inference time (or use inverted dropout)
  - Add configurable dropout rate per layer

## Low Priority

- [ ] **Vectorize Optimizer Loops with BLAS** *(~4-8 hours)*
  - Low ROI: optimizer runs once per batch, not the compute bottleneck
  - BLAS helps with axpy but not element-wise sqrt/divide
  - Consider only after batched GEMM training is implemented

- [ ] **Add Learning Rate Schedulers** *(~4-6 hours total)*
  - Less critical when using Adam optimizer (has built-in adaptive rates)
  - Consider adding:
    - [ ] Step decay *(~1-2 hours)*
    - [ ] Exponential decay *(~1-2 hours)*
    - [ ] Cosine annealing *(~2 hours)*

- [ ] **Documentation Updates** *(~1-2 hours)*
  - [ ] Update tensor_gemm header comment (no longer "not fully implemented")
  - [ ] Add example usage for new optimizers in README

- [ ] **Code Cleanup** *(~1-2 hours)*
  - [ ] Remove duplicate includes in ann.c (lines 28-35 and 107-113)
  - [ ] Remove commented-out debug code

- [ ] **Performance Improvements** *(~16-32 hours total)*
  - [ ] Consider OpenMP parallelization for non-BLAS builds *(~4-8 hours)*
  - [ ] Profile and optimize training loop hot path *(~2-4 hours)*
  - [ ] **Optimize non-BLAS tensor operations:** *(~8-16 hours)*
    - [ ] `tensor_gemm`: Use loop tiling and i-k-j loop order for cache locality *(~4-6 hours)*
    - [ ] `tensor_matvec` transpose: Improve cache access pattern for row-major storage *(~2-3 hours)*
    - [ ] Element-wise ops: Loop unrolling (4x/8x) for tensor_add/sub/mul/square/fill *(~2-4 hours)*
    - [ ] `tensor_copy`: Use memcpy instead of loop *(~30 min)*
    - [ ] `tensor_fill(t, 0)`: Use memset for zero-fill case *(~30 min)*
    - [ ] `tensor_argmax`: Use direct array access instead of get/set_element function calls *(~30 min)*
    - [ ] `tensor_outer`: Loop unrolling for hot inner loop *(~1-2 hours)*

- [ ] **Additional Layer Types** *(~24-40 hours total)*
  - Currently only dense (fully-connected) layers supported
  - Potential additions (significant undertaking):
    - [ ] Convolutional layers (Conv2D) for image processing *(~16-24 hours)*
    - [ ] Pooling layers (MaxPool, AvgPool) *(~6-10 hours)*
    - [ ] Flatten layer for 2D to 1D transitions *(~2-4 hours)*

---

## Completed

_Move items here as they are finished:_

- [x] Implement `tensor_argmax`
- [x] Implement `tensor_gemm` with BLAS and scalar paths
- [x] Implement `optimize_adagrad`
- [x] Implement `optimize_rmsprop`
- [x] Implement `optimize_adam`
- [x] Update README with new optimizer status
- [x] Add missing functions to README documentation tables
- [x] Convert all asserts to error callbacks (31 in tensor.c, 7 in ann.c)
- [x] Add bias updates to AdaGrad/RMSProp/Adam optimizers
- [x] Complete activation function backpropagation (ReLU, LeakyReLU, Tanh, Softsign)
