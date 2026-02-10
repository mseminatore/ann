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

- [x] **True Batched Training with GEMM** *(~35-50 hours)*
  - **Highest performance opportunity** - potential 10-100x speedup for typical batch sizes
  - Implementation complete:
    - [x] Restructure layer storage - added `t_batch_values`, `t_batch_dl_dz`, `t_batch_z` matrices
    - [x] Rewrite forward pass - use gemm: `Y = X × W^T + Bias` with `tensor_gemm_transB()`
    - [x] Rewrite backward pass - batch gradient computation with `tensor_gemm_transA()`
    - [x] Update activation functions - operate on matrices row-wise
    - [x] Update softmax - batch-aware per-row softmax (`softmax_batched()`)
    - [x] Update optimizers - verified compatibility with batched gradients
    - [x] Testing & debugging - all 13 tests passing
  - Added `tensor_gemm_transA()` and `tensor_gemm_transB()` with BLAS and scalar fallback
  - Dynamic batch tensor allocation via `ensure_batch_tensors()` when batch size changes

## Medium Priority

- [x] **Add Unit Tests for New Optimizers** *(~2-4 hours)*
  - AdaGrad, RMSProp, and Adam now have dedicated tests in test_optimizers.c
  - Tests verify convergence on XOR problem and proper state tensor allocation

- [x] **Gradient Clipping** *(~2-3 hours)*
  - Prevents exploding gradients, especially with ReLU activation
  - Implemented via `ann_set_gradient_clip()` - clips to [-max_grad, max_grad]
  - Applied in all optimizers before weight updates

- [x] **Activation-Aware Weight Initialization** *(~3-5 hours)*
  - He initialization for ReLU: `std = sqrt(2/fan_in)`
  - Xavier/Glorot for sigmoid/tanh: `std = sqrt(2/(fan_in + fan_out))`
  - Auto-selection based on layer activation (WEIGHT_INIT_AUTO)
  - Added `tensor_random_normal()` and `tensor_clip()` to tensor library

- [ ] **Batch Normalization Support** *(~8-16 hours)*
  - Would significantly improve training stability for deeper networks
  - Requires forward pass normalization, learnable gamma/beta, and backward pass gradients

- [x] **Dropout Regularization** *(~4-8 hours)* ✓
  - Randomly zero out neurons during training to prevent overfitting
  - Uses inverted dropout (scale during training, no adjustment at inference)
  - Configurable per-layer dropout rates via `ann_set_layer_dropout()`

- [ ] **L1/L2 Weight Regularization** *(~2-3 hours)*
  - L2 (Ridge): Penalize large weights, reduces overfitting
  - L1 (LASSO): Encourage sparse weights for feature selection
  - Add `ann_set_weight_decay()` for L2, `ann_set_l1_regularization()` for L1
  - Apply in all optimizers during weight update

- [ ] **ONNX JSON Import (Round-Trip)** *(~8-10 hours)*
  - Import models from the JSON format libann exports via `ann_export_onnx()`
  - Enables model exchange and editing outside C code
  - Implementation:
    - [ ] Add lightweight JSON parser (cJSON or minimal custom) *(~3-4 hours)*
    - [ ] Parse graph topology and validate sequential dense structure *(~2-3 hours)*
  - Import models from the JSON format libann exports via `ann_export_onnx()`
  - Enables model exchange and editing outside C code
  - Implementation:
    - [ ] Add lightweight JSON parser (cJSON or minimal custom) *(~3-4 hours)*
    - [ ] Parse graph topology and validate sequential dense structure *(~2-3 hours)*
    - [ ] Extract weights/biases from initializers *(~2 hours)*
    - [ ] Map ONNX ops to libann activations *(~1 hour)*
  - Rejects unsupported ops (Conv, Pool, etc.) with clear error message

## Low Priority

- [ ] **Complete Network Serialization for Resumable Training** *(~4-6 hours)*
  - Current save/load works for inference (weights + biases saved)
  - Missing properties for training checkpoint/resume:
    - [ ] `learning_rate`, `batchSize`, `convergence_epsilon`, `epochLimit`
    - [ ] `max_gradient` (gradient clipping threshold)
    - [ ] `train_iteration` (critical for Adam optimizer bias correction)
    - [ ] Optimizer state tensors (`t_m`, `t_v`, `t_bias_m`, `t_bias_v`)
  - Consider separate "checkpoint" format vs lightweight "inference" format
  - Would require incrementing `ANN_BINARY_FORMAT_VERSION` / `ANN_TEXT_FORMAT_VERSION`

- [ ] **Vectorize Optimizer Loops with BLAS** *(~4-8 hours)*
  - Low ROI: optimizer runs once per batch, not the compute bottleneck
  - BLAS helps with axpy but not element-wise sqrt/divide
  - Consider only after batched GEMM training is implemented

- [x] **Add Learning Rate Schedulers** *(~4-6 hours total)*
  - Callback-based API via `ann_set_lr_scheduler()`
  - Built-in schedulers:
    - [x] Step decay: `lr_scheduler_step()` - halve LR every N epochs
    - [x] Exponential decay: `lr_scheduler_exponential()` - LR *= gamma each epoch
    - [x] Cosine annealing: `lr_scheduler_cosine()` - smooth decay to min_lr

- [x] **PIKCHR Network Diagram Export** *(~2-3 hours)* ✓
  - Export network structure as PIKCHR diagram (renders to SVG)
  - `ann_export_pikchr(pnet, filename)` function
  - Simple mode: boxes with layer info for large networks (>10 nodes/layer)
  - Detailed mode: show individual nodes for small networks

- [x] **Learning Curve Export** *(~2-3 hours)* ✓
  - Store loss/accuracy history during training
  - `ann_export_learning_curve(pnet, filename)` exports CSV
  - Format: epoch,loss,learning_rate
  - Enables plotting with gnuplot, matplotlib, Excel

- [ ] **Documentation Updates** *(~1-2 hours)*
  - [ ] Update tensor_gemm header comment (no longer "not fully implemented")
  - [ ] Add example usage for new optimizers in README

- [ ] **Multi-class Confusion Matrix** *(~2-3 hours)*
  - Extend `ann_confusion_matrix` to support N-class problems
  - NxN matrix output with per-class precision/recall
  - Multi-class MCC calculation

- [ ] **Code Cleanup** *(~1-2 hours)*
  - [ ] Remove duplicate includes in ann.c (lines 28-35 and 107-113)
  - [ ] Remove commented-out debug code

- [ ] **Performance Improvements** *(~16-32 hours total)*
  - [ ] Consider OpenMP parallelization for non-BLAS builds *(~4-8 hours)*
  - [x] Profile and optimize training loop hot path
    - Replaced element-by-element input copy with memcpy in `train_pass_network`
    - Removed unused variables in training loop
    - Moved progress indicator outside inner loop (per-batch instead of per-sample)
  - [ ] **Optimize non-BLAS tensor operations:** *(~8-16 hours)*
    - [ ] `tensor_gemm`: Use loop tiling and i-k-j loop order for cache locality *(~4-6 hours)*
    - [ ] Fuse `tensor_matvec` + `tensor_add` in `eval_network` (y = Wx + b in one pass)
    - [x] `tensor_matvec` transpose: Cache-friendly row-major access pattern + 4x loop unrolling
    - [x] Element-wise ops: Loop unrolling (4x) for tensor_add/sub/mul/square/fill
    - [x] `tensor_copy`: Use memcpy instead of loop
    - [x] `tensor_fill(t, 0)`: Use memset for zero-fill case
    - [x] `tensor_argmax`: Use direct array access instead of get/set_element function calls
    - [x] `tensor_outer`: Loop unrolling (4x) for hot inner loop

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
- [x] Gradient clipping (`ann_set_gradient_clip()`, `tensor_clip()`)
- [x] Activation-aware weight initialization (He/Xavier/Glorot, `tensor_random_normal()`)
- [x] Tensor optimizations: memcpy for `tensor_copy`, memset for zero-fill, 4x loop unrolling for element-wise ops and `tensor_outer`, direct array access in `tensor_argmax`
- [x] `tensor_matvec` transpose: cache-friendly row-major access + 4x unrolling (eliminates column-stride access)
- [x] Training loop hot path: memcpy for input copy, removed redundant variables, per-batch progress output
- [x] **True Batched Training with GEMM** - process entire mini-batch in parallel using gemm for 10-100x speedup
  - Added `tensor_gemm_transA()` and `tensor_gemm_transB()` for batched matrix operations
  - Batch activation matrices: `t_batch_values`, `t_batch_dl_dz`, `t_batch_z` per layer
  - Dynamic reallocation when batch size changes via `ensure_batch_tensors()`
  - Batched forward pass: `Y = X × W^T + B` (single gemm per layer)
  - Batched backward pass: gradient computation via `dW = delta^T × A` (single gemm per layer)
