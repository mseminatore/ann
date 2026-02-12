# libann Improvement Roadmap

A prioritized list of remaining improvements and enhancements for the library.

## Medium Priority

- [ ] **Batch Normalization Support** *(~8-16 hours)*
  - Would significantly improve training stability for deeper networks
  - Requires forward pass normalization, learnable gamma/beta, and backward pass gradients

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

- [ ] **Documentation Updates** *(~1-2 hours)*
  - [ ] Add example usage for new optimizers in README

- [ ] **Multi-class Confusion Matrix** *(~2-3 hours)*
  - Extend `ann_confusion_matrix` to support N-class problems
  - NxN matrix output with per-class precision/recall
  - Multi-class MCC calculation

- [ ] **Code Cleanup** *(~1-2 hours)*
  - [x] Remove duplicate includes in ann.c ~~(lines 28-35 and 107-113)~~ ✓
  - [x] Remove dead `print_network` function ✓
  - [ ] Remove commented-out debug code (minimal remaining)

- [ ] **Performance Improvements** *(~12-24 hours remaining)*
  - [ ] Consider OpenMP parallelization for non-BLAS builds *(~4-8 hours)*
  - [ ] **Optimize non-BLAS tensor operations:**
    - [ ] `tensor_gemm`: Use loop tiling and i-k-j loop order for cache locality *(~4-6 hours)*
    - [ ] Fuse `tensor_matvec` + `tensor_add` in `eval_network` (y = Wx + b in one pass)

- [ ] **Additional Layer Types** *(~24-40 hours total)*
  - Currently only dense (fully-connected) layers supported
  - Potential additions (significant undertaking):
    - [ ] Convolutional layers (Conv2D) for image processing *(~16-24 hours)*
    - [ ] Pooling layers (MaxPool, AvgPool) *(~6-10 hours)*
    - [ ] Flatten layer for 2D to 1D transitions *(~2-4 hours)*

---

## Completed

- True Batched Training with GEMM (10-100x speedup)
- All activation function backpropagation (ReLU, LeakyReLU, Tanh, Softsign)
- Error callbacks (replaced all asserts)
- Bias updates in AdaGrad/RMSProp/Adam
- Unit tests for optimizers
- Gradient clipping (`ann_set_gradient_clip()`)
- Activation-aware weight initialization (He/Xavier/Glorot)
- Dropout regularization (`ann_set_layer_dropout()`)
- ONNX JSON export & import (`ann_export_onnx()`, `ann_import_onnx()`)
- Learning rate schedulers (step, exponential, cosine)
- PIKCHR network diagram export
- Learning curve CSV export
- Tensor optimizations (memcpy, memset, loop unrolling, cache-friendly access)
- TPE hyperparameter optimization (`hypertune_tpe_search()`)
- L1/L2 weight regularization (`ann_set_weight_decay()`, `ann_set_l1_regularization()`)
