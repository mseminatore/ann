# libann Improvement Roadmap

A prioritized list of improvements and enhancements for the library.

## High Priority

- [ ] **Complete Activation Function Backpropagation**
  - README notes ReLU/Tanh/LeakyReLU/Softsign are "not yet functional"
  - `back_propagate_relu` exists but has uncertain implementation (comment: "??? is that right?")
  - Missing implementations:
    - [ ] `back_propagate_tanh`
    - [ ] `back_propagate_leaky_relu`
    - [ ] `back_propagate_softsign`

- [x] **Convert Remaining Asserts to Error Callbacks**
  - ~~31 asserts in tensor.c (null checks, shape validation, bounds checking)~~
  - ~~7 asserts in ann.c~~
  - All converted to proper error handling with return codes

- [ ] **Add Bias Updates to AdaGrad/RMSProp/Adam**
  - Current optimizer implementations only update weights, not biases
  - Follow pattern from other optimizers that update both

## Medium Priority

- [ ] **Add Unit Tests for New Optimizers**
  - AdaGrad, RMSProp, and Adam have no dedicated tests
  - Add to test_network.c or create test_optimizers.c

- [ ] **Vectorize Optimizer Loops with BLAS**
  - All three new optimizers use scalar loops
  - Could use `tensor_axpby` for momentum updates
  - Consider custom BLAS-based operations for element-wise sqrt/divide

- [ ] **Add Learning Rate Schedulers**
  - Currently only adaptive rate adjustment exists
  - Consider adding:
    - [ ] Step decay
    - [ ] Exponential decay
    - [ ] Cosine annealing

- [ ] **Batch Normalization Support**
  - Would significantly improve training stability for deeper networks

## Low Priority

- [ ] **Documentation Updates**
  - [ ] Update tensor_gemm header comment (no longer "not fully implemented")
  - [ ] Add example usage for new optimizers in README

- [ ] **Code Cleanup**
  - [ ] Remove duplicate includes in ann.c (lines 28-35 and 107-113)
  - [ ] Remove commented-out debug code

- [ ] **Performance Improvements**
  - [ ] Consider OpenMP parallelization for non-BLAS builds
  - [ ] Profile and optimize training loop hot path

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
