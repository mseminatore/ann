# libann - AI Coding Guidelines

## Architecture Overview

This is a portable **C neural network library** with two core components:
- [tensor.c](../tensor.c) / [tensor.h](../tensor.h) - Lightweight tensor math (vectors/matrices only, no rank-3)
- [ann.c](../ann.c) / [ann.h](../ann.h) - Neural network training and inference runtime

**Integration**: Add only `ann.c` and `tensor.c` to any project, or link to `libann`.

## Key Patterns

### Error Handling Convention
All public functions return error codes. Use the validation macros consistently:
```c
CHECK_OK(ann_add_layer(pnet, 10, LAYER_HIDDEN, ACTIVATION_SIGMOID));  // Returns on error
CHECK_NULL(ptr);  // Returns ERR_NULL_PTR if NULL
```
Error codes: `ERR_OK`, `ERR_NULL_PTR`, `ERR_ALLOC`, `ERR_INVALID`, `ERR_IO`, `ERR_FAIL`

For custom error logging, use the callback system:
```c
ann_set_error_log_callback(my_handler);  // ErrorLogCallback signature in ann.h
```

### Memory Management
- Always pair `tensor_create*()` with `tensor_free()` and `ann_make_network()` with `ann_free_network()`
- Data loaded via `ann_load_csv()` must be freed by caller with `free()`
- Tensors use row-major order: `data[row * stride + col]`

### Network Construction Pattern
```c
PNetwork pnet = ann_make_network(OPT_MOMENTUM, LOSS_CATEGORICAL_CROSS_ENTROPY);
ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);      // Input: always ACTIVATION_NULL
ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);  // Hidden layers
ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);   // Output: SOFTMAX for classification
```

### BLAS Acceleration
- Controlled by `USE_BLAS` in [ann_config.h](../ann_config.h) or via `-DUSE_BLAS=1` CMake flag
- Links against OpenBLAS; update paths in CMakeLists.txt if installed elsewhere than `/opt/`
- Non-BLAS scalar path always available as fallback

## Build & Test Commands

```bash
# CMake build (preferred)
mkdir build && cd build
cmake ..                          # Standard build
cmake -DUSE_BLAS=1 ..             # With OpenBLAS acceleration
cmake --build . --config Release

# Run all tests
ctest --output-on-failure

# Make build alternative
make                              # From project root
```

## Test Framework

Uses embedded [testy](../testy/test.h) framework. Test files follow this pattern:
```c
#include "testy/test.h"

void test_main(int argc, char *argv[]) {
    MODULE("Module Name");
    SUITE("Feature Suite");
    TESTEX("descriptive test name", (expression_that_should_be_true));
}
```
Test executables: `test_tensor`, `test_network`, `test_activations`, `test_onnx_export`

## Code Conventions

- **ANSI C** for portability (Windows/macOS/Linux)
- `real` typedef (float by default, change in tensor.h for double precision)
- Prefix all public functions: `ann_*` for network, `tensor_*` for tensors
- Header documentation uses doxygen-style `@param`/`@return` comments
- Platform-specific includes handled via `#ifdef _WIN32` blocks

## Supported Features (and Limitations)

| Feature | Status |
|---------|--------|
| Activations | Sigmoid, Softmax, ReLU, LeakyReLU, Tanh, Softsign |
| Optimizers | SGD, Momentum, Adam, AdaGrad, RMSProp (not yet vectorized) |
| Loss Functions | MSE, Categorical Cross-Entropy |
| Model Export | Text (`.nna`), Binary (`.nnb`), ONNX JSON |

## Example Programs

- [logic.c](../logic.c) - Linear regression (AND/OR/NAND gates)
- [digit5x7.c](../digit5x7.c) - Multi-class image classification
- [mnist.c](../mnist.c) - Full MNIST digit/fashion training

## Additional Documentation

- [ONNX Export Guide](../docs/ONNX_EXPORT.md) - Exporting models to ONNX JSON format
- [Debugging Tips](../docs/DEBUGGING_TIPS.md) - Common pitfalls and troubleshooting
