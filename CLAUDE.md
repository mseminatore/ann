# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`libann` is a compact, portable ANSI C library for training and evaluating neural networks. It has two core modules — `tensor` (the math layer over vectors/matrices) and `ann` (the training/inference runtime built on top of `tensor`) — plus `ann_hypertune` (automated hyperparameter search) and `json` (ONNX JSON I/O). All four compile into `libann`. To embed in another project, add only `ann.c` + `tensor.c`, or link `libann`.

## Project layout

- Library sources (`.c`, `.h`) live in the **project root**
- `tests/` — test source files (`test_*.c`), use the `testy` framework
- `examples/` — example programs (`mnist.c`, `logic.c`, `digit5x7.c`, etc.)
- `data/` — CSV datasets, saved models (`.nna`, `.nnb`), ONNX exports
- `testy/` — embedded test framework
- `docs/` — topic guides (ONNX, hypertuning, debugging, GPU)

## Build

CMake is preferred and drives the test suite. There is also a root `Makefile`.

```bash
# CMake (preferred) — run from project root
mkdir build && cd build
cmake ..                          # scalar build (no BLAS)
cmake --build . --config Release

# BLAS acceleration (training speedup; inference is fast without it)
cmake -DUSE_BLAS=1 ..             # OpenBLAS, expects /opt/OpenBLAS
cmake -DUSE_CBLAS=1 ..            # CBLAS, expects /opt/cblas (requires C11 atomics)
cmake -DUSE_MKL=1 ..              # Intel MKL, uses $ONEAPI_ROOT env var

# Shared library (libann.so/.dylib/.dll)
cmake -DBUILD_SHARED=1 ..

# Make alternative
make                              # builds lib, examples, and tests
make shared                       # shared library
```

Note: the root `Makefile` currently hard-codes OpenBLAS at `/opt/OpenBLAS`. On a machine without it, edit the `CFLAGS`/`LFLAGS` BLAS lines near the top of the `Makefile` (or use the CMake scalar build). Library install paths live in `CMakeLists.txt` if yours differ.

## Test

Test source files live in `tests/`. They use the embedded `testy` framework (`testy/test.h`, linked with `testy/test_main.c`). Each test is its own executable, registered with CTest.

```bash
cd build
ctest --output-on-failure         # run all tests
ctest -R test_tensor              # run a single test by name
./test_tensor                     # or run the executable directly
```

Test executables: `test_tensor`, `test_network`, `test_activations`, `test_loss_functions`, `test_optimizers`, `test_save_load`, `test_onnx_export`, `test_hypertune`, `test_json`, `test_training_convergence`, `test_online_training`.

Test files define `void test_main(int argc, char *argv[])` and use the `MODULE(...)` / `SUITE(...)` / `TESTEX("name", expr)` macros.

## Architecture

- **Module dependencies**: `ann.c` → `tensor.c`; `ann_hypertune.c` → `ann.c`; `json.c` provides the ONNX export/import serialization. `tensor` supports only rank-1/rank-2 (vectors/matrices), no rank-3.
- **Network construction is layered and order-sensitive**: create with `ann_make_network(optimizer, loss)`, then `ann_add_layer()` per layer — input layer uses `ACTIVATION_NULL`, hidden layers a nonlinearity, output `ACTIVATION_SOFTMAX` for classification. The optimizer (`OPT_ADAM` recommended, also SGD/Momentum/AdaGrad/RMSProp), loss (`LOSS_MSE`, `LOSS_CATEGORICAL_CROSS_ENTROPY`), and per-layer activation are chosen via enums at these call sites.
- **Two training paths**: `ann_train_network()` for full batch/epoch training (resets optimizer state and weights), versus the online API `ann_train_begin()` / `ann_train_step()` / `ann_train_end()` which **preserves optimizer state and weights** — use it for streaming, fine-tuning a loaded model, or single-sample updates. `ann_predict()` is safe mid-online-training (dropout auto-disabled).
- **Model persistence**: `.nna` (text), `.nnb` (binary), and ONNX JSON via `ann_export_onnx`/`ann_import_onnx`.

## Conventions

- ANSI C for portability (Windows/macOS/Linux). Platform-specific code is guarded by `#ifdef _WIN32`.
- `real` typedef is the scalar type — `float` by default; switch to `double` in `tensor.h`.
- Public functions are prefixed `ann_*` (runtime) or `tensor_*` (math).
- All public functions return error codes (`ERR_OK`, `ERR_NULL_PTR`, `ERR_ALLOC`, `ERR_INVALID`, `ERR_IO`, `ERR_FAIL`). Use the `CHECK_OK(...)` / `CHECK_NULL(...)` macros to propagate failures; install an `ErrorLogCallback` via `ann_set_error_log_callback` for custom logging. `ann_strerror(code)` gives a readable message.
- Tensors are row-major: element at `data[row * stride + col]`.
- Pair every `tensor_create*()` with `tensor_free()` and `ann_make_network()` with `ann_free_network()`. Data from `ann_load_csv()` must be freed by the caller with `free()`.
- The BLAS toggle is `USE_BLAS` in `ann_config.h` (or the CMake flags above); the scalar path is always available as a fallback.

## More detail

For the full API/feature reference and worked examples, see `README.md` and `.github/copilot-instructions.md`. Deeper topic guides live in `docs/`: `ONNX_EXPORT.md`, `HYPERTUNING.md`, `DEBUGGING_TIPS.md`.
