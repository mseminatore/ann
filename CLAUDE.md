# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`libann` is a compact, portable ANSI C library for training and evaluating neural networks. It has two core modules — `tensor` (the math layer over vectors/matrices) and `ann` (the training/inference runtime built on top of `tensor`) — plus `ann_hypertune` (automated hyperparameter search), `json` (ONNX JSON I/O) and `ann_quantize` (int8 quantization and C header export). All five compile into `libann`. To embed in another project, add only `ann.c` + `tensor.c`, or link `libann`.

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
cmake -DUSE_ACCELERATE=1 ..       # Apple Accelerate (macOS), headers from the SDK via xcrun

# GPU backends (see docs/GPU_*.md)
cmake -DUSE_CUDA=1 ..             # NVIDIA, CUDA Toolkit
cmake -DUSE_VULKAN=1 ..           # AMD/Intel/NVIDIA, needs Vulkan headers + glslc (Vulkan SDK)

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

Test executables: `test_tensor`, `test_network`, `test_activations`, `test_loss_functions`, `test_optimizers`, `test_save_load`, `test_onnx_export`, `test_hypertune`, `test_json`, `test_training_convergence`, `test_online_training`, `test_noise`, `test_unload`, `test_quantize`, `test_conv`, `test_batchnorm`, `test_c_export` (+ `_int8`; built from headers that `gen_c_export` writes at build time).

Test files define `void test_main(int argc, char *argv[])` and use the `MODULE(...)` / `SUITE(...)` / `TESTEX("name", expr)` macros.

## Architecture

- **Module dependencies**: `ann.c` → `tensor.c`; `ann_hypertune.c` → `ann.c`; `json.c` provides the ONNX export/import serialization. `tensor` supports only rank-1/rank-2 (vectors/matrices), no rank-3.
- **Network construction is layered and order-sensitive**: create with `ann_make_network(optimizer, loss)`, then `ann_add_layer()` per layer — input layer uses `ACTIVATION_NULL`, hidden layers a nonlinearity, output `ACTIVATION_SOFTMAX` for classification. The optimizer (`OPT_ADAM` recommended, also SGD/Momentum/AdaGrad/RMSProp), loss (`LOSS_MSE`, `LOSS_CATEGORICAL_CROSS_ENTROPY`; `LOSS_BINARY_CROSS_ENTROPY` is an alias that gives BCE with a sigmoid output), and per-layer activation are chosen via enums at these call sites.
- **Input noise**: `ann_set_input_noise()` (Gaussian, masking, salt-and-pepper) corrupts each training batch's inputs right after it is copied into `layers[0].t_batch_values`, in both training paths, so the GPU backends get noisy inputs too; drawn serially with `rand()` like dropout masks. Never applied to predictions or evaluation. `examples/fashion_dae.c` uses it for a conv denoising autoencoder.
- **Two training paths**: `ann_train_network()` for full batch/epoch training (resets optimizer state and weights), versus the online API `ann_train_begin()` / `ann_train_step()` / `ann_train_end()` which **preserves optimizer state and weights** — use it for streaming, fine-tuning a loaded model, or single-sample updates. `ann_predict()` is safe mid-online-training (dropout auto-disabled).
- **Layer kinds**: every layer has a `Layer_kind` (`DENSE`, `CONV2D`, `MAXPOOL2D`, `FLATTEN`, `UPSAMPLE2D`, `RESHAPE`, `GLOBAL_AVGPOOL2D`, `BATCHNORM`), a `Layer_shape` (C, H, W; dense is `(nodes, 1, 1)`; `in_shape` is the layer below's), and owns the parameters that compute it from the layer below — `layers[i].t_weights` is `[nodes_i × nodes_{i-1}]`; the input layer has none. Trainable tensors are listed in `layers[i].params[]` (dense: W, b; `t_weights`/`t_bias` alias them). In `ann.c`, each kind implements a `Layer_ops` entry (setup, init, forward, backward, ONNX); optimizers, regularization, early stopping and freeing iterate `params[]`, and activations/dropout are applied by shared code. A new kind adds an enum value at the end, a `g_layer_ops` entry (setup validates its `config[]` and shape, so bad layers and bad v2 files are rejected), and ONNX hooks. Conv2D is im2col + GEMM per sample through stack `Tensor` views; grouped/depthwise convs (`ann_add_conv2d_grouped`, `ann_add_depthwise_conv2d`; groups in `config[CONV_GROUPS]`, W `[F x C/G*k*k]`, `ann_get_layer_groups`) keep the full im2col columns, whose rows are already grouped, and run each product as one GEMM per group on row blocks (`conv_group_gemm`); a conv with one group saves 3 config values (so older libann reads it) and a missing groups value loads as 1; MaxPool recomputes the argmax in backward. CNNs are built with `ann_add_input`/`ann_add_conv2d`/`ann_add_maxpool2d`/`ann_add_flatten`, decoders also with `ann_add_upsample2d` (nearest-neighbour) and `ann_add_reshape` (reuses Flatten's copies), and `ann_add_global_avgpool2d` gives (C, 1, 1) channel means in place of flatten + a big dense layer; `ann_add_batchnorm(net, act)` normalizes per channel (per node after dense) and applies `act` — the only kind besides dense/conv with an activation. Batch norm is the one kind that differs in training: `Layer_ops.forward_train` (batch statistics, x_hat kept for backward) vs `forward` (running statistics); its running mean/variance are `params[2..3]` with `state = 1`, so save/load, early stopping and GPU upload/sync carry them but optimizers (CPU and GPU) and clipping skip them, and quantization/C export fold it into the dense/conv layer before it (tested in `tests/test_batchnorm.c`, whose gradient checks read the training-mode loss from `ann_train_step` at learning rate 0); all tested in `tests/test_conv.c` (finite-difference gradient checks; the ONNX import cases embed reference outputs from onnxruntime). Metal runs all eight kinds on the GPU (grouped convs: their own direct kernels for the forward pass, input gradient and a threadgroup per weight for dW; other conv: im2col + one MPS GEMM per batch, with the activation derivative and bias gradient fused into the backward kernels; the weight gradient uses split-K partial sums when F*C*K*K is small; layers with at most `CONV_DIRECT_FILTERS` (4) filters convolve directly, with no im2col columns); Vulkan also runs all eight (grouped convs: `conv_grouped.comp`, the same three direct steps; other conv: im2col + one GEMM per batch, with the weight gradient split over K and bias gradients reduced per workgroup; layers with at most 8 filters convolve directly, as Metal's do up to 4; `gemm.comp` has 5 tile shapes and splits K over ~128 workgroups when a GEMM has few output tiles; `ANN_VK_PROFILE=1` prints GPU time by kernel); CUDA is dense-only, and a backend's `upload_network` returns `ERR_INVALID` for kinds it lacks, which keeps that network on the CPU. GPU training is pipelined: `train_batch` copies the batch (inputs, targets, dropout masks) and submits the step without waiting, `ann_train_network` collects the epoch's losses with `train_finish`, and Metal keeps `TRAIN_SLOTS` (2) steps in flight with per-slot buffers; anything that reads or replaces GPU state (sync, upload, free, eval) first waits with `finish_all_slots()`. Vulkan does the same with `TRAIN_SLOTS` (2) slots (own command buffer, fence, descriptor pools, staging, inputs/targets/masks/losses), waiting for the steps in flight (`finish_all_slots()`) before anything that reads or replaces shared GPU state; `ANN_VK_DEBUG=1` runs it under the validation layer with synchronization validation. CUDA runs each step synchronously behind the same interface. Metal and Vulkan CNN results are checked against the CPU by the same CNN suite in `tests/test_metal.c` and `tests/test_vulkan.c`. The Vulkan backend (`tensor_vulkan.c`, GLSL shaders in `shaders/vk/` compiled by glslc and embedded via `cmake/embed_spirv.cmake`) covers AMD/Intel/NVIDIA on Windows/Linux: it opens the Vulkan loader at run time (no link dependency), uses its own tiled GEMM shader, maps device memory when it is host-visible and large, else copies through a staging buffer, and is tested by `tests/test_vulkan.c` (also run with `ANN_VK_STAGING=1`/`0`; CI runs it on lavapipe). Each `GpuBackend` has a `name`; `ann_gpu_init()` tries the compiled-in backends in order (Metal, Vulkan, CUDA — Vulkan first since CUDA is dense-only) and the `ANN_GPU` env var (`metal|vulkan|cuda|none`) picks one; see docs/GPU_VULKAN.md.
- **Model persistence**: `.nna` (text), `.nnb` (binary), and ONNX JSON via `ann_export_onnx`/`ann_import_onnx` (the importer walks the node chain, so it handles dense and CNN models, including PyTorch-style JSON; see docs/ONNX_EXPORT.md). Saves write format v1 when every layer is dense with a flat shape (readable by older libann) and v2 otherwise (records each layer's kind, shape and config); loaders read both. See the format comment above `save_network` in `ann.c`.

## Conventions

- ANSI C for portability (Windows/macOS/Linux). Platform-specific code is guarded by `#ifdef _WIN32`.
- `real` typedef is the scalar type — `float` by default; switch to `double` in `tensor.h`.
- Public functions are prefixed `ann_*` (runtime) or `tensor_*` (math). Mark every public declaration with `ANN_API`: the shared library is built with hidden visibility, so unmarked functions are not exported (and the `../pyann` CFFI bindings rely on that surface). New enum values go at the end of their enum, since saved models and pyann store the integer values.
- All public functions return error codes (`ERR_OK`, `ERR_NULL_PTR`, `ERR_ALLOC`, `ERR_INVALID`, `ERR_IO`, `ERR_FAIL`). Use the `CHECK_OK(...)` / `CHECK_NULL(...)` macros to propagate failures; install an `ErrorLogCallback` via `ann_set_error_log_callback` for custom logging. `ann_strerror(code)` gives a readable message.
- Tensors are row-major: element at `data[row * stride + col]`.
- Pair every `tensor_create*()` with `tensor_free()` and `ann_make_network()` with `ann_free_network()`. Data from `ann_load_csv()` must be freed by the caller with `free()`.
- The BLAS toggle is `USE_BLAS` in `ann_config.h` (or the CMake flags above); the scalar path is always available as a fallback.
- **Threading**: `ann_parallel_for()` (internal, `ann_parallel.h`, pool in `tensor.c`) runs libann's own loops on worker threads; `ann_set_num_threads`/`ANN_NUM_THREADS` set the count. Rules: range functions call BLAS only when `ann_parallel_blas_ok()` allows it — where BLAS threading can be set per thread (Accelerate on macOS 15+), `ann_parallel_for` runs ranges with single-threaded BLAS so many small per-sample GEMMs (conv) run side by side; with other BLAS libraries BLAS calls stay on the calling thread, so its own threads aren't oversubscribed, and conv batches its GEMMs instead (one per product per batch). Work is split so each output is computed whole by one thread in serial order — results must stay bit-identical for any thread count (`tests/test_threads.c` checks this). BLAS calls inside range functions must have shapes that don't depend on the split, since BLAS may round an output differently depending on the GEMM's shape (Accelerate does): conv's dW GEMMs go to threads in filter blocks sized from the filter count alone (`conv_filter_block`). Nested or concurrent calls run serially. Workers are joined at exit on POSIX; on Windows a DLL's `atexit` runs under the loader lock, where joining deadlocks, so `pool_start()` pins the module instead and the workers end with the process (pyann's cffi unloads libann at interpreter exit; `tests/test_unload.c` loads a shared copy, starts the pool and checks the unload returns). `-DANN_NO_THREADS` compiles the pool out. `ann_get_backend_info()` reports the BLAS library (named in `tensor.c`, which includes the BLAS header: `ANN_BLAS_NAME` if defined, else from the build flags and header macros), GPU backend and thread count; `ann_print_props()` prints it as the training header's `Backend:` line.

## More detail

For the full API/feature reference and worked examples, see `README.md` and `.github/copilot-instructions.md`. Deeper topic guides live in `docs/`: `ONNX_EXPORT.md`, `C_EXPORT.md`, `HYPERTUNING.md`, `DEBUGGING_TIPS.md`, `GPU_CUDA.md`, `GPU_VULKAN.md`.
