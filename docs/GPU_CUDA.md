# CUDA Backend (WIP)

This repository now includes CUDA backend wiring via `GpuBackend` with:

- `tensor_cuda.cu` — CUDA backend registration + GPU memory management
- `shaders.cu` — initial CUDA kernel set
- `CMakeLists.txt` option `-DUSE_CUDA=1`

## Build

```bash
mkdir build_cuda && cd build_cuda
cmake -DUSE_CUDA=1 ..
cmake --build . --config Release
```

## Current scope

The CUDA backend currently provides:

- device/context initialization (`ann_gpu_init()` through `cuda_backend`)
- upload/download/free for network weight and bias tensors
- vtable integration for future CUDA inference/training paths

`eval_single`, `eval_batch`, and `train_batch` are scaffolded and intentionally return fallback/error until full kernel + cuBLAS training/inference paths are completed.
