# CUDA Backend

The CUDA backend implements GPU-accelerated **inference and training** through the
`GpuBackend` vtable, alongside the Metal backend. It consists of:

- `tensor_cuda.cu` — CUDA backend registration, GPU memory management, and the
  inference/training compute paths (cuBLAS GEMMs + kernel launches)
- `shaders.cu` — CUDA kernels (bias-add, activations + derivatives, softmax, bias-gradient
  reduction, gradient clipping, the five optimizers, L1/L2 regularization)
- `shaders_cuda.h` — kernel prototypes shared between `shaders.cu` and `tensor_cuda.cu`
- `CMakeLists.txt` option `-DUSE_CUDA=1`

## Build

```bash
mkdir build_cuda && cd build_cuda
cmake -DUSE_CUDA=1 ..
cmake --build . --config Release
```

Requires the CUDA Toolkit (`cudart`, `cublas`) and an NVIDIA GPU.

## What runs on the GPU

- **Inference** — `ann_predict()` (single sample) and `ann_predict_batch()` (batched) run the
  full forward pass on the GPU: cuBLAS `Y = X·Wᵀ` per layer, bias add, and the activation
  (including softmax) via kernels. Weights must be uploaded first with
  `ann_gpu_upload_network()`. If a network is not uploaded, the call transparently falls back
  to the CPU path.
- **Training** — `ann_train_network()` runs each mini-batch entirely on the GPU: forward pass
  (saving pre-activations), backward pass (two cuBLAS GEMMs per layer for `dW` and delta
  propagation, plus activation-derivative and bias-gradient kernels), and the optimizer update
  (SGD / Momentum / AdaGrad / RMSProp / Adam, with optional gradient clipping and L1/L2
  regularization). Call `ann_gpu_sync_weights()` afterward to copy trained weights back to the
  CPU for `ann_predict()` / `ann_save_network()`.

## Implementation notes

- **Row-major ↔ column-major.** libann tensors are row-major; cuBLAS is column-major. All GEMMs
  go through a single `cuda_gemm_rowmajor()` helper that computes the row-major result by issuing
  the equivalent column-major (transposed) `cublasSgemm` — operands swapped, `m`/`n` swapped.
- **Device memory is not host-visible.** Unlike Metal's unified (shared) memory, CUDA
  `cudaMalloc` buffers cannot be read directly by the CPU. The forward/backward/optimizer steps
  stay entirely on the device. The one host-side step — computing `delta = T − Y` and the loss —
  round-trips: the output activations are copied device→host, `delta`/loss are computed on the
  CPU (identical math to the Metal backend), and `delta` is copied host→device for backprop.
- **float-only.** The kernels and `cublasSgemm` operate on `float`. libann's default
  `real = float`, so the default build is fully supported. A `double` build (changing `real` in
  `tensor.h`) is **not** supported by the GPU backends (Metal or CUDA) and will fall back to /
  mismatch the CPU path — this is a shared GPU-backend limitation, not a CUDA-specific one.

## Test

`tests/test_cuda.c` (built only under `-DUSE_CUDA=1`) verifies that GPU inference and training
match the CPU path within tolerance:

```bash
cd build_cuda
ctest -R test_cuda --output-on-failure
```

`examples/bench_gpu_training.c` provides a CPU-vs-GPU training throughput comparison.

## Portability: Intel and AMD GPUs

The `GpuBackend` vtable (`ann_gpu_backend.h`) is **backend-agnostic** and already supports
additional GPU vendors with no interface changes:

- The eight vtable slots operate only on `PNetwork` and host `real *` arrays; each tensor's GPU
  handle is an opaque `void *gpu_buf`. No API-specific types leak into `ann.c`.
- `ann.c` contains zero backend-specific code outside the `#ifdef` blocks in `ann_gpu_init()`.
  Adding a backend means: a new implementation file that exports a `GpuBackend` instance, one
  `#ifdef` block in `ann_gpu_init()`, and a `-DUSE_xxx` CMake option (mirroring the CUDA block).
- Dispatch granularity is whole-batch (`eval_batch` / `train_batch`), so any compute API fits.

Concrete paths:

- **AMD (ROCm/HIP):** HIP is source-compatible with CUDA. `tensor_cuda.cu` can be `hipify`-d to
  `tensor_hip`, with `cublas` → `hipblas`/`rocblas`; the kernels carry over unchanged.
- **Intel (oneAPI):** a `tensor_sycl` backend using SYCL / Level Zero plus **oneMKL** for GEMM.
- **Cross-vendor:** a single **Vulkan compute** or **OpenCL** backend would cover NVIDIA, AMD,
  and Intel at once.

Minor (non-blocking) gaps for future work: `ann_gpu_init()` selects a backend at compile time on
a first-success-wins basis (no runtime selection among multiple compiled-in backends), and the
float-only limitation above applies to every backend.
