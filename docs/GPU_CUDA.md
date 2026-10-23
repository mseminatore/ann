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
git submodule update --init --recursive   # required: pulls the `testy` test framework
mkdir build_cuda && cd build_cuda
cmake -DUSE_CUDA=1 ..
cmake --build . --config Release
```

Requires the CUDA Toolkit (`cudart`, `cublas`) and an NVIDIA GPU.

### GPU architecture selection

The build sets `CMAKE_CUDA_ARCHITECTURES` to `75;86` (Turing + Ampere) by default, which compiles
under both CUDA 12.x and 13.x toolkits. Override it for your GPU:

```bash
cmake -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=61 ..   # Pascal (e.g. Quadro P620, GTX 10xx)
cmake -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=89 ..   # Ada (e.g. RTX 40xx)
```

> **CUDA 13.x dropped offline codegen for Maxwell/Pascal (sm_5x / sm_6x).** To target an older
> GPU you must build with a **CUDA 12.x** toolkit *and* set the matching `CMAKE_CUDA_ARCHITECTURES`.
> On Windows with the Visual Studio generator, select the toolkit with
> `-T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"`.

> **Note on host compiler flags:** CPU SIMD flags (`/arch:AVX`, `-mavx2 -mfma`, `-mcpu=native`) are
> scoped to C/C++ sources via a generator expression so they are never passed to `nvcc` (which
> rejects them with `nvcc fatal : A single input file is required ...`).

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
  regularization). When training finishes the weights and the optimizer state (momentum /
  Adam moments) are copied back to the CPU automatically, so `ann_save_network()` and later
  online training see the trained state without an explicit `ann_gpu_sync_weights()`. The next
  `ann_train_network()` copies the CPU's optimizer state back to the GPU, so training resumes
  where it left off on either side.
- **Dropout** runs on the GPU. The masks are drawn on the host with the same random sequence as
  CPU training and copied to the device each batch; `cuda_dropout_forward` applies them after
  the activation and `cuda_dropout_backward` masks the gradient and restores the pre-dropout
  activations before the activation derivative.
- **Online training** (`ann_train_begin()` / `ann_train_step()` / `ann_train_end()`) runs on the
  CPU. During the session `ann_predict()` uses the CPU weights; `ann_train_end()` re-uploads them
  so GPU inference resumes with the updated model.
- If a GPU batch fails mid-training, training continues on the CPU from the GPU's latest weights.

## Implementation notes

- **Row-major ↔ column-major.** libann tensors are row-major; cuBLAS is column-major. All GEMMs
  go through a single `cuda_gemm_rowmajor()` helper that computes the row-major result by issuing
  the equivalent column-major (transposed) `cublasSgemm` — operands swapped, `m`/`n` swapped.
- **Device memory is not host-visible.** Unlike Metal's unified (shared) memory, CUDA
  `cudaMalloc` buffers cannot be read directly by the CPU. The forward/backward/optimizer steps
  stay entirely on the device, including `delta = T − Y` and the per-element loss
  (`cuda_mse_delta_loss` / `cuda_xent_delta_loss`), which `cublasSasum` reduces. Per batch, only
  the inputs and targets go host→device and the scalar loss comes back.
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
- `ann.c` contains no backend-specific code outside the backend list in `ann_gpu_init()`.
  Adding a backend means: a new implementation file that exports a `GpuBackend` instance, an
  entry in that list, its `USE_xxx` define in the `USE_GPU` condition in `ann_config.h`, and a
  `-DUSE_xxx` CMake option (mirroring the CUDA block).
- Dispatch granularity is whole-batch (`eval_batch` / `train_batch`), so any compute API fits.

AMD and Intel GPUs are covered by the **Vulkan** backend (`-DUSE_VULKAN=1`, see
[GPU_VULKAN.md](GPU_VULKAN.md)), which also runs on NVIDIA. A build can include CUDA and Vulkan
together: `ann_gpu_init()` tries Metal, Vulkan, then CUDA (Vulkan runs every layer kind), and
the `ANN_GPU` environment variable picks one: `ANN_GPU=cuda` uses CUDA on such a build. The float-only limitation above applies to every backend.
