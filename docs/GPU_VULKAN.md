# Vulkan Backend

The Vulkan backend runs GPU **inference and training** on any GPU with a Vulkan 1.1 driver
(AMD, Intel and NVIDIA) on Windows and Linux, through the same `GpuBackend` vtable as the Metal
and CUDA backends. Like Metal, it runs every layer kind: dense, conv, max pool, upsample, flatten
and reshape. It consists of:

- `tensor_vulkan.c`: device setup, buffers, command recording, and the inference and training
  paths
- `shaders/vk/*.comp`: GLSL compute shaders (GEMM, bias + activation, softmax, activation
  derivatives, output delta and loss, bias gradients, gradient clipping, the five optimizers,
  L1/L2 regularization, dropout; im2col, conv output and gradient rows, col2im, max pooling,
  upsampling, split-K partial sums), with shared code in `shaders/vk/common.glsl`
- `cmake/embed_spirv.cmake`: embeds the compiled SPIR-V in a generated `shaders_vk_spv.h`
- `CMakeLists.txt` option `-DUSE_VULKAN=1`

## Build

Building needs the Vulkan headers and `glslc`, which compiles the shaders to SPIR-V:

- **Windows:** install the [Vulkan SDK](https://vulkan.lunarg.com/) (it sets `VULKAN_SDK`).
- **Linux:** `sudo apt-get install libvulkan-dev glslc` (or your distribution's equivalent), or
  the Vulkan SDK.

```bash
git submodule update --init --recursive   # required: pulls the `testy` test framework
mkdir build_vulkan && cd build_vulkan
cmake -DUSE_VULKAN=1 ..
cmake --build . --config Release
```

Running needs only a Vulkan driver, which current AMD, Intel and NVIDIA drivers include (on
Linux, Mesa's RADV and ANV drivers, or the vendor's). libann opens the Vulkan loader
(`vulkan-1.dll`, `libvulkan.so.1`) at run time rather than linking it, so a `USE_VULKAN` build
also runs on machines without Vulkan: `ann_gpu_init()` returns 0 and everything runs on the CPU.

## What runs on the GPU

The same as the Metal backend:

- **Inference:** `ann_predict()` and `ann_predict_batch()` once the network is uploaded with
  `ann_gpu_upload_network()`. Large batches are evaluated in pieces of up to 256 MB of
  activations.
- **Training:** `ann_train_network()` runs each mini-batch on the GPU: forward pass, output
  gradient and loss, backward pass, then gradient clipping, the optimizer update (SGD,
  Momentum, AdaGrad, RMSProp, Adam) and L1/L2 regularization. Weights and optimizer state come
  back to the CPU when training finishes. Dropout masks are drawn on the host, with the same
  random sequence as CPU training, and copied to the GPU each batch.
- **Online training** (`ann_train_begin()` / `ann_train_step()` / `ann_train_end()`) runs on the
  CPU and re-uploads the weights at `ann_train_end()`.
- **Every layer kind.** Conv layers run as im2col + one GEMM for the whole batch; max pooling
  records each window's argmax for the backward pass; upsampling is nearest-neighbour; flatten
  and reshape are copies. Backward passes are gathers, so no kernel needs atomics.

GPU and CPU training agree to float rounding (`tests/test_vulkan.c` compares them for every
optimizer, loss and activation, and for CNNs and conv autoencoders).

## Choosing the device and backend

| Environment variable | Effect |
|----------------------|--------|
| `ANN_GPU=vulkan` | Use only the Vulkan backend, in a build that also has CUDA. `ANN_GPU=none` keeps everything on the CPU. |
| `ANN_VK_DEVICE=<n>` | Use the n-th Vulkan device. By default, a discrete GPU is preferred over an integrated one, and either over a CPU implementation such as lavapipe. `vulkaninfo --summary` lists the devices in order. |
| `ANN_VK_DEBUG=1` | Enable the Khronos validation layer (installed with the Vulkan SDK), with synchronization validation, and print its warnings and errors. Much slower; for debugging. |
| `ANN_VK_STAGING=1` / `0` | Force copies through staging memory (`1`), or mapped device memory (`0`); see below. For testing. |
| `ANN_VK_PROFILE=1` | Time every GPU command with timestamp queries and print the GPU time by kernel at exit (count, total, average, share). For finding what to optimize; adds a little overhead. |
| `ANN_VK_CONV_DIRECT=<n>` | Convolve layers with at most `n` filters (up to 16) directly instead of through im2col and GEMM; default 8. For tuning on other GPUs: compare `ANN_VK_PROFILE` totals. |

`ann_gpu_init()` prints the device it chose and how data reaches it, e.g.
`[Vulkan] Device 0: AMD Radeon HD 7700 Series (copies through staging memory)`.

## Implementation notes

- **Memory.** Every tensor's `gpu_buf` is a buffer in device-local memory. When the GPU's
  device-local memory is also host-visible and about as large as its VRAM (integrated GPUs, and
  discrete GPUs with resizable BAR), buffers are mapped and the host copies straight into them,
  as with Metal. Otherwise (discrete GPUs whose host-visible VRAM is a 256 MB window) copies go
  through a host-visible staging buffer, as with CUDA. A buffer that no longer fits in mapped
  memory goes in plain device memory and uses the staging buffer, so the two can mix. CTest
  runs `test_vulkan` both ways (`test_vulkan_staging`, `test_vulkan_mapped`).
- **GEMM.** There is no vendor BLAS for Vulkan; `gemm.comp` is a tiled SGEMM (64x64 tiles, a 4x4
  block per invocation) with specialization constants for the three layouts training uses:
  `Y = X·Wᵀ`, `dW += dZᵀ·A` and `dX = dZ·W`. Matrices are row-major, as libann's tensors are.
- **Conv backward.** A conv layer's weight gradient sums over every output position of the batch
  (`batch·OH·OW` rows) into only `F x C·K·K` outputs, too few tiles to keep a GPU busy, so its
  GEMM is split over K: about 128 workgroups each sum a slice into their own block, and
  `sum_partials.comp` adds the blocks up (a workgroup per output). Bias gradients use one
  workgroup per column with a shared-memory reduction. Without these, CNN training on the GPU
  was about 10x slower than on the CPU.
- **Direct convolution.** Layers with at most 8 filters (`ANN_VK_CONV_DIRECT`) skip im2col: their
  GEMMs would fill a sliver of the 64-wide tiles. The forward pass sums each output's window
  directly; the weight gradient (`conv_direct_weight_grad.comp`) gives each invocation one
  weight column and every filter's sum, with the bias as one more column of ones, so it also
  replaces the bias reduction; the input gradient gathers `dZ·W` over the windows. With more
  filters the direct input gradient reads too much, and im2col + GEMM wins. Max pooling whose
  windows tile the input (size == stride) writes each window from its output in the backward
  pass rather than gathering. On the Radeon HD 7700 these cut fashion_cnn's GPU time per epoch
  from 4.0 s to 3.0 s at batch 32 and from 2.8 s to 1.8 s at batch 256.
- **Work buffers.** Each conv layer keeps its im2col columns, GEMM rows, column gradients and
  partial sums, and each max pool layer its argmax, between a batch's forward and backward
  passes. They grow to the largest batch seen and are freed with `ann_gpu_free_network()`.
- **Commands.** Each operation (a training step, a forward pass, an upload or download) is
  recorded into one command buffer with a full memory barrier between dependent commands.
  Descriptor sets come from pools reset before each command buffer; push descriptors would be
  simpler, but not every driver has them. Kernels loop over their items, so the workgroup count
  stays within `maxComputeWorkGroupCount`.
- **Pipelined training.** As with Metal, up to `TRAIN_SLOTS` (2) training steps are in flight:
  `train_batch` records a step and submits it without waiting, so the CPU assembles the next
  batch while the GPU runs this one, and `train_finish` collects the losses. Each slot has its
  own command buffer, fence, descriptor pools, staging buffer, and copies of what the CPU
  writes for a step (inputs, targets, dropout masks) and reads back (per-row losses). Weights,
  gradients, activations and layer work buffers are shared; steps run in submission order, each
  command buffer starting with a barrier on the work before it. Anything that reads or replaces
  that shared state (inference, weight sync, upload, free, a buffer that has to grow) first waits
  for the steps in flight.
- **float only**, like the other GPU backends; a `double` build does not compile with
  `USE_VULKAN`.

## Test

```bash
cd build_vulkan
ctest -R test_vulkan --output-on-failure      # test_vulkan, test_vulkan_staging, test_vulkan_mapped
```

`tests/test_vulkan.c` checks GPU inference and training against the CPU: odd matrix shapes that
straddle the GEMM tiles, every optimizer, categorical and binary cross-entropy, MSE through
softmax and linear outputs, L1/L2 with gradient clipping, CNNs (the CNN suite shared with
`tests/test_metal.c`: strided and padded convs, overlapping pools, conv dropout, every
activation, input noise, conv autoencoders, reshape and upsample, a batch evaluated in pieces),
and the shared consistency suite in `tests/gpu_consistency.h` (dropout, Adam state, GELU, early
stopping, online training).

CI runs the Vulkan build and tests on Linux with **lavapipe**, Mesa's Vulkan implementation on
the CPU, so the backend is checked on every push without a GPU.

`examples/bench_gpu_training.c` compares CPU and GPU training throughput for dense networks;
`examples/fashion_cnn.c -g` trains a CNN on the GPU.
