# GPU Inference with Apple Metal

libann supports GPU-accelerated inference on macOS using [Apple Metal](https://developer.apple.com/metal/) and Metal Performance Shaders (MPS). This enables forward-pass computation on the GPU, which is most beneficial for **batch inference** over many samples.

> **Note**: Metal GPU support includes both **inference and training**. Forward pass, backward pass, and optimizer updates all run on the GPU when weights are uploaded.
>
> All layer kinds run on the GPU: dense, Conv2D, MaxPool2D, Flatten, Upsample2D and Reshape. A convolution is an im2col step plus one MPS GEMM for the whole batch; max pooling records each window's argmax for the backward pass. GPU results match the CPU within float precision. Conv networks gain the most with wider layers (32+ filters); for very small CNNs the CPU is about as fast.

## Requirements

- macOS 10.13 (High Sierra) or later
- A Metal-capable GPU (all Apple Silicon Macs and most Intel Macs since 2012)
- Xcode Command Line Tools (for `metal` shader compilation)

## Building with Metal Support

```bash
mkdir build && cd build
cmake -DUSE_METAL=1 ..
cmake --build .
```

The Metal kernels are compiled from source at runtime by `ann_gpu_init()` (the source is embedded in `tensor_metal.m`), so no separate shader build step is needed.

To verify Metal is active:
```bash
cmake -DUSE_METAL=1 .. | grep Metal
# Should print: Apple Metal GPU inference build requested...
```

## API Overview

### 1. Initialize the GPU

Call once at startup, before any GPU operations:

```c
if (!ann_gpu_init()) {
    fprintf(stderr, "Metal GPU not available, using CPU\n");
}
```

### 2. Load and Upload a Network

Train or load a network, then upload weights to the GPU:

```c
// Load a trained network
PNetwork pnet = ann_load_network("my_model.nna");

// Upload weights to GPU (done once per model)
int err = ann_gpu_upload_network(pnet);
if (err != ERR_OK) {
    fprintf(stderr, "GPU upload failed, using CPU fallback\n");
}
```

After uploading, `ann_predict()` and `ann_predict_batch()` automatically use the GPU.

### 3. Run Inference

**Single-sample inference** (GPU when uploaded, CPU fallback otherwise):
```c
real input[784]  = { /* pixel values */ };
real output[10]  = {0};

ann_predict(pnet, input, output);
int predicted_class = ann_class_prediction(output, 10);
```

**Batch inference** (recommended for GPU — amortizes transfer overhead):
```c
// inputs: [batch_size × input_nodes] flat row-major array
// outputs: [batch_size × output_nodes] flat row-major array
int batch_size = 256;
real *inputs  = malloc(batch_size * 784 * sizeof(real));
real *outputs = malloc(batch_size * 10  * sizeof(real));

// ... fill inputs ...

int err = ann_predict_batch(pnet, inputs, outputs, batch_size);
if (err == ERR_OK) {
    for (int i = 0; i < batch_size; i++) {
        int cls = ann_class_prediction(outputs + i * 10, 10);
        printf("Sample %d: class %d\n", i, cls);
    }
}

free(inputs);
free(outputs);
```

### 4. Free GPU Resources

```c
ann_gpu_free_network(pnet);  // Release GPU buffers (weights stay on CPU)
ann_free_network(pnet);      // Free entire network
```

`ann_gpu_free_network()` only frees GPU-side buffers. CPU-side weights are preserved, so you can re-upload later or continue using the CPU path.

## Performance Notes

| Scenario | Recommendation |
|----------|---------------|
| Single-sample inference | GPU may be **slower** than CPU due to transfer overhead for small networks |
| Batch inference (≥32 samples) | GPU typically **faster** — transfer cost amortized |
| MNIST-scale networks (784→128→10) | GPU benefit is modest; larger networks benefit more |
| Large networks or big batches | GPU provides the most speedup |

### Rule of thumb
Use `ann_predict_batch()` with large batch sizes (64–1024+) to get meaningful GPU speedup. For latency-sensitive single-sample inference on small networks, the CPU path may be preferable.

## GPU-Accelerated Training

Once weights are uploaded with `ann_gpu_upload_network()`, **training automatically uses the GPU**. The CPU does not need to perform forward pass, backward pass, or optimizer updates — all computation happens on the GPU.

### Training Workflow

1. **Create and upload** the network:
   ```c
   PNetwork pnet = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
   ann_add_layer(pnet, 784, LAYER_INPUT, ACTIVATION_NULL);
   ann_add_layer(pnet, 128, LAYER_HIDDEN, ACTIVATION_SIGMOID);
   ann_add_layer(pnet, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
   
   // Upload weights to GPU (must be called before training)
   ann_gpu_upload_network(pnet);
   ```

2. **Train with GPU acceleration** (automatic dispatch):
   ```c
   PTensor inputs = tensor_create_from_array(rows, input_nodes, raw_data);
   PTensor outputs = tensor_create_from_array(rows, output_nodes, labels);
   
   ann_train_network(pnet, inputs, outputs, rows);
   // All batches train on GPU automatically
   ```

3. **Weights come back to the CPU automatically** when `ann_train_network()`
   returns, so you can save the network right away. `ann_gpu_sync_weights()`
   is still available and harmless to call.

### GPU Training Performance

| Scenario | Typical Speedup |
|----------|-----------------|
| MNIST-scale network (small batch) | 2–5× |
| Large network + large batch (256+) | 5–20× |
| Deep networks with many parameters | 10–50× |

GPU training is especially beneficial for:
- Networks with many layers or large hidden layers
- Large batch sizes (≥128)
- Training for many epochs on moderate datasets

For very small networks or tiny datasets, CPU training may be comparable due to GPU initialization overhead.

### Supported Optimizers (GPU)

All optimizers work with GPU training:
- `OPT_SGD` — Stochastic Gradient Descent
- `OPT_MOMENTUM` — SGD with momentum
- `OPT_ADAM` — Adaptive Moment Estimation (recommended)
- `OPT_ADAGRAD` — AdaGrad
- `OPT_RMSPROP` — RMSProp

### Important Notes

- **GPU training is automatic**: Once `ann_gpu_upload_network()` has been called, `ann_train_network()` uses GPU for all batches.
- **Unified memory**: On Apple Silicon, GPU and CPU share the same memory. Gradients are computed on GPU but directly accessible by CPU for inspection.
- **Fallback**: If `ann_gpu_upload_network()` was never called, training uses the CPU path automatically.
- **Weight sync**: `ann_train_network()` copies trained weights back to the CPU when it finishes. The online API (`ann_train_step()`) trains on the CPU and `ann_train_end()` re-uploads the weights.
- **Dropout**: runs on the GPU. Masks are drawn on the CPU with the same random sequence as CPU training, so GPU and CPU training drop the same neurons.
- **Optimizer state**: momentum and Adam moments move between CPU and GPU with the weights, so online fine-tuning after GPU training (and GPU training after online training) continues where the other left off.

## Complete Example

```c
#include <stdio.h>
#include "ann.h"

int main(void)
{
    // Initialize GPU
    if (!ann_gpu_init())
        fprintf(stderr, "Warning: Metal not available, using CPU\n");

    // Load trained model
    PNetwork pnet = ann_load_network("fashion_mnist.nna");
    if (!pnet) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Upload weights to GPU
    ann_gpu_upload_network(pnet);

    // Load test data
    real *data = NULL;
    int rows, stride;
    ann_load_csv("fashion-mnist_test.csv", CSV_HAS_HEADER, &data, &rows, &stride);

    int input_nodes  = pnet->layers[0].node_count;
    int output_nodes = pnet->layers[pnet->layer_count - 1].node_count;

    real *inputs  = malloc((size_t)rows * input_nodes  * sizeof(real));
    real *outputs = malloc((size_t)rows * output_nodes * sizeof(real));

    // Fill inputs (skip label column)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < input_nodes; j++)
            inputs[i * input_nodes + j] = data[i * stride + j + 1] / 255.0f;

    // Batch predict on GPU
    ann_predict_batch(pnet, inputs, outputs, rows);

    // Evaluate accuracy
    int correct = 0;
    for (int i = 0; i < rows; i++)
    {
        int predicted = ann_class_prediction(outputs + i * output_nodes, output_nodes);
        int expected  = (int)data[i * stride];
        if (predicted == expected) correct++;
    }
    printf("Accuracy: %.2f%%\n", 100.0f * correct / rows);

    free(inputs);
    free(outputs);
    free(data);
    ann_gpu_free_network(pnet);
    ann_free_network(pnet);
    return 0;
}
```

## Supported Activations

All libann activations are supported in the GPU path:

| Activation | Metal Kernel |
|-----------|-------------|
| `ACTIVATION_SIGMOID` | `activation_sigmoid` |
| `ACTIVATION_RELU` | `activation_relu` |
| `ACTIVATION_LEAKY_RELU` | `activation_leaky_relu` |
| `ACTIVATION_TANH` | `activation_tanh` |
| `ACTIVATION_SOFTSIGN` | `activation_softsign` |
| `ACTIVATION_SOFTMAX` | `softmax_rows` (per-row, numerically stable) |
| `ACTIVATION_NULL` | (no kernel, pass-through) |

## Troubleshooting

**"[Metal] Failed to compile shader source"**  
`ann_gpu_init()` compiles the kernels embedded in `tensor_metal.m` at runtime and prints the compiler's error. This usually means the kernel source was edited and no longer compiles; `ann_gpu_init()` returns 0 and libann uses the CPU.

**GPU predict doesn't match CPU**  
Check for float precision differences — GPU uses 32-bit float throughout. Results should match within ~1e-4. If diverging significantly, ensure weights are uploaded correctly with `ann_gpu_upload_network()`.

**Metal tests fail on first run**  
Reboot if the Metal driver was just installed. Some macOS configurations require a reboot for Metal shader compilation to work.
