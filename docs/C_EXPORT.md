# int8 Quantization & C Header Export

Run a trained network on a microcontroller or any C target: quantize it to int8, check the
accuracy in libann, and export it as a C header that needs only `ann_rt.h`, a small header-only
runtime. The int8 path is integer-only, so it runs on cores without an FPU. It needs no libc
beyond `<stddef.h>` and `<stdint.h>`, never calls `malloc`, and never recurses. A float32 export
is also available for targets with an FPU.

Supported layers: dense, conv2d (any kernel, stride, padding), maxpool2d, global average and
max pooling, flatten, upsample2d and reshape, with every activation (softmax only on the output
layer). Global average pooling keeps its input's int8 quantization and rounds each mean half
away from the zero point. Batch norm layers are folded into the dense or conv layer before them
(`W' = s * W`, `b' = s * (b - mean) + beta` with `s = gamma / sqrt(var + 1e-5)`, the batch
norm's activation becoming that layer's), so a header has no batch norm layers; a batch norm
anywhere else (on the input, after pooling, or after a layer with an activation) makes
`ann_quantize()` and `ann_export_c_header()` fail with `ERR_INVALID`.

Conv layers may be grouped or depthwise (`ann_add_conv2d_grouped`, `ann_add_depthwise_conv2d`), and
conv and pool windows non-square, including 1-D convs and pools (`ann_add_conv2d_ex`,
`ann_add_conv1d`, `ann_add_maxpool2d_ex`, `ann_add_maxpool1d`).

A generated header initializes every `ann_rt_layer` field, so it builds without
missing-initializer warnings, and checks that `ann_rt.h` is at least the version that wrote it
(6: version 2 added upsampling, 3 global average pooling, 4 grouped convolutions, 5 non-square
windows, 6 global max pooling). Networks with an embedding are rejected (`ERR_INVALID`): the int8
path quantizes its inputs, and token ids do not fit in 8 bits. Copy `ann_rt.h` from the libann that exported the model. Headers written by older
versions still build with the newer runtime (the fields they leave out read as 0, which means
one group and a square window), but
compilers warn about the missing initializers with `-Wextra`. Headers exported by libann 0.7
(pyann 0.7.0) require exactly version 1; re-export them to use the newer runtime.

## Quick start

```c
#include "ann.h"

// pnet is a trained network; calibration holds a few hundred training inputs
PQuantizedNetwork q = ann_quantize(pnet, calibration);

printf("float %.2f%%  int8 %.2f%%\n",
       100 * ann_evaluate_accuracy(pnet, x_test, y_test),
       100 * ann_quantized_evaluate_accuracy(q, x_test, y_test));

ann_export_c_header(NULL, q, "model_q.h", "model");    // int8
ann_export_c_header(pnet, NULL, "model_f.h", "modelf"); // float32
ann_free_quantized(q);
```

On the device, copy `ann_rt.h` next to the generated header:

```c
#include "model_q.h"

static int8_t workspace[MODEL_WORKSPACE_SIZE];

int classify(const int8_t *image)            // MODEL_INPUT_SIZE int8 values
{
    int8_t scores[MODEL_OUTPUT_SIZE];
    model_predict(image, scores, workspace);
    return model_argmax(scores);
}
```

`examples/export_model.c` does the whole flow on Fashion-MNIST:

```bash
./export_model -e 5 fashion-mnist_train.csv fashion-mnist_test.csv      # MLP
./export_model -c -e 5 fashion-mnist_train.csv fashion-mnist_test.csv   # CNN
./export_model -m mnist-fashion.nna fashion-mnist_train.csv fashion-mnist_test.csv
```

`examples/run_exported.c` is the other half: a program built without libann, from just the
generated `fashion_q.h` and `fashion_f.h` plus `ann_rt.h`. It runs the test set through both
headers and prints their accuracy, which should match what `export_model` printed, and the time
per image. It is not in the default build, because the headers exist only after `export_model`
runs:

```bash
cmake -DEXPORT_HEADER_DIR=<dir with the headers> .. && cmake --build . --target run_exported
./run_exported fashion-mnist_test.csv

make run_exported      # Makefile: expects the headers in the project root
```

## API

| Function | Purpose |
|----------|---------|
| `ann_quantize(pnet, calibration)` | Build an int8 model; calibration is a `[samples × inputs]` tensor |
| `ann_free_quantized(q)` | Free it |
| `ann_quantized_predict(q, in, out)` | Real in/out; int8 inside (same results as `<prefix>_predict_float`) |
| `ann_quantized_predict_int8(q, in, out)` | int8 in/out (same results as `<prefix>_predict`) |
| `ann_quantized_get_io_params(q, ...)` | Input/output scale and zero point |
| `ann_quantized_evaluate_accuracy(q, x, y)` | Classification accuracy of the int8 model |
| `ann_export_c_header(pnet, q, file, prefix)` | Write the int8 model (`q`), or the float32 network when `q` is `NULL` |

libann runs the int8 model with the same `ann_rt.h` code the header uses. The
`test_c_export` test compiles exported headers without libann and checks that their results
match libann's bit for bit.

## Generated header

For prefix `model`, an int8 header defines:

| Symbol | Meaning |
|--------|---------|
| `MODEL_INPUT_SIZE`, `MODEL_OUTPUT_SIZE` | Values per sample |
| `MODEL_WORKSPACE_SIZE` | int8 values the caller supplies as scratch space |
| `MODEL_INPUT_SCALE`, `MODEL_INPUT_ZERO_POINT` | Input quantization: `real = scale * (q - zero_point)` |
| `MODEL_OUTPUT_SCALE`, `MODEL_OUTPUT_ZERO_POINT` | Output quantization |
| `model_predict(const int8_t *in, int8_t *out, int8_t *ws)` | int8 inference |
| `model_predict_float(const float *in, float *out, int8_t *ws)` | Quantizes the input, runs int8, dequantizes the output and applies softmax if the model has it; not available with `ANN_RT_NO_FLOAT` |
| `model_argmax(const int8_t *out)` | Index of the largest output |

A float32 header defines the sizes and `model_predict(const float *in, float *out, float *ws)` /
`model_argmax(const float *out)`, with a `float` workspace.

All parameters are `static const` arrays, so they go to flash on most MCUs. The header's top
comment gives the parameter memory, int8 against float32. The workspace holds two copies of the
largest layer output, plus the input and output for the int8 model. Functions return 0, or -1 for
a bad model.

Define `ANN_RT_NO_FLOAT` before including the headers to compile out every function that uses
float math. The int8 path then needs only `<stddef.h>` and `<stdint.h>`, and it builds
freestanding (`-ffreestanding -nostdinc`). It has been cross-compiled for Cortex-M0 and
Cortex-M4 but not yet run on hardware. On cores without a 64-bit multiply, the compiler's
runtime library supplies the 64-bit helpers (`__aeabi_lmul` and so on).

### Feeding int8 inputs

Quantize sensor or pixel data with the input parameters:

```c
q = round(x / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT, clamped to [-128, 127]
```

With integer data the division often becomes a shift or a lookup table. For example, 0–255 pixels
scaled to [0, 1] during training give a scale of 1/255 and a zero point of -128, so
`q = pixel - 128`.

## Quantization scheme

TFLite-style post-training quantization:

- **Weights**: symmetric int8, one scale per output channel (dense row or conv filter):
  `scale = max|w| / 127`.
- **Biases**: int32 at scale `input_scale × weight_scale`.
- **Activations**: asymmetric int8, one scale and zero point per tensor. The ranges come from
  running the float network over the calibration inputs, widened to include 0 so that ReLU and
  zero padding are exact.
- **Dense / conv**: `acc = bias + Σ (x - x_zero_point) × w` in int32. The sum is rescaled with a
  per-channel fixed-point multiplier: a Q31 value and a right shift, applied with a 64-bit product
  and rounding.
- **Activations**: ReLU and no activation are applied as part of the clamp. Leaky ReLU, sigmoid,
  tanh, softsign and GELU rescale to an int8 pre-activation, then use a 256-entry lookup table
  built at quantization time.
- **Max pooling, upsampling, flatten and reshape** work on the int8 values directly, keeping
  their input's scale and zero point.
- **Softmax** outputs stay as int8 logits, and `argmax` of the logits is the predicted class.
  The float helpers apply softmax after dequantizing.

### Calibration tips

- Use a few hundred to a few thousand samples that look like real inputs, such as part of the
  training set.
- The calibration data must be preprocessed exactly like the inputs at inference.
- Outliers widen a range and cost resolution for everything else. If int8 accuracy drops,
  remove extreme samples from the calibration set.

## Accuracy

Fashion-MNIST test accuracy (10,000 images), calibrated on 1000 training images
(`examples/export_model.c`):

| Model | float32 | int8 | Parameters float32 → int8 |
|-------|---------|------|---------------------------|
| MLP 784-128-10 (5 epochs) | 88.51% | 88.57% | 407 KB → 103 KB |
| CNN from `fashion_cnn.c` (5 epochs) | 89.20% | 89.33% | 110 KB → 28 KB |
| `data/mnist-fashion.nna` | 86.98% | 86.82% | 407 KB → 103 KB |

int8 accuracy is within a fraction of a point of float32, in either direction. int8 parameter
memory includes the per-channel bias, multiplier and shift (9 bytes per output channel) and any
lookup tables.

## Limitations

- Post-training quantization only (no quantization-aware training).
- Per-tensor activation ranges are plain min/max over the calibration set.
- The runtime's kernels are plain C loops with no SIMD. The conv kernel works directly on the
  input, without im2col, to keep memory small.
- Right shifts of negative values are assumed to be arithmetic, as they are on every mainstream
  compiler.
