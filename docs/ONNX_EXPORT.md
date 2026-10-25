# ONNX Export & Import

Export and import trained networks using ONNX JSON format for model persistence and interoperability.

## Export

```c
ann_export_onnx(pnet, "model.onnx.json");
```

### Format Details

- Outputs JSON (not binary protobuf) - see [Converting to binary ONNX](#converting-to-binary-onnx)
- Each dense layer becomes: `MatMul` → `Add` → `Activation` nodes
- Convolutional networks export `Conv` (weights `[F, C, k, k]`, `kernel_shape`,
  `strides`, `pads`), `MaxPool` and `Flatten` nodes with a `[batch, C, H, W]`
  input; a `Flatten` is inserted before a dense layer that receives an image
- Upsampling layers export `Resize` (`mode = nearest`, `asymmetric`
  coordinates, `floor` rounding, as PyTorch writes `nn.Upsample`) with a
  `scales` initializer `[1, 1, s, s]`; reshape layers export `Reshape` with a
  shape initializer `[-1, C, H, W]`. Global average pooling layers export
  `GlobalAveragePool` (output `[batch, C, 1, 1]`, flattened before a dense
  layer). A model that starts with dense layers
  keeps a `[batch, n]` input.
- `weight_N` has dims `[in_nodes, out_nodes]`, the layout ONNX `MatMul` expects
  (libann stores weights as `[out, in]` internally and transposes them on export)
- Input shape: `[batch, input_size]` with dynamic batch dimension
- Supported activations map to ONNX ops: `Sigmoid`, `Relu`, `LeakyRelu`, `Tanh`, `Softsign`, `Softmax`, `Gelu`
- Opset 17, or opset 20 when the network uses GELU (`Gelu` with `approximate="tanh"`)
- `producer_name` is `ann-library`; `producer_version` is `1.1`
- Floats are written with 9 significant digits, so weights round-trip exactly

### Export Limitations

- `ACTIVATION_NULL` layers export without activation node
- LeakyReLU uses fixed alpha=0.01
- Softmax uses axis=-1 (last dimension)

## Import

```c
PNetwork pnet = ann_import_onnx("model.onnx.json");
if (pnet) {
    // Network loaded - ready for inference
    ann_free_network(pnet);
}
```

### Import Capabilities

The importer reads the graph as a single chain of nodes starting at its input
(`[batch, n]` or `[batch, C, H, W]`) and turns each node into a layer:

ONNX node | libann layer
--------- | ------------
`MatMul` (+ `Add` of a bias) | dense layer (weights `[in, out]`)
`Gemm` | dense layer (`transB = 1` as written by PyTorch's `Linear`, or `transB = 0`; `alpha = beta = 1`)
`Conv` | conv layer (square kernel, equal strides and pads, no dilation or groups; bias optional)
`MaxPool` | max pooling layer (square window, equal strides, no padding, `ceil_mode = 0`)
`GlobalAveragePool` | global average pooling layer (as PyTorch writes `nn.AdaptiveAvgPool2d(1)`)
`Flatten` (axis 1) | flatten layer (libann's own `flatten_input_N` nodes are skipped)
`Resize`, `Upsample` | upsampling layer (`nearest` mode, one integer scale on H and W from `scales` or output `sizes`; `asymmetric` coordinates with `floor` rounding, or `half_pixel` with rounding)
`Reshape` | reshape layer to a constant `[batch, C, H, W]` shape (`0` copies a dimension, one `-1` is inferred); a `[batch, n]` shape is a flatten
`Relu`, `LeakyRelu` (alpha 0.01), `Sigmoid`, `Tanh`, `Softsign`, `Softmax`, `Gelu` (`approximate="tanh"`) | the activation of the layer before it

- Reads libann's JSON and the JSON written by protobuf's printer
  (`json_format.MessageToJson`): lowerCamelCase or snake_case field names,
  int64 values as strings, and weights in `float_data` or `raw_data`
  (base64 little-endian float32)
- Imported networks are ready for inference (or continued training); a libann
  export imports with the same layers and identical predictions

### Files from older libann versions

Exports with `producer_version` `1.0` (libann before this fix) wrote the weight
data without transposing it, so other ONNX runtimes computed wrong results
from them. `ann_import_onnx()` recognizes these files and still loads them
correctly; re-export them to get a file other runtimes can use:

```c
PNetwork net = ann_import_onnx("old_model.onnx.json");  // 1.0 file
ann_export_onnx(net, "model.onnx.json");                // 1.1 file
```

### Import Limitations

- Reads JSON, not binary `.onnx` files: convert with `json_format.MessageToJson(onnx.load("model.onnx"))`
- Anything libann can't reproduce exactly is rejected with `NULL` and an error
  callback message: branching graphs, other ops (e.g. `AveragePool`,
  `ConvTranspose`, `BatchNormalization`), dilated or grouped convolutions,
  padded or `ceil_mode` pooling, linear or fractional `Resize`, a `Reshape`
  whose shape is computed in the graph (`Shape` → `Gather` → `Concat`), the
  exact (erf) `Gelu`, or an activation after a pool/flatten/upsample/reshape
  layer
- Does not restore: optimizer state, learning rate, dropout rates, training history
- Imported networks default to `OPT_ADAM` and `LOSS_MSE`

### Round-Trip Example

```c
// Train and export
PNetwork net1 = ann_make_network(OPT_ADAM, LOSS_CATEGORICAL_CROSS_ENTROPY);
ann_add_layer(net1, 784, LAYER_INPUT, ACTIVATION_NULL);
ann_add_layer(net1, 128, LAYER_HIDDEN, ACTIVATION_RELU);
ann_add_layer(net1, 10, LAYER_OUTPUT, ACTIVATION_SOFTMAX);
ann_train_network(net1, x_train, y_train, num_samples);
ann_export_onnx(net1, "model.onnx.json");

// Later: import and use
PNetwork net2 = ann_import_onnx("model.onnx.json");
real accuracy = ann_evaluate_accuracy(net2, x_test, y_test);
```

## Converting to binary ONNX

The JSON is the protobuf JSON form of an ONNX `ModelProto`, so the `onnx`
Python package can convert it and run it, for example with onnxruntime:

```python
import onnx, onnxruntime as ort
from google.protobuf import json_format

model = json_format.Parse(open("model.onnx.json").read(), onnx.ModelProto())
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {session.get_inputs()[0].name: x})  # x: float32 [batch, input_size]
```
