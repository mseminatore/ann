# ONNX Export & Import

Export and import trained networks using ONNX JSON format for model persistence and interoperability.

## Export

```c
ann_export_onnx(pnet, "model.onnx.json");
```

### Format Details

- Outputs JSON (not binary protobuf) - see [Converting to binary ONNX](#converting-to-binary-onnx)
- Each layer becomes: `MatMul` → `Add` → `Activation` nodes
- `weight_N` has dims `[in_nodes, out_nodes]`, the layout ONNX `MatMul` expects
  (libann stores weights as `[out, in]` internally and transposes them on export)
- Input shape: `[batch, input_size]` with dynamic batch dimension
- Supported activations map to ONNX ops: `Sigmoid`, `Relu`, `LeakyRelu`, `Tanh`, `Softsign`, `Softmax`, `Gelu`
- Opset 17, or opset 20 when the network uses GELU (`Gelu` with `approximate="tanh"`)
- `producer_name` is `ann-library`; `producer_version` is `1.1`

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

- Loads network topology from weight tensor dimensions
- Restores weights and biases from initializers
- Maps ONNX activation ops back to libann activations
- Imported networks are ready for inference (or continued training)

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

- **Only reads libann-exported JSON** - not general ONNX files
- Requires specific naming: `weight_0`, `bias_0`, `activation_0`, etc.
- Does not restore: optimizer state, learning rate, dropout rates, training history
- Imported networks default to `OPT_ADAM` and `LOSS_MSE`
- Unsupported ONNX operations will cause import to fail with error callback

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
