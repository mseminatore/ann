# ONNX Export

Export trained networks to ONNX JSON format for interoperability:
```c
ann_export_onnx(pnet, "model.onnx.json");
```

## Format Details

- Outputs JSON (not binary protobuf) - convert externally if needed
- Each layer becomes: `MatMul` → `Add` → `Activation` nodes
- Input shape: `[batch, input_size]` with dynamic batch dimension
- Supported activations map to ONNX ops: `Sigmoid`, `Relu`, `LeakyRelu`, `Tanh`, `Softsign`, `Softmax`

## Limitations

- `ACTIVATION_NULL` layers export without activation node
- LeakyReLU uses fixed alpha=0.01
- Softmax uses axis=-1 (last dimension)
