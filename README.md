# CRNN_PyTorch2TensorFlow

First, convert the pth model to onnx model and then use the onnx/onnx-tensorflow converter tool as a Tensorflow backend for ONNX or convert through the python API.

## onnx-tensorflow converter tool
```bash
Install onnx-tensorflow: pip install onnx-tf

Convert using the command line tool: onnx-tf convert -t tf -i /path/to/input.onnx -o /path/to/output.pb
```

## python API.
```bash
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("input_path")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("output_path")
```bash