import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torch


gpu_id = 0
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(gpu_id))
else:
    device = torch.device("cpu")

print('loading onnx model')
onnx_model = onnx.load('ocr-lstm.onnx')
print('prepare tf model')
b, c, w, h = 1, 1, 32, 512
input_shape = (b, c, w, h)
tf_rep = prepare(onnx_model, strict=False)
tf_rep.export_graph('ocr-lstm.pb')

