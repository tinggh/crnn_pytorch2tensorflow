import os
import sys

import torch
from torchsummary import summary
from model.keys import alphabetChinese as alphabet
from model.crnn import CRNN

nh = 256
net = CRNN(32, 1, len(alphabet) + 1, nh, n_rnn=2, leakyRelu=False, lstmFlag=True)
gpu_id = 0
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(gpu_id))
else:
    device = torch.device("cpu")
weights = torch.load('ocr-lstm.pth', map_location=device)
state_dict = {}
for k, v in weights.items():
    state_dict[k.replace("module.", "")] = v
net.load_state_dict(state_dict)
net = net.to(device)
net.eval()

print('Finished loading model!')
summary(net, (1,32,582))
##################export###############
output_onnx = 'ocr-lstm.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input"]
output_names = ["out"]
b, c, w, h = 1, 1, 32, 582
input_shape = (b, c, w, h)
inputs = torch.randn(*input_shape).to(device)

input_names=['input']
output_names=['output']
dynamic_axes= {'input':{0:'batch_size', 3:'width'}, 'output':{0:'width', 2:'batch_size'}} #adding names for better debugging

torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)


os.system("python -m onnxsim {0} {0}".format(output_onnx))
