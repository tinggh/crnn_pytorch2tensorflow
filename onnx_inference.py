
import glob
import os
import sys
import time

import cv2
import numpy as np
import onnx
import onnxruntime

from utils import decode, resizeNormalize

onnx_path = "ocr-lstm.onnx"
session = onnxruntime.InferenceSession(onnx_path)
session.get_modelmeta()



def onnx_predict(image, dynamic=False):

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    mask = resizeNormalize(image, dynamic)
    
    x = np.zeros((1, 1, 32, 512), dtype=np.float32)
    x[0] = np.expand_dims(mask, axis=0)

    preds = session.run([output_name], {input_name: x})[0]
    preds = preds.transpose((1,0,2))
    preds = preds[:,:,:]
    out = decode(preds)
    return out

if __name__ == "__main__":
    root = "./imgs/*.*p*g"

    imgs = glob.glob(root)
    for img_path in imgs:
        orig_image = cv2.imread(img_path, 0)
        pred = onnx_predict(orig_image)
        print("image:{0}====> pred:{1}".format(img_path, pred))
