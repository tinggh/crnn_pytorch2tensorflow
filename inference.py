
import glob
import os
import sys
import time
import cv2
import numpy as np
import onnx
import onnxruntime
from model.keys import alphabetChinese as alphabet

characters = alphabet + u'ç'
nclass = len(characters)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != 0 and (not (i > 0 and pred_text[i] == pred_text[i - 1])):
            char_list.append(characters[pred_text[i]-1])
    return u''.join(char_list)


def onnx_predict(image):

    onnx_path = "ocr-lstm.onnx"
    session = onnxruntime.InferenceSession(onnx_path)
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    h, w, _ = orig_image.shape
    image = orig_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w = int(w * 32.0 / h)
    image = cv2.resize(image, (w, 32))
    image = image / 255.0

    mean = 0.5
    std = 0.5
    image = (image - mean) / std
    
    x = np.zeros((1, 1, 32, w), dtype=np.float32)
    x[0] = np.expand_dims(image, axis=0)

    preds = session.run([output_name], {input_name: x})[0]
    preds = preds.transpose((1,0,2))
    preds = preds[:,:,:]
    out = decode(preds)
    return out

if __name__ == "__main__":
    root = "./imgs/*.*p*g"

    imgs = glob.glob(root)
    for img_path in imgs:
        orig_image = cv2.imread(img_path)
        pred = onnx_predict(orig_image)
        print("image:{0}====> pred:{1}".format(img_path, pred))
    

