
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
    print(pred)
    pred_text = pred.argmax(axis=1)
    print(pred_text)
    for i in range(len(pred_text)):
        if pred_text[i] != 0 and (not (i > 0 and pred_text[i] == pred_text[i - 1])):
            char_list.append(characters[pred_text[i]-1])
    return u''.join(char_list)

onnx_path = "ocr-lstm.onnx"
session = onnxruntime.InferenceSession(onnx_path)
session.get_modelmeta()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


root = "./imgs/*.*p*g"

imgs = glob.glob(root)
for img_path in imgs:
    print(img_path)
    orig_image = cv2.imread(img_path)
    h, w, _ = orig_image.shape
    
    image = orig_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # if h > w :
    #     image = np.concatenate((image , np.zeros([h,h-w,3])),axis=1)
    # elif w > h :
    #     image = np.concatenate((image, np.zeros([w - h,  w,3])), axis=0)
    
    print(image.shape)

    w = int(w * 32.0 / h)
    image = cv2.resize(image, (w, 32))
    image = image / 255.0

    
    mean = 0.5
    std = 0.5
    image = (image - mean) / std
    image = np.reshape(image, (1,1,32, -1))
    

    image = image.astype(np.float32)
    
    time_time = time.time()
    preds = session.run([output_name], {input_name: image})[0]
    T, h, c = preds.shape
    preds = np.reshape(preds, (T, -1))[:,:]

    out = decode(preds)
    print(out)

