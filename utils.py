
import cv2
import numpy as np

from model.keys import alphabetChinese as alphabet

characters = alphabet + u'ç'


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != 0 and (not (i > 0 and pred_text[i] == pred_text[i - 1])):
            char_list.append(characters[pred_text[i]-1])
    return u''.join(char_list)


def resizeNormalize(image, dynamic=False):
    h, w = image.shape   
    r = w / h
    if r<16 or dynamic:
        h = 32
        w = int(h * r)
    elif r>16:
        w = 512
        h = int(w / r)
    image = cv2.resize(image, (w,h))
    mask = np.ones((32, 512), dtype=np.uint8)*255
    mask[:h, :w] = image
    mask = mask / 255.0

    mean = 0.5
    std = 0.5
    mask = (mask - mean) / std
    return mask
