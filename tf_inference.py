import glob
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.platform import gfile

from utils import decode, resizeNormalize


model_path = "ocr-lstm.pb"

def load_graph(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

graph_def = load_graph(model_path)


def revise_placeholder(width):
    dynamic_graph = tf.GraphDef()
    for n in graph_def.node:
        if n.name == 'input':
            nn = dynamic_graph.node.add()
            nn.op = 'Placeholder'
            nn.name = n.name
            nn.attr['dtype'].type = n.attr['dtype'].type
            s = tensor_shape_pb2.TensorShapeProto()
            b = tensor_shape_pb2.TensorShapeProto.Dim()
            c = tensor_shape_pb2.TensorShapeProto.Dim()
            h = tensor_shape_pb2.TensorShapeProto.Dim()
            w = tensor_shape_pb2.TensorShapeProto.Dim()
            b.size, c.size, h.size, w.size = 1, 1, 32, width
            s.dim.extend([b,c,h,w])
            nn.attr['shape'].shape.CopyFrom(s)
        else:
            nn = dynamic_graph.node.add()
            nn.CopyFrom(n)
    return dynamic_graph

            


def tf_predict(image):
    mask, w = resizeNormalize(image, dynamic=False)
    
    x = np.zeros((1, 1, 32, w), dtype=np.float32)
    x[0] = np.expand_dims(mask, axis=0)

    dynamic_graph = revise_placeholder(w)

    with tf.Session() as sess:
        tf.import_graph_def(dynamic_graph, name="")
        input_name = sess.graph.get_tensor_by_name('input:0')
        output = sess.graph.get_tensor_by_name('output:0')
        preds = sess.run(output, {input_name: x})
    preds = preds.transpose((1,0,2))
    preds = preds[:,:,:]
    out = decode(preds)
    return out

if __name__ == "__main__":
    root = "./imgs/*.*p*g"

    imgs = glob.glob(root)
    for img_path in imgs:
        orig_image = cv2.imread(img_path,0)
        pred = tf_predict(orig_image)
        print("image:{0}====> pred:{1}".format(img_path, pred))
