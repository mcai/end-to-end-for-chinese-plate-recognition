# coding=utf-8
import cv2

import mxnet as mx
import numpy as np
from mxnet.test_utils import list_gpus

from common import chars, plate_recognition_net


def recognize_one(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.resize(img, (120, 30))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("models/cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 30, 120))]
    input_shapes = dict(data_shape)

    sym = plate_recognition_net(train=False)
    executor = sym.simple_bind(ctx=mx.gpu() if list_gpus() else mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    executor.forward(is_train=True, data=mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()

    line = ''
    for i in range(probs.shape[0]):
        if i == 0:
            result = np.argmax(probs[i][0:31])
        elif i == 1:
            result = np.argmax(probs[i][41:65]) + 41
        else:
            result = np.argmax(probs[i][31:65]) + 31

        line += chars[result]

    print('recognized as: ' + line)


if __name__ == '__main__':
    recognize_one('./plates_to_test/00.jpg')
