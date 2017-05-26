# coding=utf-8
from mxnet.test_utils import list_gpus

import mxnet as mx
import numpy as np
import cv2

from common import chars


def get_net():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    flatten = mx.symbol.Flatten(data=relu2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=120)
    fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc25 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc26 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc27 = mx.symbol.FullyConnected(data=fc1, num_hidden=65)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25, fc26, fc27], dim=0)
    return mx.symbol.SoftmaxOutput(data=fc2, name="softmax")


def recognize_one(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.resize(img, (120, 30))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 30, 120))]
    input_shapes = dict(data_shape)

    sym = get_net()
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
    recognize_one('./recognize_samples/00.jpg')
