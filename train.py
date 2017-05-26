# coding=utf-8
import mxnet as mx
from mxnet.test_utils import list_gpus

from common import plate_recognition_net
from generate_plates import *
from generate_plates import generate_plate


def rand_range(lo, hi):
    return lo + rand(hi - lo)


def generate_rand():
    label = [rand_range(0, 31), rand_range(41, 65)]
    name = chars[label[0]] + chars[label[1]]

    for i in range(5):
        label.append(rand_range(31, 65))
        name += chars[label[i + 2]]

    return name, label


def generate_sample(generate_plate, width, height):
    num, label = generate_rand()
    img = generate_plate.generate(num)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1 / 255.0)
    img = img.transpose(2, 0, 1)

    return label, img


class PlateRecognitionBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class PlateRecognitionIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(PlateRecognitionIter, self).__init__()

        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]

    def __iter__(self):
        for k in range(int(self.count / self.batch_size)):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = generate_sample(generate_plate, self.width, self.height)
                data.append(img)
                label.append(num)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch = PlateRecognitionBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


def accuracy(label, pred):
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range(int(pred.shape[0] / 7)):
        ok = True
        for j in range(7):
            k = i * 7 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total


def train():
    net = plate_recognition_net(train=True)

    model = mx.model.FeedForward(
        symbol=net, ctx=mx.gpu() if list_gpus() else mx.cpu(),
        num_epoch=1,
        learning_rate=0.001,
        wd=0.00001,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        momentum=0.9)

    batch_size = 20

    # data_train = PlateRecognitionIter(2000000, batch_size, 7, 30, 120)
    data_train = PlateRecognitionIter(5000000, batch_size, 7, 30, 120)
    data_test = PlateRecognitionIter(1000, batch_size, 7, 30, 120)

    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

    model.fit(X=data_train, eval_data=data_test, eval_metric=accuracy,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))
    model.save("models/cnn-ocr")


if __name__ == '__main__':
    train()
