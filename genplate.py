# coding=utf-8
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
from math import *

from common import chars


def rot(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]

    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])

    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])

    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_environment(img, data_set):
    index = r(len(data_set))
    env = cv2.imread(data_set[index])

    env = cv2.resize(env, (img.shape[1], img.shape[0]))

    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img


def GenCh(f, val):
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=f)
    img = img.resize((23, 70))
    A = np.array(img)

    return A


def GenCh1(f, val):
    img = Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val.encode('utf-8').decode('utf-8'), (0, 0, 0), font=f)
    A = np.array(img)
    return A


def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


def r(val):
    return int(np.random.random() * val)


def AddNoiseSingleChannel(single):
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(6), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst


def addNoise(img):
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0])
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1])
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2])
    return img


class GenPlate:
    def __init__(self, fontCh, fontEng, NoPlates):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)
        self.fontE = ImageFont.truetype(fontEng, 60, 0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread("images/template.bmp"), (226, 70))
        self.smu = cv2.imread("images/smu2.jpg")
        self.noplates_path = []
        for parent, parent_folder, filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent + "/" + filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset = 2

        self.img[0:70, offset + 8:offset + 8 + 23] = GenCh(self.fontC, val[0])
        self.img[0:70, offset + 8 + 23 + 6:offset + 8 + 23 + 6 + 23] = GenCh1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base: base + 23] = GenCh1(self.fontE, val[i + 2])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            fg = self.draw(text.encode('utf-8').decode(encoding="utf-8"))
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg, self.bg)
            com = rot(com, r(60) - 30, com.shape, 30)
            com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))

            com = tfactor(com)
            com = random_environment(com, self.noplates_path)
            com = AddGauss(com, 1 + r(4))
            com = addNoise(com)

            return com

    def genPlateString(self, pos, val):
        plateStr = ""
        box = [0, 0, 0, 0, 0, 0, 0]
        if pos != -1:
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]
                elif cpos == 1:
                    plateStr += chars[41 + r(24)]
                else:
                    plateStr += chars[31 + r(34)]

        return plateStr

    def genBatch(self, batchSize, outputPath, size):
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plateStr = G.genPlateString(-1, -1)
            img = G.generate(plateStr)
            img = cv2.resize(img, size)
            cv2.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img)


G = GenPlate("font/platech.ttf", 'font/platechar.ttf', "NoPlates")

G.genBatch(100, "./plate", (272, 72))
