# coding=utf-8
import cv2
from math import *

import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from common import chars, rand
import  random


def rotate(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]

    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])

    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])

    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, m, size)

    return dst


def rotate_random(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[rand(factor), rand(factor)], [rand(factor), shape[0] - rand(factor)], [shape[1] - rand(factor), rand(factor)],
                       [shape[1] - rand(factor), shape[0] - rand(factor)]])
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
    index = rand(len(data_set))
    env = cv2.imread(data_set[index])

    env = cv2.resize(env, (img.shape[1], img.shape[0]))

    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img


def generate_char(font, val):
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=font)
    img = img.resize((23, 70))
    return np.array(img)


def generate_char1(font, val):
    img = Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val.encode('utf-8').decode('utf-8'), (0, 0, 0), font=font)
    return np.array(img)


def add_gauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


def add_noise_single_channel(single):
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + rand(6), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst


def add_noise(img):
    img[:, :, 0] = add_noise_single_channel(img[:, :, 0])
    img[:, :, 1] = add_noise_single_channel(img[:, :, 1])
    img[:, :, 2] = add_noise_single_channel(img[:, :, 2])
    return img


def generate_plate_str(pos, val):
    plate_str = ""
    box = [0, 0, 0, 0, 0, 0, 0]
    if pos != -1:
        box[pos] = 1
    for unit, cpos in zip(box, range(len(box))):
        if unit == 1:
            plate_str += val
        else:
            if cpos == 0:
                plate_str += chars[rand(31)]
            elif cpos == 1:
                plate_str += chars[41 + rand(24)]
            else:
                plate_str += chars[31 + rand(34)]

    return plate_str


def generate_batch(batch_size, output_path, size):
    newsize=size

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(batch_size):
        plate_str = generate_plate_str(-1, -1)
        img = generate_plate.generate(plate_str)
        size=int(newsize[0]+random.random()*150),int(newsize[1]+random.random()*150)
        img = cv2.resize(img, size)
        cv2.imwrite(output_path + "/" + str(i).zfill(2) + ".jpg", img)


class GeneratePlate:
    def __init__(self, font_ch, font_eng, no_plates):
        self.fontC = ImageFont.truetype(font_ch, 43, 0)
        self.fontE = ImageFont.truetype(font_eng, 60, 0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread("images/template.bmp"), (226, 70))
        self.smu = cv2.imread("images/smu2.jpg")
        self.no_plates_path = []
        for parent, parent_folder, filenames in os.walk(no_plates):
            for filename in filenames:
                path = parent + "/" + filename
                self.no_plates_path.append(path)

    def draw(self, val):
        offset = 2

        self.img[0:70, offset + 8:offset + 8 + 23] = generate_char(self.fontC, val[0])
        self.img[0:70, offset + 8 + 23 + 6:offset + 8 + 23 + 6 + 23] = generate_char1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base: base + 23] = generate_char1(self.fontE, val[i + 2])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            fg = self.draw(text.encode('utf-8').decode(encoding="utf-8"))
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg, self.bg)
            com = rotate(com, rand(60) - 30, com.shape, 30)
            com = rotate_random(com, 10, (com.shape[1], com.shape[0]))

            com = tfactor(com)
            com = random_environment(com, self.no_plates_path)
            com = add_gauss(com, 1 + rand(4))
            com = add_noise(com)

            return com

generate_plate = GeneratePlate("fonts/plate_cn.ttf", 'fonts/plate_en.ttf', "no_plates")

generate_batch(1500, "plates", (272, 72))