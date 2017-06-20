import cv2
import os

img = cv2.imread('plates_to_test3/timg.jpg', -1)
# print(img.shape)

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def sort_func(c):
    (x, y, w, h) = cv2.boundingRect(c)
    return x, -y


contours = sorted(contours, key=sort_func)

i = 0
for c in contours:
    rect = cv2.boundingRect(c)

    print(rect)

    x, y, w, h = rect

    if 350 < w * h < 1500:
        dst = img[y:y + h, x:x + w]
        dir = 'plates_to_test3/charOut/'
        os.system('mkdir -p ' + dir)
        cv2.imwrite(dir + str(i) + 'ch.png', dst)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        i += 1
    # print((x, y), (x + w, y + h))

cv2.imwrite('plates_to_test3/charOut/preview.png', img)
# print(img.shape)
