import cv2

img = cv2.imread('plates_to_test3/timg.jpg', -1)
print(img.shape)

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    if 1500 > w * h > 350:
        dst = img[y:y + h, x:x + w]  # 输出该区域
        cv2.imwrite('plates_to_test3/charOut/' + str(i) + 'ch.png', dst)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 标注该区域
        i += 1
    print((x, y), (x + w, y + h))

cv2.imwrite('plates_to_test3/charOut/preview.png', img)
print(img.shape)
