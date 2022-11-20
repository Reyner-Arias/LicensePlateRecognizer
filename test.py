import pytesseract
import numpy as np
import cv2
img = cv2.imread('muahaha.jpg')
hImg, wImg, _ = img.shape
custom_config = r'--oem 3 --psm 6'

img = cv2.medianBlur(img, 5)

kernel = np.ones((5,5),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# img = cv2.Canny(img, 100, 200)

boxes = pytesseract.image_to_boxes(img, config = " -c tessedit_create_boxfile=1")

print(boxes)

print(pytesseract.image_to_string(img, config=custom_config))

for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
    cv2.putText(img, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)

open(r"img.jpg", "w")
cv2.imwrite(r"img.jpg", img)