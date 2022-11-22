import pytesseract
import numpy as np
import cv2

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options

img = cv2.imread('12.jpg', cv2.IMREAD_GRAYSCALE)
hImg, wImg = img.shape
custom_config = r'--oem 3 --psm 6'

img = cv2.resize(img, (1500, 750), interpolation = cv2.INTER_AREA)

img = cv2.GaussianBlur(img, (7, 7), 0)

img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = np.ones((5,5),np.uint8)

for i in range(4):
    img = cv2.erode(img, kernel, iterations = 1)
    
for i in range(4):
    img = cv2.dilate(img, kernel, iterations = 1)

"""kernel = np.ones((5,5),np.uint8)
img = cv2.erode(img, kernel, iterations = 1)"""

"""kernel = np.ones((5,5),np.uint8)
img = cv2.dilate(img, kernel, iterations = 1)"""

cv2.imshow("Prueba", img)
cv2.waitKey(0)

boxes = pytesseract.image_to_boxes(img, config = " -c tessedit_create_boxfile=1")

print(boxes)

print(pytesseract.image_to_string(img, config=build_tesseract_options()))

for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
    cv2.putText(img, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)

open(r"img.jpg", "w")
cv2.imwrite(r"img.jpg", img)