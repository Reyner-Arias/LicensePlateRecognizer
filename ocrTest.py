import cv2 
import pytesseract

img = cv2.imread('muahaha.jpg', cv2.IMREAD_GRAYSCALE)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, config=custom_config)

print(text)