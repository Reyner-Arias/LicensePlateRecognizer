import cv2 
import pytesseract
import numpy as np
import math

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options

# Show rectangles
def getLicensePlateImg(image):
    plateClassifier = cv2.CascadeClassifier('cascade.xml')

    plate = plateClassifier.detectMultiScale(image, scaleFactor = 3, minNeighbors = 120, minSize = (80, 40))

    for (x, y, w, h) in plate:
        cv2.rectangle(image, (x+math.floor(x*0.05), y) ,(x+w+math.floor(x*0.05), y+h), (0, 255, 0), 2)
        cv2.putText(image, 'License Plate', (x+math.floor(x*0.05), y-10), 2, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', image)
    cv2.waitKey(0)

    """plateClassifier = cv2.CascadeClassifier('cascade.xml')

    plate = plateClassifier.detectMultiScale(image, scaleFactor = 5, minNeighbors = 91, minSize = (200, 100))

    (x, y, w, h) = plate[0]
    
    image = image[y:y+h, x:x+w]

    open(r"imgHaar.jpg", "w")
    cv2.imwrite(r"imgHaar.jpg", image)"""

    return image

# Preprocessing function
def preprocessing(image):
    image = cv2.resize(image, (1500, 750), interpolation = cv2.INTER_AREA)

    image = cv2.GaussianBlur(image, (7, 7), 0)

    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)

    for _ in range(1):
        image = cv2.erode(image, kernel, iterations = 1)
        
    for _ in range(1):
        image = cv2.dilate(image, kernel, iterations = 1)

    open(r"imgPre.jpg", "w")
    cv2.imwrite(r"imgPre.jpg", image)

    return image

# Preprocessing function
def applyOCR(image):
    return pytesseract.image_to_string(image, config=build_tesseract_options())

def main():
    exists = False # Flag to identify a existing path
    print("Welcome to the License Plate Recognizer, please type the path where the image is located:")

    while(not exists):
        path = input()
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except:
            print("The input text is not a valid path in this system")
        else:
            exists = True

    # image = preprocessing(image) # Applying preprocessing to increase the recognition level

    # Adding custom options
    # custom_config = r'--oem 3 --psm 6'

    image = getLicensePlateImg(image)

    image = preprocessing(image)

    print(applyOCR(image))

main()