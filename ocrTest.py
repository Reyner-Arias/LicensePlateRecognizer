import cv2 
import pytesseract
import numpy as np


# Show rectangles
def showRectangles(image):
    return 0
    # boxes = pytesseract.image_to_boxes(image, config = " -c tessedit_create_boxfile=1")
    

# Preprocessing function
def preprocessing(image):
    # image = cv2.GaussianBlur(image, (5, 5), 0) # Blurring to reduce noise in original image

    # kernel = np.ones((5,5),np.uint8)
    # image = cv2.dilate(image, kernel, iterations = 1)

    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]

    # image = cv2.Canny(image, 100, 200) # Edge detection to reduce potential erroneous characters identified in the image

    return image

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

    image = preprocessing(image) # Applying preprocessing to increase the recognition level

    # Adding custom options
    # custom_config = r'--oem 3 --psm 6'

    image = showRectangles(image)

    # print(licensePlate) # Final result of the recognizer

main()