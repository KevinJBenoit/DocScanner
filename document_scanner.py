import cv2
import numpy as np
import matplotlib.pyplot as plt


#read in the image and load and resize
image = cv2.imread("test_doc.jpg")
doc_image = np.copy(image)
doc_imageS = cv2.resize(doc_image, (1134,2016))

def canny(image):
    #convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #identify strong edges
    canny_image = cv2.Canny(blur, 50, 150)

    return canny_image


canny = canny(doc_imageS)

cv2.imshow('result', canny)
cv2.waitKey(0)
