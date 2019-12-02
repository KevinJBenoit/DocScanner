from cv2 import cv2
from PIL import Image
import numpy as np
import imutils
from skimage.filters import threshold_local
from pyimagesearch import four_point_transform



def canny(image_in):
    """
    applys greyscale to the passed in image
    returns a canny copy of the image
    """
    #convert to gray
    gray = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
    #apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #identify strong edges
    canny_image = cv2.Canny(blur, 75, 200)
    return canny_image



def find_edges(image, canny_image):
    """
    applys an an outline of the document in the passed in image
    needs a canny copy of the image
    """
    #find the contours
    contours = cv2.findContours(canny_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .02 * peri, True)

        if len(approx) == 4:
            screen_contours = approx
            break

    cv2.drawContours(image, [screen_contours], -1, (0, 255, 0), 2)
    return screen_contours

def main():
    """
    run the main program
    """
    #read in the image and load and resize
    image = cv2.imread("test_doc.jpg")
    doc_image = np.copy(image)
    doc_image_s = imutils.resize(doc_image, height=2000)
    ratio = doc_image.shape[0] / 2000

    #apply canny
    canny_doc_image_s = canny(doc_image_s)

    #get the contours from the canny image and apply to the colored image
    screen_contours = find_edges(doc_image_s, canny_doc_image_s)


    #change the perspective of the image based on the contours and threshold
    warped = four_point_transform(image, screen_contours.reshape(4,2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    threshold = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > threshold).astype("uint8") * 255

    cv2.imshow("original", imutils.resize(image, height=1500))
    cv2.imshow("scanned", imutils.resize(warped, height=1500))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
