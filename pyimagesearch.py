import numpy as np
from cv2 import cv2


def order_points(pts):
    """
    Function that takes pts argument specifying (x, y) coordinates of the
    rectangle and orders them clockwise as top-left, top-right, bottom-right, bottom-left
    """
    #create container for coordinates for rectangle
    #order of entry:, top-left, top-right, bottom-right, bottom-left
    rectangle = np.zeros((4,2), dtype= "float32")

    #find the top-left(smallest sum) and bottom-right(largest sum) points
    pt_sum = pts.sum(axis=1)
    rectangle[0] = pts[np.argmin(pt_sum)]
    rectangle[2] = pts[np.argmax(pt_sum)]

    #find the top-right(smallest difference) and bottom-left(largest difference) points
    pt_diff = pts.diff(pts, paxis = 1)
    rectangle[1] = pts[np.argmin(pt_diff)]
    rectangle[3] = pts[np.argmax(pt_diff)]

    return rectangle

def four_point_transform(image, pts):
    """
    write descript
    """
    #organize the point coordinates
    rectangle = order_points(pts)

    #unpack the rectangle
    (tl, tr, br, bl) = rectangle

    #calculate the width of the top and bottom the captured rectangle points
    #uses pythagorean theorem
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    #get the maximum width between top and bottom
    max_width = max(int(width_bottom), int(width_top))

    #calculate the height of the top and bottom the captured rectangle points
    #uses pythagorean theorem
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    #get the maximum height between left and right
    max_height = max(int(height_right), int(height_left))

    #obtain a top-down view of the captured rectangle
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height -1]], dtype="float32")

    #perform perspective transorm matrix and apply it
    transformation_matrix = cv2.getPerspectiveTransform(rectangle, destination)
    warped = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))

    return warped