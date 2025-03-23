# DONE
import cv2
import numpy as np
import time


def update_circle(img):
    x = cv2.getTrackbarPos('X', 'MyImage')
    y = cv2.getTrackbarPos('Y', 'MyImage')
    img_copy = img.copy()
    cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('MyImage', img_copy)


image = cv2.imread('/home/dynamo/Music/spot_ex3.png')
image = cv2.resize(image, (640, 480))
height, width, _ = image.shape
default_x = width // 2
default_y = height // 2
cv2.namedWindow('MyImage')
cv2.createTrackbar('X', 'MyImage', default_x, width - 1, lambda x: update_circle(image))
cv2.createTrackbar('Y', 'MyImage', default_y, height - 1, lambda x: update_circle(image))
update_circle(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
