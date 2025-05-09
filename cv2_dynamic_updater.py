"""
Template/demo file on how to dynamically update a pose arrow on an image using OpenCV.
"""

import cv2
import numpy as np
import math

def update_pose(img):
    x = cv2.getTrackbarPos('X', 'MyImage')
    y = cv2.getTrackbarPos('Y', 'MyImage')
    theta_deg = cv2.getTrackbarPos('Theta', 'MyImage')
    theta_rad = np.deg2rad(theta_deg)

    img_copy = img.copy()
    # Draw circle (robot base)
    cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)

    # Draw orientation arrow
    arrow_length = 18
    end_x = int(x + arrow_length * math.cos(theta_rad))
    end_y = int(y - arrow_length * math.sin(theta_rad))  # y-axis inverted in image
    cv2.arrowedLine(img_copy, (x, y), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

    cv2.imshow('MyImage', img_copy)

# Load and prepare image
image = cv2.imread('/home/dynamo/Music/spot_ex3.png')
image = cv2.resize(image, (640, 480))
height, width, _ = image.shape

# Initial pose
default_x = width // 2
default_y = height // 2
default_theta = 0

# Create UI
cv2.namedWindow('MyImage')
cv2.createTrackbar('X', 'MyImage', default_x, width - 1, lambda x: update_pose(image))
cv2.createTrackbar('Y', 'MyImage', default_y, height - 1, lambda x: update_pose(image))
cv2.createTrackbar('Theta', 'MyImage', default_theta, 359, lambda x: update_pose(image))

# Initial render
update_pose(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
