#!/usr/bin/env python

# MESSY, NEEDS CLEANING

import cv2
import numpy as np
import os
import glob
import yaml
from checkerboard import detect_checkerboard
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", default="/home/dynamo/AMRL_Research/repos/spot_calib/3072", type=str, help="Directory containing images")
parser.add_argument("--save_calib", default="/home/dynamo/AMRL_Research/repos/spot_calib/params/cam_intrinsics_3072.yaml", type=str, help="Output calibration file")
parser.add_argument("--mode", default="filter", type=str, help="filter or calib")
parser.add_argument("--view_mode", default=False, type=bool, help="Show images")
parser.add_argument("--score_threshold", default=0.1, type=float, help="Threshold for checkerboard detection")
parser.add_argument("--checkerboard", default="8x6", type=str, help="Checkerboard size")
args = parser.parse_args()

images_dir = args.images_dir
save_calib = args.save_calib
MODE = args.mode
VIEW_MODE = args.view_mode
SCORE_THRESHOLD = args.score_threshold

CHECKERBOARD = tuple(map(int, args.checkerboard.split('x')))
print(f"Checkerboard size: {CHECKERBOARD}")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-8)

objpoints = []
imgpoints = []
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

images = sorted(glob.glob(images_dir + '/*.png'))

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)

def resize_img(img, scale_percent=60):
    _width = int(img.shape[1] * scale_percent / 100)
    _height = int(img.shape[0] * scale_percent / 100)
    dim = (_width, _height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def pretty_print(v):
    print(np.array_str(v, precision=4, suppress_small=True))

if MODE == "filter":
    tot_removed = 0

print(f"To process {len(images)} images")
for i, fname in enumerate(images):
    print(f"Processing image: {i}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners. If desired number of corners are found in the image then ret = true
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    corners, score = detect_checkerboard(gray, CHECKERBOARD)
    ret = score < SCORE_THRESHOLD
    if ret:
        corners = corners.reshape((*CHECKERBOARD, 1, 2)).transpose((1, 0, 2, 3)).reshape((-1, 1, 2)).astype(np.float32)

    if MODE == "calib":
        # ASSUMES you did the filtering already
        # Perform calib and show the images with corners where you could find all corners
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refining pixel coordinates for given 2d points.
        imgpoints.append(corners2)
        if VIEW_MODE:
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)  # Draw and display the corners
    elif MODE == "filter":
        # Perform filtering and remove images where you could not find all corners. Show the images which you are removing
        if ret == False:
            print(f"Removing image: {i} for score: {score}")
            os.remove(fname)
            tot_removed += 1

    if VIEW_MODE:
        cv2.imshow('img', resize_img(img))
        cv2.waitKey(0)


cv2.destroyAllWindows()

if MODE == "filter":
    print(f"Total removed: {tot_removed}")
elif MODE == "calib":
    """
    Performing camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel 
    coordinates of the detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"Reprojection Error: {reprojection_error}")

    print("camera_matrix:")
    pretty_print(mtx)
    print("dist_coeffs:")
    pretty_print(dist)

    # Saving the results into yaml file
    mtx_list = np.array(mtx).squeeze().tolist()
    dist_list = np.array(dist).squeeze().tolist()
    rvecs_list = np.array(rvecs).squeeze().tolist()
    tvecs_list = np.array(tvecs).squeeze().tolist()

    calibration_data = {
        "camera_matrix": mtx_list,
        "dist_coeffs": dist_list,
        "height": img.shape[0],
        "width": img.shape[1],
        "rvecs": rvecs_list,
        "tvecs": tvecs_list
    }
    with open(save_calib, 'w') as outfile:
        yaml.dump(calibration_data, outfile)