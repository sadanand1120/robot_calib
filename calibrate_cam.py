import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from homography import Homography

np.set_printoptions(precision=4, suppress=True)


def filter_images(images, checkerboard):
    total_removed = 0
    for i, fname in tqdm(enumerate(images), total=len(images), desc="Filtering images"):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            os.remove(fname)
            total_removed += 1
    print(f"Total removed: {total_removed}")


def save_rectified_yaml_from_raw(raw_path, save_path):
    with open(raw_path, "r") as f:
        raw_data = yaml.safe_load(f)
    mtx = np.asarray(raw_data["camera_matrix"])
    dist = np.asarray(raw_data["dist_coeffs"])
    h = raw_data["height"]
    w = raw_data["width"]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rect_mtx, _ = Homography.get_rectified_K(mtx, dist, w, h, alpha=0.0)
    rect_dist = np.zeros_like(dist)
    rect_img = Homography.rectifyRawCamImage(img, mtx, dist, alpha=0.0)
    rect_data = {
        "camera_matrix": rect_mtx.tolist(),
        "dist_coeffs": rect_dist.tolist(),
        "height": rect_img.shape[0],
        "width": rect_img.shape[1]
    }
    with open(save_path, "w") as f:
        yaml.dump(rect_data, f)


def calibrate_camera_from_images(images, checkerboard, save_path, save_raw_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_raw_path), exist_ok=True)
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 400, 1e-8)
    for fname in tqdm(images, desc="Calibrating camera", total=len(images)):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print(f"Reprojection Error: {total_error / len(objpoints)}")
    print("camera_matrix:\n", mtx)
    print("dist_coeffs:\n", dist)
    print("img_shape:\n", (img.shape[0], img.shape[1]))
    raw_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "height": img.shape[0],
        "width": img.shape[1]
    }
    with open(save_raw_path, "w") as f:
        yaml.dump(raw_data, f)
    save_rectified_yaml_from_raw(save_raw_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--save_calib", required=True, help="Path to save (rectified) calibration YAML")
    parser.add_argument("--save_raw_calib", required=True, help="Path to save raw calibration YAML")
    parser.add_argument("--mode", choices=["filter", "calib"], required=True, help="Operation mode")
    parser.add_argument("--checkerboard", default="8x6", help="Checkerboard size as WxH; only inner corners")
    args = parser.parse_args()

    checkerboard = tuple(map(int, args.checkerboard.split('x')))
    images = [os.path.join(args.images_dir, img) for img in sorted(os.listdir(args.images_dir))]

    if args.mode == "filter":
        filter_images(images, checkerboard)
    elif args.mode == "calib":
        calibrate_camera_from_images(images, checkerboard, args.save_calib, args.save_raw_calib)

    # save_rectified_yaml_from_raw(raw_path="spot/params/raw/cam_intrinsics_1440.yaml",
    #                              save_path="spot/params/cam_intrinsics_1440.yaml")
