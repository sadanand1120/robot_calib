"""
Optimized homography and transformation utilities.
Contains geometric transformation functions and camera operations.
"""

import cv2
import numpy as np
import os
import yaml
import math
from typing import Tuple, Union, Optional
from copy import deepcopy


class Homography:
    """Utility class for homographic transformations and 3D geometry operations."""

    def __init__(self):
        pass

    @staticmethod
    def general_project_A_to_B(inp: np.ndarray, AtoBmat: np.ndarray) -> np.ndarray:
        """
        Project points from frame A to frame B using transformation matrix.

        Args:
            inp: (N x 3) array of points in frame A
            AtoBmat: (4 x 4) transformation matrix from A to B

        Returns:
            (N x 3) array of points in frame B
        """
        inp = np.asarray(inp, dtype=np.float64)
        inp_4d = Homography.get_homo_from_ordinary(inp)
        out_4d = (AtoBmat @ inp_4d.T).T
        return Homography.get_ordinary_from_homo(out_4d)

    @staticmethod
    def to_image_fov_bounds(pixels: np.ndarray, width: int, height: int, mode: str = "skip") -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrain pixel coordinates to image field of view bounds.

        Args:
            pixels: (N x 2) array of pixel coordinates
            width: Image width
            height: Image height
            mode: "skip" (remove out-of-bounds), "clip" (clamp values), "none" (no constraints)

        Returns:
            Tuple of (constrained_pixels, mask_indicating_kept_pixels)
        """
        if mode == "skip":
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            return pixels[mask], mask
        elif mode == "clip":
            pixels_clipped = pixels.copy()
            pixels_clipped[:, 0] = np.clip(pixels_clipped[:, 0], 0, width - 1)
            pixels_clipped[:, 1] = np.clip(pixels_clipped[:, 1], 0, height - 1)
            return pixels_clipped, np.ones(pixels.shape[0], dtype=bool)
        elif mode == "none":
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        else:
            raise ValueError(f"Unknown fov bounds mode: {mode}. Supported: 'skip', 'clip', 'none'")

    @staticmethod
    def get_ordinary_from_homo(points_higherD: np.ndarray) -> np.ndarray:
        """
        Convert homogeneous coordinates to ordinary coordinates.

        Args:
            points_higherD: (N x D) array of homogeneous coordinates

        Returns:
            (N x D-1) array of ordinary coordinates
        """
        # Avoid division by zero
        last_coord = points_higherD[:, -1].reshape(-1, 1)
        last_coord = np.where(last_coord == 0, 1e-10, last_coord)
        points_normalized = points_higherD / last_coord
        return points_normalized[:, :-1]

    @staticmethod
    def get_homo_from_ordinary(points_lowerD: np.ndarray) -> np.ndarray:
        """
        Convert ordinary coordinates to homogeneous coordinates.

        Args:
            points_lowerD: (N x D) array of ordinary coordinates

        Returns:
            (N x D+1) array of homogeneous coordinates
        """
        ones = np.ones((points_lowerD.shape[0], 1), dtype=points_lowerD.dtype)
        return np.hstack([points_lowerD, ones])

    @staticmethod
    def get_std_trans(cx: float = 0, cy: float = 0, cz: float = 0) -> np.ndarray:
        """
        Create standard translation transformation matrix.

        Args:
            cx, cy, cz: Translation coordinates of O_M with respect to O_F when expressed in F

        Returns:
            (4 x 4) translation transformation matrix where M_coords = T * F_coords
        """
        return np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1]
        ], dtype=np.float64)

    @staticmethod
    def get_std_rot(axis: str, alpha: float) -> np.ndarray:
        """
        Create standard rotation transformation matrix.

        Args:
            axis: Rotation axis ("X", "Y", or "Z") of frame F
            alpha: Rotation angle in radians (positive according to right-hand rule)

        Returns:
            (4 x 4) rotation transformation matrix where M_coords = T * F_coords
        """
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, cos_a, sin_a, 0],
                [0, -sin_a, cos_a, 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [cos_a, 0, -sin_a, 0],
                [0, 1, 0, 0],
                [sin_a, 0, cos_a, 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [cos_a, sin_a, 0, 0],
                [-sin_a, cos_a, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError(f"Invalid rotation axis: {axis}. Supported: 'X', 'Y', 'Z'")
        return np.array(mat, dtype=np.float64)

    @staticmethod
    def get_rectified_K(K: np.ndarray, d: np.ndarray, w: int, h: int, alpha: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Compute optimal camera matrix for undistortion.

        Args:
            K: (3 x 3) camera intrinsic matrix
            d: (5,) distortion coefficients
            w: Image width
            h: Image height
            alpha: Free scaling parameter (0.0 = crop borders, 1.0 = preserve all pixels)

        Returns:
            Tuple of (new_camera_matrix, region_of_interest)
        """
        newK, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), alpha=alpha)
        return newK, roi

    @staticmethod
    def rectifyRawCamImage(cv2_img: np.ndarray, K: np.ndarray, d: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        Rectify (undistort) raw camera image using intrinsics and distortion coefficients.

        Args:
            cv2_img: (H x W x 3) raw camera image
            K: (3 x 3) camera intrinsic matrix
            d: (5,) distortion coefficients
            alpha: Free scaling parameter (0.0 = crop borders, 1.0 = preserve all pixels)

        Returns:
            (H x W x 3) rectified camera image
        """
        h, w = cv2_img.shape[:2]
        newK, roi = Homography.get_rectified_K(K, d, w, h, alpha)
        map1, map2 = cv2.initUndistortRectifyMap(K, d, None, newK, (w, h), cv2.CV_16SC2)
        undistorted_img = cv2.remap(cv2_img, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Apply ROI cropping if alpha is 0.0
        if alpha == 0.0 and roi is not None:
            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:  # Ensure valid ROI
                undistorted_img = undistorted_img[y:y + h_roi, x:x + w_roi]
        return undistorted_img
