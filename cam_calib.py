import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import os
import yaml
import math
from copy import deepcopy
from homography import Homography


class CamCalib:
    def __init__(self, override_intrinsics_filepath=None, override_extrinsics_filepath=None, robotname="jackal", cam_res=540):
        self.override_intrinsics_filepath = override_intrinsics_filepath
        self.override_extrinsics_filepath = override_extrinsics_filepath
        self.robotname = robotname
        self.cam_res = cam_res
        self.load_params()

    def load_params(self):
        if self.robotname == "jackal":
            from jackal.ret_mats import get_cam_int_dict, get_cam_ext_dict, compute_cam_ext_T
        elif self.robotname == "spot":
            from spot.ret_mats import get_cam_int_dict, get_cam_ext_dict, compute_cam_ext_T

        self.intrinsics_dict = get_cam_int_dict(self.override_intrinsics_filepath, self.cam_res)
        self.raw_intrinsics_dict = get_cam_int_dict(self.override_intrinsics_filepath, self.cam_res, ret_raw=True)
        self.img_height = self.intrinsics_dict['height']
        self.img_width = self.intrinsics_dict['width']
        self.extrinsics_dict = get_cam_ext_dict(self.override_extrinsics_filepath)
        self.M_ext = compute_cam_ext_T(self.extrinsics_dict)

    def projectWCStoPCS(self, wcs_coords, mode="skip"):
        """
        Projects set of points in WCS to PCS (rectified).
        wcs_coords: (N x 3) array of points in WCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        ccs_coords = self.projectWCStoCCS(wcs_coords)
        return self.projectCCStoPCS(ccs_coords, mode=mode)

    def projectWCStoCCS(self, wcs_coords):
        """
        Projects set of points in WCS to CCS.
        wcs_coords: (N x 3) array of points in WCS
        Returns: (N x 3) array of points in CCS
        """
        return Homography.general_project_A_to_B(wcs_coords, self.M_ext)

    def projectCCStoWCS(self, ccs_coords):
        """
        Projects set of points in CCS to WCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 3) array of points in WCS
        """
        return Homography.general_project_A_to_B(ccs_coords, np.linalg.inv(self.M_ext))

    def projectCCStoPCS(self, ccs_coords, mode="skip"):
        """
        Projects set of points in CCS to PCS (rectified).
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        ccs_coords = np.asarray(ccs_coords, dtype=np.float64)
        ccs_mask = (ccs_coords[:, 2] > 0)  # mask to filter out points in front of camera (ie, possibly visible in image). This is important and is not taken care of by pixel bounding
        ccs_coords = ccs_coords[ccs_mask]
        if ccs_coords.shape[0] == 0 or ccs_coords is None:
            return None, None
        R = np.zeros((3, 1), dtype=np.float64)
        T = np.zeros((3, 1), dtype=np.float64)
        K = self.intrinsics_dict['camera_matrix']
        d = self.intrinsics_dict['dist_coeffs']   # zero as its rectified
        image_points, _ = cv2.projectPoints(ccs_coords, R, T, K, d)
        image_points = image_points.reshape(-1, 2).astype(int)
        image_points, pcs_mask = Homography.to_image_fov_bounds(image_points, self.img_width, self.img_height, mode=mode)
        unified_mask = deepcopy(ccs_mask)
        unified_mask[ccs_mask] = pcs_mask
        return image_points, unified_mask

    def rectifyRawCamImage(self, cv2_img, alpha=0.0):
        """
        Rectifies (meaning undistorts for monocular) the raw camera image using the camera intrinsics and distortion coefficients.
        cv2_img: (H x W x 3) array of raw camera image
        alpha: 0.0 -> crop the image to remove black borders, 1.0 -> preserve all pixels but retains black borders
        Returns: (H x W x 3) array of rectified camera image
        """
        return Homography.rectifyRawCamImage(cv2_img, self.raw_intrinsics_dict['camera_matrix'], self.raw_intrinsics_dict['dist_coeffs'], alpha=alpha)
