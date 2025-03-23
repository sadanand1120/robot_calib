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
        self.img_height = self.intrinsics_dict['height']
        self.img_width = self.intrinsics_dict['width']
        self.extrinsics_dict = get_cam_ext_dict(self.override_extrinsics_filepath)
        self.M_ext = compute_cam_ext_T(self.extrinsics_dict)

    def projectWCStoPCS(self, wcs_coords, mode="skip"):
        """
        Projects set of points in WCS to PCS.
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
        Projects set of points in CCS to PCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        ccs_coords = np.array(ccs_coords).astype(np.float64)
        ccs_mask = (ccs_coords[:, 2] >= 0)  # mask to filter out points in front of camera (ie, possibly visible in image). This is important and is not taken care of by pixel bounding
        ccs_coords = ccs_coords[ccs_mask]
        if ccs_coords.shape[0] == 0 or ccs_coords is None:
            return None, None
        R = np.zeros((3, 1))
        T = np.zeros((3, 1))
        K = self.intrinsics_dict['camera_matrix']
        d = self.intrinsics_dict['dist_coeffs']
        image_points, _ = cv2.projectPoints(ccs_coords, R, T, K, d)
        image_points = np.swapaxes(image_points, 0, 1).astype(np.int32)

        # Additional processing for the opencv issue https://github.com/opencv/opencv/issues/17768
        image_points_nodist, _ = cv2.projectPoints(ccs_coords, R, T, K, np.zeros((5, 1)).squeeze())
        image_points_nodist = np.swapaxes(image_points_nodist, 0, 1).astype(np.int32)
        _, num_points, _ = image_points_nodist.shape
        safe_image_points_mask = Homography.get_safe_projs(d, ccs_coords)
        safe_image_points = np.zeros((1, num_points, 2))
        safe_image_points[0, safe_image_points_mask, :] = image_points[0, safe_image_points_mask, :]
        unsafe_image_points_mask = np.delete(np.arange(0, num_points), safe_image_points_mask)
        safe_image_points[0, unsafe_image_points_mask, :] = image_points_nodist[0, unsafe_image_points_mask, :]
        image_points = safe_image_points.astype(np.int32).squeeze().reshape((-1, 2))

        image_points, pcs_mask = Homography.to_image_fov_bounds(image_points, self.img_width, self.img_height, mode=mode)
        unified_mask = deepcopy(ccs_mask)
        unified_mask[ccs_mask] = pcs_mask
        return image_points, unified_mask

    def projectPCStoWCSground(self, pcs_coords, apply_dist=True, mode="skip"):
        """
        Projects set of points in PCS to WCS, assuming they are on the ground plane (z=0) in WCS.
        pcs_coords: (N x 2) array of points in PCS
        Returns: (N x 3) array of points in WCS, and a mask to indicate which pixel locs were kept during FOV bounding
        """
        pcs_coords = np.array(pcs_coords).astype(np.float64)
        K = self.intrinsics_dict['camera_matrix']
        d = self.intrinsics_dict['dist_coeffs']
        R = np.eye(3)
        undistorted_pcs_coords = cv2.undistortPoints(pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
        undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
        undistorted_pcs_coords, pcs_mask = Homography.to_image_fov_bounds(undistorted_pcs_coords, self.img_width, self.img_height, mode=mode)
        if apply_dist:
            pcs_coords = undistorted_pcs_coords
        pcs_coords_3d = Homography.get_homo_from_ordinary(pcs_coords)
        M_ext_short = self.M_ext[:, [0, 1, 3]][:-1, :].reshape((3, 3))
        H = K @ M_ext_short  # 3x3
        H_inv = np.linalg.inv(H)
        wcs_coords_3d = (H_inv @ pcs_coords_3d.T).T
        wcs_coords = Homography.get_ordinary_from_homo(wcs_coords_3d)
        zeros = np.zeros((wcs_coords.shape[0], 1))
        wcs_coords_full = np.hstack([wcs_coords, zeros]).reshape((wcs_coords.shape[0], 3))
        ccs_coords_full = self.projectWCStoCCS(wcs_coords_full)
        ccs_mask = (ccs_coords_full[:, 2] >= 0)  # this ccs_mask calculation is to cross-check if projected wcs points make sense or not
        unified_mask = deepcopy(pcs_mask)
        unified_mask[pcs_mask] = ccs_mask
        return wcs_coords_full[ccs_mask], unified_mask
