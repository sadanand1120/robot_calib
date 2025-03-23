import cv2
import numpy as np
import os
import yaml
import math
from copy import deepcopy


class Homography:
    def __init__(self):
        pass

    @staticmethod
    def general_project_A_to_B(inp, AtoBmat):
        """
        Project inp from A frame to B
        inp: (N x 3) array of points in A frame
        AtoBmat: (4 x 4) transformation matrix from A to B
        Returns: (N x 3) array of points in B frame
        """
        inp = np.asarray(inp).astype(np.float64)
        inp_4d = Homography.get_homo_from_ordinary(inp)
        out_4d = (AtoBmat @ inp_4d.T).T
        return Homography.get_ordinary_from_homo(out_4d)

    @staticmethod
    def to_image_fov_bounds(pixels, width, height, mode="skip"):
        """
        Returns the pixels coords in image bounds and a mask for indicating which pixels were kept (ie, relevant for skip mode)
        """
        if mode == "skip":
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            return pixels[mask], mask
        elif mode == "clip":
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        elif mode == "none":
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        else:
            raise ValueError("Unknown fov bounds mode!")

    @staticmethod
    def get_safe_projs(distCoeffs, obj_pts):
        """
        Vectorize this for more efficiency (Slow on large point clouds)
        Custom point projection function to fix OpenCV distortion issue for out of bound points:
        https://github.com/opencv/opencv/issues/17768
        """
        # Define a list of booleans to denote if a variable is safe
        obj_pts_safe = np.array([True] * len(obj_pts))

        # First step is to get the location of the points relative to the camera.
        obj_pts_rel_cam = deepcopy(obj_pts)

        # Define the homogenous coordiantes
        x_homo_vals = (obj_pts_rel_cam[:, 0] / obj_pts_rel_cam[:, 2]).astype(np.complex128)
        y_homo_vals = (obj_pts_rel_cam[:, 1] / obj_pts_rel_cam[:, 2]).astype(np.complex128)

        # Define the distortion terms, and vectorize calculating of powers of x_homo_vals
        #   and y_homo_vals
        k1, k2, p1, p2, k3 = distCoeffs
        x_homo_vals_2 = np.power(x_homo_vals, 2)
        y_homo_vals_2 = np.power(y_homo_vals, 2)
        x_homo_vals_4 = np.power(x_homo_vals, 4)
        y_homo_vals_4 = np.power(y_homo_vals, 4)
        x_homo_vals_6 = np.power(x_homo_vals, 6)
        y_homo_vals_6 = np.power(y_homo_vals, 6)

        # Find the bounds on the x_homo coordinate to ensure it is closer than the
        #   inflection point of x_proj as a function of x_homo
        x_homo_min = np.full(x_homo_vals.shape, np.inf)
        x_homo_max = np.full(x_homo_vals.shape, -np.inf)
        for i in range(len(y_homo_vals)):
            # Expanded projection function polynomial coefficients
            x_proj_coeffs = np.array([k3,
                                      0,
                                      k2 + 3 * k3 * y_homo_vals_2[i],
                                      0,
                                      k1 + 2 * k2 * y_homo_vals_2[i] + 3 * k3 * y_homo_vals_4[i],
                                      3 * p2,
                                      1 + k1 * y_homo_vals_2[i] + k2 * y_homo_vals_4[i] + k3 * y_homo_vals_6[i] + 2 * p1 * y_homo_vals[i],
                                      p2 * y_homo_vals_2[i]])

            # Projection function derivative polynomial coefficients
            x_proj_der_coeffs = np.polyder(x_proj_coeffs)

            # Find the root of the derivative
            roots = np.roots(x_proj_der_coeffs)

            # Get the real roots
            # Approximation of real[np.where(np.isreal(roots))]
            real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

            for real_root in real_roots:
                x_homo_min[i] = np.minimum(x_homo_min[i], real_root)
                x_homo_max[i] = np.maximum(x_homo_max[i], real_root)

        # Check that the x_homo values are within the bounds
        obj_pts_safe *= np.where(x_homo_vals > x_homo_min, True, False)
        obj_pts_safe *= np.where(x_homo_vals < x_homo_max, True, False)

        # Find the bounds on the y_homo coordinate to ensure it is closer than the
        #   inflection point of y_proj as a function of y_homo
        y_homo_min = np.full(y_homo_vals.shape, np.inf)
        y_homo_max = np.full(y_homo_vals.shape, -np.inf)
        for i in range(len(x_homo_vals)):
            # Expanded projection function polynomial coefficients
            y_proj_coeffs = np.array([k3,
                                      0,
                                      k2 + 3 * k3 * x_homo_vals_2[i],
                                      0,
                                      k1 + 2 * k2 * x_homo_vals_2[i] + 3 * k3 * x_homo_vals_4[i],
                                      3 * p1,
                                      1 + k1 * x_homo_vals_2[i] + k2 * x_homo_vals_4[i] + k3 * x_homo_vals_6[i] + 2 * p2 * x_homo_vals[i],
                                      p1 * x_homo_vals_2[i]])

            # Projection function derivative polynomial coefficients
            y_proj_der_coeffs = np.polyder(y_proj_coeffs)

            # Find the root of the derivative
            roots = np.roots(y_proj_der_coeffs)

            # Get the real roots
            # Approximation of real[np.where(np.isreal(roots))]
            real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

            for real_root in real_roots:
                y_homo_min[i] = np.minimum(y_homo_min[i], real_root)
                y_homo_max[i] = np.maximum(y_homo_max[i], real_root)

        # Check that the x_homo values are within the bounds
        obj_pts_safe *= np.where(y_homo_vals > y_homo_min, True, False)
        obj_pts_safe *= np.where(y_homo_vals < y_homo_max, True, False)

        # Return the indices where obj_pts is safe to project
        return np.where(obj_pts_safe == True)[0]

    @staticmethod
    def get_ordinary_from_homo(points_higherD):
        # Scales so that last coord is 1 and then removes last coord
        points_higherD = points_higherD / points_higherD[:, -1].reshape(-1, 1)  # scale by the last coord
        return points_higherD[:, :-1]

    @staticmethod
    def get_homo_from_ordinary(points_lowerD):
        # Append 1 to each point
        ones = np.ones((points_lowerD.shape[0], 1))  # create a column of ones
        return np.hstack([points_lowerD, ones])  # append the ones column to points

    @staticmethod
    def get_std_trans(cx=0, cy=0, cz=0):
        """
        cx, cy, cz are the coords of O_M wrt O_F when expressed in F
        Multiplication goes like M_coords = T * F_coords
        """
        mat = [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1]
        ]
        return np.array(mat)

    @staticmethod
    def get_std_rot(axis, alpha):
        """
        axis is either "X", "Y", or "Z" axis of F and alpha is positive acc to right hand thumb rule dirn
        Multiplication goes like M_coords = T * F_coords
        """
        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, math.cos(alpha), math.sin(alpha), 0],
                [0, -math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [math.cos(alpha), 0, -math.sin(alpha), 0],
                [0, 1, 0, 0],
                [math.sin(alpha), 0, math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [math.cos(alpha), math.sin(alpha), 0, 0],
                [-math.sin(alpha), math.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError("Invalid axis!")
        return np.array(mat)
