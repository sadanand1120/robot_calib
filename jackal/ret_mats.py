"""
Jackal frames description: Assume you are watching the robot head on, with the robot facing you.
- Frame 1: World Coordinate System (WCS) / base_link
    At the robot body center but on ground (though attached to robot, so moves with robot)
    +X: Forward of the robot (towards you)
    +Y: Left of the robot (to your right)
    +Z: Upwards (away from the ground, to the sky)
- Frame 2: Intermediate frame
    At the camera lens center, but axes aligned to WCS
    +X: Forward of the robot (towards you)
    +Y: Left of the robot (to your right)
    +Z: Upwards (away from the ground, to the sky)
- Frame 3: Camera Coordinate System (CCS)
    At the camera lens center, with conventional camera axes
    +X: Right of the camera (to your left)
    +Y: Down of the camera (towards the ground, may not be straight down)
    +Z: Coming out of camera along the lens axis (towards you)
- Frame 4: Pixel Coordinate System (PCS)
    At the top left (from your perspective) corner of the image, with conventional image axes
    +X: To your right (left of the image)
    +Y: Downwards

The equation goes like: PCS = M_int * M_perp * M_ext * WCS
    - M_int (3 x 3): Intrinsic matrix, gets PCS from CCS
    - M_perp (3 x 4): Perspective matrix, 3D to 2D projection
    - M_ext (4 x 4): Extrinsic matrix, gets CCS from WCS

Additional info:
    - M_int is constant for a given camera
    - You should NOT directly do M_int multiplication as shown above, but instead use cv2.projectPoints as it takes into account the distortion coefficients as well
    - Sometimes, M_perp * M_ext is combined into a single matrix, called the extrinsic matrix, where then it is (3 x 4) extrinsic matrix
    - (Corrected) lidar frame is such that it only has z displacement from base_link
"""

import cv2
import numpy as np
import os
import yaml
import math
from copy import deepcopy
from homography import Homography
from collections import OrderedDict


def get_cam_int_dict(override_intrinsics_filepath=None, cam_res=540, ret_raw=False):
    """
    ret_raw: whether to return the raw camera intrinsics or the rectified camera intrinsics
    """
    if override_intrinsics_filepath is None:
        if ret_raw:
            intrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"params/raw/zed_left_rect_intrinsics_{cam_res}.yaml")
        else:
            intrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"params/zed_left_rect_intrinsics_{cam_res}.yaml")
    else:
        intrinsics_filepath = override_intrinsics_filepath

    with open(intrinsics_filepath, 'r') as f:
        intrinsics_dict = yaml.safe_load(f)
    intrinsics_dict['camera_matrix'] = np.array(intrinsics_dict['camera_matrix']).reshape((3, 3))
    intrinsics_dict['dist_coeffs'] = np.array(intrinsics_dict['dist_coeffs']).reshape((1, 5)).squeeze()
    if 'rvecs' in intrinsics_dict:
        intrinsics_dict['rvecs'] = np.array(intrinsics_dict['rvecs']).reshape((-1, 3))
    if 'tvecs' in intrinsics_dict:
        intrinsics_dict['tvecs'] = np.array(intrinsics_dict['tvecs']).reshape((-1, 3))
    return intrinsics_dict


def get_cam_ext_dict(override_extrinsics_filepath=None):
    if override_extrinsics_filepath is None:
        extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_zed_left_extrinsics.yaml")
    else:
        extrinsics_filepath = override_extrinsics_filepath

    with open(extrinsics_filepath, 'r') as f:
        extrinsics_dict = yaml.safe_load(f)
    return extrinsics_dict


def compute_cam_ext_T(cam_ext_dict):
    """
    Returns the extrinsic matrix (4 x 4) that transforms from WCS to CCS        
    """
    # Transforms from WCS to Intermediate frame
    T1 = Homography.get_std_trans(cx=cam_ext_dict['T12']['T1']['X'] / 100,
                                  cy=cam_ext_dict['T12']['T1']['Y'] / 100,
                                  cz=cam_ext_dict['T12']['T1']['Z'] / 100)
    # Transforms from Intermediate frame to CCS
    T2 = Homography.get_std_rot(axis=cam_ext_dict['T23']['R1']['axis'],
                                alpha=np.deg2rad(cam_ext_dict['T23']['R1']['alpha']))
    T3 = Homography.get_std_rot(axis=cam_ext_dict['T23']['R2']['axis'],
                                alpha=np.deg2rad(cam_ext_dict['T23']['R2']['alpha']))
    T4 = np.array(cam_ext_dict['T23']['R3'])
    T5 = Homography.get_std_rot(axis=cam_ext_dict['T23']['R4']['axis'],
                                alpha=np.deg2rad(cam_ext_dict['T23']['R4']['alpha']))
    return T5 @ T4 @ T3 @ T2 @ T1


def compute_parameterized_cam_ext_T(xcm, ycm, zcm, r1deg, r2deg, r4deg):
    cam_ext_dict = get_cam_ext_dict()
    cam_ext_dict['T12']['T1']['X'] = xcm
    cam_ext_dict['T12']['T1']['Y'] = ycm
    cam_ext_dict['T12']['T1']['Z'] = zcm
    cam_ext_dict['T23']['R1']['alpha'] = r1deg
    cam_ext_dict['T23']['R2']['alpha'] = r2deg
    cam_ext_dict['T23']['R4']['alpha'] = r4deg
    return compute_cam_ext_T(cam_ext_dict)


def get_parameters_cam_ext_T(cam_ext_dict):
    return OrderedDict({
        "xcm": cam_ext_dict['T12']['T1']['X'],
        "ycm": cam_ext_dict['T12']['T1']['Y'],
        "zcm": cam_ext_dict['T12']['T1']['Z'],
        "r1deg": cam_ext_dict['T23']['R1']['alpha'],
        "r2deg": cam_ext_dict['T23']['R2']['alpha'],
        "r4deg": cam_ext_dict['T23']['R4']['alpha']
    })


def get_lidar_ext_dict(override_extrinsics_filepath=None):
    if override_extrinsics_filepath is None:
        extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_lidar_extrinsics.yaml")
    else:
        extrinsics_filepath = override_extrinsics_filepath

    with open(extrinsics_filepath, 'r') as f:
        extrinsics_dict = yaml.safe_load(f)
    return extrinsics_dict


def get_lidar_actual_ext_dict(override_extrinsics_filepath=None):
    if override_extrinsics_filepath is None:
        extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_actual_lidar_extrinsics.yaml")
    else:
        extrinsics_filepath = override_extrinsics_filepath

    with open(extrinsics_filepath, 'r') as f:
        extrinsics_dict = yaml.safe_load(f)
    return extrinsics_dict


def compute_lidar_ext_T(lidar_ext_dict):
    """
    Returns the extrinsic matrix (4 x 4) that transforms from WCS to VLP frame
    """
    T1 = Homography.get_std_trans(cx=lidar_ext_dict['T1']['Trans1']['X'] / 100,
                                  cy=lidar_ext_dict['T1']['Trans1']['Y'] / 100,
                                  cz=lidar_ext_dict['T1']['Trans1']['Z'] / 100)
    T2 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot1']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot1']['alpha']))
    T3 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot2']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot2']['alpha']))
    T4 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot3']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot3']['alpha']))
    return T4 @ T3 @ T2 @ T1


def compute_lidar_actual_ext_T(lidar_ext_dict):
    """
    Returns the actual extrinsic matrix (4 x 4) that transforms from WCS to real VLP frame
    """
    T1 = Homography.get_std_trans(cx=lidar_ext_dict['T1']['Trans1']['X'] / 100,
                                  cy=lidar_ext_dict['T1']['Trans1']['Y'] / 100,
                                  cz=lidar_ext_dict['T1']['Trans1']['Z'] / 100)
    T2 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot1']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot1']['alpha']))
    T3 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot2']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot2']['alpha']))
    T4 = Homography.get_std_rot(axis=lidar_ext_dict['T2']['Rot3']['axis'],
                                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot3']['alpha']))
    return T4 @ T3 @ T2 @ T1
