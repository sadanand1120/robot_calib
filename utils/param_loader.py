"""
Unified parameter loader for camera and lidar calibration parameters.
Eliminates redundancy between jackal and spot parameter files.
"""

import cv2
import numpy as np
import os
import yaml
import math
from typing import Dict, Any, Optional, Union
from collections import OrderedDict
from copy import deepcopy
from homography import Homography
from utils.constants import (
    SUPPORTED_ROBOTS, DEFAULT_CAM_RESOLUTIONS,
    CAMERA_INTRINSICS_PATTERNS, RAW_CAMERA_INTRINSICS_PATTERNS,
    CAMERA_EXTRINSICS_FILES, LIDAR_EXTRINSICS_FILES,
    ACTUAL_LIDAR_EXTRINSICS_FILES, CM_TO_M_SCALE_FACTOR
)


class ParameterLoader:
    """Unified parameter loader for different robot platforms."""

    def __init__(self, robotname: str):
        """Initialize parameter loader for specified robot."""
        if robotname not in SUPPORTED_ROBOTS:
            raise ValueError(f"Unsupported robot: {robotname}. Supported: {SUPPORTED_ROBOTS}")

        self.robotname = robotname
        self.params_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), robotname, "params")

        # Set up robot-specific configuration
        self.config = {
            "camera_extrinsics_file": CAMERA_EXTRINSICS_FILES[robotname],
            "lidar_extrinsics_file": LIDAR_EXTRINSICS_FILES[robotname],
            "actual_lidar_extrinsics_file": ACTUAL_LIDAR_EXTRINSICS_FILES[robotname]
        }

    def get_cam_intrinsics_dict(self,
                                override_filepath: Optional[str] = None,
                                cam_res: Optional[int] = None,
                                ret_raw: bool = False) -> Dict[str, Any]:
        """
        Load camera intrinsics parameters.

        Args:
            override_filepath: Path to override intrinsics file
            cam_res: Camera resolution
            ret_raw: Whether to return raw camera intrinsics or rectified

        Returns:
            Dictionary containing camera intrinsics
        """
        if override_filepath is None:
            if cam_res is None:
                cam_res = DEFAULT_CAM_RESOLUTIONS[self.robotname]

            pattern = RAW_CAMERA_INTRINSICS_PATTERNS[self.robotname] if ret_raw else CAMERA_INTRINSICS_PATTERNS[self.robotname]
            filename = pattern.format(cam_res=cam_res)
            filepath = os.path.join(self.params_dir, filename)
        else:
            filepath = override_filepath

        with open(filepath, 'r') as f:
            intrinsics_dict = yaml.safe_load(f)

        # Convert to numpy arrays
        intrinsics_dict['camera_matrix'] = np.array(intrinsics_dict['camera_matrix']).reshape((3, 3))
        intrinsics_dict['dist_coeffs'] = np.array(intrinsics_dict['dist_coeffs']).reshape((1, 5)).squeeze()

        # Handle optional parameters
        if 'rvecs' in intrinsics_dict:
            intrinsics_dict['rvecs'] = np.array(intrinsics_dict['rvecs']).reshape((-1, 3))
        if 'tvecs' in intrinsics_dict:
            intrinsics_dict['tvecs'] = np.array(intrinsics_dict['tvecs']).reshape((-1, 3))

        return intrinsics_dict

    def get_cam_extrinsics_dict(self, override_filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load camera extrinsics parameters."""
        if override_filepath is None:
            filepath = os.path.join(self.params_dir, self.config["camera_extrinsics_file"])
        else:
            filepath = override_filepath

        with open(filepath, 'r') as f:
            extrinsics_dict = yaml.safe_load(f)
        return extrinsics_dict

    def get_lidar_extrinsics_dict(self, override_filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load lidar extrinsics parameters."""
        if override_filepath is None:
            filepath = os.path.join(self.params_dir, self.config["lidar_extrinsics_file"])
        else:
            filepath = override_filepath

        with open(filepath, 'r') as f:
            extrinsics_dict = yaml.safe_load(f)
        return extrinsics_dict

    def get_lidar_actual_extrinsics_dict(self, override_filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load actual lidar extrinsics parameters."""
        if override_filepath is None:
            filepath = os.path.join(self.params_dir, self.config["actual_lidar_extrinsics_file"])
        else:
            filepath = override_filepath

        with open(filepath, 'r') as f:
            extrinsics_dict = yaml.safe_load(f)
        return extrinsics_dict

    def compute_cam_extrinsics_transform(self, cam_ext_dict: Dict[str, Any]) -> np.ndarray:
        """
        Compute camera extrinsic transformation matrix.

        Args:
            cam_ext_dict: Camera extrinsics dictionary

        Returns:
            4x4 transformation matrix from WCS to CCS
        """
        # Transforms from WCS to Intermediate frame
        T1 = Homography.get_std_trans(
            cx=cam_ext_dict['T12']['T1']['X'] / 100,
            cy=cam_ext_dict['T12']['T1']['Y'] / 100,
            cz=cam_ext_dict['T12']['T1']['Z'] / 100
        )

        # Transforms from Intermediate frame to CCS
        T2 = Homography.get_std_rot(
            axis=cam_ext_dict['T23']['R1']['axis'],
            alpha=np.deg2rad(cam_ext_dict['T23']['R1']['alpha'])
        )

        # Robot-specific transformation chain
        if self.robotname == "jackal":
            T3 = Homography.get_std_rot(
                axis=cam_ext_dict['T23']['R2']['axis'],
                alpha=np.deg2rad(cam_ext_dict['T23']['R2']['alpha'])
            )
            T4 = np.array(cam_ext_dict['T23']['R3'])
            T5 = Homography.get_std_rot(
                axis=cam_ext_dict['T23']['R4']['axis'],
                alpha=np.deg2rad(cam_ext_dict['T23']['R4']['alpha'])
            )
            return T5 @ T4 @ T3 @ T2 @ T1

        elif self.robotname == "spot":
            T3 = np.array(cam_ext_dict['T23']['R2'])
            T4 = Homography.get_std_rot(
                axis=cam_ext_dict['T23']['R3']['axis'],
                alpha=np.deg2rad(cam_ext_dict['T23']['R3']['alpha'])
            )
            T5 = Homography.get_std_rot(
                axis=cam_ext_dict['T23']['R4']['axis'],
                alpha=np.deg2rad(cam_ext_dict['T23']['R4']['alpha'])
            )
            return T5 @ T4 @ T3 @ T2 @ T1

    def compute_parameterized_cam_extrinsics_transform(self,
                                                       xcm: float,
                                                       ycm: float,
                                                       zcm: float,
                                                       r1deg: float,
                                                       r2deg: float,
                                                       r4deg: float) -> np.ndarray:
        """Compute parameterized camera extrinsic transformation matrix."""
        cam_ext_dict = self.get_cam_extrinsics_dict()
        cam_ext_dict['T12']['T1']['X'] = xcm
        cam_ext_dict['T12']['T1']['Y'] = ycm
        cam_ext_dict['T12']['T1']['Z'] = zcm
        cam_ext_dict['T23']['R1']['alpha'] = r1deg

        if self.robotname == "jackal":
            cam_ext_dict['T23']['R2']['alpha'] = r2deg
        elif self.robotname == "spot":
            cam_ext_dict['T23']['R3']['alpha'] = r2deg

        cam_ext_dict['T23']['R4']['alpha'] = r4deg
        return self.compute_cam_extrinsics_transform(cam_ext_dict)

    def get_camera_parameters(self, cam_ext_dict: Dict[str, Any]) -> OrderedDict:
        """Extract camera transformation parameters."""
        if self.robotname == "jackal":
            return OrderedDict({
                "xcm": cam_ext_dict['T12']['T1']['X'],
                "ycm": cam_ext_dict['T12']['T1']['Y'],
                "zcm": cam_ext_dict['T12']['T1']['Z'],
                "r1deg": cam_ext_dict['T23']['R1']['alpha'],
                "r2deg": cam_ext_dict['T23']['R2']['alpha'],
                "r4deg": cam_ext_dict['T23']['R4']['alpha']
            })
        elif self.robotname == "spot":
            return OrderedDict({
                "xcm": cam_ext_dict['T12']['T1']['X'],
                "ycm": cam_ext_dict['T12']['T1']['Y'],
                "zcm": cam_ext_dict['T12']['T1']['Z'],
                "r1deg": cam_ext_dict['T23']['R1']['alpha'],
                "r3deg": cam_ext_dict['T23']['R3']['alpha'],
                "r4deg": cam_ext_dict['T23']['R4']['alpha']
            })

    def compute_lidar_extrinsics_transform(self, lidar_ext_dict: Dict[str, Any]) -> np.ndarray:
        """
        Compute lidar extrinsic transformation matrix.

        Args:
            lidar_ext_dict: Lidar extrinsics dictionary

        Returns:
            4x4 transformation matrix from WCS to VLP frame
        """
        T1 = Homography.get_std_trans(
            cx=lidar_ext_dict['T1']['Trans1']['X'] / 100,
            cy=lidar_ext_dict['T1']['Trans1']['Y'] / 100,
            cz=lidar_ext_dict['T1']['Trans1']['Z'] / 100
        )
        T2 = Homography.get_std_rot(
            axis=lidar_ext_dict['T2']['Rot1']['axis'],
            alpha=np.deg2rad(lidar_ext_dict['T2']['Rot1']['alpha'])
        )
        T3 = Homography.get_std_rot(
            axis=lidar_ext_dict['T2']['Rot2']['axis'],
            alpha=np.deg2rad(lidar_ext_dict['T2']['Rot2']['alpha'])
        )

        # Robot-specific lidar transformation
        if self.robotname == "jackal":
            T4 = Homography.get_std_rot(
                axis=lidar_ext_dict['T2']['Rot3']['axis'],
                alpha=np.deg2rad(lidar_ext_dict['T2']['Rot3']['alpha'])
            )
            return T4 @ T3 @ T2 @ T1
        elif self.robotname == "spot":
            return (T2 @ T3) @ T1

    def compute_lidar_actual_extrinsics_transform(self, lidar_ext_dict: Dict[str, Any]) -> np.ndarray:
        """
        Compute actual lidar extrinsic transformation matrix.

        Args:
            lidar_ext_dict: Actual lidar extrinsics dictionary

        Returns:
            4x4 transformation matrix from WCS to real VLP frame
        """
        return self.compute_lidar_extrinsics_transform(lidar_ext_dict)


# Convenience functions for backward compatibility
def get_cam_int_dict(override_intrinsics_filepath=None, cam_res=None, ret_raw=False, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.get_cam_intrinsics_dict(override_intrinsics_filepath, cam_res, ret_raw)


def get_cam_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.get_cam_extrinsics_dict(override_extrinsics_filepath)


def compute_cam_ext_T(cam_ext_dict, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.compute_cam_extrinsics_transform(cam_ext_dict)


def compute_parameterized_cam_ext_T(xcm, ycm, zcm, r1deg, r2deg, r4deg, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.compute_parameterized_cam_extrinsics_transform(xcm, ycm, zcm, r1deg, r2deg, r4deg)


def get_parameters_cam_ext_T(cam_ext_dict, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.get_camera_parameters(cam_ext_dict)


def get_lidar_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.get_lidar_extrinsics_dict(override_extrinsics_filepath)


def get_lidar_actual_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.get_lidar_actual_extrinsics_dict(override_extrinsics_filepath)


def compute_lidar_ext_T(lidar_ext_dict, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.compute_lidar_extrinsics_transform(lidar_ext_dict)


def compute_lidar_actual_ext_T(lidar_ext_dict, robotname="jackal"):
    """Backward compatibility function."""
    loader = ParameterLoader(robotname)
    return loader.compute_lidar_actual_extrinsics_transform(lidar_ext_dict)
