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

    def get_lidar_extrinsics_dict(self,
                                  override_filepath: Optional[str] = None,
                                  use_actual: bool = False) -> Dict[str, Any]:
        """
        Load lidar extrinsics parameters.

        Args:
            override_filepath: Path to override extrinsics file
            use_actual: Whether to load actual lidar extrinsics (True) or ideal lidar extrinsics (False)

        Returns:
            Dictionary containing lidar extrinsics
        """
        if override_filepath is None:
            config_key = "actual_lidar_extrinsics_file" if use_actual else "lidar_extrinsics_file"
            filepath = os.path.join(self.params_dir, self.config[config_key])
        else:
            filepath = override_filepath

        with open(filepath, 'r') as f:
            extrinsics_dict = yaml.safe_load(f)
        return extrinsics_dict

    def compute_cam_extrinsics_transform(self,
                                         cam_ext_dict: Optional[Dict[str, Any]] = None,
                                         override_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Compute camera extrinsic transformation matrix.

        Args:
            cam_ext_dict: Camera extrinsics dictionary (loaded if None)
            override_params: Dict with keys like 'xcm', 'ycm', 'zcm', 'r1deg', 'r2deg', 'r4deg' to override parameters

        Returns:
            4x4 transformation matrix from WCS to CCS
        """
        # Load default if not provided
        if cam_ext_dict is None:
            cam_ext_dict = self.get_cam_extrinsics_dict()
        else:
            cam_ext_dict = deepcopy(cam_ext_dict)  # Don't modify original

        # Apply parameter overrides if provided
        if override_params:
            if 'xcm' in override_params:
                cam_ext_dict['T12']['T1']['X'] = override_params['xcm']
            if 'ycm' in override_params:
                cam_ext_dict['T12']['T1']['Y'] = override_params['ycm']
            if 'zcm' in override_params:
                cam_ext_dict['T12']['T1']['Z'] = override_params['zcm']
            if 'r1deg' in override_params:
                cam_ext_dict['T23']['R1']['alpha'] = override_params['r1deg']
            if 'r4deg' in override_params:
                cam_ext_dict['T23']['R4']['alpha'] = override_params['r4deg']

            # Robot-specific parameter mapping
            if self.robotname == "jackal" and 'r2deg' in override_params:
                cam_ext_dict['T23']['R2']['alpha'] = override_params['r2deg']
            elif self.robotname == "spot" and 'r2deg' in override_params:
                cam_ext_dict['T23']['R3']['alpha'] = override_params['r2deg']

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

    def compute_lidar_extrinsics_transform(self,
                                           lidar_ext_dict: Optional[Dict[str, Any]] = None,
                                           use_actual: bool = False) -> np.ndarray:
        """
        Compute lidar extrinsic transformation matrix.

        Args:
            lidar_ext_dict: Lidar extrinsics dictionary (loaded if None)
            use_actual: Whether to use actual lidar extrinsics (True) or ideal lidar extrinsics (False)

        Returns:
            4x4 transformation matrix from WCS to VLP frame
        """
        # Load default if not provided
        if lidar_ext_dict is None:
            lidar_ext_dict = self.get_lidar_extrinsics_dict(use_actual=use_actual)

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
