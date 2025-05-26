"""
Centralized robot parameter management.
Provides backward compatibility while using the unified parameter loader.
"""

from utils.param_loader import ParameterLoader


class RobotParameters:
    """Unified robot parameter management interface."""

    _loaders = {}  # Cache loaders to avoid repeated instantiation

    @classmethod
    def get_loader(cls, robotname: str) -> ParameterLoader:
        """Get or create parameter loader for specified robot."""
        if robotname not in cls._loaders:
            cls._loaders[robotname] = ParameterLoader(robotname)
        return cls._loaders[robotname]


# Convenience functions that match the old API exactly
def get_cam_int_dict(override_intrinsics_filepath=None, cam_res=None, ret_raw=False, robotname="jackal"):
    """Get camera intrinsics dictionary."""
    loader = RobotParameters.get_loader(robotname)
    return loader.get_cam_intrinsics_dict(override_intrinsics_filepath, cam_res, ret_raw)


def get_cam_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Get camera extrinsics dictionary."""
    loader = RobotParameters.get_loader(robotname)
    return loader.get_cam_extrinsics_dict(override_extrinsics_filepath)


def compute_cam_ext_T(cam_ext_dict, robotname="jackal"):
    """Compute camera extrinsic transformation matrix."""
    loader = RobotParameters.get_loader(robotname)
    return loader.compute_cam_extrinsics_transform(cam_ext_dict)


def compute_parameterized_cam_ext_T(xcm, ycm, zcm, r1deg, r2deg, r4deg, robotname="jackal"):
    """Compute parameterized camera extrinsic transformation matrix."""
    loader = RobotParameters.get_loader(robotname)
    return loader.compute_parameterized_cam_extrinsics_transform(xcm, ycm, zcm, r1deg, r2deg, r4deg)


def get_parameters_cam_ext_T(cam_ext_dict, robotname="jackal"):
    """Extract camera transformation parameters."""
    loader = RobotParameters.get_loader(robotname)
    return loader.get_camera_parameters(cam_ext_dict)


def get_lidar_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Get lidar extrinsics dictionary."""
    loader = RobotParameters.get_loader(robotname)
    return loader.get_lidar_extrinsics_dict(override_extrinsics_filepath)


def get_lidar_actual_ext_dict(override_extrinsics_filepath=None, robotname="jackal"):
    """Get actual lidar extrinsics dictionary."""
    loader = RobotParameters.get_loader(robotname)
    return loader.get_lidar_actual_extrinsics_dict(override_extrinsics_filepath)


def compute_lidar_ext_T(lidar_ext_dict, robotname="jackal"):
    """Compute lidar extrinsic transformation matrix."""
    loader = RobotParameters.get_loader(robotname)
    return loader.compute_lidar_extrinsics_transform(lidar_ext_dict)


def compute_lidar_actual_ext_T(lidar_ext_dict, robotname="jackal"):
    """Compute actual lidar extrinsic transformation matrix."""
    loader = RobotParameters.get_loader(robotname)
    return loader.compute_lidar_actual_extrinsics_transform(lidar_ext_dict)
