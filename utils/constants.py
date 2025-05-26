"""
Common constants used throughout the robot calibration codebase.
Centralizes magic numbers and configuration values.
"""

# Default camera resolutions for different robots
DEFAULT_CAM_RESOLUTIONS = {
    "jackal": 540,
    "spot": 1536
}

# Image processing constants
DEFAULT_ALPHA_RECTIFICATION = 0.0  # Crop borders during rectification
MAX_ALPHA_RECTIFICATION = 1.0      # Preserve all pixels during rectification

# Coordinate transformation constants
CM_TO_M_SCALE_FACTOR = 100          # Convert centimeters to meters
EPSILON_DIVISION_SAFETY = 1e-10     # Prevent division by zero in homogeneous coordinates

# OpenCV constants
DEFAULT_INTERPOLATION = "INTER_LINEAR"
DEFAULT_CV_MAP_TYPE = "CV_16SC2"

# Field of view constraint modes
FOV_MODES = {
    "SKIP": "skip",      # Remove out-of-bounds pixels
    "CLIP": "clip",      # Clamp pixel values to bounds
    "NONE": "none"       # No constraints
}

# Rotation axes
ROTATION_AXES = ["X", "Y", "Z"]

# Default point visualization parameters
DEFAULT_POINT_RADIUS = 4
DEFAULT_POINT_THICKNESS = -1  # Filled circle

# Default visualization resize factors
DEFAULT_RESIZE_FACTOR = 0.75

# Depth estimation constants
DEFAULT_DEPTH_MODEL_INPUT_SIZE = 518
DEFAULT_MAX_DEPTH_INDOOR = 20
DEFAULT_MAX_DEPTH_OUTDOOR = 80

# Supported depth encoders
DEPTH_ENCODERS = ["vits", "vitb", "vitl", "vitg"]

# Supported depth datasets
DEPTH_DATASETS = ["hypersim", "vkitti"]

# Default video processing parameters
DEFAULT_VIDEO_FPS = 2

# Common file extensions
YAML_EXTENSION = ".yaml"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
POINTCLOUD_EXTENSIONS = [".pcd", ".ply"]

# Default ROS topic names (can be overridden)
DEFAULT_IMAGE_TOPIC = "/camera/rgb/image_raw"
DEFAULT_POINTCLOUD_TOPIC = "/velodyne_points"
DEFAULT_COMPRESSED_IMAGE_TOPIC_SUFFIX = "/compressed"

# Parameter file naming patterns
CAMERA_INTRINSICS_PATTERNS = {
    "jackal": "zed_left_rect_intrinsics_{cam_res}.yaml",
    "spot": "cam_intrinsics_{cam_res}.yaml"
}

RAW_CAMERA_INTRINSICS_PATTERNS = {
    "jackal": "raw/zed_left_rect_intrinsics_{cam_res}.yaml",
    "spot": "raw/cam_intrinsics_{cam_res}.yaml"
}

CAMERA_EXTRINSICS_FILES = {
    "jackal": "baselink_to_zed_left_extrinsics.yaml",
    "spot": "baselink_to_cam_extrinsics.yaml"
}

LIDAR_EXTRINSICS_FILES = {
    "jackal": "baselink_to_lidar_extrinsics.yaml",
    "spot": "baselink_to_lidar_extrinsics.yaml"
}

ACTUAL_LIDAR_EXTRINSICS_FILES = {
    "jackal": "baselink_to_actual_lidar_extrinsics.yaml",
    "spot": "baselink_to_actual_lidar_extrinsics.yaml"
}

# Supported robots
SUPPORTED_ROBOTS = ["jackal", "spot"]
