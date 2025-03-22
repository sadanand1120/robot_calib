from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import math
import rosbag
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy  # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
np.set_printoptions(precision=4, suppress=True)
from copy import deepcopy
import sys
import yaml
import matplotlib.pyplot as plt
from correct_pc_publisher import get_M_ext as get_M_ext_actual_lidar

try:
    from spot_calib import SpotCameraCalibration
    from lidar_cam_calib import SpotLidarCamCalibration
except ImportError:
    pass
try:
    from .spot_calib import SpotCameraCalibration
    from .lidar_cam_calib import SpotLidarCamCalibration
except ImportError:
    pass
try:
    from spot_calib.spot_calib import SpotCameraCalibration
    from spot_calib.lidar_cam_calib import SpotLidarCamCalibration
except ImportError:
    pass


def get_first_image_and_pc(bag_filepath, image_topic='/camera/rgb/image_raw', pc_topic='/velodyne_points', skip_n=5):
    bridge, bag = CvBridge(), rosbag.Bag(bag_filepath, 'r')
    img_saved, pc_saved = False, False
    for topic, msg, _ in bag.read_messages(topics=[image_topic, pc_topic]):
        if skip_n > 0:
            skip_n -= 1
            continue
        if topic == image_topic and not img_saved:
            img = bridge.imgmsg_to_cv2(msg, "passthrough")
            img_saved = True
        elif topic == pc_topic and not pc_saved:
            pc = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape(-1)
            pc = np.column_stack((pc['x'], pc['y'], pc['z'], pc['intensity'])).astype(np.float32).reshape((-1, 4))
            pc_saved = True
        if img_saved and pc_saved:
            break
    bag.close()
    return img, pc


def save_first_image_and_pc(bag_filepath, imgpath, binpath, image_topic='/camera/rgb/image_raw', pc_topic='/velodyne_points'):
    os.makedirs(os.path.dirname(imgpath), exist_ok=True)
    os.makedirs(os.path.dirname(binpath), exist_ok=True)
    img, pc = get_first_image_and_pc(bag_filepath, image_topic, pc_topic)
    cv2.imwrite(imgpath, img)
    pc.tofile(binpath)


def transform_matrix_from_6dof(x, y, z, roll, pitch, yaw):
    """
    Compute a 4x4 transformation matrix from 6-DOF parameters.

    Args:
        x, y, z: Translation components
        roll, pitch, yaw: Rotation components (in radians)

    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    # Compute rotation matrices for roll, pitch, and yaw
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Compute the full rotation matrix
    R = Rz @ Ry @ Rx  # Equivalent to intrinsic rotation ZYX

    # Construct the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def extract_6dof_from_transform(T):
    """
    Extract 6-DOF parameters (x, y, z, roll, pitch, yaw) from a 4x4 transformation matrix.

    Args:
        T: 4x4 numpy array representing the transformation matrix.

    Returns:
        x, y, z: Translation components
        roll, pitch, yaw: Rotation components (in radians)
    """
    # Extract translation
    x, y, z = T[:3, 3]

    # Extract rotation matrix
    R = T[:3, :3]

    # Extract pitch angle (handle singularities)
    pitch = np.arcsin(-R[2, 0])

    if np.abs(R[2, 0]) < 1 - 1e-6:  # Not at singularity
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # Gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return x, y, z, roll, pitch, yaw


def put_lidar_points_on_img(_cv2_img, pc_np_xyz, camobj: SpotCameraCalibration, M_lidar_to_kinect):
    cv2_img = deepcopy(_cv2_img)
    vlp_points = np.array(pc_np_xyz).astype(np.float64)
    vlp_points_4d = SpotCameraCalibration.get_homo_from_ordinary(vlp_points)
    ccs_coords_4d = (M_lidar_to_kinect @ vlp_points_4d.T).T
    ccs_coords = SpotCameraCalibration.get_ordinary_from_homo(ccs_coords_4d)
    pcs_coords, mask = camobj.projectCCStoPCS(ccs_coords)
    ccs_zs = ccs_coords[:, 2].reshape((-1, 1))
    ccs_zs = ccs_zs[mask]

    if len(ccs_zs) == 0:
        return cv2_img
    # Normalize depth values to [0, 1] for colormap
    ccs_zs_norm = (ccs_zs - np.min(ccs_zs)) / (np.max(ccs_zs) - np.min(ccs_zs) + 1e-6)

    # Apply colormap (viridis)
    cmap = plt.get_cmap("viridis")
    colors = (cmap(ccs_zs_norm.squeeze())[:, :3] * 255).astype(np.uint8)  # Convert to 0-255

    # Ensure colors are proper BGR integer tuples
    colors_bgr = [tuple(map(int, color[::-1])) for color in colors]  # Reverse RGB to BGR

    # Draw points on image
    h, w = cv2_img.shape[:2]  # Get image dimensions for bounds checking
    for i in range(pcs_coords.shape[0]):
        px, py = pcs_coords[i, :].astype(int)
        if 0 <= px < w and 0 <= py < h:  # Ensure points are inside the image
            cv2.circle(cv2_img, (px, py), radius=4, color=colors_bgr[i], thickness=-1)

    # cv2_img = cv2.resize(cv2_img, None, fx=0.75, fy=0.75)
    # cv2_img = cv2.resize(cv2_img, (1280, 720))
    return cv2_img


def correct_pc_lidar(pc_np_xyz):
    c = SpotLidarCamCalibration(ros_flag=False, resolution=IMG_RES)
    real_lidar_to_wcs = np.linalg.inv(get_M_ext_actual_lidar())
    wcs_to_lidar = c.get_M_ext()

    # lidar_from_real_lidar = lidar_from_wcs @ wcs_from_real_lidar
    real_lidar_to_lidar = wcs_to_lidar @ real_lidar_to_wcs

    pc_np_xyz_4d = SpotCameraCalibration.get_homo_from_ordinary(pc_np_xyz)
    lidar_coords_xyz_4d = (real_lidar_to_lidar @ pc_np_xyz_4d.T).T
    lidar_coords_xyz = SpotCameraCalibration.get_ordinary_from_homo(lidar_coords_xyz_4d)

    return lidar_coords_xyz


def update_overlay(cv2img, pc_np_xyz, camcalib_obj):
    global PARAMS_UPD
    """ Callback function to update the overlay when trackbars are adjusted. """
    x = (cv2.getTrackbarPos('X', 'MyImage') - 50.0) / 100.0
    y = (cv2.getTrackbarPos('Y', 'MyImage') - 50.0) / 100.0
    z = (cv2.getTrackbarPos('Z', 'MyImage') - 50.0) / 100.0
    roll = (cv2.getTrackbarPos('Roll', 'MyImage') - 180) * np.pi / 180
    pitch = (cv2.getTrackbarPos('Pitch', 'MyImage') - 180) * np.pi / 180
    yaw = (cv2.getTrackbarPos('Yaw', 'MyImage') - 180) * np.pi / 180
    PARAMS_UPD = (x, y, z, roll, pitch, yaw)

    M_lidar_to_kinect_upd = transform_matrix_from_6dof(x, y, z, roll, pitch, yaw)
    img_overlay = put_lidar_points_on_img(cv2img, pc_np_xyz, camcalib_obj, M_lidar_to_kinect_upd)
    img_overlay = cv2.resize(img_overlay, (1280, 720))
    cv2.imshow('MyImage', img_overlay)


if __name__ == "__main__":
    PARAMS_UPD = None
    IMG_RES = 1536
    cam_calib_obj = SpotCameraCalibration(resolution=IMG_RES)
    cv2_img, pc_bin = get_first_image_and_pc("/home/dynamo/Music/metric_depthany2_calib/1536_board.bag")
    pc_bin = pc_bin[:, :3]
    pc_bin = correct_pc_lidar(pc_bin)

    if not os.path.exists("params/lidar_to_kinect.yaml"):
        lcc = SpotLidarCamCalibration(ros_flag=False, resolution=IMG_RES)
        M_baselink_to_kinect = lcc.spot_cam_calib.get_M_ext()
        M_baselink_to_lidar = lcc.get_M_ext()
        M_kinect_to_lidar = M_baselink_to_lidar @ np.linalg.inv(M_baselink_to_kinect)
        M_lidar_to_kinect = np.linalg.inv(M_kinect_to_lidar)
        params_def = extract_6dof_from_transform(M_lidar_to_kinect)
        x_def, y_def, z_def, roll_def, pitch_def, yaw_def = params_def
        with open("params/lidar_to_kinect.yaml", "w") as f:
            f.write(f"lidar_to_kinect:\n  x: {x_def}\n  y: {y_def}\n  z: {z_def}\n  roll: {roll_def}\n  pitch: {pitch_def}\n  yaw: {yaw_def}\n")

    with open("params/lidar_to_kinect.yaml", "r") as f:
        params_dict = yaml.safe_load(f)
        x_def = params_dict["lidar_to_kinect"]["x"]
        y_def = params_dict["lidar_to_kinect"]["y"]
        z_def = params_dict["lidar_to_kinect"]["z"]
        roll_def = params_dict["lidar_to_kinect"]["roll"]
        pitch_def = params_dict["lidar_to_kinect"]["pitch"]
        yaw_def = params_dict["lidar_to_kinect"]["yaw"]
    PARAMS_UPD = (x_def, y_def, z_def, roll_def, pitch_def, yaw_def)
    roll_def_deg, pitch_def_deg, yaw_def_deg = np.degrees([roll_def, pitch_def, yaw_def])
    print(f"Default 6-DOF parameters: x={x_def}, y={y_def}, z={z_def}, roll={roll_def_deg}, pitch={pitch_def_deg}, yaw={yaw_def_deg}")

    cv2.namedWindow('MyImage')
    cv2.createTrackbar('X', 'MyImage', int(x_def * 100 + 50), 200, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))  # origin centered at 50 cm
    cv2.createTrackbar('Y', 'MyImage', int(y_def * 100 + 50), 200, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))  # origin centered at 50 cm
    cv2.createTrackbar('Z', 'MyImage', int(z_def * 100 + 50), 200, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))  # origin centered at 50 cm
    cv2.createTrackbar('Roll', 'MyImage', int(roll_def_deg + 180), 540, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))   # origin centered at 180 deg
    cv2.createTrackbar('Pitch', 'MyImage', int(pitch_def_deg + 180), 540, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))  # origin centered at 180 deg
    cv2.createTrackbar('Yaw', 'MyImage', int(yaw_def_deg + 180), 540, lambda x: update_overlay(cv2_img, pc_bin, cam_calib_obj))   # origin centered at 180 deg

    update_overlay(cv2_img, pc_bin, cam_calib_obj)
    cv2.waitKey(0)

    # store in yaml
    x, y, z, roll, pitch, yaw = PARAMS_UPD
    print(f"Updated 6-DOF parameters: x={x}, y={y}, z={z}, roll={np.degrees(roll)}, pitch={np.degrees(pitch)}, yaw={np.degrees(yaw)}")
    with open("params/lidar_to_kinect.yaml", "w") as f:
        f.write(f"lidar_to_kinect:\n  x: {x}\n  y: {y}\n  z: {z}\n  roll: {roll}\n  pitch: {pitch}\n  yaw: {yaw}\n")

    cv2.destroyAllWindows()
