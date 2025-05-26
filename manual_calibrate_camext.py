"""
Script to manually tweak the camera extrinsics parameters using a static (i.e., robot not moving) bag file.
"""

from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import rosbag
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy  # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
np.set_printoptions(precision=4, suppress=True)
from copy import deepcopy
import json
import yaml
from lidar_cam_calib import LidarCamCalib
from pprint import pprint

# **************************************************************
# Parameters
STATIC_BAGFILE = "/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/1536_board.bag"
ROBOTNAME = "spot"
IMG_RES = 1536
LCC = LidarCamCalib(ros_flag=False, robotname=ROBOTNAME, cam_res=IMG_RES)
# actual bounds are -100 to 100 cms, and -180 to 180 degrees
# cv2 bounds are 0 to 200, and 0 to 360 degrees
# **************************************************************


def actual_cm_to_cv2_val(val):
    return int(val + 100)


def cv2_val_to_actual_cm(val):
    return float(val - 100)


def actual_deg_to_cv2_val(val):
    actual_neg180_180 = (val + 180) % 360 - 180
    return int(actual_neg180_180 + 180)


def cv2_val_to_actual_deg(val):
    return float(val - 180)


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
            pc = np.column_stack((pc['x'], pc['y'], pc['z'])).astype(np.float32).reshape((-1, 3))
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
    print(pc.shape)
    pc.tofile(binpath)  # can do this directly without flattening, tofile() will flatten it


def update_overlay(cv2img, pc_bin, lcc: LidarCamCalib):
    img = deepcopy(cv2img)
    T0 = cv2_val_to_actual_cm(cv2.getTrackbarPos('T0', 'MyImage'))
    T1 = cv2_val_to_actual_cm(cv2.getTrackbarPos('T1', 'MyImage'))
    T2 = cv2_val_to_actual_cm(cv2.getTrackbarPos('T2', 'MyImage'))
    R0 = cv2_val_to_actual_deg(cv2.getTrackbarPos('R0', 'MyImage'))
    R1 = cv2_val_to_actual_deg(cv2.getTrackbarPos('R1', 'MyImage'))
    R2 = cv2_val_to_actual_deg(cv2.getTrackbarPos('R2', 'MyImage'))
    override_params = {'xcm': T0, 'ycm': T1, 'zcm': T2, 'r1deg': R0, 'r2deg': R1, 'r4deg': R2}
    cam_M_ext = param_loader.compute_cam_extrinsics_transform(override_params=override_params)
    lcc.cam_calib.M_ext = cam_M_ext
    img_overlay, *_ = lcc.projectPCtoImage(pc_bin, img)
    img_overlay = cv2.resize(img_overlay, (1280, 720))
    cv2.imshow('MyImage', img_overlay)


with open(f"{ROBOTNAME}/info.json", "r") as f:
    ROBOTINFO_DICT = json.load(f)

from utils.param_loader import ParameterLoader
param_loader = ParameterLoader(ROBOTNAME)

cv2_img, pc_np_xyz = get_first_image_and_pc(STATIC_BAGFILE, image_topic=ROBOTINFO_DICT["image_topic"], pc_topic=ROBOTINFO_DICT["lidar_topic"])
cv2_img = LCC.cam_calib.rectifyRawCamImage(cv2_img)
pc_np_xyz = LCC._correct_pc(pc_np_xyz)
parameters_dict = param_loader.get_camera_parameters(LCC.cam_calib.extrinsics_dict)
parameters_keys = list(parameters_dict.keys())
print("Initial parameters:")
pprint(parameters_dict)

cv2.namedWindow('MyImage')
cv2.createTrackbar('T0', 'MyImage', actual_cm_to_cv2_val(parameters_dict[parameters_keys[0]]), 200, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
cv2.createTrackbar('T1', 'MyImage', actual_cm_to_cv2_val(parameters_dict[parameters_keys[1]]), 200, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
cv2.createTrackbar('T2', 'MyImage', actual_cm_to_cv2_val(parameters_dict[parameters_keys[2]]), 200, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
cv2.createTrackbar('R0', 'MyImage', actual_deg_to_cv2_val(parameters_dict[parameters_keys[3]]), 360, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
cv2.createTrackbar('R1', 'MyImage', actual_deg_to_cv2_val(parameters_dict[parameters_keys[4]]), 360, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
cv2.createTrackbar('R2', 'MyImage', actual_deg_to_cv2_val(parameters_dict[parameters_keys[5]]), 360, lambda x: update_overlay(cv2_img, pc_np_xyz, LCC))
update_overlay(cv2_img, pc_np_xyz, LCC)
cv2.waitKey(0)

parameters_dict[parameters_keys[0]] = cv2_val_to_actual_cm(cv2.getTrackbarPos('T0', 'MyImage'))
parameters_dict[parameters_keys[1]] = cv2_val_to_actual_cm(cv2.getTrackbarPos('T1', 'MyImage'))
parameters_dict[parameters_keys[2]] = cv2_val_to_actual_cm(cv2.getTrackbarPos('T2', 'MyImage'))
parameters_dict[parameters_keys[3]] = cv2_val_to_actual_deg(cv2.getTrackbarPos('R0', 'MyImage'))
parameters_dict[parameters_keys[4]] = cv2_val_to_actual_deg(cv2.getTrackbarPos('R1', 'MyImage'))
parameters_dict[parameters_keys[5]] = cv2_val_to_actual_deg(cv2.getTrackbarPos('R2', 'MyImage'))
print("Updated parameters: *NEED TO STORE MANUALLY IF YOU WANT TO KEEP THEM*")
pprint(parameters_dict)

cv2.destroyAllWindows()
