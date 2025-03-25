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
import argparse
from tqdm import tqdm


def save_all_images_and_pcs(bag_filepath, rootdir, image_topic='/camera/rgb/image_raw', pc_topic='/velodyne_points'):
    img_dir, bin_dir = os.path.join(rootdir, "images"), os.path.join(rootdir, "pcs")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    bridge = CvBridge()

    # First pass: count messages for tqdm
    with rosbag.Bag(bag_filepath, 'r') as bag:
        total_msgs = sum(1 for _ in bag.read_messages(topics=[image_topic, pc_topic]))

    # Second pass: process messages
    with rosbag.Bag(bag_filepath, 'r') as bag:
        for topic, msg, _ in tqdm(bag.read_messages(topics=[image_topic, pc_topic]), total=total_msgs, desc="Saving data", unit="msg"):
            ts = msg.header.stamp.to_nsec()
            if topic == image_topic:
                img = bridge.imgmsg_to_cv2(msg, "passthrough")
                cv2.imwrite(os.path.join(img_dir, f"{ts}.png"), img)
            elif topic == pc_topic:
                pc = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape(-1)
                pc_arr = np.column_stack((pc['x'], pc['y'], pc['z'])).astype(np.float32).reshape((-1, 3))
                pc_arr.tofile(os.path.join(bin_dir, f"{ts}.bin"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--bagfile", type=str, required=True, help="Path to the rosbag file")
    arg_parser.add_argument("--rootdir", type=str, default=None, help="Directory to save images and pcs; default is same as bagfile")
    arg_parser.add_argument("--imgtopic", type=str, default='/camera/rgb/image_raw', help="Image topic name")
    arg_parser.add_argument("--pctopic", type=str, default='/velodyne_points', help="Pointcloud topic name")
    args = arg_parser.parse_args()
    if args.rootdir is None:
        args.rootdir = os.path.join(os.path.dirname(args.bagfile), os.path.basename(args.bagfile).split(".")[0])
    save_all_images_and_pcs(args.bagfile, args.rootdir, args.imgtopic, args.pctopic)
