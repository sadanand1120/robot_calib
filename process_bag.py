import os
import sys
import cv2
import argparse
import numpy as np
import rosbag
from tqdm import tqdm
from cv_bridge import CvBridge
np.float = np.float64  # ros_numpy fix
import ros_numpy


def extract_images(bagfile, img_dir, image_topic):
    os.makedirs(img_dir, exist_ok=True)
    bridge = CvBridge()
    with rosbag.Bag(bagfile, "r") as bag:
        total = sum(1 for _ in bag.read_messages(topics=[image_topic]))
    with rosbag.Bag(bagfile, "r") as bag:
        for _, msg, _ in tqdm(bag.read_messages(topics=[image_topic]), total=total, desc="Saving images"):
            ts = msg.header.stamp.to_nsec()
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(img_dir, f"{ts}.png"), img)


def extract_pcs(bagfile, pc_dir, pc_topic):
    os.makedirs(pc_dir, exist_ok=True)
    with rosbag.Bag(bagfile, "r") as bag:
        total = sum(1 for _ in bag.read_messages(topics=[pc_topic]))
    with rosbag.Bag(bagfile, "r") as bag:
        for _, msg, _ in tqdm(bag.read_messages(topics=[pc_topic]), total=total, desc="Saving pointclouds"):
            ts = msg.header.stamp.to_nsec()
            pc = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape(-1)
            arr = np.column_stack((pc['x'], pc['y'], pc['z'])).astype(np.float32)
            arr.tofile(os.path.join(pc_dir, f"{ts}.bin"))


def extract_video(bagfile, video_path, image_topic, fps):
    bridge = CvBridge()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    with rosbag.Bag(bagfile, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[image_topic]):
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h), True)
            writer.write(frame)
    if writer:
        writer.release()
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images, pointclouds, and/or video from a ROS bag")
    parser.add_argument("--bagfile", required=True, help="Path to ROS bag file")
    parser.add_argument("--rootdir", default=None, help="Root directory to store outputs")
    parser.add_argument("--images", action="store_true", help="Extract images")
    parser.add_argument("--pcs", action="store_true", help="Extract pointclouds")
    parser.add_argument("--video", action="store_true", help="Extract video")
    parser.add_argument("--imgtopic", default="/camera/rgb/image_raw", help="Image topic")
    parser.add_argument("--pctopic", default="/velodyne_points", help="Pointcloud topic")
    parser.add_argument("--fps", type=int, default=2, help="FPS for video output")
    args = parser.parse_args()

    bagname = os.path.splitext(os.path.basename(args.bagfile))[0]
    if args.rootdir is None:
        args.rootdir = os.path.join(os.path.dirname(args.bagfile), bagname)

    if args.images:
        extract_images(args.bagfile, os.path.join(args.rootdir, "images"), args.imgtopic)

    if args.pcs:
        extract_pcs(args.bagfile, os.path.join(args.rootdir, "pcs"), args.pctopic)

    if args.video:
        extract_video(args.bagfile, os.path.join(args.rootdir, f"{bagname}.mp4"), args.imgtopic, args.fps)
