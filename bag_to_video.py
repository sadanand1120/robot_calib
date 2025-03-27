import os
import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a ROS bag to an MP4 video.")
    parser.add_argument("--bagfile", required=True, type=str, help="Path to the ROS bag file.")
    parser.add_argument("--output", default=None, type=str, help="Path to the output MP4 video.")
    parser.add_argument("--topic", default="/camera/rgb/image_raw", type=str, help="Image topic in the ROS bag.")
    parser.add_argument("--fps", default=2, type=int, help="Frames per second for the output video.")
    return parser.parse_args()


def bag_to_video(bagfile, output, image_topic, fps):
    # Open the ROS bag file
    bag = rosbag.Bag(bagfile, "r")
    bridge = CvBridge()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(output, fourcc, fps, (w, h), True)
        writer.write(frame)

    # Release resources
    writer.release()
    bag.close()


if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.bagfile)[0] + ".mp4"
    bag_to_video(args.bagfile, args.output, args.topic, args.fps)
