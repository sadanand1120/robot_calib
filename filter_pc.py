#!/usr/bin/env python
# Debugging utility: Filters pointcloud and publishes the filtered pc
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np


def filter_pc(msg):
    X_MIN = rospy.get_param("/fxmin", -20.0)
    X_MAX = rospy.get_param("/fxmax", 20.0)
    Y_MIN = rospy.get_param("/fymin", -20.0)
    Y_MAX = rospy.get_param("/fymax", 20.0)
    Z_MIN = rospy.get_param("/fzmin", -20.0)
    Z_MAX = rospy.get_param("/fzmax", 20.0)
    
    gen = pc2.read_points(msg, field_names=None, skip_nans=True)
    pc_np = np.array(list(gen))
    lidar_coords = np.array([row for row in pc_np if X_MIN <= row[0] <= X_MAX and Y_MIN <= row[1] <= Y_MAX and Z_MIN <= row[2] <= Z_MAX])
    lidar_coords_list = [tuple(int(el) if idx == len(row) - 2 else el for idx, el in enumerate(row)) for row in lidar_coords.tolist()]
    lidar_cloud = pc2.create_cloud(msg.header, msg.fields, lidar_coords_list)
    lidar_cloud.is_dense = True
    pub.publish(lidar_cloud)


if __name__ == "__main__":
    rospy.init_node('pointcloud_filter', anonymous=False)
    rospy.Subscriber("/corrected_velodyne_points", PointCloud2, filter_pc, queue_size=1)
    pub = rospy.Publisher('/filter_velodyne_points', PointCloud2, queue_size=1)
    rospy.spin()
