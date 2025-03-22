import cv2
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy
import os
from sensor_msgs.msg import PointCloud2, CompressedImage
import matplotlib.pyplot as plt
import rospy
import time
from cv_bridge import CvBridge
from copy import deepcopy
import yaml

try:
    from spot_calib import SpotCameraCalibration
except ImportError:
    pass
try:
    from .spot_calib import SpotCameraCalibration
except ImportError:
    pass
try:
    from spot_calib.spot_calib import SpotCameraCalibration
except ImportError:
    pass

class SpotLidarCamCalibration:
    COLMAP = [(0, 0, 0.5385), (0, 0, 0.6154),
              (0, 0, 0.6923), (0, 0, 0.7692),
              (0, 0, 0.8462), (0, 0, 0.9231),
              (0, 0, 1.0000), (0, 0.0769, 1.0000),
              (0, 0.1538, 1.0000), (0, 0.2308, 1.0000),
              (0, 0.3846, 1.0000), (0, 0.4615, 1.0000),
              (0, 0.5385, 1.0000), (0, 0.6154, 1.0000),
              (0, 0.6923, 1.0000), (0, 0.7692, 1.0000),
              (0, 0.8462, 1.0000), (0, 0.9231, 1.0000),
              (0, 1.0000, 1.0000), (0.0769, 1.0000, 0.9231),
              (0.1538, 1.0000, 0.8462), (0.2308, 1.0000, 0.7692),
              (0.3077, 1.0000, 0.6923), (0.3846, 1.0000, 0.6154),
              (0.4615, 1.0000, 0.5385), (0.5385, 1.0000, 0.4615),
              (0.6154, 1.0000, 0.3846), (0.6923, 1.0000, 0.3077),
              (0.7692, 1.0000, 0.2308), (0.8462, 1.0000, 0.1538),
              (0.9231, 1.0000, 0.0769), (1.0000, 1.0000, 0),
              (1.0000, 0.9231, 0), (1.0000, 0.8462, 0),
              (1.0000, 0.7692, 0), (1.0000, 0.6923, 0),
              (1.0000, 0.6154, 0), (1.0000, 0.5385, 0),
              (1.0000, 0.4615, 0), (1.0000, 0.3846, 0),
              (1.0000, 0.3077, 0), (1.0000, 0.2308, 0),
              (1.0000, 0.1538, 0), (1.0000, 0.0769, 0),
              (1.0000, 0, 0), (0.9231, 0, 0),
              (0.8462, 0, 0), (0.7692, 0, 0),
              (0.6923, 0, 0), (0.6154, 0, 0)]

    def __init__(self, resolution=1440, ros_flag=True, cam_extrinsics_rel_filepath=None):
        self.spot_cam_calib = SpotCameraCalibration(resolution=resolution, extrinsics_rel_filepath=cam_extrinsics_rel_filepath)
        self.latest_img = None
        self.latest_vlp_points = None
        self.ros_flag = ros_flag
        # NOTE: this is the hypothetical, NOT real spot VLP16 frame
        self.extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_lidar_extrinsics.yaml")
        self.extrinsics_dict = None
        self.load_params()
        if self.ros_flag:
            self.cv_bridge = CvBridge()
            rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1, buff_size=2**32)
            rospy.Subscriber("/corrected_velodyne_points", PointCloud2, self.pc_callback, queue_size=10)
            self.pub = rospy.Publisher("/lidar_cam/compressed", CompressedImage, queue_size=1)
            rospy.Timer(rospy.Duration(1 / 10), self.timer_callback)

    def load_params(self):
        with open(self.extrinsics_filepath, 'r') as f:
            self.extrinsics_dict = yaml.safe_load(f)

    def pc_callback(self, msg):
        pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape((1, -1))
        pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], 3), dtype=np.float32)
        pc_np[..., 0] = pc_cloud['x']
        pc_np[..., 1] = pc_cloud['y']
        pc_np[..., 2] = pc_cloud['z']
        self.latest_vlp_points = pc_np.reshape((-1, 3))

    def image_callback(self, msg):
        self.latest_img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def timer_callback(self, _):
        if self.latest_img is None or self.latest_vlp_points is None:
            return
        cur_img = deepcopy(self.latest_img)
        cur_vlp_points = deepcopy(self.latest_vlp_points)
        pcs_coords, mask, ccs_dists = self.projectVLPtoPCS(cur_vlp_points)
        colors = SpotLidarCamCalibration.get_depth_colors(list(ccs_dists.squeeze()))
        for i in range(pcs_coords.shape[0]):
            cv2.circle(cur_img, tuple(pcs_coords[i, :].astype(np.int32)), radius=2, color=colors[i], thickness=-1)
        img = cv2.resize(cur_img, None, fx=0.75, fy=0.75)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        self.pub.publish(msg)

    @staticmethod
    def get_depth_colors(dists):
        """
        Gives colors for depth values
        dists: list of distances
        Returns: list of colors in BGR format
        """
        COLMAP = SpotLidarCamCalibration.COLMAP
        colors = []
        for i in range(len(dists)):
            range_val = min(round((dists[i] / 30.0) * 49), 49)
            # TODO: do try and catch here, it showed once coz index out of range
            color = (255 * COLMAP[50 - range_val][2], 255 * COLMAP[50 - range_val][1], 255 * COLMAP[50 - range_val][0])
            colors.append(color)
        return colors

    def get_M_ext(self):
        """
        Returns the extrinsic matrix (4 x 4) that transforms from WCS to VLP frame
        """
        T1 = SpotCameraCalibration.get_std_trans(cx=self.extrinsics_dict['T1']['Trans1']['X'] / 100,
                                                 cy=self.extrinsics_dict['T1']['Trans1']['Y'] / 100,
                                                 cz=self.extrinsics_dict['T1']['Trans1']['Z'] / 100)
        T2 = SpotCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T2']['Rot1']['axis'],
                                               alpha=np.deg2rad(self.extrinsics_dict['T2']['Rot1']['alpha']))
        T3 = SpotCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T2']['Rot2']['axis'],
                                               alpha=np.deg2rad(self.extrinsics_dict['T2']['Rot2']['alpha']))
        return (T2 @ T3) @ T1

    def projectVLPtoWCS(self, vlp_points):
        """
        Project VLP points to WCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 3) numpy array of points in WCS
        """
        vlp_points = np.array(vlp_points).astype(np.float64)
        vlp_points_4d = SpotCameraCalibration.get_homo_from_ordinary(vlp_points)
        M_ext = np.linalg.inv(self.get_M_ext())
        wcs_coords_4d = (M_ext @ vlp_points_4d.T).T
        return SpotCameraCalibration.get_ordinary_from_homo(wcs_coords_4d)

    def projectVLPtoPCS(self, vlp_points, mode="skip", ret_zs=False):
        """
        Project VLP points to PCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 2) numpy array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        vlp_points = np.array(vlp_points).astype(np.float64)
        wcs_coords = self.projectVLPtoWCS(vlp_points)
        ccs_coords = self.spot_cam_calib.projectWCStoCCS(wcs_coords)
        pcs_coords, mask = self.spot_cam_calib.projectCCStoPCS(ccs_coords, mode=mode)
        ccs_dists = np.linalg.norm(ccs_coords, axis=1).reshape((-1, 1))
        ccs_dists = ccs_dists[mask]
        if ret_zs:
            ccs_zs = ccs_coords[:, 2].reshape((-1, 1))
            ccs_zs = ccs_zs[mask]
            return pcs_coords, mask, ccs_dists, ccs_zs
        return pcs_coords, mask, ccs_dists


if __name__ == "__main__":
    rospy.init_node('lidar_cam_calib_testing', anonymous=False)
    e = SpotLidarCamCalibration(resolution=1536, ros_flag=True)
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS lidar cam calib testing module")
