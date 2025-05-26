import cv2
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy
import os
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs.msg import CameraInfo
import rospy
import time
import json
from cv_bridge import CvBridge
from copy import deepcopy
from scipy.interpolate import griddata
from cam_calib import CamCalib
from homography import Homography


class LidarCamCalib:
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

    def __init__(self, ros_flag=True, override_cam_intrinsics_filepath=None, override_cam_extrinsics_filepath=None, robotname="jackal", override_lidar_extrinsics_filepath=None, override_lidar_actual_extrinsics_filepath=None, cam_res=None):
        self.ros_flag = ros_flag
        self.cam_calib = CamCalib(override_intrinsics_filepath=override_cam_intrinsics_filepath, override_extrinsics_filepath=override_cam_extrinsics_filepath, robotname=robotname, cam_res=cam_res)
        self.robotname = robotname
        self.override_lidar_extrinsics_filepath = override_lidar_extrinsics_filepath
        self.override_lidar_actual_extrinsics_filepath = override_lidar_actual_extrinsics_filepath
        self.load_params()
        if self.ros_flag:
            self.setup_ros()

    def setup_ros(self):
        self.latest_img = None
        self.latest_vlp_points = None
        self.cv_bridge = CvBridge()
        rospy.Subscriber(f"{self.info['image_topic']}/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber(self.info['lidar_topic'], PointCloud2, self.pc_callback)
        self.overlay_pub = rospy.Publisher("/lidar_cam/compressed", CompressedImage, queue_size=1)
        self.corrected_pc_pub = rospy.Publisher(f"{self.info['lidar_topic']}/corrected", PointCloud2, queue_size=1)
        self.caminfo_pub = rospy.Publisher("/lidar_cam/camera_info", CameraInfo, queue_size=1)
        rospy.Timer(rospy.Duration(1 / 10), lambda event: self.main(self.latest_img, self.latest_vlp_points))

    def image_callback(self, msg):
        latest_img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_img = self.cam_calib.rectifyRawCamImage(latest_img)

        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = rospy.Time.now()
        camera_info_msg.header.frame_id = msg.header.frame_id
        camera_info_msg.height = self.img_height
        camera_info_msg.width = self.img_width
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.D = list(self.cam_calib.intrinsics_dict['dist_coeffs'].reshape((1, 5)).squeeze())  # Distortion parameters (D)
        camera_info_msg.K = list(self.cam_calib.intrinsics_dict['camera_matrix'].reshape((1, 9)).squeeze())  # Intrinsic camera matrix (K)
        camera_info_msg.R = list(np.eye(3).reshape((1, 9)).squeeze())   # Rectification matrix (R)
        P = np.hstack((self.cam_calib.intrinsics_dict['camera_matrix'], np.zeros((3, 1))))  # Projection/camera matrix (P): For monocular cameras, P is [fx 0 cx 0, 0 fy cy 0, 0 0 1 0]
        camera_info_msg.P = list(P.reshape((1, 12)).squeeze())
        self.caminfo_pub.publish(camera_info_msg)

    def pc_callback(self, msg):
        pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape((1, -1))
        pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], 3), dtype=np.float32)
        pc_np[..., 0] = pc_cloud['x']
        pc_np[..., 1] = pc_cloud['y']
        pc_np[..., 2] = pc_cloud['z']
        latest_vlp_points = pc_np.reshape((-1, 3))
        self.latest_vlp_points = self._correct_pc(latest_vlp_points)

        corrected_pc_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            np.array(
                list(zip(
                    self.latest_vlp_points[:, 0],
                    self.latest_vlp_points[:, 1],
                    self.latest_vlp_points[:, 2]
                )),
                dtype=[
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32)
                ]
            ),
            stamp=msg.header.stamp,
            frame_id=msg.header.frame_id
        )
        self.corrected_pc_pub.publish(corrected_pc_msg)

    def load_params(self):
        from utils.param_loader import ParameterLoader
        param_loader = ParameterLoader(self.robotname)

        # Use the unified lidar extrinsics loading
        self.extrinsics_dict = param_loader.get_lidar_extrinsics_dict(self.override_lidar_extrinsics_filepath, use_actual=False)
        self.actual_extrinsics_dict = param_loader.get_lidar_extrinsics_dict(self.override_lidar_actual_extrinsics_filepath, use_actual=True)
        self.M_ext = param_loader.compute_lidar_extrinsics_transform(self.extrinsics_dict, use_actual=False)
        self.actual_M_ext = param_loader.compute_lidar_extrinsics_transform(self.actual_extrinsics_dict, use_actual=True)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{self.robotname}/info.json"), "r") as f:
            self.info = json.load(f)
        self.img_height = self.cam_calib.img_height
        self.img_width = self.cam_calib.img_width

    @staticmethod
    def plot_points_on_image(img, corresponding_dists, corresponding_pcs_coords, resize=False):
        """
        img: rectified cv2 image (BGR) to plot on
        corresponding_dists: a list of distances / 1d np array of distances
        corresponding_pcs_coords: N x 2 pixel coordinates corresponding to the distances
        """
        img2 = deepcopy(img)
        if isinstance(corresponding_dists, np.ndarray):
            corresponding_dists = list(corresponding_dists.squeeze())
        colors = LidarCamCalib.get_depth_colors(corresponding_dists)
        radius = int(4 * img.shape[0] / 1536)
        radius = radius if radius > 0 else 1
        for i in range(corresponding_pcs_coords.shape[0]):
            cv2.circle(img2, tuple(corresponding_pcs_coords[i, :].astype(np.int32)), radius=radius, color=colors[i], thickness=-1)
        if resize:
            img2 = cv2.resize(img2, None, fx=0.75, fy=0.75)
        return img2

    @staticmethod
    def interp(a, b, x, method="nearest"):
        y = griddata(a, b, x, method=method)
        # for coord, value in zip(a, b):
        #     index = np.where((x == coord).all(axis=1))[0]
        #     y[index] = value
        return y

    @staticmethod
    def double_interp(a, b, x, do_nearest=True, firstmethod="linear"):
        y = LidarCamCalib.interp(a, b, x, method=firstmethod)
        single_mask = np.isnan(y[:, 0]).squeeze()
        if do_nearest:
            x2 = x[~single_mask]
            y2 = y[~single_mask]
            y = LidarCamCalib.interp(a=x2, b=y2, x=x, method="nearest")
            return y, np.ones(x.shape[0], dtype=bool)
        else:
            return y[~single_mask], ~single_mask

    @staticmethod
    def get_depth_colors(dists):
        """
        Gives colors for depth values
        dists: list of distances
        Returns: list of colors in BGR format
        """
        COLMAP = LidarCamCalib.COLMAP
        colors = []
        for i in range(len(dists)):
            range_val = max(min(round((dists[i] / 30.0) * 49), 49), 0)
            color = (255 * COLMAP[49 - range_val][2], 255 * COLMAP[49 - range_val][1], 255 * COLMAP[49 - range_val][0])
            colors.append(color)
        return colors

    def _correct_pc(self, vlp_points):
        """
        Corrects actual pc to desired lidar location pc (i.e., transforms from actual_lidar frame to lidar frame)
        vlp_points: (N x K) numpy array of points in (uncorrected) actual VLP frame
        returns: (N x K) numpy array of points in (corrected) VLP frame
        """
        vlp_points_copy = deepcopy(vlp_points)
        actual_lidar_to_wcs = np.linalg.inv(self.actual_M_ext)
        wcs_to_lidar = self.M_ext
        actual_lidar_to_lidar = wcs_to_lidar @ actual_lidar_to_wcs   # lidar_from_actual_lidar = lidar_from_wcs @ wcs_from_actual_lidar
        vlp_points_copy[:, :3] = Homography.general_project_A_to_B(vlp_points[:, :3], actual_lidar_to_lidar)
        return vlp_points_copy

    def projectVLPtoWCS(self, vlp_points):
        """
        Project VLP points to WCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 3) numpy array of points in WCS
        """
        return Homography.general_project_A_to_B(vlp_points, np.linalg.inv(self.M_ext))

    def projectVLPtoPCS(self, vlp_points, mode="skip", ret_zs=False):
        """
        Project VLP points to PCS (rectified)
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 2) numpy array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        wcs_coords = self.projectVLPtoWCS(vlp_points)
        ccs_coords = self.cam_calib.projectWCStoCCS(wcs_coords)
        pcs_coords, mask = self.cam_calib.projectCCStoPCS(ccs_coords, mode=mode)
        ccs_dists = np.linalg.norm(ccs_coords, axis=1).reshape((-1, 1))
        ccs_dists = ccs_dists[mask]
        if ret_zs:
            ccs_zs = ccs_coords[:, 2].reshape((-1, 1))
            ccs_zs = ccs_zs[mask]
            return pcs_coords, mask, ccs_dists, ccs_zs
        return pcs_coords, mask, ccs_dists

    def projectPCtoImage(self, pc_np, img):
        """
        img: (H x W x 3) numpy array, rectified cv2 based (BGR)
        pc_np: (N x 3) numpy array of points in VLP frame
        """
        pcs_coords, mask, ccs_dists = self.projectVLPtoPCS(pc_np)
        img = LidarCamCalib.plot_points_on_image(img, ccs_dists, pcs_coords)
        return img, pcs_coords, pc_np[mask], np.asarray(ccs_dists.squeeze()).reshape((-1, 1))

    def projectPCtoImageFull(self, pc_np, img, ret_imgs=False, do_nearest=True, firstmethod="linear", resize=True):
        """
        img: (H x W x 3) numpy array, rectified cv2 based (BGR)
        pc_np: (N x 3) numpy array of points in VLP frame
        """
        _, corresponding_pcs_coords, corresponding_vlp_coords, corresponding_ccs_dists = self.projectPCtoImage(pc_np, img)
        all_ys, all_xs = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        all_vlp_coords, interp_mask = self.double_interp(a=corresponding_pcs_coords, b=corresponding_vlp_coords, x=all_pixel_locs, do_nearest=do_nearest, firstmethod=firstmethod)

        all_ccs_dists, interp_mask = self.double_interp(a=corresponding_pcs_coords, b=corresponding_ccs_dists, x=all_pixel_locs, do_nearest=do_nearest, firstmethod=firstmethod)
        all_pixel_locs = all_pixel_locs[interp_mask]
        all_vlp_zs = all_vlp_coords[:, 2].reshape((-1, 1))
        corresponding_vlp_zs = corresponding_vlp_coords[:, 2].reshape((-1, 1))
        if ret_imgs:
            img_vlp_ccs_dists = deepcopy(img)
            img_full_ccs_dists = deepcopy(img)
            img_vlp_vlp_zs = deepcopy(img)
            img_full_vlp_zs = deepcopy(img)
            img_vlp_ccs_dists = LidarCamCalib.plot_points_on_image(img_vlp_ccs_dists, corresponding_ccs_dists, corresponding_pcs_coords, resize=resize)
            img_full_ccs_dists = LidarCamCalib.plot_points_on_image(img_full_ccs_dists, all_ccs_dists, all_pixel_locs, resize=resize)
            img_vlp_vlp_zs = LidarCamCalib.plot_points_on_image(img_vlp_vlp_zs, corresponding_vlp_zs, corresponding_pcs_coords, resize=resize)
            img_full_vlp_zs = LidarCamCalib.plot_points_on_image(img_full_vlp_zs, all_vlp_zs, all_pixel_locs, resize=resize)
            side_by_side_ccs_dists = np.hstack((img_vlp_ccs_dists, img_full_ccs_dists))
            side_by_side_vlp_zs = np.hstack((img_vlp_vlp_zs, img_full_vlp_zs))
            full_img = np.vstack((side_by_side_ccs_dists, side_by_side_vlp_zs))
            imgs = [full_img, img_vlp_ccs_dists, img_full_ccs_dists, img_vlp_vlp_zs, img_full_vlp_zs]
            return all_pixel_locs, all_vlp_coords, all_ccs_dists, all_vlp_zs, interp_mask, imgs
        return all_pixel_locs, all_vlp_coords, all_ccs_dists, all_vlp_zs, interp_mask

    def main(self, img, vlp_points, event=None):
        """
        img: (H x W x 3) numpy array, rectified cv2 based (BGR)
        vlp_points: (N x 3) numpy array of points in (by deafult, corrected) VLP frame
        """
        if vlp_points is None or img is None:
            return
        img, *_ = self.projectPCtoImage(vlp_points, img)
        img = cv2.resize(img, None, fx=0.75, fy=0.75)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        self.overlay_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node('lidar_cam_calib_node', anonymous=False)
    e = LidarCamCalib(robotname="jackal", ros_flag=True, cam_res=540)
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down lidar cam calib node")
