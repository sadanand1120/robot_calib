import numpy as np
from PIL import Image as PILImage
import cv2
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
from cv_bridge import CvBridge
import torch
from matplotlib import pyplot as plt
from lidar_cam_calib import LidarCamCalib
from homography import Homography
from depthany2.metric_main import DepthAny2


class D2P:
    def __init__(self, robotname="jackal", device=None, depth_encoder='vitl', depth_dataset='hypersim', cam_res=540):
        self.lcc = LidarCamCalib(ros_flag=False, robotname=robotname, cam_res=cam_res)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.depth_obj = DepthAny2(device=self.DEVICE, model_input_size=518, max_depth=1, encoder=depth_encoder, dataset=depth_dataset)

    @staticmethod
    def depth2points(depth_arr_img: np.ndarray, cam_intrinsics_dict):
        FX = cam_intrinsics_dict['camera_matrix'][0, 0]
        FY = cam_intrinsics_dict['camera_matrix'][1, 1]
        CX = cam_intrinsics_dict['camera_matrix'][0, 2]
        CY = cam_intrinsics_dict['camera_matrix'][1, 2]
        x, y = np.meshgrid(np.arange(depth_arr_img.shape[1]), np.arange(depth_arr_img.shape[0]))
        # back project (along the camera ray) the pixel coordinates to 3D using the depth
        x = (x - CX) / FX
        y = (y - CY) / FY
        points = np.stack((np.multiply(x, depth_arr_img), np.multiply(y, depth_arr_img), depth_arr_img), axis=-1).reshape(-1, 3)
        return points

    def project_points_kinect_to_lidar(self, points_kinect):
        M_baselink_to_kinect = self.lcc.cam_calib.M_ext
        M_baselink_to_lidar = self.lcc.M_ext
        M_kinect_to_lidar = M_baselink_to_lidar @ np.linalg.inv(M_baselink_to_kinect)
        return Homography.general_project_A_to_B(points_kinect, M_kinect_to_lidar)

    @torch.inference_mode()
    def main(self, cv2_img, pc_np_xyz=None, do_lidar_correction=True):
        """
        cv2_img has to be rectified
        Pass pc_np_xyz as None if you do not want to do lidar based metric scaling
        """
        with torch.device(self.DEVICE):
            depth_rel_img = self.depth_obj.predict(cv2_img, max_depth=1)
            if pc_np_xyz is not None:
                if do_lidar_correction:
                    pc_np_xyz = self.lcc._correct_pc(pc_np_xyz)
                pcs_coords, _, _, ccs_zs = self.lcc.projectVLPtoPCS(pc_np_xyz, ret_zs=True)
                depths_rel = depth_rel_img[pcs_coords[:, 1].astype(int), pcs_coords[:, 0].astype(int)].reshape((-1, 1))
                scaling_facs = ccs_zs / depths_rel   # TODO: impl diff scaling facs over the image, ie, remove outliers, and then do a smoothen / interp over scaling array, and then multiply with this instead of a single scaling fac
                max_d = float(np.median(scaling_facs))
                depth_rel_img = depth_rel_img * max_d
            kinect_points = self.depth2points(depth_rel_img, self.lcc.cam_calib.intrinsics_dict)
            lidar_points = self.project_points_kinect_to_lidar(kinect_points)
        return lidar_points, depth_rel_img


if __name__ == "__main__":
    d2p = D2P(robotname="jackal")
    cv2_img = cv2.imread("/home/dynamo/AMRL_Research/repos/synapse/test/000000.png")
    cv2_img = d2p.lcc.cam_calib.rectifyRawCamImage(cv2_img)
    pc_np = np.fromfile("/home/dynamo/AMRL_Research/repos/synapse/test/000000.bin", dtype=np.float32).reshape((-1, 4))
    lidar_points, depth_arr = d2p.main(cv2_img, pc_np[:, :3], do_lidar_correction=True)
    plt.imshow(depth_arr)
    plt.show()
    flat_pc = lidar_points.reshape(-1).astype(np.float32)
    flat_pc.tofile("depth.bin")
