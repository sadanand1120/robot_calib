import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from lidar_cam_calib import LidarCamCalib
from depth2points import D2P


class EgoToBEV:
    def __init__(self, robotname="jackal", depth_encoder='vitl', depth_dataset='hypersim', cam_res=540):
        self.robotname = robotname
        self.d2p = D2P(robotname=robotname, depth_encoder=depth_encoder, depth_dataset=depth_dataset, cam_res=cam_res)
        # here below, I hardcoded jackal (the bev image there was much closer and zoomed in though less detailed as spot has higher res), but you can change to dynamic robotname if needed
        self.bev_lidar_cam = LidarCamCalib(ros_flag=False, robotname="jackal", override_cam_extrinsics_filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)), f"jackal/params/bev_cam_extrinsics.yaml"), cam_res=540)

    def get_bev(self, pil_img, pc_np, inpaint=True, use_depthpred=True, do_lidar_correction=True):
        pc_np = pc_np[:, :3]
        np_pil_img = np.array(pil_img)
        np_cv2_img = cv2.cvtColor(np_pil_img, cv2.COLOR_RGB2BGR)
        if do_lidar_correction:
            pc_np = self.d2p.lcc._correct_pc(pc_np)
        if use_depthpred:
            pc_np, _ = self.d2p.main(np_cv2_img, pc_np, do_lidar_correction=False)

        all_ys, all_xs = np.meshgrid(np.arange(pil_img.height), np.arange(pil_img.width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        _, all_vlp_coords, _, _, interp_mask = self.d2p.lcc.projectPCtoImageFull(pc_np, np_cv2_img, do_nearest=False)
        all_pixel_locs = all_pixel_locs[interp_mask]
        bev_np_pil_img = np.zeros((self.bev_lidar_cam.img_height, self.bev_lidar_cam.img_width, 3), dtype=np.uint8)
        bev_pixel_locs, bev_mask, _ = self.bev_lidar_cam.projectVLPtoPCS(all_vlp_coords)

        all_pixel_locs = all_pixel_locs[bev_mask]
        rows_bev, cols_bev = bev_pixel_locs[:, 1], bev_pixel_locs[:, 0]
        rows_all, cols_all = all_pixel_locs[:, 1], all_pixel_locs[:, 0]
        bev_np_pil_img[rows_bev, cols_bev] = np_pil_img[rows_all, cols_all]
        if inpaint:
            inpaint_mask = np.all(bev_np_pil_img == [0, 0, 0], axis=-1).astype(np.uint8)
            polygon_mask = np.zeros((bev_np_pil_img.shape[0], bev_np_pil_img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(polygon_mask, bev_pixel_locs, 1)
            combined_mask = cv2.bitwise_and(inpaint_mask, polygon_mask)
            bev_np_pil_img = cv2.inpaint(bev_np_pil_img, combined_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        return bev_np_pil_img


if __name__ == "__main__":
    raw_pil_img = Image.open("/home/dynamo/Music/metric_depthany2_calib/test/img.png")
    pc_np = np.fromfile("/home/dynamo/Music/metric_depthany2_calib/test/pc_np.bin", dtype=np.float32).reshape((-1, 3))

    f, axs = plt.subplots(1, 2)
    f.set_figheight(30)
    f.set_figwidth(50)
    axs[0].set_title("Raw", {'fontsize': 40})
    axs[0].imshow(raw_pil_img)
    axs[1].set_title("BEV", {'fontsize': 40})
    axs[1].imshow(EgoToBEV(depth_dataset='hypersim', robotname="spot", cam_res=1536).get_bev(raw_pil_img, pc_np, use_depthpred=True))  # vkitti much better than hypersim outdoors
    plt.show()
