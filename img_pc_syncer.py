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
from depth2points import D2P
import os
from tqdm import tqdm
import sys
from scipy.stats import kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import json
from scipy.stats import gaussian_kde

# NOTE: below is depthbased image syncer code, but it did not work well, so using manual image viz based syncer
# def img_to_allpcs(img_index, rootdir, save_dir, LCC, num=20, do_lidar_correction=True, robotname="jackal", cam_res=540):
#     img_dir = os.path.join(rootdir, "images")
#     pc_dir = os.path.join(rootdir, "pcs")
#     depths_dir = os.path.join(rootdir, "un_scaled_depths")

#     all_images = sorted(os.listdir(img_dir))
#     all_images_timestamps = [float(img.split('.')[0]) for img in all_images]
#     all_pcs = sorted(os.listdir(pc_dir))
#     all_pcs_timestamps = [float(pc.split('.')[0]) for pc in all_pcs]

#     cv2_img = cv2.imread(os.path.join(img_dir, all_images[img_index]))
#     depth_rel_img = np.fromfile(os.path.join(depths_dir, all_images[img_index].replace(".png", ".bin")), dtype=np.float32).reshape(cv2_img.shape[0], cv2_img.shape[1])

#     # find pc index for which the time is closest to the img_index's time
#     img_time = all_images_timestamps[img_index]
#     pc_index = np.argmin(np.abs(np.array(all_pcs_timestamps) - img_time))
#     start_index = max(0, pc_index - num)
#     end_index = min(len(all_pcs), pc_index + num)

#     time_delta_ms_arr = []
#     kurt_arr = []

#     for i in range(start_index, end_index):
#         pc_np_xyz = np.fromfile(os.path.join(pc_dir, all_pcs[i]), dtype=np.float32).reshape((-1, 3))
#         if do_lidar_correction:
#             pc_np_xyz = LCC._correct_pc(pc_np_xyz)
#         pcs_coords, _, _, ccs_zs = LCC.projectVLPtoPCS(pc_np_xyz, ret_zs=True)
#         depths_rel = depth_rel_img[pcs_coords[:, 1].astype(int), pcs_coords[:, 0].astype(int)].reshape((-1, 1))
#         scaling_facs = (ccs_zs / depths_rel).squeeze()

#         # plot a histogram
#         time_delta_ms = (all_pcs_timestamps[i] - img_time) / 1e6
#         kurt = kurtosis(scaling_facs)
#         time_delta_ms_arr.append(time_delta_ms)
#         kurt_arr.append(kurt)
#         # sns.kdeplot(scaling_facs, bw_adjust=0.5)  # bw_adjust controls smoothness
#         # plt.title(f"i={i}, KDE for time diff (ms): {round(time_delta_ms, 2)}, Kurtosis: {round(kurt, 2)}")
#         # plt.xlabel("Scaling factor")
#         # plt.ylabel("Density")
#         # plt.show()

#     save_path = os.path.join(save_dir, f"{all_images[img_index].replace('.png', '.json')}")
#     with open(save_path, "w") as f:
#         json.dump(dict(zip(time_delta_ms_arr, kurt_arr)), f, indent=2)


def save_all_unscaled_depths(img_dir, save_depth_dir, robotname="jackal", device=None, depth_encoder='vitl', depth_dataset='hypersim', cam_res=540):
    os.makedirs(save_depth_dir, exist_ok=True)
    all_images = sorted(os.listdir(img_dir))
    d2p = D2P(robotname=robotname, device=device, depth_encoder=depth_encoder, depth_dataset=depth_dataset, cam_res=cam_res)
    for img_name in tqdm(all_images, desc="Saving depth pcs", unit="img"):
        img_path = os.path.join(img_dir, img_name)
        cv2_img = cv2.imread(img_path)
        _, unscaled_depth_rel_arr = d2p.main(cv2_img)
        depth_path = os.path.join(save_depth_dir, img_name.replace(".png", ".bin"))
        unscaled_depth_rel_arr.tofile(depth_path)


def allimgs_sync(rootdir, num=20, do_lidar_correction=True, robotname="jackal", cam_res=540):
    save_dir = os.path.join(rootdir, "sync_cache")
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(rootdir, "images")
    all_images = sorted(os.listdir(img_dir))
    LCC = LidarCamCalib(ros_flag=False, robotname=robotname, cam_res=cam_res)
    for img_index in tqdm(range(len(all_images)), desc="Syncing images", unit="img"):
        img_to_allpcs(img_index, rootdir, save_dir, LCC, num, do_lidar_correction, robotname, cam_res)


def img_to_allpcs(img_index, rootdir, save_dir, LCC: LidarCamCalib, num=20, do_lidar_correction=True, robotname="jackal", cam_res=540):
    img_dir = os.path.join(rootdir, "images")
    pc_dir = os.path.join(rootdir, "pcs")

    all_images = sorted(os.listdir(img_dir))
    all_images_timestamps = [float(img.split('.')[0]) for img in all_images]
    all_pcs = sorted(os.listdir(pc_dir))
    all_pcs_timestamps = [float(pc.split('.')[0]) for pc in all_pcs]

    cv2_img = cv2.imread(os.path.join(img_dir, all_images[img_index]))

    cur_save_dir = os.path.join(save_dir, all_images[img_index].replace(".png", ""))
    os.makedirs(cur_save_dir, exist_ok=True)

    # find pc index for which the time is closest to the img_index's time
    img_time = all_images_timestamps[img_index]
    pc_index = np.argmin(np.abs(np.array(all_pcs_timestamps) - img_time))
    start_index = max(0, pc_index - num)
    end_index = min(len(all_pcs), pc_index + num)

    for i in range(start_index, end_index):
        pc_np_xyz = np.fromfile(os.path.join(pc_dir, all_pcs[i]), dtype=np.float32).reshape((-1, 3))
        if do_lidar_correction:
            pc_np_xyz = LCC._correct_pc(pc_np_xyz)
        img_overlay, *_ = LCC.projectPCtoImage(pc_np_xyz, cv2_img)
        time_delta_ms = round((all_pcs_timestamps[i] - img_time) / 1e6, 4)
        cv2.imwrite(os.path.join(cur_save_dir, f"{time_delta_ms}.png"), img_overlay)


def get_sorted_png_paths(root_dir):
    sorted_paths = []

    # Get all subfolders with numeric names and sort by integer value
    subfolders = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
        key=lambda x: int(x.replace('.png', ''))
    )

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        pngs = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]

        # Sort images by numeric value (can be negative)
        pngs_sorted = sorted(pngs, key=lambda x: float(x.replace('.png', '')))

        for f in pngs_sorted:
            sorted_paths.append(os.path.join(subfolder_path, f))

    return sorted_paths


def review_and_filter_images(root_dir):
    png_paths = get_sorted_png_paths(root_dir)

    print(f"Total images: {len(png_paths)}")
    cv2.namedWindow("Image Review", cv2.WINDOW_AUTOSIZE)

    idx = 0
    while idx < len(png_paths):
        path = png_paths[idx]
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load: {path}")
            idx += 1
            continue
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, None, fx=0.75, fy=0.75)
        display_text = f"{idx + 1}/{len(png_paths)} (x=delete, n=next, q=quit); timedelta: {os.path.basename(path).replace('.png', '')}"
        cv2.imshow("Image Review", img)
        cv2.setWindowTitle("Image Review", display_text)

        key = cv2.waitKey(0)
        if key == ord('x'):
            os.remove(path)
            print(f"Deleted: {path}")
        elif key == ord('q'):
            break
        idx += 1

    cv2.destroyAllWindows()


def get_kde_peak(data):
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(data), max(data), 1000)
    y_kde = kde(x_grid)
    peak_x = x_grid[np.argmax(y_kde)]
    return peak_x


def find_synctime_from_filtered(root_dir):
    png_paths = sorted([os.path.join(dp, f) for dp, _, files in os.walk(root_dir) for f in files if f.endswith(".png")])
    png_filenames = [os.path.basename(p) for p in png_paths]
    noext_names = [float(os.path.splitext(f)[0]) for f in png_filenames]

    sns.kdeplot(noext_names, fill=True)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Plot of Float Values")
    plt.grid(True)
    plt.show()

    peak = get_kde_peak(noext_names)
    print(f"Peak value (mode of density): {peak}")  # -545 ms is the answer, i.e., pc at t matches best with image at t+545ms


if __name__ == "__main__":
    # save_all_unscaled_depths(img_dir="/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/sync/images",
    #                          save_depth_dir="/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/sync/un_scaled_depths",
    #                          robotname="spot",
    #                          depth_dataset="vkitti",
    #                          cam_res=3072)

    # allimgs_sync(rootdir="/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/sync",
    #              num=20,
    #              do_lidar_correction=True,
    #              robotname="spot",
    #              cam_res=3072)

    # review_and_filter_images("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/sync/sync_cache")

    find_synctime_from_filtered("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/sync/sync_cache")
