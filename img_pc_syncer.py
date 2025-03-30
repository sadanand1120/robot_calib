import numpy as np
import cv2
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
from matplotlib import pyplot as plt
from lidar_cam_calib import LidarCamCalib
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def overlay_pcs_on_images(rootdir, num=20, do_lidar_correction=True, robotname="jackal", cam_res=540):
    """
    Assumes you already did process_bag.py to extract images and pcs from a bag file.
    rootdir: path to the directory containing the images and pcs subdirs
    """
    save_dir = os.path.join(rootdir, "sync_cache")
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(rootdir, "images")
    all_images = sorted(os.listdir(img_dir))
    LCC = LidarCamCalib(ros_flag=False, robotname=robotname, cam_res=cam_res)
    for img_index in tqdm(range(len(all_images)), desc="Syncing images", unit="img"):
        overlay_pcs_on_img(img_index, rootdir, save_dir, LCC, num, do_lidar_correction)


def overlay_pcs_on_img(img_index, rootdir, save_dir, LCC: LidarCamCalib, num=20, do_lidar_correction=True):
    img_dir = os.path.join(rootdir, "images")
    pc_dir = os.path.join(rootdir, "pcs")

    all_images = sorted(os.listdir(img_dir))
    all_images_timestamps = [float(img.split('.')[0]) for img in all_images]
    all_pcs = sorted(os.listdir(pc_dir))
    all_pcs_timestamps = [float(pc.split('.')[0]) for pc in all_pcs]

    cv2_img = cv2.imread(os.path.join(img_dir, all_images[img_index]))
    cv2_img = LCC.cam_calib.rectifyRawCamImage(cv2_img)

    cur_save_dir = os.path.join(save_dir, all_images[img_index].replace(".png", ""))
    os.makedirs(cur_save_dir, exist_ok=True)

    # find pc index for which the time is closest to the img_index's time
    img_time = all_images_timestamps[img_index]
    pc_index = np.argmin(np.abs(np.array(all_pcs_timestamps) - img_time))
    start_index = max(0, pc_index - num)
    end_index = min(len(all_pcs), pc_index + num)

    for i in tqdm(range(start_index, end_index), desc="Overlaying PCs", unit="pc", leave=False):
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


def review_and_filter_sync_overlays(sync_cache_dir):
    png_paths = get_sorted_png_paths(sync_cache_dir)

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


def find_synctime_from_filtered(sync_cache_dir):
    png_paths = sorted([os.path.join(dp, f) for dp, _, files in os.walk(sync_cache_dir) for f in files if f.endswith(".png")])
    png_filenames = [os.path.basename(p) for p in png_paths]
    noext_names = [float(os.path.splitext(f)[0]) for f in png_filenames]

    sns.kdeplot(noext_names, fill=True)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Plot of Float Values")
    plt.grid(True)
    plt.show()

    kde = gaussian_kde(noext_names)
    x_grid = np.linspace(min(noext_names), max(noext_names), 1000)
    y_kde = kde(x_grid)
    peak_x = x_grid[np.argmax(y_kde)]
    print(f"Peak value (mode of density): {peak_x}")  # -585 ms is the answer for spot, i.e., pc at t matches best with image at t+585ms


if __name__ == "__main__":
    # overlay_pcs_on_images(rootdir="/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/sync",
    #                       num=20,
    #                       do_lidar_correction=True,
    #                       robotname="spot",
    #                       cam_res=3072)

    # review_and_filter_sync_overlays("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/sync/sync_cache")

    find_synctime_from_filtered("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/sync/sync_cache")
