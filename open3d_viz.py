import PIL.Image
import open3d as o3d
import PIL
import numpy as np
import cv2
from matplotlib import pyplot as plt
from depth2points import D2P


def get_pcd_colors_from_image(pil_img: PIL.Image.Image):
    colors = np.asarray(pil_img).reshape(-1, 3) / 255.0
    return o3d.utility.Vector3dVector(colors)


def pcd_from_np(pc_np, color_rgb_list=None):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    c = [1, 0.647, 0] if color_rgb_list is None else color_rgb_list   # default orange color
    pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(xyz), 1)))
    return pcd


def save_pcd(pointcloud, filename="pointcloud.pcd"):
    """Save an Open3D point cloud to a PCD file."""
    o3d.io.write_point_cloud(filename, pointcloud)


def load_pcd(filename="pointcloud.pcd"):
    """Load a PCD file as an Open3D point cloud."""
    return o3d.io.read_point_cloud(filename)


def visualize_pcd(pcd, camera_params_jsonpath="config/open3d_cameraview_params.json", point_size=0.85):
    vis, ctr = init_vis(point_size=point_size)
    vis.add_geometry(pcd)
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    # vis.add_geometry(axis_pcd)
    if camera_params_jsonpath is not None:
        param = o3d.io.read_pinhole_camera_parameters(camera_params_jsonpath)
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.run()    # keeping the visualization window open until the user closes it
    vis.destroy_window()


def init_vis(point_size=0.85):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    ctr = vis.get_view_control()
    return vis, ctr


def write_o3d_camera_params(pcd_path, jsonpath):
    """
    Using a pcd file, calibrate the view and then write the camera parameters to a json file.
    """
    pcd = load_pcd(pcd_path)
    vis, ctr = init_vis(point_size=0.85)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(jsonpath, camera_params)
    vis.destroy_window()


def get_pcd_from_image_depth_prediction(imgpath, robotname, depth_dataset, cam_res, scalefac=1.0, lidar_pc_np_xyz=None):
    d2p = D2P(robotname=robotname, depth_dataset=depth_dataset, cam_res=cam_res)
    cv2_img = cv2.imread(imgpath)
    cv2_img = d2p.lcc.cam_calib.rectifyRawCamImage(cv2_img)
    pc_np_xyz, depth_rel_img = d2p.main(cv2_img, lidar_pc_np_xyz)
    pc_np_xyz[:, :3] *= scalefac
    pcd = pcd_from_np(pc_np_xyz)
    pcd.colors = get_pcd_colors_from_image(PIL.Image.fromarray(cv2_img))
    return pcd, depth_rel_img


if __name__ == "__main__":
    pcd, depth_rel_img = get_pcd_from_image_depth_prediction(imgpath="notrack_bags/paths/poc_cut/images/1742591008774347085.png",
                                                             robotname="spot",
                                                             depth_dataset="vkitti",
                                                             cam_res=3072,
                                                             scalefac=1.0,
                                                             lidar_pc_np_xyz=None)
    plt.imshow(depth_rel_img)
    plt.show()
    visualize_pcd(pcd, camera_params_jsonpath=None, point_size=0.75)
