import argparse
import builtins
import os
import open3d as o3d
from rich import pretty, print
import LIM.log as log
import config.config as config
from LIM.data.structures.pcloud import Painter, PCloud, Downsampler
import numpy as np
import torch

pretty.install()
builtins.print = print

os.environ["XDG_SESSION_TYPE"] = "x11"


def main():
    from LIM.data.sets.datasetI import CloudDatasetsI
    from LIM.data.sets.scanNet import ScanNet

    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()

    YELLOW = np.array([1.0, 0.706, 0.0])
    THRESHOLD = 0.1

    dataset = ScanNet.new_instance(CloudDatasetsI.SPLITS.TRAIN)
    cloud: PCloud
    implicit_l1: PCloud
    implicit_iou: PCloud

    for cloud, implicit_l1, implicit_iou in dataset:
        cloud = Painter.Uniform(YELLOW, compute_normals=True)(cloud)
        implicit_iou = Painter.Uniform(YELLOW, compute_normals=False)(implicit_iou)

        mask = np.squeeze(implicit_iou.features) < THRESHOLD
        implicit_iou.pcd.points = o3d.utility.Vector3dVector(np.asarray(implicit_iou.pcd.points)[mask, :])
        implicit_iou._features = implicit_iou.features[mask, :]
        # implicit_iou.pcd.normals = o3d.utility.Vector3dVector(np.asarray(implicit_iou.pcd.normals)[mask])
        # implicit_iou.pcd.colors = o3d.utility.Vector3dVector(np.asarray(implicit_iou.pcd.colors)[mask])

        vis1.create_window("pcloud", 1280, 720, left=0, top=720)
        vis2.create_window("implicit", 1280, 720, left=1280, top=720)
        vis1.add_geometry(cloud.pcd)
        vis2.add_geometry(implicit_iou.pcd)
        while True:
            vis1.update_geometry(cloud.pcd)
            vis2.update_geometry(implicit_iou.pcd)

            if not vis1.poll_events():
                break
            if not vis2.poll_events():
                break

            vis1.update_renderer()
            vis2.update_renderer()

        vis1.destroy_window()
        vis2.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="implicit_viewer", add_help=True)
    parser.add_argument(
        "config_file",
        help="Path to the configuration file",
        default="foo",
    )
    args = parser.parse_args()
    config.settings = config.Settings.from_yaml(args.config_file)
    main()
