from LIM.data.structures.pcloud import PCloud, Painter
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import os

os.environ["XDG_SESSION_TYPE"] = "x11"


def view(paths: list[Path]):
    YELLOW = np.array([1.0, 0.706, 0.0])
    vis = o3d.visualization.Visualizer()
    vis.create_window("viewer", width=1280, height=720)

    clouds = []
    painter = Painter.Uniform(YELLOW, compute_normals=True)
    for path in paths:
        if ".npz" in path.suffix:
            np_data = np.load(str(path))["points"]
            clouds.append(painter(PCloud.from_arr(np_data)).pcd)
        else:
            clouds.append(painter(PCloud.from_path(path)).pcd)

    for pcd in clouds:
        vis.add_geometry(pcd)
    while True:
        for pcd in clouds:
            vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="viewer",
        add_help=True,
    )
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        help="paths to the pointcloud we want to visualize",
        required=True,
    )
    args = parser.parse_args()
    print(args)
    view([Path(p) for p in args.paths])
