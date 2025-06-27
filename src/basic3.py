import builtins
import functools
import importlib.util
import sys
import time
from copy import copy
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any, Type, Callable
import polars as pl
import numpy as np
import open3d.pipelines.registration as o3d_reg
import torch
from rich import pretty, print, traceback
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from debug.decorators import identify_function
from LIM.data.sets.bunny import Bunny, Transform
from LIM.data.structures.pcloud import Downsampler, PCloud
from LIM.data.structures.pair import Pair
import matplotlib.pyplot as plt

traceback.install(show_locals=False)
pretty.install()
builtins.print = print

np.set_printoptions(precision=2, suppress=True)


@identify_function
def import_without_init(
    path_to_file: str | Path, class_name: str, dependencies: list[str | Path] | None = None
) -> Type[Any]:
    """
    Import a class from a module without running parent package __init__.py
    and with temporary sys.path dependencies.

    Args:
        module_path: Path to the Python file
        class_name: Name of the class to import
        dependencies: Optional list of directories to add to sys.path temporarily
    """
    dependencies = [] if dependencies is None else dependencies
    og_sys_path = sys.path.copy()
    try:
        for dependency in dependencies:
            sys.path.insert(0, str(Path(dependency).resolve()))
        path_to_file = path_to_file if isinstance(path_to_file, Path) else Path(path_to_file)
        if not path_to_file.is_file():
            raise FileNotFoundError(f"Could not find path at {path_to_file}")
        spec: ModuleSpec = importlib.util.spec_from_file_location(path_to_file.stem, Path(path_to_file))  # pyright: ignore[reportAssignmentType]
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
        dynamic_class = getattr(module, class_name)
    finally:
        sys.path = og_sys_path
    return dynamic_class


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
        _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = (
        torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    )  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


@identify_function
def load_encoder(load_weights: bool = True) -> torch.nn.Module:
    encoder = import_without_init(
        path_to_file=(IAE_PATH := Path("src/submodules/IAE/")) / "src/encoder/dgcnn_cls.py",
        class_name="DGCNN_Cls_Encoder",
        dependencies=[IAE_PATH],
    )(feat_dim=1024, c_dim=256, k=20)

    if load_weights:
        missing, unexpected = encoder.load_state_dict(
            state_dict=torch.load(
                f=Path("src/weights/classification/modelnet40_trained.pt"),
                weights_only=False,
            )["model_state_dict"],
            strict=False,
        )

    # monkey patch last layers and forward method
    encoder.conv6 = torch.nn.Identity()
    encoder.conv7 = torch.nn.Identity()

    @identify_function
    def _forward_up_to_conv5(self, x):
        x = x.transpose(2, 1).contiguous()  # (B, 3, N)
        B, _, N = x.shape

        # edge conv blocks
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(-1)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(-1)[0]

        # conv5
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        feat = self.conv5(x_cat)  # (B, feat_dim, N)

        return feat.permute(0, 2, 1).contiguous()  # (B, N, feat_dim)

    encoder.forward = MethodType(_forward_up_to_conv5, encoder)

    return encoder


_executor = ThreadPoolExecutor(max_workers=4)


def track_progress(func: Callable) -> Callable:
    @functools.wraps(func)
    def inner(*args, **kwargs) -> bool:
        future = _executor.submit(func, *args, **kwargs)

        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(func.__name__, total=None)

            while not future.done():
                progress.advance(task)
                time.sleep(0.1)

            return future.result()

    return inner


def RANSAC(
    source: PCloud,
    target: PCloud,
    ransac_n: int,
    max_correspondence_distance: float,
    similarity_threshold: float,
    distance_threshold: float,
    max_iteration: int,
    voxel_size: float | None = None,
) -> torch.Tensor:
    down_src = copy(source)
    down_tgt = copy(target)
    if voxel_size is not None:
        down_src = Downsampler(size=int(down_src.shape[0]), mode=Downsampler.Mode.VOXEL)(
            down_src, voxel_size=voxel_size
        )
        down_tgt = Downsampler(size=int(down_tgt.shape[0]), mode=Downsampler.Mode.VOXEL)(
            down_tgt, voxel_size=voxel_size
        )
    print(down_src, down_tgt)
    return o3d_reg.registration_ransac_based_on_feature_matching(
        source=down_src.pcd.to_legacy(),
        target=down_tgt.pcd.to_legacy(),
        source_feature=down_src.o3d_features,
        target_feature=down_tgt.o3d_features,
        mutual_filter=False,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d_reg.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d_reg.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold),
            o3d_reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d_reg.RANSACConvergenceCriteria(4000000, 500),
    )


def model_inference(pair, model) -> Pair:
    with torch.no_grad():
        pair.source.features = model(pair.source.points.unsqueeze(dim=0))[0]
        pair.target.features = model(pair.target.points.unsqueeze(dim=0))[0]
    return pair


def pose_error(T_pred: np.ndarray | torch.Tensor, T_gt: np.ndarray | torch.Tensor) -> tuple[float, float]:
    """

    Return:
        Relative Rotation Error (RRE) -> [0, pi]
        Reltaive Translation Error (RTE) -> [0, +inf)
    """
    T_pred = T_pred if isinstance(T_pred, np.ndarray) else T_pred.cpu().detach().numpy()
    T_gt = T_gt if isinstance(T_gt, np.ndarray) else T_gt.cpu().detach().numpy()
    R_pred, R_gt = T_pred[:3, :3], T_gt[:3, :3]
    t_pred, t_gt = T_pred[:3, 3], T_gt[:3, 3]

    translation_error = np.linalg.norm(t_pred - t_gt)
    raw_trace = np.trace(R_gt.T @ R_pred)
    safe_trace = np.clip(raw_trace, -1.0, 3.0)
    rotation_error = np.arccos((safe_trace - 1) / 2)

    return translation_error, rotation_error


def plot_df(df: pl.DataFrame, prefix: str) -> None:
    # Plot 1: Relative Rotation Error (RRE) vs Overlap
    plt.figure()
    sns.lineplot(
        data=df,
        x="overlap_ratio",
        y="RRE",
        estimator="mean",
        errorbar="sd",
        markers="o",
        linewidth=2,
    )
    plt.xlabel("Overlap ratio")
    plt.ylabel("Rotation Error (rad)")
    plt.title("Mean ± STD of Relative Rotation Error vs Overlap")
    plt.savefig(f"{prefix}_RRE.png")

    # Plot 2: Relative Translation Error (RTE) vs Overlap
    plt.figure()
    sns.lineplot(
        data=df,
        x="overlap_ratio",
        y="RTE",
        estimator="mean",
        errorbar="sd",
        markers="o",
        linewidth=2,
    )
    plt.xlabel("Overlap ratio")
    plt.ylabel("Translation Error (units)")
    plt.title("Mean ± STD of Relative Translation Error vs Overlap")
    plt.savefig(f"{prefix}_RTE.png")


def main():
    # OVERLAPS = np.arange(1.0, 0.0, -0.05)
    OVERLAPS = [1.0]
    N_ROTATIONS = 1
    VOXEL_SIZE = 0.0035
    records = []

    overlap_progress = Progress(
        TextColumn("[bold green]Overlap {task.completed}/{task.total}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
    )
    rotation_progress = Progress(
        TextColumn("[bold red] Rotation {task.completed}/{task.total}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
    )

    overlap_task = overlap_progress.add_task("overlaps", total=len(OVERLAPS))
    rotation_task = rotation_progress.add_task("rots", total=N_ROTATIONS)

    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(overlap_progress, title="Overlap Progress", border_style="green", padding=(1, 1)),
        Panel.fit(rotation_progress, title="Rotation Progress", border_style="red", padding=(1, 1)),
    )
    for LOAD_WEIGHTS in [True, False]:
        print(f"{LOAD_WEIGHTS=}")
        encoder = load_encoder(load_weights=LOAD_WEIGHTS)
        overlap_progress.reset(overlap_task)
        with Live(progress_table, refresh_per_second=10, transient=True):
            for overlap_ratio in OVERLAPS:
                print(f"Overlap: {overlap_ratio:.2f}")
                rotation_progress.reset(rotation_task)
                for _ in range(N_ROTATIONS):
                    pair = Bunny().split(overlap=overlap_ratio)
                    pair.target, pair.GT_tf_matrix = Transform.random(pair.target, inplace=True)
                    pair.show(pair.GT_tf_matrix)
                    #         pair = model_inference(pair, encoder)
                    #         pred = RANSAC(
                    #             pair.source,
                    #             pair.target,
                    #             ransac_n=4,
                    #             voxel_size=VOXEL_SIZE,
                    #             max_correspondence_distance=VOXEL_SIZE * 1.5 if VOXEL_SIZE is not None else 0.01,
                    #             similarity_threshold=0.9,
                    #             distance_threshold=VOXEL_SIZE * 1.5 if VOXEL_SIZE is not None else 0.01,
                    #             max_iteration=5000,
                    #         )
                    #         print(
                    #             f"Registration result with fitness={pred.fitness:.2f}, inlier_rmse={pred.inlier_rmse:.2f}, correspondence_set_size: {len(pred.correspondence_set)}"
                    #         )
                    #         R_err, t_err = pose_error(pred.transformation, pair.GT_tf_matrix)
                    #         print(f"RRE: {R_err}, RTE: {t_err}\n")
                    #         records.append(
                    #             {
                    #                 "overlap_ratio": overlap_ratio,
                    #                 "RRE": R_err,
                    #                 "RTE": t_err,
                    #             }
                    #         )
                    #         rotation_progress.advance(rotation_task)
            #     over#         lap_progress.advance(overlap_task)

            #     df = pl.#         DataFrame(records)
            #     print(df#         )
            #     df.write#         _csv(f"IAE_registration_load_weights_{LOAD_WEIGHTS}.csv")

            #     plot_df(#         df, prefix=f"IAE_registration_weights_{LOAD_WEIGHTS}")


if __name__ == "__main__":
    main()
