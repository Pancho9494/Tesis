import copy
import numpy as np
import open3d as o3d
import torch
from typing import Optional, Iterable, Tuple, Any

from config.config import settings
from LIM.data.structures.pcloud import PCloud, Painter

import LIM.cpp.neighbors.radius_neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling
import os

os.environ["XDG_SESSION_TYPE"] = "x11"


class Correspondences:
    matrix: torch.Tensor
    _source_indices: Optional[torch.Tensor] = None
    _target_indices: Optional[torch.Tensor] = None

    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix

    @property
    def source_indices(self) -> torch.Tensor:
        if self._source_indices is None:
            self._source_indices = self.matrix[:, 0].unique().long()
        return self._source_indices

    @property
    def target_indices(self) -> torch.Tensor:
        if self._target_indices is None:
            self._target_indices = self.matrix[:, 1].unique().long()
        return self._target_indices


class Pair:
    """
    Utility class that holds cloud pairs, with the transform that aligns them and their overlap
    """

    id: Optional[str] = None
    source: PCloud
    target: PCloud
    mix: Optional[PCloud] = None
    _correspondences: Optional[Correspondences] = None
    GT_tf_matrix: Optional[np.ndarray] = None
    prediction: Optional[np.ndarray] = None
    _overlap: Optional[float] = None
    device: torch.device

    def __init__(
        self, source: PCloud, target: PCloud, GT_tf_matrix: Optional[np.ndarray] = None, id: Optional[str] = None
    ) -> None:
        self.device = "cpu" if settings is None else torch.device(settings.DEVICE)
        self.id = id
        self.source = source
        self.target = target
        if GT_tf_matrix is not None:
            self.GT_tf_matrix = GT_tf_matrix

    def __repr__(self) -> str:
        out = f"Pair(source={self.source}, target={self.target}"
        # out += f", {self.overlap * 100:02.2f}%" if self._overlap is not None else ""
        out += ")"
        return out

    def __iter__(self) -> Iterable[Tuple[PCloud, PCloud]]:
        return iter((self.source, self.target))

    def overlap(self, threshold: float = 0.03) -> float:
        pcd_tree = o3d.geometry.KDTreeFlann(self.target.pcd.to_legacy())
        match_count = 0
        for i, point in enumerate(self.source.points):
            [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
            if count != 0:
                match_count += 1

        overlap_ratio = match_count / len(self.source.points)
        return overlap_ratio

    def split(self) -> "Pair":
        foo = max(tuple(self.source.shape))
        if len(self.mix.points.shape) == 2:
            self.source.points, self.target.points = (
                self.mix.points[:foo, :],
                self.mix.points[foo:, :],
            )
            self.source.features, self.target.features = (
                self.mix.features[:foo, :],
                self.mix.features[foo:, :],
            )
        elif len(self.mix.points.shape) == 3:
            self.source.points, self.target.points = (
                self.mix.points[:, :, :foo],
                self.mix.points[:, :, foo:],
            )
            self.source.features, self.target.features = (
                self.mix.features[:, :, :foo],
                self.mix.features[:, :, foo:],
            )
        else:
            raise AttributeError(
                f"Expected self.mix.points to have either 2 or 3 dimensions, but got {self.mix.points.shape}"
            )

        return self

    def join(self) -> "Pair":
        if self.mix is None:
            self.mix = copy.copy(self.source)
            self.mix.path = self.source.path + self.target.path
        dimension = torch.argmax(torch.tensor(self.source.points.shape)).item()
        self.mix.points = torch.cat(tensors=(self.source.points, self.target.points), dim=dimension)
        self.mix.features = torch.cat(tensors=(self.source.features, self.target.features), dim=dimension)
        return self

    @property
    def tag(self) -> str:
        return self.source.path[0]

    @property
    def correspondences(self) -> Correspondences:
        if self._correspondences is not None:
            return self._correspondences

        assert self.GT_tf_matrix is not None, "No ground truth transformation given, can't compute correspondences"

        SEARCH_VOXEL_SIZE = 0.0375
        temp_source = self.source.pcd.to_legacy()
        temp_source.transform(self.GT_tf_matrix)
        foo = self.target.pcd.to_legacy()
        pcd_tree = o3d.geometry.KDTreeFlann(foo)
        correspondences = []
        for i, point in enumerate(temp_source.points):
            [count, indices, _] = pcd_tree.search_radius_vector_3d(point, SEARCH_VOXEL_SIZE)
            for j in indices:
                correspondences.append([i, j])

        self._correspondences = Correspondences(torch.tensor(correspondences, device=self.device, requires_grad=False))
        self._correspondences.source_indices
        self._correspondences.target_indices
        return self._correspondences

    # @identify_method
    def compute_neighbors(self, radius: float, sampleDl: Optional[float]) -> None:
        current_mix = self.mix.last
        current_source = self.source.last
        current_target = self.target.last

        points = current_mix.points.cpu().detach().numpy()
        lengths = torch.tensor([len(current_source.points), len(current_target.points)])
        current_mix.neighbors = cpp_neighbors.batch_query(  # conv_i
            queries=points,
            supports=points,
            q_batches=lengths,
            s_batches=lengths,
            radius=radius,
        )
        current_mix.neighbors = torch.from_numpy(current_mix.neighbors[:, :40].astype(np.int64)).to(self.device)

        if sampleDl is not None:
            subsampled, subsampled_lens = cpp_subsampling.subsample_batch(  # pool_p, pool_b
                points=points,
                batches=lengths.to(dtype=torch.int32),
                sampleDl=sampleDl,
                max_p=0,
                verbose=0,
            )
            subsampled, subsampled_lens = (
                torch.from_numpy(subsampled),
                torch.from_numpy(subsampled_lens),
            )

            current_mix.pools = cpp_neighbors.batch_query(  # pool_i
                queries=subsampled,
                supports=points,
                q_batches=subsampled_lens,
                s_batches=lengths,
                radius=radius,
            )
            current_mix.pools = torch.from_numpy(current_mix.pools.astype(np.int64)).to(self.device)

            current_mix.upsamples = cpp_neighbors.batch_query(  # up_i
                queries=points,
                supports=subsampled,
                q_batches=lengths,
                s_batches=subsampled_lens,
                radius=2 * radius,
            )
            current_mix.upsamples = torch.from_numpy(current_mix.upsamples.astype(np.int64)).to(self.device)
            current_mix._super = PCloud.from_tensor(subsampled)
            current_mix._super.path = current_mix.path

            source_length = subsampled_lens[0].detach().cpu().numpy()
            current_source._super = PCloud.from_tensor(subsampled[:source_length])
            current_source._super._sub = current_source
            current_source._super.path = current_source.path
            current_target._super = PCloud.from_tensor(subsampled[source_length:])
            current_target._super._sub = current_target
            current_target._super.path = current_target.path
        else:
            current_mix.pools = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            current_mix.upsamples = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            current_mix._super = PCloud.from_tensor(current_mix.points)
            current_mix._super.path = current_mix.path

        current_mix._super._sub = current_mix
        current_mix._super.features = current_mix.features.detach().clone().to(self.device)

    def to_legacy(self) -> dict[str, Any]:
        """
        dict_keys(['points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths', 'rot', 'trans', 'correspondences',
        'src_pcd_raw', 'tgt_pcd_raw', 'sample'])
        """
        batch = {}
        batch["points"] = []
        batch["neighbors"] = []
        batch["pools"] = []
        batch["upsamples"] = []
        batch["stack_lengths"] = []

        current_mix = self.mix.first
        current_src = self.source.first
        current_tgt = self.target.first
        while current_mix is not None:
            batch["points"].append(current_mix.points)
            batch["neighbors"].append(current_mix.neighbors)
            batch["pools"].append(current_mix.pools)
            batch["upsamples"].append(current_mix.upsamples)
            batch["stack_lengths"].append((len(current_src.points), len(current_tgt.points)))

            current_mix = current_mix._super
            current_src = current_src._super if current_src._super is not None else current_src
            current_tgt = current_tgt._super if current_tgt._super is not None else current_tgt

        batch["features"] = self.mix.features
        batch["rot"] = self.GT_tf_matrix[:3, :3]
        batch["trans"] = self.GT_tf_matrix[:3, 3].reshape((3, 1))
        batch["correspondences"] = self.correspondences.matrix

        return batch

    def show(self, predicted_tf: np.ndarray | None = None) -> None:
        WIDTH, HEIGHT = 1280, 720
        ROTATE_X, ROTATE_Y = 1.0, 0.0
        YELLOW, BLUE = np.array([1.0, 0.706, 0.0]), np.array([0.0, 0.651, 0.929])
        WHITE = np.array([1, 1, 1])

        src_pcd = Painter.Uniform(YELLOW, compute_normals=True)(self.source)
        tgt_pcd = Painter.Uniform(BLUE, compute_normals=True)(self.target)

        gt_src_pcd = copy.deepcopy(src_pcd).pcd.transform(self.GT_tf_matrix)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="raw", width=WIDTH, height=HEIGHT, left=0, top=HEIGHT)
        vis.add_geometry(src_pcd.pcd)
        vis.add_geometry(tgt_pcd.pcd)

        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name="ground truth", width=WIDTH, height=HEIGHT, left=WIDTH, top=HEIGHT)
        vis2.add_geometry(gt_src_pcd)
        vis2.add_geometry(tgt_pcd.pcd)

        if predicted_tf is not None:
            vis3 = o3d.visualization.Visualizer()
            vis3.create_window(window_name="predicted", width=WIDTH, height=HEIGHT, left=WIDTH, top=int(HEIGHT * 1.7))
            pred_src_pcd = copy.deepcopy(src_pcd.pcd)
            pred_src_pcd.transform(predicted_tf)
            vis3.add_geometry(pred_src_pcd)
            vis3.add_geometry(tgt_pcd.pcd)

        while True:
            vis.update_geometry(src_pcd.pcd)
            vis.update_geometry(tgt_pcd.pcd)
            if not vis.poll_events():
                break
            ctr = vis.get_view_control()
            ctr.rotate(ROTATE_X, ROTATE_Y)
            vis.update_renderer()

            vis2.update_geometry(gt_src_pcd)
            vis2.update_geometry(tgt_pcd.pcd)
            if not vis2.poll_events():
                break
            ctr = vis2.get_view_control()
            ctr.rotate(ROTATE_X, ROTATE_Y)
            vis2.update_renderer()

            if predicted_tf is not None:
                vis3.update_geometry(pred_src_pcd)
                vis3.update_geometry(tgt_pcd.pcd)
                if not vis3.poll_events():
                    break
                ctr = vis3.get_view_control()
                ctr.rotate(ROTATE_X, ROTATE_Y)
                vis3.update_renderer()

        vis.destroy_window()
        vis2.destroy_window()

        if predicted_tf is not None:
            vis3.destroy_window()
