import copy
import numpy as np
import open3d as o3d
import torch
from typing import Optional, Iterable, Tuple, Any

from config import settings
from LIM.data.structures import PCloud

import LIM.cpp.neighbors.radius_neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling


class Overlaps:
    mix: torch.Tensor
    src: torch.Tensor
    target: torch.Tensor
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self) -> None:
        self.src = torch.tensor([], device=self.device, requires_grad=True)
        self.target = torch.tensor([], device=self.device, requires_grad=True)


class Saliencies:
    mix: torch.Tensor
    src: torch.Tensor
    target: torch.Tensor
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self) -> None:
        self.src = torch.tensor([], device=self.device, requires_grad=True)
        self.target = torch.tensor([], device=self.device, requires_grad=True)


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

    source: PCloud
    target: PCloud
    mix: Optional[PCloud] = None
    overlaps: Overlaps
    saliencies: Saliencies
    _correspondences: Optional[Correspondences] = None
    GT_tf_matrix: Optional[np.ndarray] = None
    prediction: Optional[np.ndarray] = None
    _overlap: Optional[float] = None
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self, source: PCloud, target: PCloud, GT_tf_matrix: Optional[np.ndarray] = None) -> None:
        self.source = source
        self.target = target
        self.overlaps = Overlaps()
        self.saliencies = Saliencies()
        if GT_tf_matrix is not None:
            self.GT_tf_matrix = GT_tf_matrix

    def __repr__(self) -> str:
        out = f"Pair(source={self.source}, target={self.target}"
        out += f", {self.overlap * 100:02.2f}%" if self._overlap is not None else ""
        out += ")"
        return out

    def __iter__(self) -> Iterable[Tuple[PCloud, PCloud]]:
        return iter((self.source, self.target))

    def split(self) -> "Pair":
        # def match_shape(tensor):
        #     if
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

    def set_overlaps_saliencies(self, value: torch.Tensor) -> None:
        sigmoid = torch.nn.Sigmoid()

        FINAL_FEATS_DIM = 32
        self.mix.features = torch.nn.functional.normalize(
            value[:, :FINAL_FEATS_DIM], p=2, dim=1
        )  # final feats dim = 32
        self.overlaps.mix = torch.nan_to_num(
            torch.clamp(sigmoid(value[:, FINAL_FEATS_DIM]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.saliencies.mix = torch.nan_to_num(
            torch.clamp(sigmoid(value[:, FINAL_FEATS_DIM + 1]), min=0, max=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @property
    def correspondences(self) -> Correspondences:
        if self._correspondences is not None:
            return self._correspondences

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
