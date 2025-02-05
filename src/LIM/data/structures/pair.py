import open3d as o3d
import numpy as np
from LIM.data.structures import Cloud
from typing import Optional, Iterable, Tuple
import torch
from config import settings


class Overlaps:
    src: torch.Tensor
    target: torch.Tensor
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self) -> None:
        self.src = torch.tensor([], device=self.device, requires_grad=True)
        self.target = torch.tensor([], device=self.device, requires_grad=True)


class Saliencies:
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

    _source: Cloud
    _target: Cloud
    overlaps: Overlaps
    saliencies: Saliencies
    _correspondences: Optional[Correspondences] = None
    GT_tf_matrix: Optional[np.ndarray] = None
    prediction: Optional[np.ndarray] = None
    _overlap: Optional[float] = None
    device: torch.device = torch.device(settings.DEVICE)

    def __init__(self, source: Cloud, target: Cloud, GT_tf_matrix: Optional[np.ndarray] = None) -> None:
        self._source = source
        self._target = target
        self.overlaps = Overlaps()
        self.saliencies = Saliencies()
        if GT_tf_matrix is not None:
            self.GT_tf_matrix = GT_tf_matrix

    @property
    def source(self) -> Cloud:
        return self._source

    @source.setter
    def source(self, value: Cloud) -> None:
        sigmoid = torch.nn.Sigmoid()

        HAS_SUPERPOINTS = value.superpoints is not None
        NO_SUBPOINTS = value.subpoints is None
        NON_EMPTY_FEATURES = len(value.features) > 0
        if _PREDATOR_LAST_LAYER := HAS_SUPERPOINTS and NO_SUBPOINTS and NON_EMPTY_FEATURES:
            self.overlaps.src = torch.nan_to_num(
                torch.clamp(sigmoid(value.features[:, -2]), min=0, max=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self.saliencies.src = torch.nan_to_num(
                torch.clamp(sigmoid(value.features[:, -1]), min=0, max=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            value.features = torch.nn.functional.normalize(value.features[:, :-2], p=2, dim=1)
        self._source = value

    @property
    def target(self) -> Cloud:
        return self._target

    @target.setter
    def target(self, value: Cloud) -> None:
        sigmoid = torch.nn.Sigmoid()

        HAS_SUPERPOINTS = value.superpoints is not None
        NO_SUBPOINTS = value.subpoints is None
        NON_EMPTY_FEATURES = len(value.features) > 0
        if _PREDATOR_LAST_LAYER := HAS_SUPERPOINTS and NO_SUBPOINTS and NON_EMPTY_FEATURES:
            self.overlaps.target = torch.nan_to_num(
                torch.clamp(sigmoid(value.features[:, -2]), min=0, max=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self.saliencies.target = torch.nan_to_num(
                torch.clamp(sigmoid(value.features[:, -1]), min=0, max=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            value.features = torch.nn.functional.normalize(value.features[:, :-2], p=2, dim=1)
        self._target = value

    def __repr__(self) -> str:
        out = f"Pair(source={self.source}, target={self.target}"
        out += f", {self.overlap * 100:02.2f}%" if self._overlap is not None else ""
        out += ")"
        return out

    def __iter__(self) -> Iterable[Tuple[Cloud, Cloud]]:
        return iter((self.source, self.target))

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
        return self._correspondences
