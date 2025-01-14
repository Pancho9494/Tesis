import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import torch
import copy
import matplotlib as mpl
from enum import Enum
from config import settings
import LIM.cpp.neighbors.radius_neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling

np.random.seed(42)


class Indices:
    _owner: "Cloud"
    _indices: Dict[str, torch.Tensor]

    def __init__(self, owner: "Cloud") -> None:
        self._indices = {}
        self._owner = owner


class Neighbors(Indices):
    def within(self, radius: float) -> torch.Tensor:
        """
        Uses the cpp wrapper function to compute the indices of the neighbors within the given radius, for all points
        in the point cloud

        As different points will have a different ammount of neighbors, the output of this query creates a vector
        big enough to fit the max number of neighbors found:
            [len(self), max_neighbors]
        For the points that don't have enough neighbors the index placed correspond to len(self) + 1
        Later, this points get filtered in the KPConv call, as the influence of the points decreases with the distancem
        a dummy point very far away is concatenated to the point cloud, so all these "fake neighbors" essentialy
        do nothing
        """
        if radius not in self._indices:
            indices = cpp_neighbors.batch_query(
                queries=self._owner.tensor,
                supports=self._owner.tensor,
                q_batches=torch.tensor([len(self._owner)]),
                s_batches=torch.tensor([len(self._owner)]),
                radius=radius,
            )
            self._indices[f"{radius}:2.04f"] = torch.from_numpy(indices.astype(np.int64))
        return self._indices[f"{radius}:2.04f"]


class Pools(Indices):
    def within(self, sampleDL: float, radius: float) -> torch.Tensor:
        if radius not in self._indices:
            indices, lengths = cpp_subsampling.subsample_batch(
                points=self._owner.tensor,
                batches=torch.tensor([len(self._owner)]),
                features=None,
                classes=None,
                sampleDl=sampleDL,
                method="barycenters",
                max_p=0,
                verbose=0,
            )
            indices = cpp_neighbors.batch_query(
                queries=torch.from_numpy(indices),
                supports=self._owner.tensor,
                q_batches=lengths,
                s_batches=torch.tensor([len(self._owner)]),
                radius=radius,
            )
            self._indices[f"{radius}:2.04f"] = torch.from_numpy(indices.astype(np.int64))

        return self._indices[f"{radius}:2.04f"]


class Cloud:
    pcd: o3d.t.geometry.PointCloud
    device: torch.device = torch.device(settings.DEVICE)
    neighbors: Neighbors
    pools: Pools

    _features: torch.Tensor
    __o3ddevice: str = "CUDA:0" if settings.DEVICE.lower() in ["cuda"] else "CPU:0"

    path: Optional[Path] = None

    def __init__(self) -> None:
        self._features = torch.tensor([], device=self.device)
        self.neighbors = Neighbors(owner=self)
        self.pools = Pools(owner=self)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __str__(self) -> str:
        out = f"Cloud(Path: {self.path}, Points{[v for v in self.shape]}"
        out += f"Features{[v for v in self.features.shape]}, Device: {self.device})"
        return out

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "Cloud":
        if isinstance(path, str):
            path = Path(path)

        instance = cls()
        if path.suffix.lower() in [".npz", ".npy"]:
            instance = cls.from_arr(np.load(path))
        elif path.suffix.lower() in [".pth"]:
            instance = cls.from_arr(torch.load(path, weights_only=False))
        else:
            instance.pcd = o3d.t.geometry.PointCloud()
            instance.pcd = o3d.io.read_point_cloud(str(path))

        instance.path = path
        return instance

    @classmethod
    def from_pcd(cls, pcd: o3d.t.geometry.PointCloud) -> "Cloud":
        instance = cls()
        instance.pcd = pcd
        return instance

    @classmethod
    def from_arr(cls, arr: np.ndarray) -> "Cloud":
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud()
        instance.pcd.point.positions = o3d.core.Tensor(
            arr[~np.isnan(arr).any(axis=1)].astype("float32"),  # filter nan points
            o3d.core.float32,
            o3d.core.Device(cls.__o3ddevice),
        )
        return instance

    @classmethod
    def from_tensor(cls, tens: torch.Tensor) -> "Cloud":
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud().to(o3d.core.Device(cls.__o3ddevice))
        tens = torch.nan_to_num(tens, 0.0)
        # B, N, D = tens.shape
        # tens_reshaped = tens.reshape(B * N, D)
        # mask = ~torch.any(tens_reshaped.isnan(), dim=1)
        # tens = tens_reshaped[mask]
        instance.pcd.point.positions = o3d.core.Tensor(
            tens.cpu().numpy(),
            o3d.core.Dtype.Float32,
            o3d.core.Device(cls.__o3ddevice),
        )
        return instance

    @property
    def arr(self) -> np.ndarray:
        return self.pcd.point.positions.cpu().contiguous().numpy().astype(np.float64)

    @property
    def tensor(self) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(self.pcd.point.positions.to_dlpack()).to(self.device)

    @tensor.setter
    def tensor(self, value: Union[o3d.core.Tensor, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            value = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(value))
        self.pcd.point.positions = value

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def features(self) -> torch.Tensor:
        # TODO: this method is ugly
        empty: bool = len(self._features) == 0

        if len(self.shape) == 3:
            if empty:
                self._features = torch.ones((self.shape[0], self.shape[1], 1)).to(self.device)
        else:
            if empty:
                self._features = torch.ones((self.shape[0], 1)).to(self.device)
            # self._features = self._features.reshape(-1, 1) # see features setter, this also gets relaxed

        return self._features

    @features.setter
    def features(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        # Relaxing this rule for a bit, when working with PREDATOR and superpoints we get less and less features
        # assert value.shape[0] == self.shape[0], ValueError(
        #     f"features vector {value.shape[0]} must have same size first dimension as pointcloud {self.shape[0]}"
        # )
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._features = value.to(self.device)

    def paint(self, rgb: Union[List, torch.Tensor], cmap: str = "RdBu", computeNormals: bool = False) -> None:
        """
        Paints the point cloud

        If a simple 3 value list is given then we assume its a uniform color for all points in the point cloud
        If an array is given we expect a (N, 3) array with RGB values for each point in the pointcloud

        I would've like to use match case here but we're stuck with python 3.8

        Args:
            rgb: Either a single RGB color or one RGB color for each point in the point cloud
            cmap: Which matplotlib colormap to use when rgb is an array
            computeNormals: Wether to compute the point cloud normals or not
        """
        if isinstance(rgb, list):
            self.__paint_uniform(rgb)

        elif isinstance(rgb, torch.Tensor):
            self.__paint_array(rgb, cmap)

        if computeNormals:
            self.pcd.estimate_normals()

    def __paint_uniform(self, rgb: list) -> None:
        if any(value > 1 for value in rgb):  # Open3D excpects colors in the [0, 1] range
            rgb = [value / 255 for value in rgb]
        self.pcd.paint_uniform_color(np.array(rgb))

    def __paint_array(self, rgb: Union[torch.Tensor, np.ndarray], cmap: str) -> None:
        assert len(rgb) == len(self), f"Colors array ({rgb.shape}) must match shape of point cloud {self.arr.shape}"
        rgb = rgb.cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        cmap = mpl.colormaps[cmap]
        self.pcd.point.colors = o3d.core.Tensor(cmap(rgb)[:, :3], o3d.core.float32, o3d.core.Device(self.__o3ddevice))

    class DOWNSAMPLE_MODE(Enum):
        RANDOM = "_random_downsample"
        PROBABILISTIC = "_probabilistic_downsample"

    def downsample(self, size: int, mode: DOWNSAMPLE_MODE, **kwargs) -> "Cloud":
        BATCH_SIZE, NUM_POINTS, N_DIM = self.tensor.shape
        instance = copy.deepcopy(self)
        if NUM_POINTS <= size:
            return instance

        method = getattr(self, mode.value)
        idx = method(size, **kwargs)
        instance.tensor = torch.gather(self.tensor, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, N_DIM))
        instance.features = torch.gather(self.features, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 1))
        try:  # Possibly empty arrays
            instance.pcd.point.colors = o3d.core.Tensor(
                np.asarray(instance.pcd.point.colors)[idx, :], o3d.core.float32, o3d.core.Device(self.__o3ddevice)
            )
            instance.pcd.point.normals = o3d.core.Tensor(
                np.asarray(instance.pcd.point.normals)[idx, :], o3d.core.float32, o3d.core.Device(self.__o3ddevice)
            )
        except (KeyError, IndexError):
            pass
        return instance

    def _random_downsample(self, size: int) -> torch.Tensor:
        return torch.randint(low=0, high=size, size=(self.tensor.shape[0], size), device=self.device)

    def _probabilistic_downsample(self, size: int, overlap: torch.Tensor, saliency: torch.Tensor) -> np.ndarray:
        """
        Probabilistic downsample

        Selects the points of the cloud that are most important for registration, based on the given overlap and
        saliency scores

        Args:
            overlap_score: The overlap scores for each point in the original cloud
            saliency_score: The saliency scores for each point in the original cloud
        """
        score = overlap * saliency
        temp_arr = self.arr
        probabilities = (score / score.sum()).numpy().flatten()
        return np.random.choice(np.arange(temp_arr.shape[1]), size, replace=False, p=probabilities)


def collate_cloud(batch: List["Cloud"]) -> "Cloud":
    """
    Stacks multiple clouds [N, 3] from a batch into a batch of shape [batch_size, N, 3]
    """
    cloud = Cloud.from_tensor(torch.stack([b.tensor for b in batch], dim=0))
    cloud.features = torch.stack([b.features.to(b.tensor.device) for b in batch], dim=0)

    cloud.path = [b.path for b in batch]
    try:  # Possibly empty tensors
        cloud.pcd.point.colors = o3d.core.Tensor(
            np.concatenate([np.asarray(b.pcd.point.colors) for b in batch]),
            o3d.core.Dtype.Float32,
            # o3d.core.Device(cls.__o3ddevice),
        )
        cloud.pcd.point.normals = o3d.core.Tensor(
            np.concatenate([np.asarray(b.pcd.point.normals) for b in batch]),
            o3d.core.Dtype.Float32,
            # o3d.core.Device(cls.__o3ddevice),
        )

    except KeyError:
        pass

    return cloud
