import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, List, Tuple, Optional
import torch
import copy
import matplotlib as mpl
from enum import Enum
from config import settings
import LIM.cpp.neighbors.radius_neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling

np.random.seed(42)
from debug.decorators import identify_method


class Cloud:
    pcd: o3d.t.geometry.PointCloud
    _features: torch.Tensor
    neighbors: torch.Tensor
    pools: torch.Tensor
    upsamples: torch.Tensor
    device: torch.device = torch.device(settings.DEVICE)
    o3ddevice: str = "CUDA:0" if settings.DEVICE.lower() in ["cuda"] else "CPU:0"

    subpoints: Optional["Cloud"] = None
    superpoints: Optional["Cloud"] = None
    path: Optional[Path] = None

    def __init__(self) -> None:
        self._features = torch.tensor([], device=self.device, requires_grad=False)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __repr__(self) -> str:
        out = f"Cloud(Points{[v for v in self.shape]}"
        out += f", Features{[v for v in self.features.shape]}, {self.device}"
        out += ")"
        return out

    def __copy__(self) -> "Cloud":
        cls = self.__class__.from_tensor(self.tensor.clone())
        cls.features = self.features.clone()
        cls.subpoints = self.subpoints
        cls.superpoints = self.superpoints

        cls.neighbors = self.neighbors
        cls.pools = self.pools
        cls.upsamples = self.upsamples
        return cls

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
            o3d.core.Device(cls.o3ddevice),
        )
        return instance

    @classmethod
    def from_tensor(cls, tens: torch.Tensor) -> "Cloud":
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud().to(o3d.core.Device(cls.o3ddevice))
        instance.pcd.point.positions = o3d.core.Tensor(
            torch.nan_to_num(tens, 0.0).cpu().numpy(),
            o3d.core.Dtype.Float32,
            o3d.core.Device(cls.o3ddevice),
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
        if not (_EMPTY := len(self._features) == 0):
            return self._features.to(self.device)

        if len(self.shape) == 3:
            self._features = torch.ones((self.shape[0], self.shape[1], 1))
        else:
            self._features = torch.ones((self.shape[0], 1))

        return self._features.to(self.device)

    @features.setter
    def features(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._features = value.to(self.device)

    def show_forwards(self) -> str:
        current = self
        points = []
        features = []

        traveled_layers = 1
        while current is not None:
            points.append(current.tensor.shape)
            features.append(current.features.shape)
            current = current.superpoints

        space = " " * len("points:\t\t") + " " * (len("torch.Size([12345, 3]), ") * traveled_layers)
        out = f"\n{space}current layer\n"
        out += f"{space}v\n"
        out += f"points:\t\t{points}"
        out += f"\nfeatures:\t{features}"
        return out

    def show_backwards(self) -> str:
        current = self
        points = []
        features = []

        traveled_layers = 0
        while current is not None:
            points.append(current.tensor.shape)
            features.append(current.features.shape)
            current = current.subpoints
            traveled_layers += 1

        space = " " * len("points:\t\t") + " " * (len("torch.Size([12345, 3]") * traveled_layers)
        out = f"\n{space}current layer\n"
        out += f"{space}v\n"
        out += f"points:\t\t{list(reversed(points))}"
        out += f"\nfeatures:\t{list(reversed(features))}"
        return out

    # @identify_method
    def compute_neighbors(self, radius: float, sampleDL: Optional[float]) -> None:
        if self.superpoints is not None:
            return

        numpy_tensor = self.tensor.cpu().detach().numpy()
        self.neighbors = cpp_neighbors.batch_query(  # conv_i
            queries=numpy_tensor,
            supports=numpy_tensor,
            q_batches=(length := torch.tensor([len(numpy_tensor)])),
            s_batches=length,
        )
        self.neighbors = torch.from_numpy(self.neighbors.astype(np.int64)).to(self.device)

        if sampleDL is not None:
            subsampled, subsampled_len = cpp_subsampling.subsample_batch(  # pool_p, pool_b
                points=numpy_tensor,
                batches=length.to(dtype=torch.int32),
                sampleDl=sampleDL,
                max_p=0,
                verbose=0,
            )
            subsampled, subsampled_len = torch.from_numpy(subsampled), torch.from_numpy(subsampled_len)

            self.pools = cpp_neighbors.batch_query(  # pool_i
                queries=subsampled,
                supports=numpy_tensor,
                q_batches=subsampled_len,
                s_batches=length,
                radius=radius,
            )
            self.pools = torch.from_numpy(self.pools.astype(np.int64)).to(self.device)

            self.upsamples = cpp_neighbors.batch_query(  # up_i
                queries=numpy_tensor,
                supports=subsampled,
                q_batches=length,
                s_batches=subsampled_len,
                radius=2 * radius,
            )
            self.upsamples = torch.from_numpy(self.upsamples.astype(np.int64)).to(self.device)
            self.superpoints = Cloud.from_tensor(subsampled)
            self.superpoints.subpoints = self
            self.superpoints.features = self.features.clone().to(self.device)
        else:
            self.pools = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            self.upsamples = torch.zeros((0, 1), dtype=torch.int64, device=self.device)

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
        self.pcd.point.colors = o3d.core.Tensor(cmap(rgb)[:, :3], o3d.core.float32, o3d.core.Device(self.o3ddevice))

    class DOWNSAMPLE_MODE(Enum):
        RANDOM = "_random_downsample"
        PROBABILISTIC = "_probabilistic_downsample"

    def downsample(self, size: int, mode: DOWNSAMPLE_MODE, **kwargs) -> "Cloud":
        if must_squeeze := (len(self.tensor.shape) == 2):  # TODO: I think I can make this cleaner, but not right now
            self.tensor = self.tensor.reshape((1, -1, 3))
            self.features = self.features.reshape((1, -1, 1))

        BATCH_SIZE, NUM_POINTS, N_DIM = self.tensor.shape
        instance = copy.deepcopy(self)

        if (NUM_POINTS <= size) or (size == 0):
            if must_squeeze:
                self.tensor = self.tensor.reshape((-1, 3))
                self.features = self.features.reshape((-1, 1))
                instance.tensor = instance.tensor.reshape((-1, 3))
                instance.features = instance.features.reshape((-1, 1))
            return self

        method = getattr(self, mode.value)
        idx = method(size, **kwargs)

        instance.tensor = torch.gather(self.tensor, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, N_DIM))
        instance.features = torch.gather(self.features, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 1))
        try:  # Possibly empty arrays
            instance.pcd.point.colors = o3d.core.Tensor(
                np.asarray(instance.pcd.point.colors)[idx, :], o3d.core.float32, o3d.core.Device(self.o3ddevice)
            )
            instance.pcd.point.normals = o3d.core.Tensor(
                np.asarray(instance.pcd.point.normals)[idx, :], o3d.core.float32, o3d.core.Device(self.o3ddevice)
            )
        except (KeyError, IndexError):
            pass
        if must_squeeze:
            self.tensor = self.tensor.reshape((-1, 3))
            self.features = self.features.reshape((-1, 1))
            instance.tensor = instance.tensor.reshape((-1, 3))
            instance.features = instance.features.reshape((-1, 1))
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
            # o3d.core.Device(cls.o3ddevice),
        )
        cloud.pcd.point.normals = o3d.core.Tensor(
            np.concatenate([np.asarray(b.pcd.point.normals) for b in batch]),
            o3d.core.Dtype.Float32,
            # o3d.core.Device(cls.o3ddevice),
        )

    except KeyError:
        pass

    return cloud
