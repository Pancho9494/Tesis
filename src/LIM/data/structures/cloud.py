import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, List, Tuple
import torch
import copy
import matplotlib as mpl
from enum import Enum


class Cloud:
    pcd: o3d.t.geometry.PointCloud
    _features: torch.Tensor
    __o3ddevice: str = "CUDA:1" if torch.cuda.is_available() else "CPU:0"
    __device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self) -> None:
        self._features = torch.tensor([], device=self.__device)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __str__(self) -> str:
        return f"Cloud(points={self.shape}, features={self.features.shape})"

    @classmethod
    def from_path(cls, path: Path) -> "Cloud":
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud()
        if path.suffix.lower() in [".npz", ".npy"]:
            data = np.load(path)
            instance.pcd.point.positions = o3d.core.Tensor(
                list(data.values())[0], o3d.core.float32, o3d.core.Device(cls.__o3ddevice)
            )
        else:
            instance.pcd = o3d.io.read_point_cloud(str(path))
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
            arr.astype("float32"), o3d.core.float32, o3d.core.Device(cls.__o3ddevice)
        )
        return instance

    @property
    def arr(self) -> np.ndarray:
        return self.pcd.point.positions.numpy()

    @property
    def tensor(self) -> torch.Tensor:
        tensor = torch.utils.dlpack.from_dlpack(self.pcd.point.positions.to_dlpack()).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.arr.shape

    @property
    def features(self) -> torch.Tensor:
        if len(self._features) == 0:
            self._features = torch.from_numpy(np.ones(self.shape[0])).to(
                self.__device
            )  # torch.ones() is not working somehow with rocm???
        return self._features

    @features.setter
    def features(self, value: torch.Tensor) -> None:
        assert value.shape[0] == self.shape[0], ValueError(
            f"features vector {value.shape[0]} must have same size first dimension as pointcloud {self.shape[0]}"
        )
        self._features = value.to(self.__device)

    def paint(self, rgb: Union[List, np.ndarray], cmap: str = "RdBu", computeNormals: bool = False) -> None:
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

        elif isinstance(rgb, np.ndarray):
            self.__paint_array(rgb, cmap)

        if computeNormals:
            self.pcd.estimate_normals()

    def __paint_uniform(self, rgb: list) -> None:
        if any(value > 1 for value in rgb):  # Open3D excpects colors in the [0, 1] range
            rgb = [value / 255 for value in rgb]
        self.pcd.paint_uniform_color(np.array(rgb))

    def __paint_array(self, rgb: np.ndarray, cmap: str) -> None:
        assert len(rgb) == len(self), f"Colors array ({rgb.shape}) must match shape of point cloud {self.arr.shape}"
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        cmap = mpl.colormaps[cmap]
        self.pcd.point.colors = o3d.core.Tensor(cmap(rgb)[:, :3], o3d.core.float32, o3d.core.Device(self.__o3ddevice))

    class DOWNSAMPLE_MODE(Enum):
        RANDOM = "_random_downsample"
        PROBABILISTIC = "_probabilistic_downsample"

    def downsample(self, size: int, mode: DOWNSAMPLE_MODE, inplace=True, **kwargs) -> Union["Cloud", None]:
        instance = self if inplace else copy.deepcopy(self)

        if instance.arr.shape[0] <= size:
            return None if inplace else instance

        method = getattr(self, mode.value)
        idx = method(size, **kwargs)
        instance.pcd.point.positions = o3d.core.Tensor(
            self.arr[idx, :], o3d.core.float32, o3d.core.Device(self.__o3ddevice)
        )
        instance.features = instance.features[idx]
        try:  # Possibly empty arrays
            instance.pcd.point.colors = o3d.core.Tensor(
                np.asarray(instance.pcd.point.colors)[idx, :], o3d.core.float32, o3d.core.Device(self.__o3ddevice)
            )
            instance.pcd.point.normals = o3d.core.Tensor(
                np.asarray(instance.pcd.point.normals)[idx, :], o3d.core.float32, o3d.core.Device(self.__o3ddevice)
            )
        except KeyError:
            pass
        except IndexError:
            pass

        return None if inplace else instance

    def _random_downsample(self, size: int) -> np.ndarray:
        return np.random.randint(self.arr.shape[0], size=size)

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
        return np.random.choice(np.arange(temp_arr.shape[0]), size, replace=False, p=probabilities)

    @classmethod
    def collate(cls, batch: List["Cloud"]) -> "Cloud":
        """
        Stacks multiple clouds [N, 3] from a batch into a batch of shape [batch_size, N, 3]
        """
        cloud = Cloud.from_arr(np.concatenate([b.arr for b in batch]))
        cloud.features = torch.cat([b.features for b in batch])

        try:  # Possibly empty tensors
            cloud.pcd.point.colors = o3d.core.Tensor(
                np.concatenate([np.asarray(b.pcd.point.colors) for b in batch]),
                o3d.core.Dtype.Float32,
                o3d.core.Device.CUDA,
            )
            cloud.pcd.point.normals = o3d.core.Tensor(
                np.concatenate([np.asarray(b.pcd.point.normals) for b in batch]),
                o3d.core.Dtype.Float32,
                o3d.core.Device.CUDA,
            )
        except KeyError:
            pass

        return cloud
