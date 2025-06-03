import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, List, Tuple, Optional
import torch
import matplotlib as mpl
from enum import Enum
from config.config import settings
import LIM.cpp.neighbors.radius_neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling
from debug.decorators import identify_method


class Shape:
    """
    Tuple shape wrapper that lets us indicate that we don't care about some dimensions
    """

    class ANY:
        def __repr__(self) -> str:
            return "ANY"

    expected: tuple[int | ANY] | list[int | ANY]

    def __init__(self, expected: tuple[int | ANY] | list[int | ANY]) -> None:
        self.expected = expected

    def __eq__(self, other: tuple[int | ANY] | list[int | ANY]) -> bool:
        assert isinstance(other, (tuple, list))
        assert all(isinstance(v, int) or v is Shape.ANY for v in other)
        if len(other) != len(self.expected):
            return False
        for exp, oth in zip(self.expected, other):
            if exp is Shape.ANY:
                continue
            if exp != oth:
                return False
        return True


class PCloud:
    device: torch.device
    o3ddevice: str
    pcd: o3d.t.geometry.PointCloud
    _features: torch.Tensor
    _sub: Optional["PCloud"] = None
    _super: Optional["PCloud"] = None

    neighbors: Optional[torch.Tensor] = None
    pools: Optional[torch.Tensor] = None
    upsamples: Optional[torch.Tensor] = None
    path: Optional[Path] = None

    def __init__(self) -> None:
        self.device = "cpu" if settings is None else settings.DEVICE
        PCloud.device = torch.device(self.device)
        PCloud.o3ddevice = "CUDA:0" if self.device.lower() in ["cuda"] else "CPU:0"
        self._features = torch.tensor([], device=self.device, requires_grad=False)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __repr__(self) -> str:
        out = "PCloud("
        out += (
            f"Path['{self.path.parent.parent.stem}/{self.path.parent.stem}/{self.path.stem}'], "
            if self.path is not None and isinstance(self.path, Path)
            else ""
        )
        out += f"Points{[v for v in self.shape]}"
        out += f", Features{[v for v in self.features.shape]}"
        out += f", Upsamples{list(self.upsamples.shape)}" if self.upsamples is not None else ""
        out += ")"
        return out

    def __copy__(self) -> "PCloud":
        cls = self.__class__.from_tensor(self.points.clone())
        cls.features = self.features.clone()
        cls._sub = self._sub
        cls._super = self._super
        cls.path = self.path

        cls.neighbors = self.neighbors
        cls.pools = self.pools
        cls.upsamples = self.upsamples
        return cls

    # =========================================== FACTORY ===========================================#
    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "PCloud":
        if isinstance(path, str):
            path = Path(path)

        if path.suffix.lower() in [".npz", ".npy"]:
            instance = cls.from_arr(np.load(path))
        elif path.suffix.lower() in [".pth"]:
            instance = cls.from_arr(torch.load(path, weights_only=False))
        else:
            instance.pcd = cls(pcd=o3d.t.geometry.PointCloud())
            instance.pcd = o3d.io.read_point_cloud(str(path))

        instance.path = path
        return instance

    @classmethod
    def from_pcd(cls, pcd: o3d.t.geometry.PointCloud) -> "PCloud":
        instance = cls()
        instance.pcd = pcd
        return instance

    @classmethod
    def from_arr(cls, arr: np.ndarray) -> "PCloud":
        assert arr.shape == Shape((Shape.ANY, 3)), "Array must be 3-dimensional"
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud()
        instance.pcd.point.positions = o3d.core.Tensor(
            arr[~np.isnan(arr).any(axis=1)].astype("float32"),  # filter nan points
            o3d.core.float32,
            o3d.core.Device(cls.o3ddevice),
        )
        return instance

    @classmethod
    def from_tensor(cls, tens: torch.Tensor) -> "PCloud":
        instance = cls()
        instance.pcd = o3d.t.geometry.PointCloud().to(o3d.core.Device(cls.o3ddevice))
        instance.pcd.point.positions = o3d.core.Tensor(
            torch.nan_to_num(tens, 0.0).cpu().numpy(),
            o3d.core.Dtype.Float32,
            o3d.core.Device(cls.o3ddevice),
        )
        return instance

    # =========================================== PROPERTIES ===========================================#

    @property
    def tag(self) -> str:
        match self.path:
            case str():
                return self.path
            case list():
                return self.path[0]
            case _:
                return ""

    @property
    def first(self) -> "PCloud":
        current = self
        while current._sub is not None:
            current = current._sub
        return current

    @property
    def last(self) -> "PCloud":
        current = self
        while current._super is not None:
            current = current._super
        return current

    @property
    def arr(self) -> np.ndarray:
        return self.pcd.point.positions.cpu().contiguous().numpy().astype(np.float64)

    @property
    def points(self) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(self.pcd.point.positions.to_dlpack()).to(self.device)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.points.shape

    @property
    def features(self) -> torch.Tensor:
        if not (_EMPTY := len(self._features) == 0):
            return self._features.to(self.device)

        if len(self.shape) == 3:
            self._features = torch.ones((self.shape[0], self.shape[1], 1))
        else:
            self._features = torch.ones((self.shape[0], 1))

        return self._features.to(self.device)

    @property
    def o3d_features(self) -> o3d.pipelines.registration.Feature:
        features = o3d.pipelines.registration.Feature()
        features.data = self.features.cpu().numpy().T
        return features

    @points.setter
    def points(self, value: Union[o3d.core.Tensor, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            value = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(value))
        self.pcd.point.positions = value

    @features.setter
    def features(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._features = value.to(self.device)

    # =========================================== METHODS ===========================================#
    def unsqueeze(self) -> None:
        self.points = self.points.reshape((1, -1, 3))
        self.features = self.features.reshape((1, -1, 1))

    def squeeze(self) -> None:
        self.points = self.points.reshape((-1, 3))
        self.features = self.features.reshape((-1, 1))

    def detach_from_chain(self) -> "PCloud":
        """
        Frees the gradients from all pointss within the [_sub ... self ... _super] chain
        """
        current = self._super
        while current is not None:
            current.features = current.features.detach()
            current = current._super

        current = self._sub
        while current is not None:
            current.features = current.features.detach()
            current = current._sub

        return self

    @identify_method
    def compute_neighbors(self, radius: float, sampleDL: Optional[float]) -> None:
        current = self.last

        points = current.points.cpu().detach().numpy()
        length = torch.tensor([len(points)])
        current.neighbors = cpp_neighbors.batch_query(  # conv_i
            queries=points,
            supports=points,
            q_batches=length,
            s_batches=length,
        )
        current.neighbors = torch.from_numpy(current.neighbors[:, :40].astype(np.int64)).to(self.device)

        if sampleDL is not None:
            subsampled, subsampled_len = cpp_subsampling.subsample_batch(  # pool_p, pool_b
                points=points,
                batches=length.to(dtype=torch.int32),
                sampleDl=sampleDL,
                max_p=0,
                verbose=0,
            )
            subsampled, subsampled_len = (
                torch.from_numpy(subsampled),
                torch.from_numpy(subsampled_len),
            )

            current.pools = cpp_neighbors.batch_query(  # pool_i
                queries=subsampled,
                supports=points,
                q_batches=subsampled_len,
                s_batches=length,
                radius=radius,
            )
            current.pools = torch.from_numpy(current.pools.astype(np.int64)).to(self.device)

            current.upsamples = cpp_neighbors.batch_query(  # up_i
                queries=points,
                supports=subsampled,
                q_batches=length,
                s_batches=subsampled_len,
                radius=2 * radius,
            )
            current.upsamples = torch.from_numpy(current.upsamples.astype(np.int64)).to(self.device)
            current._super = PCloud.from_tensor(subsampled)
            current._super.path = current.path
        else:
            current.pools = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            current.upsamples = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            current._super = PCloud.from_tensor(current.points)
            current._super.path = current.path

        current._super._sub = self
        current._super.features = current.features.detach().clone().to(current.device)


def collate_cloud(batch: List["PCloud"]) -> "PCloud":
    """
    Stacks multiple clouds [N, 3] from a batch into a batch of shape [batch_size, N, 3]
    """
    cloud = PCloud.from_tensor(torch.stack([b.points for b in batch], dim=0))
    cloud.features = torch.stack([b.features.to(b.points.device) for b in batch], dim=0)

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


class Downsampler:
    class Mode(str, Enum):
        RANDOM = "_random_indices"
        PROBABILISTIC = "_probabilistic"

    mode: Mode
    size: int

    def __init__(self, size: int, mode: Mode = Mode.RANDOM) -> None:
        self.size = size
        self.mode = mode

    def __call__(self, cloud: PCloud, *args, **kwargs) -> PCloud:
        temp_cloud = cloud
        match temp_cloud.points.shape:
            case (_BATCH_SIZE, NUM_POINTS, _N_DIM):
                if NUM_POINTS < self.size:
                    return temp_cloud
                temp_cloud = self._downsample_batch(temp_cloud, *args, **kwargs)

            case (NUM_POINTS, _N_DIM):
                if NUM_POINTS < self.size:
                    return temp_cloud
                temp_cloud.unsqueeze()
                temp_cloud = self._downsample_batch(temp_cloud, *args, **kwargs)
                temp_cloud.squeeze()
            case _:
                raise ValueError(f"Expected input tensor to have 2 or 3 dimensions, but got {temp_cloud.points.shape}")
        return temp_cloud

    def _downsample_batch(self, cloud: PCloud, *args, **kwargs) -> PCloud:
        BATCH_SIZE, NUM_POINTS, N_DIM = cloud.points.shape
        indices: torch.Tensor = getattr(self, self.mode)(self.size, cloud, *args, **kwargs)
        cloud.points = torch.gather(cloud.points, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, N_DIM))
        cloud.features = torch.gather(cloud.features, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 1))

        try:  # These might be empty but we don't really care

            def make_o3d_tensor(x):
                o3d.core.Tensor(np.asarray(x)[indices, :], o3d.core.float32, o3d.core.Device(self.o3ddevice))

            cloud.pcd.point.colors = make_o3d_tensor(cloud.pcd.point.colors)
            cloud.pcd.point.normals = make_o3d_tensor(cloud.pcd.point.normals)
        except (KeyError, IndexError):
            pass

        return cloud

    def _random_indices(self, size: int, cloud: PCloud) -> torch.Tensor:
        return torch.randint(low=0, high=size, size=(cloud.points.shape[0], size), device=cloud.device)

    def _probabilistic(self, size: int, cloud: PCloud, scores: torch.Tensor) -> torch.Tensor:
        n = scores.size(0)
        if n < size:
            choice = np.random.choice(n, size)
        else:
            idx = np.arange(n)
            probabilities = (scores / scores.sum()).cpu().numpy().flatten()
            choice = np.random.choice(idx, size=size, replace=False, p=probabilities)

        return torch.from_numpy(choice).unsqueeze(0).to(cloud.device)


class Painter:
    class Cmap:
        value: np.ndarray
        cmap: mpl.colors.Colormap
        compute_normals: bool

        def __init__(
            self,
            value: Union[np.ndarray, torch.tensor, List[float]],
            cmap: Union[str, mpl.colors.Colormap],
            compute_normals: bool = False,
        ) -> None:
            match value:
                case np.ndarray():
                    self.value = value
                case torch.tensor():
                    self.value = value.cpu().numpy()
                case list():
                    self.value = np.array(value)
                case _:
                    raise ValueError(
                        "Painter.Uniform's value accepts one of [np.ndarray, torch.tensor, List[float]],"
                        + f"but got {type(value)}"
                    )
            match cmap:
                case str():
                    self.cmap = mpl.colormaps[cmap]
                case mpl.colors.colormap():
                    self.cmap = cmap
                case _:
                    raise ValueError(f"Painter.Uniform's cmap accepts str or mpl.colors.Colormap, but got {type(cmap)}")

            self.compute_normals = compute_normals

        def __call__(self, cloud: PCloud) -> PCloud:
            cloud.pcd.point.colors = o3d.core.Tensor(
                self.cmap(self.value)[:, :3], o3d.core.float32, o3d.core.Device(cloud.o3ddevice)
            )
            if self.compute_normals:
                cloud = cloud.pcd.estimate_normals()
            return cloud

    class Uniform:
        value: np.ndarray
        compute_normals: bool

        def __init__(self, value: Union[list, np.ndarray], compute_normals: bool = False) -> None:
            match value:
                case list():
                    ...
                case np.ndarray():
                    assert value.shape[-1] == 3, "Array must contain three values: [R, G, B]"
                    value = np.array(value)
                case _:
                    raise ValueError(f"Painter.Cmap's value only accepts lists or numpy arrays, but got {type(value)}")

            self.value = value / 255 if np.any(value > 1) else value
            self.compute_normals = compute_normals

        def __call__(self, cloud: PCloud, to_legacy: bool = True) -> PCloud:
            if to_legacy:
                cloud.pcd = cloud.pcd.to_legacy()
            cloud.pcd.paint_uniform_color(self.value)
            if self.compute_normals:
                cloud.pcd.estimate_normals()
            return cloud
