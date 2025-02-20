import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, List, Tuple, Optional
import torch
import matplotlib as mpl
from enum import Enum
from config import settings
import LIM.cpp.neighbors.neighbors as cpp_neighbors
import LIM.cpp.subsampling.grid_subsampling as cpp_subsampling


class PCloud:
    device: torch.device = torch.device(settings.DEVICE)
    o3ddevice: str = "CUDA:0" if settings.DEVICE.lower() in ["cuda"] else "CPU:0"

    pcd: o3d.t.geometry.PointCloud
    _features: torch.Tensor
    _sub: Optional["PCloud"] = None
    _super: Optional["PCloud"] = None

    neighbors: Optional[torch.Tensor] = None
    pools: Optional[torch.Tensor] = None
    upsamples: Optional[torch.Tensor] = None
    path: Optional[Path] = None

    def __init__(self) -> None:
        self._features = torch.tensor([], device=self.device, requires_grad=False)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __repr__(self) -> str:
        out = f"PCloud(Points{[v for v in self.shape]}"
        out += f", Features{[v for v in self.features.shape]}"
        out += f", Upsamples{self.upsamples.shape}" if self.upsamples is not None else ""
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

    def compute_neighbors(self, radius: float, sampleDL: Optional[float]) -> None:
        if self._super is not None:
            return

        numpy_tensor = self.points.cpu().detach().numpy()
        self.neighbors = cpp_neighbors.batch_query(  # conv_i
            queries=numpy_tensor,
            supports=numpy_tensor,
            q_batches=(length := torch.tensor([len(numpy_tensor)])),
            s_batches=length,
        )
        print("============================")
        print(self.neighbors.shape)
        print("============================")
        self.neighbors = torch.from_numpy(self.neighbors[:, :40].astype(np.int64)).to(self.device)

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
            self._super = PCloud.from_tensor(subsampled)
        else:
            self.pools = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            self.upsamples = torch.zeros((0, 1), dtype=torch.int64, device=self.device)
            self._super = PCloud.from_tensor(self.points)

        self._super._sub = self
        self._super.features = self.features.detach().clone().to(self.device)


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

    mode: Mode
    size: int

    def __init__(self, size: int, mode: Mode = Mode.RANDOM) -> None:
        self.size = size
        self.mode = mode

    def __call__(self, cloud: PCloud) -> PCloud:
        match cloud.points.shape:
            case (_BATCH_SIZE, NUM_POINTS, _N_DIM):
                if NUM_POINTS < self.size:
                    return cloud
                cloud = self._downsample_batch(cloud)

            case (NUM_POINTS, _N_DIM):
                if NUM_POINTS < self.size:
                    return cloud
                cloud.unsqueeze()
                cloud = self._downsample_batch(cloud)
                cloud.squeeze()
            case _:
                raise ValueError(f"Expected input tensor to have 2 or 3 dimensions, but got {cloud.points.shape}")
        return cloud

    def _downsample_batch(self, cloud: PCloud) -> PCloud:
        BATCH_SIZE, NUM_POINTS, N_DIM = cloud.points.shape
        indices: torch.Tensor = getattr(self, self.mode)(self.size, cloud)
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
                    assert len(value) == 3, "Array must contain three values: [R, G, B]"
                    value = np.array(value)
                case _:
                    raise ValueError(f"Painter.Cmap's value only accepts lists or numpy arrays, but got {type(value)}")

            self.value = value / 255 if np.any(value > 1) else value
            self.compute_normals = compute_normals

        def __call__(self, cloud: PCloud) -> PCloud:
            cloud.pcd.paint_uniform_color(self.value)
            if self.compute_normals:
                cloud.pcd.estimate_normals()
            return cloud
