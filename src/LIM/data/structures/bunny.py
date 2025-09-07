from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from LIM.data.structures import PCloud
from LIM.log import log

os.environ["XDG_SESSION_TYPE"] = "x11"


class Bunny:
    _path: Path = Path("./src/LIM/data/raw/bunny")
    cloud: PCloud
    T: np.ndarray
    _seed: int | None

    class Midpoint(Enum):
        MEAN = partial(np.mean)
        MEDIAN = partial(np.median)

        def __call__(self, *args, **kwargs) -> Any:
            return self.value(*args, **kwargs)

    def __init__(self, seed: int | None = None) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        self.__download_source()
        self.cloud = PCloud.from_path(self._path / "reconstruction/bun_zipper.ply")
        self.T = np.eye(4, 4)
        self._seed = seed

    def _rng(self, idx: int | None, tag: str) -> np.random.Generator:
        """
        Deterministic per-index RNG when both self._seed and idx are provided.
        Distinct 'tag' values yield independent substreams for the same (seed, idx).
        """
        if (self._seed is None) or (idx is None):
            return np.random.default_rng()
        ss = np.random.SeedSequence(
            [
                int(self._seed),
                int(idx),
                int.from_bytes(
                    hashlib.sha256(tag.encode("utf-8")).digest()[:4],
                    "little",
                ),
            ]
        )
        return np.random.Generator(np.random.PCG64(ss))

    def __download_source(self) -> None:
        if any(self._path.iterdir()):
            return

        SOURCE = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        log.info(f"Bunny dir was not found at {self._path}, downloading it from {SOURCE}")
        urllib.request.urlretrieve(
            url=SOURCE,
            filename=(tar_path := self._path.parent / "bunny.tar.gz"),
        )
        log.info(f"Succesfully downloaded bunny file from {SOURCE}")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self._path.parent)
            log.info(f"Succesfully extracted tar to {self._path}")
        tar_path.unlink()

    def _new_bunny(self, applied_T: np.ndarray) -> Bunny:
        out_bunny = Bunny(seed=self._seed)
        out_bunny.T = deepcopy(self.T)
        if not isinstance(self.cloud.pcd, o3d.pybind.geometry.PointCloud):  # pyright: ignore[reportAttributeAccessIssue]
            pcd_copy = deepcopy(self.cloud.pcd.clone())
        else:
            pcd_copy = o3d.geometry.PointCloud(deepcopy(self.cloud.pcd))
        pcd_copy.transform(applied_T)
        out_bunny.cloud.pcd = pcd_copy
        out_bunny.T = applied_T @ out_bunny.T
        return out_bunny

    def split(self, overlap: float | int | None = None, axis: int | str | None = None, idx: int | None = None) -> Bunny:
        if overlap is None:
            if hasattr(self, "_rng"):
                rng = self._rng(idx, tag="split")
                overlap = float(rng.random())
            else:
                overlap = float(np.random.random())
        else:
            match overlap:
                case float():
                    assert 0.0 <= overlap <= 1.0, "overlap float must be in [0,1]"
                case int():
                    assert 0 <= overlap <= 100, "overlap int must be in [0,100]"
                    overlap = overlap / 100.0
                case _:
                    raise TypeError("overlap must be float, int, or None")

        if axis is None:
            rng = self._rng(idx, tag="split_axis") if hasattr(self, "_rng") else np.random.default_rng()
            ax = rng.integers(0, 3, dtype=int)
        elif isinstance(axis, str):
            ax = {"x": 0, "y": 1, "z": 2}[axis]
        else:
            ax = int(axis)

        points = np.asarray(self.cloud.points.cpu())
        axis_points = points[:, ax]

        q_low = np.quantile(axis_points, 0.5 - overlap / 2.0)
        q_high = np.quantile(axis_points, 0.5 + overlap / 2.0)

        low = axis_points <= q_low
        band = (axis_points > q_low) & (axis_points <= q_high)
        high = axis_points > q_high

        mask_S = band | low
        mask_T = band | high

        out_bunny = self._new_bunny(np.eye(4, 4))
        out_bunny.cloud.pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                points[mask_T].copy(),
            )
        )
        self.cloud = PCloud()
        self.cloud.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(deepcopy(points)[mask_S]))
        return out_bunny

    def implicit(self) -> Bunny:
        out_bunny = self._new_bunny(np.eye(4, 4))
        out_bunny.cloud.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self._to_unit_cube(out_bunny.cloud)))
        return out_bunny

    def _to_unit_cube(self, cloud: PCloud, z_level: float = 0.5) -> np.ndarray:
        points = np.asarray(self.cloud.points.cpu())
        bbox_min, bbox_max = points.min(axis=0), points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        scale = 1.0 / (bbox_max - bbox_min).max()

        T_center_to_origin = np.eye(4)
        T_center_to_origin[:-1, -1] = -bbox_center
        T_uniform_scale = np.eye(4) * scale
        T_uniform_scale[-1, -1] = 1.0

        hom = np.c_[points, np.ones((points.shape[0], 1))]
        points_cs = (T_uniform_scale @ (T_center_to_origin @ hom.T)).T[:, :3]
        z_min = points_cs[:, 2].min()
        T_set_bottom_z = np.eye(4)
        T_set_bottom_z[2, -1] = -z_min + z_level

        T_unit_cube = T_set_bottom_z @ T_uniform_scale @ T_center_to_origin
        return (T_unit_cube @ hom.T).T[:, :3]

    def rotate(
        self,
        R: np.ndarray | None = None,
        center: np.ndarray | None = None,
        idx: int | None = None,
    ) -> Bunny:
        """
        Returns a NEW Bunny rotated by applying a random rotation (uniform over SO(3))
        around the point cloud centroid
        """
        if R is None:
            # Sample a uniform random unit quaternion
            u1, u2, u3 = self._rng(idx, tag="rotate").random(3)
            qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
            qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
            qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
            qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

            # Convert quaternion (x, y, z, w) to rotation matrix
            x, y, z, w = qx, qy, qz, qw
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z

            R = np.array(
                [
                    [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
                ],
                dtype=np.float64,
            )
        if center is None:
            c = self.cloud.arr.mean(axis=0)
        else:
            c = np.asarray(center, dtype=np.float64).reshape(3)

        # Rotation about c:  x' = R x + t, with t = (I - R) c
        applied_T = np.eye(4, dtype=np.float64)
        applied_T[:3, :3] = R
        applied_T[:3, 3] = c - R @ c
        return self._new_bunny(applied_T)

    def translate(
        self, t: list[float] | np.ndarray | None = None, *, frac_of_bbox: float = 0.5, idx: int | None = None
    ) -> Bunny:
        """
        Return a NEW Bunny translated by vector `t`.

        Args:
            t: (dx, dy, dz). If None, a random translation is sampled.
            frac_of_bbox: Max fraction (per axis) of the bbox extent for random t.
        """

        def _cast_to_np3(value: Any) -> np.ndarray:
            try:
                return value.cpu().numpy().astype(np.float64)
            except AttributeError:
                return np.asarray(value, dtype=np.float64)

        if t is None:
            minb = _cast_to_np3(self.cloud.pcd.get_min_bound())
            maxb = _cast_to_np3(self.cloud.pcd.get_max_bound())
            extent = maxb - minb
            t = (self._rng(idx, tag="translate").random(3) * 2.0 - 1.0) * (frac_of_bbox * extent)

        t = np.asarray(t, dtype=np.float64).reshape(3)
        applied_T = np.eye(4, dtype=np.float64)
        applied_T[:3, 3] = t
        return self._new_bunny(applied_T)

    def noise(
        self,
        mu: float | list[float] | np.ndarray = 0.0,
        sigma: float | list[float] | np.ndarray = 0.005,
        idx: int | None = None,
    ) -> Bunny:
        """
        Return a NEW Bunny with i.i.d. Gaussian noise added to each point.
        `mu` and `sigma` can be scalars (same for all axes) or 3-vectors.
        """
        pts = np.asarray(self.cloud.pcd.points, dtype=np.float64)  # (N, 3)
        N = pts.shape[0]

        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if mu.size == 1:
            mu = np.full(3, mu.item())
        if sigma.size == 1:
            sigma = np.full(3, sigma.item())

        mu = mu.reshape(1, 3)
        sigma = sigma.reshape(1, 3)

        noise = self._rng(idx, tag="noise").normal(loc=mu, scale=sigma, size=(N, 3))
        new_pts = pts + noise

        out_bunny = deepcopy(self)
        out_bunny.cloud.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_pts))
        out_bunny.T = self.T
        return out_bunny
