from pathlib import Path
from typing import Any
import numpy as np
import LIM.log as log
import urllib.request
import tarfile
import open3d as o3d
import os
from enum import Enum
from functools import partial
from LIM.data.structures import PCloud, Pair

os.environ["XDG_SESSION_TYPE"] = "x11"


class Bunny:
    _path: Path = Path("./src/LIM/data/raw/bunny")
    pcd: o3d.geometry.PointCloud

    class Midpoint(Enum):
        MEAN = partial(np.mean)
        MEDIAN = partial(np.median)

        def __call__(self, *args, **kwargs) -> Any:
            return self.value(*args, **kwargs)

    def __init__(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        self.__download_source()
        self.pcd = o3d.io.read_point_cloud(self._path / "reconstruction/bun_zipper.ply")
        for overlap_ratio in [i / 10.0 for i in range(10, -1, -1)]:
            print(f"Showing ~{overlap_ratio * 100:0.0f}% of overlap from centroid", end="")
            self.split(overlap=overlap_ratio)

    def __download_source(self) -> None:
        if any(self._path.iterdir()):
            return

        SOURCE = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        log.info(f"Bunny dir was not found at {self._path}, downloading it from {SOURCE}")
        urllib.request.urlretrieve(url=SOURCE, filename=(tar_path := self._path.parent / "bunny.tar.gz"))
        log.info(f"Succesfully downloaded bunny file from {SOURCE}")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self._path.parent)
            log.info(f"Succesfully extracted tar to {self._path}")
        tar_path.unlink()

    def split(self, overlap: float | int) -> Pair:
        match overlap:
            case float():
                assert overlap >= 0.0 and overlap <= 1.0
            case int():
                assert overlap >= 0 and overlap <= 100
                overlap /= 100.0

        centroid = self.find_midpoint(via=Bunny.Midpoint.MEDIAN)
        points = np.asarray(self.pcd.points)
        delta = overlap * (np.max(points[:, 0]) - np.min(points[:, 0]))
        pair = Pair(
            source=PCloud.from_arr(points[points[:, 0] > centroid[0] - delta / 2.0]),
            target=PCloud.from_arr(points[points[:, 0] <= centroid[0] + delta / 2.0]),
        )
        print(f" Real overlap: {pair.overlap(0.00001):0.2f}")
        pair.show()
        return pair

    def find_midpoint(self, via: Midpoint) -> list[float]:
        return [via(ax) for ax in np.asarray(self.pcd.points).T]
