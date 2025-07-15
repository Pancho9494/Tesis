from __future__ import annotations

import functools
import json
import random
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List

import numpy as np
import torchvision
from tqdm import tqdm

import LIM.log as log
from config.config import settings
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures import transform_factory
from LIM.data.structures.pcloud import PCloud, collate_cloud

executor = ThreadPoolExecutor()


def _check_file(filepath: Path) -> bool:
    delete = False
    try:
        np.load(str(filepath))
    except (zipfile.BadZipFile, EOFError, ValueError, FileNotFoundError):
        delete = True

    return delete


class ScanNet(CloudDatasetsI):
    """
    Class representing the ScanNet dataset

    The hierarchy of the files goes like this:
        * room:
            * scene:
                * points_iou:
                    * points_iou_xx.npz
                * pointcloud:
                    * pointcloud_xx.npz

    Here the actual pointclouds are stored in the ['points'] field inside the pointcloud_xx.npz files

    """

    dir: Path = Path("./src/LIM/data/raw/scannet-clean")
    paths: List[Path]
    split: CloudDatasetsI.SPLITS | None = None
    _cached_len: int | None = None

    def __init__(self) -> None:
        self.rooms = self.dir.iterdir()

    def __repr__(self) -> str:
        if self._cached_len is None:
            samples = []
            for scene in self.scenes:
                samples.extend(scene.rglob("pointcloud/**/*.bin"))
            self._cached_len = len(samples)
        return f"ScanNet({self._cached_len})"

    def __len__(self) -> int:
        return len(self.scenes) if hasattr(self, "scenes") else 0

    def __getitem__(self, idx: int) -> tuple[PCloud, PCloud, PCloud]:
        cloud: PCloud
        implicit_l1: PCloud
        implicit_iou: PCloud

        def _load_binary(path: Path) -> np.ndarray:
            with open(f"{path}.json", "r") as f:
                metadata = json.load(f)
            return np.fromfile(f"{path}.bin", dtype=metadata["dtype"]).reshape(metadata["shape"])

        def _load_pointcloud(path: Path) -> np.ndarray:
            return _load_binary(path / "points")

        def _load_iou(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            df_value = _load_binary(path / "df_value")
            occupancies = _load_binary(path / "occupancies")
            points_iou = _load_binary(path / "points")
            return (df_value, occupancies, points_iou)

        scene = self.scenes[idx]

        choices_pointcloud = random.choice(list((scene / "pointcloud").iterdir()))
        idx_pc = int(choices_pointcloud.stem[-2:])
        choices_points_iou = list((scene / "points_iou").iterdir())
        idx_l1 = int(random.choice(choices_points_iou).stem[-2:])
        idx_iou = int(random.choice(choices_points_iou).stem[-2:])

        pointcloud_future = executor.submit(_load_pointcloud, scene / f"pointcloud/pointcloud_{idx_pc:02d}")
        l1_future = executor.submit(_load_iou, scene / f"points_iou/points_iou_{idx_l1:02d}")
        iou_future = executor.submit(_load_iou, scene / f"points_iou/points_iou_{idx_iou:02d}")
        (points), (l1_df, l1_occ, l1_points), (iou_df, iou_occ, iou_points) = (
            pointcloud_future.result(),
            l1_future.result(),
            iou_future.result(),
        )

        cloud = PCloud.from_arr(points.astype(np.float32))
        cloud.path = scene / f"pointcloud/pointcloud_{idx_pc:02d}.npz"

        implicit_l1 = PCloud.from_arr(l1_points.astype(np.float32))
        implicit_l1.features = np.expand_dims(l1_df.astype(np.float32), axis=1)
        implicit_l1.path = scene / f"points_iou/points_iou_{idx_l1:02d}.npz"

        implicit_iou = PCloud.from_arr(iou_points.astype(np.float32))
        implicit_iou.features = np.expand_dims(iou_df.astype(np.float32), axis=1)
        implicit_iou.path = scene / f"points_iou/points_iou_{idx_iou:02d}.npz"

        return cloud, implicit_l1, implicit_iou

    @classmethod
    def new_instance(cls, split: CloudDatasetsI.SPLITS) -> ScanNet:
        instance = cls()
        instance.split = split
        instance.scenes = []
        for room in instance.rooms:
            with open(room / f"{split.value}.lst") as file:
                subset = file.read().splitlines()
            instance.scenes.extend([room / filename for filename in subset if filename and (room / filename).exists()])

        return instance

    @classmethod
    def make_toy_lst(cls) -> ScanNet:
        instance = cls()
        office_scenes = [
            "scene0010_00_*",
            "scene0010_01_*",
            "scene0040_00_*",
            "scene0040_01_*",
            "scene0089_00_*",
            "scene0089_01_*",
            "scene0089_02_*",
            "scene0098_00_*",
            "scene0098_01_*",
            "scene0131_00_*",
            "scene0131_01_*",
            "scene0131_02_*",
            "scene0255_00_*",
            "scene0255_01_*",
            "scene0255_02_*",
            "scene0464_00_*",
        ]
        instance.scenes = []
        for scene in office_scenes:
            instance.scenes.extend([p.stem for p in (instance.dir / "rooms_01").glob(scene)])

        arr = np.arange(0, len(instance.scenes))
        np.random.shuffle(arr)
        train, val, test = np.split(
            arr,
            [int(0.7 * len(arr)), int(0.85 * len(arr))],  # 70%, 15%, 15%
        )
        scenes = np.array(instance.scenes)
        for split, indices in zip(("train", "val", "test"), (train, val, test)):
            with open(instance.dir / "rooms_01" / f"{split}_toy.lst", "w") as file:
                file.write("\n".join(line for line in scenes[indices]))
        return instance

    @classmethod
    async def clean_and_extract(cls) -> None:
        """
        Yes, this is a very inneficient method, we go through all the files three times, but I can't be bothered to change it right now.
        Besides, this a preparation step that happens before the training itself, so it's not even that critical
        """
        print("Cleaning ScanNet dataset")
        instance = cls()
        for room in instance.dir.iterdir():
            scenes = [scene for scene in room.iterdir() if scene.is_dir()]

        files_to_check = [file for scene in scenes for file in scene.rglob("*.npz")]
        remove = []
        with ProcessPoolExecutor() as executor:
            with tqdm(total=len(files_to_check), desc="Checking files", ncols=100) as progress_bar:
                for file, badFile in zip(files_to_check, executor.map(_check_file, files_to_check)):
                    if badFile:
                        remove.append(file)
                    progress_bar.update(1)

        print(f"Removing {len(remove)} files out of {len(files_to_check)}")
        for file in remove:
            idx = file.stem[-2:]
            (file.parent.parent / f"pointcloud/pointcloud_{idx}.npz").unlink(missing_ok=True)
            (file.parent.parent / f"points_iou/points_iou_{idx}.npz").unlink(missing_ok=True)

        for scene in scenes:
            if scene.is_dir():
                if not any((scene / "pointcloud").iterdir()) or not any((scene / "points_iou").iterdir()):
                    shutil.rmtree(scene)
        with tqdm(total=len(scenes), desc="Unpacking files") as progress_bar:
            for scene in scenes:
                for file in scene.rglob("*.npz"):
                    with np.load(file) as npz_data:
                        file = Path(file)
                        for array_name in npz_data.files:
                            save_to = file.parent / file.stem / array_name
                            save_to.parent.mkdir(parents=True, exist_ok=True)
                            array = npz_data[array_name]
                            metadata = {
                                "dtype": str(array.dtype),
                                "shape": list(array.shape),
                            }
                            bin_path = f"{save_to}.bin"
                            json_path = f"{save_to}.json"
                            # array.tofile(bin_path)
                            # with open(json_path, "w") as f:
                            #     json.dump(metadata, f, indent=2)
                print(file)
                file.unlink(missing_ok=True)
                progress_bar.update()

    @property
    def collate_fn(self) -> Callable:
        return functools.partial(collate_scannet, split=self.split)


def collate_scannet(
    batch: List[tuple[PCloud, PCloud, PCloud]],
    split: ScanNet.SPLITS,
) -> tuple[PCloud, PCloud]:
    clouds, implicit_l1s, implicit_ious = [], [], []
    for cloud, implicit_l1, implicit_iou in batch:
        clouds.append(cloud)
        implicit_l1s.append(implicit_l1)
        implicit_ious.append(implicit_iou)

    cloud_tf = torchvision.transforms.Compose(
        transform_factory(
            getattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()),
        )
    )
    cloud_batch = cloud_tf(collate_cloud(clouds))
    implicit_tf = torchvision.transforms.Compose(
        transform_factory(
            getattr(settings.TRAINER.IMPLICIT_GRID_TF, split.value.upper()),
        )
    )
    implicit_l1_batch = implicit_tf(collate_cloud(implicit_l1s))
    implicit_iou_batch = implicit_tf(collate_cloud(implicit_ious))
    cloud_batch.points = cloud_batch.points.reshape(-1, 3)
    cloud_batch.features = cloud_batch.features.reshape(-1, 1)
    implicit_l1_batch.points = implicit_l1_batch.points.reshape(-1, 3)
    implicit_l1_batch.features = implicit_l1_batch.features.reshape(-1, 1)
    implicit_iou_batch.points = implicit_iou_batch.points.reshape(-1, 3)
    implicit_iou_batch.features = implicit_iou_batch.features.reshape(-1, 1)
    return cloud_batch, implicit_l1_batch, implicit_iou_batch
