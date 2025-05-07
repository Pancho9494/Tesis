import numpy as np
from pathlib import Path
import zipfile
from typing import List, Tuple, Optional, Callable
from LIM.data.structures.pcloud import PCloud, collate_cloud
from LIM.data.sets.datasetI import CloudDatasetsI
import torchvision
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import shutil
import random
from LIM.data.structures import transform_factory
from config.config import settings
import functools

executor = ThreadPoolExecutor()


def check_file(file: Path) -> bool:
    try:
        np.load(str(file))
    except (zipfile.BadZipFile, EOFError, ValueError):
        return True
    return False


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

    dir: Path = Path("./src/LIM/data/raw/scannet")
    paths: List[Path]
    split: Optional[CloudDatasetsI.SPLITS] = None

    def __init__(self) -> None:
        self.rooms = self.dir.iterdir()

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[Optional[PCloud], Optional[PCloud]]:
        def __load_npz(path: Path) -> np.ndarray:
            return np.load(str(path))

        cloud: Optional[PCloud] = None
        implicit: Optional[PCloud] = None

        scene = self.scenes[idx]
        sub = random.choice(list((scene / "pointcloud/").iterdir()))
        sub_idx = int(sub.stem[-2:])

        future1 = executor.submit(__load_npz, scene / f"pointcloud/pointcloud_{sub_idx:02d}.npz")
        future2 = executor.submit(__load_npz, scene / f"points_iou/points_iou_{sub_idx:02d}.npz")

        pointcloud_file, iou_file = future1.result(), future2.result()
        cloud = PCloud.from_arr(pointcloud_file["points"].astype(np.float32))
        cloud.path = scene / f"pointcloud/pointcloud_{sub_idx:02d}.npz"
        implicit = PCloud.from_arr(iou_file["points"].astype(np.float32))
        implicit.features = np.expand_dims(iou_file["df_value"].astype(np.float32), axis=1)

        implicit.path = scene / f"points_iou/points_iou_{sub_idx:02d}.npz"
        return cloud, implicit

    @classmethod
    async def clean_bad_files(cls) -> None:
        print("Cleaning ScanNet dataset")
        instance = cls()
        for room in instance.dir.iterdir():
            scenes = [scene for scene in room.iterdir() if scene.is_dir()]

        files_to_check = [file for scene in scenes for file in scene.rglob("*.npz")]
        remove = []
        with ProcessPoolExecutor() as executor:
            with tqdm(total=len(files_to_check), desc="Checking files", ncols=100) as progress_bar:
                for file, badFile in zip(files_to_check, executor.map(check_file, files_to_check)):
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

    @classmethod
    def new_instance(cls, split: CloudDatasetsI.SPLITS) -> "ScanNet":
        instance = cls()
        instance.split = split
        instance.scenes = []
        for room in instance.rooms:
            with open(room / f"{split.value}.lst") as file:
                subset = file.read().splitlines()
            instance.scenes.extend([room / filename for filename in subset if filename and (room / filename).exists()])
        return instance

    @property
    def collate_fn(self) -> Callable:
        return functools.partial(collate_scannet, split=self.split)


def collate_scannet(
    batch: List[Tuple[Optional[PCloud], Optional[PCloud]]],
    split: ScanNet.SPLITS,
) -> Tuple[PCloud, PCloud]:
    clouds, implicits = [], []
    for cloud, implicit in batch:
        clouds.append(cloud)
        implicits.append(implicit)

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
    implicit_batch = implicit_tf(collate_cloud(implicits))
    # cloud_batch, implicit_batch = (
    #     (
    #         pcd_tf := torchvision.transforms.Compose(
    #             transform_factory(
    #                 getattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()),
    #             )
    #         )
    #     )(collate_cloud(clouds)),
    #     (
    #         imp_tf := torchvision.transforms.Compose(
    #             transform_factory(
    #                 getattr(settings.TRAINER.IMPLICIT_GRID_TF, split.value.upper()),
    #             )
    #         )
    #     )(collate_cloud(implicits)),
    # )
    cloud_batch.points = cloud_batch.points.reshape(-1, 3)
    cloud_batch.features = cloud_batch.features.reshape(-1, 1)
    implicit_batch.points = implicit_batch.points.reshape(-1, 3)
    implicit_batch.features = implicit_batch.features.reshape(-1, 1)
    return cloud_batch, implicit_batch
