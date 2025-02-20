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

    dir: Path
    paths: List[Path]
    cloud_tf: Optional[torchvision.transforms.Compose] = None
    implicit_tf: Optional[torchvision.transforms.Compose] = None

    def __init__(self) -> None:
        self.dir = Path("./data/scannet")
        self.scenes = []

        for room in self.dir.iterdir():
            self.scenes = [scene for scene in room.iterdir() if scene.is_dir()]

        print(f"Loaded ScanNet with {len(self)} point clouds")

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
        implicit.features = iou_file["df_value"].astype(np.float32)
        implicit.path = scene / f"points_iou/points_iou_{sub_idx:02d}.npz"
        return cloud, implicit

    async def clean_bad_files(self) -> None:
        print("Cleaning ScanNet dataset")
        files_to_check = [file for scene in self.scenes for file in scene.rglob("*.npz")]
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

        for scene in self.scenes:
            if scene.is_dir():
                if not any((scene / "pointcloud").iterdir()) or not any((scene / "points_iou").iterdir()):
                    shutil.rmtree(scene)

    def set_transforms(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)

    @property
    def collate_fn(self) -> Callable:
        return collate_scannet


def collate_scannet(
    batch: List[Tuple[Optional[PCloud], Optional[PCloud]]],
    cloud_tf: Optional[List[torch.nn.Module]],
    implicit_tf: Optional[List[torch.nn.Module]],
) -> Tuple[PCloud, PCloud]:
    clouds, implicits = [], []
    for cloud, implicit in batch:
        clouds.append(cloud)
        implicits.append(implicit)

    cloud_batch, implicit_batch = collate_cloud(clouds), collate_cloud(implicits)
    if cloud_tf is not None and implicit_tf is not None:
        cloud_tf = torchvision.transforms.Compose(cloud_tf)
        implicit_tf = torchvision.transforms.Compose(implicit_tf)
        cloud_batch, implicit_batch = cloud_tf(cloud_batch), implicit_tf(implicit_batch)
    return cloud_batch, implicit_batch
