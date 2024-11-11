import numpy as np
from pathlib import Path
import re

import zipfile
from typing import List, Tuple, Optional
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI
import torchvision
import torch


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
        self.dir = Path("/mnt/nas/scannet.zip")
        self.scenes = []

        with zipfile.ZipFile(self.dir, "r") as contents:
            for filename in contents.namelist():
                regex_match = re.search(".?scene\d{4}_\d{2}_\d{2}/(\s)*(?!.)", str(filename))
                if regex_match is None:
                    continue

                self.scenes.append(Path(filename))

        print(f"Loaded ScanNet with {len(self)} point clouds")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[Optional[Cloud], Optional[Cloud]]:
        """
        Extracts from the scannet zipfile the contents of just the pointcloud associated with idx.

        In the IAE's processed version of scannet they include the following data for each pointcloud:
            * df_value:     subsampled(np.load(iou_path)['df_value']), the computed value for the Distance Function (DF)
            * occupancies:  ???
            * points:       subsampled(np.load(iou_path)['points']), the points representing the discretized space,
                            each point has a DF value associated

        TODO: I think we need to split the dataset elements into groups based on their scenes
        """

        cloud: Optional[Cloud] = None
        implicit: Optional[Cloud] = None
        try:
            scene = self.scenes[idx]
            sub_idx = np.random.randint(5)
            with zipfile.ZipFile(self.dir, "r") as contents:
                pcd_path = str(scene / f"pointcloud/pointcloud_{sub_idx:02d}.npz")
                iou_path = str(scene / f"points_iou/points_iou_{sub_idx:02d}.npz")

                with contents.open(pcd_path) as pcdFile:
                    cloud = Cloud.from_arr(np.load(pcdFile)["points"].astype(np.float32))

                with contents.open(iou_path) as iouFile:
                    npz = np.load(iouFile)
                    implicit = Cloud.from_arr(npz["points"].astype(np.float32))
                    implicit.features = npz["df_value"].astype(np.float32)
        except zipfile.BadZipFile:
            pass
        except ValueError:
            pass

        return cloud, implicit

    def set_transforms(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None:
        self.cloud_tf = torchvision.transforms.Compose(cloud_tf)
        self.implicit_tf = torchvision.transforms.Compose(implicit_tf)

    def collate(self, batch: List[Tuple[Optional[Cloud], Optional[Cloud]]]) -> Tuple[Cloud, Cloud]:
        def __replace_nones_in_batch(batch):
            len_before = len(batch)
            batch = list(filter(lambda x: all(x), batch))  # filter elements witn None values
            len_after = len(batch)
            while len_after != len_before:
                cloud, implicit = self.__getitem__(np.random.randint(len(self)))
                if all([cloud, implicit]):
                    batch.append((cloud, implicit))
                    len_after += 1
            return batch

        clouds = []
        implicits = []
        for cloud, implicit in __replace_nones_in_batch(batch):
            clouds.append(cloud)
            implicits.append(implicit)

        cloud_batch, implicit_batch = Cloud.collate(clouds), Cloud.collate(implicits)
        if self.cloud_tf is not None and self.implicit_tf is not None:
            cloud_batch, implicit_batch = self.cloud_tf(cloud_batch), self.implicit_tf(implicit_batch)
        return cloud_batch, implicit_batch
