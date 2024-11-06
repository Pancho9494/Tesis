import numpy as np
from pathlib import Path
import re

import zipfile
from typing import List, Tuple, Optional
from LIM.data.structures.cloud import Cloud
from LIM.data.datasets.datasetI import CloudDatasetsI


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
                    points = np.load(pcdFile)["points"].astype(np.float32)
                    points = self.__crop(points)
                    cloud = Cloud.from_arr(points)

                with contents.open(iou_path) as iouFile:
                    npz = np.load(iouFile)
                    points = npz["points"].astype(np.float32)
                    implicit = Cloud.from_arr(points + 1e-4 * np.random.randn(*points.shape))
                    implicit.features = npz["df_value"].astype(np.float32)
        except zipfile.BadZipFile:
            pass
        except ValueError:
            pass

        return cloud, implicit

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
            cloud = cloud.downsample(4096, mode=Cloud.DOWNSAMPLE_MODE.RANDOM)
            noise = 0.005 * np.random.randn(*cloud.arr.shape)
            noise = noise.astype(np.float32)
            cloud.pcd.point.positions += noise
            clouds.append(cloud)
            implicits.append(implicit.downsample(2048, mode=Cloud.DOWNSAMPLE_MODE.RANDOM))

        return Cloud.collate(clouds), Cloud.collate(implicits)

    def __crop(self, data: np.ndarray) -> np.ndarray:
        random_ratio = 0.5 * np.random.random()
        min_x = data[:, 0].min()
        max_x = data[:, 0].max()

        min_y = data[:, 1].min()
        max_y = data[:, 1].max()

        remove_size_x = (max_x - min_x) * random_ratio
        remove_size_y = (max_y - min_y) * random_ratio

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        start_x = center_x - (remove_size_x / 2)
        start_y = center_y - (remove_size_y / 2)

        crop_x_idx = np.where((data[:, 0] < (start_x + remove_size_x)) & (data[:, 0] > start_x))[0]
        crop_y_idx = np.where((data[:, 1] < (start_y + remove_size_y)) & (data[:, 1] > start_y))[0]

        crop_idx = np.intersect1d(crop_x_idx, crop_y_idx)

        valid_mask = np.ones(len(data), dtype=bool)
        valid_mask[crop_idx] = 0
        return data[valid_mask]
