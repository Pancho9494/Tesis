import numpy as np
from pathlib import Path

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
        self.dir = Path("src/LIM/data/raw_data/scannet.zip")
        self.paths = []

        with zipfile.ZipFile(self.dir, "r") as contents:
            for filename in contents.namelist():
                path = Path(filename)
                if ("pointcloud" in path.parts) and (path.suffix.lower() in [".npz", ".npy"]):
                    self.paths.append(path)

        print(f"Loaded ScanNet with {len(self)} point clouds")

    def __len__(self) -> int:
        return len(self.paths)

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
            path = self.paths[idx]
            with zipfile.ZipFile(self.dir, "r") as contents:
                pcd_path = str(path)
                iou_path = str(path.parent.parent / f"points_iou/points_iou_{path.stem[-2:]}.npz")

                with contents.open(pcd_path) as pcdFile:
                    cloud = Cloud.from_arr(np.load(pcdFile)["points"])

                with contents.open(iou_path) as iouFile:
                    npz = np.load(iouFile)
                    implicit = Cloud.from_arr(npz["points"])
                    implicit.features = npz["df_value"]
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
            clouds.append(cloud)
            implicits.append(implicit)

        return Cloud.collate(clouds), Cloud.collate(implicits)
