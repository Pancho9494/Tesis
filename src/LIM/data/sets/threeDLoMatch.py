import torch
import torchvision
from pathlib import Path
from typing import List, Optional, Callable, Dict
import pickle
import numpy as np
import functools
from LIM.data.structures.pcloud import PCloud, collate_cloud, Downsampler
from LIM.data.structures.pair import Pair
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures.transforms import transform_factory
from config.config import settings


class ThreeDLoMatch(CloudDatasetsI):
    """
    Class that represents the 3DLoMatch dataset

    We got it directly from the OverlapPredator repo, and they provide the object with the paths to the pointcloud pairs
    and their respective rotation and translation ground truth matrices

    """

    dir: Path = Path("./src/LIM/data/raw/3DLoMatch/")
    src_paths: List[Path]
    tgt_paths: List[Path]
    rot_paths: List[np.ndarray]
    trans_paths: List[np.ndarray]
    overlap_paths: List[Path]
    split: Optional[CloudDatasetsI.SPLITS]

    def __repr__(self) -> str:
        return "3DLoMatch()"

    def __len__(self) -> int:
        return len(self.src_paths)

    def __getitem__(self, idx: int) -> Pair:
        src = PCloud.from_path(self.src_paths[idx])
        target = PCloud.from_path(self.tgt_paths[idx])

        if (tag := f"{src.path.parent.name}/{src.path.stem}") in self.downsample_table:
            src = Downsampler(size=int(self.downsample_table[tag] * len(src)))(src)
            target = Downsampler(size=int(self.downsample_table[tag] * len(target)))(target)

        ground_truth = np.eye(4)
        ground_truth[:3, :3] = self.rot_paths[idx]
        ground_truth[:3, 3] = self.trans_paths[idx].flatten()
        overlap = self.overlap_paths[idx]

        pair = Pair(
            id=f"{src.path.parent.name}/{src.path.stem}::{target.path.parent.name}/{target.path.stem}",
            source=src,
            target=target,
            GT_tf_matrix=ground_truth,
        )
        pair.overlap = overlap
        pair.correspondences
        return pair

    def __parse_info(self, info: Dict) -> "ThreeDLoMatch":
        self.src_paths = [self.dir / Path(p) for p in info["src"]]
        self.tgt_paths = [self.dir / Path(p) for p in info["tgt"]]
        self.rot_paths = info["rot"]
        self.trans_paths = info["trans"]
        self.overlap_paths = info["overlap"]
        return self

    @classmethod
    def new_instance(cls, split: CloudDatasetsI.SPLITS) -> "ThreeDLoMatch":
        instance = cls()
        instance.split = split
        with open(cls.dir / f"{split.value}_info.pkl", "rb") as f:
            info = pickle.load(f)
        return instance.__parse_info(info)

    @property
    def collate_fn(self) -> Callable:
        return functools.partial(collate_3dmatch, split=self.split)


def collate_3dmatch(batch: List[Pair], split: CloudDatasetsI.SPLITS) -> Pair:
    sources, targets, GT_tf_matrices = [], [], []
    for pair in batch:
        sources.append(pair.source)
        targets.append(pair.target)
        GT_tf_matrices.append(pair.GT_tf_matrix)

    source_batch, target_batch = collate_cloud(sources), collate_cloud(targets)
    GT_tf_batch = np.concatenate([np.expand_dims(arr, axis=0) for arr in GT_tf_matrices], axis=0)

    tf = torchvision.transforms.Compose(
        transform_factory(
            getattr(settings.TRAINER.POINTCLOUD_TF, split.value.upper()),
        )
    )
    source_batch, target_batch = tf(source_batch), tf(target_batch)
    source_batch.points = source_batch.points.reshape(-1, 3)
    source_batch.features = source_batch.features.reshape(-1, 1)
    target_batch.points = target_batch.points.reshape(-1, 3)
    target_batch.features = target_batch.features.reshape(-1, 1)
    return Pair(id=batch[0].id, source=source_batch, target=target_batch, GT_tf_matrix=np.squeeze(GT_tf_batch, axis=0))
