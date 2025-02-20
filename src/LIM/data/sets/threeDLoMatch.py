import torch
import torchvision
from pathlib import Path
from typing import List, Optional, Callable, Dict
import pickle
import numpy as np
from LIM.data.structures.pcloud import PCloud, collate_cloud, Downsampler
from LIM.data.structures.pair import Pair
from LIM.data.sets.datasetI import CloudDatasetsI


class ThreeDLoMatch(CloudDatasetsI):
    """
    Class that represents the 3DLoMatch dataset

    We got it directly from the OverlapPredator repo, and they provide the object with the paths to the pointcloud pairs
    and their respective rotation and translation ground truth matrices

    """

    dir: Path = Path("./src/LIM/data/raw/3DLoMatch/")
    downsample_table: Dict[str, float] = {}

    src_paths: List[Path]
    tgt_paths: List[Path]
    rot_paths: List[np.ndarray]
    trans_paths: List[np.ndarray]
    overlap_paths: List[Path]

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

        pair = Pair(source=src, target=target, GT_tf_matrix=ground_truth)
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
    def new_instance(cls, info_path: str) -> "ThreeDLoMatch":
        instance = cls()
        with open(cls.dir / info_path, "rb") as f:
            info = pickle.load(f)
        return instance.__parse_info(info)

    def train_set(self) -> "ThreeDLoMatch":
        return ThreeDLoMatch.new_instance(info_path="train_info.pkl")

    def val_set(self) -> "ThreeDLoMatch":
        return ThreeDLoMatch.new_instance(info_path="val_info.pkl")

    def test_set(self) -> "ThreeDLoMatch":
        return ThreeDLoMatch.new_instance(info_path="test_info.pkl")

    def force_downsample(self, pair: Pair) -> None:
        """
        Keep track of the max size of each pair the computer can handle in order to avoid pytorch OOMs
        """
        path = pair.source.path[0]
        if (tag := f"{path.parent.name}/{path.stem}") not in self.downsample_table:
            self.downsample_table[tag] = 1.0
        self.downsample_table[tag] = max(0.1, self.downsample_table[tag] - 0.05)

    @property
    def collate_fn(self) -> Callable:
        return collate_3dmatch


def collate_3dmatch(batch: List[Pair], tf_pipeline: Optional[List[torch.nn.Module]]) -> Pair:
    sources, targets, GT_tf_matrices = [], [], []
    for pair in batch:
        sources.append(pair.source)
        targets.append(pair.target)
        GT_tf_matrices.append(pair.GT_tf_matrix)

    source_batch, target_batch = collate_cloud(sources), collate_cloud(targets)
    GT_tf_batch = np.concatenate([np.expand_dims(arr, axis=0) for arr in GT_tf_matrices], axis=0)

    if tf_pipeline is not None:
        tf = torchvision.transforms.Compose(tf_pipeline)
        source_batch, target_batch = tf(source_batch), tf(target_batch)

    source_batch.points = source_batch.points.reshape(-1, 3)
    source_batch.features = source_batch.features.reshape(-1, 1)
    target_batch.points = target_batch.points.reshape(-1, 3)
    target_batch.features = target_batch.features.reshape(-1, 1)
    return Pair(source=source_batch, target=target_batch, GT_tf_matrix=np.squeeze(GT_tf_batch, axis=0))
