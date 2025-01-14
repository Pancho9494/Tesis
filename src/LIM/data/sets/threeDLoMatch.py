import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional
import pickle
import numpy as np
from LIM.data.structures.cloud import Cloud, collate_cloud
from LIM.data.structures.pair import Pair


class ThreeDLoMatch(Dataset):
    """
    Class that represents the 3DLoMatch dataset

    We got it directly from the OverlapPredator repo, and they provide the object with the paths to the pointcloud pairs
    and their respective rotation and translation ground truth matrices

    """

    dir: Path
    src_paths: List[Path]
    tgt_paths: List[Path]
    rot_paths: List[np.ndarray]
    trans_paths: List[np.ndarray]
    overlap_paths: List[Path]

    def __init__(self) -> None:
        self.dir = Path("./data/3DLoMatch/")
        file = "train_info"
        with open(self.dir / f"{file}.pkl", "rb") as f:
            info = pickle.load(f)

        self.src_paths = [self.dir / Path(p) for p in info["src"]]
        self.tgt_paths = [self.dir / Path(p) for p in info["tgt"]]
        self.rot_paths = info["rot"]
        self.trans_paths = info["trans"]
        self.overlap_paths = info["overlap"]

        assert (
            len(self.src_paths)
            == len(self.tgt_paths)
            == len(self.rot_paths)
            == len(self.trans_paths)
            == len(self.overlap_paths)
        ), "Provided file has wrong data, all lists in the dict need to have the same length"

        print(f"Loaded 3DLoMatch_{file} with {len(self)} point cloud pairs")

    def __len__(self) -> int:
        return len(self.rot_paths)

    def __getitem__(self, idx: int) -> Pair:
        src = Cloud.from_path(self.src_paths[idx])
        target = Cloud.from_path(self.tgt_paths[idx])
        ground_truth = np.eye(4)
        ground_truth[:3, :3] = self.rot_paths[idx]
        ground_truth[:3, 3] = self.trans_paths[idx].flatten()
        overlap = self.overlap_paths[idx]

        pair = Pair(src=src, target=target, truth=ground_truth)
        pair.overlap = overlap
        return pair


def collate_3dmatch(batch: List[Pair], tf_pipeline: Optional[List[torch.nn.Module]]) -> Pair:
    sources, targets = [], []
    for pair in batch:
        sources.append(pair.src)
        targets.append(pair.target)

    source_batch, target_batch = collate_cloud(sources), collate_cloud(targets)
    if tf_pipeline is not None:
        tf = torchvision.transforms.Compose(tf_pipeline)
        source_batch, target_batch = tf(source_batch), tf(target_batch)

    return Pair(src=source_batch, target=target_batch, truth=...)
