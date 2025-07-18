from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torchvision

import LIM.log as log
from config.config import settings
from LIM.data.sets.datasetI import CloudDatasetsI
from LIM.data.structures.pair import Pair
from LIM.data.structures.pcloud import Downsampler, PCloud, collate_cloud
from LIM.data.structures.transforms import transform_factory


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
    split: CloudDatasetsI.SPLITS

    def __repr__(self) -> str:
        return f"3DLoMatch({len(self)})"

    def __len__(self) -> int:
        return len(self.src_paths) if hasattr(self, "src_paths") else 0

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

    def __parse_info(self, info: Dict, skip_indices: list[int] | None = None) -> ThreeDLoMatch:
        """
        Args:
            info:
            skip_indices: 1 to skip element, 0 to keep
        """
        if skip_indices is None:
            skip_indices = [0 for _ in info["src"]]
        self.src_paths = [self.dir / Path(p) for p, skip in zip(info["src"], skip_indices) if not skip]
        self.tgt_paths = [self.dir / Path(p) for p, skip in zip(info["tgt"], skip_indices) if not skip]
        self.rot_paths = [r for r, skip in zip(info["rot"], skip_indices) if not skip]
        self.trans_paths = [t for t, skip in zip(info["trans"], skip_indices) if not skip]
        self.overlap_paths = [o for o, skip in zip(info["overlap"], skip_indices) if not skip]
        return self

    @classmethod
    def new_instance(cls, split: CloudDatasetsI.SPLITS) -> ThreeDLoMatch:
        instance = cls()
        instance.split = split
        info_path = cls.dir / f"{split.value}_info.pkl"
        log.info(f"ThreeDLoMatch loading {info_path}")
        with open(info_path, "rb") as f:
            info = pickle.load(f)

        out = instance.__parse_info(info)
        bar = {}
        for name in ["src", "tgt", "rot", "trans", "overlap"]:
            bar[name] = len(getattr(instance, f"{name}_paths"))
        return out

    @classmethod
    def make_toy_pkl(cls) -> None:
        log.info("Making toy dataset lists for ThreeDLoMatch")
        instance = cls()
        with open(cls.dir / "train_info.pkl", "rb") as f:
            info = pickle.load(f)
        instance = instance.__parse_info(info, [0 if "7-scenes-office" in p else 1 for p in info["src"]])

        arr = np.arange(0, len(instance.src_paths))
        np.random.shuffle(arr)
        train, val, test = np.split(
            arr,
            [int(0.8 * len(arr)), int(0.9 * len(arr))],  # 80%, 10%, 10%
        )
        new_infos: dict[str, dict[str, list[Path] | np.ndarray]] = {
            "new_train_info": {},
            "new_val_info": {},
            "new_test_info": {},
        }
        for split, indices in zip(("train", "val", "test"), (train, val, test)):
            for arr_name in ["src", "tgt"]:
                new_infos[f"new_{split}_info"][arr_name] = [
                    "/".join(str(p).split("/")[5:]) for p in np.array(getattr(instance, f"{arr_name}_paths"))[indices]
                ]

            for arr_name in ["rot", "trans", "overlap"]:
                new_infos[f"new_{split}_info"][arr_name] = np.array(getattr(instance, f"{arr_name}_paths"))[indices]

        for split in ["train", "val", "test"]:
            with open(instance.dir / f"{split}_toy_info.pkl", "wb") as file:
                foo = {k: len(v) for k, v in new_infos[f"new_{split}_info"].items()}
                log.info(f"Writing to {instance.dir / f'{split}_toy_info.pkl'}\ndict with shapes: {foo}")
                pickle.dump(new_infos[f"new_{split}_info"], file)

    @property
    def collate_fn(self) -> Callable:
        return functools.partial(collate_3dmatch, split=self.split)


def collate_3dmatch(batch: List[Pair], split: CloudDatasetsI.SPLITS) -> Pair:
    if settings is None:
        raise RuntimeError("settings has not been initialized")
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
