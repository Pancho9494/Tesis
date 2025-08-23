from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict

from config.config import settings
from torch.utils.data import Dataset

import LIM.log as log

if settings.TRAINER.SUBSET is not None:
    SUBSET = settings.TRAINER.SUBSET
else:
    SUBSET = "no"

log.info(f"Training with {SUBSET} subset")


class CloudDatasetsI(ABC, Dataset):
    downsample_table: Dict[str, float] = {}

    class SPLITS(Enum):
        TRAIN = "train"
        VAL = "val"
        TEST = "test"

        TOY_TRAIN = f"train_{SUBSET}"
        TOY_VAL = f"val_{SUBSET}"
        TOY_TEST = f"test_{SUBSET}"

        # TOY_TRAIN = "train_hand_picked_offices"
        # TOY_VAL = "val_hand_picked_offices"
        # TOY_TEST = "test_hand_picked_offices"

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    @property
    @abstractmethod
    def collate_fn(self) -> Callable: ...

    @classmethod
    @abstractmethod
    def new_instance(cls, *args, **kwargs) -> "CloudDatasetsI":
        """
        Handles specific dataset initialization
        """
        ...

    def force_downsample(self, sample: Any) -> None:
        """
        Keep track of the max size of each pair the computer can handle in order to avoid pytorch OOMs
        """
        tag = sample[0].tag if isinstance(sample, tuple) else sample.tag  # TODO: kind of an ugly solution
        if tag not in self.downsample_table:
            self.downsample_table[tag] = 1.0
        self.downsample_table[tag] = max(0.1, self.downsample_table[tag] - 0.05)
