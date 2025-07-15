from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict

from torch.utils.data import Dataset


class CloudDatasetsI(ABC, Dataset):
    downsample_table: Dict[str, float] = {}

    class SPLITS(Enum):
        TRAIN = "train"
        VAL = "val"
        TEST = "test"
        TOY_TRAIN = "train_toy"
        TOY_VAL = "val_toy"
        TOY_TEST = "test_toy"

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
