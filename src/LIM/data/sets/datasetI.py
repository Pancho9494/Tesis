from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, List, Callable
import torch
import numpy as np


class CloudDatasetsI(ABC, Dataset):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    # @abstractmethod
    # def set_transforms(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None: ...

    @property
    @abstractmethod
    def collate_fn(self) -> Callable: ...

    # @abstractmethod
    # def from_split(self, split: List[int]) -> "CloudDatasetsI": ...

    def random_split(self, validation_split: float) -> tuple["CloudDatasetsI", "CloudDatasetsI"]:
        indices = np.random.permutation(len(self))
        split = int((1 - validation_split) * len(self))
        return self.from_split(indices[:split]), self.from_split(indices[split:])
