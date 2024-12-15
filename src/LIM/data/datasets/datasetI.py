from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, List
import torch


class CloudDatasetsI(ABC, Dataset):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    @abstractmethod
    def set_transforms(self, cloud_tf: List[torch.nn.Module], implicit_tf: List[torch.nn.Module]) -> None: ...
