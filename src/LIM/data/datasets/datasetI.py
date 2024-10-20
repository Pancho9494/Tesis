from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, List


class CloudDatasetsI(ABC, Dataset):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    @abstractmethod
    def collate(self, batch: List[Any]) -> Any: ...
