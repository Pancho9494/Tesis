from abc import ABC, abstractmethod
from LIM.data.pairs import Pairs
from torch.utils.data import Dataset


class PairMetric(ABC):
    @abstractmethod
    def __call__(self, pair: Pairs) -> float: ...


class DatasetMetric(ABC):
    @abstractmethod
    def __call__(self, data: Dataset) -> float: ...
