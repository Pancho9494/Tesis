from abc import ABC, abstractmethod
import numpy as np
from LIM.data.cloud import Cloud
from LIM.data.pairs import Pairs
from typing import Union


class ModelI(ABC):
    @abstractmethod
    def forward(self, input: Union[Pairs, Cloud]) -> np.ndarray: ...
