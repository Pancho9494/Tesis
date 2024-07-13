from abc import ABC, abstractmethod
import numpy as np
from database.pairs import Pairs

class ModelI(ABC):
    
    @abstractmethod
    def __call__(self, pair: Pairs) -> np.ndarray:
        ...
