from abc import ABC, abstractmethod
import numpy as np
from database.cloudPairs import FragmentPairs

class ModelI(ABC):
    
    @abstractmethod
    def __call__(self, pair: FragmentPairs) -> np.ndarray:
        ...
